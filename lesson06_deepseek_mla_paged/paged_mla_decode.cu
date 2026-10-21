#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// ============================================================================
// 第 6 课核心算子：DeepSeek-V3/V4 专属 Paged MLA (Multi-head Latent Attention)
//                 + Online Softmax 算子实战
// 对应 FlashInfer 源码：include/flashinfer/attention/mla.cuh
// 对应 vLLM 源码：csrc/attention/mla/
// ============================================================================
// DeepSeek MLA 在 Decode 阶段的核心奥秘（矩阵吸收 Matrix Absorption）：
// 传统 MHA 每个 Head 有独立的 K 和 V Cache，显存带宽开销极大。
// DeepSeek MLA 将 KV Cache 压缩为全 Head 共享的 1 个低维潜向量：
//   - c_KV (Compressed KV Latent): 维度 D_ckv = 512 (所有 Query Head 共享！)
//   - k_pe (RoPE 位置编码 Key):    维度 D_kpe = 64  (所有 Query Head 共享！)
// 通过把解压矩阵 W_UK 吸收进 Query，把 W_UV 吸收进 Output Projection，
// Decode 阶段的 MLA 变成了一个 128 个 Query Head 共享同 1 个 (512+64=576维) KV Head 的超级 MQA！
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int D_CKV = 64; // 教学演示缩小版潜向量维度 (真实 DeepSeek 为 512，原理完全一致)
constexpr int D_KPE = 16; // 教学演示缩小版位置编码维度 (真实 DeepSeek 为 64)
constexpr int D_TOTAL = D_CKV + D_KPE; // 80

__device__ __forceinline__ float warp_reduce_sum_mla(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ----------------------------------------------------------------------------
// Paged MLA Decode Kernel (内置 FlashAttention Online Softmax 算法)
// 维度说明：
//   q_nope:      [batch_size, num_heads, D_CKV] (已吸收 W_UK 的非位置编码 Query)
//   q_pe:        [batch_size, num_heads, D_KPE] (带 RoPE 的位置编码 Query)
//   ckv_cache:   [num_pages, page_size, D_CKV]  (分页潜变量 KV Cache，同时充当 Key 和 Value！)
//   kpe_cache:   [num_pages, page_size, D_KPE]  (分页位置编码 Key Cache)
//   page_table:  [batch_size, max_num_pages]    (每个请求的物理页号映射表)
//   seq_lens:    [batch_size]                   (每个请求当前的上下文长度)
// 输出：
//   out_latent:  [batch_size, num_heads, D_CKV] (待乘以 W_UV 的潜空间注意力输出)
// ----------------------------------------------------------------------------
__global__ void paged_mla_decode_online_softmax_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ page_table,
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out_latent,
    int num_heads,
    int page_size,
    int max_num_pages,
    float sm_scale
) {
    // Grid: (batch_size, num_heads) -> 每个 Block (1 个 Warp = 32 线程) 负责 1 个 (batch, head)
    // 进阶优化提示：在 sm_120 上可让 1 个 CTA 的多个 Warp 同时处理同一个 batch 的多个 Head，
    // 在 Shared Memory 中共享同一份 ckv_cache/kpe_cache！
    int batch_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int lane_id   = threadIdx.x; // 0..31

    int seq_len = seq_lens[batch_idx];
    const int* req_page_table = page_table + batch_idx * max_num_pages;

    // 1. 将当前 Head 的 q_nope (64维) 和 q_pe (16维) 读入寄存器常驻！
    //    32 个线程，每个线程负责 D_CKV / 32 = 2 个 ckv 元素，以及 (lane_id < 16) 的 1 个 kpe 元素
    int q_Index_base_nope = (batch_idx * num_heads + head_idx) * D_CKV;
    int q_Index_base_pe   = (batch_idx * num_heads + head_idx) * D_KPE;

    float reg_q_nope[2];
    reg_q_nope[0] = __bfloat162float(q_nope[q_Index_base_nope + lane_id * 2 + 0]);
    reg_q_nope[1] = __bfloat162float(q_nope[q_Index_base_nope + lane_id * 2 + 1]);

    float reg_q_pe = (lane_id < D_KPE)
                     ? __bfloat162float(q_pe[q_Index_base_pe + lane_id])
                     : 0.0f;

    // ========================================================================
    // 2. 初始化 Online Softmax 状态寄存器 (FlashAttention 核心数学公式)：
    //    m_i: 当前遍历过的所有历史 Token 的最大打分 (初始化为 -FLT_MAX)
    //    l_i: 当前的指数分母累加和 sum(exp(s - m_i)) (初始化为 0.0f)
    //    acc_o[2]: 当前线程负责的 2 个输出维度的未归一化分子累加器
    // ========================================================================
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc_o[2] = {0.0f, 0.0f};

    // 3. 遍历该请求的所有上下文 Token (按分页表 Page Table 寻址)
    for (int token_pos = 0; token_pos < seq_len; ++token_pos) {
        int logical_page_idx = token_pos / page_size;
        int offset_in_page   = token_pos % page_size;
        int physical_page_id = req_page_table[logical_page_idx];

        int ckv_base = (physical_page_id * page_size + offset_in_page) * D_CKV;
        int kpe_base = (physical_page_id * page_size + offset_in_page) * D_KPE;

        // 读取当前 KV Token 的潜向量 c_KV (既是 Key 也是 Value！) 与位置编码 k_pe
        float k_ckv_0 = __bfloat162float(ckv_cache[ckv_base + lane_id * 2 + 0]);
        float k_ckv_1 = __bfloat162float(ckv_cache[ckv_base + lane_id * 2 + 1]);
        float k_pe_val = (lane_id < D_KPE)
                         ? __bfloat162float(kpe_cache[kpe_base + lane_id])
                         : 0.0f;

        // 计算点积 score = (q_nope · c_KV + q_pe · k_pe) * sm_scale
        float partial_dot = reg_q_nope[0] * k_ckv_0 + reg_q_nope[1] * k_ckv_1 + reg_q_pe * k_pe_val;
        float score = warp_reduce_sum_mla(partial_dot) * sm_scale;

        // ====================================================================
        // 4. Online Softmax 状态单步更新 (无需保存整个 seq_len 的 Attention 矩阵！)
        //    m_new = max(m_i, score)
        //    alpha = exp(m_i - m_new)   -> 用于缩放旧的累加和 l_i 与旧的输出分子 acc_o
        //    p     = exp(score - m_new) -> 当前 Token 的指数权重
        // ====================================================================
        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float p     = __expf(score - m_new);

        l_i = l_i * alpha + p;
        acc_o[0] = acc_o[0] * alpha + p * k_ckv_0; // 注意：MLA 中 Value 就是 c_KV 本身！
        acc_o[1] = acc_o[1] * alpha + p * k_ckv_1;
        m_i = m_new;
    }

    // 5. 最终归一化：除以分母累加和 l_i，写回显存
    float inv_l = 1.0f / (l_i + 1e-12f);
    out_latent[q_Index_base_nope + lane_id * 2 + 0] = __float2bfloat16(acc_o[0] * inv_l);
    out_latent[q_Index_base_nope + lane_id * 2 + 1] = __float2bfloat16(acc_o[1] * inv_l);
}

void launch_paged_mla_decode(
    const void* q_nope,
    const void* q_pe,
    const void* ckv_cache,
    const void* kpe_cache,
    const int* page_table,
    const int* seq_lens,
    void* out_latent,
    int batch_size,
    int num_heads,
    int page_size,
    int max_num_pages,
    float sm_scale,
    cudaStream_t stream = 0
) {
    dim3 grid(batch_size, num_heads);
    dim3 block(WARP_SIZE); // 32 threads (1 Warp)
    paged_mla_decode_online_softmax_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope),
        reinterpret_cast<const __nv_bfloat16*>(q_pe),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache),
        page_table,
        seq_lens,
        reinterpret_cast<__nv_bfloat16*>(out_latent),
        num_heads,
        page_size,
        max_num_pages,
        sm_scale
    );
}
