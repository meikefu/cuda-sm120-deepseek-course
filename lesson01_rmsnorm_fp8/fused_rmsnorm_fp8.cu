#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>

// ============================================================================
// 第 1 课核心算子：Fused Residual + RMSNorm + Dynamic Per-Token FP8 Quantization
// 对应 vLLM 源码参考：csrc/layernorm_quant_kernels.cu
// ============================================================================
// 为什么大模型推理极度需要这个融合算子？
// 在 DeepSeek / Llama 的每一层进入 FP8 Linear 或 MoE 之前，原本需要 3 个独立算子：
//   1. hidden_states = hidden_states + residual          (读写一次显存)
//   2. norm_out = RMSNorm(hidden_states, weight)         (读写一次显存)
//   3. fp8_out, scale = dynamic_fp8_quant(norm_out)      (读写一次显存)
// 融合后：数据从 GDDR7 读入寄存器 1 次，在寄存器和 Warp 内部完成所有规约，直接写出 FP8！
// 显存带宽开销直接降低约 60%！
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr float FP8_E4M3_MAX = 448.0f; // NVIDIA E4M3 标准最大值

// ----------------------------------------------------------------------------
// 知识点 1：Warp 级规约原语（Warp Shuffle Reduction）
// 不经过 Shared Memory，直接通过寄存器通道在 32 个线程间交换数据，延迟仅几个时钟周期
// ----------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// 知识点 2：Block 级规约（结合 Warp Shuffle + 极小量 Shared Memory）
// 一个 CTA (Block) 负责处理 1 个 Token（即矩阵的一行，长度为 hidden_size）
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem_reduce) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 第一步：每个 Warp 内部先规约
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        smem_reduce[warp_id] = val;
    }
    __syncthreads();

    // 第二步：由第 0 个 Warp 把所有 Warp 的结果再规约一次
    val = (threadIdx.x < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            smem_reduce[0] = val;
        }
    }
    __syncthreads();
    return smem_reduce[0];
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val, float* smem_reduce) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane_id == 0) {
        smem_reduce[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            smem_reduce[0] = val;
        }
    }
    __syncthreads();
    return smem_reduce[0];
}

// ----------------------------------------------------------------------------
// 知识点 3：主 Kernel —— 每个 Block 处理 1 个 Token (1 行, 维度为 hidden_size)
// 输入：
//   input:    [num_tokens, hidden_size] (BF16)
//   residual: [num_tokens, hidden_size] (BF16, 原地更新：residual += input)
//   weight:   [hidden_size]             (BF16, RMSNorm 的 gamma 权重)
// 输出：
//   out_fp8:  [num_tokens, hidden_size] (FP8 E4M3, 供下一层 CUTLASS FP8 GEMM 使用)
//   scales:   [num_tokens, 1]           (FP32, 当前 Token 的动态量化缩放因子)
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_dynamic_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8,
    float* __restrict__ scales,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float epsilon
) {
    // 当前 Block 负责第 token_idx 行
    const int token_idx = blockIdx.x;
    const int row_offset = token_idx * hidden_size;

    // 用于跨 Warp 规约的 Shared Memory（最多 32 个 Warp，仅需 32 个 float = 128 Bytes）
    __shared__ float smem_reduce[32];
    // 使用动态 Shared Memory 缓存残差相加后的 float 值，避免二次读取 Global Memory！
    extern __shared__ float smem_row[];

    float sum_sq = 0.0f;

    // 阶段 1：读取 input 和 residual，完成残差相加，写回新的 residual，并累加平方和
    for (int idx = threadIdx.x; idx < hidden_size; idx += BLOCK_SIZE) {
        float x = __bfloat162float(input[row_offset + idx]);
        float r = __bfloat162float(residual[row_offset + idx]);
        float h = x + r;

        // 写回更新后的 residual 供下一个 Transformer 层使用
        residual[row_offset + idx] = __float2bfloat16(h);
        // 缓存在 Shared Memory 中，省去后续再从 GDDR7 显存读取
        smem_row[idx] = h;

        sum_sq += h * h;
    }

    // 计算 1 / sqrt(mean(h^2) + eps)
    float total_sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq, smem_reduce);
    float rms_rcp = rsqrtf(total_sum_sq / static_cast<float>(hidden_size) + epsilon);

    // 阶段 2：乘上 RMSNorm 权重 weight，同时找出当前 Token 的最大绝对值 absmax
    float local_absmax = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_size; idx += BLOCK_SIZE) {
        float w = __bfloat162float(weight[idx]);
        float normed = smem_row[idx] * rms_rcp * w;
        smem_row[idx] = normed;
        local_absmax = fmaxf(local_absmax, fabsf(normed));
    }

    // 规约得到整行的 absmax，并计算 Per-Token FP8 缩放因子 scale
    float token_absmax = block_reduce_max<BLOCK_SIZE>(local_absmax, smem_reduce);
    // 防止全 0 除零，计算 scale = max(|x|) / 448.0
    float scale = fmaxf(token_absmax / FP8_E4M3_MAX, 1e-12f);
    float inv_scale = 1.0f / scale;

    if (threadIdx.x == 0) {
        scales[token_idx] = scale;
    }

    // 阶段 3：利用硬件转换指令将归一化结果量化为 FP8 E4M3 并写回全局显存
    for (int idx = threadIdx.x; idx < hidden_size; idx += BLOCK_SIZE) {
        float fp8_val = smem_row[idx] * inv_scale;
        out_fp8[row_offset + idx] = __nv_fp8_e4m3(fp8_val);
    }
}

// Host 端启动封装函数
void launch_fused_add_rmsnorm_fp8(
    void* out_fp8,
    float* scales,
    void* residual,
    const void* input,
    const void* weight,
    int num_tokens,
    int hidden_size,
    float epsilon,
    cudaStream_t stream = 0
) {
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);
    size_t smem_bytes = hidden_size * sizeof(float);

    fused_add_rmsnorm_dynamic_fp8_kernel<BLOCK_SIZE><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        scales,
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        hidden_size,
        epsilon
    );
}
