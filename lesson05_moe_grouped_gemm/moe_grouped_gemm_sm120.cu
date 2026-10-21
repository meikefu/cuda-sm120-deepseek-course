#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// 第 5 课核心算子：DeepSeek MoE 的心脏 —— 动态路由 Grouped GEMM (分组矩阵乘)
// 对应 vLLM 源码：csrc/moe/ & csrc/quantization/cutlass_w8a8/moe/
// 对应 SGLang / FlashInfer 源码：include/flashinfer/gemm/group_gemm.cuh
// ============================================================================
// 为什么 MoE 绝对不能用 Python for 循环调 cuBLAS？
// DeepSeek-V4.1-Flash 有成百上千个细粒度专家 (Routed Experts)。
// 在推理时，第 e 号专家分到的 Token 数量 M_e 是动态变化的（有的专家分到 150 个 Token，
// 有的分到 7 个 Token）。如果用 for 循环启动 256 次小 GEMM Kernel：
//   1. CPU 启动 256 个 Kernel 的 Launch Overhead 高达上千微秒，远超 GPU 计算时间；
//   2. 每个小矩阵乘连 RTX 6000D 的 100+ 个 SM 都占不满（严重的 Wave Quantization 浪费）。
// 解决方案：使用单次启动的 Grouped GEMM + 扁平化 Tile Scheduler (动态瓦片调度器)！
// ============================================================================

constexpr int TILE_M = 32;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

// ----------------------------------------------------------------------------
// 核心数据结构：在设备端通过二分查找 / 线性扫描定位当前 CTA (blockIdx.x)
// 属于哪一个 MoE 专家 (expert_id)，以及在该专家内部的 (m_tile, n_tile) 坐标！
// ----------------------------------------------------------------------------
__device__ __forceinline__ int find_expert_for_m_tile(
    const int* __restrict__ expert_tile_offsets, // 长度为 num_experts + 1 的 M-tile 前缀和数组
    int num_experts,
    int global_m_tile_idx
) {
    // 二分查找：找到满足 expert_tile_offsets[e] <= global_m_tile_idx < expert_tile_offsets[e+1] 的专家 e
    int left = 0, right = num_experts - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (global_m_tile_idx < expert_tile_offsets[mid]) {
            right = mid - 1;
        } else if (global_m_tile_idx >= expert_tile_offsets[mid + 1]) {
            left = mid + 1;
        } else {
            return mid;
        }
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Grouped GEMM 主 Kernel（带路由权重 topk_weight 尾声融合）
// 输入：
//   permuted_x:          [total_tokens, K] (按专家排好序的输入 Token 激活)
//   expert_weights:      [num_experts, N, K] (所有专家的权重张量，K 维连续)
//   expert_token_offsets:[num_experts + 1] (每个专家在 permuted_x 中的起始/结束 Token 行号前缀和)
//   expert_tile_offsets: [num_experts + 1] (每个专家占用的 M-Tile 数量前缀和，用于 CTA 调度)
//   expert_scales:       [num_experts, N] (每个专家输出通道的量化反缩放因子 / 权重缩放)
// 输出：
//   out_y:               [total_tokens, N]
// ----------------------------------------------------------------------------
__global__ void moe_grouped_gemm_sm120_kernel(
    const __nv_bfloat16* __restrict__ permuted_x,
    const __nv_bfloat16* __restrict__ expert_weights,
    const float* __restrict__ expert_scales,
    const int* __restrict__ expert_token_offsets,
    const int* __restrict__ expert_tile_offsets,
    __nv_bfloat16* __restrict__ out_y,
    int num_experts,
    int N,
    int K
) {
    // 1. 扁平化 Tile Scheduler：
    //    blockIdx.x 代表全局展平后的第 global_m_tile_idx 个 M-Tile
    //    blockIdx.y 代表第 n_tile_idx 个 N-Tile
    int global_m_tile_idx = blockIdx.x;
    int total_m_tiles = expert_tile_offsets[num_experts];
    if (global_m_tile_idx >= total_m_tiles) {
        return;
    }

    // 2. 通过二分查找瞬间定位当前 Block 归属的专家编号 expert_id
    int expert_id = find_expert_for_m_tile(expert_tile_offsets, num_experts, global_m_tile_idx);
    int local_m_tile_idx = global_m_tile_idx - expert_tile_offsets[expert_id];

    // 当前专家在 permuted_x 中的起始行与真实 Token 数 M_e
    int token_start = expert_token_offsets[expert_id];
    int token_end   = expert_token_offsets[expert_id + 1];
    int M_e         = token_end - token_start;

    int m_base = local_m_tile_idx * TILE_M;
    int n_base = blockIdx.y * TILE_N;

    // 定位当前专家的输入指针、权重指针和输出指针
    const __nv_bfloat16* A_ptr = permuted_x + static_cast<int64_t>(token_start) * K;
    const __nv_bfloat16* B_ptr = expert_weights + static_cast<int64_t>(expert_id) * N * K;
    const float* scale_ptr     = expert_scales + static_cast<int64_t>(expert_id) * N;
    __nv_bfloat16* C_ptr       = out_y + static_cast<int64_t>(token_start) * N;

    // Shared Memory 分块缓存 (TILE_M x TILE_K + TILE_N x TILE_K)
    // Padding +8 避免标量演示中的 Bank Conflict（生产级可结合第 3/4 课的 CuTe Swizzle + TiledMMA）
    __shared__ float sA[TILE_M][TILE_K + 2];
    __shared__ float sB[TILE_N][TILE_K + 2];

    // 线程布局：256 个线程 (16 x 16)，每个线程计算 (TILE_M/16) x (TILE_N/16) = 2 x 4 个输出元素
    int tx = threadIdx.x % 16; // 0..15 (对应 N 方向)
    int ty = threadIdx.x / 16; // 0..15 (对应 M 方向)

    float acc[2][4] = {0.0f};

    // 3. 沿 K 维分块循环累加
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        // 协作加载 A 分块 (32 x 32 = 1024 元素，256 线程每人加载 4 个)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = threadIdx.x + i * 256;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int global_r = m_base + r;
            int global_c = k_base + c;
            // 关键：因为每个专家的 Token 数 M_e 通常不是 TILE_M 的整数倍，必须做谓词边界保护 (Predicated Load)！
            sA[r][c] = (global_r < M_e && global_c < K)
                       ? __bfloat162float(A_ptr[global_r * K + global_c])
                       : 0.0f;
        }

        // 协作加载 B 分块 (64 x 32 = 2048 元素，256 线程每人加载 8 个)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = threadIdx.x + i * 256;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int global_r = n_base + r;
            int global_c = k_base + c;
            sB[r][c] = (global_r < N && global_c < K)
                       ? __bfloat162float(B_ptr[global_r * K + global_c])
                       : 0.0f;
        }
        __syncthreads();

        // 计算 2x4 寄存器块乘加
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_reg[2];
            a_reg[0] = sA[ty * 2 + 0][k];
            a_reg[1] = sA[ty * 2 + 1][k];

            float b_reg[4];
            b_reg[0] = sB[tx * 4 + 0][k];
            b_reg[1] = sB[tx * 4 + 1][k];
            b_reg[2] = sB[tx * 4 + 2][k];
            b_reg[3] = sB[tx * 4 + 3][k];

            #pragma unroll
            for (int im = 0; im < 2; ++im) {
                #pragma unroll
                for (int in = 0; in < 4; ++in) {
                    acc[im][in] += a_reg[im] * b_reg[in];
                }
            }
        }
        __syncthreads();
    }

    // 4. Epilogue：融合专家缩放因子并只写回 < M_e 的有效 Token 行
    #pragma unroll
    for (int im = 0; im < 2; ++im) {
        int global_m = m_base + ty * 2 + im;
        if (global_m < M_e) {
            #pragma unroll
            for (int in = 0; in < 4; ++in) {
                int global_n = n_base + tx * 4 + in;
                if (global_n < N) {
                    float val = acc[im][in] * scale_ptr[global_n];
                    C_ptr[global_m * N + global_n] = __float2bfloat16(val);
                }
            }
        }
    }
}

void launch_moe_grouped_gemm_sm120(
    const void* permuted_x,
    const void* expert_weights,
    const float* expert_scales,
    const int* expert_token_offsets,
    const int* expert_tile_offsets,
    void* out_y,
    int num_experts,
    int total_m_tiles,
    int N,
    int K,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid(total_m_tiles, (N + TILE_N - 1) / TILE_N);

    moe_grouped_gemm_sm120_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(permuted_x),
        reinterpret_cast<const __nv_bfloat16*>(expert_weights),
        expert_scales,
        expert_token_offsets,
        expert_tile_offsets,
        reinterpret_cast<__nv_bfloat16*>(out_y),
        num_experts,
        N,
        K
    );
}
