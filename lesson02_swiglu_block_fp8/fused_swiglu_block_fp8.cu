#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>

// ============================================================================
// 第 2 课核心算子：128-bit 向量化访存 + SwiGLU 激活 + 1x128 Group FP8 块缩放量化
// 对应 vLLM 源码：csrc/activation_kernels.cu & csrc/quantization/fp8/
// 对应 DeepSeek-V3/V4 核心痛点：MoE 专家层 Gate+Up 投影后的激活与细粒度块量化融合
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr float FP8_E4M3_MAX = 448.0f;

// ----------------------------------------------------------------------------
// 知识点 1：128-bit (16 Bytes) 向量化内存对齐联合体
// GPU 显存控制器单次指令最多可令每个线程读取 16 字节 (即 LDG.E.128 指令)。
// 1 个 __nv_bfloat16 是 2 字节，因此打包 8 个 BF16 恰好是 16 字节 (uint4)！
// 1 个 __nv_fp8_e4m3 是 1 字节，因此打包 8 个 FP8 恰好是 8 字节 (uint2)！
// ----------------------------------------------------------------------------
struct __align__(16) PackedBF16x8 {
    union {
        uint4 raw;               // 用于触发单条 128-bit LDG.E.128 访存指令
        __nv_bfloat162 vec2[4];  // 4 对 bfloat162，可用双路半精度硬件指令
        __nv_bfloat16 elem[8];   // 8 个独立的 bfloat16 元素
    };
};

struct __align__(8) PackedFP8x8 {
    union {
        uint2 raw;               // 用于触发单条 64-bit STG.E.64 写回指令
        __nv_fp8_e4m3 elem[8];   // 8 个独立的 FP8 E4M3 元素
    };
};

// ----------------------------------------------------------------------------
// 知识点 2：SiLU (Swish) 激活函数：silu(x) = x / (1 + exp(-x))
// 在 CUDA 中使用内联快速指数 __expf(x) 替代标准 expf(x)，直接映射到 SFU (特殊函数单元)
// 的 MUFU.EX2 硬件指令，吞吐量提升数倍。
// ----------------------------------------------------------------------------
__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ----------------------------------------------------------------------------
// 知识点 3：Sub-Warp (子线程束) 规约 —— 无需 Shared Memory 的 1x128 块规约！
// DeepSeek 的细粒度 FP8 量化以 128 个连续元素为一个 Quantization Group (块)。
// 因为每个线程通过 PackedBF16x8 一次处理 8 个元素：
//   16 个连续线程 × 8 个元素/线程 = 恰好 128 个元素！
// 而 16 个连续线程正好是 半个 Warp (Half-Warp)！
// 因此，我们完全不需要 Shared Memory，也不需要 __syncthreads()，
// 只需要用 __shfl_xor_sync 在 16 个线程内部做 4 次蝶形交换 (offset = 8, 4, 2, 1)！
// ----------------------------------------------------------------------------
__device__ __forceinline__ float half_warp_reduce_max_16(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// ----------------------------------------------------------------------------
// 知识点 4：Blackwell (sm_120 / sm_100) 原生 UE8M0 幂次缩放因子转换
// 在 DeepSeek-V3/V4 与 Blackwell Tensor Core (mxf8f6f4 指令) 中，块缩放因子支持
// UE8M0 格式（无符号 8 位纯指数，无尾数，即 2^E）。
// 将普通正浮点数 raw_scale 向上取整为最近的 2 的整数次幂 (Power-of-Two)：
//   IEEE 754 FP32 格式：1 位符号 + 8 位指数 (bits 30..23) + 23 位尾数 (bits 22..0)
//   只要尾数 > 0，就把指数加 1 并清空尾数，完全通过整数位运算完成，0 分支开销！
// ----------------------------------------------------------------------------
__device__ __forceinline__ float round_up_pow2_ue8m0(float raw_scale) {
    uint32_t bits = __float_as_uint(raw_scale);
    // 如果尾数部分 (低 23 位) 不为 0，向上进位到下一个 2 的幂次
    if ((bits & 0x007fffffu) != 0) {
        bits = (bits & 0xff800000u) + 0x00800000u;
    }
    return __uint_as_float(bits);
}

// ----------------------------------------------------------------------------
// 主 Kernel：Fused SwiGLU + 1x128 Group FP8 Quantization
// 输入：
//   gate_up: [num_tokens, 2 * inter_size] (BF16)
//            前半部分 [0, inter_size) 是 gate，后半部分 [inter_size, 2*inter_size) 是 up
//            要求 inter_size 是 128 的整数倍（DeepSeek/Llama 均满足）
// 输出：
//   out_fp8: [num_tokens, inter_size] (FP8 E4M3)
//   scales:  [num_tokens, inter_size / 128] (FP32, 每 128 个元素 1 个缩放因子)
// ----------------------------------------------------------------------------
template <bool USE_UE8M0_POW2_SCALE>
__global__ void fused_swiglu_group128_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8,
    float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ gate_up,
    const int inter_size
) {
    // 每个线程一次处理 8 个元素 (VEC_SIZE = 8)
    constexpr int VEC_SIZE = 8;
    // 每 16 个连续线程组成一个 Group，共同负责 16 * 8 = 128 个元素
    constexpr int THREADS_PER_GROUP = 16;

    const int token_idx = blockIdx.x;
    const int num_groups_per_token = inter_size / 128;

    // 计算全局线程在当前 Token 内的向量偏移
    // 支持一个 Token 跨多个 Block，或者单 Block 循环处理整个 Token
    const int64_t gate_row_base = static_cast<int64_t>(token_idx) * (2 * inter_size);
    const int64_t up_row_base   = gate_row_base + inter_size;
    const int64_t out_row_base  = static_cast<int64_t>(token_idx) * inter_size;
    const int64_t scale_row_base = static_cast<int64_t>(token_idx) * num_groups_per_token;

    const int lane_in_group = threadIdx.x % THREADS_PER_GROUP; // 0..15

    // 每次循环，一个 Block 处理 blockDim.x * 8 个元素
    for (int elem_base = threadIdx.x * VEC_SIZE;
         elem_base < inter_size;
         elem_base += blockDim.x * VEC_SIZE) {

        // 1. 128-bit 向量化读取：单条指令读入 8 个 gate 和 8 个 up (避免标量碎片化读取)
        PackedBF16x8 gate_vec, up_vec;
        gate_vec.raw = *reinterpret_cast<const uint4*>(&gate_up[gate_row_base + elem_base]);
        up_vec.raw   = *reinterpret_cast<const uint4*>(&gate_up[up_row_base + elem_base]);

        // 2. 在寄存器内计算 SwiGLU = SiLU(gate) * up，并统计当前线程 8 个元素的最大绝对值
        float swiglu_vals[VEC_SIZE];
        float thread_absmax = 0.0f;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            float g = __bfloat162float(gate_vec.elem[i]);
            float u = __bfloat162float(up_vec.elem[i]);
            float val = fast_silu(g) * u;
            swiglu_vals[i] = val;
            thread_absmax = fmaxf(thread_absmax, fabsf(val));
        }

        // 3. Sub-Warp 规约：16 个线程交换寄存器，求出这 128 个元素的 group_absmax
        // 零 Shared Memory 占用，零 Bank Conflict，零 __syncthreads() 停顿！
        float group_absmax = half_warp_reduce_max_16(thread_absmax);

        // 4. 计算当前 1x128 块的 FP8 缩放因子
        float scale = fmaxf(group_absmax / FP8_E4M3_MAX, 1e-12f);
        if constexpr (USE_UE8M0_POW2_SCALE) {
            scale = round_up_pow2_ue8m0(scale);
        }
        float inv_scale = 1.0f / scale;

        // 每个 16 线程组的第 0 号线程负责将该块的 scale 写入显存
        int group_idx = elem_base / 128;
        if (lane_in_group == 0) {
            scales[scale_row_base + group_idx] = scale;
        }

        // 5. 在寄存器内量化为 8 个 FP8 E4M3，并打包成 64-bit (uint2) 一次性写回显存
        PackedFP8x8 out_vec;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vec.elem[i] = __nv_fp8_e4m3(swiglu_vals[i] * inv_scale);
        }
        *reinterpret_cast<uint2*>(&out_fp8[out_row_base + elem_base]) = out_vec.raw;
    }
}

void launch_fused_swiglu_group128_fp8(
    void* out_fp8,
    float* scales,
    const void* gate_up,
    int num_tokens,
    int inter_size,
    bool use_ue8m0_pow2_scale,
    cudaStream_t stream = 0
) {
    // 256 个线程 = 16 个 Sub-Warp Group，单次迭代可处理 256 * 8 = 2048 个元素
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    if (use_ue8m0_pow2_scale) {
        fused_swiglu_group128_fp8_kernel<true><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
            scales,
            reinterpret_cast<const __nv_bfloat16*>(gate_up),
            inter_size
        );
    } else {
        fused_swiglu_group128_fp8_kernel<false><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
            scales,
            reinterpret_cast<const __nv_bfloat16*>(gate_up),
            inter_size
        );
    }
}
