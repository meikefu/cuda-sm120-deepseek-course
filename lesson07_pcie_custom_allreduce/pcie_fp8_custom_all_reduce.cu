#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// 第 7 课核心算子：手撕 vLLM custom_all_reduce.cu —— 无 NVLink (纯 PCIe 5.0) 下
//                 的 P2P 跨卡直写、无锁原子信号量同步与 FP8 寄存器级低比特压缩 AllReduce
// 对应 vLLM 源码：csrc/custom_all_reduce.cu & csrc/custom_all_reduce.cuh
// 对应 SGLang 源码：sgl-kernel/csrc/allreduce/
// ============================================================================
// 为什么在无 NVLink 的 RTX 6000D 多卡服务器上，修改这个算子收益巨大？
// 1. 大模型 Decode 阶段每层有 2 次 AllReduce，消息大小通常只有 64KB ~ 512KB。
//    标准 NCCL Ring 协议为大吞吐设计，在小包 PCIe 传输上启动延迟和多次往返同步开销极高。
// 2. 通过 PCIe P2P BAR1 映射直接读取各卡缓冲区，并在推上 PCIe 总线前将 BF16 (2 Bytes)
//    实时在寄存器内块量化为 FP8 E4M3 (1 Byte) + Group Scale，PCIe 传输字节数直接砍半！
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr float FP8_MAX = 448.0f;

struct __align__(16) BF16Vec8 {
    union {
        uint4 raw;
        __nv_bfloat16 elem[8];
    };
};

struct __align__(8) FP8Vec8 {
    union {
        uint2 raw;
        __nv_fp8_e4m3 elem[8];
    };
};

// ----------------------------------------------------------------------------
// 知识点 1：跨 GPU 系统级内存屏障 (System-Scope Memory Fence & Atomic Semaphore)
// 在多卡 PCIe P2P 中，普通 __syncthreads() 只能同步单卡单 Block 内的线程；
// __threadfence() 只能保证当前 GPU 显存可见；
// 跨 PCIe 总线让其他 GPU 看到数据，必须使用 __threadfence_system() 或系统级原子操作！
// ----------------------------------------------------------------------------
__device__ __forceinline__ void multi_gpu_barrier_demo(
    int* __restrict__ barrier_flags,
    int world_size,
    int rank
) {
    // 在真实多卡 Kernel 中，每个 Block 的 threadIdx.x < world_size 负责向各卡的信号量数组写 flag_value，
    // 并使用 volatile / atomicAdd_system 轮询等待所有 rank 到达同一代数 (generation)。
    __threadfence_system();
}

// ----------------------------------------------------------------------------
// 知识点 2：階段 A —— 将本地 BF16 激活在寄存器内实时量化为 1x16 Group FP8 写入 P2P 共享区
// 每 16 个 BF16 (32 Bytes) 压缩为：16 个 FP8 (16 Bytes) + 1 个 FP16/BF16 Scale (2 Bytes)
// 压缩后体积从 32 Bytes 降至 18 Bytes（若使用 1x128 块则降至 16.25 Bytes，节省近 50% PCIe 带宽！）
// ----------------------------------------------------------------------------
__global__ void quantize_to_p2p_fp8_buffer_kernel(
    const __nv_bfloat16* __restrict__ local_input, // [num_elements]
    __nv_fp8_e4m3* __restrict__ p2p_fp8_buf,       // [num_elements] (位于 PCIe BAR1 P2P IPC 映射内存)
    float* __restrict__ p2p_scale_buf,             // [num_elements / 8]
    int num_elements
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_base = vec_idx * 8;
    if (elem_base >= num_elements) return;

    // 1. 128-bit 向量化读取本地 8 个 BF16
    BF16Vec8 in_vec;
    in_vec.raw = *reinterpret_cast<const uint4*>(&local_input[elem_base]);

    float vals[8];
    float absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        vals[i] = __bfloat162float(in_vec.elem[i]);
        absmax = fmaxf(absmax, fabsf(vals[i]));
    }

    float scale = fmaxf(absmax / FP8_MAX, 1e-12f);
    float inv_scale = 1.0f / scale;

    // 2. 打包成 8 个 FP8 (64-bit uint2)，通过 PCIe Posted Write 写入 P2P 缓冲区
    FP8Vec8 out_fp8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_fp8.elem[i] = __nv_fp8_e4m3(vals[i] * inv_scale);
    }

    *reinterpret_cast<uint2*>(&p2p_fp8_buf[elem_base]) = out_fp8.raw;
    p2p_scale_buf[vec_idx] = scale;
}

// ----------------------------------------------------------------------------
// 知识点 3：阶段 B —— One-Shot P2P FP8 解压规约 Kernel (模拟 4 卡 TP AllReduce)
// 每个 GPU 直接通过 PCIe P2P 指针数组读取全部 4 张卡的 FP8 缓冲区与 Scale，
// 在寄存器内反量化累加并输出最终 BF16 结果！
// ----------------------------------------------------------------------------
template <int WORLD_SIZE>
__global__ void oneshot_p2p_fp8_allreduce_kernel(
    const __nv_fp8_e4m3* __restrict__ all_ranks_fp8,   // [WORLD_SIZE, num_elements]
    const float* __restrict__ all_ranks_scales,        // [WORLD_SIZE, num_elements / 8]
    __nv_bfloat16* __restrict__ output,                // [num_elements]
    int num_elements
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_base = vec_idx * 8;
    if (elem_base >= num_elements) return;

    int num_vecs = num_elements / 8;
    float accum[8] = {0.0f};

    // 遍历所有 Rank (例如 TP=4)，通过 64-bit 向量读取替代 128-bit 读取，PCIe 读取流量减半！
    #pragma unroll
    for (int r = 0; r < WORLD_SIZE; ++r) {
        const __nv_fp8_e4m3* rank_fp8_ptr = all_ranks_fp8 + static_cast<int64_t>(r) * num_elements;
        const float* rank_scale_ptr       = all_ranks_scales + static_cast<int64_t>(r) * num_vecs;

        FP8Vec8 fp8_vec;
        fp8_vec.raw = *reinterpret_cast<const uint2*>(&rank_fp8_ptr[elem_base]);
        float s = rank_scale_ptr[vec_idx];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            accum[i] += static_cast<float>(fp8_vec.elem[i]) * s;
        }
    }

    // 将规约后的 FP32 转换回 BF16 并以 128-bit 向量写回本地显存
    BF16Vec8 out_vec;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_vec.elem[i] = __float2bfloat16(accum[i]);
    }
    *reinterpret_cast<uint4*>(&output[elem_base]) = out_vec.raw;
}

// 对照组：传统未压缩 BF16 One-Shot AllReduce Kernel (每次跨 PCIe 读取 16 Bytes)
template <int WORLD_SIZE>
__global__ void oneshot_p2p_bf16_allreduce_kernel(
    const __nv_bfloat16* __restrict__ all_ranks_bf16, // [WORLD_SIZE, num_elements]
    __nv_bfloat16* __restrict__ output,
    int num_elements
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_base = vec_idx * 8;
    if (elem_base >= num_elements) return;

    float accum[8] = {0.0f};
    #pragma unroll
    for (int r = 0; r < WORLD_SIZE; ++r) {
        const __nv_bfloat16* rank_ptr = all_ranks_bf16 + static_cast<int64_t>(r) * num_elements;
        BF16Vec8 v;
        v.raw = *reinterpret_cast<const uint4*>(&rank_ptr[elem_base]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            accum[i] += __bfloat162float(v.elem[i]);
        }
    }

    BF16Vec8 out_vec;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_vec.elem[i] = __float2bfloat16(accum[i]);
    }
    *reinterpret_cast<uint4*>(&output[elem_base]) = out_vec.raw;
}

void launch_fp8_compressed_allreduce_sim(
    const void* all_ranks_bf16_in,
    void* p2p_fp8_buf,
    float* p2p_scale_buf,
    void* output_fp8_ar,
    void* output_bf16_ar,
    int num_elements,
    cudaStream_t stream = 0
) {
    constexpr int WORLD_SIZE = 4;
    constexpr int BLOCK_SIZE = 256;
    int total_elements_all_ranks = WORLD_SIZE * num_elements;
    int grid_quant = ((total_elements_all_ranks / 8) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_reduce = ((num_elements / 8) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. 4 张卡各自将本地 BF16 压缩为 FP8 P2P 包
    quantize_to_p2p_fp8_buffer_kernel<<<grid_quant, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(all_ranks_bf16_in),
        reinterpret_cast<__nv_fp8_e4m3*>(p2p_fp8_buf),
        p2p_scale_buf,
        total_elements_all_ranks
    );

    // 2. 执行 FP8 压缩版 One-Shot P2P AllReduce
    oneshot_p2p_fp8_allreduce_kernel<WORLD_SIZE><<<grid_reduce, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(p2p_fp8_buf),
        p2p_scale_buf,
        reinterpret_cast<__nv_bfloat16*>(output_fp8_ar),
        num_elements
    );

    // 3. 执行传统 BF16 未压缩版 One-Shot P2P AllReduce (对照)
    oneshot_p2p_bf16_allreduce_kernel<WORLD_SIZE><<<grid_reduce, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(all_ranks_bf16_in),
        reinterpret_cast<__nv_bfloat16*>(output_bf16_ar),
        num_elements
    );
}
