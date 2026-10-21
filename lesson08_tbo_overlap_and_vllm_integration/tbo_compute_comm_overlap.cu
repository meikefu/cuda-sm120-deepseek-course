#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// 第 8 课核心算子：双微批次计算与 PCIe 通信重叠 (Two-Microbatch Overlap, TBO)
//                 与高优先级多流事件调度实战
// 对应 SGLang / vLLM 源码：
//   - SGLang: python/sglang/srt/two_batch_overlap.py & sgl-kernel
//   - DeepSeek 开源技术报告核心架构：DualPipe / Two-Microbatch Overlap (TBO)
// ============================================================================
// 为什么在无 NVLink 的 RTX 6000D 上，TBO 是解决 MoE 通信瓶颈的终极方案？
// 即使我们在第 7 课把 PCIe AllReduce / All-to-All 压缩到了极致，只要通信和计算是串行的：
//   Total Time = T_compute + T_pcie_comm
// GPU 的 Tensor Core 在 PCIe 传输期间依然是 100% 闲置的！
// 通过将当前批次切分为两个微批次 (Microbatch 0 和 Microbatch 1)，并建立：
//   - compute_stream (普通优先级，占用绝大多数 SM 跑 Tensor Core GEMM / Shared Expert)
//   - comm_stream    (最高优先级，仅占用极少几个 SM 或 DMA 引擎跑 PCIe P2P 通信)
// 即可让 Microbatch 0 的 PCIe 通信 与 Microbatch 1 的 MoE/MLA 计算 100% 硬件重叠！
// ============================================================================

// 模拟高算术强度的本地专家 / Shared Expert 矩阵计算 Kernel (占用大量 Tensor Core / ALU 周期)
__global__ void simulated_moe_compute_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int num_elements,
    int compute_iters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = __bfloat162float(input[idx]);
    float w   = __bfloat162float(weight[idx % 4096]);

    // 模拟高密度乘加计算 (Compute-bound)
    #pragma unroll 16
    for (int i = 0; i < compute_iters; ++i) {
        val = fmaf(val, w, 0.0001f);
    }
    output[idx] = __float2bfloat16(val);
}

// 模拟轻量级 SM 占用的 PCIe P2P 跨卡通信与规约 Kernel (仅使用少量 Block，留出 SM 给计算流！)
// 关键工程技巧：通信 Kernel 严禁占满所有 SM，否则计算流的 Kernel 无法同时驻留 (Co-reside)！
__global__ void lightweight_pcie_comm_kernel(
    const __nv_bfloat16* __restrict__ comm_in,
    __nv_bfloat16* __restrict__ comm_out,
    int num_elements,
    int comm_delay_loops
) {
    // 使用 Grid-Stride Loop，哪怕只启动 8 ~ 16 个 Block 也能扫完整个缓冲区！
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_elements;
         idx += blockDim.x * gridDim.x) {
        float v = __bfloat162float(comm_in[idx]);
        // 模拟跨 PCIe 5.0 总线等待延迟 (Memory/Bus latency bound)
        #pragma unroll 8
        for (int d = 0; d < comm_delay_loops; ++d) {
            v = v * 1.00001f + 0.00001f;
        }
        comm_out[idx] = __float2bfloat16(v);
    }
}

// ----------------------------------------------------------------------------
// 方案 A：传统串行流水线 (Compute 0 -> Comm 0 -> Compute 1 -> Comm 1)
// ----------------------------------------------------------------------------
void run_sequential_pipeline(
    const void* in0, const void* in1, const void* weight,
    void* mid0, void* mid1, void* out0, void* out1,
    int num_elements, int compute_iters, int comm_delay_loops,
    cudaStream_t default_stream
) {
    int block = 256;
    int grid_comp = (num_elements + block - 1) / block;
    int grid_comm = 16; // 限制通信 Kernel 仅占用 16 个 CTA (约占 RTX 6000D 10% 的 SM)

    // Microbatch 0: 先计算，再通信
    simulated_moe_compute_kernel<<<grid_comp, block, 0, default_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in0),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(mid0),
        num_elements, compute_iters
    );
    lightweight_pcie_comm_kernel<<<grid_comm, block, 0, default_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(mid0),
        reinterpret_cast<__nv_bfloat16*>(out0),
        num_elements, comm_delay_loops
    );

    // Microbatch 1: 先计算，再通信
    simulated_moe_compute_kernel<<<grid_comp, block, 0, default_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in1),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(mid1),
        num_elements, compute_iters
    );
    lightweight_pcie_comm_kernel<<<grid_comm, block, 0, default_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(mid1),
        reinterpret_cast<__nv_bfloat16*>(out1),
        num_elements, comm_delay_loops
    );
}

// ----------------------------------------------------------------------------
// 方案 B：TBO 双流计算通信重叠流水线 (Compute Stream + High-Priority Comm Stream)
// ----------------------------------------------------------------------------
void run_tbo_overlapped_pipeline(
    const void* in0, const void* in1, const void* weight,
    void* mid0, void* mid1, void* out0, void* out1,
    int num_elements, int compute_iters, int comm_delay_loops,
    cudaStream_t compute_stream,
    cudaStream_t comm_stream,
    cudaEvent_t evt_comp0_done,
    cudaEvent_t evt_comp1_done,
    cudaEvent_t evt_comm_all_done
) {
    int block = 256;
    int grid_comp = (num_elements + block - 1) / block;
    int grid_comm = 16; // 仅用 16 个 CTA 跑 PCIe 通信，把 90% 的 SM 留给 compute_stream！

    // 1. 在 compute_stream 上发射 Microbatch 0 的计算，并打下硬件事件点 evt_comp0_done
    simulated_moe_compute_kernel<<<grid_comp, block, 0, compute_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in0),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(mid0),
        num_elements, compute_iters
    );
    cudaEventRecord(evt_comp0_done, compute_stream);

    // 2. 緊接着在 compute_stream 上无缝发射 Microbatch 1 的计算（无需等待 Microbatch 0 的通信！）
    simulated_moe_compute_kernel<<<grid_comp, block, 0, compute_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in1),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(mid1),
        num_elements, compute_iters
    );
    cudaEventRecord(evt_comp1_done, compute_stream);

    // 3. 在高优先级 comm_stream 上等待 evt_comp0_done，立刻与 Microbatch 1 的计算并行执行 PCIe 通信！
    cudaStreamWaitEvent(comm_stream, evt_comp0_done, 0);
    lightweight_pcie_comm_kernel<<<grid_comm, block, 0, comm_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(mid0),
        reinterpret_cast<__nv_bfloat16*>(out0),
        num_elements, comm_delay_loops
    );

    // 4. 在 comm_stream 上等待 evt_comp1_done，执行 Microbatch 1 的通信
    cudaStreamWaitEvent(comm_stream, evt_comp1_done, 0);
    lightweight_pcie_comm_kernel<<<grid_comm, block, 0, comm_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(mid1),
        reinterpret_cast<__nv_bfloat16*>(out1),
        num_elements, comm_delay_loops
    );

    // 5. 让 compute_stream 等待通信流全部收尾（保持与外部 CUDA Graph 捕获兼容！）
    cudaEventRecord(evt_comm_all_done, comm_stream);
    cudaStreamWaitEvent(compute_stream, evt_comm_all_done, 0);
}
