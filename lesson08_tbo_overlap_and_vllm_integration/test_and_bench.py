import os
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 8 课练手与评测脚本：双微批次计算与 PCIe 通信重叠 (TBO) 实战验证
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void run_sequential_pipeline(
    const void* in0, const void* in1, const void* weight,
    void* mid0, void* mid1, void* out0, void* out1,
    int num_elements, int compute_iters, int comm_delay_loops,
    cudaStream_t default_stream
);

void run_tbo_overlapped_pipeline(
    const void* in0, const void* in1, const void* weight,
    void* mid0, void* mid1, void* out0, void* out1,
    int num_elements, int compute_iters, int comm_delay_loops,
    cudaStream_t compute_stream,
    cudaStream_t comm_stream,
    cudaEvent_t evt_comp0_done,
    cudaEvent_t evt_comp1_done,
    cudaEvent_t evt_comm_all_done
);

class TBORunner {
public:
    cudaStream_t comm_stream;
    cudaEvent_t evt0, evt1, evt_done;

    TBORunner() {
        int least_prio, greatest_prio;
        cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio);
        // 创建最高硬件优先级的通信流，确保一旦 Microbatch 0 计算完，硬件调度器立即抢占 16 个 CTA 启动通信！
        cudaStreamCreateWithPriority(&comm_stream, cudaStreamNonBlocking, greatest_prio);
        cudaEventCreateWithFlags(&evt0, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evt1, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evt_done, cudaEventDisableTiming);
    }

    ~TBORunner() {
        cudaEventDestroy(evt0);
        cudaEventDestroy(evt1);
        cudaEventDestroy(evt_done);
        cudaStreamDestroy(comm_stream);
    }

    void run_seq(torch::Tensor& in0, torch::Tensor& in1, torch::Tensor& weight,
                 torch::Tensor& mid0, torch::Tensor& mid1, torch::Tensor& out0, torch::Tensor& out1,
                 int compute_iters, int comm_delay_loops) {
        cudaStream_t s = at::cuda::getCurrentCUDAStream();
        run_sequential_pipeline(
            in0.data_ptr(), in1.data_ptr(), weight.data_ptr(),
            mid0.data_ptr(), mid1.data_ptr(), out0.data_ptr(), out1.data_ptr(),
            in0.numel(), compute_iters, comm_delay_loops, s
        );
    }

    void run_tbo(torch::Tensor& in0, torch::Tensor& in1, torch::Tensor& weight,
                 torch::Tensor& mid0, torch::Tensor& mid1, torch::Tensor& out0, torch::Tensor& out1,
                 int compute_iters, int comm_delay_loops) {
        cudaStream_t comp_s = at::cuda::getCurrentCUDAStream();
        run_tbo_overlapped_pipeline(
            in0.data_ptr(), in1.data_ptr(), weight.data_ptr(),
            mid0.data_ptr(), mid1.data_ptr(), out0.data_ptr(), out1.data_ptr(),
            in0.numel(), compute_iters, comm_delay_loops,
            comp_s, comm_stream, evt0, evt1, evt_done
        );
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<TBORunner>(m, "TBORunner")
        .def(py::init<>())
        .def("run_seq", &TBORunner::run_seq)
        .def("run_tbo", &TBORunner::run_tbo);
}
"""

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本仓库同步至 RTX 6000D 服务器或 Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(cur_dir, "tbo_compute_comm_overlap.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 tbo_compute_comm_overlap.cu ...")
    ext = load_inline(
        name="lesson08_tbo_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    runner = ext.TBORunner()
    num_elements = 256 * 1024
    compute_iters = 512
    comm_delay_loops = 1024

    in0 = torch.randn((num_elements,), dtype=torch.bfloat16, device="cuda")
    in1 = torch.randn((num_elements,), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((4096,), dtype=torch.bfloat16, device="cuda")

    mid0_seq, mid1_seq = torch.empty_like(in0), torch.empty_like(in1)
    out0_seq, out1_seq = torch.empty_like(in0), torch.empty_like(in1)
    mid0_tbo, mid1_tbo = torch.empty_like(in0), torch.empty_like(in1)
    out0_tbo, out1_tbo = torch.empty_like(in0), torch.empty_like(in1)

    runner.run_seq(in0, in1, weight, mid0_seq, mid1_seq, out0_seq, out1_seq, compute_iters, comm_delay_loops)
    runner.run_tbo(in0, in1, weight, mid0_tbo, mid1_tbo, out0_tbo, out1_tbo, compute_iters, comm_delay_loops)
    torch.cuda.synchronize()

    assert torch.equal(out0_seq, out0_tbo) and torch.equal(out1_seq, out1_tbo)
    print("✅ TBO 双流计算通信重叠结果与串行执行 100% 比特级一致！")

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    iters = 100

    start_evt.record()
    for _ in range(iters):
        runner.run_seq(in0, in1, weight, mid0_seq, mid1_seq, out0_seq, out1_seq, compute_iters, comm_delay_loops)
    end_evt.record()
    torch.cuda.synchronize()
    seq_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    start_evt.record()
    for _ in range(iters):
        runner.run_tbo(in0, in1, weight, mid0_tbo, mid1_tbo, out0_tbo, out1_tbo, compute_iters, comm_delay_loops)
    end_evt.record()
    torch.cuda.synchronize()
    tbo_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    print(f"⏱️  传统串行流水线耗时:        {seq_us:.1f} us")
    print(f"🚀 TBO 双流计算通信重叠耗时:  {tbo_us:.1f} us (隐藏通信后提速: {seq_us / tbo_us:.2f}x)")

if __name__ == "__main__":
    main()
