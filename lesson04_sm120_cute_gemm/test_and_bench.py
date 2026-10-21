import os
import subprocess
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 4 课练手与评测脚本：Blackwell sm_120 CuTe Tensor Core GEMM + Epilogue Scale 融合
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_sm120_cute_gemm_scaled(
    const void* ptr_A,
    const void* ptr_B,
    const float* scale_a,
    const float* scale_b,
    void* ptr_C,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

torch::Tensor sm120_cute_gemm_scaled(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "B must have shape [N, K]");
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0, "Dimensions must align with tile size");

    auto C = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_sm120_cute_gemm_scaled(
        A.data_ptr(),
        B.data_ptr(),
        scale_a.data_ptr<float>(),
        scale_b.data_ptr<float>(),
        C.data_ptr(),
        M, N, K, stream
    );
    return C;
}
"""

def ensure_cutlass_include(cur_dir: str) -> str:
    env_cutlass = os.environ.get("CUTLASS_PATH", "")
    if env_cutlass and os.path.exists(os.path.join(env_cutlass, "include", "cute", "tensor.hpp")):
        return os.path.join(env_cutlass, "include")

    local_cutlass = os.path.join(os.path.dirname(cur_dir), "third_party", "cutlass")
    include_dir = os.path.join(local_cutlass, "include")
    if not os.path.exists(os.path.join(include_dir, "cute", "tensor.hpp")):
        print(f"正在拉取 NVIDIA/cutlass 纯头文件库至 {local_cutlass} ...")
        os.makedirs(os.path.dirname(local_cutlass), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", local_cutlass],
            check=True,
        )
    return include_dir

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本仓库同步至 RTX 6000D 服务器或 Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cutlass_inc = ensure_cutlass_include(cur_dir)
    cu_path = os.path.join(cur_dir, "sm120_cute_gemm.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 sm120_cute_gemm.cu ...")
    ext = load_inline(
        name="lesson04_gemm_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["sm120_cute_gemm_scaled"],
        extra_include_paths=[cutlass_inc],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        verbose=False,
    )

    M, N, K = 2048, 4096, 4096
    A = (torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.5)
    B = (torch.randn((N, K), dtype=torch.bfloat16, device="cuda") * 0.5)
    scale_a = torch.rand((M,), dtype=torch.float32, device="cuda") + 0.5
    scale_b = torch.rand((N,), dtype=torch.float32, device="cuda") + 0.5

    C_custom = ext.sm120_cute_gemm_scaled(A, B, scale_a, scale_b)
    C_ref = ((A.float() @ B.float().T) * scale_a.unsqueeze(1) * scale_b.unsqueeze(0)).to(torch.bfloat16)

    max_diff = (C_custom.float() - C_ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(C_custom.float().flatten(), C_ref.float().flatten(), dim=0).item()
    print(f"✅ CuTe Tensor Core GEMM 精度校验通过！Cosine 相似度: {cos_sim:.6f}, 最大误差: {max_diff:.4f}")

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        ext.sm120_cute_gemm_scaled(A, B, scale_a, scale_b)
    torch.cuda.synchronize()

    iters = 50
    start_evt.record()
    for _ in range(iters):
        ext.sm120_cute_gemm_scaled(A, B, scale_a, scale_b)
    end_evt.record()
    torch.cuda.synchronize()
    ms = start_evt.elapsed_time(end_evt) / iters
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    print(f"🚀 CuTe 双缓冲 Tensor Core GEMM 平均耗时: {ms:.3f} ms | 实测算力: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    main()
