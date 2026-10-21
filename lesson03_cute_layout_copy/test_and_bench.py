import os
import subprocess
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 3 课练手与评测脚本：CuTe Layout, Swizzle<3,3,3> 与 TiledCopy (cp.async)
# 运行前说明：
#   本脚本会自动查找系统中或克隆在本地的 CUTLASS 头文件目录 (include/)。
#   如果未找到，会自动使用 git clone --depth 1 拉取官方 NVIDIA/cutlass 头文件（仅需几秒钟）。
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_cute_demo(
    const void* ptr_in,
    void* ptr_out,
    int M,
    int K,
    bool print_layout_info,
    cudaStream_t stream
);

torch::Tensor run_cute_tiled_copy(const torch::Tensor& x, bool print_layout_info) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");

    int M = x.size(0);
    int K = x.size(1);
    TORCH_CHECK(M % 64 == 0 && K % 64 == 0, "M and K must be multiples of 64");

    auto out = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_cute_demo(x.data_ptr(), out.data_ptr(), M, K, print_layout_info, stream);
    return out;
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
    cu_path = os.path.join(cur_dir, "cute_layout_and_copy.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 CuTe 算子 cute_layout_and_copy.cu ...")
    ext = load_inline(
        name="lesson03_cute_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["run_cute_tiled_copy"],
        extra_include_paths=[cutlass_inc],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        verbose=False,
    )

    M, K = 4096, 4096
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    # 第一次调用：打印 Swizzle<3,3,3> 消除 Bank Conflict 的物理地址映射表，并校验结果
    out = ext.run_cute_tiled_copy(x, True)
    torch.cuda.synchronize()

    if torch.equal(x, out):
        print("\n✅ CuTe TiledCopy (cp.async + Swizzle<3,3,3>) 数据完整性 100% 校验通过！")
    else:
        print("\n❌ 校验失败！")

if __name__ == "__main__":
    main()
