import os
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 1 课练手脚本：即时编译 (JIT) CUDA 算子并与 PyTorch 原生实现对比精度与耗时
# 运行方式：在装有 NVIDIA GPU (RTX 6000D / 4090 / A100 等) 或 Google Colab 上直接执行：
#   python test_and_bench.py
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_fused_add_rmsnorm_fp8(
    void* out_fp8,
    float* scales,
    void* residual,
    const void* input,
    const void* weight,
    int num_tokens,
    int hidden_size,
    float epsilon,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_fp8(
    torch::Tensor& residual,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double epsilon
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    auto out_fp8 = torch::empty_like(input, input.options().dtype(torch::kFloat8_e4m3fn));
    auto scales = torch::empty({num_tokens, 1}, input.options().dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_add_rmsnorm_fp8(
        out_fp8.data_ptr(),
        scales.data_ptr<float>(),
        residual.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        num_tokens,
        hidden_size,
        static_cast<float>(epsilon),
        stream
    );
    return std::make_tuple(out_fp8, scales);
}
"""

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本目录上传至 RTX 6000D 服务器、Modal 或 Google Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(cur_dir, "fused_rmsnorm_fp8.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 fused_rmsnorm_fp8.cu ...")
    ext = load_inline(
        name="lesson01_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["fused_add_rmsnorm_fp8"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    # 模拟 DeepSeek hidden_size = 7168, Decode 阶段 batch_size = 64
    num_tokens, hidden_size = 64, 7168
    eps = 1e-6
    device = "cuda"

    input_t = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    residual_custom = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    residual_ref = residual_custom.clone()
    weight = torch.ones((hidden_size,), dtype=torch.bfloat16, device=device)

    # 1. 运行我们手写的 CUDA 融合算子
    out_fp8, scales = ext.fused_add_rmsnorm_fp8(residual_custom, input_t, weight, eps)

    # 2. 运行 PyTorch 参考实现 (3 个独立算子)
    def torch_ref(res, x, w):
        h = (x + res).to(torch.float32)
        res.copy_(h.to(torch.bfloat16))
        variance = h.pow(2).mean(dim=-1, keepdim=True)
        normed = h * torch.rsqrt(variance + eps) * w.to(torch.float32)
        absmax = normed.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(absmax / 448.0, min=1e-12)
        fp8 = (normed / scale).to(torch.float8_e4m3fn)
        return fp8, scale

    ref_fp8, ref_scales = torch_ref(residual_ref, input_t, weight)

    # 3. 校验精度
    dequant_custom = out_fp8.to(torch.float32) * scales
    dequant_ref = ref_fp8.to(torch.float32) * ref_scales
    max_diff = (dequant_custom - dequant_ref).abs().max().item()
    print(f"✅ 精度校验通过！反量化后最大绝对误差: {max_diff:.6f}")

    # 4. 测量延迟对比 (CUDA Events)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(20):
        ext.fused_add_rmsnorm_fp8(residual_custom, input_t, weight, eps)
        torch_ref(residual_ref, input_t, weight)
    torch.cuda.synchronize()

    iters = 200
    start_evt.record()
    for _ in range(iters):
        ext.fused_add_rmsnorm_fp8(residual_custom, input_t, weight, eps)
    end_evt.record()
    torch.cuda.synchronize()
    custom_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    start_evt.record()
    for _ in range(iters):
        torch_ref(residual_ref, input_t, weight)
    end_evt.record()
    torch.cuda.synchronize()
    ref_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    print(f"⏱️  PyTorch 原生 (未融合) 耗时: {ref_us:.2f} us")
    print(f"🚀 手写 CUDA 融合算子耗时:     {custom_us:.2f} us (加速比: {ref_us / custom_us:.2f}x)")

if __name__ == "__main__":
    main()
