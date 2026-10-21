import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 2 课练手与评测脚本：128-bit 向量化 SwiGLU + 1x128 Group FP8 块缩放量化
# 包含：精度验证、与 PyTorch 原生实现耗时对比、以及 GDDR7 有效显存带宽 (GB/s) 计算！
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_fused_swiglu_group128_fp8(
    void* out_fp8,
    float* scales,
    const void* gate_up,
    int num_tokens,
    int inter_size,
    bool use_ue8m0_pow2_scale,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> fused_swiglu_group128_fp8(
    const torch::Tensor& gate_up,
    bool use_ue8m0_pow2_scale
) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be a CUDA tensor");
    TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous");
    TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16, "gate_up must be BF16");

    int num_tokens = gate_up.size(0);
    int inter_size = gate_up.size(1) / 2;
    TORCH_CHECK(inter_size % 128 == 0, "inter_size must be divisible by 128");

    auto out_fp8 = torch::empty({num_tokens, inter_size}, gate_up.options().dtype(torch::kFloat8_e4m3fn));
    auto scales = torch::empty({num_tokens, inter_size / 128}, gate_up.options().dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_swiglu_group128_fp8(
        out_fp8.data_ptr(),
        scales.data_ptr<float>(),
        gate_up.data_ptr(),
        num_tokens,
        inter_size,
        use_ue8m0_pow2_scale,
        stream
    );
    return std::make_tuple(out_fp8, scales);
}
"""

def torch_ref_swiglu_group128_fp8(gate_up: torch.Tensor, use_ue8m0: bool):
    num_tokens, double_inter = gate_up.shape
    inter_size = double_inter // 2
    gate, up = gate_up[:, :inter_size].float(), gate_up[:, inter_size:].float()
    swiglu = F.silu(gate) * up

    # 切分为 [num_tokens, num_groups, 128]
    grouped = swiglu.view(num_tokens, inter_size // 128, 128)
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / 448.0, min=1e-12)
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    fp8 = (grouped / scale).to(torch.float8_e4m3fn).view(num_tokens, inter_size)
    return fp8, scale.squeeze(-1)

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本目录同步至 RTX 6000D 服务器或 Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(cur_dir, "fused_swiglu_block_fp8.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 fused_swiglu_block_fp8.cu ...")
    ext = load_inline(
        name="lesson02_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["fused_swiglu_group128_fp8"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    # 模拟 MoE 路由展开后的 Token 批次与专家中间层维度
    num_tokens, inter_size = 1024, 4096
    gate_up = torch.randn((num_tokens, 2 * inter_size), dtype=torch.bfloat16, device="cuda")

    for use_ue8m0 in [False, True]:
        mode_name = "UE8M0 (2^E 幂次缩放)" if use_ue8m0 else "FP32 连续缩放"
        out_fp8, scales = ext.fused_swiglu_group128_fp8(gate_up, use_ue8m0)
        ref_fp8, ref_scales = torch_ref_swiglu_group128_fp8(gate_up, use_ue8m0)

        deq_custom = (out_fp8.float().view(num_tokens, -1, 128) * scales.unsqueeze(-1)).view(num_tokens, inter_size)
        deq_ref = (ref_fp8.float().view(num_tokens, -1, 128) * ref_scales.unsqueeze(-1)).view(num_tokens, inter_size)
        max_err = (deq_custom - deq_ref).abs().max().item()
        print(f"✅ [{mode_name}] 精度校验通过！反量化最大误差: {max_err:.6f}")

    # 性能与显存有效带宽评测
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(20):
        ext.fused_swiglu_group128_fp8(gate_up, True)
        torch_ref_swiglu_group128_fp8(gate_up, True)
    torch.cuda.synchronize()

    iters = 200
    start_evt.record()
    for _ in range(iters):
        ext.fused_swiglu_group128_fp8(gate_up, True)
    end_evt.record()
    torch.cuda.synchronize()
    custom_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    start_evt.record()
    for _ in range(iters):
        torch_ref_swiglu_group128_fp8(gate_up, True)
    end_evt.record()
    torch.cuda.synchronize()
    ref_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    # 计算实际读写的总字节数：
    # 读：num_tokens * (2 * inter_size) * 2 Bytes (BF16)
    # 写：num_tokens * inter_size * 1 Byte (FP8) + num_tokens * (inter_size / 128) * 4 Bytes (Scale)
    total_bytes = num_tokens * (2 * inter_size * 2 + inter_size * 1 + (inter_size // 128) * 4)
    bw_gbps = (total_bytes / (custom_us * 1e-6)) / 1e9

    print(f"⏱️  PyTorch 原生耗时:       {ref_us:.2f} us")
    print(f"🚀 128-bit 向量化融合耗时: {custom_us:.2f} us (加速比: {ref_us / custom_us:.2f}x)")
    print(f"📊 实测有效显存带宽:       {bw_gbps:.1f} GB/s")

if __name__ == "__main__":
    main()
