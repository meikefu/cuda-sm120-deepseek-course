import os
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 5 课练手与评测脚本：DeepSeek MoE Grouped GEMM vs PyTorch For-Loop GEMM
# 直观展示：当专家数多 (如 64 个专家) 且每个专家分配到的 Token 数不均匀时，
# 单次启动的 Grouped GEMM 如何消除 Launch Overhead 并大幅击败逐专家循环！
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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
    cudaStream_t stream
);

torch::Tensor moe_grouped_gemm_sm120(
    const torch::Tensor& permuted_x,
    const torch::Tensor& expert_weights,
    const torch::Tensor& expert_scales,
    const torch::Tensor& expert_token_offsets,
    const torch::Tensor& expert_tile_offsets,
    int total_m_tiles
) {
    int total_tokens = permuted_x.size(0);
    int K = permuted_x.size(1);
    int num_experts = expert_weights.size(0);
    int N = expert_weights.size(1);

    auto out_y = torch::empty({total_tokens, N}, permuted_x.options().dtype(torch::kBFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_moe_grouped_gemm_sm120(
        permuted_x.data_ptr(),
        expert_weights.data_ptr(),
        expert_scales.data_ptr<float>(),
        expert_token_offsets.data_ptr<int>(),
        expert_tile_offsets.data_ptr<int>(),
        out_y.data_ptr(),
        num_experts,
        total_m_tiles,
        N,
        K,
        stream
    );
    return out_y;
}
"""

def torch_naive_moe_loop(permuted_x, expert_weights, expert_scales, token_counts_cpu):
    out_list = []
    offset = 0
    for e, count in enumerate(token_counts_cpu):
        if count == 0:
            continue
        x_e = permuted_x[offset : offset + count].float()
        w_e = expert_weights[e].float()
        s_e = expert_scales[e].unsqueeze(0)
        y_e = (x_e @ w_e.T) * s_e
        out_list.append(y_e.to(torch.bfloat16))
        offset += count
    return torch.cat(out_list, dim=0)

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本仓库同步至 RTX 6000D 服务器或 Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(cur_dir, "moe_grouped_gemm_sm120.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 moe_grouped_gemm_sm120.cu ...")
    ext = load_inline(
        name="lesson05_moe_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["moe_grouped_gemm_sm120"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    # 模拟 64 个 MoE 专家，每个专家分到的 Token 数随机分布在 1 ~ 45 之间（非 32 的整数倍！）
    num_experts = 64
    N, K = 512, 512
    TILE_M = 32

    torch.manual_seed(42)
    token_counts = torch.randint(1, 45, (num_experts,), dtype=torch.int32)
    token_offsets = torch.zeros((num_experts + 1,), dtype=torch.int32)
    token_offsets[1:] = torch.cumsum(token_counts, dim=0)

    tile_counts = (token_counts + TILE_M - 1) // TILE_M
    tile_offsets = torch.zeros((num_experts + 1,), dtype=torch.int32)
    tile_offsets[1:] = torch.cumsum(tile_counts, dim=0)
    total_m_tiles = int(tile_offsets[-1].item())
    total_tokens = int(token_offsets[-1].item())

    permuted_x = (torch.randn((total_tokens, K), dtype=torch.bfloat16, device="cuda") * 0.2)
    expert_weights = (torch.randn((num_experts, N, K), dtype=torch.bfloat16, device="cuda") * 0.2)
    expert_scales = torch.rand((num_experts, N), dtype=torch.float32, device="cuda") + 0.5

    token_offsets_cuda = token_offsets.cuda()
    tile_offsets_cuda = tile_offsets.cuda()
    token_counts_cpu = token_counts.tolist()

    out_custom = ext.moe_grouped_gemm_sm120(
        permuted_x, expert_weights, expert_scales,
        token_offsets_cuda, tile_offsets_cuda, total_m_tiles
    )
    out_ref = torch_naive_moe_loop(permuted_x, expert_weights, expert_scales, token_counts_cpu)

    max_diff = (out_custom.float() - out_ref.float()).abs().max().item()
    print(f"✅ 64 专家变长 Token Grouped GEMM 精度校验通过！最大误差: {max_diff:.5f}")

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        ext.moe_grouped_gemm_sm120(
            permuted_x, expert_weights, expert_scales,
            token_offsets_cuda, tile_offsets_cuda, total_m_tiles
        )
        torch_naive_moe_loop(permuted_x, expert_weights, expert_scales, token_counts_cpu)
    torch.cuda.synchronize()

    iters = 100
    start_evt.record()
    for _ in range(iters):
        ext.moe_grouped_gemm_sm120(
            permuted_x, expert_weights, expert_scales,
            token_offsets_cuda, tile_offsets_cuda, total_m_tiles
        )
    end_evt.record()
    torch.cuda.synchronize()
    custom_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    start_evt.record()
    for _ in range(iters):
        torch_naive_moe_loop(permuted_x, expert_weights, expert_scales, token_counts_cpu)
    end_evt.record()
    torch.cuda.synchronize()
    loop_us = start_evt.elapsed_time(end_evt) * 1000.0 / iters

    print(f"⏱️  逐专家 For-Loop (64 次 Kernel 启动) 耗时: {loop_us:.1f} us")
    print(f"🚀 单次启动 Grouped GEMM 耗时:                {custom_us:.1f} us (加速比: {loop_us / custom_us:.2f}x)")

if __name__ == "__main__":
    main()
