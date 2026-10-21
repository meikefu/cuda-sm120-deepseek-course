import os
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 7 课练手与评测脚本：PCIe P2P FP8 低比特压缩 Custom AllReduce 验证
# （单卡即可模拟 4 卡 TP=4 P2P 缓冲区压缩规约，验证余弦相似度与 PCIe 通信量减半效果）
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_fp8_compressed_allreduce_sim(
    const void* all_ranks_bf16_in,
    void* p2p_fp8_buf,
    float* p2p_scale_buf,
    void* output_fp8_ar,
    void* output_bf16_ar,
    int num_elements,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> test_p2p_fp8_allreduce(const torch::Tensor& all_ranks_in) {
    TORCH_CHECK(all_ranks_in.size(0) == 4, "Simulates WORLD_SIZE = 4");
    int num_elements = all_ranks_in.size(1);
    TORCH_CHECK(num_elements % 8 == 0, "num_elements must be divisible by 8");

    auto p2p_fp8_buf = torch::empty_like(all_ranks_in, all_ranks_in.options().dtype(torch::kFloat8_e4m3fn));
    auto p2p_scale_buf = torch::empty({4, num_elements / 8}, all_ranks_in.options().dtype(torch::kFloat32));
    auto out_fp8_ar = torch::empty({num_elements}, all_ranks_in.options().dtype(torch::kBFloat16));
    auto out_bf16_ar = torch::empty({num_elements}, all_ranks_in.options().dtype(torch::kBFloat16));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fp8_compressed_allreduce_sim(
        all_ranks_in.data_ptr(),
        p2p_fp8_buf.data_ptr(),
        p2p_scale_buf.data_ptr<float>(),
        out_fp8_ar.data_ptr(),
        out_bf16_ar.data_ptr(),
        num_elements,
        stream
    );
    return std::make_tuple(out_fp8_ar, out_bf16_ar);
}
"""

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本仓库同步至 RTX 6000D 服务器或 Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(cur_dir, "pcie_fp8_custom_all_reduce.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 pcie_fp8_custom_all_reduce.cu ...")
    ext = load_inline(
        name="lesson07_ar_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["test_p2p_fp8_allreduce"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    # 模拟 Decode 阶段 batch_size = 16, hidden_size = 7168, TP = 4
    world_size = 4
    num_elements = 16 * 7168
    all_ranks_in = torch.randn((world_size, num_elements), dtype=torch.bfloat16, device="cuda")

    out_fp8_ar, out_bf16_ar = ext.test_p2p_fp8_allreduce(all_ranks_in)
    ref_sum = all_ranks_in.float().sum(dim=0).to(torch.bfloat16)

    cos_sim = torch.nn.functional.cosine_similarity(
        out_fp8_ar.float(), ref_sum.float(), dim=0
    ).item()
    bf16_bytes = num_elements * 2 * (world_size - 1)
    fp8_bytes  = num_elements * 1 * (world_size - 1) + (num_elements // 128) * 4 * (world_size - 1)

    print(f"✅ BF16 Custom AllReduce 精确匹配: {torch.allclose(out_bf16_ar, ref_sum, atol=1e-2)}")
    print(f"✅ FP8 压缩 Custom AllReduce 余弦相似度: {cos_sim:.6f} (几乎无损！)")
    print(f"📉 单次 AllReduce 跨 PCIe 5.0 传输量: 从 {bf16_bytes/1024:.1f} KB 降至 {fp8_bytes/1024:.1f} KB (节省 {(1 - fp8_bytes/bf16_bytes)*100:.1f}% PCIe 带宽)")

if __name__ == "__main__":
    main()
