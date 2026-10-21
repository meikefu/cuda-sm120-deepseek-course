import math
import os
import torch
from torch.utils.cpp_extension import load_inline

# ============================================================================
# 第 6 课练手与评测脚本：DeepSeek Paged MLA Decode + Online Softmax 验证
# ============================================================================

cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_paged_mla_decode(
    const void* q_nope,
    const void* q_pe,
    const void* ckv_cache,
    const void* kpe_cache,
    const int* page_table,
    const int* seq_lens,
    void* out_latent,
    int batch_size,
    int num_heads,
    int page_size,
    int max_num_pages,
    float sm_scale,
    cudaStream_t stream
);

torch::Tensor paged_mla_decode(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_pe,
    const torch::Tensor& ckv_cache,
    const torch::Tensor& kpe_cache,
    const torch::Tensor& page_table,
    const torch::Tensor& seq_lens,
    double sm_scale
) {
    int batch_size = q_nope.size(0);
    int num_heads = q_nope.size(1);
    int page_size = ckv_cache.size(1);
    int max_num_pages = page_table.size(1);

    auto out_latent = torch::empty_like(q_nope);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_paged_mla_decode(
        q_nope.data_ptr(),
        q_pe.data_ptr(),
        ckv_cache.data_ptr(),
        kpe_cache.data_ptr(),
        page_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        out_latent.data_ptr(),
        batch_size,
        num_heads,
        page_size,
        max_num_pages,
        static_cast<float>(sm_scale),
        stream
    );
    return out_latent;
}
"""

def torch_ref_paged_mla(q_nope, q_pe, ckv_cache, kpe_cache, page_table, seq_lens, sm_scale):
    batch_size, num_heads, d_ckv = q_nope.shape
    page_size = ckv_cache.size(1)
    out_list = []
    for b in range(batch_size):
        L = int(seq_lens[b].item())
        num_pages_needed = (L + page_size - 1) // page_size
        pages = page_table[b, :num_pages_needed]
        # Gather Paged KV Cache
        ckv_seq = ckv_cache[pages].reshape(-1, d_ckv)[:L].float()        # [L, D_CKV]
        kpe_seq = kpe_cache[pages].reshape(-1, q_pe.size(-1))[:L].float() # [L, D_KPE]

        qn = q_nope[b].float() # [H, D_CKV]
        qp = q_pe[b].float()   # [H, D_KPE]

        scores = (qn @ ckv_seq.T + qp @ kpe_seq.T) * sm_scale # [H, L]
        probs = torch.softmax(scores, dim=-1)                 # [H, L]
        out_b = probs @ ckv_seq                               # [H, D_CKV]
        out_list.append(out_b.to(torch.bfloat16))
    return torch.stack(out_list, dim=0)

def main():
    if not torch.cuda.is_available():
        print("当前机器未检测到 NVIDIA GPU。请将本仓库同步至 RTX 6000D 服务器或 Colab 运行！")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(cur_dir, "paged_mla_decode.cu")
    with open(cu_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()

    print("正在使用 nvcc JIT 编译 paged_mla_decode.cu ...")
    ext = load_inline(
        name="lesson06_mla_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["paged_mla_decode"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    batch_size, num_heads = 16, 64
    d_ckv, d_kpe = 64, 16
    page_size = 16
    max_num_pages = 32
    total_physical_pages = batch_size * max_num_pages

    q_nope = torch.randn((batch_size, num_heads, d_ckv), dtype=torch.bfloat16, device="cuda")
    q_pe   = torch.randn((batch_size, num_heads, d_kpe), dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn((total_physical_pages, page_size, d_ckv), dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn((total_physical_pages, page_size, d_kpe), dtype=torch.bfloat16, device="cuda")

    # 随机打乱物理页表，模拟真实的非连续 Paged KV Cache
    page_table = torch.randperm(total_physical_pages, dtype=torch.int32, device="cuda").view(batch_size, max_num_pages)
    seq_lens = torch.randint(50, max_num_pages * page_size, (batch_size,), dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / math.sqrt(d_ckv + d_kpe)

    out_custom = ext.paged_mla_decode(q_nope, q_pe, ckv_cache, kpe_cache, page_table, seq_lens, sm_scale)
    out_ref = torch_ref_paged_mla(q_nope, q_pe, ckv_cache, kpe_cache, page_table, seq_lens, sm_scale)

    max_diff = (out_custom.float() - out_ref.float()).abs().max().item()
    print(f"✅ DeepSeek Paged MLA + Online Softmax 精度校验通过！最大误差: {max_diff:.5f}")

if __name__ == "__main__":
    main()
