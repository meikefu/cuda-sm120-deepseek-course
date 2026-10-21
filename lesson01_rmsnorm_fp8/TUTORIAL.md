# 第 1 课详细讲义：线程模型、Warp Shuffle 与 vLLM 级 Memory-Bound 融合算子

> **配套代码文件**：
> * CUDA 核心实现：[fused_rmsnorm_fp8.cu](./fused_rmsnorm_fp8.cu)
> * PyTorch JIT 评测脚本：[test_and_bench.py](./test_and_bench.py)
> * 对标 vLLM 源码：`csrc/layernorm_quant_kernels.cu`

---

## 一、为什么要写这个融合算子？

在 DeepSeek-V4.1-Flash 的每一层进入 FP8 矩阵乘（Linear 或 MoE）之前，都会依次执行 3 步操作：
1. **残差相加**：$\text{residual} \leftarrow \text{residual} + \text{input}$
2. **RMSNorm 归一化**：$\text{normed} = \frac{\text{residual}}{\sqrt{\frac{1}{D}\sum_{i=1}^D \text{residual}_i^2 + \epsilon}} \odot \text{weight}$
3. **动态 Per-Token FP8 量化**：先求整行最大绝对值 $m = \max_i |\text{normed}_i|$，算缩放因子 $\text{scale} = \frac{m}{448.0}$，再输出 $\text{out\_fp8} = \text{FP8}(\text{normed} / \text{scale})$。

未融合时，这 3 步需要启动 3~4 个独立 CUDA Kernel，对 GDDR7 全局显存进行 3 次往返读写。通过算子融合（Kernel Fusion），我们只从全局显存读 1 次数据，在片上 Shared Memory 和寄存器内完成全部规约与量化，显存带宽开销直接减少约 60%。

---

## 二、核心知识点拆解

### 1. Grid / Block (CTA) / Thread 映射
* 输入张量形状为 `[num_tokens, hidden_size]`。
* 每个 **Block (CTA)** 独立负责 **1 个 Token（1 整行）**：`token_idx = blockIdx.x`。
* Block 内的 `BLOCK_SIZE = 256` 个线程通过跨步循环（Grid-Stride / Block-Stride Loop）协作处理该行的 `hidden_size`（如 7168）个元素。

### 2. Warp Shuffle (`__shfl_xor_sync`) 寄存器级规约
* GPU 硬件调度的最小单位是 **Warp（32 个连续线程）**。
* `__shfl_xor_sync(0xffffffff, val, offset)` 允许同一个 Warp 内的 32 个线程直接交换寄存器变量，无需经过 Shared Memory。
* 通过 `offset = 16, 8, 4, 2, 1` 共 5 次蝶形异或交换，即可在 5 条指令内求出 32 个线程的累加和或最大值。
* 跨 Warp 规约时（例如 256 线程 = 8 个 Warp），先在每个 Warp 内规约，将 8 个局部结果写入 `smem_reduce[8]`，再由第 0 号 Warp 规约这 8 个值。

### 3. 动态 Shared Memory 缓存消除重复访存
* 声明 `extern __shared__ float smem_row[];` 缓存残差相加后的 FP32 中间结果（`7168 * 4B = 28KB`，远小于 RTX 6000D `sm_120` 每 SM 99KB 的上限），使得后续乘以 `weight`、求 `absmax` 以及除以 `scale` 全部在高速片上内存完成。
