# 第 6 课详细讲义：Attention 进阶 —— 手撕 FlashInfer Online Softmax 与 DeepSeek Paged MLA 算子

> **配套代码文件**：
> * CUDA 核心实现：[paged_mla_decode.cu](./paged_mla_decode.cu)
> * PyTorch JIT 评测脚本：[test_and_bench.py](./test_and_bench.py)
> * 对标 FlashInfer / vLLM 源码：
>   * FlashInfer: `include/flashinfer/attention/mla.cuh`
>   * vLLM: `csrc/attention/mla/`

---

## 一、核心概念 1：DeepSeek 的 MLA（Multi-head Latent Attention）为什么能省 90% 的 KV Cache 带宽？

在传统 MHA / GQA 中，Decode 阶段每生成一个新 Token，都要把历史所有 Token 的 Key Cache 和 Value Cache 从 GDDR7 显存读进 SM。
DeepSeek 的 **MLA（多头潜变量注意力）** 做了两个革命性设计：

1. **低秩潜向量压缩（Low-Rank KV Compression）**：
   不再存储每个 Head 独立的 Key 和 Value，而是存储所有 Head 共享的 **1 个压缩潜向量 $c_{KV} \in \mathbb{R}^{512}$** 和 **1 个带 RoPE 位置编码的向量 $k_{pe} \in \mathbb{R}^{64}$**。每个 Token 的 KV Cache 只有 $512 + 64 = 576$ 个数！
2. **矩阵吸收（Matrix Absorption）**：
   * 在推理 Decode 阶段，将解压 Key 的投影矩阵 $W_{UK}$ 提前乘进 Query（得到 $q_{\text{nope}} = q \cdot W_{UK} \in \mathbb{R}^{H \times 512}$）；
   * 将解压 Value 的投影矩阵 $W_{UV}$ 延迟到 Attention 算完之后再乘到输出上！
   * **惊人结论**：经过矩阵吸收后，Decode 阶段的 MLA 等价于 **128 个 Query Head 共享同 1 个 KV Head（其中 Key 是 $[c_{KV}, k_{pe}]$，Value 就是 $c_{KV}$ 本身！）** 的超级 MQA（Multi-Query Attention）！

---

## 二、核心概念 2：FlashAttention 的数学心脏 —— Online Softmax（单趟在线归约）

为什么传统 Attention 慢？因为传统 Softmax 需要走 **两趟（2-Pass）** 显存循环：
* 第 1 趟：算出所有历史位置 $t = 1 \dots L$ 的打分 $s_t = q \cdot k_t$，求全局最大值 $m = \max_t s_t$ 和分母 $l = \sum_t e^{s_t - m}$；
* 第 2 趟：算概率 $p_t = e^{s_t - m} / l$，再乘上 $v_t$ 累加得到输出 $O$。

**Online Softmax（在线归约）** 让我们可以 **只遍历 1 趟 Paged KV Cache**，边读 $k_t, v_t$ 边动态修正最大值和输出累加器！

看 [paged_mla_decode.cu](./paged_mla_decode.cu) 的第 74-121 行，每个线程只需维护 3 个状态变量：
1. `m_i`：到目前为止见过的最大打分（初始为 `-FLT_MAX`）；
2. `l_i`：以当前 `m_i` 为基准的指数分母和 $\sum e^{s - m_i}$（初始为 `0.0f`）；
3. `acc_o`：以当前 `m_i` 为基准的未归一化输出分子 $\sum e^{s - m_i} \cdot v$（初始为 `0.0f`）。

当读入一个新的 KV Token 并算出打分 `score` 时，执行以下 **4 行黄金更新公式**：
```cpp
float m_new = fmaxf(m_i, score);          // 1. 更新当前历史最大值
float alpha = __expf(m_i - m_new);        // 2. 计算旧最大值的指数修正系数 e^(m_old - m_new)
float p     = __expf(score - m_new);      // 3. 计算新 Token 的指数权重 e^(score - m_new)

l_i      = l_i * alpha + p;               // 4. 修正旧分母并加上新权重
acc_o[d] = acc_o[d] * alpha + p * v[d];   // 5. 修正旧分子并加上新 Value (在 MLA 中即 c_KV)
m_i      = m_new;
```
当循环结束（读完所有 $L$ 个历史 Token）后，只需要在寄存器里做一次除法 `out = acc_o / l_i`，就能得到与标准两趟 Softmax **100% 数学等价** 的精确输出！

---

## 三、核心概念 3：Paged KV Cache 非连续页表寻址与 RTX 6000D (`sm_120`) 优化策略

### 1. Paged KV Cache 页表寻址（第 83-90 行）
在 vLLM 和 SGLang 中，每个请求的 KV Cache 被切成固定大小（`page_size = 16` 或 `64`）的逻辑页，散落在显存的不同物理页中：
```cpp
int logical_page_idx = token_pos / page_size;              // 逻辑第几页
int offset_in_page   = token_pos % page_size;              // 页内第几个槽位
int physical_page_id = req_page_table[logical_page_idx];   // 查表得到物理页号
```

### 2. 在 RTX 6000D (`sm_120`, 99KB SMEM) 上如何把 MLA Decode 写到极致？
理解了本课的单 Warp 基准实现后，你在 FlashInfer / vLLM 中为 `sm_120` 定制生产级 MLA 算子时，有两个关键升级点：
1. **跨 Head 共享 KV Cache（CTA 级 MQA Group 聚合）**：
   因为同一个请求的所有 128 个 Query Head 共享完全相同的 $c_{KV}$ 和 $k_{pe}$，如果每个 Head 各自从 GDDR7 读一遍 KV Cache，带宽浪费了 128 倍！
   **优化**：让 1 个 CTA（Block）同时加载该请求的 $16 \sim 64$ 个 Query Head 进寄存器，把 Paged $c_{KV}$ 分块异步拷贝 (`cp.async`) 到 Shared Memory（例如缓存 32 个 Token 的 $c_{KV}$：$32 \times 576 \times 2\text{B} = 36\text{KB}$，完全装得进 `sm_120` 的 99KB SMEM！），然后直接调第 4 课学的 **Tensor Core `TiledMMA`** 计算 `Q_heads [64, 576] x K_chunk^T [576, 32]`！这就把原本 Memory-bound 的向量点积变成了 Compute-bound 的矩阵乘！
2. **长上下文 Split-K（FlashDecoding）**：
   当 `batch_size` 很小（如 1~4）但上下文 `seq_len` 很长（如 32K~128K）时，只有几个 Block 无法占满 RTX 6000D 的 ~180 个 SM。此时需要将 `seq_len` 切成多个 Partition（Split-K），每个 Block 计算一段子序列的局部 `(m_part, l_part, acc_o_part)`，最后用一个极轻量的 Combine Kernel 用同样的 Online Softmax 公式合并各段结果。
