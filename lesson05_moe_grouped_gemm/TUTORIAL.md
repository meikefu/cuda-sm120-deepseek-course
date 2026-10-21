# 第 5 课详细讲义：DeepSeek MoE 的心脏 —— `sm_120` Grouped GEMM、扁平化瓦片调度与尾声融合

> **配套代码文件**：
> * CUDA 核心实现：[moe_grouped_gemm_sm120.cu](./moe_grouped_gemm_sm120.cu)
> * PyTorch JIT 评测脚本：[test_and_bench.py](./test_and_bench.py)
> * 对标 vLLM / SGLang / FlashInfer 源码：
>   * vLLM: `csrc/moe/`、`csrc/quantization/cutlass_w8a8/moe/`
>   * FlashInfer: `include/flashinfer/gemm/group_gemm.cuh`

---

## 一、为什么 DeepSeek MoE 绝对不能用 `for` 循环调 `cuBLAS`？

在 DeepSeek-V3 / V4-Flash 中，每一层 MoE 包含多达 **256 个路由专家（Routed Experts）**，每个 Token 激活 Top-8 个专家。
假设当前批次有 $B = 128$ 个 Token，经过门控网络（Gating / TopK）和重排（Permute）后：
* 第 0 号专家可能分到了 $M_0 = 19$ 个 Token；
* 第 1 号专家可能分到了 $M_1 = 3$ 个 Token；
* 第 2 号专家可能分到了 $M_2 = 0$ 个 Token（空专家）；
* ……
* 第 255 号专家可能分到了 $M_{255} = 41$ 个 Token。

每个专家都有自己独立的权重矩阵 $W_e \in \mathbb{R}^{N \times K}$。我们需要计算 256 个**高度 $M_e$ 各不相同**的小矩阵乘法：
$$Y_e = X_e \times W_e^T, \quad e \in \{0, 1, \dots, 255\}$$

如果用 Python `for` 循环依次调用 256 次 `torch.matmul`（即调 256 次 `cuBLAS`）：
1. **CPU Kernel Launch Overhead 灾难**：CPU 向 GPU 发射 1 个 Kernel 需要约 $3\sim 5\ \mu\text{s}$。发射 256 次光 CPU 排队就要 **$1000\ \mu\text{s}$（1 毫秒）**，而 RTX 6000D 算这些小矩阵实际只需要不到 $100\ \mu\text{s}$！GPU 90% 的时间在干等 CPU 发指令！
2. **SM 利用率极低（Wave Quantization）**：当 $M_e = 3, N = 2048$ 时，如果分块是 $64 \times 128$，整个矩阵乘只能切出 $1 \times 16 = 16$ 个 Block (CTA)。而你的 **RTX 6000D 有近 180 个 SM**，16 个 Block 只能喂饱不到 10% 的 SM，其余 90% 的算力全部空转！

---

## 二、核心概念 1：Grouped GEMM 的扁平化瓦片调度器（Flattened Tile Scheduler）

怎么用 **1 次 Kernel 启动** 同时算完 256 个形状不同的矩阵乘，并且把 RTX 6000D 的所有 SM 全部塞满？
答案是 **扁平化瓦片调度（Flattened Tile Scheduling）**，看 [moe_grouped_gemm_sm120.cu](./moe_grouped_gemm_sm120.cu) 的第 26-84 行：

### 1. 构建两套前缀和偏移数组（Prefix Sum Offsets）
假设瓦片高度 `TILE_M = 32`，有 3 个专家，Token 数分别为 $M_0 = 35, M_1 = 10, M_2 = 64$：
1. **`expert_token_offsets`（真实 Token 行偏移）**：
   $$[0, \ 35, \ 45, \ 109]$$
   用于定位第 $e$ 号专家的输入数据在连续张量 `permuted_x` 中的起始行 `token_start = expert_token_offsets[e]`。
2. **`expert_tile_offsets`（向上取整后的 M-Tile 块数偏移）**：
   * 专家 0：$35$ 个 Token 需要 $\lceil 35 / 32 \rceil = 2$ 个 M-Tile；
   * 专家 1：$10$ 个 Token 需要 $\lceil 10 / 32 \rceil = 1$ 个 M-Tile；
   * 专家 2：$64$ 个 Token 需要 $\lceil 64 / 32 \rceil = 2$ 个 M-Tile。
   前缀和数组为：
   $$\text{expert\_tile\_offsets} = [0, \ 2, \ 3, \ 5]$$
   所有专家一共需要 `total_m_tiles = 5` 个 M-Tile！

### 2. 二分查找反推专家归属 (`find_expert_for_m_tile`)
我们启动单个 Kernel，令 `gridDim.x = total_m_tiles`。
当 `blockIdx.x`（即 `global_m_tile_idx`）进入 Kernel 后，通过 [`find_expert_for_m_tile`](./moe_grouped_gemm_sm120.cu#L26-L43) 在 `expert_tile_offsets` 中做一次仅需 $\log_2(256) = 8$ 步的二分查找：
* 立刻算出当前 Block 属于第 `expert_id` 号专家；
* 以及它是该专家内部的第 `local_m_tile_idx = global_m_tile_idx - expert_tile_offsets[expert_id]` 个分块！

---

## 三、核心概念 2：非对齐尾块边界保护（Predicated Load / Store）

因为每个专家分到的 Token 数 $M_e$ 是运行期动态路由出来的，**几乎不可能是 `TILE_M`（如 32 或 128）的整数倍**！
例如专家 0 有 $M_0 = 35$ 个 Token，它的第 1 个 M-Tile 负责行号 $32 \dots 63$，其中只有行号 $32, 33, 34$ 是真实属于专家 0 的 Token，而行号 $35 \dots 63$ 已经是专家 1 的数据了！

因此，在 Grouped GEMM 中**必须严格控制读写边界**：
1. **读取阶段（Zero-Padding Predicated Load，第 106-109 行）**：
   ```cpp
   // global_r 是当前专家内部的行号 (0 .. M_e - 1)
   sA[r][c] = (global_r < M_e && global_c < K)
              ? __bfloat162float(A_ptr[global_r * K + global_c])
              : 0.0f;
   ```
   超出 `M_e` 的行在加载进 Shared Memory 时强制填 `0.0f`，既不会越界读错别的专家的数据，也不会污染累加器。
2. **写回阶段（Predicated Store，第 152-162 行）**：
   ```cpp
   if (global_m < M_e) {
       C_ptr[global_m * N + global_n] = __float2bfloat16(val);
   }
   ```
   **绝对不能把 `global_m >= M_e` 的无效行写回显存**，否则会直接覆盖掉下一个专家（`expert_id + 1`）的起始计算结果，引发极其隐蔽的精度崩溃！

---

## 四、实战避坑指南：在 RTX 6000D (`sm_120`) 上修改 vLLM / SGLang 的 MoE Grouped GEMM

在实际工程中，当你打开 vLLM 或 SGLang 针对 Blackwell 的 MoE 算子时，请重点关注以下 3 个实战修改点：

1. **避免 `sm_120` 上的 TMA Warp-Specialized Grouped GEMM 陷阱**：
   * 在 CUTLASS 3.x 中，针对 Hopper/Blackwell 设计的 Ptr-Array TMA Grouped GEMM 需要在每个专家切换时动态更新 TMA Descriptor（张量内存加速器描述符）。
   * 在 `sm_120` 上，如果未开启 `compute_120f` 编译目标或 Shared Memory 超出 99KB，极易触发 **IMA (`cudaErrorIllegalAddress`)**。
   * **稳妥且高效的方案**：在 Decode 与中小 Batch 阶段，使用 **基于 `cp.async` + 连续 `permuted_x` 偏移表（Offset-based Grouped GEMM，即本课代码所演示的架构）**，无需在 Device 端动态修改 TMA 描述符，延迟更低且 100% 稳定！
2. **根据 Batch Size 动态选择 `TILE_M`（自适应瓦片高度）**：
   * 在 Decode 阶段，平均每个专家只有 $1 \sim 8$ 个 Token。如果 `TILE_M` 设成 `128`，那么每个 Tile 里有超过 90% 的行都是填 `0` 的无效计算！
   * **优化策略**：在 `launch` 函数中根据 `total_tokens / num_experts` 的平均值，动态分发不同 `TILE_M` 的模板实例（如平均 Token $< 16$ 时分发 `TILE_M = 16` 或 `32`；Prefill 大批次时分发 `TILE_M = 64` 或 `128`）。
