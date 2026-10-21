# cuda-sm120-deepseek-course

> **Blackwell (`sm_120`) 架构大模型推理 CUDA & CUTLASS 算子优化实战课程（针对 DeepSeek-V4.1-Flash / vLLM / SGLang）**

本仓库是专为大模型推理底层算子研发定制的 **4 周（8 门核心课）** 渐进式实战课程，聚焦 **RTX 6000D (`sm_120`, 84GB GDDR7, 无 NVLink PCIe 5.0)** 上的 **DeepSeek-V4.1-Flash** 算子级优化与工程落地。

每门课目录下均包含：
1. **`TUTORIAL.md`**：深度图文讲义与底层硬件原理解析；
2. **`*.cu`**：完整、带逐行中文注释的 CUDA C++ / CuTe 核心算子源码；
3. **`test_and_bench.py`**：使用 PyTorch `load_inline` 即时编译（3 秒出结果）的精度校验与性能 Benchmark 脚本。

---

## 课程总览与快速导航

### 第一周：现代 CUDA 内存层次、Warp 原语与 Memory-Bound 算子融合
* **第 1 课**：线程模型、Warp Shuffle 与 vLLM 级融合算子（手写 `RMSNorm + Residual + Dynamic FP8 Quant`）
  * 📘 [详细讲义](./lesson01_rmsnorm_fp8/TUTORIAL.md) | 💻 [CUDA 源码](./lesson01_rmsnorm_fp8/fused_rmsnorm_fp8.cu) | 🧪 [评测脚本](./lesson01_rmsnorm_fp8/test_and_bench.py)
* **第 2 课**：128-bit 向量化访存 (`uint4`)、Sub-Warp 规约、Shared Memory Bank Conflict 与 MoE 块缩放激活融合（手写 `SwiGLU + 1x128 Group FP8 / UE8M0`）
  * 📘 [详细讲义](./lesson02_swiglu_block_fp8/TUTORIAL.md) | 💻 [CUDA 源码](./lesson02_swiglu_block_fp8/fused_swiglu_block_fp8.cu) | 🧪 [评测脚本](./lesson02_swiglu_block_fp8/test_and_bench.py)

### 第二周：CuTe 与 CUTLASS 3.x/4.x 核心（攻克 Blackwell `sm_120` Tensor Core）
* **第 3 课**：CuTe 的灵魂——`Layout` 代数、`Swizzle<3,3,3>` 零冲突重排与 `TiledCopy` (`cp.async`)
  * 📘 [详细讲义](./lesson03_cute_layout_copy/TUTORIAL.md) | 💻 [CuTe 源码](./lesson03_cute_layout_copy/cute_layout_and_copy.cu) | 🧪 [评测脚本](./lesson03_cute_layout_copy/test_and_bench.py)
* **第 4 课**：`sm_120` Tensor Core 实战——用 CuTe `TiledMMA` 与双缓冲软件流水线编写矩阵乘（适配 `sm_120` 99KB Shared Memory 限制 + Epilogue Scale 融合）
  * 📘 [详细讲义](./lesson04_sm120_cute_gemm/TUTORIAL.md) | 💻 [CuTe 源码](./lesson04_sm120_cute_gemm/sm120_cute_gemm.cu) | 🧪 [评测脚本](./lesson04_sm120_cute_gemm/test_and_bench.py)

### 第三周：DeepSeek 核心算子拆解（MoE Grouped GEMM 与 FlashInfer MLA）
* **第 5 课**：DeepSeek MoE 的心脏——`sm_120` 动态路由 **Grouped GEMM**、扁平化瓦片调度器（Flattened Tile Scheduler）与非对齐尾块保护
  * 📘 [详细讲义](./lesson05_moe_grouped_gemm/TUTORIAL.md) | 💻 [CUDA 源码](./lesson05_moe_grouped_gemm/moe_grouped_gemm_sm120.cu) | 🧪 [评测脚本](./lesson05_moe_grouped_gemm/test_and_bench.py)
* **第 6 课**：Attention 进阶——手撕 **FlashInfer Online Softmax** 与 DeepSeek **MLA (Multi-head Latent Attention)** 矩阵吸收 + Paged Decode 算子
  * 📘 [详细讲义](./lesson06_deepseek_mla_paged/TUTORIAL.md) | 💻 [CUDA 源码](./lesson06_deepseek_mla_paged/paged_mla_decode.cu) | 🧪 [评测脚本](./lesson06_deepseek_mla_paged/test_and_bench.py)

### 第四周：无 NVLink (PCIe 5.0) 多卡通信优化与 vLLM/SGLang 完整闭环
* **第 7 课**：手撕 vLLM `custom_all_reduce.cu`——PCIe BAR1 P2P 跨卡直写、IPC 信号量同步与 **FP8 寄存器级低比特压缩 AllReduce**
  * 📘 [详细讲义](./lesson07_pcie_custom_allreduce/TUTORIAL.md) | 💻 [CUDA 源码](./lesson07_pcie_custom_allreduce/pcie_fp8_custom_all_reduce.cu) | 🧪 [评测脚本](./lesson07_pcie_custom_allreduce/test_and_bench.py)
* **第 8 课**：双微批次计算通信重叠（Two-Microbatch Overlap, TBO）、跨 NUMA 拓扑双层规约与修改 vLLM/SGLang 工程闭环指南
  * 📘 [详细讲义](./lesson08_tbo_overlap_and_vllm_integration/TUTORIAL.md) | 💻 [CUDA 源码](./lesson08_tbo_overlap_and_vllm_integration/tbo_compute_comm_overlap.cu) | 🧪 [评测脚本](./lesson08_tbo_overlap_and_vllm_integration/test_and_bench.py)

---

## 环境要求与快速上手

### 推荐运行环境
- **GPU 架构**：Blackwell 架构（`sm_120`，如 RTX 6000D / RTX 5090）或其他支持 FP8 的 NVIDIA GPU（如 Hopper `sm_90` / Ada Lovelace `sm_89`）
- **CUDA Toolkit**：12.4+（推荐 CUDA 12.8 以获得完整 Blackwell 原生支持）
- **Python / PyTorch**：Python 3.10+, PyTorch >= 2.4.0（需支持 `torch.float8_e4m3fn`）
- **C++ 编译器**：支持 C++17 的 GCC / Clang
- **CUTLASS**：CUTLASS 3.x / 4.x（针对第 3、4 课 CuTe 算子）

### 运行方式（即时 JIT 评测）
每门课程均提供独立的单测与性能基准脚本，直接运行即可自动完成 JIT 编译与基线精度验证：

```bash
# 示例：运行第 1 课 RMSNorm + FP8 融合算子验证
python lesson01_rmsnorm_fp8/test_and_bench.py

# 示例：运行第 3 课 CuTe Layout & TiledCopy 验证
python lesson03_cute_layout_copy/test_and_bench.py

# 示例：运行第 5 课 MoE Grouped GEMM 验证
python lesson05_moe_grouped_gemm/test_and_bench.py
```
