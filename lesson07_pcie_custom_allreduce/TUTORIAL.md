# 第 7 课详细讲义：无 NVLink 多卡通信破局（上）—— 手撕 vLLM `custom_all_reduce.cu` 与 PCIe 5.0 FP8 压缩规约

> **配套代码文件**：
> * CUDA 核心实现：[pcie_fp8_custom_all_reduce.cu](./pcie_fp8_custom_all_reduce.cu)
> * PyTorch JIT 评测脚本：[test_and_bench.py](./test_and_bench.py)
> * 对标 vLLM / SGLang 真实源码位置：
>   * vLLM: `csrc/custom_all_reduce.cu`、`csrc/custom_all_reduce.cuh`
>   * SGLang: `sgl-kernel/csrc/allreduce/`

---

## 一、为什么无 NVLink 的 RTX 6000D 多卡推理会被 NCCL 卡脖子？

RTX 6000D 没有 NVLink 桥接器，多卡之间完全通过主板或 PCIe Switch 上的 **PCIe 5.0 x16 总线** 通信（单向理论带宽 64 GB/s，实测有效约 48~52 GB/s）。

在跑 DeepSeek-V4.1-Flash 的 Decode 阶段时，如果开启张量并行（比如 `TP = 4` 或 `TP = 8`），每经过一层 Attention 和一层 MoE 都要做一次 `AllReduce`：
* **每次传输的数据包极小**：比如 `batch_size = 8, hidden_size = 7168`，BF16 张量大小只有 $8 \times 7168 \times 2\text{B} = \mathbf{112\text{ KB}}$！
* **标准 NCCL Ring AllReduce 的痛点**：
  NCCL 的 Ring 协议是为了几十 MB 的训练梯度设计的。在 4 卡 Ring 中，一次 AllReduce 需要分成 $2 \times (4 - 1) = 6$ 跳（3 跳 Reduce-Scatter + 3 跳 All-Gather）。每次跨 PCIe 跳跃都要经历硬件协议握手和同步延迟（约 $5 \sim 10\ \mu\text{s}$），6 跳下来光延迟就要 $40\ \mu\text{s}$ 以上，而 112 KB 数据真正传输其实只需要 $2\ \mu\text{s}$！

---

## 二、核心概念 1：vLLM `custom_all_reduce.cu` 的底层原理（CUDA IPC + PCIe BAR1 P2P）

为了消灭 NCCL 的多跳延迟，vLLM 和 SGLang 编写了专门针对小包（通常 $< 2\text{MB} \sim 8\text{MB}$）的 `custom_all_reduce.cu`，核心靠三步：

### 1. 用 `cudaIpcGetMemHandle` 把其他 GPU 的显存直接映射到本卡地址空间
在初始化阶段：
1. 每张卡（Rank $r$）用 `cudaMalloc` 分配一块固定的通信缓冲区 `comm_buf[r]` 和一块信号量数组 `barrier_flags[r]`。
2. 调用 `cudaIpcGetMemHandle(&handle, comm_buf[r])`，并通过 CPU 进程间通信把 `handle` 广播给所有其他卡。
3. 其他卡调用 `cudaIpcOpenMemHandle(&peer_ptr[r], handle, cudaIpcMemLazyEnablePeerAccess)`。
4. **神奇效果**：现在，Rank 0 的 CUDA Kernel 里可以直接解引用指针 `*peer_ptr[1]`，硬件会自动将其转换为一条跨 **PCIe BAR1 P2P 总线** 的读写事务，直接访问 Rank 1 的 GDDR7 显存，完全绕过 CPU 内存！

### 2. 无锁原子信号量屏障（Lock-Free Semaphore Barrier）
因为 4 张卡上的 Kernel 是异步启动的，Rank 0 怎么知道 Rank 1, 2, 3 已经把这一层的激活值写进各自的 P2P 缓冲区了？
不能回退到 CPU 同步！vLLM 在 GPU 显存里维护了一个递增的代数计数器 `flag_value += 1`：
* 进入 Kernel 时，每个 Block 的前 4 个线程（`threadIdx.x < 4`）通过 PCIe P2P 向全部 4 张卡的信号量地址写入当前的 `flag_value`，并调用 **`__threadfence_system()`**（注意：必须带 `_system` 后缀，强制刷新 PCIe 写缓冲，确保跨卡可见！）；
* 然后在寄存器里 `while (volatile_load(my_flags[peer]) < flag_value) {}` 自旋等待，耗时不到 $1\ \mu\text{s}$！

---

## 三、核心概念 2：One-Shot vs Two-Shot 与 PCIe P2P 读写黄金法则

在修改无 NVLink 的 `custom_all_reduce.cu` 时，有一条硬件铁律必须牢记：
> **在 PCIe 总线上，Posted Write（主动往远端卡写数据）比 Non-Posted Read（从远端卡拉数据）效率高得多！**
> 因为 PCIe Read 需要发一个请求包过去、等待远端显卡显存控制器响应、再把数据包发回来（经历完整的往返延迟 RTT）；而 PCIe Write 是“发后即忘（Posted）”流水线式的！

因此，在 RTX 6000D 的 PCIe 拓扑上，优化 `custom_all_reduce.cu` 有两大杀手锏：

### 杀手锏 1：寄存器级 FP8 实时压缩 AllReduce（本课代码实现）
看 [pcie_fp8_custom_all_reduce.cu](./pcie_fp8_custom_all_reduce.cu) 的第 49-125 行：
* 原本传输 8 个 `BF16` 需要跨 PCIe 搬运 **16 字节 (`uint4`)**；
* 我们在把数据写入 P2P 共享区前，直接在寄存器内算出 8 个（或 16/128 个）元素的局部 `absmax`，将其量化为 **8 个 `__nv_fp8_e4m3`（仅 8 字节，`uint2`）** + 极小的缩放因子 `scale`！
* **收益**：跨 PCIe 5.0 传输的数据量直接砍掉近 **50%**，而实测余弦相似度高达 `0.9995+`，对大模型生成质量几乎零影响！

### 杀手锏 2：Two-Shot（Reduce-Scatter + All-Gather）替代 One-Shot
* 当消息大于 `256 KB` 时，如果用 One-Shot（每张卡都去读另外 3 张卡的完整数据），每块数据会在 PCIe 总线上被重复拉取 3 次，总通信量是 $3 \times N$。
* 改用 **Two-Shot 模式**：
  * **第 1 拍**：4 张卡把数据切成 4 份，第 $r$ 号卡只负责拉取并累加第 $r$ 份（通信量 $\frac{3}{4} N$）；
  * **第 2 拍**：第 $r$ 号卡把算好的第 $r$ 份最终结果通过 **PCIe Posted Write** 直接推（Push）给另外 3 张卡（通信量 $\frac{3}{4} N$）！
  * 总通信量从 $3N$ 降到 $1.5N$，再叠加上面的 **FP8 压缩**，直接降到 **$0.75N$（比原版快 4 倍）**！
