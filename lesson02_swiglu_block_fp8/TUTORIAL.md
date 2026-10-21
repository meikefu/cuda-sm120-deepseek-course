# 第 2 课详细讲义：128-bit 向量化访存、Sub-Warp 规约、Bank Conflict 与 Blackwell `UE8M0` 块量化

> **配套代码文件**：
> * CUDA 核心实现：[fused_swiglu_block_fp8.cu](./fused_swiglu_block_fp8.cu)
> * PyTorch JIT 评测脚本：[test_and_bench.py](./test_and_bench.py)
> * 对标 vLLM / DeepGEMM 源码：`csrc/activation_kernels.cu`

---

## 一、核心知识点 1：128-bit 向量化访存 (`uint4` / `LDG.E.128`)

* GPU 显存控制器单线程单条指令最高支持 **16 字节（128-bit）** 读写（对应 SASS 指令 `LDG.E.128` / `STG.E.128`）。
* 1 个 `__nv_bfloat16` 占 2 字节，因此使用 16 字节对齐的 `union PackedBF16x8`（内含 `uint4 raw`）可一次性读入 **8 个连续的 `bfloat16`**，将访存指令发射次数降低为原来的 1/8，跑满 RTX 6000D 的 GDDR7 带宽。
* 同理，8 个 `__nv_fp8_e4m3`（共 8 字节）使用 `uint2`（64-bit `STG.E.64`）打包一次性写出。

---

## 二、核心知识点 2：Sub-Warp（半线程束）规约 —— 零 Shared Memory 搞定 1x128 块量化

* DeepSeek 的细粒度 FP8 量化要求每 **128 个连续元素** 独立计算一个缩放因子 `scale`。
* 由于每个线程一次向量化读取 8 个元素，**16 个连续线程恰好负责 $16 \times 8 = 128$ 个元素**！
* 16 个连续线程正好构成半个 Warp（Half-Warp：线程 `0..15` 为第 0 组，线程 `16..31` 为第 1 组）。
* 因此无需 Shared Memory 和 `__syncthreads()`，只需执行 `offset = 8, 4, 2, 1` 的 4 步 `__shfl_xor_sync` 即可在寄存器内直接完成 128 元素块的最大值规约！

---

## 三、核心知识点 3：Shared Memory Bank Conflict（存储体冲突）

* Shared Memory 在硬件上分为 **32 个 Bank**，每 4 字节轮流映射到一个 Bank（每 128 字节循环一圈）。
* 当同一个 Warp 内的多个线程在同一周期访问**同一个 Bank 的不同地址**时，硬件会将访存串行化，最高导致 32 倍的性能退化（32-way Bank Conflict）。
* 若多个线程访问同一个 Bank 的**完全相同地址**，则会触发硬件广播（Broadcast），不会产生冲突。

---

## 四、核心知识点 4：Blackwell (`sm_120`) 原生 `UE8M0` 幂次缩放因子

* Blackwell 第 5 代 Tensor Core 块缩放指令支持 **`UE8M0`（无符号 8 位纯指数，尾数为 0）** 格式，即缩放因子取 $2^E$。
* 利用 `__float_as_uint` 检查 IEEE-754 FP32 的低 23 位尾数（`0x007fffff`）：若尾数非零，则将指数位加 1（`+ 0x00800000`）并清空尾数，即可用零分支整数位运算完成向上取整到 2 的整数次幂。
