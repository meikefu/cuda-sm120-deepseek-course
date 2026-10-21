# 第 4 课详细讲义：Blackwell `sm_120` Tensor Core 实战 —— 用 CuTe `TiledMMA` 与双缓冲流水线编写矩阵乘

> **配套代码文件**：
> * CUDA / CuTe 核心实现：[sm120_cute_gemm.cu](./sm120_cute_gemm.cu)
> * PyTorch JIT 编译与算力评测脚本：[test_and_bench.py](./test_and_bench.py)
> * 对标 vLLM / CUTLASS 源码：
>   * vLLM: `csrc/quantization/cutlass_w8a8/`、`csrc/quantization/fp4/`
>   * CUTLASS: `examples/70_blackwell_gemm/`、`include/cutlass/gemm/collective/`

---

## 一、核心认知：为什么 RTX 6000D (`sm_120`) 的 Tensor Core 编程与 B200 (`sm_100`) 完全不同？

这是你在 RTX 6000D 上修改 vLLM / SGLang / DeepGEMM 算子时**最核心的硬件背景**：

| 硬件特性 | 数据中心 Hopper (`sm_90`, H100) | 数据中心 Blackwell (`sm_100`, B200) | 工作站/消费级 Blackwell (`sm_120`, RTX 6000D / 5090) |
| :--- | :--- | :--- | :--- |
| **Tensor Core 指令族** | `wgmma.mma_async` (4 个 Warp 协同异步执行) | `tcgen05.mma` (第 5 代指令，支持 2-CTA 协同) | **`mma.sync.aligned` (Warp 级寄存器同步指令，含新型块缩放指令)** |
| **累加器存储位置** | 寄存器 (Register File) | 专用 **TMEM (Tensor Memory)** (每 SM 256KB) | **寄存器 (Register File)**（无 TMEM！） |
| **每 SM 最大 Shared Memory** | 228 KB | 228 KB | **99 KB**（严格受限！） |
| **原生低比特支持** | FP8 (`e4m3`/`e5m2`) | FP8, NVFP4, MXFP4 (`UE8M0` 块缩放) | **FP8, NVFP4, MXFP4 (`UE8M0` 块缩放)** |

### 关键结论：
1. **不能照搬 `sm_100` 或 `sm_90` 算子**：任何写死了 `wgmma` 或 `tcgen05` 的算子（如原版 FlashAttention-3 或部分 DeepGEMM 分支）在 RTX 6000D (`sm_120`) 上会直接编译失败或报 `Illegal Instruction`。
2. **`sm_120` 的正确打开方式**：使用基于 **Warp 级寄存器 (`mma.sync`)** 的 CuTe `MMA_Atom`（如 `SM80_16x8x16` / `SM120_BlockScaled_MMA`），配合 **小于 99KB 的 Shared Memory 分块与多级异步流水线 (`cp.async` / TMA)**！

---

## 二、核心概念 1：`TiledMMA` —— 如何指挥 Tensor Core 干活？

在第 3 课我们学了指挥数据搬运的 `TiledCopy`；这一课我们学指挥 Tensor Core 做矩阵乘累加 $D = A \times B^T + C$ 的 **`TiledMMA`**。

看 [sm120_cute_gemm.cu](./sm120_cute_gemm.cu) 的第 168-174 行：
```cpp
using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
auto tiled_mma = make_tiled_mma(
    MmaAtom{},                                 // 参数 1：单条硬件 Tensor Core 指令 (MMA Atom)
    make_layout(make_shape(_2{}, _2{}, _1{})), // 参数 2：Warp 怎么排布？(2x2 = 4 个 Warp = 128 线程)
    Tile<_32, _32, _16>{}                      // 参数 3：单次 MMA 覆盖的置换瓦片大小
);
```

### 1. 拆解 `SM80_16x8x16_F32BF16BF16F32_TN` 的名字密码
CUTLASS 里所有 `MMA_Atom` 的名字都遵循严格格式：
* **`16x8x16`**：1 个 Warp（32 个线程）执行**单条**硬件 `mma.sync.aligned.m16n8k16` 指令时，能够计算一个 **$M=16, N=8, K=16$** 的小矩阵乘法！
* **`F32BF16BF16F32`**：依次代表 $D$（输出 `float32`）、$A$（输入 `bfloat16`）、$B$（输入 `bfloat16`）、$C$（累加器 `float32`）的数据类型。
* **`TN`（极为重要！）**：代表矩阵 $A$ 是 **T (Transposed / K-Major，即行优先 `[M, K]`，$K$ 维连续)**，矩阵 $B$ 是 **N (Normal / K-Major，即 `[N, K]`，$K$ 维连续)**！
  * 为什么大模型推理最爱 `TN` 布局？因为 PyTorch 的 `nn.Linear(in_features=K, out_features=N)` 默认权重 `weight` 的形状就是 **`[N, K]`**，且输入激活 `x` 的形状是 **`[M, K]`**，两者的最后一维都是 $K$ 连续！直接用 `TN` 指令，权重和激活都不需要提前转置！

### 2. `thr_mma.partition_A/B/C` 与 `make_fragment_A/B/C`
看 Kernel 的第 67-76 行：
```cpp
ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA); // 当前线程在 Shared Memory sA 中需要读取的位置视图
Tensor tCsB = thr_mma.partition_B(sB); // 当前线程在 Shared Memory sB 中需要读取的位置视图
Tensor tCgC = thr_mma.partition_C(gC); // 当前线程在 Global Memory gC 中负责写回的位置视图

// 在寄存器 (Register File) 中开辟真正的存储空间：
Tensor tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0));
Tensor tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0));
Tensor tCrC = thr_mma.make_fragment_C(tCgC);
clear(tCrC); // 累加器清零
```
随后在主循环里，只需要两行 `cute::copy` 把数据从 Shared Memory 装进寄存器 `tCrA`、`tCrB`，再调一行 `cute::gemm(tiled_mma, tCrA, tCrB, tCrC)`，CuTe 就会自动展开循环并发射满屏的 `mma.sync` 硬件指令！

---

## 三、核心概念 2：针对 `sm_120` (99KB SMEM) 的分块计算与双缓冲流水线

### 1. 如何为 RTX 6000D (`sm_120`) 精打细算 Shared Memory？
假设我们选择的 CTA 分块大小为 $b_M = 128, b_N = 128, b_K = 32$，开启 **2 级流水线（Double Buffering，`Stages = 2`）**：
* 矩阵 $A$ 的 Shared Memory 占用：$128 \times 32 \times 2\text{ Bytes (BF16)} \times 2\text{ Stages} = 16\text{ KB}$
* 矩阵 $B$ 的 Shared Memory 占用：$128 \times 32 \times 2\text{ Bytes (BF16)} \times 2\text{ Stages} = 16\text{ KB}$
* **单 CTA 总 Shared Memory** $= 16\text{ KB} + 16\text{ KB} = \mathbf{32\text{ KB}}$！
* 因为 RTX 6000D 每个 SM 有 **99 KB** Shared Memory：
  $$\lfloor 99\text{ KB} / 32\text{ KB} \rfloor = \mathbf{3\text{ 个 CTA / SM！}}$$
  这意味着每个 SM 可以同时驻留 2~3 个 Block（对应 `__launch_bounds__(128, 2)`），完美隐藏指令延迟！如果你盲目照搬 H100 的配置（比如 $128 \times 256 \times 64 \times 3\text{ Stages} = 144\text{ KB}$），在 RTX 6000D 上就会直接报 `cudaErrorInvalidConfiguration` 启动失败！

### 2. 双缓冲软件流水线（Software Pipelining）是怎么把访存耗时藏起来的？
如果不做流水线，GPU 会陷入“**搬一块数据（等几百周期） $\rightarrow$ Tensor Core 算一下 $\rightarrow$ 再搬下一块（又等几百周期）**”的串行等待，Tensor Core 一半时间在闲置。

看 [sm120_cute_gemm.cu](./sm120_cute_gemm.cu) 的第 81-118 行，我们把 Shared Memory 分成了第 `0` 槽位和第 `1` 槽位（即最后一个维度 `_2`）：
1. **Prologue（预热阶段，进循环前）**：
   先发射异步指令 `cute::copy`，把第 `0` 个 $K$ 分块预取到 Shared Memory 的 **`Stage 0`**。
2. **Mainloop（乒乓循环，处理第 `k_tile` 块）**：
   * `read_stage = k_tile % 2`（当前要算的槽位），`write_stage = (k_tile + 1) % 2`（下一个预取的槽位）。
   * **第 1 步（异步发射下一块）**：立刻向 **`write_stage`** 发射第 `k_tile + 1` 块的 `cp.async` 异步读取指令（此时硬件后台开始从 GDDR7 搬运，线程不阻塞！）。
   * **第 2 步（等待当前块就绪）**：调用 `cp_async_wait<1>()` —— 这里的 `<1>` 极其关键！它告诉硬件：“**只要在途未完成的异步批次不超过 1 个（也就是说刚发射的 `write_stage` 可以继续在后台飞，但更早发射的 `read_stage` 必须已经落地），我就继续往下走！**”
   * **第 3 步（Tensor Core 全速计算）**：当内存控制器在后台搬运 `write_stage` 的同时，Tensor Core 正在全速计算 `read_stage` 的矩阵乘 `cute::gemm`！

这就是所有现代高性能算子库（CUTLASS、FlashAttention、DeepGEMM）最核心的 **Compute-Memory Overlap（算存重叠）** 技巧。

---

## 四、核心概念 3：Epilogue（尾声融合）与 `make_identity_tensor` 坐标反查

在量化矩阵乘（如 W8A8 FP8 或 W4A4 NVFP4）中，Tensor Core 算完累加器 `tCrC` 后，它的元素还躺在 128 个线程各自的寄存器里。因为经过了复杂的 `TiledMMA` 交错排布，**第 `threadIdx.x` 号线程手里的第 `i` 个寄存器 `tCrC(i)`，到底对应输出矩阵 $C$ 的第几行 (`m_idx`)、第几列 (`n_idx`)？**

千万不要自己去翻 PTX 手册手算线程寄存器映射！CuTe 提供了一个万能神器 **`make_identity_tensor`**（见第 123-138 行）：
```cpp
// 1. 创建一个虚拟的“坐标张量” idC，它在 (m, n) 位置存的值就是坐标 (m, n) 本身（不占任何物理内存！）
Tensor idC = make_identity_tensor(make_shape(size<0>(cta_tiler), size<1>(cta_tiler)));

// 2. 用跟累加器完全相同的 thr_mma.partition_C 对这个坐标张量做切片！
Tensor tCcC = thr_mma.partition_C(idC);

// 3. 现在，tCrC(i) 的逻辑坐标就是 tCcC(i)！
CUTE_UNROLL
for (int i = 0; i < size(tCrC); ++i) {
    auto coord = tCcC(i);
    int m_idx = cta_m_base + get<0>(coord); // 直接拿到全局行号 m
    int n_idx = cta_n_base + get<1>(coord); // 直接拿到全局列号 n
    // 在寄存器里直接乘上 Per-Token 激活缩放 scale_a[m] 和 Per-Channel 权重缩放 scale_b[n]！
    float scaled_val = tCrC(i) * scale_a[m_idx] * scale_b[n_idx];
    tCgC(i) = __float2bfloat16(scaled_val);
}
```
无论是融合 `scale_a * scale_b`、加 `bias`、还是融合激活函数，全都可以在这 10 行 Epilogue 代码里在寄存器内零开销完成！
