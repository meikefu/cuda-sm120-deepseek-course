# 第 3 课详细讲义：CuTe 的灵魂 —— `Layout` 代数、`Swizzle` 零冲突重排与 `TiledCopy` 异步搬运

> **配套代码文件**：
> * CUDA / CuTe 核心实现：[cute_layout_and_copy.cu](./cute_layout_and_copy.cu)
> * PyTorch JIT 编译与验证脚本：[test_and_bench.py](./test_and_bench.py)
> * 对应 vLLM / FlashInfer 真实源码位置：
>   * vLLM: `csrc/quantization/cutlass_w8a8/`、`csrc/quantization/fp4/`
>   * FlashInfer: `include/flashinfer/gemm/`、`include/flashinfer/attention/`

---

## 一、为什么改 vLLM / SGLang 算子必须先懂 CuTe？

当你打开 vLLM 的 Blackwell (`sm_120`) 矩阵乘源码或者 FlashInfer 的 MLA 算子源码时，你会发现里面几乎没有传统 CUDA 教程里的 `row * K + col` 下标计算，取而代之的是满屏的 CuTe 代码：

```cpp
using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{},
                                Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
using TiledCopyG2S = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, bfloat16_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{}));

Tensor gA = local_tile(mA, cta_tiler, cta_coord);
ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
Tensor tAgA = thr_copy.partition_S(gA);
Tensor tAsA = thr_copy.partition_D(sA);
cute::copy(tiled_copy, tAgA, tAsA);
```

很多同学看到这里直接劝退。其实，**CuTe (CUDA Tensors)** 是 CUTLASS 3.x / 4.x 引入的革命性核心库，它只做了一件极其优雅的事：
> **把繁琐易错的“多维数组下标计算”和“线程分工映射”，统一抽象成了数学上的函数映射 `Layout = (Shape, Stride)`。**

只要花 30 分钟读懂本讲义的 **4 个核心概念**，上面那段“天书”在你眼里就会像 Python 代码一样直白！

---

## 二、核心概念 1：`Layout = (Shape, Stride)` 与编译期常量 `_N`

### 1. 什么是 `Layout`？
在 CuTe 中，任何一个 `Layout` 都由两个元组构成：**形状 `Shape`** 和 **步长 `Stride`**。
它本质上是一个从 **逻辑坐标 $(m, k)$** 到 **一维物理内存偏移 $\text{offset}$** 的点积函数：
$$\text{offset}(m, k) = m \times \text{stride}_0 + k \times \text{stride}_1$$

以一个 $8 \times 64$ 的二维矩阵为例：
* **行优先（Row-Major / K-Major）**：往下走 1 行需要跨过 64 个元素，往右走 1 列跨过 1 个元素：
  ```cpp
  // Shape = (8, 64), Stride = (64, 1)
  // 坐标 (r, c) 的物理偏移 = r * 64 + c * 1
  auto row_major = make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{}));
  ```
* **列优先（Column-Major / M-Major）**：往下走 1 行跨过 1 个元素，往右走 1 列跨过 8 个元素：
  ```cpp
  // Shape = (8, 64), Stride = (1, 8)
  // 坐标 (r, c) 的物理偏移 = r * 1 + c * 8
  auto col_major = make_layout(make_shape(_8{}, _64{}), make_stride(_1{}, _8{}));
  ```

### 2. 为什么代码里到处都是带下划线的 `_8{}`、`_64{}`，而不是直接写 `8`、`64`？
在 CuTe 中：
* 写 `64` 代表 **运行期变量（Dynamic Int）**，必须占用 GPU 的宝贵寄存器，且每次算地址都要执行真正的整数乘除法指令；
* 写 `_64{}`（等价于 `cute::Int<64>{}` 或 `cute::C<64>{}`）代表 **C++ 编译期常量（Static Int）**！
  * **核心威力**：因为 Shared Memory 的分块大小（比如 $64 \times 64$）、Warp 的线程数（32）、向量长度（8）在编译时是绝对固定的，使用 `_64{}` 后，`nvcc` 编译器会在编译阶段把所有复杂的坐标映射、循环边界和移位操作**全部常量折叠（Constant Folding）**，生成零额外开销的极致机器码！
  * **经验法则**：全局矩阵的总维度 `M`, `N`, `K`（比如 `num_tokens`）在运行期才确定，用普通 `int`；而 **CTA 分块大小 (`bM`, `bN`, `bK`)、Shared Memory Layout、线程排布全部用 `_N{}` 编译期常量**！

### 3. `local_tile`：如何把大矩阵切给每个 Block (CTA)？
看 [cute_layout_and_copy.cu](./cute_layout_and_copy.cu) 的第 67-76 行：
```cpp
// 1. mA 是整个 Global Memory 上的大矩阵，形状 (M, K)
Tensor mA = make_tensor(make_gmem_ptr(ptr_in), make_shape(M, K), make_stride(K, _1{}));

// 2. cta_tiler 是每个 Block 负责的小块大小 (_64, _64)
auto cta_tiler = make_shape(_64{}, _64{});
auto cta_coord = make_coord(blockIdx.y, blockIdx.x);

// 3. local_tile 一行代码切出第 (blockIdx.y, blockIdx.x) 块 64x64 的子张量 gA！
Tensor gA = local_tile(mA, cta_tiler, cta_coord); // gA 的形状自动变成 (_64, _64)
```
再也不用自己手写 `ptr_in + blockIdx.y * 64 * K + blockIdx.x * 64` 了！

---

## 三、核心概念 2：`Swizzle<B, M, S>` —— 为什么它是 RTX 6000D (`sm_120`) 的救星？

### 1. 为什么在矩阵乘里绝对不能用 Padding（填充列）来消除 Bank Conflict？
在第 2 课我们讲过，GPU 的 Shared Memory 有 32 个 Bank（每 4 字节 1 个 Bank，整圈 $32 \times 4\text{B} = 128\text{B}$）。
当我们把一个 $8 \times 64$ 的 `bfloat16` 矩阵块按行优先（每行 $64 \times 2\text{B} = 128\text{B}$，刚好绕 32 个 Bank 一整圈！）放进 Shared Memory 时：
* 第 0 行第 0 列在 **Bank 0**；
* 第 1 行第 0 列（偏移 $128\text{B}$）**又回到了 Bank 0**！
* 第 $2, 3, \dots, 7$ 行的第 0 列**全都在 Bank 0**！
当 Tensor Core 需要按列方向或跨行读取矩阵块到寄存器（比如 `ldmatrix` 指令）时，所有线程全部撞在同一个 Bank 上，触发严重的 **Bank Conflict**！

**那为什么不用传统教材里的 Padding（比如声明 `smem[8][64 + 8]`，每行多塞 8 个空元素）？**
因为两个致命原因：
1. **破坏 128-bit (16B) 向量对齐与 TMA 硬件约束**：一旦加了不规则 Padding，或者是更小粒度的数据类型，极易破坏硬件异步拷贝指令（`cp.async` / TMA `cp.async.bulk`）强制要求的 16B / 128B 地址对齐！
2. **浪费 RTX 6000D (`sm_120`) 极其吃紧的 Shared Memory**：记住，数据中心 B200 (`sm_100`) 每个 SM 有 228KB 共享内存，而你的 **RTX 6000D (`sm_120`) 每个 SM 只有 99KB 共享内存**！如果用 Padding 浪费掉 10% 的共享内存，原本能放下 3 级流水线（3-Stage Pipeline）的矩阵分块可能直接因为超限而编译失败或掉 Occupancy！

### 2. 图解 `Swizzle<B, M, S>` 的异或重排魔法
CuTe 的 `Swizzle<B, M, S>` 不增加哪怕 1 个字节的额外内存，而是**把物理地址中的“行号位”与“列号位”做一次按位异或 (`XOR, ^`)**！

以 CUTLASS / vLLM 源码里最常见的 **`Swizzle<3, 3, 3>`**（配合 `8行 x 64列 bfloat16`）为例：
* **`M = 3`（最小原子块大小 $= 2^3 = 8$ 个元素）**：
  因为我们每次用 128-bit 指令一次读写 **8 个连续的 `bfloat16`（16 字节）**，这 8 个元素内部绝对不能被打散！所以最底层的 $2^3 = 8$ 个元素绑死成一个“128-bit 向量原子（Vector Atom）”，位坐标的最低 3 位（bit `0..2`）保持不动。
* **`S = 3`（列向量位数 $= 3$ 位）**：
  一行有 64 个 `bfloat16`，既然每 8 个绑成 1 个向量，那么一行就有 $64 / 8 = 8 = 2^3$ 个列向量（列向量编号 `col_vec` 占 3 个二进制位，即 bit `3..5`）。
* **`B = 3`（参与异或的行号位数 $= 3$ 位）**：
  每 $2^3 = 8$ 行为一个循环周期（行号 `row` 占 3 个二进制位，即 bit `6..8`）。
* **异或公式**：
  $$\text{new\_col\_vec} = \text{col\_vec} \oplus \text{row}$$
  看！对于原本全撞在 `col_vec = 0`（Bank 0..3）的第 $0 \sim 7$ 行：
  * 第 0 行 (`row=0`)：$0 \oplus 0 = 0 \implies$ 落在第 0 号向量位置（**Bank 0..3**）
  * 第 1 行 (`row=1`)：$0 \oplus 1 = 1 \implies$ 落在第 1 号向量位置（**Bank 4..7**）
  * 第 2 行 (`row=2`)：$0 \oplus 2 = 2 \implies$ 落在第 2 号向量位置（**Bank 8..11**）
  * ……
  * 第 7 行 (`row=7`)：$0 \oplus 7 = 7 \implies$ 落在第 7 号向量位置（**Bank 28..31**）

**结论**：加了一行 `composition(Swizzle<3, 3, 3>{}, layout)` 后，无论是按行写（Global $\rightarrow$ Shared），还是按列读（Shared $\rightarrow$ Register），32 个 Bank 全部满载无冲突，且 **0 字节内存浪费**！

---

## 四、核心概念 3：`TiledCopy` 与 CUTLASS 变量命名密码

### 1. `make_tiled_copy` 的三个参数怎么看？
看 [cute_layout_and_copy.cu](./cute_layout_and_copy.cu) 的第 127-132 行：
```cpp
using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, __nv_bfloat16>;
auto tiled_copy_g2s = make_tiled_copy(
    CopyAtom{},                                                    // 参数 1：用什么硬件指令搬？
    make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})), // 参数 2：线程怎么排队 (ThrLayout)？
    make_layout(make_shape(_1{}, _8{}))                            // 参数 3：单线程一次拿几个数 (ValLayout)？
);
```
遇到任何 `make_tiled_copy`，按下面三步直接心算：
1. **看 `CopyAtom`**：`SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>` 代表调用硬件 **`cp.async.cg.shared.global`** 指令。
   * *为什么它比普通赋值快？* 普通写法 `sA[i] = gA[i]` 必须先把数据从 GDDR7 读进**寄存器（Register）**，再从寄存器写进 **Shared Memory**；而 `cp.async` 直接让内存控制器把数据从 L2 Cache **硬件旁路直写进 Shared Memory**，完全不经过寄存器，省下大量寄存器给 Tensor Core 用！
2. **看 `ThrLayout`**：`Shape<_16, _8>, Stride<_8, _1>` 代表总共 $16 \times 8 = 128$ 个线程，按行优先排成 **16 行 $\times$ 8 列** 的方阵。
3. **看 `ValLayout`**：`Shape<_1, _8>` 代表每个线程每次指令处理 **1 行 $\times$ 8 列** 个 `bfloat16`（正好 $8 \times 2\text{B} = 16\text{B} = \text{uint128\_t}$）。
4. **合起来算覆盖范围**：
   * 行方向覆盖：$16 \text{ (线程行)} \times 1 \text{ (元素/线程)} = 16$ 行；
   * 列方向覆盖：$8 \text{ (线程列)} \times 8 \text{ (元素/线程)} = 64$ 列。
   * 而我们的 Tile `gA` 大小是 $64 \times 64$，所以当调用 `cute::copy(tiled_copy_g2s, tAgA, tAsA)` 时，CuTe 会在编译期自动帮每个线程生成 $64 / 16 = 4$ 条 `cp.async` 指令，把整个 $64 \times 64$ 分块搬完！

### 2. 秒懂 vLLM / CUTLASS 源码里的变量名（匈牙利命名法）
以后看 CUTLASS / FlashInfer 源码，记住这张前缀对照表，永不迷路：

| 变量名示例 | 拆解含义 | 解释 |
| :--- | :--- | :--- |
| **`mA`** | **m**atrix **A** | 全局显存上的完整大矩阵 $A$（形状 $M \times K$） |
| **`gA`** | **g**lobal tile **A** | 当前 CTA (Block) 通过 `local_tile` 切出来的全局显存分块（如 $64 \times 64$） |
| **`sA`** | **s**hared tile **A** | 当前 CTA 在 Shared Memory 上的分块（通常带 `Swizzle`） |
| **`rA` / `tArA`** | **r**egister fragment **A** | 当前线程在寄存器（Register）里持有的切片数组 |
| **`tAgA`** | **t**hread-partitioned **A** in **gA** | `thr_copy.partition_S(gA)`：当前线程负责从 `gA` 读取的源数据视图 |
| **`tAsA`** | **t**hread-partitioned **A** in **sA** | `thr_copy.partition_D(sA)`：当前线程负责写入 `sA` 的目标数据视图 |

---

## 五、动手实验与课后思考

### 1. 运行本课代码
在你的 GPU 环境（如 RTX 6000D 服务器或 Colab）中运行：
```bash
python lesson03_cute_layout_copy/test_and_bench.py
```
你会看到终端直接打印出：
* 未加 `Swizzle` 时，8 行数据全部撞在 `Bank 0` 的抓包证据；
* 加上 `Swizzle<3, 3, 3>` 后，8 行数据完美散列到 `Bank 0..31` 的映射表；
* 以及 `cp.async` 分块搬运 100% 精度通过的提示！

### 2. 课后小思考（通往第 4 课 Tensor Core 矩阵乘的桥梁）
现在我们已经用 `TiledCopy` + `cp.async` + `Swizzle<3, 3, 3>` 把矩阵 $A$ 的分块 (`sA`) 和矩阵 $B$ 的分块 (`sB`) 零冲突地搬进了 RTX 6000D 的 Shared Memory。
下一步，在 **第 4 课** 中：
> 我们如何用 CuTe 的另一个神器 **`TiledMMA`**，把 `sA` 和 `sB` 里的数据喂给 **Blackwell `sm_120` 的 Tensor Core**，完成真正的高性能矩阵乘 $C = A \times B$？
> 为什么说 `sm_120` 不能用 B200 (`sm_100`) 的 `tcgen05` MMA Atom，而必须用基于寄存器的 Warp-level MMA Atom？
