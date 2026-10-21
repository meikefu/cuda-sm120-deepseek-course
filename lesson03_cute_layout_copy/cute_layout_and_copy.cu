#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

// 引入 CUTLASS 3.x/4.x 的核心 CuTe 头文件
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

using namespace cute;

// ============================================================================
// 第 3 课核心算子：使用 CuTe Layout、Swizzle<3,3,3> 与 TiledCopy (cp.async)
// 实现零 Bank Conflict 的 Global -> Shared -> Register -> Global 高速分块搬运与转置
// ============================================================================
// 学习目标：
// 1. 看懂 vLLM / FlashInfer / DeepGEMM 里满屏的 Shape<_M, _K>、Stride、local_tile
// 2. 看懂 Swizzle<B, M, S> 如何在不浪费 RTX 6000D (sm_120) 宝贵的 99KB SMEM 的前提下消除 Bank Conflict
// 3. 看懂 make_tiled_copy 与 tAgA / tAsA 命名规范，掌握 cp.async 硬件异步旁路拷贝
// ============================================================================

// ----------------------------------------------------------------------------
// 演示函数 1：在 Host/Device 端打印并验证 CuTe Layout 与 Swizzle 的地址映射
// ----------------------------------------------------------------------------
__global__ void inspect_cute_layout_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("========== [1] 基础 Row-Major Layout (8行 x 64列 BF16) ==========\n");
        // _8 和 _64 是编译期常量 (C<8>, C<64>)，编译器会直接将下标乘法折叠为常量指令！
        // 行优先 (Row-Major)：跨 1 行需要跳过 64 个元素，跨 1 列跳过 1 个元素 -> Stride<_64, _1>
        auto s_layout_naive = make_layout( make_shape(_8{}, _64{}),
                                           make_stride(_64{}, _1{}) );

        // 检查同一列 (col = 0..7) 不同行 (row = 0..7) 在 Shared Memory 中的 Bank 分布：
        // 每个 BF16 占 2 字节，每 2 个 BF16 (4 字节) 属于 1 个 Bank，每 8 个 BF16 (16 字节, 128-bit) 占 4 个连续 Bank。
        // 如果不加 Swizzle，按列读取 128-bit 向量 (col_vec = 0) 时：
        printf("未加 Swizzle 时，第 0 列向量 (col=0) 在各行的物理元素偏移与所属 Bank:\n");
        for (int r = 0; r < 8; ++r) {
            int offset = s_layout_naive(r, 0); // 传入逻辑坐标 (row=r, col=0)
            int bank = (offset * sizeof(__nv_bfloat16) / 4) % 32;
            printf("  row=%d, col=0 -> offset=%3d -> Bank %2d\n", r, offset, bank);
        }
        printf("  [警告] 偶数行全部撞在 Bank 0，奇数行全部撞在 Bank 0！产生严重的 Bank Conflict！\n\n");

        printf("========== [2] 加上 Swizzle<3, 3, 3> 后的零冲突 Layout ==========\n");
        // Swizzle<B=3, M=3, S=3> 含义：
        //   M = 3: 最小连续不拆分的块大小 = 2^3 = 8 个 BF16 (即 16 Bytes = 128-bit 向量保持完整！)
        //   B = 3: 参与异或的行数周期 = 2^3 = 8 行
        //   S = 3: 行坐标右移 3 位与列向量坐标异或（因为每行有 64 个 BF16 = 8 个 128-bit 向量 = 2^3）
        auto s_layout_swizzled = composition( Swizzle<3, 3, 3>{}, s_layout_naive );

        printf("加上 Swizzle<3,3,3> 后，第 0 列向量 (col=0) 在各行的物理元素偏移与所属 Bank:\n");
        for (int r = 0; r < 8; ++r) {
            int offset = s_layout_swizzled(r, 0);
            int bank = (offset * sizeof(__nv_bfloat16) / 4) % 32;
            printf("  row=%d, col=0 -> offset=%3d -> Bank %2d..%2d (128-bit 占 4 个 Bank)\n",
                   r, offset, bank, bank + 3);
        }
        printf("  [完美] 8 行的 128-bit 访问被均匀打散到 Bank 0..31，0 Bank Conflict，且 0 字节内存浪费！\n");
    }
}

// ----------------------------------------------------------------------------
// 演示函数 2：用 CuTe TiledCopy + Swizzle + cp.async 编写高速分块矩阵拷贝 Kernel
// 每个 CTA (Block) 负责一个 bM x bK = 64 x 64 的 BF16 矩阵分块
// ----------------------------------------------------------------------------
template <class ProblemShape, class CtaTiler, class SmemLayout, class TiledCopyG2S>
__global__ void cute_tiled_copy_kernel(
    const __nv_bfloat16* __restrict__ ptr_in,
    __nv_bfloat16* __restrict__ ptr_out,
    ProblemShape problem_shape, // (M, K)
    CtaTiler cta_tiler,         // (_64, _64)
    SmemLayout sA_layout,       // Swizzled Shared Memory Layout (_64, _64)
    TiledCopyG2S tiled_copy_g2s
) {
    // 分配对齐的 Shared Memory（64 * 64 * 2 Bytes = 8 KB，远小于 sm_120 的 99KB 限制）
    __shared__ __align__(128) __nv_bfloat16 smem_buf[cosize_v<SmemLayout>];

    int M = size<0>(problem_shape);
    int K = size<1>(problem_shape);

    // 1. 将裸指针包装为 CuTe 的全局 Tensor (mA, mOut)，形状为 (M, K)，行优先 Stride(K, _1)
    // CUTLASS 命名习惯：m 前缀代表整个 Global Memory 矩阵 (Matrix)
    Tensor mA   = make_tensor(make_gmem_ptr(ptr_in),  make_shape(M, K), make_stride(K, _1{}));
    Tensor mOut = make_tensor(make_gmem_ptr(ptr_out), make_shape(M, K), make_stride(K, _1{}));

    // 2. 使用 local_tile 切出当前 CTA (blockIdx.y, blockIdx.x) 负责的 64x64 分块
    // CUTLASS 命名习惯：g 前缀代表当前 CTA 负责的 Global Memory Tile
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x);
    Tensor gA   = local_tile(mA,   cta_tiler, cta_coord); // Shape: (_64, _64)
    Tensor gOut = local_tile(mOut, cta_tiler, cta_coord); // Shape: (_64, _64)

    // 3. 将 Shared Memory 包装为带 Swizzle<3,3,3> 的 CuTe Tensor sA
    // CUTLASS 命名习惯：s 前缀代表 Shared Memory Tile
    Tensor sA = make_tensor(make_smem_ptr(smem_buf), sA_layout); // Shape: (_64, _64)

    // 4. 获取当前线程 (threadIdx.x) 在 TiledCopy 中的专属切片器 (ThrCopy)
    ThrCopy thr_copy = tiled_copy_g2s.get_slice(threadIdx.x);

    // CUTLASS 核心命名法（读懂 vLLM/FlashInfer 源码的关键）：
    //   tAgA: Thread-partitioned A in Global A (源切片 Source)
    //   tAsA: Thread-partitioned A in Shared A (目标切片 Destination)
    Tensor tAgA   = thr_copy.partition_S(gA);
    Tensor tAsA   = thr_copy.partition_D(sA);
    Tensor tAgOut = thr_copy.partition_D(gOut);

    // 5. 发射硬件异步拷贝指令 cp.async (Global -> Swizzled Shared Memory)
    // 数据直接从 GDDR7 旁路写入 Shared Memory，不占用宝贵的寄存器！
    cute::copy(tiled_copy_g2s, tAgA, tAsA);

    // 提交异步拷贝批次并等待完成
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // 6. 分配当前线程的寄存器 Fragment (tArA)，从 Swizzled Shared Memory 读入寄存器再写回全局内存
    // CUTLASS 命名习惯：r 前缀代表 Register Fragment
    Tensor tArA = make_fragment_like(tAgOut);
    cute::copy(tAsA, tArA);
    cute::copy(tArA, tAgOut);
}

void launch_cute_demo(
    const void* ptr_in,
    void* ptr_out,
    int M,
    int K,
    bool print_layout_info,
    cudaStream_t stream = 0
) {
    if (print_layout_info) {
        inspect_cute_layout_kernel<<<1, 1, 0, stream>>>();
    }

    // 定义每个 CTA 处理的 Tile 大小：bM = 64, bK = 64
    auto bM = _64{};
    auto bK = _64{};
    auto cta_tiler = make_shape(bM, bK);

    // 定义带 Swizzle<3, 3, 3> 的 Shared Memory Layout:
    // 先定义基础 8x64 Atom，再 tile_to_shape 扩展到 64x64
    auto smem_atom = composition(Swizzle<3, 3, 3>{},
                                 make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{})));
    auto sA_layout = tile_to_shape(smem_atom, make_shape(bM, bK));

    // 定义 TiledCopy：
    // 1. Copy_Atom: 使用 SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>，单条指令异步搬运 16 Bytes (8 个 BF16)
    // 2. ThrLayout: 128 个线程排成 16 行 x 8 列 (Stride<_8, _1>)
    // 3. ValLayout: 每个线程沿列方向连续读取 8 个 BF16 (Shape<_1, _8>)
    //    单次拷贝覆盖范围：(16 * 1) 行 x (8 * 8) 列 = 16 x 64，循环 4 次恰好覆盖 64 x 64！
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, __nv_bfloat16>;
    auto tiled_copy_g2s = make_tiled_copy(
        CopyAtom{},
        make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})), // ThrLayout: 16x8 = 128 threads
        make_layout(make_shape(_1{}, _8{}))                            // ValLayout: 1x8 BF16 (128-bit)
    );

    dim3 block(size(tiled_copy_g2s)); // 128 threads
    dim3 grid(K / size(bK), M / size(bM));

    cute_tiled_copy_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(ptr_in),
        reinterpret_cast<__nv_bfloat16*>(ptr_out),
        make_shape(M, K),
        cta_tiler,
        sA_layout,
        tiled_copy_g2s
    );
}
