#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

// ============================================================================
// 第 4 课核心算子：针对 Blackwell sm_120 (RTX 6000D, 99KB SMEM 限制)
// 使用 CuTe TiledMMA + ldmatrix + cp.async 双缓冲流水线编写 Tensor Core GEMM
// 支持 Epilogue Per-Token / Block Scale 融合缩放：D = alpha * (scale_a * scale_b) * (A * B^T)
// ============================================================================
// 为什么这门课是 RTX 6000D (sm_120) 算子优化的分水岭？
// 1. 数据中心 B200 (sm_100) 使用 tcgen05 指令和 TMEM；H100 (sm_90) 使用 wgmma 指令。
//    而 RTX 6000D (sm_120) 既没有 TMEM 也没有 wgmma，它的 Tensor Core 全部走
//    Warp-level 寄存器路径 (mma.sync.aligned.m16n8k16 / m16n8k32)！
// 2. RTX 6000D 每个 SM 只有 99KB Shared Memory。通过精心设计的 (bM=128, bN=128, bK=32, Stages=2)
//    流水线，本算子仅占用 32KB Shared Memory，每个 SM 可同时驻留 3 个 CTA，吃满算力！
// ============================================================================

template <
    class ProblemShape,
    class CtaTiler,
    class SmemLayoutA,
    class SmemLayoutB,
    class TiledCopyA,
    class TiledCopyB,
    class TiledMma
>
__global__ void __launch_bounds__(128, 2)
sm120_cute_gemm_scaled_kernel(
    const __nv_bfloat16* __restrict__ ptr_A, // [M, K] Row-Major (K-Major)
    const __nv_bfloat16* __restrict__ ptr_B, // [N, K] Row-Major (即 B^T 为 Column-Major, K 维连续!)
    const float* __restrict__ scale_a,       // [M] Per-token 激活缩放因子
    const float* __restrict__ scale_b,       // [N] Per-channel 权重缩放因子
    __nv_bfloat16* __restrict__ ptr_C,       // [M, N] Row-Major 输出
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    SmemLayoutA sA_layout,                   // (bM, bK, _2) 双缓冲
    SmemLayoutB sB_layout,                   // (bN, bK, _2) 双缓冲
    TiledCopyA g2s_copy_a,
    TiledCopyB g2s_copy_b,
    TiledMma tiled_mma
) {
    // 分配双缓冲 Shared Memory: (128*32 + 128*32) * 2 stages * 2B = 32 KB (完美契合 sm_120 99KB 限制)
    __shared__ __align__(128) __nv_bfloat16 smem_A[cosize_v<SmemLayoutA>];
    __shared__ __align__(128) __nv_bfloat16 smem_B[cosize_v<SmemLayoutB>];

    int M = size<0>(problem_shape);
    int N = size<1>(problem_shape);
    int K = size<2>(problem_shape);

    // 1. 构建全局内存 Tensor：注意大模型 Linear 层权重 B 形状是 [N, K]，K 维始终连续 (Stride<K, _1>)！
    Tensor mA = make_tensor(make_gmem_ptr(ptr_A), make_shape(M, K), make_stride(K, _1{}));
    Tensor mB = make_tensor(make_gmem_ptr(ptr_B), make_shape(N, K), make_stride(K, _1{}));
    Tensor mC = make_tensor(make_gmem_ptr(ptr_C), make_shape(M, N), make_stride(N, _1{}));

    // 2. 切出当前 CTA (blockIdx.x, blockIdx.y) 负责的 Tile
    //    gA: (bM, bK, k_tiles), gB: (bN, bK, k_tiles), gC: (bM, bN)
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{});

    // 3. 构建 Shared Memory Tensor: (bM, bK, _2), (bN, bK, _2)
    Tensor sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem_B), sB_layout);

    // 4. Global -> Shared 的线程切片视图 (tAgA, tAsA, tBgB, tBsB)
    ThrCopy thr_copy_a = g2s_copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k_tiles)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, _2)

    ThrCopy thr_copy_b = g2s_copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    // 5. Tensor Core TiledMMA 的线程切片视图与寄存器累加器分配
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);            // (MMA, MMA_M, MMA_K, _2)
    Tensor tCsB = thr_mma.partition_B(sB);            // (MMA, MMA_N, MMA_K, _2)
    Tensor tCgC = thr_mma.partition_C(gC);            // (MMA, MMA_M, MMA_N)

    // 分配寄存器 Fragment：tCrA 存 A 的寄存器块，tCrB 存 B 的寄存器块，tCrC 存 FP32 累加结果
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0));
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0));
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC); // 初始化累加器为 0.0f

    // ========================================================================
    // 阶段 A：Prologue (流水线预取第 0 个 K-tile 到 Stage 0 的 Shared Memory)
    // ========================================================================
    int k_tile_count = size<3>(tAgA);
    cute::copy(g2s_copy_a, tAgA(_, _, _, 0), tAsA(_, _, _, 0));
    cute::copy(g2s_copy_b, tBgB(_, _, _, 0), tBsB(_, _, _, 0));
    cp_async_fence();

    // ========================================================================
    // 阶段 B：Mainloop (双缓冲乒乓软件流水线：边用 Tensor Core 算当前块，边预取下一块)
    // ========================================================================
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int read_stage  = k_tile % 2;
        int write_stage = (k_tile + 1) % 2;

        // 如果还有下一个 K-tile，立即异步发起 Global -> Shared 预取！
        if (k_tile + 1 < k_tile_count) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, k_tile + 1), tAsA(_, _, _, write_stage));
            cute::copy(g2s_copy_b, tBgB(_, _, _, k_tile + 1), tBsB(_, _, _, write_stage));
        }
        cp_async_fence();

        // 等待当前 read_stage 的数据到达 Shared Memory
        if (k_tile + 1 < k_tile_count) {
            cp_async_wait<1>(); // 允许在途保留 1 个异步批次（即刚发射的 write_stage）
        } else {
            cp_async_wait<0>(); // 最后一个块，等待全部完成
        }
        __syncthreads();

        // 将当前 Stage 的数据从 Shared Memory 搬入寄存器，并调用 Tensor Core (cute::gemm)
        cute::copy(tCsA(_, _, _, read_stage), tCrA);
        cute::copy(tCsB(_, _, _, read_stage), tCrB);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

        __syncthreads();
    }

    // ========================================================================
    // 阶段 C：Epilogue (尾声融合：在寄存器内直接乘以 scale_a[m] * scale_b[n] 并转 BF16 写回)
    // 建一张逻辑坐标网格 cC，用来查询当前线程每个寄存器元素对应的全局行号 m 和列号 n
    // ========================================================================
    Tensor idC = make_identity_tensor(make_shape(size<0>(cta_tiler), size<1>(cta_tiler)));
    Tensor tCcC = thr_mma.partition_C(idC);

    int cta_m_base = blockIdx.x * size<0>(cta_tiler);
    int cta_n_base = blockIdx.y * size<1>(cta_tiler);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        int m_idx = cta_m_base + get<0>(coord);
        int n_idx = cta_n_base + get<1>(coord);
        if (m_idx < M && n_idx < N) {
            float scaled_val = tCrC(i) * scale_a[m_idx] * scale_b[n_idx];
            tCgC(i) = __float2bfloat16(scaled_val);
        }
    }
}

void launch_sm120_cute_gemm_scaled(
    const void* ptr_A,
    const void* ptr_B,
    const float* scale_a,
    const float* scale_b,
    void* ptr_C,
    int M,
    int N,
    int K,
    cudaStream_t stream = 0
) {
    // 专为 sm_120 (99KB SMEM) 设计的黃金分块大小：128 x 128 x 32，双缓冲 (2 Stages)
    auto bM = _128{};
    auto bN = _128{};
    auto bK = _32{};
    auto bStages = _2{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // 带 Swizzle<2, 3, 3> 的 Shared Memory Layout (bK=32 个 BF16 = 64 Bytes)
    auto smem_atom = composition(Swizzle<2, 3, 3>{},
                                 make_layout(make_shape(_8{}, _32{}), make_stride(_32{}, _1{})));
    auto sA_layout = tile_to_shape(smem_atom, make_shape(bM, bK, bStages));
    auto sB_layout = tile_to_shape(smem_atom, make_shape(bN, bK, bStages));

    // 128 线程 (4 个 Warp) 异步拷贝 TiledCopy: 32行 x 4列，每线程一次搬 8 个 BF16 (128-bit)
    using CopyAtomG2S = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, __nv_bfloat16>;
    auto g2s_copy = make_tiled_copy(
        CopyAtomG2S{},
        make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
        make_layout(make_shape(_1{}, _8{}))
    );

    // Tensor Core TiledMMA: 使用 SM80_16x8x16_F32BF16BF16F32_TN 硬件指令
    // (在 Ampere / Hopper / Blackwell sm_120 上完全原生支持！)
    // 4 个 Warp 排成 2x2 矩阵 (Layout<Shape<_2, _2>>)，单次覆盖 32x16x16
    using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    auto tiled_mma = make_tiled_mma(
        MmaAtom{},
        make_layout(make_shape(_2{}, _2{}, _1{})), // 2x2 Warps = 128 Threads
        Tile<_32, _32, _16>{}
    );

    dim3 block(size(tiled_mma)); // 128 threads
    dim3 grid(M / size(bM), N / size(bN));

    sm120_cute_gemm_scaled_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(ptr_A),
        reinterpret_cast<const __nv_bfloat16*>(ptr_B),
        scale_a,
        scale_b,
        reinterpret_cast<__nv_bfloat16*>(ptr_C),
        make_shape(M, N, K),
        cta_tiler,
        sA_layout,
        sB_layout,
        g2s_copy,
        g2s_copy,
        tiled_mma
    );
}
