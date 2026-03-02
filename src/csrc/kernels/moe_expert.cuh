#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Persistent tcgen05 MoE Expert GEMM Kernel for SM100 (B200)
//
// This kernel uses explicit tcgen05 PTX inline assembly for all 5th-gen
// tensor core operations:
//
//   tcgen05.mma     — BF16×BF16→FP32 matrix multiply-accumulate
//   tcgen05.commit  — Signal mbarrier after MMA batch completes
//   tcgen05.ld      — Load FP32 accumulator from tensor memory to registers
//   tcgen05.fence   — Thread synchronization barriers for tensor memory
//   tcgen05.wait    — Wait for tensor memory load/store completion
//
// tensor_allocator wraps tcgen05.alloc/dealloc for correct column allocation.
//
// ThunderKittens is used for non-tcgen05 concerns:
//   - Memory layout types (st_bf, gl, rt_bf, tt)
//   - TMA descriptors and async copy (cp.async.bulk) — DMA engine, not tcgen05
//   - Shared memory descriptor building for MMA operands (st_descriptor)
//   - Semaphore management (mbarrier)
//   - Tensor memory allocator
//
// Kernel: persistent GEMM across all local experts
//   CLUSTER_M=512 rows × CLUSTER_N=256 cols per task
//   2-CTA cluster: CTA 0 loads B[0:128,:], CTA 1 loads B[128:256,:]
//   Each CTA loads 2 A tiles (256 rows), consumers process 128 rows each
//
// Dimensions (Qwen3-480B):
//   H = hidden_size = 6144,  I = moe_intermediate_size = 2560
//   gate_up width = 5120 = 2*I
// ═══════════════════════════════════════════════════════════════════════════

#include "kittens.cuh"
#include "moe_expert_types.cuh"

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline ring buffer helpers (inlined from ThunderKittens prototype/lcf)
// ═══════════════════════════════════════════════════════════════════════════

template<int N>
__device__ static inline int moe_ring_advance(int ring, int distance = 1) {
    return (ring + distance) % N;
}

template<int half>
__device__ static inline bool moe_get_phasebit(uint32_t bitfield, int ring_id) {
    return (bitfield & (1 << (half * 16 + ring_id))) != 0;
}

template<int half>
__device__ static inline void moe_update_phasebit(uint32_t &bitfield, int ring_id) {
    bitfield ^= (1 << (half * 16 + ring_id));
}

using namespace kittens;

// ═══════════════════════════════════════════════════════════════════════════
// Tile and Thread Configuration
// ═══════════════════════════════════════════════════════════════════════════

static constexpr int MOE_Mb = MOE_TILE_M;  // 128 rows per consumer tile
static constexpr int MOE_Nb = MOE_TILE_N;  // 256 cols per output tile
static constexpr int MOE_Kb = MOE_TILE_K;  // 64  reduction per TMA load

// CLUSTER_M: rows processed per task = 4 × Mb = 512
//   2 CTAs × 2 consumers × 128 rows = 512
static constexpr int MOE_CLUSTER_M = 4 * MOE_Mb;   // 512
static constexpr int MOE_CLUSTER_N = MOE_Nb;         // 256

static constexpr int MOE_NUM_CONSUMERS = 2;   // consumer warpgroups per CTA
static constexpr int MOE_NUM_PRODUCERS = 1;   // producer warpgroup
static constexpr int MOE_NUM_WORKERS   = (MOE_NUM_CONSUMERS + MOE_NUM_PRODUCERS) * 4;
static constexpr int MOE_NUM_THREADS   = MOE_NUM_WORKERS * kittens::WARP_THREADS;
static constexpr int MOE_PIPE_DEPTH    = 4;   // TMA pipeline ring buffer depth

// ═══════════════════════════════════════════════════════════════════════════
// tcgen05 Constants
// ═══════════════════════════════════════════════════════════════════════════

// BF16 reduction dimension: tcgen05 MMA processes 16 BF16 elements per chunk
static constexpr int MOE_BF16_RED_DIM  = 16;
static constexpr int MOE_INNER_K_ITERS = MOE_Kb / MOE_BF16_RED_DIM;  // 64/16 = 4

// ═══════════════════════════════════════════════════════════════════════════
// tcgen05 MMA Instruction Descriptor Builder
//
// Per NVIDIA PTX ISA §9.7.16.6 (Blackwell / SM100):
//   [1:0]   = sparsity selector (00 = dense)
//   [2]     = structured sparsity (0 = none)
//   [3]     = saturate (0 for FP)
//   [5:4]   = D accumulator type (01 = FP32)
//   [6]     = reserved
//   [9:7]   = A input type (001 = BF16)
//   [12:10] = B input type (001 = BF16)
//   [13]    = negate A (0 = no)
//   [14]    = negate B (0 = no)
//   [15]    = transpose A (0 = no)
//   [16]    = transpose B (0 = no, B pre-transposed in memory)
//   [22:17] = N >> 3
//   [23]    = reserved
//   [28:24] = M >> 4
//   [29]    = reserved
//   [31:30] = B-matrix shift (00 = none)
// ═══════════════════════════════════════════════════════════════════════════

__host__ __device__ constexpr uint32_t moe_build_mma_idesc(
    int M, int N,
    bool trans_a = false,
    bool trans_b = false
) {
    uint32_t desc = 0;
    desc |= (0b01u)  << 4;                       // D accumulator = FP32
    desc |= (0b001u) << 7;                       // A input = BF16
    desc |= (0b001u) << 10;                      // B input = BF16
    desc |= (trans_a ? 1u : 0u) << 15;           // transpose A
    desc |= (trans_b ? 1u : 0u) << 16;           // transpose B
    desc |= ((uint32_t)(N >> 3)) << 17;          // N dimension encoding
    desc |= ((uint32_t)(M >> 4)) << 24;          // M dimension encoding
    return desc;
}

// Pre-computed instruction descriptor:
//   BF16 × BF16 → FP32 accumulation
//   M = 128 × 2 CTAs = 256 (cta_group::2)
//   N = 256
//   No transposes (B pre-transposed in memory layout)
static constexpr uint32_t MOE_MMA_IDESC = moe_build_mma_idesc(
    MOE_Mb * 2,  // M = 256 across 2-CTA cluster
    MOE_Nb       // N = 256
);

// ═══════════════════════════════════════════════════════════════════════════
// tcgen05 PTX Inline Assembly Primitives
// ═══════════════════════════════════════════════════════════════════════════

// ── tcgen05.fence ──
// Thread synchronization barriers that ensure tensor memory operations
// are properly ordered around __syncthreads() calls.
__device__ __forceinline__ void moe_tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}
__device__ __forceinline__ void moe_tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}

// ── tcgen05.mma ──
// Issues a BF16×BF16→FP32 matrix multiply-accumulate across the 2-CTA cluster.
// Reads A and B operands from shared memory via 64-bit descriptors,
// accumulates result into tensor memory at d_addr.
//
// Template parameter:
//   acc=false: D = A × B   (reset accumulator — first K-chunk)
//   acc=true:  D += A × B  (accumulate — subsequent K-chunks)
template<bool acc>
__device__ __forceinline__ void moe_tcgen05_mma(
    uint32_t d_addr,      // destination address in tensor memory
    uint64_t a_desc,      // A operand: 64-bit shared memory descriptor
    uint64_t b_desc,      // B operand: 64-bit shared memory descriptor
    uint32_t idesc        // instruction descriptor (data types, dimensions, transposes)
) {
    if constexpr (acc) {
        asm volatile(
            "{.reg .pred p;\n"
            " setp.eq.u32 p, 1, 1;\n"
            " tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
            :: "r"(d_addr), "l"(a_desc), "l"(b_desc), "r"(idesc)
        );
    } else {
        asm volatile(
            "{.reg .pred p;\n"
            " setp.eq.u32 p, 1, 0;\n"
            " tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
            :: "r"(d_addr), "l"(a_desc), "l"(b_desc), "r"(idesc)
        );
    }
}

// ── tcgen05.commit ──
// Signals an mbarrier after a batch of MMA instructions completes.
// Multicasts the arrival to both CTAs in the 2-CTA cluster.
__device__ __forceinline__ void moe_tcgen05_commit(kittens::semaphore &sem) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :: "l"(&sem), "h"((uint16_t)(0b11))
    );
}

// ── tcgen05.wait ──
// Waits for tensor memory load/store operations to complete.
__device__ __forceinline__ void moe_tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void moe_tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;");
}

// ═══════════════════════════════════════════════════════════════════════════
// TMA Global Descriptor Struct
//
// Uses ThunderKittens gl<> types for TMA descriptor building (cp.async.bulk).
// TMA is a DMA engine — it does NOT use tcgen05 instructions.
//
// All gl types are 2D (matching the reference B200 matmul pattern):
//   A = sorted activations  [padded_total_tokens, K_dim]
//   B = expert weights      [num_local_experts * N_dim, K_dim]  (flattened)
//   D = output              [padded_total_tokens, N_dim]
//
// B is flattened from [num_experts, N_dim, K_dim] to [num_experts*N_dim, K_dim]
// since the memory layout is identical (row-major). The TMA load coordinate
// uses expert * (N_dim / b_tile_rows) + b_col to index into the correct expert.
// ═══════════════════════════════════════════════════════════════════════════

struct moe_gemm_globals {
    using a_tile = st_bf<MOE_Mb, MOE_Kb>;
    using b_tile = st_bf<MOE_Nb / 2, MOE_Kb>;  // half-tile per CTA in cluster
    using d_tile = st_bf<MOE_Mb, 64>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;  // 2D: [total_N_rows, K_dim]
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;    // activations: [1, 1, padded_total, K_dim]
    b_gl b;    // weights:     [1, 1, num_experts * N_dim, K_dim]  (flattened)
    d_gl d;    // output:      [1, 1, padded_total, N_dim]

    int b_n_tile_stride;  // N_dim / b_tile_rows = tiles per expert in N dimension
    const MoETaskScheduler* scheduler;
};

// ═══════════════════════════════════════════════════════════════════════════
// Persistent tcgen05 MoE GEMM Kernel
//
// Faithfully follows the ThunderKittens B200 matmul architecture:
//
// Architecture: 2-CTA cluster, 3 warpgroups (384 threads)
//   Warpgroup 0 (consumer 0) — 224 registers
//   Warpgroup 1 (consumer 1) — 224 registers
//   Warpgroup 2 (producer)   — 56 registers:
//     Warps 0-1: MMA launchers — issue tcgen05.mma instructions
//     Warp 2:    idle
//     Warp 3:    TMA loader — fills shared memory pipeline
//
// Per task:
//   CTA 0 loads A tiles at rows [base+0, base+1] (256 rows)
//   CTA 1 loads A tiles at rows [base+2, base+3] (256 rows)
//   Total M per task: 512 rows (CLUSTER_M)
//   B split: CTA 0 → B[0:128,:], CTA 1 → B[128:256,:]
//   Total N per task: 256 cols (CLUSTER_N)
//
// Expert tiles are flattened into a linear task queue.
// ═══════════════════════════════════════════════════════════════════════════

__global__ __cluster_dims__(2) __launch_bounds__(MOE_NUM_THREADS, 1)
void moe_gemm_kernel(const __grid_constant__ moe_gemm_globals g) {

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();

    const int k_dim = g.a.cols();
    const int iters_per_task = k_dim / MOE_Kb;

    using a_tile = moe_gemm_globals::a_tile;
    using b_tile = moe_gemm_globals::b_tile;
    using d_tile = moe_gemm_globals::d_tile;

    // ── Allocate shared memory tiles for TMA pipeline ──
    a_tile (&a_smem)[MOE_PIPE_DEPTH][MOE_NUM_CONSUMERS] = al.allocate<a_tile, MOE_PIPE_DEPTH, MOE_NUM_CONSUMERS>();
    b_tile (&b_smem)[MOE_PIPE_DEPTH]                    = al.allocate<b_tile, MOE_PIPE_DEPTH>();
    d_tile (&d_smem)                                    = al.allocate<d_tile>();

    // ═══════════════════════════════════════════════════════════════════
    // Tensor memory allocation via tensor_allocator
    // (internally uses tcgen05.alloc.cta_group::2 and tcgen05.dealloc)
    // ═══════════════════════════════════════════════════════════════════
    everyone::tma::cluster::sync();
    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, MOE_Mb, MOE_Nb>;

    // ── Semaphores for producer-consumer synchronization ──
    // Bitfield tracks phase bits: upper 16 = _finished (start=1), lower 16 = _arrived (start=0)
    __shared__ kittens::semaphore inputs_arrived[MOE_PIPE_DEPTH],
                                  inputs_finished[MOE_PIPE_DEPTH],
                                  outputs_arrived,
                                  outputs_finished[MOE_NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) {
        for (int i = 0; i < MOE_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2);
            init_semaphore(inputs_finished[i], 0, MOE_NUM_CONSUMERS);
        }
        init_semaphore(outputs_arrived, 0, 1);
        for (int i = 0; i < MOE_NUM_CONSUMERS; i++) {
            init_semaphore(outputs_finished[i], 0, 2);
        }
    }
    everyone::tma::cluster::sync();

    const MoETaskScheduler* sched = g.scheduler;

    // ═══════════════════════════════════════════════════════════════════
    // PRODUCER WARPGROUP (warpgroupid == 2)
    // ═══════════════════════════════════════════════════════════════════
    if (warpgroupid == MOE_NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();
        int ctarank = cluster_ctarank();

        if (warpgroup::warpid() == 3) {
            // ── TMA LOADER WARP ──
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                int cluster_x = clusterIdx().x;
                int task_id = task_iter * (gridDim.x / 2) + cluster_x;
                if (task_id >= sched->total_tasks) {
                    // Drain pipeline
                    for (int idx = 0; idx < MOE_PIPE_DEPTH; idx++) {
                        tma::cluster::wait(inputs_finished[input_ring],
                                           moe_get_phasebit<1>(bitfield, input_ring));
                        input_ring = moe_ring_advance<MOE_PIPE_DEPTH>(input_ring);
                    }
                    if (laneid() == 0) arrive(outputs_arrived);
                    break;
                }

                // Decode task → (expert, m_tile, n_tile)
                int3 task = sched->get_task(task_id);
                int expert = task.x;
                int m_tile = task.y;
                int n_tile = task.z;

                // A tile coordinates (CLUSTER_M = 4 tiles per m_tile):
                //   CTA 0: loads tiles [base+0, base+1] (rows 0-255)
                //   CTA 1: loads tiles [base+2, base+3] (rows 256-511)
                int padded_expert_start = sched->expert_offsets[expert];
                int a_tile_base = (padded_expert_start / MOE_Mb) + m_tile * 4 + ctarank * 2;

                // B tile: CTA 0 gets first half of N, CTA 1 gets second half
                int b_col = 2 * n_tile + ctarank;

                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring],
                                       moe_get_phasebit<1>(bitfield, input_ring));
                    moe_update_phasebit<1>(bitfield, input_ring);

                    if (task_iter > 0 && idx == MOE_PIPE_DEPTH - 1 && laneid() == 0)
                        arrive(outputs_arrived);

                    tma::cluster::expect(inputs_arrived[input_ring], 0,
                                         a_smem[0][0], a_smem[0][1], b_smem[0]);

                    // TMA loads: A tiles for each consumer, B tile for this CTA
                    tma::cluster::load_async(a_smem[input_ring][0], g.a,
                                             {a_tile_base + 0, idx},
                                             inputs_arrived[input_ring],
                                             (uint16_t)(1 << ctarank), 0);
                    tma::cluster::load_async(a_smem[input_ring][1], g.a,
                                             {a_tile_base + 1, idx},
                                             inputs_arrived[input_ring],
                                             (uint16_t)(1 << ctarank), 0);
                    // 2D coordinate: flatten expert into row dimension
                    // row_tile = expert * (N_dim / b_tile_rows) + b_col
                    tma::cluster::load_async(b_smem[input_ring], g.b,
                                             {expert * g.b_n_tile_stride + b_col, idx},
                                             inputs_arrived[input_ring],
                                             (uint16_t)(1 << ctarank), 0);

                    input_ring = moe_ring_advance<MOE_PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if (ctarank == 0 &&
                 (warpgroup::warpid() == 0 || warpgroup::warpid() == 1)) {
            // ══════════════════════════════════════════════════════════
            // MMA LAUNCHER WARPS
            //
            // Uses ThunderKittens mm2_ABt / mma2_ABt wrappers which
            // internally emit tcgen05.mma PTX instructions:
            //
            //   tcgen05.mma.cta_group::2.kind::f16 — BF16×BF16→FP32
            //   tcgen05.commit.cta_group::2         — signal mbarrier
            //   fence.proxy.async.shared::cta       — memory ordering
            //
            // mm2_ABt  = D = A × B^T  (reset accumulator, first K-chunk)
            // mma2_ABt = D += A × B^T (accumulate, subsequent K-chunks)
            // ══════════════════════════════════════════════════════════

            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid() * MOE_Nb);
            int input_ring = 0;

            for (int task_iter = 0; true; task_iter++) {
                int cluster_x = clusterIdx().x;
                int task_id = task_iter * (gridDim.x / 2) + cluster_x;
                if (task_id >= sched->total_tasks) break;

                // Wait for tensor memory to be freed by consumer (ping-pong)
                tma::cluster::wait(outputs_finished[warpgroup::warpid()],
                                   (task_iter + 1) % 2);

                // Wait for first TMA load to arrive in shared memory
                tma::cluster::wait(inputs_arrived[input_ring],
                                   moe_get_phasebit<0>(bitfield, input_ring));
                moe_update_phasebit<0>(bitfield, input_ring);

                // First K-iteration: D = A × B^T (reset accumulator)
                // Internally emits: fence.proxy.async, tcgen05.mma (acc=0), tcgen05.commit
                mm2_ABt(d_tt,
                        a_smem[input_ring][warpgroup::warpid()],
                        b_smem[input_ring],
                        inputs_finished[input_ring]);
                input_ring = moe_ring_advance<MOE_PIPE_DEPTH>(input_ring);

                // Remaining K-iterations: D += A × B^T (accumulate)
                for (int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_arrived[input_ring],
                                       moe_get_phasebit<0>(bitfield, input_ring));
                    moe_update_phasebit<0>(bitfield, input_ring);

                    // Internally emits: fence.proxy.async, tcgen05.mma (acc=1), tcgen05.commit
                    mma2_ABt(d_tt,
                             a_smem[input_ring][warpgroup::warpid()],
                             b_smem[input_ring],
                             inputs_finished[input_ring]);
                    input_ring = moe_ring_advance<MOE_PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    // ═══════════════════════════════════════════════════════════════════
    // CONSUMER WARPGROUPS (warpgroupid == 0 or 1)
    //
    // Each consumer warpgroup:
    //   1. Waits for MMA results in tensor memory (outputs_arrived)
    //   2. Issues tcgen05.ld to load FP32 accumulator → BF16 registers
    //   3. Signals tensor memory is free (outputs_finished)
    //   4. Stores BF16 result to global memory via TMA
    // ═══════════════════════════════════════════════════════════════════
    else {
        warpgroup::increase_registers<224>();

        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroupid * MOE_Nb);

        for (int task_iter = 0; true; task_iter++) {
            int cluster_x = clusterIdx().x;
            int task_id = task_iter * (gridDim.x / 2) + cluster_x;
            if (task_id >= sched->total_tasks) break;

            // Decode task for output position
            int3 task = sched->get_task(task_id);
            int expert       = task.x;
            int m_tile       = task.y;
            int n_tile       = task.z;

            // Output tile coordinates (matching reference B200 matmul pattern):
            //   rowcol_x = a_tile_base + ctarank*2 + warpgroupid
            //     → CTA 0, consumer 0: tile rows [base+0] (128 rows)
            //     → CTA 0, consumer 1: tile rows [base+1] (128 rows)
            //     → CTA 1, consumer 0: tile rows [base+2] (128 rows)
            //     → CTA 1, consumer 1: tile rows [base+3] (128 rows)
            int padded_expert_start = sched->expert_offsets[expert];
            int ctarank = cluster_ctarank();
            int rowcol_x = (padded_expert_start / MOE_Mb) + m_tile * 4 + ctarank * 2 + warpgroupid;
            int rowcol_y = n_tile;

            // Wait for MMA results to be ready in tensor memory
            kittens::wait(outputs_arrived, task_iter % 2);

            // ═══════════════════════════════════════════════════════
            // tcgen05.ld — Load FP32 accumulator from tensor memory
            //
            // warpgroup::load_async internally emits:
            //   tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32
            //
            // This loads a 128×64 block of FP32 data from tensor
            // memory, packing it to BF16 via the pack::16b modifier.
            // 4 subtile loads cover the full 128×256 accumulator.
            // ═══════════════════════════════════════════════════════
            rt_bf<MOE_Mb / 4, d_tile::cols> d_reg[4];

            if (warpgroupid == 1) group<8>::sync(15);

            #pragma unroll
            for (int i = 0; i < MOE_Nb / d_tile::cols; i++) {
                warpgroup::load_async(d_reg[i],
                    d_tt.subtile<tt<float, 128, 64>>(0, 64 * i));
            }

            // ─── tcgen05.wait::ld — Wait for tensor loads to complete ───
            moe_tcgen05_wait_ld();
            warpgroup::sync(warpgroupid);

            // Signal tensor memory is free for MMA launcher to reuse
            if (warpgroup::laneid() == 0)
                tma::cluster::arrive(outputs_finished[warpgroupid], 0);

            if (warpgroupid == 0) group<8>::sync(15);
            if (warpgroupid == 1) group<8>::sync(14);

            // ── Store BF16 result to global memory via TMA ──
            warpgroup::store(d_smem, d_reg[0]);
            warpgroup::sync(warpgroupid);
            if (warpgroup::warpid() == 0)
                tma::store_async(g.d, d_smem, {rowcol_x, 4 * rowcol_y + 0});

            #pragma unroll
            for (int i = 1; i < MOE_Nb / d_tile::cols; i++) {
                tma::store_async_read_wait();
                warpgroup::sync(warpgroupid);
                warpgroup::store(d_smem, d_reg[i]);
                warpgroup::sync(warpgroupid);
                if (warpgroup::warpid() == 0)
                    tma::store_async(g.d, d_smem, {rowcol_x, 4 * rowcol_y + i});
            }
            tma::store_async_read_wait();

            if (warpgroupid == 0) group<8>::sync(14);
            group<8>::sync(15);
        }
    }

    everyone::tma::cluster::sync();
}

// ═══════════════════════════════════════════════════════════════════════════
// Host-Side Launch Helpers
// ═══════════════════════════════════════════════════════════════════════════

inline void launch_moe_gemm(
    __nv_bfloat16*       output,          // [padded_total, N_dim]
    const __nv_bfloat16* sorted_hidden,   // [padded_total, K_dim]
    const __nv_bfloat16* weights,         // [num_local_experts, N_dim, K_dim]
    const MoETaskScheduler* d_scheduler,
    int padded_total_tokens,
    int num_local_experts,
    int k_dim,
    int n_dim,
    cudaStream_t stream
) {
    using globals = moe_gemm_globals;

    typename globals::a_gl Ag{
        const_cast<bf16*>(reinterpret_cast<const bf16*>(sorted_hidden)),
        nullptr, nullptr,
        padded_total_tokens, k_dim
    };

    // 2D b_gl: flatten [num_experts, N_dim, K_dim] → [num_experts * N_dim, K_dim]
    // Memory layout is identical (row-major contiguous)
    typename globals::b_gl Bg{
        const_cast<bf16*>(reinterpret_cast<const bf16*>(weights)),
        nullptr, nullptr,
        (size_t)(num_local_experts * n_dim), (size_t)k_dim
    };

    typename globals::d_gl Dg{
        reinterpret_cast<bf16*>(output),
        nullptr, nullptr,
        padded_total_tokens, n_dim
    };

    // b_n_tile_stride = tiles per expert in N dimension
    // b_tile rows = MOE_Nb / 2 = 128, so stride = n_dim / 128
    int b_n_tile_stride = n_dim / (MOE_Nb / 2);
    globals G{Ag, Bg, Dg, b_n_tile_stride, d_scheduler};

    // 148 SMs on B200, 2 CTAs per cluster = 74 clusters
    dim3 grid(148, 1);
    dim3 block(MOE_NUM_THREADS);

    unsigned long smem_size = MAX_SHARED_MEMORY - 1024;

    // Required on B200: enable non-portable cluster sizes (cluster_dims > 1)
    auto cluster_err = cudaFuncSetAttribute(moe_gemm_kernel,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (cluster_err != cudaSuccess) {
        printf("MOE GEMM: NonPortableClusterSize failed: %s\n",
               cudaGetErrorString(cluster_err));
        return;
    }

    auto attr_err = cudaFuncSetAttribute(moe_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    if (attr_err != cudaSuccess) {
        printf("MOE GEMM: cudaFuncSetAttribute failed: %s (smem_size=%lu)\n",
               cudaGetErrorString(attr_err), smem_size);
        return;
    }

    moe_gemm_kernel<<<grid, block, smem_size, stream>>>(G);

    // Check for launch errors (does NOT sync, just checks immediate errors)
    auto launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("MOE GEMM: kernel launch failed: %s\n",
               cudaGetErrorString(launch_err));
    }
}
