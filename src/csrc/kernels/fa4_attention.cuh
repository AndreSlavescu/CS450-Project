#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// FA4 Forward Attention Kernel — Raw CUDA
//
// FlashAttention-style forward-only kernel for Qwen3 inference.
// Designed for BS=1 low-latency decode and multi-token prefill.
//
// Features:
//   - Online softmax across KV tiles (numerically stable, single pass)
//   - GQA (grouped query attention): Q heads share KV heads
//   - Causal masking with block-level skip optimization
//   - BF16 input/output, FP32 internal accumulation
//   - Optional LSE output for ring attention partial accumulation
//
// Thread mapping:
//   Each thread owns one Q row and maintains its own O accumulator
//   and online softmax state (running_max, running_sum).
//   K/V tiles are cooperatively loaded into shared memory.
//   Q is read from global memory (const __restrict__, L1 cached).
//
// Memory layout (all row-major):
//   Q: [num_q_heads, seq_q, head_dim]
//   K: [num_kv_heads, seq_kv, head_dim]
//   V: [num_kv_heads, seq_kv, head_dim]
//   O: [num_q_heads, seq_q, head_dim]
//
// Register budget per thread (HEAD_DIM=128, BLOCK_KV=64):
//   o_acc[128]   = 128 regs (FP32 output accumulator)
//   scores[64]   =  64 regs (attention scores, reused as weights)
//   running_max  =   1 reg
//   running_sum  =   1 reg
//   misc         = ~16 regs
//   Total        = ~210 regs (under 255 limit)
//
// Shared memory (HEAD_DIM=128, BLOCK_KV=64):
//   sK[64][128]  =  16 KB (K tile)
//   sV[64][128]  =  16 KB (V tile)
//   Total        =  32 KB
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Forward attention kernel
//
// Template params:
//   HEAD_DIM:  head dimension (128 for Qwen3)
//   BLOCK_KV:  number of KV rows per shared memory tile
//   BLOCK_Q:   number of threads = number of Q rows per block
//
// Grid: (ceil(seq_q / BLOCK_Q), num_q_heads, 1)
// Block: (BLOCK_Q)
// ---------------------------------------------------------------------------

// Cache-hint constants borrowed from optimized SM100 kernels.
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;

// Elect one lane from the current warp.
__device__ inline uint32_t elect_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t pred = 0;
    asm volatile("{\n\t"
                 ".reg .pred %%px;\n\t"
                 "elect.sync _|%%px, %1;\n\t"
                 "@%%px mov.s32 %0, 1;\n\t"
                 "}"
                 : "+r"(pred)
                 : "r"(0xFFFFFFFF));
    return pred;
#else
    return (threadIdx.x == 0) ? 1U : 0U;
#endif
}

__device__ inline void prefetch_l2_bulk(const void* src, int bytes, uint64_t cache_policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(src), "r"(bytes),
                 "l"(cache_policy)
                 : "memory");
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // Fallback on older architectures: prefetch one cacheline.
    asm volatile("prefetch.global.L2 [%0];" ::"l"(src) : "memory");
    (void)bytes;
    (void)cache_policy;
#else
    (void)src;
    (void)bytes;
    (void)cache_policy;
#endif
}

template <int HEAD_DIM = 128, int BLOCK_KV = 64, int BLOCK_Q = 64, bool USE_BULK_PREFETCH = false>
__global__ void __launch_bounds__(BLOCK_Q)
    fa4_forward_kernel(__nv_bfloat16* __restrict__ O_out,   // [num_q_heads, seq_q, HEAD_DIM]
                       float* __restrict__ lse_out,         // [num_q_heads, seq_q] or nullptr
                       const __nv_bfloat16* __restrict__ Q, // [num_q_heads, seq_q, HEAD_DIM]
                       const __nv_bfloat16* __restrict__ K, // [num_kv_heads, seq_kv, HEAD_DIM]
                       const __nv_bfloat16* __restrict__ V, // [num_kv_heads, seq_kv, HEAD_DIM]
                       const int seq_q, const int seq_kv, const float scale, const int gqa_ratio, const bool causal,
                       const int q_offset, // global Q position offset (for ring attention)
                       const int kv_offset // global KV position offset (for ring attention)
    ) {
    static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even");

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int q_tile = blockIdx.x;
    const int q_head = blockIdx.y;
    const int kv_head = q_head / gqa_ratio;

    // This thread's Q row index
    const int q_idx = q_tile * BLOCK_Q + tid;

    // Per-head base pointers
    const long long q_head_off = (long long)q_head * seq_q * HEAD_DIM;
    const long long kv_head_off = (long long)kv_head * seq_kv * HEAD_DIM;

    const __nv_bfloat16* Q_head = Q + q_head_off;
    const __nv_bfloat16* K_head = K + kv_head_off;
    const __nv_bfloat16* V_head = V + kv_head_off;
    __nv_bfloat16* O_head = O_out + q_head_off;

    // Shared memory for K and V tiles
    __shared__ __nv_bfloat16 sK[BLOCK_KV * HEAD_DIM];
    __shared__ __nv_bfloat16 sV[BLOCK_KV * HEAD_DIM];

    // Per-thread output accumulator and online softmax state
    float o_acc[HEAD_DIM];
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        o_acc[d] = 0.0f;
    }

    // Block-level causal bound: the last Q row in this block determines
    // the furthest KV position any thread in this block could need.
    // All threads must participate in shared memory loading, so we use
    // the block-level bound (not per-thread) for the KV loop.
    int kv_end = seq_kv;
    if (causal) {
        int block_last_q = min(q_tile * BLOCK_Q + BLOCK_Q - 1, seq_q - 1);
        int last_q_global = block_last_q + q_offset;
        kv_end = min(seq_kv, last_q_global - kv_offset + 1);
    }

    // Per-thread global Q position (for causal masking in compute phase)
    const int q_global = q_idx + q_offset;

    // Pointer to this thread's Q row in global memory (L1 cached)
    const __nv_bfloat16* q_row = (q_idx < seq_q) ? (Q_head + (long long)q_idx * HEAD_DIM) : nullptr;

    // Stream KV tiles through shared memory
    for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_KV) {
        const int tile_rows = min(BLOCK_KV, seq_kv - kv_start);
        const int next_kv_start = kv_start + BLOCK_KV;
        const int next_rows = (next_kv_start < kv_end) ? min(BLOCK_KV, kv_end - next_kv_start) : 0;

        // One elected lane prefetches the next tile into L2 to reduce
        // long-scoreboard stalls in the following iteration.
        if constexpr (USE_BULK_PREFETCH) {
            if (warp_id == 0 && elect_sync() && next_kv_start < kv_end) {
                const int next_bytes = next_rows * HEAD_DIM * static_cast<int>(sizeof(__nv_bfloat16));
                const __nv_bfloat16* next_k = K_head + static_cast<long long>(next_kv_start) * HEAD_DIM;
                const __nv_bfloat16* next_v = V_head + static_cast<long long>(next_kv_start) * HEAD_DIM;
                prefetch_l2_bulk(next_k, next_bytes, EVICT_FIRST);
                prefetch_l2_bulk(next_v, next_bytes, EVICT_LAST);
            }
        }

        // --- Cooperative load: K and V tiles into shared memory ---
        // All threads participate regardless of whether they have a valid Q row.
        // Total elements per buffer: BLOCK_KV * HEAD_DIM
        // Each thread loads ceil(BLOCK_KV * HEAD_DIM / BLOCK_Q) elements.
        for (int i = tid; i < BLOCK_KV * HEAD_DIM; i += BLOCK_Q) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (row < tile_rows) {
                int global_row = kv_start + row;
                sK[i] = K_head[(long long)global_row * HEAD_DIM + col];
                sV[i] = V_head[(long long)global_row * HEAD_DIM + col];
            } else {
                // Zero-pad partial tiles
                sK[i] = __float2bfloat16(0.0f);
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // --- Compute: only threads with valid Q rows ---
        if (q_idx < seq_q) {
            // Phase 1: Compute attention scores S[j] = Q[q_idx] · K[kv_start+j] * scale
            float scores[BLOCK_KV];
            float local_max = -FLT_MAX;

#pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                // Out-of-bounds KV row
                if (j >= tile_rows) {
                    scores[j] = -FLT_MAX;
                    continue;
                }

                // Causal mask: skip if KV position > Q position
                if (causal) {
                    int kv_global = kv_start + j + kv_offset;
                    if (kv_global > q_global) {
                        scores[j] = -FLT_MAX;
                        continue;
                    }
                }

                // Dot product Q[q_idx] · K[kv_start + j]
                float dot = 0.0f;
                const __nv_bfloat16* k_row = sK + j * HEAD_DIM;

#pragma unroll 8
                for (int d = 0; d < HEAD_DIM; d += 2) {
                    float q0 = __bfloat162float(q_row[d]);
                    float q1 = __bfloat162float(q_row[d + 1]);
                    float k0 = __bfloat162float(k_row[d]);
                    float k1 = __bfloat162float(k_row[d + 1]);
                    dot = fmaf(q0, k0, dot);
                    dot = fmaf(q1, k1, dot);
                }

                scores[j] = dot * scale;
                local_max = fmaxf(local_max, scores[j]);
            }

            // Phase 2: Online softmax update
            // Rescale previous accumulator when max increases
            float new_max = fmaxf(running_max, local_max);

            if (running_max > -FLT_MAX) {
                float rescale = __expf(running_max - new_max);
                running_sum *= rescale;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    o_acc[d] *= rescale;
                }
            }
            running_max = new_max;

// Convert scores to weights in-place: w[j] = exp(score[j] - max)
// Also accumulate running_sum
#pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                if (scores[j] > -FLT_MAX + 1.0f) {
                    scores[j] = __expf(scores[j] - running_max);
                    running_sum += scores[j];
                } else {
                    scores[j] = 0.0f;
                }
            }

// Phase 3: Accumulate O += weights @ V
#pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                if (scores[j] > 0.0f) {
                    const __nv_bfloat16* v_row = sV + j * HEAD_DIM;
                    float w = scores[j];
#pragma unroll 8
                    for (int d = 0; d < HEAD_DIM; d += 2) {
                        o_acc[d] = fmaf(w, __bfloat162float(v_row[d]), o_acc[d]);
                        o_acc[d + 1] = fmaf(w, __bfloat162float(v_row[d + 1]), o_acc[d + 1]);
                    }
                }
            }
        }

        __syncthreads();
    }

    // --- Write output ---
    if (q_idx < seq_q) {
        float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

        __nv_bfloat16* o_row = O_head + (long long)q_idx * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            o_row[d] = __float2bfloat16(o_acc[d] * inv_sum);
        }

        // Output log-sum-exp for ring attention accumulation
        if (lse_out != nullptr) {
            lse_out[(long long)q_head * seq_q + q_idx] = running_max + __logf(fmaxf(running_sum, 1e-10f));
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
//
// Instantiates the kernel with HEAD_DIM=128 (Qwen3) and launches it.
// Handles GQA ratio computation from head counts.
// ---------------------------------------------------------------------------

inline void fa4_forward(__nv_bfloat16* O,
                        float* lse, // nullable
                        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, int num_q_heads,
                        int num_kv_heads, int seq_q, int seq_kv, float scale, bool causal, int q_offset, int kv_offset,
                        cudaStream_t stream) {
    constexpr int HD = 128;
    constexpr int BKV = 64;
    constexpr int BQ_FALLBACK = 64;
    constexpr int BQ_SM100 = 128;

    int gqa_ratio = num_q_heads / num_kv_heads;
    static int cached_sm = -1;
    if (cached_sm < 0) {
        int device = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            cached_sm = prop.major * 10 + prop.minor;
        } else {
            cached_sm = 0;
        }
    }
    const bool is_sm100a = (cached_sm >= 100);

    if (is_sm100a && seq_q >= BQ_SM100) {
        auto kernel = fa4_forward_kernel<HD, BKV, BQ_SM100, true>;
        // Bias carveout toward SMEM on Blackwell; helps sustain tile residency.
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        dim3 grid((seq_q + BQ_SM100 - 1) / BQ_SM100, num_q_heads, 1);
        dim3 block(BQ_SM100);
        kernel<<<grid, block, 0, stream>>>(O, lse, Q, K, V, seq_q, seq_kv, scale, gqa_ratio, causal, q_offset,
                                           kv_offset);
    } else {
        dim3 grid((seq_q + BQ_FALLBACK - 1) / BQ_FALLBACK, num_q_heads, 1);
        dim3 block(BQ_FALLBACK);
        fa4_forward_kernel<HD, BKV, BQ_FALLBACK, false>
            <<<grid, block, 0, stream>>>(O, lse, Q, K, V, seq_q, seq_kv, scale, gqa_ratio, causal, q_offset, kv_offset);
    }
}
