#pragma once

#include "qwen3_dims.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace lmhead_events {
enum : int {
    EV_RMSNORM     = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC      = 2,
    EV_MATVEC_END  = 3,
};
} // namespace lmhead_events

/*
 * Final RMSNorm + LM Head projection
 *
 * post_ln = RMSNorm(hidden_states, norm_w)
 * logits = lm_head_w @ post_ln     (matvec: [vocab_size, HIDDEN_DIM] @ [HIDDEN_DIM])
 *
 * Note: For Qwen3-1.7B, lm_head_w = embed_tokens.weight (tied embeddings).
 *
 * Shared memory layout:
 *   s_post_ln[HIDDEN_DIM]  = 2048 floats = 8 KB
 *   s_reduce[WARP_SIZE]    = 32 floats   = 128 B
 *   prof                   = profiler state
 */
__device__ void rms_lm_head_device(
    float*                      logits,         // [vocab_size] output
    const float* __restrict__   hidden,         // [HIDDEN_DIM]
    const float* __restrict__   norm_w,         // [HIDDEN_DIM]
    const float* __restrict__   lm_head_w,      // [vocab_size, HIDDEN_DIM]
    int vocab_size,
    profiler::event_record* g_events,
    int* g_counts
) {
    using namespace lmhead_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_post_ln = (float*)smem;
    float* s_reduce  = s_post_ln + HIDDEN_DIM;
    profiler::block_state* prof = (profiler::block_state*)(s_reduce + kernels::WARP_SIZE);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    // ===== Phase 1: RMSNorm =====
    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM);

    kernels::rmsnorm(s_post_ln, hidden, norm_w, HIDDEN_DIM,
                     s_reduce, tid, num_threads, lane_id, warp_id, num_warps);
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM_END);

    // ===== Phase 2: LM Head MatVec =====
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    for (int out_idx = tid; out_idx < vocab_size; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = lm_head_w + (long long)out_idx * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            acc += row[j] * s_post_ln[j];
        }
        logits[out_idx] = acc;
    }

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
