#pragma once

#include "qwen3_dims.cuh"
#include "utils.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace oproj_events {
enum : int {
    EV_MATVEC     = 0,
    EV_MATVEC_END = 1,
};
} // namespace oproj_events

/*
 * O-Projection + Residual Add
 *
 * proj = o_proj_weight @ attn_out   (matvec: [HIDDEN_DIM, HIDDEN_DIM] @ [HIDDEN_DIM])
 * hidden_states += proj             (residual)
 *
 * Shared memory layout:
 *   s_attn[HIDDEN_DIM]  = 2048 floats = 8 KB
 *   prof                = profiler state
 */
__device__ void oproj_residual_device(
    float*                      hidden_states,  // [HIDDEN_DIM] — in/out (residual added in-place)
    const float* __restrict__   attn_out,       // [HIDDEN_DIM]
    const float* __restrict__   o_proj_w,       // [HIDDEN_DIM, HIDDEN_DIM] row-major
    profiler::event_record* g_events,
    int* g_counts
) {
    using namespace oproj_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_attn = (float*)smem;
    profiler::block_state* prof = (profiler::block_state*)(s_attn + HIDDEN_DIM);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    // Load attn_out into shared memory
    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        s_attn[i] = attn_out[i];
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    // MatVec: proj = o_proj_w @ attn_out, then residual add
    for (int out_idx = tid; out_idx < HIDDEN_DIM; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = o_proj_w + (long long)out_idx * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            acc += row[j] * s_attn[j];
        }
        hidden_states[out_idx] += acc;
    }

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
