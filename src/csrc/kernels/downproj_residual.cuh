#pragma once

#include "qwen3_dims.cuh"
#include "utils.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace downproj_events {
enum : int {
    EV_MATVEC     = 0,
    EV_MATVEC_END = 1,
};
} // namespace downproj_events

/*
 * Down Projection + Residual Add
 *
 * proj = down_proj_w @ silu_out   (matvec: [HIDDEN_DIM, INTERMEDIATE_DIM] @ [INTERMEDIATE_DIM])
 * hidden_states += proj           (residual)
 *
 * Shared memory layout:
 *   prof = profiler state only
 */
__device__ void downproj_residual_device(
    float*                      hidden_states,  // [HIDDEN_DIM] — in/out
    const float* __restrict__   silu_out,       // [INTERMEDIATE_DIM]
    const float* __restrict__   down_proj_w,    // [HIDDEN_DIM, INTERMEDIATE_DIM]
    profiler::event_record* g_events,
    int* g_counts
) {
    using namespace downproj_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    profiler::block_state* prof = (profiler::block_state*)smem;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    for (int out_idx = tid; out_idx < HIDDEN_DIM; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = down_proj_w + (long long)out_idx * INTERMEDIATE_DIM;
        for (int j = 0; j < INTERMEDIATE_DIM; j++) {
            acc += row[j] * silu_out[j];
        }
        hidden_states[out_idx] += acc;
    }

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
