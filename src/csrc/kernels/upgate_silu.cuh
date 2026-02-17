#pragma once

#include "qwen3_dims.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace upgate_events {
enum : int {
    EV_RMSNORM     = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC      = 2,
    EV_MATVEC_END  = 3,
    EV_SILU        = 4,
    EV_SILU_END    = 5,
};
} // namespace upgate_events

/*
 * RMSNorm + Gate/Up double MatVec + SiLU
 *
 * post_ln = RMSNorm(hidden_states, mlp_ln_w)
 * gate = gate_w @ post_ln          [INTERMEDIATE_DIM]
 * up   = up_w   @ post_ln          [INTERMEDIATE_DIM]
 * silu_out = SiLU(gate) * up        [INTERMEDIATE_DIM]
 *
 * Shared memory layout:
 *   s_post_ln[HIDDEN_DIM]  = 2048 floats = 8 KB
 *   s_reduce[WARP_SIZE]    = 32 floats   = 128 B
 *   prof                   = profiler state
 */
__device__ void upgate_silu_device(
    float*                      silu_out,       // [INTERMEDIATE_DIM] output
    const float* __restrict__   hidden,         // [HIDDEN_DIM]
    const float* __restrict__   mlp_ln_w,       // [HIDDEN_DIM]
    const float* __restrict__   gate_w,         // [INTERMEDIATE_DIM, HIDDEN_DIM]
    const float* __restrict__   up_w,           // [INTERMEDIATE_DIM, HIDDEN_DIM]
    profiler::event_record* g_events,
    int* g_counts
) {
    using namespace upgate_events;
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

    kernels::rmsnorm(s_post_ln, hidden, mlp_ln_w, HIDDEN_DIM,
                     s_reduce, tid, num_threads, lane_id, warp_id, num_warps);
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM_END);

    // ===== Phase 2+3: Gate + Up MatVec, then fused SiLU * up =====
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);
    if (tid == 0 && has_profiler) prof->record(EV_SILU);

    for (int out_idx = tid; out_idx < INTERMEDIATE_DIM; out_idx += num_threads) {
        const float* gate_row = gate_w + (long long)out_idx * HIDDEN_DIM;
        const float* up_row   = up_w   + (long long)out_idx * HIDDEN_DIM;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        for (int j = 0; j < HIDDEN_DIM; j++) {
            float post_ln_j = s_post_ln[j];
            gate_acc += gate_row[j] * post_ln_j;
            up_acc   += up_row[j]   * post_ln_j;
        }

        // Fused SiLU(gate) * up using Rishu's scalar implementation
        silu_out[out_idx] = kernels::silu_multiply_scalar(gate_acc, up_acc);
    }

    if (tid == 0 && has_profiler) prof->record(EV_SILU_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
