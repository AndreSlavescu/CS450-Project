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

    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    {
        constexpr int ILP = 4;
        const float4* input4 = reinterpret_cast<const float4*>(silu_out);
        for (int out_base = tid * ILP; out_base < HIDDEN_DIM; out_base += num_threads * ILP) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            const float4* row0 = reinterpret_cast<const float4*>(down_proj_w + (long long)(out_base + 0) * INTERMEDIATE_DIM);
            const float4* row1 = reinterpret_cast<const float4*>(down_proj_w + (long long)(out_base + 1) * INTERMEDIATE_DIM);
            const float4* row2 = reinterpret_cast<const float4*>(down_proj_w + (long long)(out_base + 2) * INTERMEDIATE_DIM);
            const float4* row3 = reinterpret_cast<const float4*>(down_proj_w + (long long)(out_base + 3) * INTERMEDIATE_DIM);
            for (int j = 0; j < INTERMEDIATE_DIM / 4; j++) {
                float4 x = input4[j];
                float4 w0 = row0[j]; acc0 += w0.x * x.x + w0.y * x.y + w0.z * x.z + w0.w * x.w;
                float4 w1 = row1[j]; acc1 += w1.x * x.x + w1.y * x.y + w1.z * x.z + w1.w * x.w;
                float4 w2 = row2[j]; acc2 += w2.x * x.x + w2.y * x.y + w2.z * x.z + w2.w * x.w;
                float4 w3 = row3[j]; acc3 += w3.x * x.x + w3.y * x.y + w3.z * x.z + w3.w * x.w;
            }
            hidden_states[out_base + 0] += acc0;
            hidden_states[out_base + 1] += acc1;
            hidden_states[out_base + 2] += acc2;
            hidden_states[out_base + 3] += acc3;
        }
    }

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
