#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace oproj_events {
enum : int {
    EV_MATVEC = 0,
    EV_MATVEC_END = 1,
};
} // namespace oproj_events

/*
 * O-Projection + Residual Add
 *
 * proj = o_proj_weight @ attn_out   (matvec: [QWEN3_1_7B.hidden_size, QWEN3_1_7B.hidden_size] @
 * [QWEN3_1_7B.hidden_size]) hidden_states += proj             (residual)
 *
 * Shared memory layout:
 *   s_attn[QWEN3_1_7B.hidden_size]  = 2048 floats = 8 KB
 *   prof                = profiler state
 */
__device__ void
oproj_residual_device(float* hidden_states,               // [QWEN3_1_7B.hidden_size] — in/out (residual added in-place)
                      const float* __restrict__ attn_out, // [QWEN3_1_7B.hidden_size]
                      const float* __restrict__ o_proj_w, // [QWEN3_1_7B.hidden_size, QWEN3_1_7B.hidden_size] row-major
                      profiler::event_record* g_events, int* g_counts) {
    using namespace oproj_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_attn = reinterpret_cast<float*>(smem);
    profiler::block_state* prof = reinterpret_cast<profiler::block_state*>(s_attn + QWEN3_1_7B.hidden_size);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler)
        prof->init();
    __syncthreads();

    // Load attn_out into shared memory
    for (int i = tid; i < QWEN3_1_7B.hidden_size; i += num_threads) {
        s_attn[i] = attn_out[i];
    }
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC);

    // MatVec: proj = o_proj_w @ attn_out, then residual add
    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    {
        constexpr int ILP = 4;
        const float4* input4 = reinterpret_cast<const float4*>(s_attn);
        for (int out_base = tid * ILP; out_base < QWEN3_1_7B.hidden_size; out_base += num_threads * ILP) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            const float4* row0 =
                reinterpret_cast<const float4*>(o_proj_w + (long long)(out_base + 0) * QWEN3_1_7B.hidden_size);
            const float4* row1 =
                reinterpret_cast<const float4*>(o_proj_w + (long long)(out_base + 1) * QWEN3_1_7B.hidden_size);
            const float4* row2 =
                reinterpret_cast<const float4*>(o_proj_w + (long long)(out_base + 2) * QWEN3_1_7B.hidden_size);
            const float4* row3 =
                reinterpret_cast<const float4*>(o_proj_w + (long long)(out_base + 3) * QWEN3_1_7B.hidden_size);
            for (int j = 0; j < QWEN3_1_7B.hidden_size / 4; j++) {
                float4 x = input4[j];
                float4 w0 = __ldcg(row0 + j);
                acc0 += w0.x * x.x + w0.y * x.y + w0.z * x.z + w0.w * x.w;
                float4 w1 = __ldcg(row1 + j);
                acc1 += w1.x * x.x + w1.y * x.y + w1.z * x.z + w1.w * x.w;
                float4 w2 = __ldcg(row2 + j);
                acc2 += w2.x * x.x + w2.y * x.y + w2.z * x.z + w2.w * x.w;
                float4 w3 = __ldcg(row3 + j);
                acc3 += w3.x * x.x + w3.y * x.y + w3.z * x.z + w3.w * x.w;
            }
            hidden_states[out_base + 0] += acc0;
            hidden_states[out_base + 1] += acc1;
            hidden_states[out_base + 2] += acc2;
            hidden_states[out_base + 3] += acc3;
        }
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
