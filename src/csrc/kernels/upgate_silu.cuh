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
    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);
    if (tid == 0 && has_profiler) prof->record(EV_SILU);

    {
        constexpr int ILP = 4;
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
        for (int out_base = tid * ILP; out_base < INTERMEDIATE_DIM; out_base += num_threads * ILP) {
            float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
            float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
            const float4* gr0 = reinterpret_cast<const float4*>(gate_w + (long long)(out_base + 0) * HIDDEN_DIM);
            const float4* gr1 = reinterpret_cast<const float4*>(gate_w + (long long)(out_base + 1) * HIDDEN_DIM);
            const float4* gr2 = reinterpret_cast<const float4*>(gate_w + (long long)(out_base + 2) * HIDDEN_DIM);
            const float4* gr3 = reinterpret_cast<const float4*>(gate_w + (long long)(out_base + 3) * HIDDEN_DIM);
            const float4* ur0 = reinterpret_cast<const float4*>(up_w + (long long)(out_base + 0) * HIDDEN_DIM);
            const float4* ur1 = reinterpret_cast<const float4*>(up_w + (long long)(out_base + 1) * HIDDEN_DIM);
            const float4* ur2 = reinterpret_cast<const float4*>(up_w + (long long)(out_base + 2) * HIDDEN_DIM);
            const float4* ur3 = reinterpret_cast<const float4*>(up_w + (long long)(out_base + 3) * HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM / 4; j++) {
                float4 x = input4[j];
                float4 gw0 = __ldcg(gr0 + j); gate_acc0 += gw0.x * x.x + gw0.y * x.y + gw0.z * x.z + gw0.w * x.w;
                float4 gw1 = __ldcg(gr1 + j); gate_acc1 += gw1.x * x.x + gw1.y * x.y + gw1.z * x.z + gw1.w * x.w;
                float4 gw2 = __ldcg(gr2 + j); gate_acc2 += gw2.x * x.x + gw2.y * x.y + gw2.z * x.z + gw2.w * x.w;
                float4 gw3 = __ldcg(gr3 + j); gate_acc3 += gw3.x * x.x + gw3.y * x.y + gw3.z * x.z + gw3.w * x.w;
                float4 uw0 = __ldcg(ur0 + j); up_acc0 += uw0.x * x.x + uw0.y * x.y + uw0.z * x.z + uw0.w * x.w;
                float4 uw1 = __ldcg(ur1 + j); up_acc1 += uw1.x * x.x + uw1.y * x.y + uw1.z * x.z + uw1.w * x.w;
                float4 uw2 = __ldcg(ur2 + j); up_acc2 += uw2.x * x.x + uw2.y * x.y + uw2.z * x.z + uw2.w * x.w;
                float4 uw3 = __ldcg(ur3 + j); up_acc3 += uw3.x * x.x + uw3.y * x.y + uw3.z * x.z + uw3.w * x.w;
            }
            // Fused SiLU(gate) * up using Rishu's scalar implementation
            silu_out[out_base + 0] = kernels::silu_multiply_scalar(gate_acc0, up_acc0);
            silu_out[out_base + 1] = kernels::silu_multiply_scalar(gate_acc1, up_acc1);
            silu_out[out_base + 2] = kernels::silu_multiply_scalar(gate_acc2, up_acc2);
            silu_out[out_base + 3] = kernels::silu_multiply_scalar(gate_acc3, up_acc3);
        }
    }

    if (tid == 0 && has_profiler) prof->record(EV_SILU_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
