#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace upgate_events {
enum : int {
    EV_RMSNORM = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC = 2,
    EV_MATVEC_END = 3,
    EV_SILU = 4,
    EV_SILU_END = 5,
};
} // namespace upgate_events

/*
 * RMSNorm + Gate/Up double MatVec + SiLU
 *
 * post_ln = RMSNorm(hidden_states, mlp_ln_w)
 * gate = gate_w @ post_ln          [QWEN3_1_7B.intermediate_size]
 * up   = up_w   @ post_ln          [QWEN3_1_7B.intermediate_size]
 * silu_out = SiLU(gate) * up        [QWEN3_1_7B.intermediate_size]
 *
 * Shared memory layout:
 *   s_post_ln[QWEN3_1_7B.hidden_size]  = 2048 floats = 8 KB
 *   s_reduce[WARP_SIZE]    = 32 floats   = 128 B
 *   prof                   = profiler state
 */
__device__ void upgate_silu_device(
    float* silu_out,                          // [QWEN3_1_7B.intermediate_size] output
    const float* __restrict__ hidden,         // [QWEN3_1_7B.hidden_size]
    const float* __restrict__ mlp_ln_w,       // [QWEN3_1_7B.hidden_size]
    const __nv_bfloat16* __restrict__ gate_w, // [QWEN3_1_7B.intermediate_size, QWEN3_1_7B.hidden_size] bf16
    const __nv_bfloat16* __restrict__ up_w,   // [QWEN3_1_7B.intermediate_size, QWEN3_1_7B.hidden_size] bf16
    profiler::event_record* g_events, int* g_counts) {
    using namespace upgate_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + QWEN3_1_7B.hidden_size;
    profiler::block_state* prof = reinterpret_cast<profiler::block_state*>(s_reduce + kernels::WARP_SIZE);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    if (tid == 0 && has_profiler)
        prof->init();
    __syncthreads();

    // ===== Phase 1: RMSNorm =====
    if (tid == 0 && has_profiler)
        prof->record(EV_RMSNORM);

    kernels::rmsnorm(s_post_ln, hidden, mlp_ln_w, QWEN3_1_7B.hidden_size, s_reduce, tid, num_threads, lane_id, warp_id,
                     num_warps);
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_RMSNORM_END);

    // ===== Phase 2+3: Gate + Up MatVec, then fused SiLU * up =====
    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC);
    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC_END);
    if (tid == 0 && has_profiler)
        prof->record(EV_SILU);

    // Warp-reduce GEMV: each warp owns one output row for both gate and up.
    // With 128 blocks × 8 warps = 1024 total warps, all blocks stay active.
    {
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
        int lane = threadIdx.x & 31;
        int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int total_warps = (blockDim.x * gridDim.x) >> 5;
        for (int out_row = global_warp; out_row < QWEN3_1_7B.intermediate_size; out_row += total_warps) {
            const uint4* gr8 = reinterpret_cast<const uint4*>(gate_w + (long long)out_row * QWEN3_1_7B.hidden_size);
            const uint4* ur8 = reinterpret_cast<const uint4*>(up_w + (long long)out_row * QWEN3_1_7B.hidden_size);
            float gate_dot = 0.f, up_dot = 0.f;
            for (int k = lane; k < QWEN3_1_7B.hidden_size / 8; k += 32) {
                float4 x0 = input4[k * 2], x1 = input4[k * 2 + 1];
                uint4 graw = __ldcg(gr8 + k);
                float2 gf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
                float2 gf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
                float2 gf45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.z));
                float2 gf67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.w));
                gate_dot += gf01.x * x0.x + gf01.y * x0.y + gf23.x * x0.z + gf23.y * x0.w + gf45.x * x1.x +
                            gf45.y * x1.y + gf67.x * x1.z + gf67.y * x1.w;
                uint4 uraw = __ldcg(ur8 + k);
                float2 uf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.x));
                float2 uf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.y));
                float2 uf45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.z));
                float2 uf67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.w));
                up_dot += uf01.x * x0.x + uf01.y * x0.y + uf23.x * x0.z + uf23.y * x0.w + uf45.x * x1.x +
                          uf45.y * x1.y + uf67.x * x1.z + uf67.y * x1.w;
            }
            for (int s = 16; s > 0; s >>= 1) {
                gate_dot += __shfl_down_sync(0xffffffff, gate_dot, s);
                up_dot += __shfl_down_sync(0xffffffff, up_dot, s);
            }
            if (lane == 0)
                silu_out[out_row] = kernels::silu_multiply(gate_dot, up_dot);
        }
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_SILU_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
