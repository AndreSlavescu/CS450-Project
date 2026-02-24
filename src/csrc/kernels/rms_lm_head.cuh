#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs
namespace lmhead_events {
enum : int {
    EV_RMSNORM = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC = 2,
    EV_MATVEC_END = 3,
};
} // namespace lmhead_events

/*
 * Final RMSNorm + LM Head projection
 *
 * post_ln = RMSNorm(hidden_states, norm_w)
 * logits = lm_head_w @ post_ln     (matvec: [vocab_size, QWEN3_1_7B.hidden_size] @ [QWEN3_1_7B.hidden_size])
 *
 * Note: For Qwen3-1.7B, lm_head_w = embed_tokens.weight (tied embeddings).
 *
 * Shared memory layout:
 *   s_post_ln[QWEN3_1_7B.hidden_size]  = 2048 floats = 8 KB
 *   s_reduce[WARP_SIZE]    = 32 floats   = 128 B
 *   prof                   = profiler state
 */
__device__ void
rms_lm_head_device(float* logits,                               // [vocab_size] output
                   const float* __restrict__ hidden,            // [QWEN3_1_7B.hidden_size]
                   const float* __restrict__ norm_w,            // [QWEN3_1_7B.hidden_size]
                   const __nv_bfloat16* __restrict__ lm_head_w, // [vocab_size, QWEN3_1_7B.hidden_size] bf16
                   int vocab_size, profiler::event_record* g_events, int* g_counts) {
    using namespace lmhead_events;
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

    kernels::rmsnorm(s_post_ln, hidden, norm_w, QWEN3_1_7B.hidden_size, s_reduce, tid, num_threads, lane_id, warp_id,
                     num_warps);
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_RMSNORM_END);

    // ===== Phase 2: LM Head MatVec =====
    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC);

    // Warp-reduce GEMV: each warp owns one output row via lane reduction.
    // With 128 blocks × 8 warps = 1024 total warps, all 128 blocks stay active.
    {
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
        int lane = threadIdx.x & 31;
        int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int total_warps = (blockDim.x * gridDim.x) >> 5;
        for (int out_row = global_warp; out_row < vocab_size; out_row += total_warps) {
            const uint4* row8 = reinterpret_cast<const uint4*>(lm_head_w + (long long)out_row * QWEN3_1_7B.hidden_size);
            float dot = 0.f;
            for (int k = lane; k < QWEN3_1_7B.hidden_size / 8; k += 32) {
                uint4 raw = __ldcg(row8 + k);
                float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
                float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));
                float4 x0 = input4[k * 2], x1 = input4[k * 2 + 1];
                dot += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                       f67.x * x1.z + f67.y * x1.w;
            }
            for (int s = 16; s > 0; s >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, s);
            if (lane == 0)
                logits[out_row] = dot;
        }
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
