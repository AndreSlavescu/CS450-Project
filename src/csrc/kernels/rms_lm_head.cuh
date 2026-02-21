#pragma once

#include "qwen3_dims.cuh"
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
 * logits = lm_head_w @ post_ln     (matvec: [vocab_size, HIDDEN_DIM] @ [HIDDEN_DIM])
 *
 * Note: For Qwen3-1.7B, lm_head_w = embed_tokens.weight (tied embeddings).
 *
 * Shared memory layout:
 *   s_post_ln[HIDDEN_DIM]  = 2048 floats = 8 KB
 *   s_reduce[WARP_SIZE]    = 32 floats   = 128 B
 *   prof                   = profiler state
 */
__device__ void rms_lm_head_device(float* logits,                       // [vocab_size] output
                                   const float* __restrict__ hidden,    // [HIDDEN_DIM]
                                   const float* __restrict__ norm_w,    // [HIDDEN_DIM]
                                   const float* __restrict__ lm_head_w, // [vocab_size, HIDDEN_DIM]
                                   int vocab_size, profiler::event_record* g_events, int* g_counts) {
    using namespace lmhead_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + HIDDEN_DIM;
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

    kernels::rmsnorm(s_post_ln, hidden, norm_w, HIDDEN_DIM, s_reduce, tid, num_threads, lane_id, warp_id, num_warps);
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_RMSNORM_END);

    // ===== Phase 2: LM Head MatVec =====
    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC);

    {
        constexpr int ILP = 4;
        int aligned_vocab = vocab_size & ~(ILP - 1);
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
        for (int out_base = tid * ILP; out_base < aligned_vocab; out_base += num_threads * ILP) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            const float4* row0 = reinterpret_cast<const float4*>(lm_head_w + (long long)(out_base + 0) * HIDDEN_DIM);
            const float4* row1 = reinterpret_cast<const float4*>(lm_head_w + (long long)(out_base + 1) * HIDDEN_DIM);
            const float4* row2 = reinterpret_cast<const float4*>(lm_head_w + (long long)(out_base + 2) * HIDDEN_DIM);
            const float4* row3 = reinterpret_cast<const float4*>(lm_head_w + (long long)(out_base + 3) * HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM / 4; j++) {
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
            logits[out_base + 0] = acc0;
            logits[out_base + 1] = acc1;
            logits[out_base + 2] = acc2;
            logits[out_base + 3] = acc3;
        }
        // Tail: handle remaining rows if vocab_size not divisible by 4
        for (int out_idx = aligned_vocab + tid; out_idx < vocab_size; out_idx += num_threads) {
            float acc = 0.0f;
            const float4* row = reinterpret_cast<const float4*>(lm_head_w + (long long)out_idx * HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM / 4; j++) {
                float4 x = input4[j];
                float4 w = __ldcg(row + j);
                acc += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
            }
            logits[out_idx] = acc;
        }
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
