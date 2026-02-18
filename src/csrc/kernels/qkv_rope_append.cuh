#pragma once

#include "qwen3_dims.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "../profiler/gpu_profiler.cuh"

// Profiler event IDs (paired: even=begin, odd=end)
namespace qkv_rope_events {
enum : int {
    EV_RMSNORM     = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC      = 2,
    EV_MATVEC_END  = 3,
    EV_QKNORM      = 4,
    EV_QKNORM_END  = 5,
    EV_ROPE        = 6,
    EV_ROPE_END    = 7,
    EV_CACHE       = 8,
    EV_CACHE_END   = 9,
};
} // namespace qkv_rope_events

/*
 * Fused kernel: RMSNorm → QKV MatVec → Q/K per-head RMSNorm → RoPE → KV Cache Append
 *
 * All operations run in a single block for BS=1 single-token decode.
 *
 * Shared memory layout (separate Q/K/V buffers):
 *   s_post_ln[HIDDEN_DIM]  = 2048 floats = 8 KB
 *   s_q[Q_DIM]             = 2048 floats = 8 KB
 *   s_k[K_DIM]             = 1024 floats = 4 KB
 *   s_v[V_DIM]             = 1024 floats = 4 KB
 *   s_reduce[WARP_SIZE]    = 32 floats   = 128 B
 *   prof                   = profiler state
 */
__device__ void qkv_rope_append_device(
    float*                      q_out,          // [Q_DIM]  output
    float*                      k_out,          // [K_DIM]  output
    float*                      v_out,          // [V_DIM]  output
    float*                      k_cache,        // [max_seq, NUM_KV_HEADS, HEAD_DIM]
    float*                      v_cache,        // [max_seq, NUM_KV_HEADS, HEAD_DIM]
    const float* __restrict__   hidden,         // [HIDDEN_DIM]
    const float* __restrict__   attn_ln_w,      // [HIDDEN_DIM]
    const float* __restrict__   qkv_weight,     // [QKV_DIM, HIDDEN_DIM] row-major
    const float* __restrict__   q_norm_w,       // [HEAD_DIM]
    const float* __restrict__   k_norm_w,       // [HEAD_DIM]
    const float* __restrict__   cos_cached,     // [HEAD_DIM] for this position
    const float* __restrict__   sin_cached,     // [HEAD_DIM] for this position
    int pos_id,
    int max_seq_len,
    profiler::event_record* g_events,
    int* g_counts
) {
    using namespace qkv_rope_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_post_ln = (float*)smem;
    float* s_q       = s_post_ln + HIDDEN_DIM;
    float* s_k       = s_q + Q_DIM;
    float* s_v       = s_k + K_DIM;
    float* s_reduce  = s_v + V_DIM;
    profiler::block_state* prof = (profiler::block_state*)(s_reduce + kernels::WARP_SIZE);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    // ===== Phase 1: RMSNorm on hidden_states =====
    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM);

    kernels::rmsnorm(s_post_ln, hidden, attn_ln_w, HIDDEN_DIM,
                     s_reduce, tid, num_threads, lane_id, warp_id, num_warps);
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM_END);

    // ===== Phase 2: Q/K/V MatVec with separate output buffers =====
    // Optimized with float4 vectorized loads + ILP (4 rows per iteration)
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    {
        constexpr int ILP = 4;
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);

        // Q matvec: q_weight[Q_DIM, HIDDEN_DIM] @ post_ln → s_q
        const float* q_weight = qkv_weight;
        for (int out_base = tid * ILP; out_base < Q_DIM; out_base += num_threads * ILP) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            const float4* row0 = reinterpret_cast<const float4*>(q_weight + (long long)(out_base + 0) * HIDDEN_DIM);
            const float4* row1 = reinterpret_cast<const float4*>(q_weight + (long long)(out_base + 1) * HIDDEN_DIM);
            const float4* row2 = reinterpret_cast<const float4*>(q_weight + (long long)(out_base + 2) * HIDDEN_DIM);
            const float4* row3 = reinterpret_cast<const float4*>(q_weight + (long long)(out_base + 3) * HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM / 4; j++) {
                float4 x = input4[j];
                float4 w0 = __ldcg(row0 + j); acc0 += w0.x * x.x + w0.y * x.y + w0.z * x.z + w0.w * x.w;
                float4 w1 = __ldcg(row1 + j); acc1 += w1.x * x.x + w1.y * x.y + w1.z * x.z + w1.w * x.w;
                float4 w2 = __ldcg(row2 + j); acc2 += w2.x * x.x + w2.y * x.y + w2.z * x.z + w2.w * x.w;
                float4 w3 = __ldcg(row3 + j); acc3 += w3.x * x.x + w3.y * x.y + w3.z * x.z + w3.w * x.w;
            }
            s_q[out_base + 0] = acc0;
            s_q[out_base + 1] = acc1;
            s_q[out_base + 2] = acc2;
            s_q[out_base + 3] = acc3;
        }

        // K matvec: k_weight[K_DIM, HIDDEN_DIM] @ post_ln → s_k
        const float* k_weight = qkv_weight + (long long)Q_DIM * HIDDEN_DIM;
        for (int out_base = tid * ILP; out_base < K_DIM; out_base += num_threads * ILP) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            const float4* row0 = reinterpret_cast<const float4*>(k_weight + (long long)(out_base + 0) * HIDDEN_DIM);
            const float4* row1 = reinterpret_cast<const float4*>(k_weight + (long long)(out_base + 1) * HIDDEN_DIM);
            const float4* row2 = reinterpret_cast<const float4*>(k_weight + (long long)(out_base + 2) * HIDDEN_DIM);
            const float4* row3 = reinterpret_cast<const float4*>(k_weight + (long long)(out_base + 3) * HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM / 4; j++) {
                float4 x = input4[j];
                float4 w0 = __ldcg(row0 + j); acc0 += w0.x * x.x + w0.y * x.y + w0.z * x.z + w0.w * x.w;
                float4 w1 = __ldcg(row1 + j); acc1 += w1.x * x.x + w1.y * x.y + w1.z * x.z + w1.w * x.w;
                float4 w2 = __ldcg(row2 + j); acc2 += w2.x * x.x + w2.y * x.y + w2.z * x.z + w2.w * x.w;
                float4 w3 = __ldcg(row3 + j); acc3 += w3.x * x.x + w3.y * x.y + w3.z * x.z + w3.w * x.w;
            }
            s_k[out_base + 0] = acc0;
            s_k[out_base + 1] = acc1;
            s_k[out_base + 2] = acc2;
            s_k[out_base + 3] = acc3;
        }

        // V matvec: v_weight[V_DIM, HIDDEN_DIM] @ post_ln → s_v
        const float* v_weight = qkv_weight + (long long)(Q_DIM + K_DIM) * HIDDEN_DIM;
        for (int out_base = tid * ILP; out_base < V_DIM; out_base += num_threads * ILP) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            const float4* row0 = reinterpret_cast<const float4*>(v_weight + (long long)(out_base + 0) * HIDDEN_DIM);
            const float4* row1 = reinterpret_cast<const float4*>(v_weight + (long long)(out_base + 1) * HIDDEN_DIM);
            const float4* row2 = reinterpret_cast<const float4*>(v_weight + (long long)(out_base + 2) * HIDDEN_DIM);
            const float4* row3 = reinterpret_cast<const float4*>(v_weight + (long long)(out_base + 3) * HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM / 4; j++) {
                float4 x = input4[j];
                float4 w0 = __ldcg(row0 + j); acc0 += w0.x * x.x + w0.y * x.y + w0.z * x.z + w0.w * x.w;
                float4 w1 = __ldcg(row1 + j); acc1 += w1.x * x.x + w1.y * x.y + w1.z * x.z + w1.w * x.w;
                float4 w2 = __ldcg(row2 + j); acc2 += w2.x * x.x + w2.y * x.y + w2.z * x.z + w2.w * x.w;
                float4 w3 = __ldcg(row3 + j); acc3 += w3.x * x.x + w3.y * x.y + w3.z * x.z + w3.w * x.w;
            }
            s_v[out_base + 0] = acc0;
            s_v[out_base + 1] = acc1;
            s_v[out_base + 2] = acc2;
            s_v[out_base + 3] = acc3;
        }
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    // ===== Phase 3: Q/K per-head RMSNorm =====
    if (tid == 0 && has_profiler) prof->record(EV_QKNORM);

    kernels::rmsnorm_per_head(s_q, q_norm_w, NUM_Q_HEADS, HEAD_DIM,
                              s_reduce, tid, num_threads, lane_id, warp_id, num_warps);

    kernels::rmsnorm_per_head(s_k, k_norm_w, NUM_KV_HEADS, HEAD_DIM,
                              s_reduce, tid, num_threads, lane_id, warp_id, num_warps);

    if (tid == 0 && has_profiler) prof->record(EV_QKNORM_END);

    // ===== Phase 4: RoPE on Q and K =====
    if (tid == 0 && has_profiler) prof->record(EV_ROPE);

    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int offset = h * HEAD_DIM;
        for (int i = tid; i < HALF_HEAD_DIM; i += num_threads) {
            float x_first  = s_q[offset + i];
            float x_second = s_q[offset + i + HALF_HEAD_DIM];
            float c = cos_cached[i];
            float s = sin_cached[i];
            s_q[offset + i]                 = x_first * c - x_second * s;
            s_q[offset + i + HALF_HEAD_DIM] = x_second * c + x_first * s;
        }
    }

    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int offset = h * HEAD_DIM;
        for (int i = tid; i < HALF_HEAD_DIM; i += num_threads) {
            float x_first  = s_k[offset + i];
            float x_second = s_k[offset + i + HALF_HEAD_DIM];
            float c = cos_cached[i];
            float s = sin_cached[i];
            s_k[offset + i]                 = x_first * c - x_second * s;
            s_k[offset + i + HALF_HEAD_DIM] = x_second * c + x_first * s;
        }
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_ROPE_END);

    // ===== Phase 5: Write outputs + KV cache append =====
    if (tid == 0 && has_profiler) prof->record(EV_CACHE);

    for (int i = tid; i < Q_DIM; i += num_threads) {
        q_out[i] = s_q[i];
    }

    for (int i = tid; i < K_DIM; i += num_threads) {
        float k_val = s_k[i];
        k_out[i] = k_val;
        k_cache[pos_id * K_DIM + i] = k_val;
    }

    for (int i = tid; i < V_DIM; i += num_threads) {
        float v_val = s_v[i];
        v_out[i] = v_val;
        v_cache[pos_id * V_DIM + i] = v_val;
    }

    if (tid == 0 && has_profiler) prof->record(EV_CACHE_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
