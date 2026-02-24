#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "../profiler/gpu_profiler.cuh"

/*
 * Phase 1a — grid-parallel QKV MatVec.
 *
 * Called with ALL blocks (grid_size = 128).
 *
 * Each block:
 *   1. Computes RMSNorm(hidden, attn_ln_w) redundantly into local s_post_ln.
 *   2. Runs Q/K/V matrix-vector products partitioned by global thread id:
 *        global_tid = blockIdx.x * blockDim.x + threadIdx.x
 *      Each global thread owns a stride of ILP=4 output rows, so the weight
 *      rows are evenly spread across all 128 * 256 = 32 768 threads.
 *
 * Writes to global g_q[q_dim], g_k[kv_dim], g_v[kv_dim].
 * These are coherent after a grid_sync.sync() by the caller.
 *
 * Shared memory layout (within the kernel's fixed allocation):
 *   s_post_ln[hidden_size]  = 2048 × 4 = 8 192 B
 *   s_reduce[WARP_SIZE]     =   32 × 4 =   128 B
 */
__device__ void
qkv_matvec_device(float* __restrict__ g_q,                        // [q_dim]  global output
                  float* __restrict__ g_k,                        // [kv_dim] global output
                  float* __restrict__ g_v,                        // [kv_dim] global output
                  const float* __restrict__ hidden,               // [hidden_size]
                  const float* __restrict__ attn_ln_w,            // [hidden_size]
                  const __nv_bfloat16* __restrict__ qkv_weight) { // [qkv_dim, hidden_size] row-major, bf16
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;

    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + hidden_size;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    // ===== Phase 1: RMSNorm (all blocks, redundant, uses local smem) =====
    kernels::rmsnorm(s_post_ln, hidden, attn_ln_w, hidden_size, s_reduce, tid, num_threads, lane_id, warp_id,
                     num_warps);
    __syncthreads();

    // ===== Phase 2: Q/K/V warp-reduce GEMV (BF16 weights) =====
    // Each warp owns one output row; uint4 loads = 8 bf16 per load, halving HBM traffic.
    // With 128 blocks × 8 warps = 1024 total warps, even Q (2048 rows) gives
    // every warp 2 rows of real work — all 128 SMs hit HBM simultaneously.
    {
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
        int lane = threadIdx.x & 31;
        int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int total_warps = (blockDim.x * gridDim.x) >> 5; // 1024

        // --- Q warp-reduce GEMV ---
        const __nv_bfloat16* q_weight = qkv_weight;
        for (int out_row = global_warp; out_row < q_dim; out_row += total_warps) {
            const uint4* row8 = reinterpret_cast<const uint4*>(q_weight + (long long)out_row * hidden_size);
            float dot = 0.f;
            for (int k = lane; k < hidden_size / 8; k += 32) {
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
                g_q[out_row] = dot;
        }

        // --- K warp-reduce GEMV ---
        const __nv_bfloat16* k_weight = qkv_weight + (long long)q_dim * hidden_size;
        for (int out_row = global_warp; out_row < kv_dim; out_row += total_warps) {
            const uint4* row8 = reinterpret_cast<const uint4*>(k_weight + (long long)out_row * hidden_size);
            float dot = 0.f;
            for (int k = lane; k < hidden_size / 8; k += 32) {
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
                g_k[out_row] = dot;
        }

        // --- V warp-reduce GEMV ---
        const __nv_bfloat16* v_weight = qkv_weight + (long long)(q_dim + kv_dim) * hidden_size;
        for (int out_row = global_warp; out_row < kv_dim; out_row += total_warps) {
            const uint4* row8 = reinterpret_cast<const uint4*>(v_weight + (long long)out_row * hidden_size);
            float dot = 0.f;
            for (int k = lane; k < hidden_size / 8; k += 32) {
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
                g_v[out_row] = dot;
        }
    }
    // No __syncthreads() needed: caller does grid_sync.sync() before reading outputs.
}

/*
 * Phase 1b — Q/K per-head RMSNorm, RoPE, and KV-cache append.
 *
 * MUST be called only from block 0 (guarded at the call site).
 * All 256 threads of block 0 participate.
 *
 * After a grid_sync.sync(), g_q and g_k are fully written by qkv_matvec_device.
 * This function applies in-place modifications and writes the KV cache.
 *
 * Shared memory layout:
 *   s_reduce[WARP_SIZE] = 32 × 4 = 128 B  (for per-head block_reduce_sum)
 */
__device__ void
qknorm_rope_kvcache_device(float* __restrict__ g_q,              // [q_dim]  in/out: norm + RoPE applied in-place
                           float* __restrict__ g_k,              // [kv_dim] in/out: norm + RoPE applied in-place
                           const float* __restrict__ g_v,        // [kv_dim] read-only
                           float* __restrict__ k_cache,          // [max_seq * kv_dim]
                           float* __restrict__ v_cache,          // [max_seq * kv_dim]
                           const float* __restrict__ q_norm_w,   // [head_dim]
                           const float* __restrict__ k_norm_w,   // [head_dim]
                           const float* __restrict__ cos_cached, // [head_dim] for this position (first half filled)
                           const float* __restrict__ sin_cached, // [head_dim] for this position
                           int pos_id) {
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int num_q_heads = QWEN3_1_7B.num_attention_heads;
    constexpr int num_kv_heads = QWEN3_1_7B.num_key_value_heads;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int half_head = head_dim / 2;

    // s_reduce sits at the start of the block's shared memory (128 B).
    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    // ===== Q per-head RMSNorm (in-place on global g_q) =====
    kernels::rmsnorm_per_head(g_q, q_norm_w, num_q_heads, head_dim, s_reduce, tid, num_threads, lane_id, warp_id,
                              num_warps);

    // ===== K per-head RMSNorm (in-place on global g_k) =====
    kernels::rmsnorm_per_head(g_k, k_norm_w, num_kv_heads, head_dim, s_reduce, tid, num_threads, lane_id, warp_id,
                              num_warps);

    // ===== Q RoPE (in-place) =====
    for (int h = 0; h < num_q_heads; h++) {
        int off = h * head_dim;
        for (int i = tid; i < half_head; i += num_threads) {
            float x0 = g_q[off + i], x1 = g_q[off + i + half_head];
            float c = cos_cached[i], s = sin_cached[i];
            g_q[off + i] = x0 * c - x1 * s;
            g_q[off + i + half_head] = x1 * c + x0 * s;
        }
    }

    // ===== K RoPE (in-place) =====
    for (int h = 0; h < num_kv_heads; h++) {
        int off = h * head_dim;
        for (int i = tid; i < half_head; i += num_threads) {
            float x0 = g_k[off + i], x1 = g_k[off + i + half_head];
            float c = cos_cached[i], s = sin_cached[i];
            g_k[off + i] = x0 * c - x1 * s;
            g_k[off + i + half_head] = x1 * c + x0 * s;
        }
    }
    __syncthreads();

    // ===== KV cache append =====
    for (int i = tid; i < kv_dim; i += num_threads) {
        k_cache[pos_id * kv_dim + i] = g_k[i];
        v_cache[pos_id * kv_dim + i] = g_v[i];
    }
    __syncthreads();
}
