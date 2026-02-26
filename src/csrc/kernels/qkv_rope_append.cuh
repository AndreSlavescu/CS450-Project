#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "../profiler/gpu_profiler.cuh"

__device__ void qkv_matvec_device(float* __restrict__ g_q, float* __restrict__ g_k, float* __restrict__ g_v,
                                  const float* __restrict__ hidden, const float* __restrict__ attn_ln_w,
                                  const __nv_bfloat16* __restrict__ qkv_weight) {
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

    kernels::rmsnorm(s_post_ln, hidden, attn_ln_w, hidden_size, s_reduce, tid, num_threads, lane_id, warp_id,
                     num_warps);
    __syncthreads();

    {
        const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
        int lane = threadIdx.x & 31;
        int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int total_warps = (blockDim.x * gridDim.x) >> 5;

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
}

__device__ void qknorm_rope_kvcache_device(float* __restrict__ g_q, float* __restrict__ g_k,
                                           const float* __restrict__ g_v, float* __restrict__ k_cache,
                                           float* __restrict__ v_cache, const float* __restrict__ q_norm_w,
                                           const float* __restrict__ k_norm_w, const float* __restrict__ cos_cached,
                                           const float* __restrict__ sin_cached, int pos_id) {
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int num_q_heads = QWEN3_1_7B.num_attention_heads;
    constexpr int num_kv_heads = QWEN3_1_7B.num_key_value_heads;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int half_head = head_dim / 2;

    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    kernels::rmsnorm_per_head(g_q, q_norm_w, num_q_heads, head_dim, s_reduce, tid, num_threads, lane_id, warp_id,
                              num_warps);

    kernels::rmsnorm_per_head(g_k, k_norm_w, num_kv_heads, head_dim, s_reduce, tid, num_threads, lane_id, warp_id,
                              num_warps);

    for (int h = 0; h < num_q_heads; h++) {
        int off = h * head_dim;
        for (int i = tid; i < half_head; i += num_threads) {
            float x0 = g_q[off + i], x1 = g_q[off + i + half_head];
            float c = cos_cached[i], s = sin_cached[i];
            g_q[off + i] = x0 * c - x1 * s;
            g_q[off + i + half_head] = x1 * c + x0 * s;
        }
    }

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

    for (int i = tid; i < kv_dim; i += num_threads) {
        k_cache[pos_id * kv_dim + i] = g_k[i];
        v_cache[pos_id * kv_dim + i] = g_v[i];
    }
    __syncthreads();
}
