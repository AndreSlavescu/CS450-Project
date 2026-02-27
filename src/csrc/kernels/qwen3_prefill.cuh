#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"

static constexpr int MAX_PREFILL_TOKENS = 64;

static constexpr int PREFILL_TILE = 16;

static constexpr int SMEM_CHUNK = 2048;

static constexpr int SMEM_MAX_N = 32;

__device__ __forceinline__ void prefill_load_bf16_chunk(__nv_bfloat16* __restrict__ s_buf,
                                                        const float* __restrict__ g_input, int N, int input_dim,
                                                        int col_start, int chunk_cols) {
    const int total = N * chunk_cols;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int t = i / chunk_cols;
        const int c = i % chunk_cols;
        s_buf[i] = __float2bfloat16(g_input[t * input_dim + col_start + c]);
    }
}

__device__ __forceinline__ void prefill_thin_gemm(float* __restrict__ output, const float* __restrict__ input,
                                                  const __nv_bfloat16* __restrict__ weight, int N, int output_dim,
                                                  int input_dim) {
    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;
    const int input_dim8 = input_dim / 8;

    for (int out_row = global_warp; out_row < output_dim; out_row += total_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(weight + (long long)out_row * input_dim);

        for (int t0 = 0; t0 < N; t0 += PREFILL_TILE) {
            const int tile_end = min(t0 + PREFILL_TILE, N);
            const int tile_n = tile_end - t0;

            float dot[PREFILL_TILE];
#pragma unroll
            for (int i = 0; i < PREFILL_TILE; i++)
                dot[i] = 0.f;

            for (int k = lane; k < input_dim8; k += 32) {
                uint4 raw = __ldcg(row8 + k);
                float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
                float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));

                for (int t = 0; t < tile_n; t++) {
                    const float4* inp4 = reinterpret_cast<const float4*>(input + (long long)(t0 + t) * input_dim);
                    float4 x0 = inp4[k * 2], x1 = inp4[k * 2 + 1];
                    dot[t] += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                              f67.x * x1.z + f67.y * x1.w;
                }
            }

            for (int t = 0; t < tile_n; t++) {
#pragma unroll
                for (int s = 16; s > 0; s >>= 1)
                    dot[t] += __shfl_down_sync(0xffffffff, dot[t], s);
                if (lane == 0)
                    output[(long long)(t0 + t) * output_dim + out_row] = dot[t];
            }
        }
    }
}

__device__ __forceinline__ void prefill_qkv_thin_gemm(float* __restrict__ g_q, float* __restrict__ g_k,
                                                      float* __restrict__ g_v, const float* __restrict__ input,
                                                      const __nv_bfloat16* __restrict__ qkv_w, int N) {
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = QWEN3_1_7B.qkv_output_dim();
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int hs8 = hidden_size / 8;

    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int out_row = global_warp; out_row < qkv_dim; out_row += total_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(qkv_w + (long long)out_row * hidden_size);

        float* out_buf;
        int out_stride, local_row;
        if (out_row < q_dim) {
            out_buf = g_q;
            out_stride = q_dim;
            local_row = out_row;
        } else if (out_row < q_dim + kv_dim) {
            out_buf = g_k;
            out_stride = kv_dim;
            local_row = out_row - q_dim;
        } else {
            out_buf = g_v;
            out_stride = kv_dim;
            local_row = out_row - q_dim - kv_dim;
        }

        for (int t0 = 0; t0 < N; t0 += PREFILL_TILE) {
            const int tile_end = min(t0 + PREFILL_TILE, N);
            const int tile_n = tile_end - t0;

            float dot[PREFILL_TILE];
#pragma unroll
            for (int i = 0; i < PREFILL_TILE; i++)
                dot[i] = 0.f;

            for (int k = lane; k < hs8; k += 32) {
                uint4 raw = __ldcg(row8 + k);
                float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
                float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));

                for (int t = 0; t < tile_n; t++) {
                    const float4* inp4 = reinterpret_cast<const float4*>(input + (long long)(t0 + t) * hidden_size);
                    float4 x0 = inp4[k * 2], x1 = inp4[k * 2 + 1];
                    dot[t] += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                              f67.x * x1.z + f67.y * x1.w;
                }
            }

            for (int t = 0; t < tile_n; t++) {
#pragma unroll
                for (int s = 16; s > 0; s >>= 1)
                    dot[t] += __shfl_down_sync(0xffffffff, dot[t], s);
                if (lane == 0)
                    out_buf[(long long)(t0 + t) * out_stride + local_row] = dot[t];
            }
        }
    }
}

__device__ __forceinline__ void prefill_thin_gemm_residual(float* __restrict__ output, const float* __restrict__ input,
                                                           const __nv_bfloat16* __restrict__ weight, int N,
                                                           int output_dim, int input_dim) {
    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;
    const int input_dim8 = input_dim / 8;

    for (int out_row = global_warp; out_row < output_dim; out_row += total_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(weight + (long long)out_row * input_dim);

        for (int t0 = 0; t0 < N; t0 += PREFILL_TILE) {
            const int tile_end = min(t0 + PREFILL_TILE, N);
            const int tile_n = tile_end - t0;

            float dot[PREFILL_TILE];
#pragma unroll
            for (int i = 0; i < PREFILL_TILE; i++)
                dot[i] = 0.f;

            for (int k = lane; k < input_dim8; k += 32) {
                uint4 raw = __ldcg(row8 + k);
                float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
                float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));

                for (int t = 0; t < tile_n; t++) {
                    const float4* inp4 = reinterpret_cast<const float4*>(input + (long long)(t0 + t) * input_dim);
                    float4 x0 = inp4[k * 2], x1 = inp4[k * 2 + 1];
                    dot[t] += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                              f67.x * x1.z + f67.y * x1.w;
                }
            }

            for (int t = 0; t < tile_n; t++) {
#pragma unroll
                for (int s = 16; s > 0; s >>= 1)
                    dot[t] += __shfl_down_sync(0xffffffff, dot[t], s);
                if (lane == 0)
                    output[(long long)(t0 + t) * output_dim + out_row] += dot[t];
            }
        }
    }
}

__device__ __forceinline__ void prefill_upgate_silu_thin_gemm(float* __restrict__ silu_out,
                                                              const float* __restrict__ input,
                                                              const __nv_bfloat16* __restrict__ gate_w,
                                                              const __nv_bfloat16* __restrict__ up_w, int N) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    constexpr int hs8 = hidden_size / 8;

    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int out_row = global_warp; out_row < intermediate_size; out_row += total_warps) {
        const uint4* gr8 = reinterpret_cast<const uint4*>(gate_w + (long long)out_row * hidden_size);
        const uint4* ur8 = reinterpret_cast<const uint4*>(up_w + (long long)out_row * hidden_size);

        for (int t0 = 0; t0 < N; t0 += PREFILL_TILE) {
            const int tile_end = min(t0 + PREFILL_TILE, N);
            const int tile_n = tile_end - t0;

            float gate_dot[PREFILL_TILE], up_dot[PREFILL_TILE];
#pragma unroll
            for (int i = 0; i < PREFILL_TILE; i++) {
                gate_dot[i] = 0.f;
                up_dot[i] = 0.f;
            }

            for (int k = lane; k < hs8; k += 32) {
                uint4 graw = __ldcg(gr8 + k);
                float2 gf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
                float2 gf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
                float2 gf45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.z));
                float2 gf67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.w));

                uint4 uraw = __ldcg(ur8 + k);
                float2 uf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.x));
                float2 uf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.y));
                float2 uf45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.z));
                float2 uf67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.w));

                for (int t = 0; t < tile_n; t++) {
                    const float4* inp4 = reinterpret_cast<const float4*>(input + (long long)(t0 + t) * hidden_size);
                    float4 x0 = inp4[k * 2], x1 = inp4[k * 2 + 1];
                    gate_dot[t] += gf01.x * x0.x + gf01.y * x0.y + gf23.x * x0.z + gf23.y * x0.w + gf45.x * x1.x +
                                   gf45.y * x1.y + gf67.x * x1.z + gf67.y * x1.w;
                    up_dot[t] += uf01.x * x0.x + uf01.y * x0.y + uf23.x * x0.z + uf23.y * x0.w + uf45.x * x1.x +
                                 uf45.y * x1.y + uf67.x * x1.z + uf67.y * x1.w;
                }
            }

            for (int t = 0; t < tile_n; t++) {
#pragma unroll
                for (int s = 16; s > 0; s >>= 1) {
                    gate_dot[t] += __shfl_down_sync(0xffffffff, gate_dot[t], s);
                    up_dot[t] += __shfl_down_sync(0xffffffff, up_dot[t], s);
                }
                if (lane == 0)
                    silu_out[(long long)(t0 + t) * intermediate_size + out_row] =
                        kernels::silu_multiply(gate_dot[t], up_dot[t]);
            }
        }
    }
}

__device__ __forceinline__ void prefill_rmsnorm_all(float* __restrict__ g_normed, const float* __restrict__ g_input,
                                                    const float* __restrict__ weight, int N, int dim) {
    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lane_id = tid % kernels::WARP_SIZE;
    const int warp_id = tid / kernels::WARP_SIZE;
    const int num_warps = num_threads / kernels::WARP_SIZE;

    for (int t = blockIdx.x; t < N; t += gridDim.x) {
        const float* inp = g_input + (long long)t * dim;
        float* out = g_normed + (long long)t * dim;

        float thread_ss = 0.0f;
        for (int i = tid; i < dim; i += 2 * num_threads) {
            float x0 = inp[i];
            thread_ss += x0 * x0;
            int j = i + num_threads;
            if (j < dim) {
                float x1 = inp[j];
                thread_ss += x1 * x1;
            }
        }

        float total_ss = kernels::block_reduce_sum(thread_ss, s_reduce, lane_id, warp_id, num_warps);
        float rms = rsqrtf(total_ss / dim + QWEN3_1_7B.rms_norm_eps);

        for (int i = tid; i < dim; i += 2 * num_threads) {
            out[i] = inp[i] * rms * weight[i];
            int j = i + num_threads;
            if (j < dim)
                out[j] = inp[j] * rms * weight[j];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void prefill_qknorm_rope_kvcache(
    float* __restrict__ g_q, float* __restrict__ g_k, const float* __restrict__ g_v, float* __restrict__ k_cache,
    float* __restrict__ v_cache, const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
    const float* __restrict__ cos_cache, const float* __restrict__ sin_cache, int N, int start_pos) {
    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int num_q_heads = QWEN3_1_7B.num_attention_heads;
    constexpr int num_kv_heads = QWEN3_1_7B.num_key_value_heads;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int half_head = head_dim / 2;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lane_id = tid % kernels::WARP_SIZE;
    const int warp_id = tid / kernels::WARP_SIZE;
    const int num_warps = num_threads / kernels::WARP_SIZE;

    for (int t = blockIdx.x; t < N; t += gridDim.x) {
        float* q_t = g_q + (long long)t * q_dim;
        float* k_t = g_k + (long long)t * kv_dim;
        const float* v_t = g_v + (long long)t * kv_dim;

        const int pos = start_pos + t;
        const float* cos_pos = cos_cache + (long long)pos * head_dim;
        const float* sin_pos = sin_cache + (long long)pos * head_dim;

        kernels::rmsnorm_per_head(q_t, q_norm_w, num_q_heads, head_dim, s_reduce, tid, num_threads, lane_id, warp_id,
                                  num_warps);

        kernels::rmsnorm_per_head(k_t, k_norm_w, num_kv_heads, head_dim, s_reduce, tid, num_threads, lane_id, warp_id,
                                  num_warps);

        for (int h = 0; h < num_q_heads; h++) {
            int off = h * head_dim;
            for (int i = tid; i < half_head; i += num_threads) {
                float x0 = q_t[off + i], x1 = q_t[off + i + half_head];
                float c = cos_pos[i], s = sin_pos[i];
                q_t[off + i] = x0 * c - x1 * s;
                q_t[off + i + half_head] = x1 * c + x0 * s;
            }
        }

        for (int h = 0; h < num_kv_heads; h++) {
            int off = h * head_dim;
            for (int i = tid; i < half_head; i += num_threads) {
                float x0 = k_t[off + i], x1 = k_t[off + i + half_head];
                float c = cos_pos[i], s = sin_pos[i];
                k_t[off + i] = x0 * c - x1 * s;
                k_t[off + i + half_head] = x1 * c + x0 * s;
            }
        }
        __syncthreads();

        for (int i = tid; i < kv_dim; i += num_threads) {
            k_cache[(long long)pos * kv_dim + i] = k_t[i];
            v_cache[(long long)pos * kv_dim + i] = v_t[i];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void prefill_causal_attention(float* __restrict__ g_attn_out, const float* __restrict__ g_q,
                                                         const float* __restrict__ g_k, const float* __restrict__ g_v,
                                                         int N, float scale) {
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;
    constexpr int NUM_KV_HEADS = QWEN3_1_7B.num_key_value_heads;
    constexpr int HEAD_DIM = QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;
    constexpr int Q_DIM = NUM_Q_HEADS * HEAD_DIM;
    constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / kernels::WARP_SIZE;

    if (blockIdx.x >= NUM_Q_HEADS)
        return;

    const int q_h = blockIdx.x;
    const int kv_h = q_h / GQA_RATIO;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    for (int qt = warp; qt < N; qt += num_warps) {
        const float* q_head = g_q + (long long)qt * Q_DIM + q_h * HEAD_DIM;

        float m_acc = -INFINITY;
        float d_acc = 0.0f;
        float o_acc[ELEMS_PER_LANE] = {};

        for (int kv_pos = 0; kv_pos <= qt; kv_pos++) {
            const float* k = g_k + (long long)kv_pos * KV_DIM + kv_h * HEAD_DIM;
            const float* v = g_v + (long long)kv_pos * KV_DIM + kv_h * HEAD_DIM;

            float dot = 0.0f;
#pragma unroll
            for (int d = lane; d < HEAD_DIM; d += kernels::WARP_SIZE)
                dot += q_head[d] * k[d];
#pragma unroll
            for (int s = 16; s > 0; s >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, s);
            const float score = __shfl_sync(0xffffffff, dot, 0) * scale;

            const float m_new = fmaxf(m_acc, score);
            const float alpha = kernels::fast_exp_approx(m_acc - m_new);
            const float beta = kernels::fast_exp_approx(score - m_new);
            d_acc = d_acc * alpha + beta;
            m_acc = m_new;

            const float4 v4 = *reinterpret_cast<const float4*>(v + lane * ELEMS_PER_LANE);
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++)
                o_acc[i] = o_acc[i] * alpha + beta * (&v4.x)[i];
        }

        const float inv_d = kernels::approx_reciprocal(d_acc);
        float* out = g_attn_out + (long long)qt * Q_DIM + q_h * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++)
            out[lane * ELEMS_PER_LANE + i] = o_acc[i] * inv_d;
    }
    __syncthreads();
}

template <bool RESIDUAL>
__device__ __forceinline__ void bf16_thin_gemm_row(float* __restrict__ output,
                                                   const __nv_bfloat16* __restrict__ s_input,
                                                   const __nv_bfloat16* __restrict__ weight_row, int N, int out_stride,
                                                   int out_row, int chunk_cols, int lane) {
    const int chunk8 = chunk_cols / 8;

    for (int t0 = 0; t0 < N; t0 += PREFILL_TILE) {
        const int tile_end = min(t0 + PREFILL_TILE, N);
        const int tile_n = tile_end - t0;

        __nv_bfloat162 a0[PREFILL_TILE], a1[PREFILL_TILE];
        __nv_bfloat162 a2[PREFILL_TILE], a3[PREFILL_TILE];
#pragma unroll
        for (int i = 0; i < PREFILL_TILE; i++) {
            a0[i] = __float2bfloat162_rn(0.f);
            a1[i] = __float2bfloat162_rn(0.f);
            a2[i] = __float2bfloat162_rn(0.f);
            a3[i] = __float2bfloat162_rn(0.f);
        }

        const uint4* w8 = reinterpret_cast<const uint4*>(weight_row);

        for (int k = lane; k < chunk8; k += 32) {
            uint4 raw_w = __ldcg(w8 + k);
            __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&raw_w.x);
            __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&raw_w.y);
            __nv_bfloat162 w45 = *reinterpret_cast<const __nv_bfloat162*>(&raw_w.z);
            __nv_bfloat162 w67 = *reinterpret_cast<const __nv_bfloat162*>(&raw_w.w);

            for (int t = 0; t < tile_n; t++) {
                const uint4* x8 = reinterpret_cast<const uint4*>(s_input + (long long)(t0 + t) * chunk_cols);
                uint4 raw_x = x8[k];
                __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&raw_x.x);
                __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&raw_x.y);
                __nv_bfloat162 x45 = *reinterpret_cast<const __nv_bfloat162*>(&raw_x.z);
                __nv_bfloat162 x67 = *reinterpret_cast<const __nv_bfloat162*>(&raw_x.w);

                a0[t] = __hfma2(w01, x01, a0[t]);
                a1[t] = __hfma2(w23, x23, a1[t]);
                a2[t] = __hfma2(w45, x45, a2[t]);
                a3[t] = __hfma2(w67, x67, a3[t]);
            }
        }

        for (int t = 0; t < tile_n; t++) {
            float2 f0 = __bfloat1622float2(a0[t]);
            float2 f1 = __bfloat1622float2(a1[t]);
            float2 f2 = __bfloat1622float2(a2[t]);
            float2 f3 = __bfloat1622float2(a3[t]);
            float dot = f0.x + f0.y + f1.x + f1.y + f2.x + f2.y + f3.x + f3.y;
#pragma unroll
            for (int s = 16; s > 0; s >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, s);
            if (lane == 0) {
                long long idx = (long long)(t0 + t) * out_stride + out_row;
                if constexpr (RESIDUAL)
                    output[idx] += dot;
                else
                    output[idx] = dot;
            }
        }
    }
}

__device__ __forceinline__ void prefill_qkv_thin_gemm_bf16(float* __restrict__ g_q, float* __restrict__ g_k,
                                                           float* __restrict__ g_v, const float* __restrict__ input,
                                                           const __nv_bfloat16* __restrict__ qkv_w, int N,
                                                           __nv_bfloat16* __restrict__ s_buf) {
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = QWEN3_1_7B.qkv_output_dim();
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;

    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int t_base = 0; t_base < N; t_base += SMEM_MAX_N) {
        const int bn = min(SMEM_MAX_N, N - t_base);

        prefill_load_bf16_chunk(s_buf, input + (long long)t_base * hidden_size, bn, hidden_size, 0, hidden_size);
        __syncthreads();

        for (int out_row = global_warp; out_row < qkv_dim; out_row += total_warps) {
            float* out_buf;
            int out_stride, local_row;
            if (out_row < q_dim) {
                out_buf = g_q + (long long)t_base * q_dim;
                out_stride = q_dim;
                local_row = out_row;
            } else if (out_row < q_dim + kv_dim) {
                out_buf = g_k + (long long)t_base * kv_dim;
                out_stride = kv_dim;
                local_row = out_row - q_dim;
            } else {
                out_buf = g_v + (long long)t_base * kv_dim;
                out_stride = kv_dim;
                local_row = out_row - q_dim - kv_dim;
            }

            const __nv_bfloat16* w_row = qkv_w + (long long)out_row * hidden_size;
            bf16_thin_gemm_row<false>(out_buf, s_buf, w_row, bn, out_stride, local_row, hidden_size, lane);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void prefill_thin_gemm_residual_bf16(float* __restrict__ output,
                                                                const float* __restrict__ input,
                                                                const __nv_bfloat16* __restrict__ weight, int N,
                                                                int output_dim, int input_dim,
                                                                __nv_bfloat16* __restrict__ s_buf) {
    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int t_base = 0; t_base < N; t_base += SMEM_MAX_N) {
        const int bn = min(SMEM_MAX_N, N - t_base);

        for (int col_base = 0; col_base < input_dim; col_base += SMEM_CHUNK) {
            const int chunk_cols = min(SMEM_CHUNK, input_dim - col_base);

            prefill_load_bf16_chunk(s_buf, input + (long long)t_base * input_dim, bn, input_dim, col_base, chunk_cols);
            __syncthreads();

            for (int out_row = global_warp; out_row < output_dim; out_row += total_warps) {
                const __nv_bfloat16* w_row = weight + (long long)out_row * input_dim + col_base;

                bf16_thin_gemm_row<true>(output + (long long)t_base * output_dim, s_buf, w_row, bn, output_dim, out_row,
                                         chunk_cols, lane);
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void prefill_upgate_silu_thin_gemm_bf16(float* __restrict__ silu_out,
                                                                   const float* __restrict__ input,
                                                                   const __nv_bfloat16* __restrict__ gate_w,
                                                                   const __nv_bfloat16* __restrict__ up_w, int N,
                                                                   __nv_bfloat16* __restrict__ s_buf) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    constexpr int hs8 = hidden_size / 8;

    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int t_base = 0; t_base < N; t_base += SMEM_MAX_N) {
        const int bn = min(SMEM_MAX_N, N - t_base);

        prefill_load_bf16_chunk(s_buf, input + (long long)t_base * hidden_size, bn, hidden_size, 0, hidden_size);
        __syncthreads();

        for (int out_row = global_warp; out_row < intermediate_size; out_row += total_warps) {
            const uint4* gr8 = reinterpret_cast<const uint4*>(gate_w + (long long)out_row * hidden_size);
            const uint4* ur8 = reinterpret_cast<const uint4*>(up_w + (long long)out_row * hidden_size);

            for (int t0 = 0; t0 < bn; t0 += PREFILL_TILE) {
                const int tile_end = min(t0 + PREFILL_TILE, bn);
                const int tile_n = tile_end - t0;

                __nv_bfloat162 ga0[PREFILL_TILE], ga1[PREFILL_TILE];
                __nv_bfloat162 ga2[PREFILL_TILE], ga3[PREFILL_TILE];
                __nv_bfloat162 ua0[PREFILL_TILE], ua1[PREFILL_TILE];
                __nv_bfloat162 ua2[PREFILL_TILE], ua3[PREFILL_TILE];
#pragma unroll
                for (int i = 0; i < PREFILL_TILE; i++) {
                    ga0[i] = ga1[i] = ga2[i] = ga3[i] = __float2bfloat162_rn(0.f);
                    ua0[i] = ua1[i] = ua2[i] = ua3[i] = __float2bfloat162_rn(0.f);
                }

                for (int k = lane; k < hs8; k += 32) {
                    uint4 graw = __ldcg(gr8 + k);
                    __nv_bfloat162 gw01 = *reinterpret_cast<const __nv_bfloat162*>(&graw.x);
                    __nv_bfloat162 gw23 = *reinterpret_cast<const __nv_bfloat162*>(&graw.y);
                    __nv_bfloat162 gw45 = *reinterpret_cast<const __nv_bfloat162*>(&graw.z);
                    __nv_bfloat162 gw67 = *reinterpret_cast<const __nv_bfloat162*>(&graw.w);

                    uint4 uraw = __ldcg(ur8 + k);
                    __nv_bfloat162 uw01 = *reinterpret_cast<const __nv_bfloat162*>(&uraw.x);
                    __nv_bfloat162 uw23 = *reinterpret_cast<const __nv_bfloat162*>(&uraw.y);
                    __nv_bfloat162 uw45 = *reinterpret_cast<const __nv_bfloat162*>(&uraw.z);
                    __nv_bfloat162 uw67 = *reinterpret_cast<const __nv_bfloat162*>(&uraw.w);

                    for (int t = 0; t < tile_n; t++) {
                        const uint4* x8 = reinterpret_cast<const uint4*>(s_buf + (long long)(t0 + t) * hidden_size);
                        uint4 rx = x8[k];
                        __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&rx.x);
                        __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&rx.y);
                        __nv_bfloat162 x45 = *reinterpret_cast<const __nv_bfloat162*>(&rx.z);
                        __nv_bfloat162 x67 = *reinterpret_cast<const __nv_bfloat162*>(&rx.w);

                        ga0[t] = __hfma2(gw01, x01, ga0[t]);
                        ga1[t] = __hfma2(gw23, x23, ga1[t]);
                        ga2[t] = __hfma2(gw45, x45, ga2[t]);
                        ga3[t] = __hfma2(gw67, x67, ga3[t]);

                        ua0[t] = __hfma2(uw01, x01, ua0[t]);
                        ua1[t] = __hfma2(uw23, x23, ua1[t]);
                        ua2[t] = __hfma2(uw45, x45, ua2[t]);
                        ua3[t] = __hfma2(uw67, x67, ua3[t]);
                    }
                }

                for (int t = 0; t < tile_n; t++) {
                    float2 gf0 = __bfloat1622float2(ga0[t]), gf1 = __bfloat1622float2(ga1[t]);
                    float2 gf2 = __bfloat1622float2(ga2[t]), gf3 = __bfloat1622float2(ga3[t]);
                    float gate_dot = gf0.x + gf0.y + gf1.x + gf1.y + gf2.x + gf2.y + gf3.x + gf3.y;

                    float2 uf0 = __bfloat1622float2(ua0[t]), uf1 = __bfloat1622float2(ua1[t]);
                    float2 uf2 = __bfloat1622float2(ua2[t]), uf3 = __bfloat1622float2(ua3[t]);
                    float up_dot = uf0.x + uf0.y + uf1.x + uf1.y + uf2.x + uf2.y + uf3.x + uf3.y;

#pragma unroll
                    for (int s = 16; s > 0; s >>= 1) {
                        gate_dot += __shfl_down_sync(0xffffffff, gate_dot, s);
                        up_dot += __shfl_down_sync(0xffffffff, up_dot, s);
                    }
                    if (lane == 0)
                        silu_out[(long long)(t_base + t0 + t) * intermediate_size + out_row] =
                            kernels::silu_multiply(gate_dot, up_dot);
                }
            }
        }
        __syncthreads();
    }
}
