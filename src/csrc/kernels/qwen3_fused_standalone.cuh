#pragma once

#include <cuda_bf16.h>
#include "qwen3.cuh"
#include "utils.cuh"

static constexpr int FUSED_TILE_K = 256;
static constexpr int FUSED_ROWS_PER_BLOCK = 8;
static constexpr int FUSED_THREADS = 256;

template <int M_BATCH>
__global__ void __launch_bounds__(FUSED_THREADS)
    rmsnorm_qkv_split_bf16_kernel(__nv_bfloat16* __restrict__ g_q, __nv_bfloat16* __restrict__ g_k,
                                  __nv_bfloat16* __restrict__ g_v, const __nv_bfloat16* __restrict__ input,
                                  const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ norm_w,
                                  int M) {
    constexpr int HIDDEN = QWEN3_1_7B.hidden_size;
    constexpr int Q_DIM = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;
    constexpr int TILE_K = FUSED_TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = FUSED_THREADS >> 5;

    const int row_base = blockIdx.x * FUSED_ROWS_PER_BLOCK;
    const int my_row = row_base + warp_id;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float* s_reduce = reinterpret_cast<float*>(smem_raw + M_BATCH * TILE_K * sizeof(__nv_bfloat16));
    float* s_rms = s_reduce + kernels::WARP_SIZE;

    for (int m = 0; m < M; m++) {
        float ss = 0.0f;
        for (int k = threadIdx.x; k < HIDDEN; k += FUSED_THREADS) {
            float val = __bfloat162float(input[m * HIDDEN + k]);
            ss += val * val;
        }
        float total = kernels::block_reduce_sum(ss, s_reduce, lane_id, warp_id, num_warps);
        if (threadIdx.x == 0) {
            s_rms[m] = rsqrtf(total / HIDDEN + QWEN3_1_7B.rms_norm_eps);
        }
        __syncthreads();
    }

    if (my_row >= QKV_DIM)
        return;

    float acc[M_BATCH];
#pragma unroll
    for (int m = 0; m < M_BATCH; m++)
        acc[m] = 0.0f;

    const __nv_bfloat16* w_row = weight + (long long)my_row * HIDDEN;

    for (int tile_start = 0; tile_start < HIDDEN; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, HIDDEN);
        const int tile_len = tile_end - tile_start;

        {
            const int total_elems = M * tile_len;
            for (int idx = threadIdx.x; idx < total_elems; idx += FUSED_THREADS) {
                const int m = idx / tile_len;
                const int k = idx % tile_len;
                const int gk = tile_start + k;
                float val = __bfloat162float(input[m * HIDDEN + gk]);
                float nw = __bfloat162float(norm_w[gk]);
                s_input[m * TILE_K + k] = __float2bfloat16(val * s_rms[m] * nw);
            }
        }
        __syncthreads();

        for (int k = lane_id; k < tile_len; k += 32) {
            float w = __bfloat162float(w_row[tile_start + k]);
#pragma unroll
            for (int m = 0; m < M_BATCH; m++) {
                float in_val = __bfloat162float(s_input[m * TILE_K + k]);
                acc[m] += w * in_val;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < M_BATCH; m++) {
        if (m >= M)
            break;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);

        if (lane_id == 0) {
            __nv_bfloat16 result = __float2bfloat16(acc[m]);
            if (my_row < Q_DIM) {
                g_q[m * Q_DIM + my_row] = result;
            } else if (my_row < Q_DIM + KV_DIM) {
                g_k[m * KV_DIM + (my_row - Q_DIM)] = result;
            } else {
                g_v[m * KV_DIM + (my_row - Q_DIM - KV_DIM)] = result;
            }
        }
    }
}

template <int M_BATCH>
__global__ void __launch_bounds__(FUSED_THREADS)
    thingemm_residual_bf16_kernel(__nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ input,
                                  const __nv_bfloat16* __restrict__ weight, int M, int input_dim, int output_dim) {
    constexpr int TILE_K = FUSED_TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int row_base = blockIdx.x * FUSED_ROWS_PER_BLOCK;
    const int my_row = row_base + warp_id;
    if (my_row >= output_dim)
        return;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    float acc[M_BATCH];
#pragma unroll
    for (int m = 0; m < M_BATCH; m++)
        acc[m] = 0.0f;

    const __nv_bfloat16* w_row = weight + (long long)my_row * input_dim;

    for (int tile_start = 0; tile_start < input_dim; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, input_dim);
        const int tile_len = tile_end - tile_start;

        {
            const int total_elems = M * tile_len;
            for (int idx = threadIdx.x; idx < total_elems; idx += FUSED_THREADS) {
                const int m = idx / tile_len;
                const int k = idx % tile_len;
                s_input[m * TILE_K + k] = input[m * input_dim + tile_start + k];
            }
        }
        __syncthreads();

        for (int k = lane_id; k < tile_len; k += 32) {
            float w = __bfloat162float(w_row[tile_start + k]);
#pragma unroll
            for (int m = 0; m < M_BATCH; m++) {
                float in_val = __bfloat162float(s_input[m * TILE_K + k]);
                acc[m] += w * in_val;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < M_BATCH; m++) {
        if (m >= M)
            break;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);

        if (lane_id == 0) {
            float prev = __bfloat162float(residual[m * output_dim + my_row]);
            residual[m * output_dim + my_row] = __float2bfloat16(prev + acc[m]);
        }
    }
}

template <int M_BATCH>
__global__ void __launch_bounds__(FUSED_THREADS)
    rmsnorm_thingemm_bf16_kernel(__nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ input,
                                 const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ norm_w,
                                 int M, int input_dim, int output_dim) {
    constexpr int TILE_K = FUSED_TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = FUSED_THREADS >> 5;

    const int row_base = blockIdx.x * FUSED_ROWS_PER_BLOCK;
    const int my_row = row_base + warp_id;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float* s_reduce = reinterpret_cast<float*>(smem_raw + M_BATCH * TILE_K * sizeof(__nv_bfloat16));
    float* s_rms = s_reduce + kernels::WARP_SIZE;

    for (int m = 0; m < M; m++) {
        float ss = 0.0f;
        for (int k = threadIdx.x; k < input_dim; k += FUSED_THREADS) {
            float val = __bfloat162float(input[m * input_dim + k]);
            ss += val * val;
        }
        float total = kernels::block_reduce_sum(ss, s_reduce, lane_id, warp_id, num_warps);
        if (threadIdx.x == 0) {
            s_rms[m] = rsqrtf(total / input_dim + QWEN3_1_7B.rms_norm_eps);
        }
        __syncthreads();
    }

    if (my_row >= output_dim)
        return;

    float acc[M_BATCH];
#pragma unroll
    for (int m = 0; m < M_BATCH; m++)
        acc[m] = 0.0f;

    const __nv_bfloat16* w_row = weight + (long long)my_row * input_dim;

    for (int tile_start = 0; tile_start < input_dim; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, input_dim);
        const int tile_len = tile_end - tile_start;

        {
            const int total_elems = M * tile_len;
            for (int idx = threadIdx.x; idx < total_elems; idx += FUSED_THREADS) {
                const int m = idx / tile_len;
                const int k = idx % tile_len;
                const int gk = tile_start + k;
                float val = __bfloat162float(input[m * input_dim + gk]);
                float nw = __bfloat162float(norm_w[gk]);
                s_input[m * TILE_K + k] = __float2bfloat16(val * s_rms[m] * nw);
            }
        }
        __syncthreads();

        for (int k = lane_id; k < tile_len; k += 32) {
            float w = __bfloat162float(w_row[tile_start + k]);
#pragma unroll
            for (int m = 0; m < M_BATCH; m++) {
                float in_val = __bfloat162float(s_input[m * TILE_K + k]);
                acc[m] += w * in_val;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < M_BATCH; m++) {
        if (m >= M)
            break;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);

        if (lane_id == 0) {
            output[m * output_dim + my_row] = __float2bfloat16(acc[m]);
        }
    }
}

__global__ void __launch_bounds__(256)
    silu_gate_mul_strided_bf16_kernel(__nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ gate_up,
                                      int M, int intermediate_size) {
    const int total = M * intermediate_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    const int m = idx / intermediate_size;
    const int j = idx % intermediate_size;
    const int stride = 2 * intermediate_size;

    float g = __bfloat162float(gate_up[m * stride + j]);
    float u = __bfloat162float(gate_up[m * stride + intermediate_size + j]);
    float s = g / (1.0f + expf(-g));
    output[idx] = __float2bfloat16(s * u);
}

template <int M_BATCH>
__global__ void __launch_bounds__(FUSED_THREADS)
    strided_oproj_residual_bf16_kernel(__nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ sdpa_out,
                                       const __nv_bfloat16* __restrict__ weight, int M, int num_heads, int head_dim) {
    constexpr int TILE_K = FUSED_TILE_K;
    const int output_dim = num_heads * head_dim;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int row_base = blockIdx.x * FUSED_ROWS_PER_BLOCK;
    const int my_row = row_base + warp_id;
    if (my_row >= output_dim)
        return;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    float acc[M_BATCH];
#pragma unroll
    for (int m = 0; m < M_BATCH; m++)
        acc[m] = 0.0f;

    const __nv_bfloat16* w_row = weight + (long long)my_row * output_dim;

    for (int tile_start = 0; tile_start < output_dim; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, output_dim);
        const int tile_len = tile_end - tile_start;

        {
            const int total_elems = M * tile_len;
            for (int idx = threadIdx.x; idx < total_elems; idx += FUSED_THREADS) {
                const int m = idx / tile_len;
                const int k_offset = idx % tile_len;
                const int k = tile_start + k_offset;
                const int h = k / head_dim;
                const int d = k % head_dim;
                s_input[m * TILE_K + k_offset] = sdpa_out[h * M * head_dim + m * head_dim + d];
            }
        }
        __syncthreads();

        for (int k = lane_id; k < tile_len; k += 32) {
            float w = __bfloat162float(w_row[tile_start + k]);
#pragma unroll
            for (int m = 0; m < M_BATCH; m++) {
                float in_val = __bfloat162float(s_input[m * TILE_K + k]);
                acc[m] += w * in_val;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < M_BATCH; m++) {
        if (m >= M)
            break;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);

        if (lane_id == 0) {
            float prev = __bfloat162float(hidden[m * output_dim + my_row]);
            hidden[m * output_dim + my_row] = __float2bfloat16(prev + acc[m]);
        }
    }
}

__global__ void __launch_bounds__(256) qkv_split_norm_rope_kvcache_bf16_kernel(
    __nv_bfloat16* __restrict__ g_q_out, __nv_bfloat16* __restrict__ g_k_out, __nv_bfloat16* __restrict__ g_v_out,
    const __nv_bfloat16* __restrict__ g_qkv, const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w, const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache, float* __restrict__ k_cache, float* __restrict__ v_cache, int N,
    int start_pos) {
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;
    constexpr int NUM_KV_HEADS = QWEN3_1_7B.num_key_value_heads;
    constexpr int HEAD_DIM_C = QWEN3_1_7B.head_dim;
    constexpr int HALF_HEAD = HEAD_DIM_C / 2;
    constexpr int Q_DIM_C = NUM_Q_HEADS * HEAD_DIM_C;
    constexpr int KV_DIM_C = NUM_KV_HEADS * HEAD_DIM_C;
    constexpr int QKV_DIM_C = Q_DIM_C + 2 * KV_DIM_C;

    const int t = blockIdx.x;
    if (t >= N)
        return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);

    const int lane_id = tid % kernels::WARP_SIZE;
    const int warp_id = tid / kernels::WARP_SIZE;
    const int num_warps = num_threads / kernels::WARP_SIZE;

    const __nv_bfloat16* qkv_t = g_qkv + (long long)t * QKV_DIM_C;
    const __nv_bfloat16* q_in = qkv_t;
    const __nv_bfloat16* k_in = qkv_t + Q_DIM_C;
    const __nv_bfloat16* v_in = qkv_t + Q_DIM_C + KV_DIM_C;

    __nv_bfloat16* q_out = g_q_out + (long long)t * Q_DIM_C;
    __nv_bfloat16* k_out = g_k_out + (long long)t * KV_DIM_C;
    __nv_bfloat16* v_out = g_v_out + (long long)t * KV_DIM_C;

    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int off = h * HEAD_DIM_C;
        float head_ss = 0.0f;
        for (int i = tid; i < HEAD_DIM_C; i += num_threads) {
            float val = __bfloat162float(q_in[off + i]);
            head_ss += val * val;
        }
        float total = kernels::block_reduce_sum(head_ss, s_reduce, lane_id, warp_id, num_warps);
        float rms = rsqrtf(total / HEAD_DIM_C + QWEN3_1_7B.rms_norm_eps);
        for (int i = tid; i < HEAD_DIM_C; i += num_threads) {
            float val = __bfloat162float(q_in[off + i]);
            float w = __bfloat162float(q_norm_w[i]);
            q_out[off + i] = __float2bfloat16(val * rms * w);
        }
        __syncthreads();
    }

    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int off = h * HEAD_DIM_C;
        float head_ss = 0.0f;
        for (int i = tid; i < HEAD_DIM_C; i += num_threads) {
            float val = __bfloat162float(k_in[off + i]);
            head_ss += val * val;
        }
        float total = kernels::block_reduce_sum(head_ss, s_reduce, lane_id, warp_id, num_warps);
        float rms = rsqrtf(total / HEAD_DIM_C + QWEN3_1_7B.rms_norm_eps);
        for (int i = tid; i < HEAD_DIM_C; i += num_threads) {
            float val = __bfloat162float(k_in[off + i]);
            float w = __bfloat162float(k_norm_w[i]);
            k_out[off + i] = __float2bfloat16(val * rms * w);
        }
        __syncthreads();
    }

    for (int i = tid; i < KV_DIM_C; i += num_threads) {
        v_out[i] = v_in[i];
    }

    const int pos = start_pos + t;
    const __nv_bfloat16* cos_pos = cos_cache + (long long)pos * HEAD_DIM_C;
    const __nv_bfloat16* sin_pos = sin_cache + (long long)pos * HEAD_DIM_C;

    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int off = h * HEAD_DIM_C;
        for (int i = tid; i < HALF_HEAD; i += num_threads) {
            float x0 = __bfloat162float(q_out[off + i]);
            float x1 = __bfloat162float(q_out[off + i + HALF_HEAD]);
            float c = __bfloat162float(cos_pos[i]);
            float s = __bfloat162float(sin_pos[i]);
            q_out[off + i] = __float2bfloat16(x0 * c - x1 * s);
            q_out[off + i + HALF_HEAD] = __float2bfloat16(x1 * c + x0 * s);
        }
    }

    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int off = h * HEAD_DIM_C;
        for (int i = tid; i < HALF_HEAD; i += num_threads) {
            float x0 = __bfloat162float(k_out[off + i]);
            float x1 = __bfloat162float(k_out[off + i + HALF_HEAD]);
            float c = __bfloat162float(cos_pos[i]);
            float s = __bfloat162float(sin_pos[i]);
            k_out[off + i] = __float2bfloat16(x0 * c - x1 * s);
            k_out[off + i + HALF_HEAD] = __float2bfloat16(x1 * c + x0 * s);
        }
    }

    float* kc = k_cache + (long long)pos * KV_DIM_C;
    float* vc = v_cache + (long long)pos * KV_DIM_C;
    for (int i = tid; i < KV_DIM_C; i += num_threads) {
        kc[i] = __bfloat162float(k_out[i]);
        vc[i] = __bfloat162float(v_out[i]);
    }
}

__global__ void __launch_bounds__(256)
    qkv_split_bf16_kernel(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
                          const __nv_bfloat16* __restrict__ qkv, int M, int q_dim, int kv_dim) {
    const int qkv_dim = q_dim + 2 * kv_dim;
    const int total = M * qkv_dim;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int m = i / qkv_dim;
        const int j = i % qkv_dim;
        __nv_bfloat16 val = qkv[i];

        if (j < q_dim) {
            q[m * q_dim + j] = val;
        } else if (j < q_dim + kv_dim) {
            k[m * kv_dim + (j - q_dim)] = val;
        } else {
            v[m * kv_dim + (j - q_dim - kv_dim)] = val;
        }
    }
}

__global__ void __launch_bounds__(FUSED_THREADS)
    rmsnorm_lmhead_bf16_kernel(__nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ input,
                               const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ norm_w,
                               int vocab_size) {
    constexpr int HIDDEN = QWEN3_1_7B.hidden_size;
    constexpr int TILE_K = FUSED_TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = FUSED_THREADS >> 5;

    const int row_base = blockIdx.x * FUSED_ROWS_PER_BLOCK;
    const int my_row = row_base + warp_id;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float* s_reduce = reinterpret_cast<float*>(smem_raw + TILE_K * sizeof(__nv_bfloat16));

    float ss = 0.0f;
    for (int k = threadIdx.x; k < HIDDEN; k += FUSED_THREADS) {
        float val = __bfloat162float(input[k]);
        ss += val * val;
    }
    float total = kernels::block_reduce_sum(ss, s_reduce, lane_id, warp_id, num_warps);
    float rms = rsqrtf(total / HIDDEN + QWEN3_1_7B.rms_norm_eps);
    __syncthreads();

    if (my_row >= vocab_size)
        return;

    float acc = 0.0f;
    const __nv_bfloat16* w_row = weight + (long long)my_row * HIDDEN;

    for (int tile_start = 0; tile_start < HIDDEN; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, HIDDEN);
        const int tile_len = tile_end - tile_start;

        for (int k = threadIdx.x; k < tile_len; k += FUSED_THREADS) {
            const int gk = tile_start + k;
            float val = __bfloat162float(input[gk]);
            float nw = __bfloat162float(norm_w[gk]);
            s_input[k] = __float2bfloat16(val * rms * nw);
        }
        __syncthreads();

        for (int k = lane_id; k < tile_len; k += 32) {
            acc += __bfloat162float(w_row[tile_start + k]) * __bfloat162float(s_input[k]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    if (lane_id == 0) {
        output[my_row] = __float2bfloat16(acc);
    }
}
