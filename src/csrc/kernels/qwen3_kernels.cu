#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "qwen3.cuh"
#include "qwen3_types.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"
#include "qkv_rope_append.cuh"
#include "oproj_residual.cuh"
#include "upgate_silu.cuh"
#include "downproj_residual.cuh"
#include "rms_lm_head.cuh"
#include "qwen3_prefill.cuh"
#include "qwen3_vm_ops.cuh"
#include "qwen3_fused_standalone.cuh"
#include "gpu_profiler.cuh"

__device__ __forceinline__ void prefetch_l2(const void* __restrict__ ptr, long long nbytes, int tid, int ntotal) {
    const char* p = static_cast<const char*>(ptr);
    const long long nlines = (nbytes + 127LL) >> 7;
    for (long long i = tid; i < nlines; i += ntotal) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p + (i << 7)) : "memory");
    }
}

__device__ void flash_decode_gqa_device(float* attn_out, const float* g_q, const float* k_cache, const float* v_cache,
                                        int seq_len, float scale) {
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;
    constexpr int NUM_KV_HEADS = QWEN3_1_7B.num_key_value_heads;
    constexpr int HEAD_DIM = QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;
    constexpr int GQA_RATIO = QWEN3_1_7B.gqa_ratio();
    constexpr int ELEMS_PER_LANE = HEAD_DIM / kernels::WARP_SIZE;

    if (blockIdx.x >= NUM_Q_HEADS)
        return;
    const int q_h = blockIdx.x;
    const int kv_h = q_h / GQA_RATIO;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    extern __shared__ char smem_raw[];
    float* s_o = reinterpret_cast<float*>(smem_raw);
    float* s_m = s_o + num_warps * HEAD_DIM;
    float* s_d = s_m + num_warps;

    const float* q_head = g_q + q_h * HEAD_DIM;

    const int chunk = (seq_len + num_warps - 1) / num_warps;
    const int kv_start = warp * chunk;
    const int kv_end = min(kv_start + chunk, seq_len);

    float m_w = -INFINITY;
    float d_w = 0.0f;
    float o_w[ELEMS_PER_LANE] = {};

    for (int pos = kv_start; pos < kv_end; pos++) {
        const float* k = k_cache + (long long)pos * KV_DIM + kv_h * HEAD_DIM;
        const float* v = v_cache + (long long)pos * KV_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int d = lane; d < HEAD_DIM; d += kernels::WARP_SIZE) {
            dot += q_head[d] * k[d];
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, s);
        }
        const float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        const float m_new = fmaxf(m_w, score);
        const float alpha = kernels::fast_exp_approx(m_w - m_new);
        const float beta = kernels::fast_exp_approx(score - m_new);
        d_w = d_w * alpha + beta;
        m_w = m_new;

        const float4 v4 = *reinterpret_cast<const float4*>(v + lane * ELEMS_PER_LANE);
        o_w[0] = o_w[0] * alpha + beta * v4.x;
        o_w[1] = o_w[1] * alpha + beta * v4.y;
        o_w[2] = o_w[2] * alpha + beta * v4.z;
        o_w[3] = o_w[3] * alpha + beta * v4.w;
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
        s_o[warp * HEAD_DIM + lane * ELEMS_PER_LANE + i] = o_w[i];
    }
    if (lane == 0) {
        s_m[warp] = m_w;
        s_d[warp] = d_w;
    }
    __syncthreads();

    if (warp == 0) {
        float m_acc = s_m[0];
        float d_acc = s_d[0];
        float o_acc[ELEMS_PER_LANE];
#pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            o_acc[i] = s_o[lane * ELEMS_PER_LANE + i];
        }

        for (int w = 1; w < num_warps; w++) {
            const float m_w2 = s_m[w];
            const float d_w2 = s_d[w];
            const float m_new2 = fmaxf(m_acc, m_w2);
            const float a1 = kernels::fast_exp_approx(m_acc - m_new2);
            const float a2 = kernels::fast_exp_approx(m_w2 - m_new2);
            d_acc = d_acc * a1 + d_w2 * a2;
            m_acc = m_new2;
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++) {
                o_acc[i] = o_acc[i] * a1 + s_o[w * HEAD_DIM + lane * ELEMS_PER_LANE + i] * a2;
            }
        }

        const float inv_d = kernels::approx_reciprocal(d_acc);
        float* out = attn_out + q_h * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            out[lane * ELEMS_PER_LANE + i] = o_acc[i] * inv_d;
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(512, 1)
    qwen3_decode_persistent(float* hidden, float* g_q, float* g_k, float* g_v, float* g_attn_out, float* g_silu,
                            const Qwen3LayerWeights* __restrict__ layer_weights, const float* __restrict__ cos_cache,
                            const float* __restrict__ sin_cache, int pos_id, int max_seq_len, float attn_scale,
                            unsigned int* kv_flag, GridBarrier* bar) {
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;

    unsigned int local_epoch = 0u;

    for (int layer = 0; layer < num_layers; layer++) {
        const Qwen3LayerWeights& w = layer_weights[layer];
        const float* cos_pos = cos_cache + (long long)pos_id * QWEN3_1_7B.head_dim;
        const float* sin_pos = sin_cache + (long long)pos_id * QWEN3_1_7B.head_dim;

        qkv_matvec_device(g_q, g_k, g_v, hidden, w.attn_ln_w, w.qkv_w);
        grid_barrier_sync(bar, 128u, local_epoch);

        if (blockIdx.x == 0) {
            qknorm_rope_kvcache_device(g_q, g_k, g_v, w.k_cache, w.v_cache, w.q_norm_w, w.k_norm_w, cos_pos, sin_pos,
                                       pos_id);
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicExch(kv_flag, 1u);
        } else if (blockIdx.x < NUM_Q_HEADS) {
            volatile unsigned int* vflag = kv_flag;
            while (*vflag == 0u) {
            }
        }

        flash_decode_gqa_device(g_attn_out, g_q, w.k_cache, w.v_cache, pos_id + 1, attn_scale);

        if (blockIdx.x >= NUM_Q_HEADS) {
            constexpr long long hs = QWEN3_1_7B.hidden_size;
            constexpr long long is = QWEN3_1_7B.intermediate_size;
            constexpr long long bf = sizeof(__nv_bfloat16);
            const int idle_tid = (blockIdx.x - NUM_Q_HEADS) * blockDim.x + threadIdx.x;
            const int idle_total = (gridDim.x - NUM_Q_HEADS) * blockDim.x;
            prefetch_l2(w.o_proj_w, hs * hs * bf, idle_tid, idle_total);
            prefetch_l2(w.gate_w, is * hs * bf, idle_tid, idle_total);
            prefetch_l2(w.up_w, is * hs * bf, idle_tid, idle_total);
            prefetch_l2(w.down_proj_w, hs * is * bf, idle_tid, idle_total);
        }
        grid_barrier_sync(bar, 128u, local_epoch);

        if (blockIdx.x == 0 && threadIdx.x == 0)
            *kv_flag = 0u;

        oproj_residual_device(hidden, g_attn_out, w.o_proj_w, nullptr, nullptr);
        grid_barrier_sync(bar, 128u, local_epoch);

        upgate_silu_device(g_silu, hidden, w.mlp_ln_w, w.gate_w, w.up_w, nullptr, nullptr);
        grid_barrier_sync(bar, 128u, local_epoch);

        downproj_residual_device(hidden, g_silu, w.down_proj_w, nullptr, nullptr);
        grid_barrier_sync(bar, 128u, local_epoch);
    }
}

__global__ void __launch_bounds__(256, 4)
    qwen3_lm_head_kernel(float* logits, const float* __restrict__ hidden, const float* __restrict__ norm_w,
                         const __nv_bfloat16* __restrict__ lm_head_w, int vocab_size) {
    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + QWEN3_1_7B.hidden_size;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lane_id = tid % kernels::WARP_SIZE;
    const int warp_id = tid / kernels::WARP_SIZE;
    const int num_warps = num_threads / kernels::WARP_SIZE;

    kernels::rmsnorm(s_post_ln, hidden, norm_w, QWEN3_1_7B.hidden_size, s_reduce, tid, num_threads, lane_id, warp_id,
                     num_warps);
    __syncthreads();

    const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
    const int lane = tid & 31;
    const int global_warp = (blockIdx.x * num_threads + tid) >> 5;
    const int total_warps = (gridDim.x * num_threads) >> 5;

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

torch::Tensor qwen3_decode_persistent_forward(torch::Tensor hidden, torch::Tensor attn_ln_ws, torch::Tensor qkv_weights,
                                              torch::Tensor q_norm_ws, torch::Tensor k_norm_ws, torch::Tensor cos_cache,
                                              torch::Tensor sin_cache, torch::Tensor k_caches, torch::Tensor v_caches,
                                              torch::Tensor o_proj_ws, torch::Tensor mlp_ln_ws, torch::Tensor gate_ws,
                                              torch::Tensor up_ws, torch::Tensor down_proj_ws, torch::Tensor norm_w,
                                              torch::Tensor lm_head_w, int pos_id) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = QWEN3_1_7B.qkv_output_dim();
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;

    int max_seq_len = static_cast<int>(k_caches.size(1)) / kv_dim;
    int vocab_size = static_cast<int>(lm_head_w.size(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    std::vector<Qwen3LayerWeights> h_weights(num_layers);
    for (int l = 0; l < num_layers; l++) {
        h_weights[l].attn_ln_w = attn_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].qkv_w = bf16_ptr(qkv_weights) + (long long)l * qkv_dim * hidden_size;
        h_weights[l].q_norm_w = q_norm_ws.data_ptr<float>() + (long long)l * head_dim;
        h_weights[l].k_norm_w = k_norm_ws.data_ptr<float>() + (long long)l * head_dim;
        h_weights[l].o_proj_w = bf16_ptr(o_proj_ws) + (long long)l * hidden_size * hidden_size;
        h_weights[l].mlp_ln_w = mlp_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].gate_w = bf16_ptr(gate_ws) + (long long)l * intermediate_size * hidden_size;
        h_weights[l].up_w = bf16_ptr(up_ws) + (long long)l * intermediate_size * hidden_size;
        h_weights[l].down_proj_w = bf16_ptr(down_proj_ws) + (long long)l * hidden_size * intermediate_size;
        h_weights[l].k_cache = k_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
        h_weights[l].v_cache = v_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
    }
    static Qwen3LayerWeights* d_weights_decode = nullptr;
    if (d_weights_decode == nullptr)
        cudaMalloc(&d_weights_decode, num_layers * sizeof(Qwen3LayerWeights));
    cudaMemcpyAsync(d_weights_decode, h_weights.data(), num_layers * sizeof(Qwen3LayerWeights), cudaMemcpyHostToDevice,
                    stream);

    auto output_hidden = hidden.clone();
    auto g_q = torch::empty({q_dim}, hidden.options());
    auto g_k = torch::empty({kv_dim}, hidden.options());
    auto g_v = torch::empty({kv_dim}, hidden.options());
    auto g_attn_out = torch::empty({q_dim}, hidden.options());
    auto g_silu = torch::empty({intermediate_size}, hidden.options());
    auto logits = torch::empty({vocab_size}, hidden.options());

    auto kv_flag = torch::zeros({1}, torch::dtype(torch::kInt32).device(hidden.device()));

    static GridBarrier* d_bar = nullptr;
    if (d_bar == nullptr)
        cudaMalloc(&d_bar, sizeof(GridBarrier));
    cudaMemsetAsync(d_bar, 0, sizeof(GridBarrier), stream);

    float attn_scale = QWEN3_1_7B.attn_scale();

    const int block_size = 512;
    const int grid_size = 128;
    size_t smem_bytes = (hidden_size + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    float* hidden_ptr = output_hidden.data_ptr<float>();
    float* g_q_ptr = g_q.data_ptr<float>();
    float* g_k_ptr = g_k.data_ptr<float>();
    float* g_v_ptr = g_v.data_ptr<float>();
    float* g_attn_out_ptr = g_attn_out.data_ptr<float>();
    float* g_silu_ptr = g_silu.data_ptr<float>();
    int max_seq_val = max_seq_len;
    unsigned int* kv_flag_ptr = reinterpret_cast<unsigned int*>(kv_flag.data_ptr<int>());

    const float* cos_ptr = cos_cache.data_ptr<float>();
    const float* sin_ptr = sin_cache.data_ptr<float>();
    void* args[] = {&hidden_ptr, &g_q_ptr, &g_k_ptr, &g_v_ptr,     &g_attn_out_ptr, &g_silu_ptr,  &d_weights_decode,
                    &cos_ptr,    &sin_ptr, &pos_id,  &max_seq_val, &attn_scale,     &kv_flag_ptr, &d_bar};

    dim3 grid(grid_size), block(block_size);
    cudaLaunchCooperativeKernel((void*)qwen3_decode_persistent, grid, block, args, smem_bytes, stream);

    const int lm_blocks = 512;
    const int lm_threads = 256;
    size_t smem_lm = (hidden_size + kernels::WARP_SIZE) * sizeof(float);
    const float* norm_w_ptr = norm_w.data_ptr<float>();
    const __nv_bfloat16* lm_ptr = reinterpret_cast<const __nv_bfloat16*>(lm_head_w.data_ptr<at::BFloat16>());
    float* logits_ptr = logits.data_ptr<float>();

    qwen3_lm_head_kernel<<<lm_blocks, lm_threads, smem_lm, stream>>>(logits_ptr, hidden_ptr, norm_w_ptr, lm_ptr,
                                                                     vocab_size);

    return logits;
}

__global__ void __launch_bounds__(512, 1)
    qwen3_prefill_persistent(float* hidden, float* g_normed, float* g_q, float* g_k, float* g_v, float* g_attn_out,
                             float* g_silu, const Qwen3LayerWeights* __restrict__ layer_weights,
                             const float* __restrict__ cos_cache, const float* __restrict__ sin_cache, int N,
                             int start_pos, float attn_scale, GridBarrier* bar) {
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    unsigned int local_epoch = 0u;

    for (int layer = 0; layer < num_layers; layer++) {
        const Qwen3LayerWeights& w = layer_weights[layer];

        prefill_rmsnorm_all(g_normed, hidden, w.attn_ln_w, N, hidden_size);
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_qkv_thin_gemm_bf16(g_q, g_k, g_v, g_normed, w.qkv_w, N, s_buf);
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_qknorm_rope_kvcache(g_q, g_k, g_v, w.k_cache, w.v_cache, w.q_norm_w, w.k_norm_w, cos_cache, sin_cache,
                                    N, start_pos);
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_causal_attention(g_attn_out, g_q, g_k, g_v, N, attn_scale);

        if (blockIdx.x >= QWEN3_1_7B.num_attention_heads) {
            constexpr long long hs = QWEN3_1_7B.hidden_size;
            constexpr long long is = QWEN3_1_7B.intermediate_size;
            constexpr long long bf = sizeof(__nv_bfloat16);
            const int idle_tid = (blockIdx.x - QWEN3_1_7B.num_attention_heads) * blockDim.x + threadIdx.x;
            const int idle_total = (gridDim.x - QWEN3_1_7B.num_attention_heads) * blockDim.x;
            prefetch_l2(w.o_proj_w, hs * hs * bf, idle_tid, idle_total);
            prefetch_l2(w.gate_w, is * hs * bf, idle_tid, idle_total);
            prefetch_l2(w.up_w, is * hs * bf, idle_tid, idle_total);
            prefetch_l2(w.down_proj_w, hs * is * bf, idle_tid, idle_total);
        }
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_thin_gemm_residual_bf16(hidden, g_attn_out, w.o_proj_w, N, hidden_size, hidden_size, s_buf);
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_rmsnorm_all(g_normed, hidden, w.mlp_ln_w, N, hidden_size);
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_upgate_silu_thin_gemm_bf16(g_silu, g_normed, w.gate_w, w.up_w, N, s_buf);
        grid_barrier_sync(bar, 128u, local_epoch);

        prefill_thin_gemm_residual_bf16(hidden, g_silu, w.down_proj_w, N, hidden_size, intermediate_size, s_buf);
        grid_barrier_sync(bar, 128u, local_epoch);
    }
}

torch::Tensor qwen3_prefill_persistent_forward(torch::Tensor hidden, torch::Tensor attn_ln_ws,
                                               torch::Tensor qkv_weights, torch::Tensor q_norm_ws,
                                               torch::Tensor k_norm_ws, torch::Tensor cos_cache,
                                               torch::Tensor sin_cache, torch::Tensor k_caches, torch::Tensor v_caches,
                                               torch::Tensor o_proj_ws, torch::Tensor mlp_ln_ws, torch::Tensor gate_ws,
                                               torch::Tensor up_ws, torch::Tensor down_proj_ws, torch::Tensor norm_w,
                                               torch::Tensor lm_head_w, int start_pos) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = QWEN3_1_7B.qkv_output_dim();
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;

    const int N = static_cast<int>(hidden.size(0));
    int max_seq_len = static_cast<int>(k_caches.size(1)) / kv_dim;
    int vocab_size = static_cast<int>(lm_head_w.size(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    std::vector<Qwen3LayerWeights> h_weights(num_layers);
    for (int l = 0; l < num_layers; l++) {
        h_weights[l].attn_ln_w = attn_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].qkv_w = bf16_ptr(qkv_weights) + (long long)l * qkv_dim * hidden_size;
        h_weights[l].q_norm_w = q_norm_ws.data_ptr<float>() + (long long)l * QWEN3_1_7B.head_dim;
        h_weights[l].k_norm_w = k_norm_ws.data_ptr<float>() + (long long)l * QWEN3_1_7B.head_dim;
        h_weights[l].o_proj_w = bf16_ptr(o_proj_ws) + (long long)l * hidden_size * hidden_size;
        h_weights[l].mlp_ln_w = mlp_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].gate_w = bf16_ptr(gate_ws) + (long long)l * intermediate_size * hidden_size;
        h_weights[l].up_w = bf16_ptr(up_ws) + (long long)l * intermediate_size * hidden_size;
        h_weights[l].down_proj_w = bf16_ptr(down_proj_ws) + (long long)l * hidden_size * intermediate_size;
        h_weights[l].k_cache = k_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
        h_weights[l].v_cache = v_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
    }
    static Qwen3LayerWeights* d_weights_prefill = nullptr;
    if (d_weights_prefill == nullptr)
        cudaMalloc(&d_weights_prefill, num_layers * sizeof(Qwen3LayerWeights));
    cudaMemcpyAsync(d_weights_prefill, h_weights.data(), num_layers * sizeof(Qwen3LayerWeights), cudaMemcpyHostToDevice,
                    stream);

    auto opts = torch::dtype(torch::kFloat32).device(hidden.device());
    auto output_hidden = hidden.clone();
    auto g_normed = torch::empty({N, hidden_size}, opts);
    auto g_q = torch::empty({N, q_dim}, opts);
    auto g_k = torch::empty({N, kv_dim}, opts);
    auto g_v = torch::empty({N, kv_dim}, opts);
    auto g_attn = torch::empty({N, q_dim}, opts);
    auto g_silu = torch::empty({N, intermediate_size}, opts);
    auto logits = torch::empty({vocab_size}, opts);

    static GridBarrier* d_bar_prefill = nullptr;
    if (d_bar_prefill == nullptr)
        cudaMalloc(&d_bar_prefill, sizeof(GridBarrier));
    cudaMemsetAsync(d_bar_prefill, 0, sizeof(GridBarrier), stream);

    float attn_scale_val = QWEN3_1_7B.attn_scale();

    const int block_size = 512;
    const int grid_size = 128;
    size_t smem_bytes = (size_t)N * SMEM_CHUNK * sizeof(__nv_bfloat16) + kernels::WARP_SIZE * sizeof(float);

    static size_t last_smem_configured = 0;
    if (smem_bytes > last_smem_configured) {
        cudaFuncSetAttribute(qwen3_prefill_persistent, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        last_smem_configured = smem_bytes;
    }

    float* hidden_ptr = output_hidden.data_ptr<float>();
    float* normed_ptr = g_normed.data_ptr<float>();
    float* g_q_ptr = g_q.data_ptr<float>();
    float* g_k_ptr = g_k.data_ptr<float>();
    float* g_v_ptr = g_v.data_ptr<float>();
    float* attn_ptr = g_attn.data_ptr<float>();
    float* silu_ptr = g_silu.data_ptr<float>();
    int n_val = N;
    int sp_val = start_pos;
    const float* cos_ptr = cos_cache.data_ptr<float>();
    const float* sin_ptr = sin_cache.data_ptr<float>();

    void* args[] = {&hidden_ptr,        &normed_ptr, &g_q_ptr, &g_k_ptr, &g_v_ptr, &attn_ptr,       &silu_ptr,
                    &d_weights_prefill, &cos_ptr,    &sin_ptr, &n_val,   &sp_val,  &attn_scale_val, &d_bar_prefill};

    dim3 grid(grid_size), block(block_size);
    cudaLaunchCooperativeKernel((void*)qwen3_prefill_persistent, grid, block, args, smem_bytes, stream);

    const int lm_blocks = 512;
    const int lm_threads = 256;
    size_t smem_lm = (hidden_size + kernels::WARP_SIZE) * sizeof(float);
    float* last_hidden = hidden_ptr + (long long)(N - 1) * hidden_size;
    const float* norm_w_ptr = norm_w.data_ptr<float>();
    const __nv_bfloat16* lm_ptr = reinterpret_cast<const __nv_bfloat16*>(lm_head_w.data_ptr<at::BFloat16>());
    float* logits_ptr = logits.data_ptr<float>();

    qwen3_lm_head_kernel<<<lm_blocks, lm_threads, smem_lm, stream>>>(logits_ptr, last_hidden, norm_w_ptr, lm_ptr,
                                                                     vocab_size);

    return logits;
}

__global__ void __launch_bounds__(512, 1)
    qwen3_vm_persistent(float* hidden, float* g_q, float* g_k, float* g_v, float* g_attn_out, float* g_attn_partial,
                        float* g_attn_lse, float* g_silu, float* logits,
                        const Qwen3LayerWeights* __restrict__ layer_weights, const float* __restrict__ cos_cache,
                        const float* __restrict__ sin_cache, const float* __restrict__ norm_w,
                        const __nv_bfloat16* __restrict__ lm_head_w, const int* __restrict__ instructions,
                        unsigned int* barriers, int queue_len, int pos_id, int max_seq_len, int vocab_size,
                        float attn_scale) {
    const int sm_id = blockIdx.x;
    const int* my_queue = instructions + (long long)sm_id * queue_len * 32;

    VMState st;
    st.hidden = hidden;
    st.g_q = g_q;
    st.g_k = g_k;
    st.g_v = g_v;
    st.g_attn_out = g_attn_out;
    st.g_attn_partial = g_attn_partial;
    st.g_attn_lse = g_attn_lse;
    st.g_silu = g_silu;
    st.logits = logits;
    st.layers = layer_weights;
    st.cos_cache = cos_cache;
    st.sin_cache = sin_cache;
    st.barriers = barriers;
    st.pos_id = pos_id;
    st.max_seq_len = max_seq_len;
    st.attn_scale = attn_scale;

    for (int pc = 0; pc < queue_len; pc++) {
        const int* instr = my_queue + pc * 32;
        const int opcode = instr[0];
        switch (opcode) {
        case 0:
            break;
        case 1:
            vm_exec_qkv(instr, st);
            break;
        case 2:
            vm_exec_flash_decode(instr, st);
            break;
        case 3:
            vm_exec_attn_reduction(instr, st);
            break;
        case 4:
            vm_exec_oproj_residual(instr, st);
            break;
        case 5:
            vm_exec_upgate_silu(instr, st);
            break;
        case 6:
            vm_exec_downproj_residual(instr, st);
            break;
        case 7:
            vm_exec_lm_head(instr, st, norm_w, lm_head_w, vocab_size);
            break;
        case 8:
            vm_exec_qknorm_rope_cache(instr, st);
            break;
        }
    }
}

torch::Tensor qwen3_vm_forward(torch::Tensor hidden, torch::Tensor attn_ln_ws, torch::Tensor qkv_weights,
                               torch::Tensor q_norm_ws, torch::Tensor k_norm_ws, torch::Tensor cos_cache,
                               torch::Tensor sin_cache, torch::Tensor k_caches, torch::Tensor v_caches,
                               torch::Tensor o_proj_ws, torch::Tensor mlp_ln_ws, torch::Tensor gate_ws,
                               torch::Tensor up_ws, torch::Tensor down_proj_ws, torch::Tensor norm_w,
                               torch::Tensor lm_head_w, torch::Tensor instructions, torch::Tensor barrier_buf,
                               int pos_id) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = QWEN3_1_7B.qkv_output_dim();
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;
    constexpr int num_q_heads = QWEN3_1_7B.num_attention_heads;
    constexpr int max_partials = 24;

    int max_seq_len = static_cast<int>(k_caches.size(1)) / kv_dim;
    int vocab_size = static_cast<int>(lm_head_w.size(0));
    int queue_len = static_cast<int>(instructions.size(1));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    std::vector<Qwen3LayerWeights> h_weights(num_layers);
    for (int l = 0; l < num_layers; l++) {
        h_weights[l].attn_ln_w = attn_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].qkv_w = bf16_ptr(qkv_weights) + (long long)l * qkv_dim * hidden_size;
        h_weights[l].q_norm_w = q_norm_ws.data_ptr<float>() + (long long)l * head_dim;
        h_weights[l].k_norm_w = k_norm_ws.data_ptr<float>() + (long long)l * head_dim;
        h_weights[l].o_proj_w = bf16_ptr(o_proj_ws) + (long long)l * hidden_size * hidden_size;
        h_weights[l].mlp_ln_w = mlp_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].gate_w = bf16_ptr(gate_ws) + (long long)l * intermediate_size * hidden_size;
        h_weights[l].up_w = bf16_ptr(up_ws) + (long long)l * intermediate_size * hidden_size;
        h_weights[l].down_proj_w = bf16_ptr(down_proj_ws) + (long long)l * hidden_size * intermediate_size;
        h_weights[l].k_cache = k_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
        h_weights[l].v_cache = v_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
    }
    static Qwen3LayerWeights* d_weights_vm = nullptr;
    if (d_weights_vm == nullptr)
        cudaMalloc(&d_weights_vm, num_layers * sizeof(Qwen3LayerWeights));
    cudaMemcpyAsync(d_weights_vm, h_weights.data(), num_layers * sizeof(Qwen3LayerWeights), cudaMemcpyHostToDevice,
                    stream);

    auto output_hidden = hidden.clone();
    auto g_q = torch::empty({q_dim}, hidden.options());
    auto g_k = torch::empty({kv_dim}, hidden.options());
    auto g_v = torch::empty({kv_dim}, hidden.options());
    auto g_attn_out = torch::empty({q_dim}, hidden.options());
    auto g_attn_partial = torch::empty({num_q_heads, max_partials, head_dim}, hidden.options());
    auto g_attn_lse = torch::empty({num_q_heads, max_partials}, hidden.options());
    auto g_silu = torch::empty({intermediate_size}, hidden.options());
    auto logits = torch::empty({vocab_size}, hidden.options());

    float attn_scale = QWEN3_1_7B.attn_scale();

    const int block_size = 512;
    const int grid_size = 128;
    size_t smem_bytes = (hidden_size + kernels::WARP_SIZE) * sizeof(float);

    float* hidden_ptr = output_hidden.data_ptr<float>();
    float* g_q_ptr = g_q.data_ptr<float>();
    float* g_k_ptr = g_k.data_ptr<float>();
    float* g_v_ptr = g_v.data_ptr<float>();
    float* attn_out_ptr = g_attn_out.data_ptr<float>();
    float* attn_part_ptr = g_attn_partial.data_ptr<float>();
    float* attn_lse_ptr = g_attn_lse.data_ptr<float>();
    float* silu_ptr = g_silu.data_ptr<float>();
    float* logits_ptr = logits.data_ptr<float>();
    const float* cos_ptr = cos_cache.data_ptr<float>();
    const float* sin_ptr = sin_cache.data_ptr<float>();
    const float* norm_w_ptr = norm_w.data_ptr<float>();
    const __nv_bfloat16* lm_ptr = reinterpret_cast<const __nv_bfloat16*>(lm_head_w.data_ptr<at::BFloat16>());
    const int* instr_ptr = instructions.data_ptr<int>();
    unsigned int* bar_ptr = reinterpret_cast<unsigned int*>(barrier_buf.data_ptr<int>());
    int msv = max_seq_len;
    int ql = queue_len;
    int vs = vocab_size;

    void* args[] = {&hidden_ptr,   &g_q_ptr,  &g_k_ptr,    &g_v_ptr,      &attn_out_ptr, &attn_part_ptr,
                    &attn_lse_ptr, &silu_ptr, &logits_ptr, &d_weights_vm, &cos_ptr,      &sin_ptr,
                    &norm_w_ptr,   &lm_ptr,   &instr_ptr,  &bar_ptr,      &ql,           &pos_id,
                    &msv,          &vs,       &attn_scale};

    dim3 grid(grid_size), block(block_size);
    cudaLaunchCooperativeKernel((void*)qwen3_vm_persistent, grid, block, args, smem_bytes, stream);

    return logits;
}

__global__ void __launch_bounds__(256)
    qknorm_rope_kvcache_bf16_kernel(__nv_bfloat16* __restrict__ g_q, __nv_bfloat16* __restrict__ g_k,
                                    const __nv_bfloat16* __restrict__ g_v, const __nv_bfloat16* __restrict__ q_norm_w,
                                    const __nv_bfloat16* __restrict__ k_norm_w,
                                    const __nv_bfloat16* __restrict__ cos_cache,
                                    const __nv_bfloat16* __restrict__ sin_cache, float* __restrict__ k_cache,
                                    float* __restrict__ v_cache, int N, int start_pos) {
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;
    constexpr int NUM_KV_HEADS = QWEN3_1_7B.num_key_value_heads;
    constexpr int HEAD_DIM = QWEN3_1_7B.head_dim;
    constexpr int HALF_HEAD = HEAD_DIM / 2;
    constexpr int Q_DIM = NUM_Q_HEADS * HEAD_DIM;
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;

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

    __nv_bfloat16* q_t = g_q + (long long)t * Q_DIM;
    __nv_bfloat16* k_t = g_k + (long long)t * KV_DIM;
    const __nv_bfloat16* v_t = g_v + (long long)t * KV_DIM;

    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int off = h * HEAD_DIM;
        float head_ss = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float val = __bfloat162float(q_t[off + i]);
            head_ss += val * val;
        }
        float total = kernels::block_reduce_sum(head_ss, s_reduce, lane_id, warp_id, num_warps);
        float rms = rsqrtf(total / HEAD_DIM + QWEN3_1_7B.rms_norm_eps);
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float val = __bfloat162float(q_t[off + i]);
            float w = __bfloat162float(q_norm_w[i]);
            q_t[off + i] = __float2bfloat16(val * rms * w);
        }
        __syncthreads();
    }

    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int off = h * HEAD_DIM;
        float head_ss = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float val = __bfloat162float(k_t[off + i]);
            head_ss += val * val;
        }
        float total = kernels::block_reduce_sum(head_ss, s_reduce, lane_id, warp_id, num_warps);
        float rms = rsqrtf(total / HEAD_DIM + QWEN3_1_7B.rms_norm_eps);
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float val = __bfloat162float(k_t[off + i]);
            float w = __bfloat162float(k_norm_w[i]);
            k_t[off + i] = __float2bfloat16(val * rms * w);
        }
        __syncthreads();
    }

    const int pos = start_pos + t;
    const __nv_bfloat16* cos_pos = cos_cache + (long long)pos * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_cache + (long long)pos * HEAD_DIM;

    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int off = h * HEAD_DIM;
        for (int i = tid; i < HALF_HEAD; i += num_threads) {
            float x0 = __bfloat162float(q_t[off + i]);
            float x1 = __bfloat162float(q_t[off + i + HALF_HEAD]);
            float c = __bfloat162float(cos_pos[i]);
            float s = __bfloat162float(sin_pos[i]);
            q_t[off + i] = __float2bfloat16(x0 * c - x1 * s);
            q_t[off + i + HALF_HEAD] = __float2bfloat16(x1 * c + x0 * s);
        }
    }

    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int off = h * HEAD_DIM;
        for (int i = tid; i < HALF_HEAD; i += num_threads) {
            float x0 = __bfloat162float(k_t[off + i]);
            float x1 = __bfloat162float(k_t[off + i + HALF_HEAD]);
            float c = __bfloat162float(cos_pos[i]);
            float s = __bfloat162float(sin_pos[i]);
            k_t[off + i] = __float2bfloat16(x0 * c - x1 * s);
            k_t[off + i + HALF_HEAD] = __float2bfloat16(x1 * c + x0 * s);
        }
    }

    float* kc = k_cache + (long long)pos * KV_DIM;
    float* vc = v_cache + (long long)pos * KV_DIM;
    for (int i = tid; i < KV_DIM; i += num_threads) {
        kc[i] = __bfloat162float(k_t[i]);
        vc[i] = __bfloat162float(v_t[i]);
    }
}

torch::Tensor qknorm_rope_kvcache_bf16_forward(torch::Tensor g_q, torch::Tensor g_k, torch::Tensor g_v,
                                               torch::Tensor q_norm_w, torch::Tensor k_norm_w, torch::Tensor cos_cache,
                                               torch::Tensor sin_cache, torch::Tensor k_cache, torch::Tensor v_cache,
                                               int start_pos) {
    const int N = g_q.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 256;
    const int blocks = N;
    const size_t smem = kernels::WARP_SIZE * sizeof(float);

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

    qknorm_rope_kvcache_bf16_kernel<<<blocks, threads, smem, stream>>>(
        bf16_ptr(g_q), bf16_ptr(g_k), bf16_cptr(g_v), bf16_cptr(q_norm_w), bf16_cptr(k_norm_w), bf16_cptr(cos_cache),
        bf16_cptr(sin_cache), k_cache.data_ptr<float>(), v_cache.data_ptr<float>(), N, start_pos);

    return g_q;
}

__global__ void __launch_bounds__(256)
    silu_gate_mul_bf16_kernel(__nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ gate,
                              const __nv_bfloat16* __restrict__ up, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float s = g / (1.0f + expf(-g));
    output[idx] = __float2bfloat16(s * u);
}

torch::Tensor silu_gate_mul_bf16_forward(torch::Tensor gate, torch::Tensor up) {
    auto output = torch::empty_like(gate);
    const int total = gate.numel();
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

    silu_gate_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(bf16_ptr(output), bf16_cptr(gate), bf16_cptr(up), total);

    return output;
}

#define DISPATCH_M_BATCH(M, KERNEL_CALL)                                                                               \
    do {                                                                                                               \
        if ((M) <= 16) {                                                                                               \
            KERNEL_CALL(16);                                                                                           \
        } else if ((M) <= 32) {                                                                                        \
            KERNEL_CALL(32);                                                                                           \
        } else if ((M) <= 64) {                                                                                        \
            KERNEL_CALL(64);                                                                                           \
        } else {                                                                                                       \
            TORCH_CHECK(false, "M_BATCH > 64 not supported; use cuBLAS");                                              \
        }                                                                                                              \
    } while (0)

std::vector<torch::Tensor> fused_rmsnorm_qkv_split_bf16_forward(torch::Tensor input, torch::Tensor weight,
                                                                torch::Tensor norm_w) {
    const int M = input.size(0);
    constexpr int Q_DIM = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;

    auto opts = input.options();
    auto q = torch::empty({M, Q_DIM}, opts);
    auto k = torch::empty({M, KV_DIM}, opts);
    auto v = torch::empty({M, KV_DIM}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (QKV_DIM + FUSED_ROWS_PER_BLOCK - 1) / FUSED_ROWS_PER_BLOCK;

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

#define QKV_LAUNCH(MB)                                                                                                 \
    rmsnorm_qkv_split_bf16_kernel<MB>                                                                                  \
        <<<blocks, FUSED_THREADS,                                                                                      \
           (MB) * FUSED_TILE_K * sizeof(__nv_bfloat16) + kernels::WARP_SIZE * sizeof(float) + (MB) * sizeof(float),    \
           stream>>>(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_cptr(input), bf16_cptr(weight), bf16_cptr(norm_w), M)
    DISPATCH_M_BATCH(M, QKV_LAUNCH);
#undef QKV_LAUNCH

    return {q, k, v};
}

void fused_thingemm_residual_bf16_forward(torch::Tensor residual, torch::Tensor input, torch::Tensor weight) {
    const int M = input.size(0);
    const int input_dim = input.size(1);
    const int output_dim = weight.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (output_dim + FUSED_ROWS_PER_BLOCK - 1) / FUSED_ROWS_PER_BLOCK;
    const size_t smem = (size_t)std::min(M, 64) * FUSED_TILE_K * sizeof(__nv_bfloat16);

    if (smem > 48 * 1024) {
        static size_t last_configured = 0;
        if (smem > last_configured) {
            cudaFuncSetAttribute(thingemm_residual_bf16_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem);
            last_configured = smem;
        }
    }

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

#define RESIDUAL_LAUNCH(MB)                                                                                            \
    thingemm_residual_bf16_kernel<MB><<<blocks, FUSED_THREADS, (MB) * FUSED_TILE_K * sizeof(__nv_bfloat16), stream>>>( \
        bf16_ptr(residual), bf16_cptr(input), bf16_cptr(weight), M, input_dim, output_dim)
    DISPATCH_M_BATCH(M, RESIDUAL_LAUNCH);
#undef RESIDUAL_LAUNCH
}

torch::Tensor fused_rmsnorm_thingemm_bf16_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor norm_w) {
    const int M = input.size(0);
    const int input_dim = input.size(1);
    const int output_dim = weight.size(0);

    auto output = torch::empty({M, output_dim}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (output_dim + FUSED_ROWS_PER_BLOCK - 1) / FUSED_ROWS_PER_BLOCK;
    const size_t smem =
        (size_t)std::min(M, 64) * FUSED_TILE_K * sizeof(__nv_bfloat16) + kernels::WARP_SIZE * sizeof(float);

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

#define NORM_GEMM_LAUNCH(MB)                                                                                           \
    rmsnorm_thingemm_bf16_kernel<MB>                                                                                   \
        <<<blocks, FUSED_THREADS,                                                                                      \
           (MB) * FUSED_TILE_K * sizeof(__nv_bfloat16) + kernels::WARP_SIZE * sizeof(float) + (MB) * sizeof(float),    \
           stream>>>(bf16_ptr(output), bf16_cptr(input), bf16_cptr(weight), bf16_cptr(norm_w), M, input_dim,           \
                     output_dim)
    DISPATCH_M_BATCH(M, NORM_GEMM_LAUNCH);
#undef NORM_GEMM_LAUNCH

    return output;
}

torch::Tensor fused_silu_gate_mul_strided_bf16_forward(torch::Tensor gate_up) {
    const int M = gate_up.size(0);
    const int intermediate_size = gate_up.size(1) / 2;
    const int total = M * intermediate_size;

    auto output = torch::empty({M, intermediate_size}, gate_up.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    silu_gate_mul_strided_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr<at::BFloat16>()), M, intermediate_size);

    return output;
}

std::vector<torch::Tensor> qkv_split_norm_rope_kvcache_bf16_forward(torch::Tensor qkv, torch::Tensor q_norm_w,
                                                                    torch::Tensor k_norm_w, torch::Tensor cos_cache,
                                                                    torch::Tensor sin_cache, torch::Tensor k_cache,
                                                                    torch::Tensor v_cache, int start_pos) {
    constexpr int Q_DIM = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;

    const int N = qkv.size(0);
    auto q = torch::empty({N, Q_DIM}, qkv.options());
    auto k = torch::empty({N, KV_DIM}, qkv.options());
    auto v = torch::empty({N, KV_DIM}, qkv.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = N;
    const size_t smem = kernels::WARP_SIZE * sizeof(float);

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

    qkv_split_norm_rope_kvcache_bf16_kernel<<<blocks, threads, smem, stream>>>(
        bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_cptr(qkv), bf16_cptr(q_norm_w), bf16_cptr(k_norm_w),
        bf16_cptr(cos_cache), bf16_cptr(sin_cache), k_cache.data_ptr<float>(), v_cache.data_ptr<float>(), N, start_pos);

    return {q, k, v};
}

void strided_oproj_residual_bf16_forward(torch::Tensor hidden, torch::Tensor sdpa_out, torch::Tensor weight) {
    const int num_heads = sdpa_out.size(0);
    const int M = sdpa_out.size(1);
    const int head_dim = sdpa_out.size(2);
    const int output_dim = num_heads * head_dim;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (output_dim + FUSED_ROWS_PER_BLOCK - 1) / FUSED_ROWS_PER_BLOCK;

    auto bf16_ptr = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
    auto bf16_cptr = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };

#define OPROJ_LAUNCH(MB)                                                                                               \
    strided_oproj_residual_bf16_kernel<MB>                                                                             \
        <<<blocks, FUSED_THREADS, (MB) * FUSED_TILE_K * sizeof(__nv_bfloat16), stream>>>(                              \
            bf16_ptr(hidden), bf16_cptr(sdpa_out), bf16_cptr(weight), M, num_heads, head_dim)
    DISPATCH_M_BATCH(M, OPROJ_LAUNCH);
#undef OPROJ_LAUNCH
}

std::vector<torch::Tensor> qkv_split_bf16_forward(torch::Tensor qkv) {
    constexpr int Q_DIM = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;

    const int M = qkv.size(0);
    auto q = torch::empty({M, Q_DIM}, qkv.options());
    auto k = torch::empty({M, KV_DIM}, qkv.options());
    auto v = torch::empty({M, KV_DIM}, qkv.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    const int total = M * (Q_DIM + 2 * KV_DIM);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    qkv_split_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()), M, Q_DIM, KV_DIM);

    return {q, k, v};
}

torch::Tensor fused_rmsnorm_lmhead_bf16_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor norm_w) {
    const int vocab_size = weight.size(0);
    auto output = torch::empty({vocab_size}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int blocks = (vocab_size + FUSED_ROWS_PER_BLOCK - 1) / FUSED_ROWS_PER_BLOCK;
    const size_t smem = FUSED_TILE_K * sizeof(__nv_bfloat16) + kernels::WARP_SIZE * sizeof(float);

    rmsnorm_lmhead_bf16_kernel<<<blocks, FUSED_THREADS, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()), vocab_size);

    return output;
}

#undef DISPATCH_M_BATCH

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qwen3_decode_persistent_forward", &qwen3_decode_persistent_forward);
    m.def("qwen3_prefill_persistent_forward", &qwen3_prefill_persistent_forward);
    m.def("qwen3_vm_forward", &qwen3_vm_forward);
    m.def("qknorm_rope_kvcache_bf16_forward", &qknorm_rope_kvcache_bf16_forward);
    m.def("silu_gate_mul_bf16_forward", &silu_gate_mul_bf16_forward);
    m.def("qkv_split_bf16_forward", &qkv_split_bf16_forward);
    m.def("qkv_split_norm_rope_kvcache_bf16_forward", &qkv_split_norm_rope_kvcache_bf16_forward);
    m.def("strided_oproj_residual_bf16_forward", &strided_oproj_residual_bf16_forward);
    m.def("fused_rmsnorm_qkv_split_bf16_forward", &fused_rmsnorm_qkv_split_bf16_forward);
    m.def("fused_thingemm_residual_bf16_forward", &fused_thingemm_residual_bf16_forward);
    m.def("fused_rmsnorm_thingemm_bf16_forward", &fused_rmsnorm_thingemm_bf16_forward);
    m.def("fused_silu_gate_mul_strided_bf16_forward", &fused_silu_gate_mul_strided_bf16_forward);
    m.def("fused_rmsnorm_lmhead_bf16_forward", &fused_rmsnorm_lmhead_bf16_forward);
}
