/*
 * qwen3_kernels.cu — Persistent megakernel for Qwen3-1.7B single-token decode.
 *
 * Architecture follows the AlpinDale qwen_megakernel pattern:
 *   - Qwen3LayerWeights struct (like LDGLayerWeights) bundles all per-layer ptrs
 *   - AtomicGridSync replaces cooperative grid barriers
 *   - qwen3_decode_persistent loops over all 28 layers in one launch:
 *       per layer: QKV+RoPE → GQA attention → O-proj+MLP
 *   - rms_lm_head at the end of the persistent kernel
 *
 * All per-op logic lives in __device__ functions (no additional __global__ wrappers
 * except the one persistent kernel entry point).
 *
 * Compile via torch.utils.cpp_extension.load() with:
 *   extra_include_paths=["path/to/kernels", "path/to/profiler"]
 *   extra_cuda_cflags=["-std=c++20", "-O2", "-arch=sm_XXX"]
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Device function headers
#include "qwen3.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"
#include "qkv_rope_append.cuh"
#include "oproj_residual.cuh"
#include "upgate_silu.cuh"
#include "downproj_residual.cuh"
#include "rms_lm_head.cuh"

// ============================================================
// AtomicGridSync — atomic barrier for persistent kernels
// (mirrors the AlpinDale qwen_megakernel pattern exactly)
// ============================================================

struct AtomicGridSync {
    unsigned int* counter;
    unsigned int* generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int* vgen = reinterpret_cast<volatile unsigned int*>(generation);
                while (*vgen <= my_gen) {
                }
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

// ============================================================
// Qwen3LayerWeights — per-layer weight bundle (like LDGLayerWeights)
// ============================================================

struct Qwen3LayerWeights {
    const float* attn_ln_w;   // [hidden_size]
    const float* qkv_w;       // [qkv_dim, hidden_size]
    const float* q_norm_w;    // [head_dim]
    const float* k_norm_w;    // [head_dim]
    const float* o_proj_w;    // [hidden_size, hidden_size]
    const float* mlp_ln_w;    // [hidden_size]
    const float* gate_w;      // [intermediate_size, hidden_size]
    const float* up_w;        // [intermediate_size, hidden_size]
    const float* down_proj_w; // [hidden_size, intermediate_size]
    float* k_cache;           // [max_seq, kv_dim]
    float* v_cache;           // [max_seq, kv_dim]
};

// ============================================================
// GQA attention device function for single-token decode
//
// Reads Q from global g_q [q_dim] and K/V from the KV cache
// [seq_len, kv_dim], writes result to attn_out [q_dim].
// Uses smem[0..WARP_SIZE) as scratch for block reductions.
// ============================================================

__device__ void attention_gqa_device(float* attn_out,      // [q_dim]
                                     const float* g_q,     // [q_dim]
                                     const float* k_cache, // [seq_len, kv_dim]
                                     const float* v_cache, // [seq_len, kv_dim]
                                     int seq_len, float scale) {
    constexpr int num_q_heads = QWEN3_1_7B.num_attention_heads;
    constexpr int num_kv_heads = QWEN3_1_7B.num_key_value_heads;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int kv_dim = num_kv_heads * head_dim;
    constexpr int gqa_ratio = QWEN3_1_7B.gqa_ratio();

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = num_threads / kernels::WARP_SIZE;

    using kernels::block_reduce_sum;

    // Reuse smem[0..WARP_SIZE) as reduction scratch — safe since previous
    // device function has completed and synced.
    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);

    for (int kv_h = 0; kv_h < num_kv_heads; kv_h++) {
        for (int g = 0; g < gqa_ratio; g++) {
            int q_h = kv_h * gqa_ratio + g;
            const float* q_head = g_q + q_h * head_dim;
            float* out_head = attn_out + q_h * head_dim;

            float m = -INFINITY; // running max
            float denom = 0.0f;  // running denominator
            int my_d = (tid < head_dim) ? tid : -1;
            float acc = 0.0f;

            for (int pos = 0; pos < seq_len; pos++) {
                const float* k_pos = k_cache + (long long)pos * kv_dim + kv_h * head_dim;
                const float* v_pos = v_cache + (long long)pos * kv_dim + kv_h * head_dim;

                // Dot product over head_dim — threads stride, then block reduce
                float thread_dot = 0.0f;
                for (int d = tid; d < head_dim; d += num_threads) {
                    thread_dot += q_head[d] * k_pos[d];
                }
                float score = block_reduce_sum(thread_dot, s_reduce, lane_id, warp_id, num_warps) * scale;

                // Online softmax update
                float m_new = fmaxf(m, score);
                float alpha = expf(m - m_new);
                float beta = expf(score - m_new);
                denom = denom * alpha + beta;
                m = m_new;

                if (my_d >= 0) {
                    acc = acc * alpha + beta * v_pos[my_d];
                }
                __syncthreads();
            }

            if (my_d >= 0) {
                out_head[my_d] = acc / denom;
            }
            __syncthreads();
        }
    }
}

// ============================================================
// Persistent decode kernel — one launch, all 28 layers
//
// Follows the AlpinDale ldg_decode_kernel_persistent pattern:
//   for each layer: QKV+RoPE → attention → O-proj+MLP
//   after all layers: RMSNorm + LM head
// ============================================================

__global__ void qwen3_decode_persistent(float* hidden,     // [hidden_size] — token hidden state, updated in-place
                                        float* g_q,        // [q_dim]       — scratch for Q (persists into attention)
                                        float* g_k,        // [kv_dim]      — scratch for K output (redundant w/ cache)
                                        float* g_v,        // [kv_dim]      — scratch for V output (redundant w/ cache)
                                        float* g_attn_out, // [q_dim]       — scratch for attention output
                                        float* g_silu,     // [intermediate_size] — scratch for SiLU*up
                                        const Qwen3LayerWeights* __restrict__ layer_weights, // [num_layers]
                                        const float* __restrict__ cos_cache,                 // [max_seq, head_dim]
                                        const float* __restrict__ sin_cache,                 // [max_seq, head_dim]
                                        int pos_id, int max_seq_len,
                                        const float* __restrict__ norm_w,    // [hidden_size]  — final RMSNorm
                                        const float* __restrict__ lm_head_w, // [vocab_size, hidden_size]
                                        int vocab_size,
                                        float* logits, // [vocab_size] — output
                                        unsigned int* sync_counter, unsigned int* sync_generation) {
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;
    constexpr float attn_scale = QWEN3_1_7B.attn_scale();

    AtomicGridSync grid_sync = {sync_counter, sync_generation, gridDim.x, 0};

    for (int layer = 0; layer < num_layers; layer++) {
        const Qwen3LayerWeights& w = layer_weights[layer];

        const float* cos_pos = cos_cache + (long long)pos_id * QWEN3_1_7B.head_dim;
        const float* sin_pos = sin_cache + (long long)pos_id * QWEN3_1_7B.head_dim;

        // --- Phase 1: QKV projection + Q/K RMSNorm + RoPE + KV cache write ---
        qkv_rope_append_device(g_q, g_k, g_v, w.k_cache, w.v_cache, hidden, w.attn_ln_w, w.qkv_w, w.q_norm_w,
                               w.k_norm_w, cos_pos, sin_pos, pos_id, max_seq_len, nullptr, nullptr);
        grid_sync.sync();

        // --- Phase 2: GQA attention (reads g_q + KV cache, writes g_attn_out) ---
        attention_gqa_device(g_attn_out, g_q, w.k_cache, w.v_cache, pos_id + 1, attn_scale);
        grid_sync.sync();

        // --- Phase 3: O-proj + residual + MLP ---
        oproj_residual_device(hidden, g_attn_out, w.o_proj_w, nullptr, nullptr);
        upgate_silu_device(g_silu, hidden, w.mlp_ln_w, w.gate_w, w.up_w, nullptr, nullptr);
        downproj_residual_device(hidden, g_silu, w.down_proj_w, nullptr, nullptr);
        grid_sync.sync();
    }

    // --- Final: RMSNorm + LM head projection ---
    rms_lm_head_device(logits, hidden, norm_w, lm_head_w, vocab_size, nullptr, nullptr);
}

// ============================================================
// PyTorch binding — persistent full-model decode
// ============================================================

torch::Tensor
qwen3_decode_persistent_forward(torch::Tensor hidden,       // [hidden_size]
                                torch::Tensor attn_ln_ws,   // [num_layers, hidden_size]
                                torch::Tensor qkv_weights,  // [num_layers, qkv_dim * hidden_size] (contiguous)
                                torch::Tensor q_norm_ws,    // [num_layers, head_dim]
                                torch::Tensor k_norm_ws,    // [num_layers, head_dim]
                                torch::Tensor cos_cache,    // [max_seq, head_dim]
                                torch::Tensor sin_cache,    // [max_seq, head_dim]
                                torch::Tensor k_caches,     // [num_layers, max_seq * kv_dim] (contiguous)
                                torch::Tensor v_caches,     // [num_layers, max_seq * kv_dim] (contiguous)
                                torch::Tensor o_proj_ws,    // [num_layers, hidden_size * hidden_size]
                                torch::Tensor mlp_ln_ws,    // [num_layers, hidden_size]
                                torch::Tensor gate_ws,      // [num_layers, intermediate_size * hidden_size]
                                torch::Tensor up_ws,        // [num_layers, intermediate_size * hidden_size]
                                torch::Tensor down_proj_ws, // [num_layers, hidden_size * intermediate_size]
                                torch::Tensor norm_w,       // [hidden_size]
                                torch::Tensor lm_head_w,    // [vocab_size, hidden_size]
                                int pos_id) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = QWEN3_1_7B.qkv_output_dim();
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int num_layers = QWEN3_1_7B.num_hidden_layers;

    // k_caches shape is [num_layers, max_seq * kv_dim]; size(1) = max_seq * kv_dim
    int max_seq_len = static_cast<int>(k_caches.size(1)) / kv_dim;
    int vocab_size = static_cast<int>(lm_head_w.size(0));

    // Build Qwen3LayerWeights host array, then copy to device
    std::vector<Qwen3LayerWeights> h_weights(num_layers);
    for (int l = 0; l < num_layers; l++) {
        h_weights[l].attn_ln_w = attn_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].qkv_w = qkv_weights.data_ptr<float>() + (long long)l * qkv_dim * hidden_size;
        h_weights[l].q_norm_w = q_norm_ws.data_ptr<float>() + (long long)l * head_dim;
        h_weights[l].k_norm_w = k_norm_ws.data_ptr<float>() + (long long)l * head_dim;
        h_weights[l].o_proj_w = o_proj_ws.data_ptr<float>() + (long long)l * hidden_size * hidden_size;
        h_weights[l].mlp_ln_w = mlp_ln_ws.data_ptr<float>() + (long long)l * hidden_size;
        h_weights[l].gate_w = gate_ws.data_ptr<float>() + (long long)l * intermediate_size * hidden_size;
        h_weights[l].up_w = up_ws.data_ptr<float>() + (long long)l * intermediate_size * hidden_size;
        h_weights[l].down_proj_w = down_proj_ws.data_ptr<float>() + (long long)l * hidden_size * intermediate_size;
        h_weights[l].k_cache = k_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
        h_weights[l].v_cache = v_caches.data_ptr<float>() + (long long)l * max_seq_len * kv_dim;
    }

    Qwen3LayerWeights* d_weights;
    cudaMalloc(&d_weights, num_layers * sizeof(Qwen3LayerWeights));
    cudaMemcpy(d_weights, h_weights.data(), num_layers * sizeof(Qwen3LayerWeights), cudaMemcpyHostToDevice);

    // Scratch buffers
    auto output_hidden = hidden.clone();
    auto g_q = torch::empty({q_dim}, hidden.options());
    auto g_k = torch::empty({kv_dim}, hidden.options());
    auto g_v = torch::empty({kv_dim}, hidden.options());
    auto g_attn_out = torch::empty({q_dim}, hidden.options());
    auto g_silu = torch::empty({intermediate_size}, hidden.options());
    auto logits = torch::empty({vocab_size}, hidden.options());

    // AtomicGridSync state
    auto sync_ctr = torch::zeros({1}, torch::dtype(torch::kInt32).device(hidden.device()));
    auto sync_gen = torch::zeros({1}, torch::dtype(torch::kInt32).device(hidden.device()));

    const int block_size = 256;
    const int grid_size = 1; // single block; AtomicGridSync ready for multi-block expansion
    // smem: max across all phases = QKV layout
    size_t smem_bytes = (hidden_size + qkv_dim + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    qwen3_decode_persistent<<<grid_size, block_size, smem_bytes>>>(
        output_hidden.data_ptr<float>(), g_q.data_ptr<float>(), g_k.data_ptr<float>(), g_v.data_ptr<float>(),
        g_attn_out.data_ptr<float>(), g_silu.data_ptr<float>(), d_weights, cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(), pos_id, max_seq_len, norm_w.data_ptr<float>(), lm_head_w.data_ptr<float>(),
        vocab_size, logits.data_ptr<float>(), reinterpret_cast<unsigned int*>(sync_ctr.data_ptr<int>()),
        reinterpret_cast<unsigned int*>(sync_gen.data_ptr<int>()));

    cudaFree(d_weights);
    return logits;
}

// ============================================================
// Standalone SiLU-Multiply (Rishu's vectorized implementation)
// ============================================================

torch::Tensor silu_multiply(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");
    auto output = torch::empty_like(gate);
    int N = gate.numel();
    int N_vec = N / 4;
    int N_tail = N % 4;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int blockSize = 256;

    if (N_vec > 0) {
        int numBlocks = (N_vec + blockSize - 1) / blockSize;
        kernels::silu_multiply_kernel<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(), N);
    }
    if (N_tail > 0) {
        int numBlocks = (N_tail + blockSize - 1) / blockSize;
        kernels::silu_multiply_kernel_tail<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(), N_vec * 4, N);
    }
    return output;
}

// ============================================================
// PYBIND11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Persistent full-model decode: one launch per token across all 28 layers
    m.def("qwen3_decode_persistent_forward", &qwen3_decode_persistent_forward,
          "Persistent decode: all layers in one kernel launch (CUDA)");

    // Standalone SiLU-Multiply (vectorized)
    m.def("silu_multiply", &silu_multiply, "Fused SiLU-multiply: output = SiLU(gate) * up (vectorized)");
}
