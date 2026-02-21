/*
 * qwen3_kernels.cu — Persistent megakernel for Qwen3-1.7B single-token decode.
 *
 * Architecture follows the AlpinDale qwen_megakernel pattern:
 *   - Qwen3LayerWeights struct (like LDGLayerWeights) bundles all per-layer ptrs
 *   - AtomicGridSync replaces cooperative grid barriers
 *   - qwen3_decode_persistent loops over all 28 layers in one launch:
 *       per layer: QKV+RoPE → flash-decode GQA attention → O-proj+MLP
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
// Flash-decoding GQA attention — single-token decode
//
// Splits KV positions across num_warps warp groups. Each warp
// independently computes partial (m, d, O) with warp-local online
// softmax (warp shuffle reduces, zero cross-warp syncs in the hot
// loop). After all KV positions are processed, warp 0 merges the
// num_warps partial results via the standard log-sum-exp trick.
//
// Complexity: O(NUM_Q_HEADS) __syncthreads()
//   vs        O(NUM_Q_HEADS * seq_len) for the naive block-reduce.
//
// smem layout (reuses the block's existing allocation):
//   s_o: [num_warps][HEAD_DIM] float   — partial output
//   s_m: [num_warps]           float   — partial row-max
//   s_d: [num_warps]           float   — partial denominator
//   Total: 8 * (128+2) * 4 = 4160 bytes  (<< QKV-phase smem ~24 KB)
// ============================================================

__device__ void flash_decode_gqa_device(float* attn_out,      // [q_dim]
                                        const float* g_q,     // [q_dim]
                                        const float* k_cache, // [max_seq, kv_dim]
                                        const float* v_cache, // [max_seq, kv_dim]
                                        int seq_len, float scale) {
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;   // 16
    constexpr int NUM_KV_HEADS = QWEN3_1_7B.num_key_value_heads;  // 8
    constexpr int HEAD_DIM = QWEN3_1_7B.head_dim;                 // 128
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;               // 1024
    constexpr int GQA_RATIO = QWEN3_1_7B.gqa_ratio();             // 2
    constexpr int ELEMS_PER_LANE = HEAD_DIM / kernels::WARP_SIZE; // 4

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5; // 8

    extern __shared__ char smem_raw[];
    float* s_o = reinterpret_cast<float*>(smem_raw); // [num_warps * HEAD_DIM]
    float* s_m = s_o + num_warps * HEAD_DIM;         // [num_warps]
    float* s_d = s_m + num_warps;                    // [num_warps]

    // Split KV sequence evenly across warps
    const int chunk = (seq_len + num_warps - 1) / num_warps;
    const int kv_start = warp * chunk;
    const int kv_end = min(kv_start + chunk, seq_len);

    for (int q_h = 0; q_h < NUM_Q_HEADS; q_h++) {
        const int kv_h = q_h / GQA_RATIO;
        const float* q_head = g_q + q_h * HEAD_DIM;

        // Warp-local softmax state
        float m_w = -INFINITY;
        float d_w = 0.0f;
        float o_w[ELEMS_PER_LANE] = {};

        // Hot loop: no cross-warp syncs — warp-internal shuffle only
        for (int pos = kv_start; pos < kv_end; pos++) {
            const float* k = k_cache + (long long)pos * KV_DIM + kv_h * HEAD_DIM;
            const float* v = v_cache + (long long)pos * KV_DIM + kv_h * HEAD_DIM;

            // QK dot: each lane sums ELEMS_PER_LANE terms, then warp-reduce
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

            // Warp-local online softmax update
            const float m_new = fmaxf(m_w, score);
            const float alpha = expf(m_w - m_new);
            const float beta = expf(score - m_new);
            d_w = d_w * alpha + beta;
            m_w = m_new;

            // Accumulate V: lane l owns dims [l*4 .. l*4+3] (coalesced load)
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++) {
                o_w[i] = o_w[i] * alpha + beta * v[lane * ELEMS_PER_LANE + i];
            }
        }

        // Store warp partial to smem (coalesced: lane l writes dims [l*4..l*4+3])
#pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            s_o[warp * HEAD_DIM + lane * ELEMS_PER_LANE + i] = o_w[i];
        }
        if (lane == 0) {
            s_m[warp] = m_w;
            s_d[warp] = d_w;
        }
        __syncthreads();

        // Warp 0: merge all partial (m, d, O) results via log-sum-exp
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
                const float a1 = expf(m_acc - m_new2);
                const float a2 = expf(m_w2 - m_new2);
                d_acc = d_acc * a1 + d_w2 * a2;
                m_acc = m_new2;
#pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++) {
                    o_acc[i] = o_acc[i] * a1 + s_o[w * HEAD_DIM + lane * ELEMS_PER_LANE + i] * a2;
                }
            }

            // Normalize and write output (coalesced: lane l writes dims [l*4..l*4+3])
            const float inv_d = 1.0f / d_acc;
            float* out = attn_out + q_h * HEAD_DIM;
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++) {
                out[lane * ELEMS_PER_LANE + i] = o_acc[i] * inv_d;
            }
        }
        __syncthreads();
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

        // --- Phase 2: Flash-decode GQA attention (reads g_q + KV cache, writes g_attn_out) ---
        flash_decode_gqa_device(g_attn_out, g_q, w.k_cache, w.v_cache, pos_id + 1, attn_scale);
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
    // smem: max across all phases = QKV layout (~24 KB)
    // flash-decode attention needs num_warps * (head_dim + 2) * 4 = 4160 B — well within budget
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
// SiLU-Multiply global kernels (private to this TU)
// Call the device-function overloads in silu.cuh.
// ============================================================

namespace {
__global__ void silu_multiply_vec_kernel(float* out, const float* __restrict__ gate, const float* __restrict__ up,
                                         int N) {
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi * 4 + 3 < N) {
        float4 g = reinterpret_cast<const float4*>(gate)[vi];
        float4 u = reinterpret_cast<const float4*>(up)[vi];
        reinterpret_cast<float4*>(out)[vi] = kernels::silu_multiply(g, u);
    }
}

__global__ void silu_multiply_tail_kernel(float* out, const float* __restrict__ gate, const float* __restrict__ up,
                                          int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = kernels::silu_multiply(gate[idx], up[idx]);
    }
}
} // namespace

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
        silu_multiply_vec_kernel<<<numBlocks, blockSize, 0, stream>>>(output.data_ptr<float>(), gate.data_ptr<float>(),
                                                                      up.data_ptr<float>(), N);
    }
    if (N_tail > 0) {
        int numBlocks = (N_tail + blockSize - 1) / blockSize;
        silu_multiply_tail_kernel<<<numBlocks, blockSize, 0, stream>>>(output.data_ptr<float>(), gate.data_ptr<float>(),
                                                                       up.data_ptr<float>(), N_vec * 4, N);
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
