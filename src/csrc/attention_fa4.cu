/**
 * CS450 Project - Waterloo Team
 * FlashAttention-style decode attention for Qwen3 GQA.
 *
 * Improvements over attention_partial.cu baseline:
 * - Both Q heads processed in parallel (vs sequential loop)
 * - Warp-local dot products using shuffle (no cross-warp reduction for scores)
 * - 4 KV positions per sync point (4x fewer barriers)
 * - Float4 vectorized loads for K and V
 * - All 256 threads active (no idle threads)
 *
 * Thread mapping (256 threads = 8 warps):
 *   Warps 0-3: head 0, each warp computes one KV score per batch
 *   Warps 4-7: head 1, each warp computes one KV score per batch
 *   Per warp: 32 threads x 4 elements = 128 = HEAD_DIM
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "gpu_profiler.cuh"
#include "qwen3_dims.cuh"

// Number of KV positions processed between __syncthreads() calls
// = number of warps per head (each warp computes one score)
static constexpr int POS_PER_BATCH = 4;

// Profiler event IDs
enum : int {
    EV_ATTN       = 0,
    EV_ATTN_END   = 1,
};

__global__ void attention_fa4_kernel(
    float* __restrict__ out,             // [GQA_RATIO, HEAD_DIM]
    float* __restrict__ lse_out,         // [GQA_RATIO]
    const float* __restrict__ q,         // [GQA_RATIO, HEAD_DIM]
    const float* __restrict__ k_cache,   // [seq_len, HEAD_DIM]
    const float* __restrict__ v_cache,   // [seq_len, HEAD_DIM]
    int seq_len,
    float scale,                         // 1/sqrt(HEAD_DIM)
    profiler::event_record* g_events,
    int* g_counts
) {
    bool has_profiler = (g_events != nullptr);

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;           // 0-7
    const int lane = tid & 31;              // 0-31
    const int head = warp_id >> 2;          // 0 or 1 (which GQA head)
    const int warp_in_head = warp_id & 3;   // 0-3 (which warp within the head)
    const int dim_base = lane << 2;         // lane * 4: base output dim (0,4,8,...,124)

    // Shared memory layout: 8 floats for score exchange + profiler state
    extern __shared__ char smem[];
    float* s_scores = (float*)smem;
    profiler::block_state* prof = (profiler::block_state*)(s_scores + 8);

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();
    if (tid == 0 && has_profiler) prof->record(EV_ATTN);

    // Load Q into registers: 4 values per thread, pre-multiplied by scale
    float4 q4 = *reinterpret_cast<const float4*>(q + head * HEAD_DIM + dim_base);
    float q0 = q4.x * scale;
    float q1 = q4.y * scale;
    float q2 = q4.z * scale;
    float q3 = q4.w * scale;

    // Per-thread output accumulators and online softmax state
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Process KV cache in batches of POS_PER_BATCH positions
    for (int base_pos = 0; base_pos < seq_len; base_pos += POS_PER_BATCH) {

        // ---- Step 1: Each warp computes one dot product (warp-local) ----
        int my_pos = base_pos + warp_in_head;
        float score;

        if (my_pos < seq_len) {
            // Float4 vectorized K load (128-bit, coalesced across warp)
            float4 k4 = *reinterpret_cast<const float4*>(k_cache + my_pos * HEAD_DIM + dim_base);
            float dot = q0 * k4.x + q1 * k4.y + q2 * k4.z + q3 * k4.w;

            // Warp shuffle reduction: 32 partial sums → 1 score (no shared memory!)
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            // Broadcast result from lane 0 to all lanes
            score = __shfl_sync(0xffffffff, dot, 0);
        } else {
            score = -INFINITY;  // padding for out-of-bounds positions
        }

        // ---- Step 2: Exchange scores across warps of same head ----
        if (lane == 0) s_scores[warp_id] = score;
        __syncthreads();

        float s0 = s_scores[head * POS_PER_BATCH + 0];
        float s1 = s_scores[head * POS_PER_BATCH + 1];
        float s2 = s_scores[head * POS_PER_BATCH + 2];
        float s3 = s_scores[head * POS_PER_BATCH + 3];

        // ---- Step 3: Online softmax + V accumulation for 4 positions ----

        // Position 0
        if (base_pos < seq_len) {
            float old_max = running_max;
            running_max = fmaxf(running_max, s0);
            float ed = expf(old_max - running_max);
            float es = expf(s0 - running_max);
            running_sum = running_sum * ed + es;
            float4 v4 = *reinterpret_cast<const float4*>(v_cache + base_pos * HEAD_DIM + dim_base);
            acc0 = acc0 * ed + es * v4.x;
            acc1 = acc1 * ed + es * v4.y;
            acc2 = acc2 * ed + es * v4.z;
            acc3 = acc3 * ed + es * v4.w;
        }

        // Position 1
        if (base_pos + 1 < seq_len) {
            float old_max = running_max;
            running_max = fmaxf(running_max, s1);
            float ed = expf(old_max - running_max);
            float es = expf(s1 - running_max);
            running_sum = running_sum * ed + es;
            float4 v4 = *reinterpret_cast<const float4*>(v_cache + (base_pos + 1) * HEAD_DIM + dim_base);
            acc0 = acc0 * ed + es * v4.x;
            acc1 = acc1 * ed + es * v4.y;
            acc2 = acc2 * ed + es * v4.z;
            acc3 = acc3 * ed + es * v4.w;
        }

        // Position 2
        if (base_pos + 2 < seq_len) {
            float old_max = running_max;
            running_max = fmaxf(running_max, s2);
            float ed = expf(old_max - running_max);
            float es = expf(s2 - running_max);
            running_sum = running_sum * ed + es;
            float4 v4 = *reinterpret_cast<const float4*>(v_cache + (base_pos + 2) * HEAD_DIM + dim_base);
            acc0 = acc0 * ed + es * v4.x;
            acc1 = acc1 * ed + es * v4.y;
            acc2 = acc2 * ed + es * v4.z;
            acc3 = acc3 * ed + es * v4.w;
        }

        // Position 3
        if (base_pos + 3 < seq_len) {
            float old_max = running_max;
            running_max = fmaxf(running_max, s3);
            float ed = expf(old_max - running_max);
            float es = expf(s3 - running_max);
            running_sum = running_sum * ed + es;
            float4 v4 = *reinterpret_cast<const float4*>(v_cache + (base_pos + 3) * HEAD_DIM + dim_base);
            acc0 = acc0 * ed + es * v4.x;
            acc1 = acc1 * ed + es * v4.y;
            acc2 = acc2 * ed + es * v4.z;
            acc3 = acc3 * ed + es * v4.w;
        }

        __syncthreads();
    }

    // ---- Finalize: divide by softmax denominator ----
    float inv_sum = 1.0f / running_sum;
    *reinterpret_cast<float4*>(out + head * HEAD_DIM + dim_base) =
        make_float4(acc0 * inv_sum, acc1 * inv_sum, acc2 * inv_sum, acc3 * inv_sum);

    // Store LSE (one thread per head)
    if (warp_in_head == 0 && lane == 0) {
        lse_out[head] = running_max + logf(running_sum);
    }

    if (tid == 0 && has_profiler) prof->record(EV_ATTN_END);
    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// ============================================================
// PyTorch bindings
// ============================================================

std::vector<torch::Tensor> attention_partial_forward(
    torch::Tensor q,         // [GQA_RATIO, HEAD_DIM]
    torch::Tensor k_cache,   // [seq_len, HEAD_DIM]
    torch::Tensor v_cache,   // [seq_len, HEAD_DIM]
    int seq_len,
    float scale
) {
    auto out = torch::empty({GQA_RATIO, HEAD_DIM}, q.options());
    auto lse = torch::empty({GQA_RATIO}, q.options());

    const int block_size = 256;
    size_t smem_bytes = 8 * sizeof(float) + sizeof(profiler::block_state);

    attention_fa4_kernel<<<1, block_size, smem_bytes>>>(
        out.data_ptr<float>(),
        lse.data_ptr<float>(),
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        seq_len,
        scale,
        nullptr, nullptr
    );

    return {out, lse};
}

std::vector<torch::Tensor> attention_partial_forward_profiled(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int seq_len,
    float scale,
    const std::string& trace_path
) {
    auto out = torch::empty({GQA_RATIO, HEAD_DIM}, q.options());
    auto lse = torch::empty({GQA_RATIO}, q.options());

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = 8 * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    attention_fa4_kernel<<<grid_size, block_size, smem_bytes>>>(
        out.data_ptr<float>(),
        lse.data_ptr<float>(),
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        seq_len,
        scale,
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_ATTN, "fa4_attention");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_partial_forward", &attention_partial_forward,
          "FA4 decode attention (CUDA)");
    m.def("attention_partial_forward_profiled", &attention_partial_forward_profiled,
          "FA4 decode attention with profiling (CUDA)");
}
