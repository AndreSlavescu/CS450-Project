#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include "gpu_profiler.cuh"
#include "qwen3.cuh"
#include "utils.cuh"
using kernels::WARP_SIZE;

// Profiler event IDs
enum : int {
    EV_QK = 0,
    EV_QK_END = 1,
    EV_SOFTMAX_AV = 2,
    EV_SOFTMAX_AV_END = 3,
};

/*
 * Partial attention for a single KV head with GQA.
 *
 * For one KV head, processes QWEN3_1_7B.gqa_ratio()=2 Q heads against the KV cache
 * using online softmax (FlashAttention-style numerically stable).
 *
 * Grid: 1 block per KV head (or partial)
 * Block: 256 threads
 *
 * Inputs:
 *   q:       [QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim] = [2, 128] — the query vectors for this KV head group
 *   k_cache: [seq_len, QWEN3_1_7B.head_dim]   — K cache for this KV head
 *   v_cache: [seq_len, QWEN3_1_7B.head_dim]   — V cache for this KV head
 *
 * Outputs:
 *   out: [QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim] = [2, 128] — attention output
 *   lse: [QWEN3_1_7B.gqa_ratio()]           = [2]      — log-sum-exp for reduction
 */
__global__ void attention_partial_kernel(float* __restrict__ out,     // [QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim]
                                         float* __restrict__ lse_out, // [QWEN3_1_7B.gqa_ratio()]
                                         const float* __restrict__ q, // [QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim]
                                         const float* __restrict__ k_cache, // [seq_len, QWEN3_1_7B.head_dim]
                                         const float* __restrict__ v_cache, // [seq_len, QWEN3_1_7B.head_dim]
                                         int seq_len,
                                         float scale, // 1/sqrt(QWEN3_1_7B.head_dim)
                                         profiler::event_record* g_events, int* g_counts) {
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);
    profiler::block_state* prof = reinterpret_cast<profiler::block_state*>(s_reduce + WARP_SIZE);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = num_threads / WARP_SIZE;

    if (tid == 0 && has_profiler)
        prof->init();
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_QK);

    // Online softmax attention for each Q head
    // For each Q head h in [0, QWEN3_1_7B.gqa_ratio()):
    //   Process KV cache entries one by one (or in blocks)
    //   Maintain running max, sum_exp, and weighted V accumulator
    for (int h = 0; h < QWEN3_1_7B.gqa_ratio(); h++) {
        const float* q_h = q + h * QWEN3_1_7B.head_dim;
        float* out_h = out + h * QWEN3_1_7B.head_dim;

        float running_max = -INFINITY;
        float running_sum_exp = 0.0f;

        // Initialize output accumulator in registers
        // Each thread handles QWEN3_1_7B.head_dim/num_threads elements
        float acc[4]; // QWEN3_1_7B.head_dim/num_threads = 128/256 < 1, so we need different parallelization
        // Actually with 256 threads and 128 elements, we have more threads than elements.
        // Let's have each thread handle at most 1 element of the output.
        // But we also need to parallelize over seq_len.

        // Strategy: parallelize dot product over threads, serialize over seq_len positions
        // Each position: compute score = q @ k_t * scale, then online softmax update

        // For accumulator: each thread maintains its own subset of the output vector
        float local_acc[1]; // each thread handles 1 or 0 output elements
        int my_out_idx = tid < QWEN3_1_7B.head_dim ? tid : -1;
        if (my_out_idx >= 0)
            local_acc[0] = 0.0f;

        for (int pos = 0; pos < seq_len; pos++) {
            const float* k_pos = k_cache + pos * QWEN3_1_7B.head_dim;
            const float* v_pos = v_cache + pos * QWEN3_1_7B.head_dim;

            // Compute dot product: score = q_h . k_pos
            float thread_dot = 0.0f;
            for (int d = tid; d < QWEN3_1_7B.head_dim; d += num_threads) {
                thread_dot += q_h[d] * k_pos[d];
            }

            // Reduce dot product across block
            namespace cg = cooperative_groups;
            cg::thread_block block = cg::this_thread_block();
            cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

            float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>{});
            if (lane_id == 0)
                s_reduce[warp_id] = warp_dot;
            __syncthreads();

            warp_dot = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
            float score = cg::reduce(warp, warp_dot, cg::plus<float>{}) * scale;

            // Online softmax update
            float old_max = running_max;
            running_max = fmaxf(running_max, score);

            float exp_diff = expf(old_max - running_max);
            float exp_score = expf(score - running_max);

            running_sum_exp = running_sum_exp * exp_diff + exp_score;

            // Update accumulator: scale old acc, add new v contribution
            if (my_out_idx >= 0) {
                local_acc[0] = local_acc[0] * exp_diff + exp_score * v_pos[my_out_idx];
            }

            __syncthreads();
        }

        // Finalize: divide by sum_exp
        if (my_out_idx >= 0) {
            out_h[my_out_idx] = local_acc[0] / running_sum_exp;
        }

        // Store LSE (log-sum-exp) = running_max + log(running_sum_exp)
        if (tid == 0) {
            lse_out[h] = running_max + logf(running_sum_exp);
        }
        __syncthreads();
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_QK_END);

    // Flush profiler
    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// ============================================================
// PyTorch bindings
// ============================================================

std::vector<torch::Tensor> attention_partial_forward(torch::Tensor q, // [QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim]
                                                     torch::Tensor k_cache, // [seq_len, QWEN3_1_7B.head_dim]
                                                     torch::Tensor v_cache, // [seq_len, QWEN3_1_7B.head_dim]
                                                     int seq_len, float scale) {
    auto out = torch::empty({QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim}, q.options());
    auto lse = torch::empty({QWEN3_1_7B.gqa_ratio()}, q.options());

    const int block_size = 256;
    size_t smem_bytes = WARP_SIZE * sizeof(float) + sizeof(profiler::block_state);

    attention_partial_kernel<<<1, block_size, smem_bytes>>>(
        out.data_ptr<float>(), lse.data_ptr<float>(), q.data_ptr<float>(), k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(), seq_len, scale, nullptr, nullptr);

    return {out, lse};
}

std::vector<torch::Tensor> attention_partial_forward_profiled(torch::Tensor q, torch::Tensor k_cache,
                                                              torch::Tensor v_cache, int seq_len, float scale,
                                                              const std::string& trace_path) {
    auto out = torch::empty({QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim}, q.options());
    auto lse = torch::empty({QWEN3_1_7B.gqa_ratio()}, q.options());

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = WARP_SIZE * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    attention_partial_kernel<<<grid_size, block_size, smem_bytes>>>(
        out.data_ptr<float>(), lse.data_ptr<float>(), q.data_ptr<float>(), k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(), seq_len, scale, prof_buf.d_events, prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_QK, "qk_softmax_av");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_partial_forward", &attention_partial_forward, "Partial attention (CUDA)");
    m.def("attention_partial_forward_profiled", &attention_partial_forward_profiled,
          "Partial attention with profiling (CUDA)");
}
