#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "gpu_profiler.cuh"
#include "qwen3.cuh"
#include "utils.cuh"

// Profiler event IDs
enum : int {
    EV_REDUCE = 0,
    EV_REDUCE_END = 1,
};

/*
 * Attention reduction: combine partial attention outputs using LSE-weighted combination.
 *
 * Given N partial results (out_i, lse_i), compute the final output:
 *   final_out = sum_i(exp(lse_i - global_lse) * out_i)
 *   where global_lse = log(sum_i(exp(lse_i)))
 *
 * This is the standard FlashAttention reduction for split-KV attention.
 *
 * Grid: 1 block
 * Block: 256 threads
 */
__global__ void attention_reduction_kernel(
    float* __restrict__ final_out,          // [QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim]
    const float* __restrict__ partial_outs, // [num_partials, QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim]
    const float* __restrict__ partial_lses, // [num_partials, QWEN3_1_7B.gqa_ratio()]
    int num_partials, profiler::event_record* g_events, int* g_counts) {
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    profiler::block_state* prof = reinterpret_cast<profiler::block_state*>(smem);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler)
        prof->init();
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_REDUCE);

    for (int h = 0; h < QWEN3_1_7B.gqa_ratio(); h++) {
        // Find global max LSE for numerical stability
        float max_lse = -INFINITY;
        for (int p = 0; p < num_partials; p++) {
            max_lse = fmaxf(max_lse, partial_lses[p * QWEN3_1_7B.gqa_ratio() + h]);
        }

        // Compute exp(lse_i - max_lse) weights and sum
        // Then weighted-average the partial outputs
        for (int d = tid; d < QWEN3_1_7B.head_dim; d += num_threads) {
            float acc = 0.0f;
            float weight_sum = 0.0f;

            for (int p = 0; p < num_partials; p++) {
                float lse = partial_lses[p * QWEN3_1_7B.gqa_ratio() + h];
                float w = expf(lse - max_lse);
                weight_sum += w;
                acc += w * partial_outs[p * QWEN3_1_7B.gqa_ratio() * QWEN3_1_7B.head_dim + h * QWEN3_1_7B.head_dim + d];
            }

            final_out[h * QWEN3_1_7B.head_dim + d] = acc / weight_sum;
        }

        __syncthreads();
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_REDUCE_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor
attention_reduction_forward(torch::Tensor partial_outs, // [num_partials, QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim]
                            torch::Tensor partial_lses, // [num_partials, QWEN3_1_7B.gqa_ratio()]
                            int num_partials) {
    auto final_out = torch::empty({QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim}, partial_outs.options());

    const int block_size = 256;
    size_t smem_bytes = sizeof(profiler::block_state);

    attention_reduction_kernel<<<1, block_size, smem_bytes>>>(
        final_out.data_ptr<float>(), partial_outs.data_ptr<float>(), partial_lses.data_ptr<float>(), num_partials,
        nullptr, nullptr);

    return final_out;
}

torch::Tensor attention_reduction_forward_profiled(torch::Tensor partial_outs, torch::Tensor partial_lses,
                                                   int num_partials, const std::string& trace_path) {
    auto final_out = torch::empty({QWEN3_1_7B.gqa_ratio(), QWEN3_1_7B.head_dim}, partial_outs.options());

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    attention_reduction_kernel<<<grid_size, block_size, smem_bytes>>>(
        final_out.data_ptr<float>(), partial_outs.data_ptr<float>(), partial_lses.data_ptr<float>(), num_partials,
        prof_buf.d_events, prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_REDUCE, "attn_reduce");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_reduction_forward", &attention_reduction_forward, "Attention reduction (CUDA)");
    m.def("attention_reduction_forward_profiled", &attention_reduction_forward_profiled,
          "Attention reduction with profiling (CUDA)");
}
