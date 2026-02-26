#pragma once

#include "qwen3.cuh"
#include "utils.cuh"

namespace kernels {

__device__ __forceinline__ void rmsnorm(float* output, const float* input, const float* weight, int dim,
                                        float* shared_reduce, int tid, int num_threads, int lane_id, int warp_id,
                                        int num_warps) {
    float thread_ss = 0.0f;
    for (int i = tid; i < dim; i += 2 * num_threads) {
        float x0 = input[i];
        thread_ss += x0 * x0;
        int j = i + num_threads;
        if (j < dim) {
            float x1 = input[j];
            thread_ss += x1 * x1;
        }
    }

    float total_ss = block_reduce_sum(thread_ss, shared_reduce, lane_id, warp_id, num_warps);
    float rms = rsqrtf(total_ss / dim + QWEN3_1_7B.rms_norm_eps);

    for (int i = tid; i < dim; i += 2 * num_threads) {
        output[i] = input[i] * rms * weight[i];
        int j = i + num_threads;
        if (j < dim)
            output[j] = input[j] * rms * weight[j];
    }
}

__device__ __forceinline__ void rmsnorm_per_head(float* data, const float* weight, int num_heads, int head_dim,
                                                 float* shared_reduce, int tid, int num_threads, int lane_id,
                                                 int warp_id, int num_warps) {
    for (int h = 0; h < num_heads; h++) {
        int offset = h * head_dim;

        float head_ss = 0.0f;
        for (int i = tid; i < head_dim; i += num_threads) {
            float val = data[offset + i];
            head_ss += val * val;
        }

        float total_head_ss = block_reduce_sum(head_ss, shared_reduce, lane_id, warp_id, num_warps);
        float head_rms = rsqrtf(total_head_ss / head_dim + QWEN3_1_7B.rms_norm_eps);

        for (int i = tid; i < head_dim; i += num_threads) {
            data[offset + i] *= head_rms * weight[i];
        }
        __syncthreads();
    }
}

} // namespace kernels
