#pragma once

#include "qwen3_dims.cuh"
#include "utils.cuh"

namespace kernels {

/*
 * RMSNorm device function.
 *
 * Computes: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
 *
 * Writes normalized result to `output` (may alias `input` if in-place is desired,
 * but caller must ensure no race — typically use shared memory as output).
 *
 * Requires `shared_reduce` to have at least WARP_SIZE floats of shared memory.
 * Caller must __syncthreads() after this returns before reading `output`.
 */
__device__ __forceinline__ void rmsnorm(
    float*       output,
    const float* input,
    const float* weight,
    int          dim,
    float*       shared_reduce,
    int          tid,
    int          num_threads,
    int          lane_id,
    int          warp_id,
    int          num_warps
) {
    float thread_ss = 0.0f;
    for (int i = tid; i < dim; i += num_threads) {
        float xi = input[i];
        thread_ss += xi * xi;
    }

    float total_ss = block_reduce_sum(thread_ss, shared_reduce, lane_id, warp_id, num_warps);
    float rms = rsqrtf(total_ss / dim + EPS);

    for (int i = tid; i < dim; i += num_threads) {
        output[i] = input[i] * rms * weight[i];
    }
}

/*
 * Per-head RMSNorm device function.
 *
 * Normalizes `num_heads` heads of `head_dim` elements each, starting at `data[0]`.
 * Applies `weight[0..head_dim-1]` (shared across all heads).
 * In-place: overwrites data.
 *
 * Caller must __syncthreads() between successive calls.
 */
__device__ __forceinline__ void rmsnorm_per_head(
    float*       data,
    const float* weight,
    int          num_heads,
    int          head_dim,
    float*       shared_reduce,
    int          tid,
    int          num_threads,
    int          lane_id,
    int          warp_id,
    int          num_warps
) {
    for (int h = 0; h < num_heads; h++) {
        int offset = h * head_dim;

        float head_ss = 0.0f;
        for (int i = tid; i < head_dim; i += num_threads) {
            float val = data[offset + i];
            head_ss += val * val;
        }

        float total_head_ss = block_reduce_sum(head_ss, shared_reduce, lane_id, warp_id, num_warps);
        float head_rms = rsqrtf(total_head_ss / head_dim + EPS);

        for (int i = tid; i < head_dim; i += num_threads) {
            data[offset + i] *= head_rms * weight[i];
        }
        __syncthreads();
    }
}

} // namespace kernels
