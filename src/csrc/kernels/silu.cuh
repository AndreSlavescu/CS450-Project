#pragma once

#include <cuda_runtime.h>

/*
 * Fused SiLU-Multiply: output = SiLU(gate) * up
 *
 * SiLU(x) = x / (1 + exp(-x))
 *
 * Vectorized float4 implementation with scalar tail handling.
 * Based on Rishabh Sharma's implementation (PR #10).
 */

namespace kernels {

// Vectorized kernel: processes 4 elements at a time via float4
__global__ void silu_multiply_kernel(float* output, const float* __restrict__ gate, const float* __restrict__ up,
                                     int N) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = vec_idx * 4;

    if (idx + 3 < N) {
        float4 g = reinterpret_cast<const float4*>(gate)[vec_idx];
        float4 u = reinterpret_cast<const float4*>(up)[vec_idx];

        float4 result;
        result.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + expf(-g.w))) * u.w;

        reinterpret_cast<float4*>(output)[vec_idx] = result;
    }
}

// Scalar tail kernel: handles remaining elements not divisible by 4
__global__ void silu_multiply_kernel_tail(float* output, const float* __restrict__ gate, const float* __restrict__ up,
                                          int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g = gate[idx];
        float u = up[idx];
        output[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

// Device-level fused SiLU for use within a kernel (scalar, per-element)
__device__ __forceinline__ float silu_multiply_scalar(float gate_val, float up_val) {
    return (gate_val / (1.0f + expf(-gate_val))) * up_val;
}

} // namespace kernels
