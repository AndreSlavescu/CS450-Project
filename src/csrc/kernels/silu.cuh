#pragma once

#include <cuda_runtime.h>

namespace kernels {

__device__ __forceinline__ float approx_reciprocal(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_exp2_approx(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_exp_approx(float x) {
    constexpr float log_2_e = 1.44269504088896340736f;
    return fast_exp2_approx(x * log_2_e);
}

__device__ __forceinline__ float fast_silu(float x) {
    return x * approx_reciprocal(1.0f + fast_exp_approx(-x));
}

__device__ __forceinline__ float silu_multiply(float gate_val, float up_val) {
    return fast_silu(gate_val) * up_val;
}

__device__ __forceinline__ float4 silu_multiply(float4 gate, float4 up) {
    return make_float4(silu_multiply(gate.x, up.x), silu_multiply(gate.y, up.y), silu_multiply(gate.z, up.z),
                       silu_multiply(gate.w, up.w));
}

} // namespace kernels
