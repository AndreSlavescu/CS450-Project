#pragma once

#include <cuda_runtime.h>

/*
 * SiLU device functions — fast approximation using PTX intrinsics.
 *   fast_silu(x) = x * rcp.approx(1 + ex2.approx(x * log2(e) * -1))
 *
 * This header is kernel-free: only __device__ functions live here.
 * Global launch wrappers (vectorized + tail) are in qwen3_kernels.cu.
 * Based on Rishabh Sharma's implementation (PR #10).
 */

namespace kernels {

// ============================================================
// PTX approximate primitives
// ============================================================

// Approximate reciprocal, flushes denormals to zero.
// Maps to a single rcp.approx.ftz.f32 instruction (~4 cycles vs ~20 for div).
__device__ __forceinline__ float approx_reciprocal(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Approximate base-2 exponential, flushes denormals to zero.
// Maps to a single ex2.approx.ftz.f32 instruction.
__device__ __forceinline__ float fast_exp2_approx(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Fast natural exponential via base-2 conversion: e^x = 2^(x * log2(e)).
// Avoids the slow expf() transcendental; max relative error ~1.7e-7.
__device__ __forceinline__ float fast_exp_approx(float x) {
    constexpr float log_2_e = 1.44269504088896340736f;
    return fast_exp2_approx(x * log_2_e);
}

// ============================================================
// SiLU device functions (scalar + vectorized overloads)
// ============================================================

// Fast SiLU: x * rcp(1 + exp(-x)), using PTX approx instructions.
__device__ __forceinline__ float fast_silu(float x) {
    return x * approx_reciprocal(1.0f + fast_exp_approx(-x));
}

// Scalar fused SiLU-multiply.
// Used by upgate_silu.cuh and the tail global kernel in qwen3_kernels.cu.
__device__ __forceinline__ float silu_multiply(float gate_val, float up_val) {
    return fast_silu(gate_val) * up_val;
}

// Vectorized fused SiLU-multiply (float4).
// Used by the vectorized global kernel in qwen3_kernels.cu.
__device__ __forceinline__ float4 silu_multiply(float4 gate, float4 up) {
    return make_float4(silu_multiply(gate.x, up.x), silu_multiply(gate.y, up.y), silu_multiply(gate.z, up.z),
                       silu_multiply(gate.w, up.w));
}

} // namespace kernels
