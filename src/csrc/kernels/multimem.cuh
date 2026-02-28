#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace multimem {

struct uint128_t {
    uint32_t x, y, z, w;
};

__device__ __forceinline__ uint128_t ld_reduce_add_bf16x8(const void* mc_ptr) {
    uint128_t r;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(mc_ptr)
        : "memory");
    return r;
}

__device__ __forceinline__ uint128_t ld_reduce_add_f16x8(const void* mc_ptr) {
    uint128_t r;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(mc_ptr)
        : "memory");
    return r;
}

__device__ __forceinline__ uint128_t ld_reduce_add_f32x4(const void* mc_ptr) {
    uint128_t r;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(mc_ptr)
        : "memory");
    return r;
}

__device__ __forceinline__ void st_b128(void* mc_ptr, uint128_t val) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(mc_ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
}

__device__ __forceinline__ void red_release_sys_add1(void* mc_ptr) {
    asm volatile(
        "multimem.red.release.sys.global.add.u32 [%0], 1;"
        :
        : "l"(mc_ptr)
        : "memory");
}

__device__ __forceinline__ void red_relaxed_gpu_add1(void* mc_ptr) {
    asm volatile(
        "multimem.red.relaxed.gpu.global.add.u32 [%0], 1;"
        :
        : "l"(mc_ptr)
        : "memory");
}

__device__ __forceinline__ void red_release_sys(void* mc_ptr, uint32_t val) {
    asm volatile(
        "red.release.sys.global.add.u32 [%0], %1;"
        :
        : "l"(mc_ptr), "r"(val)
        : "memory");
}

__device__ __forceinline__ void red_relaxed_sys(void* mc_ptr, uint32_t val) {
    asm volatile(
        "red.relaxed.sys.global.add.u32 [%0], %1;"
        :
        : "l"(mc_ptr), "r"(val)
        : "memory");
}

__device__ __forceinline__ void spin_wait_eq_reset(uint32_t* flag_ptr, uint32_t expected, uint32_t reset) {
    uint32_t result = 0;
    while (result != expected) {
        asm volatile(
            "atom.relaxed.sys.global.cas.b32 %0, [%1], %2, %3;"
            : "=r"(result)
            : "l"(flag_ptr), "r"(expected), "r"(reset)
            : "memory");
    }
}

__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void fence_acq_rel_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

}  // namespace multimem
