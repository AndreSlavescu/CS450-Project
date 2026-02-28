#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "multimem.cuh"

#define CUDA_CHECK(cmd)                                                                \
    do {                                                                               \
        cudaError_t e = cmd;                                                           \
        if (e != cudaSuccess) {                                                        \
            printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__,                      \
                   cudaGetErrorString(e));                                              \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

constexpr int MULTIMEM_BF16_ELEMS_PER_OP = 8;
constexpr int MULTIMEM_BYTES_PER_OP = 16;

__global__ void multimem_allreduce_kernel(void* __restrict__ mc_ptr,
                                          uint32_t* __restrict__ flag_mc_ptr,
                                          uint32_t* __restrict__ flag_local_ptr,
                                          int num_elems, int local_rank, int world_size) {
    int num_ops = num_elems / MULTIMEM_BF16_ELEMS_PER_OP;
    int total_threads = gridDim.x * blockDim.x;
    char* base = static_cast<char*>(mc_ptr);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_ops; i += total_threads) {
        void* addr = base + static_cast<size_t>(i) * MULTIMEM_BYTES_PER_OP;
        multimem::uint128_t val = multimem::ld_reduce_add_bf16x8(addr);
        multimem::st_b128(addr, val);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        multimem::red_release_sys_add1(flag_mc_ptr + blockIdx.x);
        multimem::spin_wait_eq_reset(
            flag_local_ptr + blockIdx.x,
            static_cast<uint32_t>(world_size), 0u);
    }
}

__global__ void multimem_reduce_scatter_kernel(void* __restrict__ mc_ptr,
                                               void* __restrict__ dst_local,
                                               uint32_t* __restrict__ flag_mc_ptr,
                                               uint32_t* __restrict__ flag_local_ptr,
                                               int total_elems, int local_rank,
                                               int world_size) {
    int shard_elems = total_elems / world_size;
    int shard_ops = shard_elems / MULTIMEM_BF16_ELEMS_PER_OP;
    int shard_offset_bytes = local_rank * shard_elems * 2;
    int total_threads = gridDim.x * blockDim.x;
    char* mc_base = static_cast<char*>(mc_ptr) + shard_offset_bytes;
    char* dst_base = static_cast<char*>(dst_local);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < shard_ops; i += total_threads) {
        size_t byte_off = static_cast<size_t>(i) * MULTIMEM_BYTES_PER_OP;
        multimem::uint128_t val = multimem::ld_reduce_add_bf16x8(mc_base + byte_off);
        uint32_t* out = reinterpret_cast<uint32_t*>(dst_base + byte_off);
        out[0] = val.x;
        out[1] = val.y;
        out[2] = val.z;
        out[3] = val.w;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        multimem::red_release_sys_add1(flag_mc_ptr + blockIdx.x);
        multimem::spin_wait_eq_reset(
            flag_local_ptr + blockIdx.x,
            static_cast<uint32_t>(world_size), 0u);
    }
}

__global__ void multimem_reduce_scatter_residual_kernel(
    void* __restrict__ mc_ptr,
    void* __restrict__ dst_local,
    const __nv_bfloat16* __restrict__ residual,
    uint32_t* __restrict__ flag_mc_ptr,
    uint32_t* __restrict__ flag_local_ptr,
    int total_elems, int local_rank, int world_size) {
    int shard_elems = total_elems / world_size;
    int shard_ops = shard_elems / MULTIMEM_BF16_ELEMS_PER_OP;
    int shard_offset_bytes = local_rank * shard_elems * 2;
    int total_threads = gridDim.x * blockDim.x;
    char* mc_base = static_cast<char*>(mc_ptr) + shard_offset_bytes;
    char* dst_base = static_cast<char*>(dst_local);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < shard_ops; i += total_threads) {
        size_t byte_off = static_cast<size_t>(i) * MULTIMEM_BYTES_PER_OP;
        multimem::uint128_t val = multimem::ld_reduce_add_bf16x8(mc_base + byte_off);

        int elem_base = i * MULTIMEM_BF16_ELEMS_PER_OP;
        __nv_bfloat162 r0 = *reinterpret_cast<const __nv_bfloat162*>(&residual[elem_base + 0]);
        __nv_bfloat162 r1 = *reinterpret_cast<const __nv_bfloat162*>(&residual[elem_base + 2]);
        __nv_bfloat162 r2 = *reinterpret_cast<const __nv_bfloat162*>(&residual[elem_base + 4]);
        __nv_bfloat162 r3 = *reinterpret_cast<const __nv_bfloat162*>(&residual[elem_base + 6]);

        __nv_bfloat162 v0 = *reinterpret_cast<__nv_bfloat162*>(&val.x);
        __nv_bfloat162 v1 = *reinterpret_cast<__nv_bfloat162*>(&val.y);
        __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&val.z);
        __nv_bfloat162 v3 = *reinterpret_cast<__nv_bfloat162*>(&val.w);

        v0 = __hadd2(v0, r0);
        v1 = __hadd2(v1, r1);
        v2 = __hadd2(v2, r2);
        v3 = __hadd2(v3, r3);

        uint32_t* out = reinterpret_cast<uint32_t*>(dst_base + byte_off);
        out[0] = *reinterpret_cast<uint32_t*>(&v0);
        out[1] = *reinterpret_cast<uint32_t*>(&v1);
        out[2] = *reinterpret_cast<uint32_t*>(&v2);
        out[3] = *reinterpret_cast<uint32_t*>(&v3);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        multimem::red_release_sys_add1(flag_mc_ptr + blockIdx.x);
        multimem::spin_wait_eq_reset(
            flag_local_ptr + blockIdx.x,
            static_cast<uint32_t>(world_size), 0u);
    }
}

__global__ void multimem_allreduce_two_shot_kernel(void* __restrict__ mc_ptr,
                                                   uint32_t* __restrict__ flag_mc_ptr,
                                                   uint32_t* __restrict__ flag_local_ptr,
                                                   int num_elems, int local_rank,
                                                   int world_size) {
    int chunk_elems = num_elems / world_size;
    int chunk_ops = chunk_elems / MULTIMEM_BF16_ELEMS_PER_OP;
    int chunk_offset_bytes = local_rank * chunk_elems * 2;
    int total_threads = gridDim.x * blockDim.x;
    char* base = static_cast<char*>(mc_ptr) + chunk_offset_bytes;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < chunk_ops; i += total_threads) {
        size_t byte_off = static_cast<size_t>(i) * MULTIMEM_BYTES_PER_OP;
        multimem::uint128_t val = multimem::ld_reduce_add_bf16x8(base + byte_off);
        multimem::st_b128(base + byte_off, val);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        multimem::red_release_sys_add1(flag_mc_ptr + blockIdx.x);
        multimem::spin_wait_eq_reset(
            flag_local_ptr + blockIdx.x,
            static_cast<uint32_t>(world_size), 0u);
    }
}

struct MultimemLaunchConfig {
    int grid_size;
    int block_size;
};

inline MultimemLaunchConfig get_multimem_launch_config(int num_elems) {
    int num_ops = num_elems / MULTIMEM_BF16_ELEMS_PER_OP;
    int block = 256;
    int grid = (num_ops + block - 1) / block;
    grid = min(grid, 132);
    return {grid, block};
}

inline void multimem_allreduce(void* mc_ptr, uint32_t* flag_mc, uint32_t* flag_local,
                                int num_elems, int local_rank, int world_size,
                                cudaStream_t stream) {
    auto cfg = get_multimem_launch_config(num_elems);
    multimem_allreduce_kernel<<<cfg.grid_size, cfg.block_size, 0, stream>>>(
        mc_ptr, flag_mc, flag_local, num_elems, local_rank, world_size);
}

inline void multimem_allreduce_two_shot(void* mc_ptr, uint32_t* flag_mc,
                                         uint32_t* flag_local, int num_elems,
                                         int local_rank, int world_size,
                                         cudaStream_t stream) {
    auto cfg = get_multimem_launch_config(num_elems / world_size);
    multimem_allreduce_two_shot_kernel<<<cfg.grid_size, cfg.block_size, 0, stream>>>(
        mc_ptr, flag_mc, flag_local, num_elems, local_rank, world_size);
}

inline void multimem_reduce_scatter(void* mc_ptr, void* dst_local,
                                     uint32_t* flag_mc, uint32_t* flag_local,
                                     int total_elems, int local_rank, int world_size,
                                     cudaStream_t stream) {
    int shard_elems = total_elems / world_size;
    auto cfg = get_multimem_launch_config(shard_elems);
    multimem_reduce_scatter_kernel<<<cfg.grid_size, cfg.block_size, 0, stream>>>(
        mc_ptr, dst_local, flag_mc, flag_local, total_elems, local_rank, world_size);
}

inline void multimem_reduce_scatter_residual(void* mc_ptr, void* dst_local,
                                              const __nv_bfloat16* residual,
                                              uint32_t* flag_mc, uint32_t* flag_local,
                                              int total_elems, int local_rank,
                                              int world_size, cudaStream_t stream) {
    int shard_elems = total_elems / world_size;
    auto cfg = get_multimem_launch_config(shard_elems);
    multimem_reduce_scatter_residual_kernel<<<cfg.grid_size, cfg.block_size, 0, stream>>>(
        mc_ptr, dst_local, residual, flag_mc, flag_local,
        total_elems, local_rank, world_size);
}
