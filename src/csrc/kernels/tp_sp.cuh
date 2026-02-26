#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

struct P2PState {
    __nv_bfloat16** all_gather_ptrs;
    __nv_bfloat16** reduce_scatter_ptrs;
    int tp_size;
    int tp_rank;
};

__global__ void p2p_all_gather_kernel(__nv_bfloat16** __restrict__ dst_ptrs, const __nv_bfloat16* __restrict__ src,
                                      int shard_size, int my_rank, int tp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_size)
        return;

    __nv_bfloat16 val = src[idx];

    int offset = my_rank * shard_size + idx;
    for (int r = 0; r < tp_size; r++) {
        dst_ptrs[r][offset] = val;
    }
}

__global__ void p2p_reduce_scatter_kernel(__nv_bfloat16* __restrict__ dst, __nv_bfloat16** __restrict__ src_ptrs,
                                          int shard_size, int my_rank, int tp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_size)
        return;

    int offset = my_rank * shard_size + idx;

    float sum = 0.0f;
    for (int r = 0; r < tp_size; r++) {
        sum += __bfloat162float(src_ptrs[r][offset]);
    }

    dst[idx] = __float2bfloat16(sum);
}

__global__ void residual_add_bf16_kernel(__nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ a,
                                         const __nv_bfloat16* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

__global__ void p2p_reduce_scatter_residual_kernel(__nv_bfloat16* __restrict__ dst,
                                                   __nv_bfloat16** __restrict__ src_ptrs,
                                                   const __nv_bfloat16* __restrict__ residual, int shard_size,
                                                   int my_rank, int tp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_size)
        return;

    int offset = my_rank * shard_size + idx;

    float sum = 0.0f;
    for (int r = 0; r < tp_size; r++) {
        sum += __bfloat162float(src_ptrs[r][offset]);
    }
    sum += __bfloat162float(residual[idx]);

    dst[idx] = __float2bfloat16(sum);
}

inline void tp_all_gather_p2p(P2PState& state, const __nv_bfloat16* local_shard, int shard_size, cudaStream_t stream) {
    if (state.tp_size == 1)
        return;

    int block = 256;
    int grid = (shard_size + block - 1) / block;
    p2p_all_gather_kernel<<<grid, block, 0, stream>>>(state.all_gather_ptrs, local_shard, shard_size, state.tp_rank,
                                                      state.tp_size);
}

inline void tp_reduce_scatter_p2p(P2PState& state, __nv_bfloat16* dst, int shard_size, cudaStream_t stream) {
    if (state.tp_size == 1)
        return;

    int block = 256;
    int grid = (shard_size + block - 1) / block;
    p2p_reduce_scatter_kernel<<<grid, block, 0, stream>>>(dst, state.reduce_scatter_ptrs, shard_size, state.tp_rank,
                                                          state.tp_size);
}

inline void tp_reduce_scatter_residual_p2p(P2PState& state, __nv_bfloat16* dst, const __nv_bfloat16* residual,
                                           int shard_size, cudaStream_t stream) {
    if (state.tp_size == 1) {
        int block = 256;
        int grid = (shard_size + block - 1) / block;
        residual_add_bf16_kernel<<<grid, block, 0, stream>>>(dst, dst, residual, shard_size);
        return;
    }

    int block = 256;
    int grid = (shard_size + block - 1) / block;
    p2p_reduce_scatter_residual_kernel<<<grid, block, 0, stream>>>(dst, state.reduce_scatter_ptrs, residual, shard_size,
                                                                   state.tp_rank, state.tp_size);
}

inline void enable_p2p_access(int tp_size, int tp_rank) {
    for (int r = 0; r < tp_size; r++) {
        if (r == tp_rank)
            continue;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, tp_rank, r));
        if (can_access) {
            cudaError_t err = cudaDeviceEnablePeerAccess(r, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            }
        }
    }
}
