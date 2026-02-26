#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdint>
#include <cstdio>

#include "qwen3.cuh"

#ifndef NCCL_CHECK
#define NCCL_CHECK(cmd)                                                                                                \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif

struct ZigZagAssignment {
    int block_a;
    int block_b;
    int pos_a;
    int pos_b;
};

__host__ __device__ inline ZigZagAssignment get_zigzag_assignment(int gpu_rank, int n_gpus, int seq_len,
                                                                  int block_size) {
    int total_blocks = (seq_len + block_size - 1) / block_size;

    int padded_blocks = 2 * n_gpus;

    int block_a = gpu_rank;
    int block_b = padded_blocks - 1 - gpu_rank;

    ZigZagAssignment a;
    a.block_a = (block_a < total_blocks) ? block_a : -1;
    a.block_b = (block_b < total_blocks) ? block_b : -1;
    a.pos_a = block_a * block_size;
    a.pos_b = block_b * block_size;
    return a;
}

struct RingCommState {
    __nv_bfloat16* kv_send_buf;
    __nv_bfloat16* kv_recv_buf;
    int buf_size;
    ncclComm_t comm;
    int rank;
    int world_size;
};

inline void ring_send_recv_kv(RingCommState& state, __nv_bfloat16* local_kv, __nv_bfloat16* recv_kv, int kv_elements,
                              cudaStream_t stream) {
    int send_to = (state.rank + 1) % state.world_size;
    int recv_from = (state.rank - 1 + state.world_size) % state.world_size;

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(local_kv, kv_elements, ncclBfloat16, send_to, state.comm, stream));
    NCCL_CHECK(ncclRecv(recv_kv, kv_elements, ncclBfloat16, recv_from, state.comm, stream));
    NCCL_CHECK(ncclGroupEnd());
}
