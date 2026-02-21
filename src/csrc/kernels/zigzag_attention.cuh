#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdint>
#include <cstdio>

#include "qwen3.cuh"

// ---------------------------------------------------------------------------
// ZigZag Ring Attention — Assignment and Communication Helpers
//
// Implements the zigzag variant of ring attention for load-balanced
// long-context prefill across multiple GPUs.
//
// Standard ring attention splits the sequence into n contiguous blocks
// (one per GPU).  Due to causal masking, the first GPU does ~1/n of the
// work while the last does ~100%, creating severe load imbalance.
//
// ZigZag fixes this by splitting into 2n blocks and distributing them
// in mirrored pairs:
//   GPU 0: (b_0, b_{2n-1})
//   GPU 1: (b_1, b_{2n-2})
//   ...
//   GPU k: (b_k, b_{2n-1-k})
//
// Each GPU's combined causal work is equal, achieving near-perfect
// load balance.
//
// The attention compute kernel is in fmha_attention.cuh.
// The ring orchestration is in src/python/Qwen3/zigzag_ring.py.
// This file provides the block assignment logic and NCCL ring helpers
// used by the C++ zigzag_attention.cu reference path.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ZigZag block assignment
//
// Given n_gpus and a sequence of length seq_len, compute which blocks
// are assigned to each GPU.
//
// Returns two block indices per GPU: the "forward" block and the
// "mirrored" block.
// ---------------------------------------------------------------------------

struct ZigZagAssignment {
    int block_a; // forward block index
    int block_b; // mirrored block index
    int pos_a;   // starting token position of block_a
    int pos_b;   // starting token position of block_b
};

__host__ __device__ inline ZigZagAssignment get_zigzag_assignment(int gpu_rank, int n_gpus, int seq_len,
                                                                  int block_size) {
    int total_blocks = (seq_len + block_size - 1) / block_size;

    // Pad to 2 * n_gpus blocks if needed
    int padded_blocks = 2 * n_gpus;
    // Each GPU gets blocks: gpu_rank and (padded_blocks - 1 - gpu_rank)
    int block_a = gpu_rank;
    int block_b = padded_blocks - 1 - gpu_rank;

    ZigZagAssignment a;
    a.block_a = (block_a < total_blocks) ? block_a : -1;
    a.block_b = (block_b < total_blocks) ? block_b : -1;
    a.pos_a = block_a * block_size;
    a.pos_b = block_b * block_size;
    return a;
}

// ---------------------------------------------------------------------------
// Ring communication helpers
//
// In ring attention, at each step:
//   - Send local KV to next rank (rank + 1) % n
//   - Recv KV from previous rank (rank - 1 + n) % n
// Overlapped with local attention compute on the current KV.
// ---------------------------------------------------------------------------

struct RingCommState {
    __nv_bfloat16* kv_send_buf; // double-buffered send
    __nv_bfloat16* kv_recv_buf; // double-buffered recv
    int buf_size;               // elements per KV buffer (kv_rows * head_dim * 2 for K+V)
    ncclComm_t comm;
    int rank;
    int world_size;
};

// Non-blocking ring send/recv of KV blocks
inline void ring_send_recv_kv(RingCommState& state,
                              __nv_bfloat16* local_kv, // [kv_rows * head_dim * 2] K and V concatenated
                              __nv_bfloat16* recv_kv,  // [kv_rows * head_dim * 2] buffer for received KV
                              int kv_elements,         // total bf16 elements (kv_rows * head_dim * 2)
                              cudaStream_t stream) {
    int send_to = (state.rank + 1) % state.world_size;
    int recv_from = (state.rank - 1 + state.world_size) % state.world_size;

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(local_kv, kv_elements, ncclBfloat16, send_to, state.comm, stream));
    NCCL_CHECK(ncclRecv(recv_kv, kv_elements, ncclBfloat16, recv_from, state.comm, stream));
    NCCL_CHECK(ncclGroupEnd());
}
