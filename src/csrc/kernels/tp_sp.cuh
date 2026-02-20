#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// TP+SP (Tensor Parallelism + Sequence Parallelism) communication primitives
//
// For BS=1 decode the activation tensors are tiny:
//   - Qwen3-1.7B: 2048 bf16 = 4 KB
//   - Qwen3-8B:   4096 bf16 = 8 KB
//
// NCCL has ~5-10us launch overhead per collective, which dominates for
// messages this small.  Instead we use P2P direct memory access over
// NVLink: each GPU maps every other GPU's memory via
// cudaDeviceEnablePeerAccess, then a simple kernel writes/reads across
// GPUs with no collective launch overhead.
//
// Pattern (Megatron-LM TP+SP):
//   - Column-parallel regions (QKV, gate/up): all_gather before compute
//   - Row-parallel regions (O, down):        reduce_scatter after compute
// ---------------------------------------------------------------------------

#define CUDA_CHECK(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// P2P state — pointers to every rank's buffers, accessible from any GPU
// ---------------------------------------------------------------------------

struct P2PState {
    // Pointers to each rank's shard buffer, mapped via cudaDeviceEnablePeerAccess.
    // all_gather_ptrs[r] points to rank r's shard in the all_gather output.
    // reduce_scatter_ptrs[r] points to rank r's partial-sum staging buffer.
    __nv_bfloat16** all_gather_ptrs;     // [tp_size] device pointers
    __nv_bfloat16** reduce_scatter_ptrs; // [tp_size] device pointers
    int tp_size;
    int tp_rank;
};

// ---------------------------------------------------------------------------
// All-gather via P2P write
//
// Each rank writes its local shard directly into every other rank's
// output buffer at the correct offset.  No collective, no staging —
// the write goes straight over NVLink.
//
// Output layout: [rank_0_shard | rank_1_shard | ... | rank_{n-1}_shard]
// Each rank writes to output[rank * shard_size .. (rank+1) * shard_size]
// on ALL GPUs (including itself).
// ---------------------------------------------------------------------------

__global__ void
p2p_all_gather_kernel(__nv_bfloat16** __restrict__ dst_ptrs, // [tp_size] pointers to each rank's output buf
                      const __nv_bfloat16* __restrict__ src, // local shard [shard_size]
                      int shard_size, int my_rank, int tp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_size)
        return;

    __nv_bfloat16 val = src[idx];

    // Write our shard into every rank's output buffer at our rank's offset
    int offset = my_rank * shard_size + idx;
    for (int r = 0; r < tp_size; r++) {
        dst_ptrs[r][offset] = val;
    }
}

// ---------------------------------------------------------------------------
// Reduce-scatter via P2P read + local reduce
//
// Each rank reads the relevant shard from every other rank's full
// output buffer, sums them locally, and writes the result.
//
// Input: each rank has [shard_size * tp_size] partial results.
// Rank r's output = sum over all ranks of input[r * shard_size .. (r+1) * shard_size].
// Instead of a collective, rank r reads its slice from every GPU directly.
// ---------------------------------------------------------------------------

__global__ void
p2p_reduce_scatter_kernel(__nv_bfloat16* __restrict__ dst,       // local output [shard_size]
                          __nv_bfloat16** __restrict__ src_ptrs, // [tp_size] pointers to each rank's full buf
                          int shard_size, int my_rank, int tp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_size)
        return;

    // My shard starts at offset my_rank * shard_size in each rank's buffer
    int offset = my_rank * shard_size + idx;

    float sum = 0.0f;
    for (int r = 0; r < tp_size; r++) {
        sum += __bfloat162float(src_ptrs[r][offset]);
    }

    dst[idx] = __float2bfloat16(sum);
}

// ---------------------------------------------------------------------------
// Residual add: dst = a + b  (bf16, element-wise)
// ---------------------------------------------------------------------------

__global__ void residual_add_bf16_kernel(__nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ a,
                                         const __nv_bfloat16* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

// ---------------------------------------------------------------------------
// Fused reduce-scatter + residual: dst = reduce_scatter(partials) + residual
//
// Single kernel: reads from all peers, sums, adds residual.
// Avoids an extra kernel launch + intermediate buffer.
// ---------------------------------------------------------------------------

__global__ void
p2p_reduce_scatter_residual_kernel(__nv_bfloat16* __restrict__ dst,            // [shard_size]
                                   __nv_bfloat16** __restrict__ src_ptrs,      // [tp_size] peer partial buffers
                                   const __nv_bfloat16* __restrict__ residual, // [shard_size]
                                   int shard_size, int my_rank, int tp_size) {
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

// ---------------------------------------------------------------------------
// Host-side wrappers
// ---------------------------------------------------------------------------

inline void tp_all_gather_p2p(P2PState& state,
                              const __nv_bfloat16* local_shard, // [shard_size]
                              int shard_size, cudaStream_t stream) {
    if (state.tp_size == 1)
        return; // output already == input in caller

    int block = 256;
    int grid = (shard_size + block - 1) / block;
    p2p_all_gather_kernel<<<grid, block, 0, stream>>>(state.all_gather_ptrs, local_shard, shard_size, state.tp_rank,
                                                      state.tp_size);
}

inline void tp_reduce_scatter_p2p(P2PState& state,
                                  __nv_bfloat16* dst, // [shard_size]
                                  int shard_size, cudaStream_t stream) {
    if (state.tp_size == 1)
        return;

    int block = 256;
    int grid = (shard_size + block - 1) / block;
    p2p_reduce_scatter_kernel<<<grid, block, 0, stream>>>(dst, state.reduce_scatter_ptrs, shard_size, state.tp_rank,
                                                          state.tp_size);
}

inline void tp_reduce_scatter_residual_p2p(P2PState& state,
                                           __nv_bfloat16* dst,            // [shard_size]
                                           const __nv_bfloat16* residual, // [shard_size]
                                           int shard_size, cudaStream_t stream) {
    if (state.tp_size == 1) {
        // Just residual add (proj_out == dst already in-place)
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

// ---------------------------------------------------------------------------
// P2P initialization
//
// Call once at startup.  Enables peer access between all GPU pairs and
// exchanges buffer pointers via cudaIpcGetMemHandle / cudaIpcOpenMemHandle
// (or simpler: just pass pointers around if using a single process with
// multiple GPUs via cudaSetDevice).
// ---------------------------------------------------------------------------

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
