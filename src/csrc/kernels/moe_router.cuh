#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include "moe_expert_types.cuh"

// ---------------------------------------------------------------------------
// Router matmul:  hidden_states [T, hidden_size] x router_weight [num_experts, hidden_size]^T
//                 → router_logits [T, num_experts]
//
// This is a tiny GEMM (N=160 output rows) — use warp-per-row pattern.
// ---------------------------------------------------------------------------

__device__ void moe_router_matmul_kernel(float* router_logits,               // [T, num_experts]  output (fp32)
                                         const __nv_bfloat16* hidden_states, // [T, hidden_size]  input
                                         const __nv_bfloat16* router_weight, // [num_experts, hidden_size]  weight
                                         int T,                              // number of tokens
                                         int hidden_size,                    // 6144
                                         int num_experts                     // 160
) {
    // One warp per (token, expert) pair
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (blockDim.x * gridDim.x) >> 5;

    for (int idx = warp; idx < T * num_experts; idx += nwarps) {
        int tok = idx / num_experts;
        int exp = idx % num_experts;

        const __nv_bfloat16* x = hidden_states + tok * hidden_size;
        const __nv_bfloat16* w = router_weight + exp * hidden_size;

        float acc = 0.0f;
        // Each lane processes 8 elements at a time
        for (int k = lane * 8; k < hidden_size; k += 32 * 8) {
            // Load 8 BF16 values as 4 pairs
            const float4* xp = reinterpret_cast<const float4*>(x + k);
            const float4* wp = reinterpret_cast<const float4*>(w + k);
            float4 xv = __ldcg(xp);
            float4 wv = __ldcg(wp);

            // Unpack bf16x2 pairs and accumulate
            __nv_bfloat162 x0 = *reinterpret_cast<__nv_bfloat162*>(&xv.x);
            __nv_bfloat162 w0 = *reinterpret_cast<__nv_bfloat162*>(&wv.x);
            __nv_bfloat162 x1 = *reinterpret_cast<__nv_bfloat162*>(&xv.y);
            __nv_bfloat162 w1 = *reinterpret_cast<__nv_bfloat162*>(&wv.y);
            __nv_bfloat162 x2 = *reinterpret_cast<__nv_bfloat162*>(&xv.z);
            __nv_bfloat162 w2 = *reinterpret_cast<__nv_bfloat162*>(&wv.z);
            __nv_bfloat162 x3 = *reinterpret_cast<__nv_bfloat162*>(&xv.w);
            __nv_bfloat162 w3 = *reinterpret_cast<__nv_bfloat162*>(&wv.w);

            float2 f0 = __bfloat1622float2(__hmul2(x0, w0));
            float2 f1 = __bfloat1622float2(__hmul2(x1, w1));
            float2 f2 = __bfloat1622float2(__hmul2(x2, w2));
            float2 f3 = __bfloat1622float2(__hmul2(x3, w3));

            acc += f0.x + f0.y + f1.x + f1.y + f2.x + f2.y + f3.x + f3.y;
        }

// Warp reduction
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);

        if (lane == 0)
            router_logits[tok * num_experts + exp] = acc;
    }
}

// ---------------------------------------------------------------------------
// Softmax + Top-K:  router_logits [T, num_experts] → top-k selection
//
// One block per token.  Computes softmax in shared memory, then finds top-k.
// ---------------------------------------------------------------------------

__device__ void moe_softmax_topk_kernel(int* selected_experts,      // [T, top_k]  output — global expert index
                                        float* routing_weights,     // [T, top_k]  output — normalized weights
                                        const float* router_logits, // [T, num_experts]  input
                                        int T,
                                        int num_experts, // 160
                                        int top_k,       // 8
                                        bool norm_topk_prob) {
    extern __shared__ float smem[];
    float* logits = smem;                                      // [num_experts]
    float* topk_vals = smem + num_experts;                     // [top_k]
    int* topk_ids = reinterpret_cast<int*>(topk_vals + top_k); // [top_k]

    int tid = threadIdx.x;
    int tok = blockIdx.x;
    if (tok >= T)
        return;

    const float* src = router_logits + tok * num_experts;

    // Load logits to shared memory + find max for softmax stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_experts; i += blockDim.x) {
        float v = src[i];
        logits[i] = v;
        local_max = fmaxf(local_max, v);
    }
    // Block-reduce max
    __shared__ float shared_max;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    if (tid == 0)
        shared_max = local_max;
    __syncthreads();
    float max_val = shared_max;

    // Softmax: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_experts; i += blockDim.x) {
        float v = expf(logits[i] - max_val);
        logits[i] = v;
        local_sum += v;
    }
    __shared__ float shared_sum;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (tid == 0)
        shared_sum = local_sum;
    __syncthreads();
    float sum_val = shared_sum;

    // Normalize
    for (int i = tid; i < num_experts; i += blockDim.x)
        logits[i] /= sum_val;
    __syncthreads();

    // Top-K selection (simple serial scan on thread 0 — K=8 is tiny)
    if (tid == 0) {
        for (int k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            int best_idx = -1;
            for (int i = 0; i < num_experts; i++) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_idx = i;
                }
            }
            topk_vals[k] = best_val;
            topk_ids[k] = best_idx;
            logits[best_idx] = -1.0f; // mask out selected
        }

        // Normalize top-k weights if requested
        if (norm_topk_prob) {
            float wsum = 0.0f;
            for (int k = 0; k < top_k; k++)
                wsum += topk_vals[k];
            for (int k = 0; k < top_k; k++)
                topk_vals[k] /= wsum;
        }

        // Write outputs
        for (int k = 0; k < top_k; k++) {
            selected_experts[tok * top_k + k] = topk_ids[k];
            routing_weights[tok * top_k + k] = topk_vals[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Sort tokens by expert for contiguous dispatch.
//
// Given selected_experts [T, top_k] and routing_weights [T, top_k]:
//   1. Filter to local experts only (those in [local_offset, local_offset + n_local))
//   2. Build sorted_token_ids, expert_offsets, routing_weights_sorted
//
// Each assignment = (token_id, local_expert_idx, weight).
// Sort by local_expert_idx so that all tokens for expert 0 come first, then 1, etc.
// ---------------------------------------------------------------------------

__device__ void moe_build_assignment_kernel(
    // Outputs (per-assignment)
    int* assignment_token_ids,  // [max_assignments]
    int* assignment_expert_ids, // [max_assignments]  (local expert index)
    float* assignment_weights,  // [max_assignments]
    int* num_assignments_out,   // [1] — atomic counter
    // Inputs
    const int* selected_experts,  // [T, top_k]  (global expert indices)
    const float* routing_weights, // [T, top_k]
    int T, int top_k, int local_expert_offset, int num_local_experts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * top_k;
    if (tid >= total)
        return;

    int tok = tid / top_k;
    int sel = tid % top_k;
    int global_expert = selected_experts[tok * top_k + sel];

    // Check if this expert is local
    int local_idx = global_expert - local_expert_offset;
    if (local_idx >= 0 && local_idx < num_local_experts) {
        int slot = atomicAdd(num_assignments_out, 1);
        assignment_token_ids[slot] = tok;
        assignment_expert_ids[slot] = local_idx;
        assignment_weights[slot] = routing_weights[tok * top_k + sel];
    }
}

// Sorting is done via histogram + scatter (see kernels below).
// No CUB sort needed — histogram approach is optimal for small expert counts.

// ---------------------------------------------------------------------------
// Histogram + scatter sort: O(N + num_local_experts) — optimal for small expert count
// ---------------------------------------------------------------------------

// Phase 1: Count tokens per expert
__device__ void moe_histogram_kernel(int* expert_counts,               // [num_local_experts] — output
                                     const int* assignment_expert_ids, // [num_assignments]
                                     int num_assignments) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_assignments)
        return;
    atomicAdd(&expert_counts[assignment_expert_ids[tid]], 1);
}

// Phase 2: Scatter assignments into sorted order using prefix-sum offsets
__device__ void moe_scatter_kernel(int* sorted_token_ids,           // output [num_assignments]
                                   float* sorted_weights,           // output [num_assignments]
                                   int* scatter_counters,           // [num_local_experts] — atomic per-expert slot
                                   const int* expert_offsets,       // [num_local_experts + 1] — prefix sum
                                   const int* assignment_token_ids, // input
                                   const int* assignment_expert_ids, const float* assignment_weights,
                                   int num_assignments) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_assignments)
        return;

    int expert = assignment_expert_ids[tid];
    int slot = expert_offsets[expert] + atomicAdd(&scatter_counters[expert], 1);
    sorted_token_ids[slot] = assignment_token_ids[tid];
    sorted_weights[slot] = assignment_weights[tid];
}

// ---------------------------------------------------------------------------
// Gather sorted activations for contiguous TMA loads
//   sorted_hidden[i] = hidden_states[sorted_token_ids[i]]
// ---------------------------------------------------------------------------

__device__ void moe_gather_tokens_kernel(__nv_bfloat16* sorted_hidden,       // [total_assignments, hidden_size]
                                         const __nv_bfloat16* hidden_states, // [T, hidden_size]
                                         const int* sorted_token_ids,        // [total_assignments]
                                         int total_assignments, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = total_assignments * hidden_size;
    if (idx >= total_elems)
        return;

    int assignment = idx / hidden_size;
    int col = idx % hidden_size;
    int src_tok = sorted_token_ids[assignment];

    sorted_hidden[assignment * hidden_size + col] = hidden_states[src_tok * hidden_size + col];
}

// ---------------------------------------------------------------------------
// SiLU fusion kernel:
//   intermediate[t, j] = SiLU(gate_up_out[t, j]) * gate_up_out[t, j + intermediate_size]
//   for j in [0, intermediate_size)
//
// gate_up_out layout: [total_assignments, 2 * intermediate_size]
//   columns [0, intermediate_size) = gate output
//   columns [intermediate_size, 2*intermediate_size) = up output
// ---------------------------------------------------------------------------

__device__ void moe_silu_fusion_kernel(__nv_bfloat16* intermediate,      // [total_assignments, intermediate_size]
                                       const __nv_bfloat16* gate_up_out, // [total_assignments, 2 * intermediate_size]
                                       int total_assignments,
                                       int intermediate_size // 2560
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = total_assignments * intermediate_size;
    if (idx >= total_elems)
        return;

    int row = idx / intermediate_size;
    int col = idx % intermediate_size;

    float gate_val = __bfloat162float(gate_up_out[row * 2 * intermediate_size + col]);
    float up_val = __bfloat162float(gate_up_out[row * 2 * intermediate_size + intermediate_size + col]);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu = gate_val / (1.0f + expf(-gate_val));
    float result = silu * up_val;

    intermediate[row * intermediate_size + col] = __float2bfloat16(result);
}

// ---------------------------------------------------------------------------
// Scatter-accumulate kernel for down_proj output:
//   output[original_token, col] += routing_weight * down_out[sorted_idx, col]
//
// Uses BF16 atomic add (SM100 native).
// ---------------------------------------------------------------------------

__device__ void moe_scatter_accumulate_kernel(float* output, // [T, hidden_size] — FP32 accumulated output
                                              const __nv_bfloat16* down_out, // [total_assignments, hidden_size]
                                              const int* sorted_token_ids,   // [total_assignments]
                                              const float* sorted_weights,   // [total_assignments]
                                              int total_assignments, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = total_assignments * hidden_size;
    if (idx >= total_elems)
        return;

    int assignment = idx / hidden_size;
    int col = idx % hidden_size;
    int src_tok = sorted_token_ids[assignment];
    float weight = sorted_weights[assignment];

    float val = __bfloat162float(down_out[assignment * hidden_size + col]);
    float weighted = val * weight;

    // Atomic add to FP32 output buffer
    atomicAdd(&output[src_tok * hidden_size + col], weighted);
}

// ---------------------------------------------------------------------------
// GPU-side prefix scan for small arrays (num_local_experts ≤ 32)
// Replaces CPU prefix scan + host sync in the router pipeline.
// ---------------------------------------------------------------------------

__device__ void moe_prefix_scan_kernel(int* expert_offsets, // [num_local_experts + 1] output
                                       const int* expert_counts, // [num_local_experts] input
                                       int num_local_experts) {
    if (threadIdx.x != 0)
        return;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_local_experts; i++)
        expert_offsets[i + 1] = expert_offsets[i] + expert_counts[i];
}

// ---------------------------------------------------------------------------
// Vectorized gather: one block per assignment row, float4 (8 bf16) per thread
// ---------------------------------------------------------------------------

__device__ void moe_gather_tokens_vec_kernel(__nv_bfloat16* sorted_hidden,       // [total_assignments, hidden_size]
                                             const __nv_bfloat16* hidden_states, // [T, hidden_size]
                                             const int* sorted_token_ids,        // [total_assignments]
                                             int total_assignments, int hidden_size) {
    int assignment = blockIdx.x;
    if (assignment >= total_assignments)
        return;
    int src_tok = sorted_token_ids[assignment];

    const float4* src = reinterpret_cast<const float4*>(hidden_states + (long)src_tok * hidden_size);
    float4* dst = reinterpret_cast<float4*>(sorted_hidden + (long)assignment * hidden_size);
    int nvec = hidden_size >> 3; // 8 bf16 per float4
    for (int i = threadIdx.x; i < nvec; i += blockDim.x)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Vectorized SiLU fusion: one block per row, float4 vectorized reads/writes
//   intermediate[t, j] = SiLU(gate_up[t, j]) * gate_up[t, j + intermediate_size]
// ---------------------------------------------------------------------------

__device__ __forceinline__ __nv_bfloat162 silu_mul_bf16x2(__nv_bfloat162 gate, __nv_bfloat162 up) {
    float2 gf = __bfloat1622float2(gate);
    float2 uf = __bfloat1622float2(up);
    float r0 = (gf.x / (1.0f + expf(-gf.x))) * uf.x;
    float r1 = (gf.y / (1.0f + expf(-gf.y))) * uf.y;
    return __floats2bfloat162_rn(r0, r1);
}

__device__ void moe_silu_fusion_vec_kernel(__nv_bfloat16* intermediate,      // [total_assignments, intermediate_size]
                                           const __nv_bfloat16* gate_up_out, // [total_assignments, 2 * intermediate_size]
                                           int total_assignments,
                                           int intermediate_size) {
    int row = blockIdx.x;
    if (row >= total_assignments)
        return;

    const __nv_bfloat16* gate_row = gate_up_out + (long)row * 2 * intermediate_size;
    const __nv_bfloat16* up_row = gate_row + intermediate_size;
    __nv_bfloat16* out_row = intermediate + (long)row * intermediate_size;

    int nvec = intermediate_size >> 3; // 8 bf16 per float4
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 g_raw = reinterpret_cast<const float4*>(gate_row)[i];
        float4 u_raw = reinterpret_cast<const float4*>(up_row)[i];

        float4 result;
        *reinterpret_cast<__nv_bfloat162*>(&result.x) =
            silu_mul_bf16x2(*reinterpret_cast<__nv_bfloat162*>(&g_raw.x), *reinterpret_cast<__nv_bfloat162*>(&u_raw.x));
        *reinterpret_cast<__nv_bfloat162*>(&result.y) =
            silu_mul_bf16x2(*reinterpret_cast<__nv_bfloat162*>(&g_raw.y), *reinterpret_cast<__nv_bfloat162*>(&u_raw.y));
        *reinterpret_cast<__nv_bfloat162*>(&result.z) =
            silu_mul_bf16x2(*reinterpret_cast<__nv_bfloat162*>(&g_raw.z), *reinterpret_cast<__nv_bfloat162*>(&u_raw.z));
        *reinterpret_cast<__nv_bfloat162*>(&result.w) =
            silu_mul_bf16x2(*reinterpret_cast<__nv_bfloat162*>(&g_raw.w), *reinterpret_cast<__nv_bfloat162*>(&u_raw.w));

        reinterpret_cast<float4*>(out_row)[i] = result;
    }
}

// ---------------------------------------------------------------------------
// Vectorized scatter-accumulate: one block per assignment row
//   output[token, col] += weight * down_out[assignment, col]
// Uses float4 loads, scalar FP32 atomics (minimal contention with EP)
// ---------------------------------------------------------------------------

__device__ void moe_scatter_accumulate_vec_kernel(float* output,                  // [T, hidden_size] FP32
                                                  const __nv_bfloat16* down_out,  // [total_assignments, hidden_size]
                                                  const int* sorted_token_ids,    // [total_assignments]
                                                  const float* sorted_weights,    // [total_assignments]
                                                  int total_assignments, int hidden_size) {
    int assignment = blockIdx.x;
    if (assignment >= total_assignments)
        return;

    int src_tok = sorted_token_ids[assignment];
    float weight = sorted_weights[assignment];
    float* dst = output + (long)src_tok * hidden_size;
    const float4* src = reinterpret_cast<const float4*>(down_out + (long)assignment * hidden_size);

    int nvec = hidden_size >> 3; // 8 bf16 per float4
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 raw = src[i];
        __nv_bfloat162 v0 = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
        __nv_bfloat162 v1 = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
        __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&raw.z);
        __nv_bfloat162 v3 = *reinterpret_cast<__nv_bfloat162*>(&raw.w);
        float2 f0 = __bfloat1622float2(v0);
        float2 f1 = __bfloat1622float2(v1);
        float2 f2 = __bfloat1622float2(v2);
        float2 f3 = __bfloat1622float2(v3);
        int col = i * 8;
        atomicAdd(&dst[col + 0], f0.x * weight);
        atomicAdd(&dst[col + 1], f0.y * weight);
        atomicAdd(&dst[col + 2], f1.x * weight);
        atomicAdd(&dst[col + 3], f1.y * weight);
        atomicAdd(&dst[col + 4], f2.x * weight);
        atomicAdd(&dst[col + 5], f2.y * weight);
        atomicAdd(&dst[col + 6], f3.x * weight);
        atomicAdd(&dst[col + 7], f3.y * weight);
    }
}

// ---------------------------------------------------------------------------
// Zero-initialize a float buffer
// ---------------------------------------------------------------------------

__device__ void moe_zero_kernel(float* buf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        buf[idx] = 0.0f;
}

// ---------------------------------------------------------------------------
// __global__ wrappers for standalone kernel launches
// (Thin shims — the megakernel calls the __device__ functions directly)
// ---------------------------------------------------------------------------

__global__ void moe_router_matmul_kernel_launch(float* router_logits, const __nv_bfloat16* hidden_states,
                                                const __nv_bfloat16* router_weight, int T, int hidden_size,
                                                int num_experts) {
    moe_router_matmul_kernel(router_logits, hidden_states, router_weight, T, hidden_size, num_experts);
}

__global__ void moe_softmax_topk_kernel_launch(int* selected_experts, float* routing_weights,
                                               const float* router_logits, int T, int num_experts, int top_k,
                                               bool norm_topk_prob) {
    moe_softmax_topk_kernel(selected_experts, routing_weights, router_logits, T, num_experts, top_k, norm_topk_prob);
}

__global__ void moe_build_assignment_kernel_launch(int* assignment_token_ids, int* assignment_expert_ids,
                                                   float* assignment_weights, int* num_assignments_out,
                                                   const int* selected_experts, const float* routing_weights, int T,
                                                   int top_k, int local_expert_offset, int num_local_experts) {
    moe_build_assignment_kernel(assignment_token_ids, assignment_expert_ids, assignment_weights, num_assignments_out,
                                selected_experts, routing_weights, T, top_k, local_expert_offset, num_local_experts);
}

__global__ void moe_histogram_kernel_launch(int* expert_counts, const int* assignment_expert_ids, int num_assignments) {
    moe_histogram_kernel(expert_counts, assignment_expert_ids, num_assignments);
}

__global__ void moe_scatter_kernel_launch(int* sorted_token_ids, float* sorted_weights, int* scatter_counters,
                                          const int* expert_offsets, const int* assignment_token_ids,
                                          const int* assignment_expert_ids, const float* assignment_weights,
                                          int num_assignments) {
    moe_scatter_kernel(sorted_token_ids, sorted_weights, scatter_counters, expert_offsets, assignment_token_ids,
                       assignment_expert_ids, assignment_weights, num_assignments);
}

__global__ void moe_gather_tokens_kernel_launch(__nv_bfloat16* sorted_hidden, const __nv_bfloat16* hidden_states,
                                                const int* sorted_token_ids, int total_assignments, int hidden_size) {
    moe_gather_tokens_vec_kernel(sorted_hidden, hidden_states, sorted_token_ids, total_assignments, hidden_size);
}

__global__ void moe_silu_fusion_kernel_launch(__nv_bfloat16* intermediate, const __nv_bfloat16* gate_up_out,
                                              int total_assignments, int intermediate_size) {
    moe_silu_fusion_vec_kernel(intermediate, gate_up_out, total_assignments, intermediate_size);
}

__global__ void moe_scatter_accumulate_kernel_launch(float* output, const __nv_bfloat16* down_out,
                                                     const int* sorted_token_ids, const float* sorted_weights,
                                                     int total_assignments, int hidden_size) {
    moe_scatter_accumulate_vec_kernel(output, down_out, sorted_token_ids, sorted_weights, total_assignments, hidden_size);
}

__global__ void moe_zero_kernel_launch(float* buf, int n) {
    moe_zero_kernel(buf, n);
}

__global__ void moe_prefix_scan_kernel_launch(int* expert_offsets, const int* expert_counts, int num_local_experts) {
    moe_prefix_scan_kernel(expert_offsets, expert_counts, num_local_experts);
}
