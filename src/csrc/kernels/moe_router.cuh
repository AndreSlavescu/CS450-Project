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

__global__ void moe_router_matmul_kernel(
    float*                 router_logits,   // [T, num_experts]  output (fp32)
    const __nv_bfloat16*   hidden_states,   // [T, hidden_size]  input
    const __nv_bfloat16*   router_weight,   // [num_experts, hidden_size]  weight
    int                    T,               // number of tokens
    int                    hidden_size,     // 6144
    int                    num_experts       // 160
) {
    // One warp per (token, expert) pair
    const int lane   = threadIdx.x & 31;
    const int warp   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
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

        if (lane == 0) router_logits[tok * num_experts + exp] = acc;
    }
}

// ---------------------------------------------------------------------------
// Softmax + Top-K:  router_logits [T, num_experts] → top-k selection
//
// One block per token.  Computes softmax in shared memory, then finds top-k.
// ---------------------------------------------------------------------------

__global__ void moe_softmax_topk_kernel(
    int*       selected_experts,   // [T, top_k]  output — global expert index
    float*     routing_weights,    // [T, top_k]  output — normalized weights
    const float* router_logits,    // [T, num_experts]  input
    int        T,
    int        num_experts,        // 160
    int        top_k,              // 8
    bool       norm_topk_prob
) {
    extern __shared__ float smem[];
    float* logits = smem;                      // [num_experts]
    float* topk_vals = smem + num_experts;     // [top_k]
    int*   topk_ids  = (int*)(topk_vals + top_k); // [top_k]

    int tid = threadIdx.x;
    int tok = blockIdx.x;
    if (tok >= T) return;

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
    if (tid == 0) shared_max = local_max;
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
    if (tid == 0) shared_sum = local_sum;
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
            int   best_idx = -1;
            for (int i = 0; i < num_experts; i++) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_idx = i;
                }
            }
            topk_vals[k] = best_val;
            topk_ids[k]  = best_idx;
            logits[best_idx] = -1.0f; // mask out selected
        }

        // Normalize top-k weights if requested
        if (norm_topk_prob) {
            float wsum = 0.0f;
            for (int k = 0; k < top_k; k++) wsum += topk_vals[k];
            for (int k = 0; k < top_k; k++) topk_vals[k] /= wsum;
        }

        // Write outputs
        for (int k = 0; k < top_k; k++) {
            selected_experts[tok * top_k + k] = topk_ids[k];
            routing_weights[tok * top_k + k]  = topk_vals[k];
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

__global__ void moe_build_assignment_kernel(
    // Outputs (per-assignment)
    int*   assignment_token_ids,  // [max_assignments]
    int*   assignment_expert_ids, // [max_assignments]  (local expert index)
    float* assignment_weights,    // [max_assignments]
    int*   num_assignments_out,   // [1] — atomic counter
    // Inputs
    const int*   selected_experts,  // [T, top_k]  (global expert indices)
    const float* routing_weights,   // [T, top_k]
    int T, int top_k,
    int local_expert_offset,
    int num_local_experts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * top_k;
    if (tid >= total) return;

    int tok = tid / top_k;
    int sel = tid % top_k;
    int global_expert = selected_experts[tok * top_k + sel];

    // Check if this expert is local
    int local_idx = global_expert - local_expert_offset;
    if (local_idx >= 0 && local_idx < num_local_experts) {
        int slot = atomicAdd(num_assignments_out, 1);
        assignment_token_ids[slot]  = tok;
        assignment_expert_ids[slot] = local_idx;
        assignment_weights[slot]    = routing_weights[tok * top_k + sel];
    }
}

// Sorting is done via histogram + scatter (see kernels below).
// No CUB sort needed — histogram approach is optimal for small expert counts.

// ---------------------------------------------------------------------------
// Histogram + scatter sort: O(N + num_local_experts) — optimal for small expert count
// ---------------------------------------------------------------------------

// Phase 1: Count tokens per expert
__global__ void moe_histogram_kernel(
    int*       expert_counts,            // [num_local_experts] — output
    const int* assignment_expert_ids,    // [num_assignments]
    int        num_assignments
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_assignments) return;
    atomicAdd(&expert_counts[assignment_expert_ids[tid]], 1);
}

// Phase 2: Scatter assignments into sorted order using prefix-sum offsets
__global__ void moe_scatter_kernel(
    int*       sorted_token_ids,         // output [num_assignments]
    float*     sorted_weights,           // output [num_assignments]
    int*       scatter_counters,         // [num_local_experts] — atomic per-expert slot
    const int* expert_offsets,           // [num_local_experts + 1] — prefix sum
    const int* assignment_token_ids,     // input
    const int* assignment_expert_ids,
    const float* assignment_weights,
    int        num_assignments
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_assignments) return;

    int expert = assignment_expert_ids[tid];
    int slot = expert_offsets[expert] + atomicAdd(&scatter_counters[expert], 1);
    sorted_token_ids[slot] = assignment_token_ids[tid];
    sorted_weights[slot]   = assignment_weights[tid];
}

// ---------------------------------------------------------------------------
// Gather sorted activations for contiguous TMA loads
//   sorted_hidden[i] = hidden_states[sorted_token_ids[i]]
// ---------------------------------------------------------------------------

__global__ void moe_gather_tokens_kernel(
    __nv_bfloat16*       sorted_hidden,   // [total_assignments, hidden_size]
    const __nv_bfloat16* hidden_states,   // [T, hidden_size]
    const int*           sorted_token_ids,// [total_assignments]
    int                  total_assignments,
    int                  hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = total_assignments * hidden_size;
    if (idx >= total_elems) return;

    int assignment = idx / hidden_size;
    int col        = idx % hidden_size;
    int src_tok    = sorted_token_ids[assignment];

    sorted_hidden[assignment * hidden_size + col] =
        hidden_states[src_tok * hidden_size + col];
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

__global__ void moe_silu_fusion_kernel(
    __nv_bfloat16*       intermediate,   // [total_assignments, intermediate_size]
    const __nv_bfloat16* gate_up_out,    // [total_assignments, 2 * intermediate_size]
    int                  total_assignments,
    int                  intermediate_size  // 2560
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = total_assignments * intermediate_size;
    if (idx >= total_elems) return;

    int row = idx / intermediate_size;
    int col = idx % intermediate_size;

    float gate_val = __bfloat162float(gate_up_out[row * 2 * intermediate_size + col]);
    float up_val   = __bfloat162float(gate_up_out[row * 2 * intermediate_size + intermediate_size + col]);

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

__global__ void moe_scatter_accumulate_kernel(
    float*               output,            // [T, hidden_size] — FP32 accumulated output
    const __nv_bfloat16* down_out,          // [total_assignments, hidden_size]
    const int*           sorted_token_ids,  // [total_assignments]
    const float*         sorted_weights,    // [total_assignments]
    int                  total_assignments,
    int                  hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = total_assignments * hidden_size;
    if (idx >= total_elems) return;

    int assignment = idx / hidden_size;
    int col        = idx % hidden_size;
    int src_tok    = sorted_token_ids[assignment];
    float weight   = sorted_weights[assignment];

    float val = __bfloat162float(down_out[assignment * hidden_size + col]);
    float weighted = val * weight;

    // Atomic add to FP32 output buffer
    atomicAdd(&output[src_tok * hidden_size + col], weighted);
}

// ---------------------------------------------------------------------------
// Zero-initialize a float buffer
// ---------------------------------------------------------------------------

__global__ void moe_zero_kernel(float* buf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = 0.0f;
}
