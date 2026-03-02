#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Tile constants (shared between task scheduler and GEMM kernels)
// ---------------------------------------------------------------------------

static constexpr int MOE_TILE_M = 128; // rows per output tile
static constexpr int MOE_TILE_N = 256; // cols per output tile
static constexpr int MOE_TILE_K = 64;  // reduction dimension per iteration

// Maximum local experts (size of offset arrays)
static constexpr int MOE_MAX_LOCAL_EXPERTS = 32;

// ---------------------------------------------------------------------------
// MoE configuration for Qwen3-Coder-480B  (EP=8 → 20 local experts/GPU)
// ---------------------------------------------------------------------------

struct MoEConfig {
    int hidden_size;           // 6144
    int moe_intermediate_size; // 2560
    int num_experts;           // 160  (total)
    int num_local_experts;     // 20   (num_experts / ep_size)
    int top_k;                 // 8
    int local_expert_offset;   // ep_rank * num_local_experts

    // Derived: gate_up concatenated width  (2 * moe_intermediate_size)
    __host__ __device__ int gate_up_width() const { return 2 * moe_intermediate_size; }
};

constexpr MoEConfig QWEN3_480B_MOE = {
    .hidden_size = 6144,
    .moe_intermediate_size = 2560,
    .num_experts = 160,
    .num_local_experts = 20,
    .top_k = 8,
    .local_expert_offset = 0, // set at runtime per rank
};

// ---------------------------------------------------------------------------
// Persistent tile scheduler for multi-expert GEMM
//
// Flattens all (expert, m_tile, n_tile) tasks into a linear sequence.
// A global atomic counter hands out task IDs to CTAs.
// ---------------------------------------------------------------------------

struct MoETaskScheduler {
    // Host-populated arrays (copied to device constant / global memory)
    int expert_offsets[MOE_MAX_LOCAL_EXPERTS + 1];      // prefix sum of token counts
    int expert_tile_offsets[MOE_MAX_LOCAL_EXPERTS + 1]; // prefix sum of tile counts
    int total_tasks;                                    // sum of all expert tiles
    int n_tiles;                                        // output N / MOE_TILE_N
    int num_local_experts;                              // actual number of local experts

    // Decode a flat task_id into (expert, m_tile, n_tile)
    __device__ int3 get_task(int task_id) const {
        if (task_id >= total_tasks)
            return make_int3(-1, -1, -1);

        // Binary search for expert using actual num_local_experts
        int lo = 0, hi = num_local_experts;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (expert_tile_offsets[mid + 1] <= task_id)
                lo = mid + 1;
            else
                hi = mid;
        }
        int expert = lo;
        int local_task = task_id - expert_tile_offsets[expert];

        int m_tile = local_task / n_tiles;
        int n_tile = local_task % n_tiles;

        return make_int3(expert, m_tile, n_tile);
    }
};

// ---------------------------------------------------------------------------
// Router output — describes the sorted token-to-expert assignment
// ---------------------------------------------------------------------------

struct MoERouterOutput {
    int* sorted_token_ids;         // [total_assignments] — original token index
    int* expert_offsets;           // [num_local_experts + 1] — prefix sum
    float* routing_weights_sorted; // [total_assignments] — normalized gate weight
    int total_assignments;         // sum of tokens assigned to local experts
};
