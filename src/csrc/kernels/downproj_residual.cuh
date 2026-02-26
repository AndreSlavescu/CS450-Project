#pragma once

#include "qwen3.cuh"
#include "utils.cuh"
#include "../profiler/gpu_profiler.cuh"

namespace downproj_events {
enum : int {
    EV_MATVEC = 0,
    EV_MATVEC_END = 1,
};
}

__device__ void downproj_residual_device(float* hidden_states, const float* __restrict__ silu_out,
                                         const __nv_bfloat16* __restrict__ down_proj_w,
                                         profiler::event_record* g_events, int* g_counts) {
    using namespace downproj_events;
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    profiler::block_state* prof = reinterpret_cast<profiler::block_state*>(smem);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler)
        prof->init();
    __syncthreads();

    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC);

    {
        const float4* input4 = reinterpret_cast<const float4*>(silu_out);
        int lane = threadIdx.x & 31;
        int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int total_warps = (blockDim.x * gridDim.x) >> 5;
        for (int out_row = global_warp; out_row < QWEN3_1_7B.hidden_size; out_row += total_warps) {
            const uint4* row8 =
                reinterpret_cast<const uint4*>(down_proj_w + (long long)out_row * QWEN3_1_7B.intermediate_size);
            float dot = 0.f;
            for (int k = lane; k < QWEN3_1_7B.intermediate_size / 8; k += 32) {
                uint4 raw = __ldcg(row8 + k);
                float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
                float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));
                float4 x0 = __ldcg(input4 + k * 2), x1 = __ldcg(input4 + k * 2 + 1);
                dot += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                       f67.x * x1.z + f67.y * x1.w;
            }
            for (int s = 16; s > 0; s >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, s);
            if (lane == 0)
                hidden_states[out_row] += dot;
        }
    }

    if (tid == 0 && has_profiler)
        prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}
