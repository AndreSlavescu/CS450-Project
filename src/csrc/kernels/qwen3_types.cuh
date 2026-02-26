#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

struct GridBarrier {
    unsigned int counter;
    unsigned int epoch;
};

__device__ __forceinline__ void grid_barrier_sync(GridBarrier* __restrict__ bar, unsigned int nblocks,
                                                  unsigned int& local_epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int my_epoch = local_epoch++;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const unsigned int prev = atomicAdd(&bar->counter, 1u);
        if (prev + 1u == nblocks) {
            bar->counter = 0u;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(&bar->epoch, 1u);
        } else {
            volatile const unsigned int* ve = &bar->epoch;
            while (*ve == my_epoch) {
            }
        }
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
}

struct Qwen3LayerWeights {
    const float* attn_ln_w;
    const __nv_bfloat16* qkv_w;
    const float* q_norm_w;
    const float* k_norm_w;
    const __nv_bfloat16* o_proj_w;
    const float* mlp_ln_w;
    const __nv_bfloat16* gate_w;
    const __nv_bfloat16* up_w;
    const __nv_bfloat16* down_proj_w;
    float* k_cache;
    float* v_cache;
};
