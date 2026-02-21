#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace kernels {

static constexpr unsigned WARP_SIZE = 32;

// Block-level sum reduction using cooperative groups.
// Requires shared memory of at least WARP_SIZE floats at `shared_reduce`.
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_reduce, int lane_id, int warp_id,
                                                  int num_warps) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float warp_sum = cg::reduce(warp, val, cg::plus<float>{});
    if (lane_id == 0)
        shared_reduce[warp_id] = warp_sum;
    __syncthreads();

    warp_sum = (lane_id < num_warps) ? shared_reduce[lane_id] : 0.0f;
    float total = cg::reduce(warp, warp_sum, cg::plus<float>{});
    return total;
}

} // namespace kernels
