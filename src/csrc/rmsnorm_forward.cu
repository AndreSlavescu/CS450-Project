#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include "gpu_profiler.cuh"

// Profiler event IDs (paired: even=begin, odd=end)
enum : int {
    EV_REDUCE     = 0,
    EV_REDUCE_END = 1,
    EV_NORM       = 2,
    EV_NORM_END   = 3,
};

// RMSNorm coop-group kernel with optional profiling
__global__ void rmsnorm_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    int B,
    int T,
    int C,
    profiler::event_record* g_events,
    int* g_counts
) {
    namespace cg = cooperative_groups;
    static constexpr unsigned WARP_SIZE = 32;
    static constexpr float eps = 1e-6f;

    bool has_profiler = (g_events != nullptr);

    __shared__ profiler::block_state prof;
    __shared__ float shared[WARP_SIZE];

    int num_warps = blockDim.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int idx = blockIdx.x;

    if (threadIdx.x == 0 && has_profiler)
        prof.init();
    __syncthreads();

    if (threadIdx.x == 0 && has_profiler)
        prof.record(EV_REDUCE);

    const float *x = inp + idx * C;

    float thread_sum_of_squares = 0.0f;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum_of_squares += xi * xi;
    }

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    float warp_sum_of_squares = cg::reduce(warp, thread_sum_of_squares, cg::plus<float>{}); // sum(x * x)
    if (lane_id == 0) {
        shared[warp_id] = warp_sum_of_squares;
    }
    __syncthreads();

    warp_sum_of_squares = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
    float block_sum_of_squares = cg::reduce(warp, warp_sum_of_squares, cg::plus<float>{}); // sum(x * x)

    // compute rms
    float rms_val = rsqrtf(block_sum_of_squares / C + eps);

    if (threadIdx.x == 0 && has_profiler)
        prof.record(EV_REDUCE_END);

    if (threadIdx.x == 0 && has_profiler)
        prof.record(EV_NORM);

    float *o = out + idx * C;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n =  __ldcs(x+i) * rms_val;
        __stcs(o+i, n * weight[i]);
    }

    if (threadIdx.x == 0 && has_profiler)
        prof.record(EV_NORM_END);

    __syncthreads();
    if (threadIdx.x == 0 && has_profiler) {
        prof.flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// Standard forward (no profiling)
torch::Tensor rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int B,
    int T,
    int C
) {
    auto output = torch::empty_like(input);

    const int block_size = 128;
    const int grid_size = B * T;

    rmsnorm_forward_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        B, T, C,
        nullptr, nullptr
    );

    return output;
}

// Profiled forward (writes Perfetto trace to file)
torch::Tensor rmsnorm_forward_profiled(
    torch::Tensor input,
    torch::Tensor weight,
    int B,
    int T,
    int C,
    const std::string& trace_path
) {
    auto output = torch::empty_like(input);

    const int block_size = 128;
    const int grid_size = B * T;

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    rmsnorm_forward_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        B, T, C,
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_REDUCE, "reduce");
    names.set(EV_NORM, "normalize");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
    m.def("rmsnorm_forward_profiled", &rmsnorm_forward_profiled, "RMSNorm forward with profiling (CUDA)");
}
