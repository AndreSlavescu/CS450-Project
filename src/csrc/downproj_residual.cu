#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include "gpu_profiler.cuh"
#include "qwen3_dims.cuh"

// Profiler event IDs
enum : int {
    EV_MATVEC     = 0,
    EV_MATVEC_END = 1,
};

/*
 * Down Projection + Residual Add
 *
 * proj = down_proj_w @ silu_out   (matvec: [HIDDEN_DIM, INTERMEDIATE_DIM] @ [INTERMEDIATE_DIM])
 * hidden_states += proj           (residual)
 *
 * Grid: 1 block
 * Block: 256 threads
 */
__global__ void downproj_residual_kernel(
    float* __restrict__ hidden_states,       // [HIDDEN_DIM] — in/out
    const float* __restrict__ silu_out,      // [INTERMEDIATE_DIM]
    const float* __restrict__ down_proj_w,   // [HIDDEN_DIM, INTERMEDIATE_DIM]
    profiler::event_record* g_events,
    int* g_counts
) {
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    profiler::block_state* prof = (profiler::block_state*)smem;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    // MatVec: proj = down_proj_w @ silu_out
    // Each thread computes several output elements
    for (int out_idx = tid; out_idx < HIDDEN_DIM; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = down_proj_w + (long long)out_idx * INTERMEDIATE_DIM;
        for (int j = 0; j < INTERMEDIATE_DIM; j++) {
            acc += row[j] * silu_out[j];
        }
        // Residual add
        hidden_states[out_idx] += acc;
    }

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor downproj_residual_forward(
    torch::Tensor hidden_states,   // [HIDDEN_DIM]
    torch::Tensor silu_out,        // [INTERMEDIATE_DIM]
    torch::Tensor down_proj_w      // [HIDDEN_DIM, INTERMEDIATE_DIM]
) {
    auto output = hidden_states.clone();

    const int block_size = 256;
    size_t smem_bytes = sizeof(profiler::block_state);

    downproj_residual_kernel<<<1, block_size, smem_bytes>>>(
        output.data_ptr<float>(),
        silu_out.data_ptr<float>(),
        down_proj_w.data_ptr<float>(),
        nullptr, nullptr
    );

    return output;
}

torch::Tensor downproj_residual_forward_profiled(
    torch::Tensor hidden_states,
    torch::Tensor silu_out,
    torch::Tensor down_proj_w,
    const std::string& trace_path
) {
    auto output = hidden_states.clone();

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    downproj_residual_kernel<<<grid_size, block_size, smem_bytes>>>(
        output.data_ptr<float>(),
        silu_out.data_ptr<float>(),
        down_proj_w.data_ptr<float>(),
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_MATVEC, "downproj_matvec_residual");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("downproj_residual_forward", &downproj_residual_forward,
          "Down Projection + Residual (CUDA)");
    m.def("downproj_residual_forward_profiled", &downproj_residual_forward_profiled,
          "Down Projection + Residual with profiling (CUDA)");
}
