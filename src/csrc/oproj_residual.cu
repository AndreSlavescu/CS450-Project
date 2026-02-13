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
    EV_RESID      = 2,
    EV_RESID_END  = 3,
};

/*
 * O-Projection + Residual Add
 *
 * proj = o_proj_weight @ attn_out   (matvec: [HIDDEN_DIM, HIDDEN_DIM] @ [HIDDEN_DIM])
 * hidden_states += proj             (residual)
 *
 * For BS=1 decode, this is a [2048, 2048] @ [2048] matvec.
 *
 * Grid: 1 block
 * Block: 256 threads
 */
__global__ void oproj_residual_kernel(
    float* __restrict__ hidden_states,    // [HIDDEN_DIM] — in/out (residual added in-place)
    const float* __restrict__ attn_out,   // [HIDDEN_DIM]
    const float* __restrict__ o_proj_w,   // [HIDDEN_DIM, HIDDEN_DIM] row-major
    profiler::event_record* g_events,
    int* g_counts
) {
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_attn = (float*)smem;  // [HIDDEN_DIM]
    profiler::block_state* prof = (profiler::block_state*)(s_attn + HIDDEN_DIM);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    // Load attn_out into shared memory
    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        s_attn[i] = attn_out[i];
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    // MatVec: proj = o_proj_w @ attn_out
    // Each thread computes HIDDEN_DIM / num_threads output elements
    for (int out_idx = tid; out_idx < HIDDEN_DIM; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = o_proj_w + (long long)out_idx * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            acc += row[j] * s_attn[j];
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

torch::Tensor oproj_residual_forward(
    torch::Tensor hidden_states,  // [HIDDEN_DIM] — will be modified in place
    torch::Tensor attn_out,       // [HIDDEN_DIM]
    torch::Tensor o_proj_w        // [HIDDEN_DIM, HIDDEN_DIM]
) {
    auto output = hidden_states.clone();  // don't modify original

    const int block_size = 256;
    size_t smem_bytes = HIDDEN_DIM * sizeof(float) + sizeof(profiler::block_state);

    oproj_residual_kernel<<<1, block_size, smem_bytes>>>(
        output.data_ptr<float>(),
        attn_out.data_ptr<float>(),
        o_proj_w.data_ptr<float>(),
        nullptr, nullptr
    );

    return output;
}

torch::Tensor oproj_residual_forward_profiled(
    torch::Tensor hidden_states,
    torch::Tensor attn_out,
    torch::Tensor o_proj_w,
    const std::string& trace_path
) {
    auto output = hidden_states.clone();

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = HIDDEN_DIM * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    oproj_residual_kernel<<<grid_size, block_size, smem_bytes>>>(
        output.data_ptr<float>(),
        attn_out.data_ptr<float>(),
        o_proj_w.data_ptr<float>(),
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_MATVEC, "oproj_matvec_residual");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("oproj_residual_forward", &oproj_residual_forward,
          "O-Projection + Residual (CUDA)");
    m.def("oproj_residual_forward_profiled", &oproj_residual_forward_profiled,
          "O-Projection + Residual with profiling (CUDA)");
}
