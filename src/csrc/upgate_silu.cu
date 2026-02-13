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
    EV_RMSNORM     = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC      = 2,
    EV_MATVEC_END  = 3,
    EV_SILU        = 4,
    EV_SILU_END    = 5,
};

// Helper: block-level sum reduction
__device__ float block_reduce_sum(float val, float* shared_reduce, int lane_id, int warp_id, int num_warps) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float warp_sum = cg::reduce(warp, val, cg::plus<float>{});
    if (lane_id == 0) shared_reduce[warp_id] = warp_sum;
    __syncthreads();

    warp_sum = (lane_id < num_warps) ? shared_reduce[lane_id] : 0.0f;
    return cg::reduce(warp, warp_sum, cg::plus<float>{});
}

/*
 * RMSNorm + Gate/Up double MatVec + SiLU
 *
 * post_ln = RMSNorm(hidden_states, mlp_ln_w)
 * gate = gate_w @ post_ln          [INTERMEDIATE_DIM]
 * up   = up_w   @ post_ln          [INTERMEDIATE_DIM]
 * silu_out = SiLU(gate) * up        [INTERMEDIATE_DIM]
 *
 * Grid: 1 block
 * Block: 256 threads
 */
__global__ void upgate_silu_kernel(
    float* __restrict__ silu_out,        // [INTERMEDIATE_DIM] output
    const float* __restrict__ hidden,    // [HIDDEN_DIM]
    const float* __restrict__ mlp_ln_w,  // [HIDDEN_DIM]
    const float* __restrict__ gate_w,    // [INTERMEDIATE_DIM, HIDDEN_DIM]
    const float* __restrict__ up_w,      // [INTERMEDIATE_DIM, HIDDEN_DIM]
    profiler::event_record* g_events,
    int* g_counts
) {
    bool has_profiler = (g_events != nullptr);

    extern __shared__ char smem[];
    float* s_post_ln = (float*)smem;
    float* s_reduce  = s_post_ln + HIDDEN_DIM;
    profiler::block_state* prof = (profiler::block_state*)(s_reduce + WARP_SIZE);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = num_threads / WARP_SIZE;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    // ===== Phase 1: RMSNorm =====
    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM);

    float thread_ss = 0.0f;
    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        float xi = hidden[i];
        thread_ss += xi * xi;
    }

    float total_ss = block_reduce_sum(thread_ss, s_reduce, lane_id, warp_id, num_warps);
    float rms = rsqrtf(total_ss / HIDDEN_DIM + EPS);

    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        s_post_ln[i] = hidden[i] * rms * mlp_ln_w[i];
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM_END);

    // ===== Phase 2: Gate + Up MatVec =====
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    // Each thread computes multiple output elements
    // gate[i] = gate_w[i, :] @ post_ln, up[i] = up_w[i, :] @ post_ln
    // Then immediately compute SiLU(gate) * up

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);
    if (tid == 0 && has_profiler) prof->record(EV_SILU);

    for (int out_idx = tid; out_idx < INTERMEDIATE_DIM; out_idx += num_threads) {
        const float* gate_row = gate_w + (long long)out_idx * HIDDEN_DIM;
        const float* up_row   = up_w   + (long long)out_idx * HIDDEN_DIM;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        for (int j = 0; j < HIDDEN_DIM; j++) {
            float post_ln_j = s_post_ln[j];
            gate_acc += gate_row[j] * post_ln_j;
            up_acc   += up_row[j]   * post_ln_j;
        }

        // SiLU(gate) * up
        float silu_gate = gate_acc / (1.0f + expf(-gate_acc));
        silu_out[out_idx] = silu_gate * up_acc;
    }

    if (tid == 0 && has_profiler) prof->record(EV_SILU_END);

    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor upgate_silu_forward(
    torch::Tensor hidden,    // [HIDDEN_DIM]
    torch::Tensor mlp_ln_w,  // [HIDDEN_DIM]
    torch::Tensor gate_w,    // [INTERMEDIATE_DIM, HIDDEN_DIM]
    torch::Tensor up_w       // [INTERMEDIATE_DIM, HIDDEN_DIM]
) {
    auto silu_out = torch::empty({INTERMEDIATE_DIM}, hidden.options());

    const int block_size = 256;
    size_t smem_bytes = (HIDDEN_DIM + WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    upgate_silu_kernel<<<1, block_size, smem_bytes>>>(
        silu_out.data_ptr<float>(),
        hidden.data_ptr<float>(),
        mlp_ln_w.data_ptr<float>(),
        gate_w.data_ptr<float>(),
        up_w.data_ptr<float>(),
        nullptr, nullptr
    );

    return silu_out;
}

torch::Tensor upgate_silu_forward_profiled(
    torch::Tensor hidden,
    torch::Tensor mlp_ln_w,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    const std::string& trace_path
) {
    auto silu_out = torch::empty({INTERMEDIATE_DIM}, hidden.options());

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = (HIDDEN_DIM + WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    upgate_silu_kernel<<<grid_size, block_size, smem_bytes>>>(
        silu_out.data_ptr<float>(),
        hidden.data_ptr<float>(),
        mlp_ln_w.data_ptr<float>(),
        gate_w.data_ptr<float>(),
        up_w.data_ptr<float>(),
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_RMSNORM, "rmsnorm");
    names.set(EV_MATVEC, "upgate_matvec");
    names.set(EV_SILU, "silu_mul");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return silu_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upgate_silu_forward", &upgate_silu_forward,
          "RMSNorm + Gate/Up + SiLU (CUDA)");
    m.def("upgate_silu_forward_profiled", &upgate_silu_forward_profiled,
          "RMSNorm + Gate/Up + SiLU with profiling (CUDA)");
}
