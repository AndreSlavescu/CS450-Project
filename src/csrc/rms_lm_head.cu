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
};

// Helper: block-level sum reduction
__device__ float block_reduce_sum_lmhead(float val, float* shared_reduce, int lane_id, int warp_id, int num_warps) {
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
 * Final RMSNorm + LM Head projection
 *
 * post_ln = RMSNorm(hidden_states, norm_w)
 * logits = lm_head_w @ post_ln     (matvec: [VOCAB_SIZE, HIDDEN_DIM] @ [HIDDEN_DIM])
 *
 * Note: For Qwen3-1.7B, lm_head_w = embed_tokens.weight (tied embeddings).
 *
 * Grid: 1 block
 * Block: 256 threads
 */
__global__ void rms_lm_head_kernel(
    float* __restrict__ logits,        // [vocab_size] output
    const float* __restrict__ hidden,  // [HIDDEN_DIM]
    const float* __restrict__ norm_w,  // [HIDDEN_DIM]
    const float* __restrict__ lm_head_w, // [vocab_size, HIDDEN_DIM]
    int vocab_size,
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

    float total_ss = block_reduce_sum_lmhead(thread_ss, s_reduce, lane_id, warp_id, num_warps);
    float rms = rsqrtf(total_ss / HIDDEN_DIM + EPS);

    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        s_post_ln[i] = hidden[i] * rms * norm_w[i];
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM_END);

    // ===== Phase 2: LM Head MatVec =====
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    // logits[i] = lm_head_w[i, :] @ post_ln
    for (int out_idx = tid; out_idx < vocab_size; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = lm_head_w + (long long)out_idx * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            acc += row[j] * s_post_ln[j];
        }
        logits[out_idx] = acc;
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

torch::Tensor rms_lm_head_forward(
    torch::Tensor hidden,     // [HIDDEN_DIM]
    torch::Tensor norm_w,     // [HIDDEN_DIM]
    torch::Tensor lm_head_w,  // [vocab_size, HIDDEN_DIM]
    int vocab_size
) {
    auto logits = torch::empty({vocab_size}, hidden.options());

    const int block_size = 256;
    size_t smem_bytes = (HIDDEN_DIM + WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    rms_lm_head_kernel<<<1, block_size, smem_bytes>>>(
        logits.data_ptr<float>(),
        hidden.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        lm_head_w.data_ptr<float>(),
        vocab_size,
        nullptr, nullptr
    );

    return logits;
}

torch::Tensor rms_lm_head_forward_profiled(
    torch::Tensor hidden,
    torch::Tensor norm_w,
    torch::Tensor lm_head_w,
    int vocab_size,
    const std::string& trace_path
) {
    auto logits = torch::empty({vocab_size}, hidden.options());

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = (HIDDEN_DIM + WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    rms_lm_head_kernel<<<grid_size, block_size, smem_bytes>>>(
        logits.data_ptr<float>(),
        hidden.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        lm_head_w.data_ptr<float>(),
        vocab_size,
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_RMSNORM, "rmsnorm");
    names.set(EV_MATVEC, "lm_head_matvec");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_lm_head_forward", &rms_lm_head_forward,
          "RMSNorm + LM Head (CUDA)");
    m.def("rms_lm_head_forward_profiled", &rms_lm_head_forward_profiled,
          "RMSNorm + LM Head with profiling (CUDA)");
}
