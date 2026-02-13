#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include "gpu_profiler.cuh"
#include "qwen3_dims.cuh"

// Profiler event IDs (paired: even=begin, odd=end)
enum : int {
    EV_RMSNORM     = 0,
    EV_RMSNORM_END = 1,
    EV_MATVEC      = 2,
    EV_MATVEC_END  = 3,
    EV_QKNORM      = 4,
    EV_QKNORM_END  = 5,
    EV_ROPE        = 6,
    EV_ROPE_END    = 7,
    EV_CACHE       = 8,
    EV_CACHE_END   = 9,
};

// Helper: warp+block reduction for sum of squares
__device__ float block_reduce_sum(float val, float* shared_reduce, int lane_id, int warp_id, int num_warps) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float warp_sum = cg::reduce(warp, val, cg::plus<float>{});
    if (lane_id == 0) shared_reduce[warp_id] = warp_sum;
    __syncthreads();

    warp_sum = (lane_id < num_warps) ? shared_reduce[lane_id] : 0.0f;
    float total = cg::reduce(warp, warp_sum, cg::plus<float>{});
    return total;
}

/*
 * Fused kernel: RMSNorm → QKV MatVec → Q/K per-head RMSNorm → RoPE → KV Cache Append
 *
 * All operations run in a single block for BS=1 single-token decode.
 * Grid: 1 block, Block: 256 threads.
 */
__global__ void qkv_rope_append_kernel(
    float* __restrict__ q_out,           // [Q_DIM]  output
    float* __restrict__ k_out,           // [K_DIM]  output
    float* __restrict__ v_out,           // [V_DIM]  output
    float* __restrict__ k_cache,         // [max_seq, NUM_KV_HEADS, HEAD_DIM]
    float* __restrict__ v_cache,         // [max_seq, NUM_KV_HEADS, HEAD_DIM]
    const float* __restrict__ hidden,    // [HIDDEN_DIM]
    const float* __restrict__ attn_ln_w, // [HIDDEN_DIM]
    const float* __restrict__ qkv_weight,// [QKV_DIM, HIDDEN_DIM] row-major
    const float* __restrict__ q_norm_w,  // [HEAD_DIM]
    const float* __restrict__ k_norm_w,  // [HEAD_DIM]
    const float* __restrict__ cos_cached,// [HEAD_DIM] for this position
    const float* __restrict__ sin_cached,// [HEAD_DIM] for this position
    int pos_id,
    int max_seq_len,
    profiler::event_record* g_events,
    int* g_counts
) {
    bool has_profiler = (g_events != nullptr);

    // Shared memory layout:
    //   s_post_ln[HIDDEN_DIM]  = 2048 floats = 8 KB
    //   s_qkv[QKV_DIM]         = 4096 floats = 16 KB
    //   s_reduce[WARP_SIZE]     = 32 floats   = 128 B
    //   prof                    = profiler state
    // Total: ~24 KB — well within SMEM limits
    extern __shared__ char smem[];
    float* s_post_ln = (float*)smem;
    float* s_qkv     = s_post_ln + HIDDEN_DIM;
    float* s_reduce  = s_qkv + QKV_DIM;
    profiler::block_state* prof = (profiler::block_state*)(s_reduce + WARP_SIZE);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = num_threads / WARP_SIZE;

    if (tid == 0 && has_profiler) prof->init();
    __syncthreads();

    // ===== Phase 1: RMSNorm on hidden_states =====
    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM);

    float thread_ss = 0.0f;
    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        float xi = hidden[i];
        thread_ss += xi * xi;
    }

    float total_ss = block_reduce_sum(thread_ss, s_reduce, lane_id, warp_id, num_warps);
    float rms = rsqrtf(total_ss / HIDDEN_DIM + EPS);

    for (int i = tid; i < HIDDEN_DIM; i += num_threads) {
        s_post_ln[i] = hidden[i] * rms * attn_ln_w[i];
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_RMSNORM_END);

    // ===== Phase 2: QKV MatVec: qkv = qkv_weight[4096, 2048] @ post_ln[2048] =====
    if (tid == 0 && has_profiler) prof->record(EV_MATVEC);

    for (int out_idx = tid; out_idx < QKV_DIM; out_idx += num_threads) {
        float acc = 0.0f;
        const float* row = qkv_weight + (long long)out_idx * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            acc += row[j] * s_post_ln[j];
        }
        s_qkv[out_idx] = acc;
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_MATVEC_END);

    // ===== Phase 3: Q/K per-head RMSNorm =====
    // q_norm_w and k_norm_w are shape [HEAD_DIM=128], shared across all heads
    if (tid == 0 && has_profiler) prof->record(EV_QKNORM);

    // Normalize each Q head (16 heads, HEAD_DIM=128 each)
    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int offset = h * HEAD_DIM;

        float head_ss = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float val = s_qkv[offset + i];
            head_ss += val * val;
        }

        float total_head_ss = block_reduce_sum(head_ss, s_reduce, lane_id, warp_id, num_warps);
        float head_rms = rsqrtf(total_head_ss / HEAD_DIM + EPS);

        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            s_qkv[offset + i] *= head_rms * q_norm_w[i];
        }
        __syncthreads();
    }

    // Normalize each K head (8 heads, HEAD_DIM=128 each)
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int offset = Q_DIM + h * HEAD_DIM;

        float head_ss = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            float val = s_qkv[offset + i];
            head_ss += val * val;
        }

        float total_head_ss = block_reduce_sum(head_ss, s_reduce, lane_id, warp_id, num_warps);
        float head_rms = rsqrtf(total_head_ss / HEAD_DIM + EPS);

        for (int i = tid; i < HEAD_DIM; i += num_threads) {
            s_qkv[offset + i] *= head_rms * k_norm_w[i];
        }
        __syncthreads();
    }

    if (tid == 0 && has_profiler) prof->record(EV_QKNORM_END);

    // ===== Phase 4: RoPE on Q and K =====
    // Standard HF RoPE (rotate_half, NOT interleaved):
    //   out = x * cos + rotate_half(x) * sin
    //   rotate_half(x) = cat(-x[half:], x[:half])
    //
    // For pair (i, i+half):
    //   out[i]      = x[i] * cos[i]      - x[i+half] * sin[i]
    //   out[i+half] = x[i+half] * cos[i+half] + x[i] * sin[i+half]
    //               = x[i+half] * cos[i] + x[i] * sin[i]
    //   (since cos/sin are repeated: cos[i] == cos[i+half])
    if (tid == 0 && has_profiler) prof->record(EV_ROPE);

    // RoPE on all Q heads
    for (int h = 0; h < NUM_Q_HEADS; h++) {
        int offset = h * HEAD_DIM;
        for (int i = tid; i < HALF_HEAD_DIM; i += num_threads) {
            float x_first  = s_qkv[offset + i];
            float x_second = s_qkv[offset + i + HALF_HEAD_DIM];
            float c = cos_cached[i];
            float s = sin_cached[i];
            s_qkv[offset + i]                 = x_first * c - x_second * s;
            s_qkv[offset + i + HALF_HEAD_DIM] = x_second * c + x_first * s;
        }
    }

    // RoPE on all K heads
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        int offset = Q_DIM + h * HEAD_DIM;
        for (int i = tid; i < HALF_HEAD_DIM; i += num_threads) {
            float x_first  = s_qkv[offset + i];
            float x_second = s_qkv[offset + i + HALF_HEAD_DIM];
            float c = cos_cached[i];
            float s = sin_cached[i];
            s_qkv[offset + i]                 = x_first * c - x_second * s;
            s_qkv[offset + i + HALF_HEAD_DIM] = x_second * c + x_first * s;
        }
    }
    __syncthreads();

    if (tid == 0 && has_profiler) prof->record(EV_ROPE_END);

    // ===== Phase 5: Write outputs + KV cache append =====
    if (tid == 0 && has_profiler) prof->record(EV_CACHE);

    // Write Q output
    for (int i = tid; i < Q_DIM; i += num_threads) {
        q_out[i] = s_qkv[i];
    }

    // Write K output and append to cache
    for (int i = tid; i < K_DIM; i += num_threads) {
        float k_val = s_qkv[Q_DIM + i];
        k_out[i] = k_val;
        // k_cache layout: [max_seq, NUM_KV_HEADS, HEAD_DIM]
        k_cache[pos_id * K_DIM + i] = k_val;
    }

    // Write V output and append to cache
    for (int i = tid; i < V_DIM; i += num_threads) {
        float v_val = s_qkv[Q_DIM + K_DIM + i];
        v_out[i] = v_val;
        v_cache[pos_id * V_DIM + i] = v_val;
    }

    if (tid == 0 && has_profiler) prof->record(EV_CACHE_END);

    // Flush profiler
    __syncthreads();
    if (tid == 0 && has_profiler) {
        prof->flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

// ============================================================
// PyTorch bindings
// ============================================================

// Standard forward (no profiling)
std::vector<torch::Tensor> qkv_rope_append_forward(
    torch::Tensor hidden,       // [HIDDEN_DIM]
    torch::Tensor attn_ln_w,    // [HIDDEN_DIM]
    torch::Tensor qkv_weight,   // [QKV_DIM, HIDDEN_DIM]
    torch::Tensor q_norm_w,     // [HEAD_DIM]
    torch::Tensor k_norm_w,     // [HEAD_DIM]
    torch::Tensor cos_cached,   // [HEAD_DIM]
    torch::Tensor sin_cached,   // [HEAD_DIM]
    torch::Tensor k_cache,      // [max_seq, NUM_KV_HEADS, HEAD_DIM]
    torch::Tensor v_cache,      // [max_seq, NUM_KV_HEADS, HEAD_DIM]
    int pos_id
) {
    int max_seq_len = k_cache.size(0);

    auto q_out = torch::empty({Q_DIM}, hidden.options());
    auto k_out = torch::empty({K_DIM}, hidden.options());
    auto v_out = torch::empty({V_DIM}, hidden.options());

    const int block_size = 256;
    // Shared memory: post_ln + qkv + reduce + profiler
    size_t smem_bytes = (HIDDEN_DIM + QKV_DIM + WARP_SIZE) * sizeof(float)
                        + sizeof(profiler::block_state);

    qkv_rope_append_kernel<<<1, block_size, smem_bytes>>>(
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        v_out.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        hidden.data_ptr<float>(),
        attn_ln_w.data_ptr<float>(),
        qkv_weight.data_ptr<float>(),
        q_norm_w.data_ptr<float>(),
        k_norm_w.data_ptr<float>(),
        cos_cached.data_ptr<float>(),
        sin_cached.data_ptr<float>(),
        pos_id,
        max_seq_len,
        nullptr, nullptr
    );

    return {q_out, k_out, v_out};
}

// Profiled forward
std::vector<torch::Tensor> qkv_rope_append_forward_profiled(
    torch::Tensor hidden,
    torch::Tensor attn_ln_w,
    torch::Tensor qkv_weight,
    torch::Tensor q_norm_w,
    torch::Tensor k_norm_w,
    torch::Tensor cos_cached,
    torch::Tensor sin_cached,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int pos_id,
    const std::string& trace_path
) {
    int max_seq_len = k_cache.size(0);

    auto q_out = torch::empty({Q_DIM}, hidden.options());
    auto k_out = torch::empty({K_DIM}, hidden.options());
    auto v_out = torch::empty({V_DIM}, hidden.options());

    const int block_size = 256;
    const int grid_size = 1;

    size_t smem_bytes = (HIDDEN_DIM + QKV_DIM + WARP_SIZE) * sizeof(float)
                        + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    qkv_rope_append_kernel<<<grid_size, block_size, smem_bytes>>>(
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        v_out.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        hidden.data_ptr<float>(),
        attn_ln_w.data_ptr<float>(),
        qkv_weight.data_ptr<float>(),
        q_norm_w.data_ptr<float>(),
        k_norm_w.data_ptr<float>(),
        cos_cached.data_ptr<float>(),
        sin_cached.data_ptr<float>(),
        pos_id,
        max_seq_len,
        prof_buf.d_events, prof_buf.d_counts
    );
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(EV_RMSNORM, "rmsnorm");
    names.set(EV_MATVEC,  "matvec");
    names.set(EV_QKNORM,  "qk_norm");
    names.set(EV_ROPE,    "rope");
    names.set(EV_CACHE,   "kv_cache");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, /*paired=*/true);
    prof_buf.free();

    return {q_out, k_out, v_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_rope_append_forward", &qkv_rope_append_forward,
          "QKV + Q/K Norm + RoPE + KV Cache Append (CUDA)");
    m.def("qkv_rope_append_forward_profiled", &qkv_rope_append_forward_profiled,
          "QKV + Q/K Norm + RoPE + KV Cache Append with profiling (CUDA)");
}
