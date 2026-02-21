/*
 * qwen3_kernels.cu — Single compilation unit for all Qwen3-1.7B CUDA kernels.
 *
 * This file contains:
 *   1. __global__ kernel wrappers that call __device__ functions from .cuh headers
 *   2. PyTorch C++ extension bindings (standard + profiled variants)
 *
 * Compile via torch.utils.cpp_extension.load() with:
 *   extra_include_paths=["path/to/kernels", "path/to/profiler"]
 *   extra_cuda_cflags=["-std=c++20", "-O2", "-arch=sm_XXX"]
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Device function headers
#include "qwen3_dims.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"
#include "qkv_rope_append.cuh"
#include "oproj_residual.cuh"
#include "upgate_silu.cuh"
#include "downproj_residual.cuh"
#include "rms_lm_head.cuh"

// ============================================================
// __global__ kernel wrappers
// ============================================================

__global__ void qkv_rope_append_kernel(float* q_out, float* k_out, float* v_out, float* k_cache, float* v_cache,
                                       const float* __restrict__ hidden, const float* __restrict__ attn_ln_w,
                                       const float* __restrict__ qkv_weight, const float* __restrict__ q_norm_w,
                                       const float* __restrict__ k_norm_w, const float* __restrict__ cos_cached,
                                       const float* __restrict__ sin_cached, int pos_id, int max_seq_len,
                                       profiler::event_record* g_events, int* g_counts) {
    qkv_rope_append_device(q_out, k_out, v_out, k_cache, v_cache, hidden, attn_ln_w, qkv_weight, q_norm_w, k_norm_w,
                           cos_cached, sin_cached, pos_id, max_seq_len, g_events, g_counts);
}

__global__ void oproj_residual_kernel(float* hidden_states, const float* __restrict__ attn_out,
                                      const float* __restrict__ o_proj_w, profiler::event_record* g_events,
                                      int* g_counts) {
    oproj_residual_device(hidden_states, attn_out, o_proj_w, g_events, g_counts);
}

__global__ void upgate_silu_kernel(float* silu_out, const float* __restrict__ hidden,
                                   const float* __restrict__ mlp_ln_w, const float* __restrict__ gate_w,
                                   const float* __restrict__ up_w, profiler::event_record* g_events, int* g_counts) {
    upgate_silu_device(silu_out, hidden, mlp_ln_w, gate_w, up_w, g_events, g_counts);
}

__global__ void downproj_residual_kernel(float* hidden_states, const float* __restrict__ silu_out,
                                         const float* __restrict__ down_proj_w, profiler::event_record* g_events,
                                         int* g_counts) {
    downproj_residual_device(hidden_states, silu_out, down_proj_w, g_events, g_counts);
}

__global__ void rms_lm_head_kernel(float* logits, const float* __restrict__ hidden, const float* __restrict__ norm_w,
                                   const float* __restrict__ lm_head_w, int vocab_size,
                                   profiler::event_record* g_events, int* g_counts) {
    rms_lm_head_device(logits, hidden, norm_w, lm_head_w, vocab_size, g_events, g_counts);
}

// ---------------------------------------------------------------------------
// Megakernel: fuses post-attention + FFN ops into a single launch per layer.
//
// Execution order (all in one block, shared memory reused between phases):
//   1. oproj_residual  — O-projection + residual add into hidden_states
//   2. upgate_silu     — RMSNorm + gate/up MatVec + fused SiLU*up
//   3. downproj_residual — Down-projection + residual add into hidden_states
//
// Replaces 3 separate kernel launches with 1, saving ~2 launches per layer
// (56 launches across all 28 Qwen3-1.7B layers).
//
// Shared memory: max of the three devices = upgate layout
//   = (HIDDEN_DIM + WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state)
// ---------------------------------------------------------------------------
__global__ void megakernel_layer_kernel(float* hidden_states,                   // [HIDDEN_DIM] in/out
                                        float* silu_buf,                        // [INTERMEDIATE_DIM] scratch
                                        const float* __restrict__ attn_out,     // [HIDDEN_DIM] from attention
                                        const float* __restrict__ o_proj_w,     // [HIDDEN_DIM, HIDDEN_DIM]
                                        const float* __restrict__ mlp_ln_w,     // [HIDDEN_DIM]
                                        const float* __restrict__ gate_w,       // [INTERMEDIATE_DIM, HIDDEN_DIM]
                                        const float* __restrict__ up_w,         // [INTERMEDIATE_DIM, HIDDEN_DIM]
                                        const float* __restrict__ down_proj_w) {// [HIDDEN_DIM, INTERMEDIATE_DIM]
    // Phase 1: O-proj + residual (ends with __syncthreads internally)
    oproj_residual_device(hidden_states, attn_out, o_proj_w, nullptr, nullptr);

    // Phase 2: RMSNorm + gate/up MatVec + SiLU (ends with __syncthreads internally)
    upgate_silu_device(silu_buf, hidden_states, mlp_ln_w, gate_w, up_w, nullptr, nullptr);

    // Phase 3: Down-proj + residual (ends with __syncthreads internally)
    downproj_residual_device(hidden_states, silu_buf, down_proj_w, nullptr, nullptr);
}

// ============================================================
// PyTorch bindings — standard (no profiling)
// ============================================================

std::vector<torch::Tensor> qkv_rope_append_forward(torch::Tensor hidden, torch::Tensor attn_ln_w,
                                                   torch::Tensor qkv_weight, torch::Tensor q_norm_w,
                                                   torch::Tensor k_norm_w, torch::Tensor cos_cached,
                                                   torch::Tensor sin_cached, torch::Tensor k_cache,
                                                   torch::Tensor v_cache, int pos_id) {
    int max_seq_len = k_cache.size(0);
    auto q_out = torch::empty({Q_DIM}, hidden.options());
    auto k_out = torch::empty({K_DIM}, hidden.options());
    auto v_out = torch::empty({V_DIM}, hidden.options());

    const int block_size = 256;
    size_t smem_bytes = (HIDDEN_DIM + QKV_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    qkv_rope_append_kernel<<<1, block_size, smem_bytes>>>(
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(), k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(), hidden.data_ptr<float>(), attn_ln_w.data_ptr<float>(), qkv_weight.data_ptr<float>(),
        q_norm_w.data_ptr<float>(), k_norm_w.data_ptr<float>(), cos_cached.data_ptr<float>(),
        sin_cached.data_ptr<float>(), pos_id, max_seq_len, nullptr, nullptr);

    return {q_out, k_out, v_out};
}

torch::Tensor oproj_residual_forward(torch::Tensor hidden_states, torch::Tensor attn_out, torch::Tensor o_proj_w) {
    auto output = hidden_states.clone();
    const int block_size = 256;
    size_t smem_bytes = HIDDEN_DIM * sizeof(float) + sizeof(profiler::block_state);

    oproj_residual_kernel<<<1, block_size, smem_bytes>>>(output.data_ptr<float>(), attn_out.data_ptr<float>(),
                                                         o_proj_w.data_ptr<float>(), nullptr, nullptr);

    return output;
}

torch::Tensor upgate_silu_forward(torch::Tensor hidden, torch::Tensor mlp_ln_w, torch::Tensor gate_w,
                                  torch::Tensor up_w) {
    auto silu_out = torch::empty({INTERMEDIATE_DIM}, hidden.options());
    const int block_size = 256;
    size_t smem_bytes = (HIDDEN_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    upgate_silu_kernel<<<1, block_size, smem_bytes>>>(silu_out.data_ptr<float>(), hidden.data_ptr<float>(),
                                                      mlp_ln_w.data_ptr<float>(), gate_w.data_ptr<float>(),
                                                      up_w.data_ptr<float>(), nullptr, nullptr);

    return silu_out;
}

torch::Tensor downproj_residual_forward(torch::Tensor hidden_states, torch::Tensor silu_out,
                                        torch::Tensor down_proj_w) {
    auto output = hidden_states.clone();
    const int block_size = 256;
    size_t smem_bytes = sizeof(profiler::block_state);

    downproj_residual_kernel<<<1, block_size, smem_bytes>>>(output.data_ptr<float>(), silu_out.data_ptr<float>(),
                                                            down_proj_w.data_ptr<float>(), nullptr, nullptr);

    return output;
}

torch::Tensor rms_lm_head_forward(torch::Tensor hidden, torch::Tensor norm_w, torch::Tensor lm_head_w, int vocab_size) {
    auto logits = torch::empty({vocab_size}, hidden.options());
    const int block_size = 256;
    size_t smem_bytes = (HIDDEN_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    rms_lm_head_kernel<<<1, block_size, smem_bytes>>>(logits.data_ptr<float>(), hidden.data_ptr<float>(),
                                                      norm_w.data_ptr<float>(), lm_head_w.data_ptr<float>(), vocab_size,
                                                      nullptr, nullptr);

    return logits;
}

torch::Tensor megakernel_layer_forward(torch::Tensor hidden_states, torch::Tensor attn_out,
                                       torch::Tensor o_proj_w, torch::Tensor mlp_ln_w, torch::Tensor gate_w,
                                       torch::Tensor up_w, torch::Tensor down_proj_w) {
    auto output = hidden_states.clone();
    auto silu_buf = torch::empty({INTERMEDIATE_DIM}, hidden_states.options());
    const int block_size = 256;
    // Shared memory: max of the three device function layouts = upgate_silu's requirement
    size_t smem_bytes = (HIDDEN_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    megakernel_layer_kernel<<<1, block_size, smem_bytes>>>(
        output.data_ptr<float>(), silu_buf.data_ptr<float>(), attn_out.data_ptr<float>(),
        o_proj_w.data_ptr<float>(), mlp_ln_w.data_ptr<float>(), gate_w.data_ptr<float>(),
        up_w.data_ptr<float>(), down_proj_w.data_ptr<float>());

    return output;
}

// ============================================================
// PyTorch bindings — profiled variants
// ============================================================

std::vector<torch::Tensor> qkv_rope_append_forward_profiled(torch::Tensor hidden, torch::Tensor attn_ln_w,
                                                            torch::Tensor qkv_weight, torch::Tensor q_norm_w,
                                                            torch::Tensor k_norm_w, torch::Tensor cos_cached,
                                                            torch::Tensor sin_cached, torch::Tensor k_cache,
                                                            torch::Tensor v_cache, int pos_id,
                                                            const std::string& trace_path) {
    int max_seq_len = k_cache.size(0);
    auto q_out = torch::empty({Q_DIM}, hidden.options());
    auto k_out = torch::empty({K_DIM}, hidden.options());
    auto v_out = torch::empty({V_DIM}, hidden.options());

    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = (HIDDEN_DIM + QKV_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    qkv_rope_append_kernel<<<grid_size, block_size, smem_bytes>>>(
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(), k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(), hidden.data_ptr<float>(), attn_ln_w.data_ptr<float>(), qkv_weight.data_ptr<float>(),
        q_norm_w.data_ptr<float>(), k_norm_w.data_ptr<float>(), cos_cached.data_ptr<float>(),
        sin_cached.data_ptr<float>(), pos_id, max_seq_len, prof_buf.d_events, prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(qkv_rope_events::EV_RMSNORM, "rmsnorm");
    names.set(qkv_rope_events::EV_MATVEC, "matvec");
    names.set(qkv_rope_events::EV_QKNORM, "qk_norm");
    names.set(qkv_rope_events::EV_ROPE, "rope");
    names.set(qkv_rope_events::EV_CACHE, "kv_cache");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, true);
    prof_buf.free();

    return {q_out, k_out, v_out};
}

torch::Tensor oproj_residual_forward_profiled(torch::Tensor hidden_states, torch::Tensor attn_out,
                                              torch::Tensor o_proj_w, const std::string& trace_path) {
    auto output = hidden_states.clone();
    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = HIDDEN_DIM * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    oproj_residual_kernel<<<grid_size, block_size, smem_bytes>>>(output.data_ptr<float>(), attn_out.data_ptr<float>(),
                                                                 o_proj_w.data_ptr<float>(), prof_buf.d_events,
                                                                 prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(oproj_events::EV_MATVEC, "oproj_matvec_residual");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, true);
    prof_buf.free();

    return output;
}

torch::Tensor upgate_silu_forward_profiled(torch::Tensor hidden, torch::Tensor mlp_ln_w, torch::Tensor gate_w,
                                           torch::Tensor up_w, const std::string& trace_path) {
    auto silu_out = torch::empty({INTERMEDIATE_DIM}, hidden.options());
    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = (HIDDEN_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    upgate_silu_kernel<<<grid_size, block_size, smem_bytes>>>(
        silu_out.data_ptr<float>(), hidden.data_ptr<float>(), mlp_ln_w.data_ptr<float>(), gate_w.data_ptr<float>(),
        up_w.data_ptr<float>(), prof_buf.d_events, prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(upgate_events::EV_RMSNORM, "rmsnorm");
    names.set(upgate_events::EV_MATVEC, "upgate_matvec");
    names.set(upgate_events::EV_SILU, "silu_mul");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, true);
    prof_buf.free();

    return silu_out;
}

torch::Tensor downproj_residual_forward_profiled(torch::Tensor hidden_states, torch::Tensor silu_out,
                                                 torch::Tensor down_proj_w, const std::string& trace_path) {
    auto output = hidden_states.clone();
    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    downproj_residual_kernel<<<grid_size, block_size, smem_bytes>>>(
        output.data_ptr<float>(), silu_out.data_ptr<float>(), down_proj_w.data_ptr<float>(), prof_buf.d_events,
        prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(downproj_events::EV_MATVEC, "downproj_matvec_residual");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, true);
    prof_buf.free();

    return output;
}

torch::Tensor rms_lm_head_forward_profiled(torch::Tensor hidden, torch::Tensor norm_w, torch::Tensor lm_head_w,
                                           int vocab_size, const std::string& trace_path) {
    auto logits = torch::empty({vocab_size}, hidden.options());
    const int block_size = 256;
    const int grid_size = 1;
    size_t smem_bytes = (HIDDEN_DIM + kernels::WARP_SIZE) * sizeof(float) + sizeof(profiler::block_state);

    profiler::host_buffer prof_buf;
    prof_buf.allocate(grid_size);

    rms_lm_head_kernel<<<grid_size, block_size, smem_bytes>>>(logits.data_ptr<float>(), hidden.data_ptr<float>(),
                                                              norm_w.data_ptr<float>(), lm_head_w.data_ptr<float>(),
                                                              vocab_size, prof_buf.d_events, prof_buf.d_counts);
    cudaDeviceSynchronize();

    profiler::event_names names;
    names.set(lmhead_events::EV_RMSNORM, "rmsnorm");
    names.set(lmhead_events::EV_MATVEC, "lm_head_matvec");

    prof_buf.print_report(&names);
    prof_buf.export_perfetto_json(trace_path.c_str(), &names, true);
    prof_buf.free();

    return logits;
}

// Standalone SiLU-Multiply (Rishu's vectorized implementation)
torch::Tensor silu_multiply(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");
    auto output = torch::empty_like(gate);
    int N = gate.numel();
    int N_vec = N / 4;
    int N_tail = N % 4;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int blockSize = 256;

    if (N_vec > 0) {
        int numBlocks = (N_vec + blockSize - 1) / blockSize;
        kernels::silu_multiply_kernel<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(), N);
    }
    if (N_tail > 0) {
        int numBlocks = (N_tail + blockSize - 1) / blockSize;
        kernels::silu_multiply_kernel_tail<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(), N_vec * 4, N);
    }
    return output;
}

// ============================================================
// PYBIND11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Op 2: QKV + Q/K Norm + RoPE + KV Cache Append
    m.def("qkv_rope_append_forward", &qkv_rope_append_forward, "QKV + Q/K Norm + RoPE + KV Cache Append (CUDA)");
    m.def("qkv_rope_append_forward_profiled", &qkv_rope_append_forward_profiled,
          "QKV + Q/K Norm + RoPE + KV Cache Append with profiling (CUDA)");

    // Op 5: O-Projection + Residual
    m.def("oproj_residual_forward", &oproj_residual_forward, "O-Projection + Residual (CUDA)");
    m.def("oproj_residual_forward_profiled", &oproj_residual_forward_profiled,
          "O-Projection + Residual with profiling (CUDA)");

    // Op 6: RMSNorm + Gate/Up + SiLU
    m.def("upgate_silu_forward", &upgate_silu_forward, "RMSNorm + Gate/Up + SiLU (CUDA)");
    m.def("upgate_silu_forward_profiled", &upgate_silu_forward_profiled,
          "RMSNorm + Gate/Up + SiLU with profiling (CUDA)");

    // Op 7: Down Projection + Residual
    m.def("downproj_residual_forward", &downproj_residual_forward, "Down Projection + Residual (CUDA)");
    m.def("downproj_residual_forward_profiled", &downproj_residual_forward_profiled,
          "Down Projection + Residual with profiling (CUDA)");

    // Op 8: RMSNorm + LM Head
    m.def("rms_lm_head_forward", &rms_lm_head_forward, "RMSNorm + LM Head (CUDA)");
    m.def("rms_lm_head_forward_profiled", &rms_lm_head_forward_profiled, "RMSNorm + LM Head with profiling (CUDA)");

    // Standalone SiLU-Multiply (vectorized)
    m.def("silu_multiply", &silu_multiply, "Fused SiLU-multiply: output = SiLU(gate) * up (vectorized)");

    // Megakernel: fused post-attention + FFN in one launch
    m.def("megakernel_layer_forward", &megakernel_layer_forward,
          "Fused layer: oproj_residual + upgate_silu + downproj_residual (single kernel launch)");
}
