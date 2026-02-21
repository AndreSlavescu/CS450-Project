#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "zigzag_attention.cuh"
#include "qwen3.cuh"

// ---------------------------------------------------------------------------
// PyTorch extension for ZigZag Ring Attention
//
// The multi-GPU ring attention loop has moved to Python
// (src/python/Qwen3/zigzag_ring.py) which calls the FA4 CUDA kernel
// for each ring step.  This module retains:
//
//   1. NCCL communicator management (init/destroy) for legacy C++ callers
//   2. A local single-GPU reference implementation for correctness testing
//
// The FMHA attention kernel is in fmha_attention.cuh / fmha_attention.cu.
// ---------------------------------------------------------------------------

static ncclComm_t g_zigzag_comm = nullptr;
static int g_world_size = 1;
static int g_rank = 0;

// ---------------------------------------------------------------------------
// NCCL communicator management
// ---------------------------------------------------------------------------

void init_zigzag_comm(int64_t world_size, int64_t rank, int64_t nccl_unique_id_ptr) {
    g_world_size = static_cast<int>(world_size);
    g_rank = static_cast<int>(rank);

    ncclUniqueId* id = reinterpret_cast<ncclUniqueId*>(nccl_unique_id_ptr);
    NCCL_CHECK(ncclCommInitRank(&g_zigzag_comm, g_world_size, *id, g_rank));
}

void destroy_zigzag_comm() {
    if (g_zigzag_comm != nullptr) {
        ncclCommDestroy(g_zigzag_comm);
        g_zigzag_comm = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Local (single-GPU) reference attention for correctness testing
//
// Computes standard scaled dot-product attention with optional causal mask
// and GQA.  No ring communication — runs entirely on one GPU.
// Useful for validating FA4 kernel output against a known-correct baseline.
// ---------------------------------------------------------------------------
torch::Tensor zigzag_attention_local(torch::Tensor Q, // [batch, num_heads, seq_len, head_dim]
                                     torch::Tensor K, // [batch, num_kv_heads, seq_len, head_dim]
                                     torch::Tensor V, // [batch, num_kv_heads, seq_len, head_dim]
                                     double scale, bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);

    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int num_kv_heads = K.size(1);
    int gqa_ratio = num_heads / num_kv_heads;

    auto K_expanded = K.repeat_interleave(gqa_ratio, 1);
    auto V_expanded = V.repeat_interleave(gqa_ratio, 1);

    // Scale Q
    auto Q_scaled = Q.to(torch::kFloat32) * static_cast<float>(scale);
    auto K_float = K_expanded.to(torch::kFloat32);

    // S = Q @ K^T
    auto S = torch::matmul(Q_scaled, K_float.transpose(-1, -2));

    // Causal mask
    if (causal) {
        auto mask = torch::ones({seq_len, seq_len}, torch::dtype(torch::kBool).device(Q.device())).triu(1);
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -1e9f);
    }

    // Softmax
    auto attn = torch::softmax(S, -1).to(torch::kBFloat16);

    // O = attn @ V
    auto O = torch::matmul(attn, V_expanded.to(torch::kFloat32)).to(torch::kBFloat16);

    return O;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ZigZag Ring Attention for Qwen3";

    m.def("init_zigzag_comm", &init_zigzag_comm, "Initialize NCCL communicator for zigzag ring attention",
          py::arg("world_size"), py::arg("rank"), py::arg("nccl_unique_id_ptr"));
    m.def("destroy_zigzag_comm", &destroy_zigzag_comm, "Destroy zigzag NCCL communicator");

    m.def("zigzag_attention_local", &zigzag_attention_local, "Local reference attention for single-GPU testing",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"), py::arg("causal") = true);
}
