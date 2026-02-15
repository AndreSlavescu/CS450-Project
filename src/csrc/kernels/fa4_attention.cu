#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "fa4_attention.cuh"
#include "qwen3.cuh"

// ---------------------------------------------------------------------------
// PyTorch extension for FA4 Forward Attention
//
// Provides a Python-callable interface for the raw CUDA FlashAttention-style
// forward kernel.  Supports GQA, causal masking, and optional LSE output
// for ring attention accumulation.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// fa4_forward
//
// Args:
//   Q:  [num_q_heads, seq_q, head_dim]   bf16
//   K:  [num_kv_heads, seq_kv, head_dim] bf16
//   V:  [num_kv_heads, seq_kv, head_dim] bf16
//   scale:      attention scale (1/sqrt(head_dim))
//   causal:     apply causal masking
//   return_lse: if true, also return log-sum-exp per query row
//   q_offset:   global position offset for Q (for ring attention)
//   kv_offset:  global position offset for KV (for ring attention)
//
// Returns:
//   [O]             if return_lse is false
//   [O, lse]        if return_lse is true
//
//   O:   [num_q_heads, seq_q, head_dim] bf16
//   lse: [num_q_heads, seq_q]           fp32
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fa4_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale,
                                                 bool causal, bool return_lse, int64_t q_offset, int64_t kv_offset) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be 3-dimensional [heads, seq, head_dim]");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Q, K, V must be contiguous");

    int num_q_heads = Q.size(0);
    int seq_q = Q.size(1);
    int head_dim = Q.size(2);
    int num_kv_heads = K.size(0);
    int seq_kv = K.size(1);

    TORCH_CHECK(head_dim == 128, "FA4 kernel requires head_dim=128 (Qwen3)");
    TORCH_CHECK(K.size(2) == head_dim && V.size(2) == head_dim, "K, V head_dim must match Q");
    TORCH_CHECK(V.size(0) == num_kv_heads && V.size(1) == seq_kv, "V shape must match K");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)");

    // Allocate output
    auto O = torch::empty_like(Q);

    // Optionally allocate LSE
    float* lse_ptr = nullptr;
    torch::Tensor lse;
    if (return_lse) {
        lse = torch::empty({num_q_heads, seq_q}, torch::dtype(torch::kFloat32).device(Q.device()));
        lse_ptr = lse.data_ptr<float>();
    }

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    fa4_forward(reinterpret_cast<__nv_bfloat16*>(O.data_ptr()), lse_ptr,
                reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), num_q_heads, num_kv_heads, seq_q, seq_kv,
                static_cast<float>(scale), causal, static_cast<int>(q_offset), static_cast<int>(kv_offset), stream);

    if (return_lse) {
        return {O, lse};
    }
    return {O};
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FA4 Forward Attention for Qwen3";

    m.def("forward", &fa4_attention_forward, "FA4 forward attention (GQA, causal, optional LSE)", py::arg("Q"),
          py::arg("K"), py::arg("V"), py::arg("scale"), py::arg("causal") = true, py::arg("return_lse") = false,
          py::arg("q_offset") = 0, py::arg("kv_offset") = 0);
}
