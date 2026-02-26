#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "zigzag_attention.cuh"
#include "qwen3.cuh"

static ncclComm_t g_zigzag_comm = nullptr;
static int g_world_size = 1;
static int g_rank = 0;

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

torch::Tensor zigzag_attention_local(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);

    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int num_kv_heads = K.size(1);
    int gqa_ratio = num_heads / num_kv_heads;

    auto K_expanded = K.repeat_interleave(gqa_ratio, 1);
    auto V_expanded = V.repeat_interleave(gqa_ratio, 1);

    auto Q_scaled = Q.to(torch::kFloat32) * static_cast<float>(scale);
    auto K_float = K_expanded.to(torch::kFloat32);

    auto S = torch::matmul(Q_scaled, K_float.transpose(-1, -2));

    if (causal) {
        auto mask = torch::ones({seq_len, seq_len}, torch::dtype(torch::kBool).device(Q.device())).triu(1);
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -1e9f);
    }

    auto attn = torch::softmax(S, -1).to(torch::kBFloat16);

    auto O = torch::matmul(attn, V_expanded.to(torch::kFloat32)).to(torch::kBFloat16);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ZigZag Ring Attention for Qwen3";

    m.def("init_zigzag_comm", &init_zigzag_comm, "Initialize NCCL communicator for zigzag ring attention",
          py::arg("world_size"), py::arg("rank"), py::arg("nccl_unique_id_ptr"));
    m.def("destroy_zigzag_comm", &destroy_zigzag_comm, "Destroy zigzag NCCL communicator");

    m.def("zigzag_attention_local", &zigzag_attention_local, "Local reference attention for single-GPU testing",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"), py::arg("causal") = true);
}
