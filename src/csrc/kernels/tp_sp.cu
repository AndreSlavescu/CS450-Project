#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "tp_sp.cuh"
#include "qwen3.cuh"

static P2PState g_p2p_state = {};
static bool g_p2p_initialized = false;

void init_p2p_state(int64_t tp_size, int64_t tp_rank, int64_t all_gather_ptrs_addr, int64_t reduce_scatter_ptrs_addr) {
    g_p2p_state.tp_size = static_cast<int>(tp_size);
    g_p2p_state.tp_rank = static_cast<int>(tp_rank);
    g_p2p_state.all_gather_ptrs = reinterpret_cast<__nv_bfloat16**>(all_gather_ptrs_addr);
    g_p2p_state.reduce_scatter_ptrs = reinterpret_cast<__nv_bfloat16**>(reduce_scatter_ptrs_addr);

    enable_p2p_access(g_p2p_state.tp_size, g_p2p_state.tp_rank);
    g_p2p_initialized = true;
}

void destroy_p2p_state() {
    g_p2p_state = {};
    g_p2p_initialized = false;
}

void tp_all_gather_op(torch::Tensor local_shard) {
    TORCH_CHECK(g_p2p_initialized, "P2P state not initialized");
    TORCH_CHECK(local_shard.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(local_shard.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(local_shard.is_contiguous(), "input must be contiguous");

    if (g_p2p_state.tp_size == 1)
        return;

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    tp_all_gather_p2p(g_p2p_state, reinterpret_cast<const __nv_bfloat16*>(local_shard.data_ptr()), local_shard.numel(),
                      stream);
}

torch::Tensor tp_reduce_scatter_op(int64_t shard_size) {
    TORCH_CHECK(g_p2p_initialized, "P2P state not initialized");

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, g_p2p_state.tp_rank);
    auto dst = torch::empty({shard_size}, options);

    if (g_p2p_state.tp_size == 1)
        return dst;

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    tp_reduce_scatter_p2p(g_p2p_state, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()), static_cast<int>(shard_size),
                          stream);

    return dst;
}

torch::Tensor tp_reduce_scatter_residual_op(torch::Tensor residual, int64_t shard_size) {
    TORCH_CHECK(g_p2p_initialized, "P2P state not initialized");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual.dtype() == torch::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual.numel() == shard_size, "residual size must match shard_size");

    auto dst = torch::empty_like(residual);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    tp_reduce_scatter_residual_p2p(g_p2p_state, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
                                   reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
                                   static_cast<int>(shard_size), stream);

    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TP+SP P2P communication primitives for Qwen3";

    m.def("init_p2p_state", &init_p2p_state, "Initialize P2P state with peer buffer pointers", py::arg("tp_size"),
          py::arg("tp_rank"), py::arg("all_gather_ptrs_addr"), py::arg("reduce_scatter_ptrs_addr"));
    m.def("destroy_p2p_state", &destroy_p2p_state, "Reset P2P state");

    m.def("all_gather", &tp_all_gather_op, "P2P all-gather: write local shard to all peers", py::arg("local_shard"));
    m.def("reduce_scatter", &tp_reduce_scatter_op, "P2P reduce-scatter: read shards from all peers and sum",
          py::arg("shard_size"));
    m.def("reduce_scatter_residual", &tp_reduce_scatter_residual_op, "Fused P2P reduce-scatter + residual add",
          py::arg("residual"), py::arg("shard_size"));
}
