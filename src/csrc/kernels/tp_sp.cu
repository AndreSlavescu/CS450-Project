#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "tp_sp.cuh"
#include "distributed.cuh"
#include "qwen3.cuh"

static DistributedState g_dist_state = {};
static bool g_dist_initialized = false;
static constexpr int MAX_SMS = 148;

void init_distributed_state(int64_t world_size, int64_t local_rank, int64_t data_buf_size) {
    TORCH_CHECK(!g_dist_initialized, "Distributed state already initialized");
    g_dist_state = create_distributed_state(static_cast<int>(world_size), static_cast<int>(local_rank),
                                            static_cast<size_t>(data_buf_size), MAX_SMS);
    g_dist_initialized = true;
}

void destroy_distributed_state_op() {
    if (!g_dist_initialized)
        return;
    destroy_distributed_state(g_dist_state);
    g_dist_initialized = false;
}

int64_t get_mc_data_ptr() {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    return static_cast<int64_t>(g_dist_state.data_mc.mc_addr);
}

int64_t get_local_data_ptr() {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    return static_cast<int64_t>(g_dist_state.data_mc.local_addr);
}

void copy_to_mc_buffer(torch::Tensor src) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    TORCH_CHECK(src.is_cuda() && src.dtype() == torch::kBFloat16 && src.is_contiguous());
    size_t nbytes = src.numel() * sizeof(__nv_bfloat16);
    TORCH_CHECK(nbytes <= g_dist_state.data_mc.buf_size, "Source tensor exceeds multicast buffer size");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(reinterpret_cast<void*>(g_dist_state.data_mc.local_addr), src.data_ptr(), nbytes,
                    cudaMemcpyDeviceToDevice, stream);
}

void multimem_allreduce_op(int64_t num_elems) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    TORCH_CHECK(num_elems % MULTIMEM_BF16_ELEMS_PER_OP == 0, "num_elems must be divisible by ",
                MULTIMEM_BF16_ELEMS_PER_OP);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce(reinterpret_cast<void*>(g_dist_state.data_mc.mc_addr),
                       reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.mc_addr),
                       reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.local_addr), static_cast<int>(num_elems),
                       g_dist_state.local_rank, g_dist_state.world_size, stream);
}

void multimem_allreduce_two_shot_op(int64_t num_elems) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    TORCH_CHECK(num_elems % (MULTIMEM_BF16_ELEMS_PER_OP * g_dist_state.world_size) == 0,
                "num_elems must be divisible by elems_per_op * world_size");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_two_shot(reinterpret_cast<void*>(g_dist_state.data_mc.mc_addr),
                                reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.mc_addr),
                                reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.local_addr),
                                static_cast<int>(num_elems), g_dist_state.local_rank, g_dist_state.world_size, stream);
}

torch::Tensor multimem_reduce_scatter_op(int64_t total_elems) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    int ws = g_dist_state.world_size;
    TORCH_CHECK(total_elems % (MULTIMEM_BF16_ELEMS_PER_OP * ws) == 0);
    int shard = static_cast<int>(total_elems) / ws;
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, g_dist_state.local_rank);
    auto dst = torch::empty({shard}, options);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    multimem_reduce_scatter(reinterpret_cast<void*>(g_dist_state.data_mc.mc_addr), dst.data_ptr(),
                            reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.mc_addr),
                            reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.local_addr), static_cast<int>(total_elems),
                            g_dist_state.local_rank, ws, stream);
    return dst;
}

torch::Tensor multimem_reduce_scatter_residual_op(torch::Tensor residual, int64_t total_elems) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    TORCH_CHECK(residual.is_cuda() && residual.dtype() == torch::kBFloat16 && residual.is_contiguous());
    int ws = g_dist_state.world_size;
    TORCH_CHECK(total_elems % (MULTIMEM_BF16_ELEMS_PER_OP * ws) == 0);
    int shard = static_cast<int>(total_elems) / ws;
    TORCH_CHECK(residual.numel() == shard);
    auto dst = torch::empty_like(residual);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    multimem_reduce_scatter_residual(reinterpret_cast<void*>(g_dist_state.data_mc.mc_addr), dst.data_ptr(),
                                     reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
                                     reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.mc_addr),
                                     reinterpret_cast<uint32_t*>(g_dist_state.flag_mc.local_addr),
                                     static_cast<int>(total_elems), g_dist_state.local_rank, ws, stream);
    return dst;
}

torch::Tensor read_mc_buffer(int64_t num_elems) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, g_dist_state.local_rank);
    auto out = torch::empty({num_elems}, options);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(out.data_ptr(), reinterpret_cast<void*>(g_dist_state.data_mc.local_addr),
                    num_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor wrap_mc_buffer(int64_t num_elems) {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    TORCH_CHECK(static_cast<size_t>(num_elems) * sizeof(__nv_bfloat16) <= g_dist_state.data_mc.buf_size,
                "Requested size exceeds multicast buffer");
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, g_dist_state.local_rank);
    return torch::from_blob(reinterpret_cast<void*>(g_dist_state.data_mc.local_addr), {num_elems}, options);
}

int64_t get_mc_buf_size_elems() {
    TORCH_CHECK(g_dist_initialized, "Distributed state not initialized");
    return static_cast<int64_t>(g_dist_state.data_mc.buf_size / sizeof(__nv_bfloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Distributed TP/SP via multimem/NVLS";

    m.def("init_distributed", &init_distributed_state);
    m.def("destroy_distributed", &destroy_distributed_state_op);
    m.def("get_mc_data_ptr", &get_mc_data_ptr);
    m.def("get_local_data_ptr", &get_local_data_ptr);
    m.def("copy_to_mc_buffer", &copy_to_mc_buffer);
    m.def("read_mc_buffer", &read_mc_buffer);
    m.def("wrap_mc_buffer", &wrap_mc_buffer);
    m.def("get_mc_buf_size_elems", &get_mc_buf_size_elems);
    m.def("multimem_allreduce", &multimem_allreduce_op);
    m.def("multimem_allreduce_two_shot", &multimem_allreduce_two_shot_op);
    m.def("multimem_reduce_scatter", &multimem_reduce_scatter_op);
    m.def("multimem_reduce_scatter_residual", &multimem_reduce_scatter_residual_op);
}
