/**
 * CS450 Project - Waterloo Team
 * Custom CUDA SiLU Activation: SiLU(x) = x / (1 + exp(-x))
 * Target: Qwen3-1.7B MLP blocks
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void silu_multiply_kernel(float* output, const float* gate, const float* up, int N) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = vec_idx * 4;

    if (idx + 3 < N) {
        float4 g = reinterpret_cast<const float4*>(gate)[vec_idx];
        float4 u = reinterpret_cast<const float4*>(up)[vec_idx];

        float4 result;
        result.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + expf(-g.w))) * u.w;

        reinterpret_cast<float4*>(output)[vec_idx] = result;
    }
}

__global__ void silu_multiply_kernel_tail(float* output, const float* gate, const float* up, int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g = gate[idx];
        float u = up[idx];
        output[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

// ============================================================
// PyTorch C++ Extension Bindings
// ============================================================

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
        silu_multiply_kernel<<<numBlocks, blockSize, 0, stream>>>(output.data_ptr<float>(), gate.data_ptr<float>(),
                                                                  up.data_ptr<float>(), N);
    }
    if (N_tail > 0) {
        int numBlocks = (N_tail + blockSize - 1) / blockSize;
        silu_multiply_kernel_tail<<<numBlocks, blockSize, 0, stream>>>(output.data_ptr<float>(), gate.data_ptr<float>(),
                                                                       up.data_ptr<float>(), N_vec * 4, N);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_multiply", &silu_multiply, "Fused SiLU-multiply: output = SiLU(gate) * up");
}
