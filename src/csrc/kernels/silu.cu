/**
 * CS450 Project - Waterloo Team
 * Custom CUDA SiLU Activation: SiLU(x) = x / (1 + exp(-x))
 * Target: Qwen3-1.7B MLP blocks
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Naive kernel: 1 thread per element (baseline)
__global__ void silu_kernel_naive(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float x = input[idx];
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// Vectorized kernel: Process 4 floats per thread (float4 coalesced memory)
__global__ void silu_kernel_vectorized(float* output, const float* input, int N) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = vec_idx * 4;
    
    if (idx + 3 < N) {
        // Load 4 floats at once (128-bit coalesced load)
        float4 x = reinterpret_cast<const float4*>(input)[vec_idx];
        
        // Compute SiLU for each element
        float4 result;
        result.x = x.x / (1.0f + expf(-x.x));
        result.y = x.y / (1.0f + expf(-x.y));
        result.z = x.z / (1.0f + expf(-x.z));
        result.w = x.w / (1.0f + expf(-x.w));
        
        // Store 4 floats at once (128-bit coalesced store)
        reinterpret_cast<float4*>(output)[vec_idx] = result;
    }
}

// Handle remaining elements not divisible by 4
__global__ void silu_kernel_tail(float* output, const float* input, int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// Fused kernel: output = SiLU(gate) * up (saves memory bandwidth)
__global__ void silu_multiply_kernel(float* output, const float* gate, const float* up, int N) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = vec_idx * 4;
    
    if (idx + 3 < N) {
        float4 g = reinterpret_cast<const float4*>(gate)[vec_idx];
        float4 u = reinterpret_cast<const float4*>(up)[vec_idx];
        
        float4 result;
        float sigmoid_g_x = 1.0f / (1.0f + expf(-g.x));
        float sigmoid_g_y = 1.0f / (1.0f + expf(-g.y));
        float sigmoid_g_z = 1.0f / (1.0f + expf(-g.z));
        float sigmoid_g_w = 1.0f / (1.0f + expf(-g.w));
        
        result.x = (g.x * sigmoid_g_x) * u.x;
        result.y = (g.y * sigmoid_g_y) * u.y;
        result.z = (g.z * sigmoid_g_z) * u.z;
        result.w = (g.w * sigmoid_g_w) * u.w;
        
        reinterpret_cast<float4*>(output)[vec_idx] = result;
    }
}

__global__ void silu_multiply_kernel_tail(float* output, const float* gate, const float* up, int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g = gate[idx];
        float u = up[idx];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[idx] = (g * sigmoid_g) * u;
    }
}

// ============================================================
// PyTorch C++ Extension Bindings
// ============================================================

torch::Tensor silu_naive(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    silu_kernel_naive<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N
    );
    
    return output;
}

torch::Tensor silu_vectorized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    int N_vec = N / 4;
    int N_tail = N % 4;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int blockSize = 256;
    
    if (N_vec > 0) {
        int numBlocks = (N_vec + blockSize - 1) / blockSize;
        silu_kernel_vectorized<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), input.data_ptr<float>(), N);
    }
    if (N_tail > 0) {
        int numBlocks = (N_tail + blockSize - 1) / blockSize;
        silu_kernel_tail<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), input.data_ptr<float>(), N_vec * 4, N);
    }
    
    return output;
}

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
        silu_multiply_kernel<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(), N);
    }
    if (N_tail > 0) {
        int numBlocks = (N_tail + blockSize - 1) / blockSize;
        silu_multiply_kernel_tail<<<numBlocks, blockSize, 0, stream>>>(
            output.data_ptr<float>(), gate.data_ptr<float>(), up.data_ptr<float>(), N_vec * 4, N);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_naive", &silu_naive, "SiLU activation (naive)");
    m.def("silu_vectorized", &silu_vectorized, "SiLU activation (vectorized)");
    m.def("silu_multiply", &silu_multiply, "SiLU-multiply fused");
}
