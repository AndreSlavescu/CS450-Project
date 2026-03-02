// ---------------------------------------------------------------------------
// PyBind11 bindings for the tcgen05 fused MoE expert kernel
//
// Exposes the following Python functions:
//   moe_router_topk(hidden_states, router_weight, top_k, norm_topk, local_offset, n_local)
//   moe_gate_up_gemm(sorted_hidden, gate_up_weights, scheduler_data)
//   moe_silu_fusion(gate_up_out, intermediate_size)
//   moe_down_gemm(intermediate, down_weights, scheduler_data)
//   moe_scatter_accumulate(down_out, sorted_token_ids, sorted_weights, T, hidden_size)
//   moe_gather_tokens(hidden_states, sorted_token_ids, hidden_size)
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "moe_expert_types.cuh"
#include "moe_router.cuh"

// tcgen05 kernel (SM100) — currently disabled due to TK/driver compatibility
// #if defined(KITTENS_BLACKWELL)
// #include "moe_expert.cuh"
// #endif

// ---------------------------------------------------------------------------
// Router + Top-K + Sort
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> moe_router_topk_op(
    torch::Tensor hidden_states,   // [T, hidden_size]  bf16
    torch::Tensor router_weight,   // [num_experts, hidden_size]  bf16
    int top_k,
    bool norm_topk_prob,
    int local_expert_offset,
    int num_local_experts
) {
    TORCH_CHECK(hidden_states.is_cuda() && router_weight.is_cuda());
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);

    int T = hidden_states.size(0);
    int hidden_size = hidden_states.size(1);
    int num_experts = router_weight.size(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 1. Router matmul → logits [T, num_experts] in FP32
    auto router_logits = torch::empty({T, num_experts},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    {
        int threads = 256;
        int warps_needed = T * num_experts;
        int blocks = (warps_needed * 32 + threads - 1) / threads;
        blocks = std::min(blocks, 1024);

        moe_router_matmul_kernel<<<blocks, threads, 0, stream>>>(
            router_logits.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(router_weight.data_ptr()),
            T, hidden_size, num_experts
        );
    }

    // 2. Softmax + Top-K → selected_experts [T, top_k], routing_weights [T, top_k]
    auto selected_experts = torch::empty({T, top_k},
        torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto routing_weights = torch::empty({T, top_k},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));

    {
        int smem = (num_experts + top_k) * sizeof(float) + top_k * sizeof(int);
        moe_softmax_topk_kernel<<<T, 32, smem, stream>>>(
            selected_experts.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            router_logits.data_ptr<float>(),
            T, num_experts, top_k, norm_topk_prob
        );
    }

    // 3. Build assignments (filter to local experts)
    int max_assignments = T * top_k;
    auto assignment_token_ids  = torch::empty({max_assignments}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto assignment_expert_ids = torch::empty({max_assignments}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto assignment_weights    = torch::empty({max_assignments}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
    auto num_assignments       = torch::zeros({1}, torch::dtype(torch::kInt32).device(hidden_states.device()));

    {
        int total = T * top_k;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        moe_build_assignment_kernel<<<blocks, threads, 0, stream>>>(
            assignment_token_ids.data_ptr<int>(),
            assignment_expert_ids.data_ptr<int>(),
            assignment_weights.data_ptr<float>(),
            num_assignments.data_ptr<int>(),
            selected_experts.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            T, top_k,
            local_expert_offset, num_local_experts
        );
    }

    // Copy num_assignments to host
    int h_num_assignments = 0;
    cudaMemcpyAsync(&h_num_assignments, num_assignments.data_ptr<int>(),
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_num_assignments == 0) {
        // No tokens assigned to local experts
        auto sorted_token_ids = torch::empty({0}, torch::dtype(torch::kInt32).device(hidden_states.device()));
        auto sorted_weights   = torch::empty({0}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
        auto expert_offsets    = torch::zeros({num_local_experts + 1}, torch::dtype(torch::kInt32).device(hidden_states.device()));
        return {sorted_token_ids, sorted_weights, expert_offsets};
    }

    // 4. Histogram: count tokens per expert
    auto expert_counts = torch::zeros({num_local_experts},
        torch::dtype(torch::kInt32).device(hidden_states.device()));

    {
        int threads = 256;
        int blocks = (h_num_assignments + threads - 1) / threads;
        moe_histogram_kernel<<<blocks, threads, 0, stream>>>(
            expert_counts.data_ptr<int>(),
            assignment_expert_ids.data_ptr<int>(),
            h_num_assignments
        );
    }

    // 5. Prefix sum → expert_offsets [num_local_experts + 1]
    auto expert_offsets = torch::zeros({num_local_experts + 1},
        torch::dtype(torch::kInt32).device(hidden_states.device()));
    {
        // Simple inclusive scan on CPU (num_local_experts=20 is tiny)
        std::vector<int> h_counts(num_local_experts);
        cudaMemcpyAsync(h_counts.data(), expert_counts.data_ptr<int>(),
                        num_local_experts * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<int> h_offsets(num_local_experts + 1, 0);
        for (int i = 0; i < num_local_experts; i++)
            h_offsets[i + 1] = h_offsets[i] + h_counts[i];

        cudaMemcpyAsync(expert_offsets.data_ptr<int>(), h_offsets.data(),
                        (num_local_experts + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    // 6. Scatter into sorted order
    auto sorted_token_ids = torch::empty({h_num_assignments},
        torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto sorted_weights   = torch::empty({h_num_assignments},
        torch::dtype(torch::kFloat32).device(hidden_states.device()));
    auto scatter_counters = torch::zeros({num_local_experts},
        torch::dtype(torch::kInt32).device(hidden_states.device()));

    {
        int threads = 256;
        int blocks = (h_num_assignments + threads - 1) / threads;
        moe_scatter_kernel<<<blocks, threads, 0, stream>>>(
            sorted_token_ids.data_ptr<int>(),
            sorted_weights.data_ptr<float>(),
            scatter_counters.data_ptr<int>(),
            expert_offsets.data_ptr<int>(),
            assignment_token_ids.data_ptr<int>(),
            assignment_expert_ids.data_ptr<int>(),
            assignment_weights.data_ptr<float>(),
            h_num_assignments
        );
    }

    return {sorted_token_ids, sorted_weights, expert_offsets};
}

// ---------------------------------------------------------------------------
// Gather sorted activations
// ---------------------------------------------------------------------------

torch::Tensor moe_gather_tokens_op(
    torch::Tensor hidden_states,     // [T, hidden_size] bf16
    torch::Tensor sorted_token_ids,  // [total_assignments] int32
    int hidden_size
) {
    int total_assignments = sorted_token_ids.size(0);
    if (total_assignments == 0) {
        return torch::empty({0, hidden_size},
            torch::dtype(torch::kBFloat16).device(hidden_states.device()));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto sorted_hidden = torch::empty({total_assignments, hidden_size},
        torch::dtype(torch::kBFloat16).device(hidden_states.device()));

    int total = total_assignments * hidden_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    moe_gather_tokens_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(sorted_hidden.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        sorted_token_ids.data_ptr<int>(),
        total_assignments, hidden_size
    );

    return sorted_hidden;
}

// ---------------------------------------------------------------------------
// SiLU fusion
// ---------------------------------------------------------------------------

torch::Tensor moe_silu_fusion_op(
    torch::Tensor gate_up_out,   // [total_assignments, 2 * intermediate_size] bf16
    int intermediate_size        // 2560
) {
    int total_assignments = gate_up_out.size(0);
    if (total_assignments == 0) {
        return torch::empty({0, intermediate_size},
            torch::dtype(torch::kBFloat16).device(gate_up_out.device()));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto intermediate = torch::empty({total_assignments, intermediate_size},
        torch::dtype(torch::kBFloat16).device(gate_up_out.device()));

    int total = total_assignments * intermediate_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    moe_silu_fusion_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(intermediate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up_out.data_ptr()),
        total_assignments, intermediate_size
    );

    return intermediate;
}

// ---------------------------------------------------------------------------
// Scatter-accumulate (down_proj output back to original token order)
// ---------------------------------------------------------------------------

torch::Tensor moe_scatter_accumulate_op(
    torch::Tensor down_out,          // [total_assignments, hidden_size] bf16
    torch::Tensor sorted_token_ids,  // [total_assignments] int32
    torch::Tensor sorted_weights,    // [total_assignments] float32
    int T,                           // number of original tokens
    int hidden_size
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Output accumulator in FP32 (initialized to zero)
    auto output_f32 = torch::zeros({T, hidden_size},
        torch::dtype(torch::kFloat32).device(down_out.device()));

    int total_assignments = sorted_token_ids.size(0);
    if (total_assignments > 0) {
        int total = total_assignments * hidden_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        blocks = std::min(blocks, 65535);

        moe_scatter_accumulate_kernel<<<blocks, threads, 0, stream>>>(
            output_f32.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(down_out.data_ptr()),
            sorted_token_ids.data_ptr<int>(),
            sorted_weights.data_ptr<float>(),
            total_assignments, hidden_size
        );
    }

    // Convert FP32 accumulator to BF16
    return output_f32.to(torch::kBFloat16);
}

// ---------------------------------------------------------------------------
// cuBLAS grouped GEMM for MoE expert projections
//
// Dispatches one cuBLAS GEMM per local expert. Each expert computes:
//   C[T_e, N] = A[T_e, K] × B[N, K]^T
//
// For gate+up:  K=6144, N=5120  (20 experts → 20 cuBLAS calls)
// For down:     K=2560, N=6144  (20 experts → 20 cuBLAS calls)
//
// Total: 40 cuBLAS calls per MoE layer (vs 60+ in naive PyTorch loop)
// ---------------------------------------------------------------------------

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

torch::Tensor moe_gate_up_gemm_op(
    torch::Tensor sorted_hidden,    // [total_sorted, hidden_size] bf16
    torch::Tensor gate_up_weights,  // [num_local_experts, gate_up_width, hidden_size] bf16
    torch::Tensor expert_offsets,   // [num_local_experts + 1] int32
    int hidden_size,                // K = 6144
    int gate_up_width               // N = 5120
) {
    int total_sorted = sorted_hidden.size(0);
    int num_local_experts = expert_offsets.size(0) - 1;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (total_sorted == 0) {
        return torch::empty({0, gate_up_width},
            torch::dtype(torch::kBFloat16).device(sorted_hidden.device()));
    }

    // Copy expert offsets to host
    std::vector<int> h_offsets(num_local_experts + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data_ptr<int>(),
                    (num_local_experts + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto output = torch::zeros({total_sorted, gate_up_width},
        torch::dtype(torch::kBFloat16).device(sorted_hidden.device()));

    auto handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const auto* A_base = reinterpret_cast<const __nv_bfloat16*>(sorted_hidden.data_ptr());
    const auto* B_base = reinterpret_cast<const __nv_bfloat16*>(gate_up_weights.data_ptr());
    auto* C_base = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    float alpha = 1.0f, beta = 0.0f;
    int K = hidden_size;
    int N = gate_up_width;

    for (int e = 0; e < num_local_experts; e++) {
        int T_e = h_offsets[e + 1] - h_offsets[e];
        if (T_e == 0) continue;

        const auto* A_ptr = A_base + h_offsets[e] * (long)K;
        const auto* B_ptr = B_base + e * (long)N * (long)K;
        auto* C_ptr = C_base + h_offsets[e] * (long)N;

        // Row-major: C[T_e, N] = A[T_e, K] × B[N, K]^T
        // cuBLAS col-major: C_col[N, T_e] = B_col^T × A_col
        cublasGemmEx(handle,
            CUBLAS_OP_T,    // transpose B (weight)
            CUBLAS_OP_N,    // no transpose A (activation)
            N, T_e, K,
            &alpha,
            B_ptr, CUDA_R_16BF, K,   // B[N,K] row-major → col[K,N], ld=K
            A_ptr, CUDA_R_16BF, K,   // A[T_e,K] row-major → col[K,T_e], ld=K
            &beta,
            C_ptr, CUDA_R_16BF, N,   // C[T_e,N] row-major → col[N,T_e], ld=N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }

    return output;
}

torch::Tensor moe_down_gemm_op(
    torch::Tensor intermediate,     // [total_sorted, intermediate_size] bf16
    torch::Tensor down_weights,     // [num_local_experts, hidden_size, intermediate_size] bf16
    torch::Tensor expert_offsets,   // [num_local_experts + 1] int32
    int hidden_size,                // N = 6144
    int intermediate_size           // K = 2560
) {
    int total_sorted = intermediate.size(0);
    int num_local_experts = expert_offsets.size(0) - 1;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (total_sorted == 0) {
        return torch::empty({0, hidden_size},
            torch::dtype(torch::kBFloat16).device(intermediate.device()));
    }

    std::vector<int> h_offsets(num_local_experts + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data_ptr<int>(),
                    (num_local_experts + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto output = torch::zeros({total_sorted, hidden_size},
        torch::dtype(torch::kBFloat16).device(intermediate.device()));

    auto handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const auto* A_base = reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr());
    const auto* B_base = reinterpret_cast<const __nv_bfloat16*>(down_weights.data_ptr());
    auto* C_base = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    float alpha = 1.0f, beta = 0.0f;
    int K = intermediate_size;
    int N = hidden_size;

    for (int e = 0; e < num_local_experts; e++) {
        int T_e = h_offsets[e + 1] - h_offsets[e];
        if (T_e == 0) continue;

        const auto* A_ptr = A_base + h_offsets[e] * (long)K;
        const auto* B_ptr = B_base + e * (long)N * (long)K;
        auto* C_ptr = C_base + h_offsets[e] * (long)N;

        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, T_e, K,
            &alpha,
            B_ptr, CUDA_R_16BF, K,
            A_ptr, CUDA_R_16BF, K,
            &beta,
            C_ptr, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }

    return output;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused MoE expert kernel with cuBLAS grouped GEMM + custom fusion";

    // Router + dispatch
    m.def("moe_router_topk",        &moe_router_topk_op,
          "Router matmul + softmax + top-k + sort");
    m.def("moe_gather_tokens",      &moe_gather_tokens_op,
          "Gather sorted activations for contiguous TMA loads");
    m.def("moe_silu_fusion",        &moe_silu_fusion_op,
          "SiLU(gate) * up elementwise fusion");
    m.def("moe_scatter_accumulate", &moe_scatter_accumulate_op,
          "Scatter-accumulate down_proj output to original token order");

    // cuBLAS GEMM (per-expert dispatch, works on all architectures)
    m.def("moe_gate_up_gemm",  &moe_gate_up_gemm_op,
          "cuBLAS GEMM for gate+up projection across all experts");
    m.def("moe_down_gemm",     &moe_down_gemm_op,
          "cuBLAS GEMM for down projection across all experts");
    m.def("has_tcgen05", []() {
#if defined(KITTENS_BLACKWELL)
        return true;
#else
        return false;
#endif
    }, "Check if tcgen05 kernels are available (build flag)");
}
