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

// tcgen05 persistent GEMM kernel (SM100 / Blackwell)
#if defined(KITTENS_BLACKWELL)
#include "moe_expert.cuh"

// -----------------------------------------------------------------------
// tcgen05 Persistent GEMM Integration
//
// Wraps launch_moe_gemm() with padding logic:
//   1. Pad expert boundaries to CLUSTER_M (512) alignment for TMA
//   2. Copy compact input → padded positions (vectorized float4)
//   3. Launch single persistent tcgen05 GEMM kernel across all experts
//   4. Extract compact output from padded positions
//
// Replaces 20 sequential cuBLAS calls with ONE persistent kernel launch.
// -----------------------------------------------------------------------

static constexpr int TCGEN05_CLUSTER_M = 4 * MOE_TILE_M; // 512

// Scatter rows from compact [total_sorted, W] to padded [padded_total, W]
__global__ void moe_compact_to_padded_kernel(
    __nv_bfloat16* __restrict__ padded_out,
    const __nv_bfloat16* __restrict__ compact_in,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ padded_offsets,
    int total_sorted, int width, int num_local_experts)
{
    int row = blockIdx.x;
    if (row >= total_sorted) return;

    // Binary search for expert owning this row
    int lo = 0, hi = num_local_experts;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (expert_offsets[mid + 1] <= row) lo = mid + 1;
        else hi = mid;
    }
    int padded_row = padded_offsets[lo] + (row - expert_offsets[lo]);

    const float4* src = reinterpret_cast<const float4*>(compact_in + (long)row * width);
    float4* dst = reinterpret_cast<float4*>(padded_out + (long)padded_row * width);
    int nvec = width >> 3; // 8 bf16 per float4
    for (int i = threadIdx.x; i < nvec; i += blockDim.x)
        dst[i] = src[i];
}

// Gather rows from padded [padded_total, W] back to compact [total_sorted, W]
__global__ void moe_padded_to_compact_kernel(
    __nv_bfloat16* __restrict__ compact_out,
    const __nv_bfloat16* __restrict__ padded_in,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ padded_offsets,
    int total_sorted, int width, int num_local_experts)
{
    int row = blockIdx.x;
    if (row >= total_sorted) return;

    int lo = 0, hi = num_local_experts;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (expert_offsets[mid + 1] <= row) lo = mid + 1;
        else hi = mid;
    }
    int padded_row = padded_offsets[lo] + (row - expert_offsets[lo]);

    const float4* src = reinterpret_cast<const float4*>(padded_in + (long)padded_row * width);
    float4* dst = reinterpret_cast<float4*>(compact_out + (long)row * width);
    int nvec = width >> 3;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x)
        dst[i] = src[i];
}

// Main tcgen05 GEMM: pad → persistent kernel → extract
static torch::Tensor moe_gemm_tcgen05(
    torch::Tensor sorted_input,            // [total_sorted, k_dim] bf16
    torch::Tensor weights,                 // [num_local_experts, n_dim, k_dim] bf16
    const std::vector<int>& h_offsets,     // expert_offsets already on host
    torch::Tensor expert_offsets_d,        // device copy
    int k_dim, int n_dim, cudaStream_t stream)
{
    int total_sorted = sorted_input.size(0);
    int num_local_experts = (int)expert_offsets_d.size(0) - 1;
    auto dev = sorted_input.device();

    // Build padded offsets + scheduler on host
    std::vector<int> h_padded(num_local_experts + 1, 0);
    MoETaskScheduler sched = {};
    sched.num_local_experts = num_local_experts;
    sched.n_tiles = n_dim / MOE_TILE_N;
    sched.expert_offsets[0] = 0;
    sched.expert_tile_offsets[0] = 0;

    for (int e = 0; e < num_local_experts; e++) {
        int count = h_offsets[e + 1] - h_offsets[e];
        int padded = count > 0
            ? ((count + TCGEN05_CLUSTER_M - 1) / TCGEN05_CLUSTER_M) * TCGEN05_CLUSTER_M
            : 0;
        h_padded[e + 1] = h_padded[e] + padded;
        sched.expert_offsets[e + 1] = h_padded[e + 1];
        int m_tiles = padded / TCGEN05_CLUSTER_M;
        sched.expert_tile_offsets[e + 1] =
            sched.expert_tile_offsets[e] + m_tiles * sched.n_tiles;
    }
    sched.total_tasks = sched.expert_tile_offsets[num_local_experts];
    int padded_total = h_padded[num_local_experts];

    if (sched.total_tasks == 0 || padded_total == 0)
        return torch::zeros({total_sorted, n_dim}, torch::dtype(torch::kBFloat16).device(dev));

    // Allocate padded buffers (zero-init for padding rows)
    auto padded_input = torch::zeros({padded_total, k_dim}, torch::dtype(torch::kBFloat16).device(dev));
    auto padded_output = torch::zeros({padded_total, n_dim}, torch::dtype(torch::kBFloat16).device(dev));

    // Padded offsets → device
    auto padded_offsets_d = torch::empty({num_local_experts + 1}, torch::dtype(torch::kInt32).device(dev));
    cudaMemcpyAsync(padded_offsets_d.data_ptr<int>(), h_padded.data(),
                    (num_local_experts + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Scheduler → device
    MoETaskScheduler* d_sched;
    cudaMallocAsync(&d_sched, sizeof(MoETaskScheduler), stream);
    cudaMemcpyAsync(d_sched, &sched, sizeof(MoETaskScheduler), cudaMemcpyHostToDevice, stream);

    // Step 1: Compact → Padded (vectorized, one block per row)
    if (total_sorted > 0) {
        moe_compact_to_padded_kernel<<<total_sorted, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(padded_input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(sorted_input.data_ptr()),
            expert_offsets_d.data_ptr<int>(), padded_offsets_d.data_ptr<int>(),
            total_sorted, k_dim, num_local_experts);
    }

    // Step 2: Persistent tcgen05 GEMM (single launch, all experts)
    launch_moe_gemm(
        reinterpret_cast<__nv_bfloat16*>(padded_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(padded_input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr()),
        d_sched, padded_total, num_local_experts, k_dim, n_dim, stream);

    // Step 3: Padded → Compact (vectorized, one block per row)
    auto output = torch::empty({total_sorted, n_dim}, torch::dtype(torch::kBFloat16).device(dev));
    if (total_sorted > 0) {
        moe_padded_to_compact_kernel<<<total_sorted, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(padded_output.data_ptr()),
            expert_offsets_d.data_ptr<int>(), padded_offsets_d.data_ptr<int>(),
            total_sorted, n_dim, num_local_experts);
    }

    cudaFreeAsync(d_sched, stream);
    return output;
}
#endif // KITTENS_BLACKWELL

// ---------------------------------------------------------------------------
// cuBLAS handle (shared by router matmul and expert GEMMs)
// ---------------------------------------------------------------------------

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

// ---------------------------------------------------------------------------
// Router + Top-K + Sort
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> moe_router_topk_op(torch::Tensor hidden_states, // [T, hidden_size]  bf16
                                              torch::Tensor router_weight, // [num_experts, hidden_size]  bf16
                                              int top_k, bool norm_topk_prob, int local_expert_offset,
                                              int num_local_experts) {
    TORCH_CHECK(hidden_states.is_cuda() && router_weight.is_cuda());
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);

    int T = hidden_states.size(0);
    int hidden_size = hidden_states.size(1);
    int num_experts = router_weight.size(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 1. Router matmul → logits [T, num_experts] in FP32
    //    C[T, N] = A[T, K] × B[N, K]^T   where N=num_experts, K=hidden_size
    //    Uses cuBLAS tensor cores instead of custom warp-per-row kernel (~3x faster)
    auto router_logits = torch::empty({T, num_experts}, torch::dtype(torch::kFloat32).device(hidden_states.device()));

    {
        auto handle = get_cublas_handle();
        cublasSetStream(handle, stream);
        float alpha = 1.0f, beta = 0.0f;
        // Row-major: C[T,N] = A[T,K] × B[N,K]^T
        // cuBLAS col-major: C_col[N,T] = B_col^T × A_col
        cublasGemmEx(handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     num_experts, T, hidden_size,           // m=N, n=T, k=K
                     &alpha,
                     reinterpret_cast<const __nv_bfloat16*>(router_weight.data_ptr()),
                     CUDA_R_16BF, hidden_size,              // B[N,K] row-major → col[K,N], ld=K
                     reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
                     CUDA_R_16BF, hidden_size,              // A[T,K] row-major → col[K,T], ld=K
                     &beta,
                     router_logits.data_ptr<float>(),
                     CUDA_R_32F, num_experts,               // C[T,N] row-major → col[N,T], ld=N
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // 2. Softmax + Top-K → selected_experts [T, top_k], routing_weights [T, top_k]
    auto selected_experts = torch::empty({T, top_k}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto routing_weights = torch::empty({T, top_k}, torch::dtype(torch::kFloat32).device(hidden_states.device()));

    {
        int smem = (num_experts + top_k) * sizeof(float) + top_k * sizeof(int);
        moe_softmax_topk_kernel_launch<<<T, 32, smem, stream>>>(
            selected_experts.data_ptr<int>(), routing_weights.data_ptr<float>(), router_logits.data_ptr<float>(), T,
            num_experts, top_k, norm_topk_prob);
    }

    // 3. Build assignments (filter to local experts)
    int max_assignments = T * top_k;
    auto assignment_token_ids =
        torch::empty({max_assignments}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto assignment_expert_ids =
        torch::empty({max_assignments}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto assignment_weights =
        torch::empty({max_assignments}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
    auto num_assignments = torch::zeros({1}, torch::dtype(torch::kInt32).device(hidden_states.device()));

    {
        int total = T * top_k;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        moe_build_assignment_kernel_launch<<<blocks, threads, 0, stream>>>(
            assignment_token_ids.data_ptr<int>(), assignment_expert_ids.data_ptr<int>(),
            assignment_weights.data_ptr<float>(), num_assignments.data_ptr<int>(), selected_experts.data_ptr<int>(),
            routing_weights.data_ptr<float>(), T, top_k, local_expert_offset, num_local_experts);
    }

    // Copy num_assignments to host
    int h_num_assignments = 0;
    cudaMemcpyAsync(&h_num_assignments, num_assignments.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_num_assignments == 0) {
        // No tokens assigned to local experts
        auto sorted_token_ids = torch::empty({0}, torch::dtype(torch::kInt32).device(hidden_states.device()));
        auto sorted_weights = torch::empty({0}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
        auto expert_offsets =
            torch::zeros({num_local_experts + 1}, torch::dtype(torch::kInt32).device(hidden_states.device()));
        return {sorted_token_ids, sorted_weights, expert_offsets};
    }

    // 4. Histogram: count tokens per expert
    auto expert_counts = torch::zeros({num_local_experts}, torch::dtype(torch::kInt32).device(hidden_states.device()));

    {
        int threads = 256;
        int blocks = (h_num_assignments + threads - 1) / threads;
        moe_histogram_kernel_launch<<<blocks, threads, 0, stream>>>(
            expert_counts.data_ptr<int>(), assignment_expert_ids.data_ptr<int>(), h_num_assignments);
    }

    // 5. GPU prefix scan → expert_offsets [num_local_experts + 1]
    //    Replaces CPU scan + host sync — single block kernel for 20 elements
    auto expert_offsets =
        torch::zeros({num_local_experts + 1}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    moe_prefix_scan_kernel_launch<<<1, 32, 0, stream>>>(
        expert_offsets.data_ptr<int>(), expert_counts.data_ptr<int>(), num_local_experts);

    // 6. Scatter into sorted order
    auto sorted_token_ids =
        torch::empty({h_num_assignments}, torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto sorted_weights =
        torch::empty({h_num_assignments}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
    auto scatter_counters =
        torch::zeros({num_local_experts}, torch::dtype(torch::kInt32).device(hidden_states.device()));

    {
        int threads = 256;
        int blocks = (h_num_assignments + threads - 1) / threads;
        moe_scatter_kernel_launch<<<blocks, threads, 0, stream>>>(
            sorted_token_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(), scatter_counters.data_ptr<int>(),
            expert_offsets.data_ptr<int>(), assignment_token_ids.data_ptr<int>(), assignment_expert_ids.data_ptr<int>(),
            assignment_weights.data_ptr<float>(), h_num_assignments);
    }

    return {sorted_token_ids, sorted_weights, expert_offsets};
}

// ---------------------------------------------------------------------------
// Gather sorted activations
// ---------------------------------------------------------------------------

torch::Tensor moe_gather_tokens_op(torch::Tensor hidden_states,    // [T, hidden_size] bf16
                                   torch::Tensor sorted_token_ids, // [total_assignments] int32
                                   int hidden_size) {
    int total_assignments = sorted_token_ids.size(0);
    if (total_assignments == 0) {
        return torch::empty({0, hidden_size}, torch::dtype(torch::kBFloat16).device(hidden_states.device()));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto sorted_hidden =
        torch::empty({total_assignments, hidden_size}, torch::dtype(torch::kBFloat16).device(hidden_states.device()));

    // Vectorized: one block per row, float4 (8 bf16) per thread
    moe_gather_tokens_kernel_launch<<<total_assignments, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(sorted_hidden.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()), sorted_token_ids.data_ptr<int>(),
        total_assignments, hidden_size);

    return sorted_hidden;
}

// ---------------------------------------------------------------------------
// SiLU fusion
// ---------------------------------------------------------------------------

torch::Tensor moe_silu_fusion_op(torch::Tensor gate_up_out, // [total_assignments, 2 * intermediate_size] bf16
                                 int intermediate_size      // 2560
) {
    int total_assignments = gate_up_out.size(0);
    if (total_assignments == 0) {
        return torch::empty({0, intermediate_size}, torch::dtype(torch::kBFloat16).device(gate_up_out.device()));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto intermediate = torch::empty({total_assignments, intermediate_size},
                                     torch::dtype(torch::kBFloat16).device(gate_up_out.device()));

    // Vectorized: one block per row, float4 (8 bf16) SiLU+mul per thread
    moe_silu_fusion_kernel_launch<<<total_assignments, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(intermediate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up_out.data_ptr()), total_assignments, intermediate_size);

    return intermediate;
}

// ---------------------------------------------------------------------------
// Scatter-accumulate (down_proj output back to original token order)
// ---------------------------------------------------------------------------

torch::Tensor moe_scatter_accumulate_op(torch::Tensor down_out,         // [total_assignments, hidden_size] bf16
                                        torch::Tensor sorted_token_ids, // [total_assignments] int32
                                        torch::Tensor sorted_weights,   // [total_assignments] float32
                                        int T,                          // number of original tokens
                                        int hidden_size) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Output accumulator in FP32 (initialized to zero)
    auto output_f32 = torch::zeros({T, hidden_size}, torch::dtype(torch::kFloat32).device(down_out.device()));

    int total_assignments = sorted_token_ids.size(0);
    if (total_assignments > 0) {
        // Vectorized: one block per assignment row, float4 loads + scalar FP32 atomics
        moe_scatter_accumulate_kernel_launch<<<total_assignments, 256, 0, stream>>>(
            output_f32.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(down_out.data_ptr()),
            sorted_token_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(), total_assignments, hidden_size);
    }

    // Convert FP32 accumulator to BF16
    return output_f32.to(torch::kBFloat16);
}

// ---------------------------------------------------------------------------
// MoE Expert GEMM Dispatch
//
// On B200 (KITTENS_BLACKWELL): uses the tcgen05 persistent GEMM kernel
//   — single kernel launch processes all 20 local experts
//   — 2-CTA cluster, persistent tile scheduler, TMA loads
//   — Set MOE_FORCE_CUBLAS=1 to force cuBLAS fallback for debugging
//
// Fallback (H100/A100): cuBLAS per-expert GEMM loop
//   — 20 cuBLAS calls per projection
// ---------------------------------------------------------------------------

#if defined(KITTENS_BLACKWELL)
static bool use_tcgen05_gemm() {
    static int val = -1;
    if (val < 0) {
        const char* env = getenv("MOE_FORCE_CUBLAS");
        val = (env && env[0] == '1') ? 0 : 1;
    }
    return val != 0;
}
#endif

// cuBLAS fallback: one GEMM per expert
static torch::Tensor moe_gemm_cublas(
    torch::Tensor sorted_input,    // [total_sorted, k_dim]
    torch::Tensor weights,         // [num_local_experts, n_dim, k_dim]
    const std::vector<int>& h_offsets,
    int k_dim, int n_dim, cudaStream_t stream)
{
    int total_sorted = sorted_input.size(0);
    int num_local_experts = (int)h_offsets.size() - 1;

    auto output = torch::zeros({total_sorted, n_dim},
                               torch::dtype(torch::kBFloat16).device(sorted_input.device()));

    auto handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const auto* A_base = reinterpret_cast<const __nv_bfloat16*>(sorted_input.data_ptr());
    const auto* B_base = reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr());
    auto* C_base = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    float alpha = 1.0f, beta = 0.0f;

    for (int e = 0; e < num_local_experts; e++) {
        int T_e = h_offsets[e + 1] - h_offsets[e];
        if (T_e == 0) continue;

        const auto* A_ptr = A_base + h_offsets[e] * (long)k_dim;
        const auto* B_ptr = B_base + e * (long)n_dim * (long)k_dim;
        auto* C_ptr = C_base + h_offsets[e] * (long)n_dim;

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     n_dim, T_e, k_dim, &alpha,
                     B_ptr, CUDA_R_16BF, k_dim,
                     A_ptr, CUDA_R_16BF, k_dim,
                     &beta, C_ptr, CUDA_R_16BF, n_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    return output;
}

torch::Tensor moe_gate_up_gemm_op(torch::Tensor sorted_hidden,   // [total_sorted, hidden_size] bf16
                                  torch::Tensor gate_up_weights, // [num_local_experts, gate_up_width, hidden_size] bf16
                                  torch::Tensor expert_offsets,  // [num_local_experts + 1] int32
                                  int hidden_size,               // K = 6144
                                  int gate_up_width              // N = 5120
) {
    int total_sorted = sorted_hidden.size(0);
    int num_local_experts = expert_offsets.size(0) - 1;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (total_sorted == 0)
        return torch::empty({0, gate_up_width}, torch::dtype(torch::kBFloat16).device(sorted_hidden.device()));

    // Copy expert offsets to host (one sync, shared by both paths)
    std::vector<int> h_offsets(num_local_experts + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data_ptr<int>(),
                    (num_local_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

#if defined(KITTENS_BLACKWELL)
    if (use_tcgen05_gemm())
        return moe_gemm_tcgen05(sorted_hidden, gate_up_weights, h_offsets, expert_offsets,
                                hidden_size, gate_up_width, stream);
#endif
    return moe_gemm_cublas(sorted_hidden, gate_up_weights, h_offsets, hidden_size, gate_up_width, stream);
}

torch::Tensor moe_down_gemm_op(torch::Tensor intermediate,   // [total_sorted, intermediate_size] bf16
                               torch::Tensor down_weights,   // [num_local_experts, hidden_size, intermediate_size] bf16
                               torch::Tensor expert_offsets, // [num_local_experts + 1] int32
                               int hidden_size,              // N = 6144
                               int intermediate_size         // K = 2560
) {
    int total_sorted = intermediate.size(0);
    int num_local_experts = expert_offsets.size(0) - 1;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (total_sorted == 0)
        return torch::empty({0, hidden_size}, torch::dtype(torch::kBFloat16).device(intermediate.device()));

    std::vector<int> h_offsets(num_local_experts + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data_ptr<int>(),
                    (num_local_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

#if defined(KITTENS_BLACKWELL)
    if (use_tcgen05_gemm())
        return moe_gemm_tcgen05(intermediate, down_weights, h_offsets, expert_offsets,
                                intermediate_size, hidden_size, stream);
#endif
    return moe_gemm_cublas(intermediate, down_weights, h_offsets, intermediate_size, hidden_size, stream);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused MoE expert kernel with cuBLAS grouped GEMM + custom fusion";

    // Router + dispatch
    m.def("moe_router_topk", &moe_router_topk_op, "Router matmul + softmax + top-k + sort");
    m.def("moe_gather_tokens", &moe_gather_tokens_op, "Gather sorted activations for contiguous TMA loads");
    m.def("moe_silu_fusion", &moe_silu_fusion_op, "SiLU(gate) * up elementwise fusion");
    m.def("moe_scatter_accumulate", &moe_scatter_accumulate_op,
          "Scatter-accumulate down_proj output to original token order");

    // cuBLAS GEMM (per-expert dispatch, works on all architectures)
    m.def("moe_gate_up_gemm", &moe_gate_up_gemm_op, "cuBLAS GEMM for gate+up projection across all experts");
    m.def("moe_down_gemm", &moe_down_gemm_op, "cuBLAS GEMM for down projection across all experts");
    m.def(
        "has_tcgen05",
        []() {
#if defined(KITTENS_BLACKWELL)
            return true;
#else
            return false;
#endif
        },
        "Check if tcgen05 kernels are available (build flag)");
}
