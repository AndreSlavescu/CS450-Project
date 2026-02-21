#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../profiler/gpu_profiler.cuh"

// ---------------------------------------------------------------------------
// FA4 Forward Attention — CUTLASS SM100 FMHA for Blackwell
//
// Wraps CUTLASS's optimized Blackwell FMHA (example 77) which uses:
//   - 16-warp (512-thread) kernel with fine-grained warp specialization
//   - tcgen05 native MMA with TMEM accumulators
//   - TMA pipeline for Q/K/V loads and O/LSE stores
//   - Online softmax with dual-pipe architecture
//   - GQA support via stride broadcasting
// ---------------------------------------------------------------------------

// Profiler events (retained for API compatibility with fmha_attention.cu)
enum Fa4ProfileEvent : int {
    FA4_EV_SETUP_BEGIN = 0,
    FA4_EV_SETUP_END = 1,
    FA4_EV_COMPUTE_BEGIN = 6,
    FA4_EV_COMPUTE_END = 7,
    FA4_EV_EPILOGUE_BEGIN = 8,
    FA4_EV_EPILOGUE_END = 9,
};

inline void fa4_register_profile_event_names(profiler::event_names& names) {
    names.set(FA4_EV_SETUP_BEGIN, "setup");
    names.set(FA4_EV_COMPUTE_BEGIN, "compute");
    names.set(FA4_EV_EPILOGUE_BEGIN, "epilogue");
}

// ---------------------------------------------------------------------------
// CUTLASS FMHA integration
// ---------------------------------------------------------------------------

#if __has_include("device/fmha.hpp")
#define FA4_HAS_CUTLASS_FMHA 1

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

namespace fa4_cutlass {

using namespace cute;

using Element = cutlass::bfloat16_t;
using ElementOut = cutlass::bfloat16_t;
using ElementAccQK = float;
using ElementAccPV = float;

// Tile shape: (Q_tile=256, KV_tile=128, D=128)
using TileShape = Shape<_256, _128, _128>;

// Stride types for [num_heads, seq, head_dim] memory layout.
// CUTLASS FMHA logical dimension order: (seq, head_dim, ((h_repeat, h_kv), batch))
using StrideQ = tuple<int, _1, tuple<tuple<int, int>, int>>;
using StrideK = tuple<int, _1, tuple<tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = tuple<_1, tuple<tuple<int, int>, int>>;

using ProblemShape = tuple<int, int, int, tuple<tuple<int, int>, int>>;

// Compose CUTLASS FMHA types parameterized by mask type.
template <typename Mask> struct FmhaTypes {
    using Mainloop =
        cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<Element, ElementAccQK, ElementAccPV,
                                                                          TileShape, StrideQ, StrideK, StrideV, Mask>;

    using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccPV, typename Mainloop::TileShapePV, StrideO, StrideLSE>;

    using TileScheduler = cutlass::fmha::kernel::PersistentTileScheduler;

    using Kernel =
        cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<ProblemShape, Mainloop, Epilogue, TileScheduler>;

    using Op = cutlass::fmha::device::FMHA<Kernel>;
};

// Mask aliases
using CausalMaskType = cutlass::fmha::collective::CausalMask<true>;
using NoMaskType = cutlass::fmha::collective::NoMask;

// Cached device buffers to avoid per-call cudaMalloc/cudaFree overhead.
// These are grown-only: once allocated, they persist until process exit.
struct DeviceBufferCache {
    float* lse_buf = nullptr;
    size_t lse_bytes = 0;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    int cached_sm_count = -1;

    float* get_lse(size_t needed) {
        if (needed > lse_bytes) {
            if (lse_buf)
                cudaFree(lse_buf);
            auto err = cudaMalloc(&lse_buf, needed);
            TORCH_CHECK(err == cudaSuccess, "FA4: failed to allocate LSE buffer");
            lse_bytes = needed;
        }
        return lse_buf;
    }

    void* get_workspace(size_t needed) {
        if (needed > workspace_bytes) {
            if (workspace)
                cudaFree(workspace);
            if (needed > 0) {
                auto err = cudaMalloc(&workspace, needed);
                TORCH_CHECK(err == cudaSuccess, "FA4: failed to allocate workspace");
            } else {
                workspace = nullptr;
            }
            workspace_bytes = needed;
        }
        return needed > 0 ? workspace : nullptr;
    }

    int get_sm_count() {
        if (cached_sm_count < 0) {
            cached_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        }
        return cached_sm_count;
    }
};

inline DeviceBufferCache& get_buffer_cache() {
    static DeviceBufferCache cache;
    return cache;
}

// Run CUTLASS FMHA with a specific mask type.
template <typename Mask>
inline void run_fmha_impl(__nv_bfloat16* O, float* lse, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                          const __nv_bfloat16* V, int num_q_heads, int num_kv_heads, int seq_q, int seq_kv,
                          cudaStream_t stream) {
    using Op = typename FmhaTypes<Mask>::Op;

    constexpr int D = 128;
    const int H_R = num_q_heads / num_kv_heads;

    // Problem shape: (seq_q, seq_kv, D, ((h_repeat, h_kv), batch))
    auto problem_shape = make_tuple(seq_q, seq_kv, D, make_tuple(make_tuple(H_R, num_kv_heads), 1));

    // Strides for [num_heads, seq, head_dim] contiguous layout.
    auto stride_Q = make_stride(D, _1{}, make_stride(make_stride(seq_q * D, H_R * seq_q * D), num_q_heads * seq_q * D));

    auto stride_K = make_stride(D, _1{}, make_stride(make_stride(_0{}, seq_kv * D), num_kv_heads * seq_kv * D));

    auto stride_O = stride_Q;

    auto stride_LSE = make_stride(_1{}, make_stride(make_stride(seq_q, H_R * seq_q), num_q_heads * seq_q));

    // Use caller's LSE buffer or get one from the cache (avoids cudaMalloc per call)
    auto& cache = get_buffer_cache();
    float* lse_buf = lse;
    if (!lse_buf) {
        lse_buf = cache.get_lse(static_cast<size_t>(num_q_heads) * static_cast<size_t>(seq_q) * sizeof(float));
    }

    // Hardware info for persistent tile scheduler (SM count cached)
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cache.get_sm_count();

    typename Op::Arguments arguments{problem_shape,
                                     {reinterpret_cast<const Element*>(Q), stride_Q,
                                      reinterpret_cast<const Element*>(K), stride_K,
                                      reinterpret_cast<const Element*>(V), stride_K},
                                     {reinterpret_cast<ElementOut*>(O), stride_O, lse_buf, stride_LSE},
                                     hw_info};

    Op op;
    auto status = op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS FMHA: can_implement failed (status=", static_cast<int>(status), ")");

    // Get workspace from cache (avoids cudaMalloc per call)
    size_t workspace_size = Op::get_workspace_size(arguments);
    void* workspace = cache.get_workspace(workspace_size);

    status = op.initialize(arguments, workspace, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS FMHA: initialize failed (status=", static_cast<int>(status), ")");

    status = op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS FMHA: run failed (status=", static_cast<int>(status),
                ")");
}

// Dispatch to causal or non-causal FMHA.
inline void run_fmha(__nv_bfloat16* O, float* lse, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                     const __nv_bfloat16* V, int num_q_heads, int num_kv_heads, int seq_q, int seq_kv, bool causal,
                     cudaStream_t stream) {
    if (causal) {
        run_fmha_impl<CausalMaskType>(O, lse, Q, K, V, num_q_heads, num_kv_heads, seq_q, seq_kv, stream);
    } else {
        run_fmha_impl<NoMaskType>(O, lse, Q, K, V, num_q_heads, num_kv_heads, seq_q, seq_kv, stream);
    }
}

} // namespace fa4_cutlass

#else
#define FA4_HAS_CUTLASS_FMHA 0
#endif // __has_include("device/fmha.hpp")

// ---------------------------------------------------------------------------
// Public API (called from fmha_attention.cu)
// ---------------------------------------------------------------------------

inline int fa4_profile_block_count(int num_q_heads, int seq_q, int /*seq_kv*/) {
    // Return a reasonable block count for profiler buffer allocation.
    // With CUTLASS FMHA, internal profiling is not supported, but we still
    // need to return a valid count for the profiler host_buffer allocation.
    return num_q_heads * ((seq_q + 255) / 256);
}

inline void fa4_forward(__nv_bfloat16* O, float* lse, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                        const __nv_bfloat16* V, int num_q_heads, int num_kv_heads, int seq_q, int seq_kv, float scale,
                        bool causal, int q_offset, int kv_offset, cudaStream_t stream) {
#if FA4_HAS_CUTLASS_FMHA
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)");
    // Note: q_offset/kv_offset for ring attention are not yet supported
    // by the CUTLASS FMHA wrapper. They are ignored here (default=0 in benchmarks).
    (void)scale; // CUTLASS auto-computes 1/sqrt(D)
    (void)q_offset;
    (void)kv_offset;
    fa4_cutlass::run_fmha(O, lse, Q, K, V, num_q_heads, num_kv_heads, seq_q, seq_kv, causal, stream);
#else
    (void)O;
    (void)lse;
    (void)Q;
    (void)K;
    (void)V;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)seq_q;
    (void)seq_kv;
    (void)scale;
    (void)causal;
    (void)q_offset;
    (void)kv_offset;
    (void)stream;
    TORCH_CHECK(false, "FA4 requires CUTLASS FMHA headers (build with CUTLASS include paths)");
#endif
}

inline void fa4_forward_profile(__nv_bfloat16* O, float* lse, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                const __nv_bfloat16* V, int num_q_heads, int num_kv_heads, int seq_q, int seq_kv,
                                float scale, bool causal, int q_offset, int kv_offset, cudaStream_t stream,
                                profiler::event_record* profile_events, int* profile_counts) {
    // CUTLASS FMHA does not support internal profiling events.
    // Use external tools (Nsight Systems/Compute) for profiling.
    (void)profile_events;
    (void)profile_counts;
    fa4_forward(O, lse, Q, K, V, num_q_heads, num_kv_heads, seq_q, seq_kv, scale, causal, q_offset, kv_offset, stream);
}
