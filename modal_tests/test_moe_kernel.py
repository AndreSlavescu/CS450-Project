"""Synthetic MoE kernel test + benchmark on a single B200.

Tests the kernel build, correctness (vs PyTorch), and performance
without needing the full 480B model weights.

Uses cuBLAS grouped GEMM for expert projections with custom fusion
kernels for SiLU and scatter-accumulate.

Usage:
    modal run modal_tests/test_moe_kernel.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent
force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

app = modal.App("cs450-moe-kernel-test")

test_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.b200", force_build=force_rebuild)
    .pip_install("transformers>=4.51.0,<5.0", "accelerate", "sentencepiece")
    .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
)


@app.function(image=test_image, gpu="B200", timeout=1200)
def run_moe_kernel_test():
    import subprocess
    import sys
    import time

    import torch

    results = {}

    # ══════════════════════════════════════════════════════════════════
    # Step 0: GPU Diagnostics
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Step 0: GPU Diagnostics")
    print("=" * 60)

    drv_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"  GPU: {drv_info.stdout.strip()}")

    cuda_ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    for line in cuda_ver.stdout.strip().split("\n"):
        if "release" in line.lower():
            print(f"  CUDA: {line.strip()}")

    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    props = torch.cuda.get_device_properties(dev)
    print(f"  Compute capability: {cap[0]}.{cap[1]}")
    print(f"  SM count: {props.multi_processor_count}")
    print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA (torch): {torch.version.cuda}")

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Build the MoE kernel
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 1: Building MoE kernel (cuBLAS GEMM + custom fusion)")
    print("=" * 60)

    torch_incs = subprocess.check_output(
        [
            "python3",
            "-c",
            "from torch.utils.cpp_extension import include_paths; print(' '.join('-I' + p for p in include_paths()))",
        ],
        text=True,
    ).strip()
    torch_libs = subprocess.check_output(
        [
            "python3",
            "-c",
            "from torch.utils.cpp_extension import library_paths; print(' '.join('-L' + p for p in library_paths()))",
        ],
        text=True,
    ).strip()
    extra_flags = f"{torch_incs} {torch_libs} -ltorch -ltorch_cpu -lc10 -ltorch_python"

    build_result = subprocess.run(
        [
            "make",
            "-C",
            "/workspace/src/csrc/kernels",
            "GPU=B200",
            f"EXTRA_NVCCFLAGS={extra_flags}",
            "moe_expert.cpython-312-x86_64-linux-gnu.so",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    print("STDOUT:", build_result.stdout[-2000:] if len(build_result.stdout) > 2000 else build_result.stdout)
    if build_result.returncode != 0:
        print("STDERR:", build_result.stderr[-3000:] if len(build_result.stderr) > 3000 else build_result.stderr)
        results["build"] = "FAILED"
        results["build_error"] = build_result.stderr[-2000:]
        return results

    print("Build successful!")
    results["build"] = "PASSED"

    # Import the kernel
    sys.path.insert(0, "/workspace/src/csrc/kernels")
    import moe_expert as moe

    print(f"  has_tcgen05() = {moe.has_tcgen05()}")

    # Disable tcgen05 persistent GEMM by default for correctness testing.
    # Set MOE_FORCE_CUBLAS=0 to enable tcgen05 path.
    if "MOE_FORCE_CUBLAS" not in os.environ:
        os.environ["MOE_FORCE_CUBLAS"] = "1"
    print(f"  MOE_FORCE_CUBLAS = {os.environ.get('MOE_FORCE_CUBLAS', '0')}")

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Test non-GEMM components (router, gather, silu, scatter)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 2: Testing non-GEMM components")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda:0"

    # Test dimensions (Qwen3-480B scale)
    T = 256  # tokens
    hidden_size = 6144
    num_experts = 160
    num_local_experts = 20
    local_expert_offset = 0
    top_k = 8
    intermediate_size = 2560

    hidden_states = torch.randn(T, hidden_size, dtype=torch.bfloat16, device=device)
    router_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device)

    # 2a. Router + top-k + sort
    print("  Testing moe_router_topk...", end=" ", flush=True)
    sorted_ids, sorted_weights, expert_offsets = moe.moe_router_topk(
        hidden_states,
        router_weight,
        top_k,
        True,
        local_expert_offset,
        num_local_experts,
    )
    total_assignments = sorted_ids.shape[0]
    print(f"OK  (total_assignments={total_assignments})")
    assert total_assignments > 0, "No tokens assigned to local experts"
    assert expert_offsets.shape[0] == num_local_experts + 1
    offsets_cpu = expert_offsets.cpu().tolist()
    for i in range(len(offsets_cpu) - 1):
        assert offsets_cpu[i] <= offsets_cpu[i + 1], f"Non-monotonic offsets at {i}"
    assert offsets_cpu[-1] == total_assignments
    results["router"] = "PASSED"

    # 2b. Gather tokens
    print("  Testing moe_gather_tokens...", end=" ", flush=True)
    sorted_hidden = moe.moe_gather_tokens(hidden_states, sorted_ids, hidden_size)
    assert sorted_hidden.shape == (total_assignments, hidden_size)
    ref_gathered = hidden_states[sorted_ids.long()]
    max_err = (sorted_hidden.float() - ref_gathered.float()).abs().max().item()
    print(f"OK  (max_err={max_err:.2e})")
    assert max_err < 1e-4, f"Gather error too large: {max_err}"
    results["gather"] = "PASSED"

    # 2c. SiLU fusion
    print("  Testing moe_silu_fusion...", end=" ", flush=True)
    gate_up_fake = torch.randn(total_assignments, 2 * intermediate_size, dtype=torch.bfloat16, device=device)
    silu_out = moe.moe_silu_fusion(gate_up_fake, intermediate_size)
    assert silu_out.shape == (total_assignments, intermediate_size)
    gate = gate_up_fake[:, :intermediate_size].float()
    up = gate_up_fake[:, intermediate_size:].float()
    ref_silu = (gate * torch.sigmoid(gate)) * up
    max_err = (silu_out.float() - ref_silu).abs().max().item()
    rel_err = max_err / (ref_silu.abs().max().item() + 1e-8)
    print(f"OK  (max_err={max_err:.2e}, rel_err={rel_err:.2e})")
    assert rel_err < 0.05, f"SiLU relative error too large: {rel_err}"
    results["silu"] = "PASSED"

    # 2d. Scatter-accumulate
    print("  Testing moe_scatter_accumulate...", end=" ", flush=True)
    down_fake = torch.randn(total_assignments, hidden_size, dtype=torch.bfloat16, device=device)
    scatter_out = moe.moe_scatter_accumulate(
        down_fake,
        sorted_ids,
        sorted_weights,
        T,
        hidden_size,
    )
    assert scatter_out.shape == (T, hidden_size)
    ref_scatter = torch.zeros(T, hidden_size, dtype=torch.float32, device=device)
    for i in range(total_assignments):
        tok = sorted_ids[i].item()
        w = sorted_weights[i].item()
        ref_scatter[tok] += w * down_fake[i].float()
    ref_scatter_bf16 = ref_scatter.bfloat16()
    max_err = (scatter_out.float() - ref_scatter_bf16.float()).abs().max().item()
    rel_err = max_err / (ref_scatter_bf16.float().abs().max().item() + 1e-8)
    print(f"OK  (max_err={max_err:.2e}, rel_err={rel_err:.2e})")
    assert rel_err < 0.05, f"Scatter relative error too large: {rel_err}"
    results["scatter"] = "PASSED"

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Test cuBLAS GEMM kernels
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 3: Testing cuBLAS grouped GEMM kernels")
    print("=" * 60)

    # 3a. Gate+Up GEMM: [T_sorted, 6144] x [20, 5120, 6144]^T -> [T_sorted, 5120]
    print("  Testing moe_gate_up_gemm...", end=" ", flush=True)
    gate_up_width = 2 * intermediate_size  # 5120
    gate_up_weights = torch.randn(
        num_local_experts,
        gate_up_width,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    gate_up_out = moe.moe_gate_up_gemm(
        sorted_hidden,
        gate_up_weights,
        expert_offsets,
        hidden_size,
        gate_up_width,
    )
    assert gate_up_out.shape == (total_assignments, gate_up_width)

    # Reference: per-expert PyTorch matmul
    ref_gate_up = torch.zeros(total_assignments, gate_up_width, dtype=torch.float32, device=device)
    for e in range(num_local_experts):
        start = offsets_cpu[e]
        end = offsets_cpu[e + 1]
        if start == end:
            continue
        a = sorted_hidden[start:end].float()
        b = gate_up_weights[e].float()
        ref_gate_up[start:end] = a @ b.T

    ref_bf16 = ref_gate_up.bfloat16()
    max_err = (gate_up_out.float() - ref_bf16.float()).abs().max().item()
    rel_err = max_err / (ref_bf16.float().abs().max().item() + 1e-8)
    print(f"OK  (max_err={max_err:.2e}, rel_err={rel_err:.2e})")
    if rel_err >= 0.1:
        print(f"    WARNING: rel_err={rel_err:.4f} is high (BF16 precision expected ~0.01-0.05)")
    results["gate_up_gemm"] = f"rel_err={rel_err:.4e}"

    # 3b. Down GEMM: [T_sorted, 2560] x [20, 6144, 2560]^T -> [T_sorted, 6144]
    print("  Testing moe_down_gemm...", end=" ", flush=True)
    intermediate_in = torch.randn(
        total_assignments,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    down_weights = torch.randn(
        num_local_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )

    down_out = moe.moe_down_gemm(
        intermediate_in,
        down_weights,
        expert_offsets,
        hidden_size,
        intermediate_size,
    )
    assert down_out.shape == (total_assignments, hidden_size)

    ref_down = torch.zeros(total_assignments, hidden_size, dtype=torch.float32, device=device)
    for e in range(num_local_experts):
        start = offsets_cpu[e]
        end = offsets_cpu[e + 1]
        if start == end:
            continue
        a = intermediate_in[start:end].float()
        b = down_weights[e].float()
        ref_down[start:end] = a @ b.T

    ref_bf16 = ref_down.bfloat16()
    max_err = (down_out.float() - ref_bf16.float()).abs().max().item()
    rel_err = max_err / (ref_bf16.float().abs().max().item() + 1e-8)
    print(f"OK  (max_err={max_err:.2e}, rel_err={rel_err:.2e})")
    if rel_err >= 0.1:
        print(f"    WARNING: rel_err={rel_err:.4f} is high")
    results["down_gemm"] = f"rel_err={rel_err:.4e}"

    # ══════════════════════════════════════════════════════════════════
    # Step 4: End-to-end MoE layer test (shared routing)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 4: End-to-end MoE layer (cuBLAS kernel vs PyTorch)")
    print("=" * 60)

    def pytorch_moe_from_routing(
        hidden, experts_gate_up, experts_down, sorted_ids, sorted_weights_t, expert_offsets_t, n_local, T_in
    ):
        """PyTorch MoE compute using pre-computed routing (same as CUDA path)."""
        H = hidden.shape[1]
        inter_size = experts_down.shape[2]
        offs = expert_offsets_t.cpu().tolist()

        sorted_hidden = hidden[sorted_ids.long()]  # [total_assignments, H]
        output = torch.zeros(T_in, H, dtype=torch.float32, device=hidden.device)

        for e in range(n_local):
            start, end = offs[e], offs[e + 1]
            if start == end:
                continue
            tokens = sorted_hidden[start:end].float()
            gate_w = experts_gate_up[e, :inter_size].float()
            up_w = experts_gate_up[e, inter_size:].float()
            down_w = experts_down[e].float()

            gate_out = tokens @ gate_w.T
            up_out = tokens @ up_w.T
            silu_out = (gate_out * torch.sigmoid(gate_out)) * up_out
            down_out = silu_out @ down_w.T

            # Scatter-accumulate with routing weights
            for i in range(start, end):
                tok_idx = sorted_ids[i].item()
                w = sorted_weights_t[i].item()
                output[tok_idx] += w * down_out[i - start]

        return output.bfloat16()

    def cuda_moe_forward(hidden, router_w, experts_gate_up, experts_down, top_k, local_offset, n_local, norm_topk):
        """MoE forward using cuBLAS GEMM + custom fusion kernel pipeline."""
        T_in, H = hidden.shape
        inter_size = experts_down.shape[2]

        sorted_ids, sorted_w, exp_offs = moe.moe_router_topk(
            hidden,
            router_w,
            top_k,
            norm_topk,
            local_offset,
            n_local,
        )

        n_assign = sorted_ids.shape[0]
        if n_assign == 0:
            return torch.zeros(T_in, H, dtype=torch.bfloat16, device=hidden.device)

        sorted_h = moe.moe_gather_tokens(hidden, sorted_ids, H)
        gate_up = moe.moe_gate_up_gemm(sorted_h, experts_gate_up, exp_offs, H, 2 * inter_size)
        inter = moe.moe_silu_fusion(gate_up, inter_size)
        down = moe.moe_down_gemm(inter, experts_down, exp_offs, H, inter_size)
        output = moe.moe_scatter_accumulate(down, sorted_ids, sorted_w, T_in, H)
        return output

    T_e2e = 256
    H_e2e = 6144
    I_e2e = 2560
    n_experts_total = 160
    n_local_e2e = 20
    top_k_e2e = 8

    hidden_e2e = torch.randn(T_e2e, H_e2e, dtype=torch.bfloat16, device=device)
    router_w_e2e = torch.randn(n_experts_total, H_e2e, dtype=torch.bfloat16, device=device) * 0.01
    gate_up_w_e2e = torch.randn(n_local_e2e, 2 * I_e2e, H_e2e, dtype=torch.bfloat16, device=device) * 0.01
    down_w_e2e = torch.randn(n_local_e2e, H_e2e, I_e2e, dtype=torch.bfloat16, device=device) * 0.01

    # 4a. Router accuracy: compare CUDA router vs PyTorch router
    print("  4a. Router comparison (CUDA vs PyTorch)...", flush=True)
    import torch.nn.functional as F

    pt_logits = hidden_e2e.float() @ router_w_e2e.float().T
    pt_probs = F.softmax(pt_logits, dim=-1)
    _, pt_topk_experts = torch.topk(pt_probs, top_k_e2e, dim=-1)

    sorted_ids_e2e, sorted_w_e2e, exp_offs_e2e = moe.moe_router_topk(
        hidden_e2e,
        router_w_e2e,
        top_k_e2e,
        True,
        0,
        n_local_e2e,
    )
    print(f"    CUDA total_assignments = {sorted_ids_e2e.shape[0]}")

    pt_local_count = 0
    for t in range(T_e2e):
        for k in range(top_k_e2e):
            e = pt_topk_experts[t, k].item()
            if 0 <= e < n_local_e2e:
                pt_local_count += 1
    cuda_local_count = sorted_ids_e2e.shape[0]
    print(f"    PyTorch local assignments: {pt_local_count}")
    print(f"    CUDA local assignments:    {cuda_local_count}")
    results["router_comparison"] = f"pt={pt_local_count}, cuda={cuda_local_count}"

    # 4b. E2E with shared routing (tests GEMM + SiLU + scatter pipeline)
    print("  4b. E2E with shared routing (isolates compute from routing)...", flush=True)

    print("    Running PyTorch reference (shared routing)...", flush=True)
    ref_out = pytorch_moe_from_routing(
        hidden_e2e,
        gate_up_w_e2e,
        down_w_e2e,
        sorted_ids_e2e,
        sorted_w_e2e,
        exp_offs_e2e,
        n_local_e2e,
        T_e2e,
    )

    print("    Running cuBLAS kernel pipeline (shared routing)...", flush=True)
    sorted_h_e2e = moe.moe_gather_tokens(hidden_e2e, sorted_ids_e2e, H_e2e)
    gate_up_e2e = moe.moe_gate_up_gemm(sorted_h_e2e, gate_up_w_e2e, exp_offs_e2e, H_e2e, 2 * I_e2e)
    inter_e2e = moe.moe_silu_fusion(gate_up_e2e, I_e2e)
    down_e2e = moe.moe_down_gemm(inter_e2e, down_w_e2e, exp_offs_e2e, H_e2e, I_e2e)
    cuda_out = moe.moe_scatter_accumulate(down_e2e, sorted_ids_e2e, sorted_w_e2e, T_e2e, H_e2e)

    diff = (cuda_out.float() - ref_out.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_norm = ref_out.float().abs().max().item()
    rel_err = max_err / (ref_norm + 1e-8)

    print(f"    Max absolute error: {max_err:.4e}")
    print(f"    Mean absolute error: {mean_err:.4e}")
    print(f"    Reference max: {ref_norm:.4e}")
    print(f"    Relative error: {rel_err:.4e}")

    if rel_err < 0.1:
        print("    PASSED (within BF16 tolerance)")
        results["e2e"] = "PASSED"
    else:
        print(f"    WARNING: relative error {rel_err:.4f} is high")
        results["e2e"] = f"HIGH_ERROR (rel={rel_err:.4e})"

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Performance benchmarks
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 5: Performance benchmarks")
    print("=" * 60)

    WARMUP = 5
    ITERS = 20

    # 5a. Gate+Up GEMM benchmark
    print("\n  Gate+Up GEMM: [T_sorted, 6144] x [5120, 6144]^T -> [T_sorted, 5120]")
    for T_bench in [256, 512, 1024, 2048, 4096]:
        tokens_per_expert = T_bench // num_local_experts
        if tokens_per_expert < 1:
            continue

        offs = [0]
        for e in range(num_local_experts):
            offs.append(offs[-1] + tokens_per_expert)
        total_sorted = offs[-1]
        exp_offs_t = torch.tensor(offs, dtype=torch.int32, device=device)

        sorted_h = torch.randn(total_sorted, hidden_size, dtype=torch.bfloat16, device=device)

        for _ in range(WARMUP):
            moe.moe_gate_up_gemm(sorted_h, gate_up_weights, exp_offs_t, hidden_size, gate_up_width)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            moe.moe_gate_up_gemm(sorted_h, gate_up_weights, exp_offs_t, hidden_size, gate_up_width)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / ITERS * 1000
        flops = 2.0 * total_sorted * gate_up_width * hidden_size
        tflops = flops / (ms / 1000) / 1e12
        print(f"    T={T_bench:>5} ({total_sorted:>5} sorted):  {ms:.3f} ms  ({tflops:.1f} TFLOPS)")
        results[f"gate_up_gemm_T{T_bench}"] = {"ms": ms, "tflops": tflops}

    # 5b. Down GEMM benchmark
    print("\n  Down GEMM: [T_sorted, 2560] x [6144, 2560]^T -> [T_sorted, 6144]")
    for T_bench in [256, 512, 1024, 2048, 4096]:
        tokens_per_expert = T_bench // num_local_experts
        if tokens_per_expert < 1:
            continue

        offs = [0]
        for e in range(num_local_experts):
            offs.append(offs[-1] + tokens_per_expert)
        total_sorted = offs[-1]
        exp_offs_t = torch.tensor(offs, dtype=torch.int32, device=device)

        inter_in = torch.randn(total_sorted, intermediate_size, dtype=torch.bfloat16, device=device)

        for _ in range(WARMUP):
            moe.moe_down_gemm(inter_in, down_weights, exp_offs_t, hidden_size, intermediate_size)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            moe.moe_down_gemm(inter_in, down_weights, exp_offs_t, hidden_size, intermediate_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / ITERS * 1000
        flops = 2.0 * total_sorted * hidden_size * intermediate_size
        tflops = flops / (ms / 1000) / 1e12
        print(f"    T={T_bench:>5} ({total_sorted:>5} sorted):  {ms:.3f} ms  ({tflops:.1f} TFLOPS)")
        results[f"down_gemm_T{T_bench}"] = {"ms": ms, "tflops": tflops}

    # 5c. Full MoE layer benchmark (end-to-end)
    print("\n  Full MoE layer (router + gather + gate_up + silu + down + scatter):")
    for T_bench in [256, 512, 1024, 2048, 4096]:
        hidden_bench = torch.randn(T_bench, hidden_size, dtype=torch.bfloat16, device=device) * 0.02

        for _ in range(WARMUP):
            cuda_moe_forward(
                hidden_bench,
                router_weight,
                gate_up_weights,
                down_weights,
                top_k,
                0,
                num_local_experts,
                True,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            cuda_moe_forward(
                hidden_bench,
                router_weight,
                gate_up_weights,
                down_weights,
                top_k,
                0,
                num_local_experts,
                True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / ITERS * 1000
        print(f"    T={T_bench:>5}:  {ms:.3f} ms")
        results[f"full_moe_T{T_bench}"] = {"ms": ms}

    # 5d. PyTorch baseline for comparison
    def pytorch_moe_forward_bench(
        hidden, router_w, experts_gate_up, experts_down, top_k_b, local_offset, n_local, norm_topk
    ):
        """Pure PyTorch MoE forward (for benchmarking)."""
        T_in, H = hidden.shape
        router_logits = hidden.float() @ router_w.float().T
        probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_experts = torch.topk(probs, top_k_b, dim=-1)
        if norm_topk:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden.dtype)

        output = torch.zeros(T_in, H, dtype=torch.float32, device=hidden.device)
        inter_size = experts_down.shape[2]

        for local_idx in range(n_local):
            global_idx = local_offset + local_idx
            mask = topk_experts == global_idx
            token_mask = mask.any(dim=-1)
            if not token_mask.any():
                continue
            weight = (topk_weights.float() * mask.float()).sum(dim=-1)
            tokens = hidden[token_mask].float()

            gate_w = experts_gate_up[local_idx, :inter_size].float()
            up_w = experts_gate_up[local_idx, inter_size:].float()
            down_w = experts_down[local_idx].float()

            gate_out = tokens @ gate_w.T
            up_out = tokens @ up_w.T
            silu_out = (gate_out * torch.sigmoid(gate_out)) * up_out
            down_out = silu_out @ down_w.T

            output[token_mask] += weight[token_mask].unsqueeze(-1) * down_out

        return output.bfloat16()

    print("\n  PyTorch baseline (loop over experts, nn.Linear equivalent):")
    for T_bench in [256, 512, 1024, 2048, 4096]:
        hidden_bench = torch.randn(T_bench, hidden_size, dtype=torch.bfloat16, device=device) * 0.02

        for _ in range(WARMUP):
            pytorch_moe_forward_bench(
                hidden_bench,
                router_weight,
                gate_up_weights,
                down_weights,
                top_k,
                0,
                num_local_experts,
                True,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            pytorch_moe_forward_bench(
                hidden_bench,
                router_weight,
                gate_up_weights,
                down_weights,
                top_k,
                0,
                num_local_experts,
                True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / ITERS * 1000
        print(f"    T={T_bench:>5}:  {ms:.3f} ms")
        results[f"pytorch_moe_T{T_bench}"] = {"ms": ms}

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Summary + comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Summary: cuBLAS MoE kernel vs PyTorch baseline")
    print("=" * 60)
    print(f"{'Tokens':>8} {'cuBLAS (ms)':>14} {'PyTorch (ms)':>14} {'Speedup':>10}")
    print("-" * 60)
    for T_bench in [256, 512, 1024, 2048, 4096]:
        cuda_key = f"full_moe_T{T_bench}"
        pt_key = f"pytorch_moe_T{T_bench}"
        if cuda_key in results and pt_key in results:
            cuda_ms = results[cuda_key]["ms"]
            pt_ms = results[pt_key]["ms"]
            speedup = pt_ms / cuda_ms if cuda_ms > 0 else 0
            print(f"{T_bench:>8} {cuda_ms:>14.3f} {pt_ms:>14.3f} {speedup:>9.2f}x")

    print("\n" + "=" * 60)
    print("Per-layer TTFT impact (Qwen3-480B has 62 MoE layers):")
    print("=" * 60)
    if "full_moe_T4096" in results:
        layer_ms = results["full_moe_T4096"]["ms"]
        total_ms = layer_ms * 62
        print(f"  4K-token prefill, per MoE layer: {layer_ms:.3f} ms")
        print(f"  Total MoE compute (62 layers):   {total_ms:.1f} ms")
        print(f"  Estimated TTFT contribution:     {total_ms:.1f} ms")

    return results


@app.local_entrypoint()
def main():
    result = run_moe_kernel_test.remote()
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    if result.get("build") == "FAILED":
        print(f"BUILD FAILED: {result.get('build_error', 'unknown')}")
    else:
        for key in ["router", "gather", "silu", "scatter", "e2e"]:
            if key in result:
                print(f"  {key}: {result[key]}")
        for key in ["gate_up_gemm", "down_gemm"]:
            if key in result:
                print(f"  {key}: {result[key]}")
