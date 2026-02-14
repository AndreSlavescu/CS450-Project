"""
Test FA4 attention kernel vs baseline on Modal.

Usage:
    modal run modal_tests/test_fa4_modal.py --gpu b200
    modal run modal_tests/test_fa4_modal.py --gpu h100
"""

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

_target_gpu = "b200"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _target_gpu = sys.argv[i + 1].lower()
        break

app = modal.App("cs450-fa4-test")

GPU_CONFIGS = {
    "h100": {
        "cuda_image": "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        "modal_gpu": "H100",
        "arch": "sm_90a",
        "torch_index": "https://download.pytorch.org/whl/cu128",
    },
    "b200": {
        "cuda_image": "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        "modal_gpu": "B200",
        "arch": "sm_100a",
        "torch_index": "https://download.pytorch.org/whl/nightly/cu130",
    },
}


def _build_image(gpu: str) -> modal.Image:
    cfg = GPU_CONFIGS[gpu]
    return (
        modal.Image.from_registry(cfg["cuda_image"], force_build=force_rebuild)
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install(
            "software-properties-common",
            "build-essential",
            "cmake",
            "ninja-build",
            "git",
            "curl",
        )
        .run_commands(
            "add-apt-repository ppa:deadsnakes/ppa",
            "apt-get update && apt-get install -y python3.12 python3.12-dev",
            "ln -sf /usr/bin/python3.12 /usr/bin/python3 && ln -sf /usr/bin/python3.12 /usr/bin/python",
            "curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12",
        )
        .run_commands(
            "pip install numpy ninja setuptools>=64.0.0",
            f"pip install --pre torch --index-url {cfg['torch_index']}",
        )
        .add_local_dir(
            str(PROJECT_ROOT / "src" / "csrc"), "/workspace/src/csrc"
        )
    )


_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    h100_image = _build_image("h100")
    b200_image = _placeholder
else:
    h100_image = _placeholder
    b200_image = _build_image("b200")


def _jit_compile(name, source_file, arch):
    from torch.utils.cpp_extension import load
    print(f"  Compiling {name}...")
    mod = load(
        name=name,
        sources=[source_file],
        extra_include_paths=["/workspace/src/csrc/profiler"],
        extra_cuda_cflags=["-std=c++20", "-O2", f"-arch={arch}"],
        verbose=False,
    )
    print(f"  {name} compiled OK.")
    return mod


def _run_fa4_test(arch: str) -> dict:
    import math
    import torch

    os.chdir("/workspace")

    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA: {device_name}")
    print(f"{'='*60}")
    print(f"FA4 Attention Kernel Test & Benchmark")
    print(f"{'='*60}")

    # Compile both kernels
    baseline = _jit_compile("attn_baseline", "/workspace/src/csrc/attention_partial.cu", arch)
    fa4 = _jit_compile("attn_fa4", "/workspace/src/csrc/attention_fa4.cu", arch)

    results = {}

    # ============================================================
    # Correctness tests at various sequence lengths
    # ============================================================
    HEAD_DIM = 128
    GQA = 2
    test_seq_lens = [1, 4, 7, 16, 32, 64, 128, 256, 512]

    print(f"\n{'Correctness Tests':-^60}")
    all_correct = True

    for seq_len in test_seq_lens:
        torch.manual_seed(42)
        q = torch.randn(GQA, HEAD_DIM, device="cuda", dtype=torch.float32)
        k_c = torch.randn(seq_len, HEAD_DIM, device="cuda", dtype=torch.float32)
        v_c = torch.randn(seq_len, HEAD_DIM, device="cuda", dtype=torch.float32)
        scale = 1.0 / math.sqrt(HEAD_DIM)

        # PyTorch reference
        scores = (q @ k_c.T) * scale
        probs = torch.softmax(scores, dim=-1)
        ref_out = probs @ v_c

        # Baseline
        base_out, base_lse = baseline.attention_partial_forward(q, k_c, v_c, seq_len, scale)
        base_diff = (base_out - ref_out).abs().max().item()

        # FA4
        fa4_out, fa4_lse = fa4.attention_partial_forward(q, k_c, v_c, seq_len, scale)
        fa4_diff = (fa4_out - ref_out).abs().max().item()

        # Cross-check: FA4 vs baseline
        cross_diff = (fa4_out - base_out).abs().max().item()

        passed = fa4_diff < 1e-3
        if not passed:
            all_correct = False

        print(f"  seq_len={seq_len:4d}: baseline={base_diff:.2e}  fa4={fa4_diff:.2e}  cross={cross_diff:.2e}  {'PASS' if passed else 'FAIL'}")

    results["correctness"] = {"all_pass": all_correct}

    # ============================================================
    # Benchmark
    # ============================================================
    print(f"\n{'Benchmark':-^60}")

    bench_seq_lens = [32, 64, 128, 256, 512]
    bench_results = {}

    for seq_len in bench_seq_lens:
        torch.manual_seed(42)
        q = torch.randn(GQA, HEAD_DIM, device="cuda", dtype=torch.float32)
        k_c = torch.randn(seq_len, HEAD_DIM, device="cuda", dtype=torch.float32)
        v_c = torch.randn(seq_len, HEAD_DIM, device="cuda", dtype=torch.float32)
        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Warmup
        for _ in range(10):
            baseline.attention_partial_forward(q, k_c, v_c, seq_len, scale)
            fa4.attention_partial_forward(q, k_c, v_c, seq_len, scale)
        torch.cuda.synchronize()

        # Benchmark baseline
        N_ITERS = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        for _ in range(N_ITERS):
            baseline.attention_partial_forward(q, k_c, v_c, seq_len, scale)
        end.record()
        torch.cuda.synchronize()
        base_ms = start.elapsed_time(end) / N_ITERS

        # Benchmark FA4
        torch.cuda.synchronize()
        start.record()
        for _ in range(N_ITERS):
            fa4.attention_partial_forward(q, k_c, v_c, seq_len, scale)
        end.record()
        torch.cuda.synchronize()
        fa4_ms = start.elapsed_time(end) / N_ITERS

        speedup = base_ms / fa4_ms if fa4_ms > 0 else 0
        bench_results[seq_len] = {
            "baseline_ms": base_ms,
            "fa4_ms": fa4_ms,
            "speedup": speedup,
        }

        print(f"  seq_len={seq_len:4d}: baseline={base_ms:.4f}ms  fa4={fa4_ms:.4f}ms  speedup={speedup:.2f}x")

    results["benchmark"] = bench_results

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    if all_correct:
        print(f"Correctness: ALL PASS")
    else:
        print(f"Correctness: SOME FAILED")

    avg_speedup = sum(r["speedup"] for r in bench_results.values()) / len(bench_results)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Device: {device_name}")
    print(f"{'='*60}")

    results["device"] = device_name
    results["all_pass"] = all_correct
    results["avg_speedup"] = avg_speedup

    return results


@app.function(image=h100_image, gpu="H100", timeout=900)
def run_h100() -> dict:
    return _run_fa4_test("sm_90a")


@app.function(image=b200_image, gpu="B200", timeout=900)
def run_b200() -> dict:
    return _run_fa4_test("sm_100a")


@app.local_entrypoint()
def main(gpu: str = "b200"):
    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    print(f"Running FA4 test on {gpu.upper()}...\n")

    if gpu == "h100":
        result = run_h100.remote()
    else:
        result = run_b200.remote()

    if result["all_pass"]:
        print(f"\nFA4 kernel: ALL TESTS PASSED on {result['device']}")
        print(f"Average speedup: {result['avg_speedup']:.2f}x")
    else:
        print(f"\nFA4 kernel: SOME TESTS FAILED")
