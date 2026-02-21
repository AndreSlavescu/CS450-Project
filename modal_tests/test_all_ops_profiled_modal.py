"""
Profile Qwen3 CUDA ops on Modal and report timing / bandwidth.

Usage:
    modal run modal_tests/test_all_ops_profiled_modal.py --gpu b200
    modal run modal_tests/test_all_ops_profiled_modal.py --gpu h100

    # force rebuild:
    FORCE_REBUILD=1 modal run modal_tests/test_all_ops_profiled_modal.py --gpu b200
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

app = modal.App("cs450-profiled-ops")

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
        .add_local_dir(str(PROJECT_ROOT / "src" / "csrc"), "/workspace/src/csrc")
    )


_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    h100_image = _build_image("h100")
    b200_image = _placeholder
else:
    h100_image = _placeholder
    b200_image = _build_image("b200")


def _jit_compile_unified(arch):
    from torch.utils.cpp_extension import load

    print("\n  Compiling unified qwen3_kernels...")
    mod = load(
        name="qwen3_kernels",
        sources=["/workspace/src/csrc/kernels/qwen3_kernels.cu"],
        extra_include_paths=[
            "/workspace/src/csrc/kernels",
            "/workspace/src/csrc/profiler",
        ],
        extra_cuda_cflags=["-std=c++20", "-O2", f"-arch={arch}"],
        verbose=False,
    )
    print("  qwen3_kernels compiled OK.")
    return mod


def _time_kernel(fn, warmup=5, iters=20):
    """Measure kernel latency using CUDA events. Returns ms."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    for _ in range(iters):
        fn()
    ev_end.record()
    torch.cuda.synchronize()
    return ev_start.elapsed_time(ev_end) / iters


def _run_profiled(arch: str) -> dict:
    import torch

    os.chdir("/workspace")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    kernels = _jit_compile_unified(arch)

    INTER = 6144

    results = {}

    # ============================
    # SiLU-Multiply bandwidth benchmark
    # Bandwidth = (gate_in + up_in + out) = 3 * N * 4 bytes
    # ============================
    silu_sizes = {
        "silu_1k": 1024,
        "silu_hidden": 2048,
        "silu_inter": INTER,
        "silu_batched": 28 * INTER,
    }
    print("\n[SiLU] Bandwidth benchmark (fast PTX approx vs PyTorch F.silu):")
    for label, N in silu_sizes.items():
        torch.manual_seed(42)
        gate = torch.randn(N, device="cuda", dtype=torch.float32)
        up = torch.randn(N, device="cuda", dtype=torch.float32)

        t_custom = _time_kernel(lambda: kernels.silu_multiply(gate, up))
        t_ref = _time_kernel(lambda: torch.nn.functional.silu(gate).mul_(up))

        bytes_transferred = 3 * N * 4  # gate + up reads + out write
        gb_s = bytes_transferred / (t_custom * 1e-3) / 1e9
        speedup = t_ref / t_custom if t_custom > 0 else float("inf")

        results[label] = {
            "N": N,
            "time_custom_ms": round(t_custom, 4),
            "time_ref_ms": round(t_ref, 4),
            "speedup": round(speedup, 2),
            "bandwidth_gb_s": round(gb_s, 1),
        }
        print(
            f"  {label:15s} N={N:7d}: custom={t_custom:.4f}ms "
            f"ref={t_ref:.4f}ms speedup={speedup:.2f}x bw={gb_s:.1f} GB/s"
        )

    return results


@app.function(image=h100_image, gpu="H100", timeout=900)
def run_profiled_h100() -> dict:
    return _run_profiled("sm_90a")


@app.function(image=b200_image, gpu="B200", timeout=1800)
def run_profiled_b200() -> dict:
    return _run_profiled("sm_100a")


@app.local_entrypoint()
def main(gpu: str = "b200"):
    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    print(f"Profiling ops on {gpu.upper()}...\n")

    if gpu == "h100":
        results = run_profiled_h100.remote()
    else:
        results = run_profiled_b200.remote()

    print("\nResults:")
    for label, r in results.items():
        print(
            f"  {label:15s}: {r['time_custom_ms']:.4f}ms "
            f"({r['speedup']:.2f}x speedup, {r['bandwidth_gb_s']:.1f} GB/s)"
        )
