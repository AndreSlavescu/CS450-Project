"""
Profile all Qwen3 CUDA ops on Modal and export Perfetto traces.

Usage:
    modal run modal_tests/test_all_ops_profiled_modal.py --gpu b200
    modal run modal_tests/test_all_ops_profiled_modal.py --gpu h100

    # force rebuild:
    FORCE_REBUILD=1 modal run modal_tests/test_all_ops_profiled_modal.py --gpu b200

Traces are saved locally to src/csrc/profiler/traces/<label>/
Open them at https://ui.perfetto.dev
"""

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

# Label for this profiling run (default: "baseline", override with --label)
_trace_label = "baseline"
_target_gpu = "b200"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _target_gpu = sys.argv[i + 1].lower()
    if arg == "--label" and i + 1 < len(sys.argv):
        _trace_label = sys.argv[i + 1]

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


def _run_profiled(arch: str) -> dict:
    import json
    import torch

    os.chdir("/workspace")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    kernels = _jit_compile_unified(arch)

    HIDDEN = 2048; NQ = 16; NKV = 8; HD = 128
    Q_DIM = NQ * HD; K_DIM = NKV * HD; V_DIM = NKV * HD; QKV_DIM = Q_DIM + K_DIM + V_DIM
    EPS = 1e-6; MAX_SEQ = 128; POS = 5; INTER = 6144

    def rmsnorm_fn(x, w, eps=1e-6):
        return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    traces = {}

    # ===== Op 2: QKV + Q/K Norm + RoPE (profiled) =====
    torch.manual_seed(42)
    hidden = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    attn_ln_w = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    qkv_w = torch.randn(QKV_DIM, HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    q_norm_w = torch.randn(HD, device="cuda", dtype=torch.float32).abs() + 0.5
    k_norm_w = torch.randn(HD, device="cuda", dtype=torch.float32).abs() + 0.5
    k_cache = torch.zeros(MAX_SEQ, K_DIM, device="cuda", dtype=torch.float32)
    v_cache = torch.zeros(MAX_SEQ, V_DIM, device="cuda", dtype=torch.float32)

    inv_freq = 1.0 / (1e6 ** (torch.arange(0, HD, 2, dtype=torch.float32, device="cuda") / HD))
    freqs = inv_freq * POS
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_c, sin_c = emb.cos(), emb.sin()

    trace_path = "/workspace/trace_qkv_rope.json"
    kernels.qkv_rope_append_forward_profiled(
        hidden, attn_ln_w, qkv_w, q_norm_w, k_norm_w, cos_c, sin_c,
        k_cache, v_cache, POS, trace_path
    )
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            traces["qkv_rope"] = json.load(f)
        print(f"[Op 2] QKV trace: {len(traces['qkv_rope'].get('traceEvents', []))} events")

    # ===== Op 5: O-Proj + Residual (profiled) =====
    torch.manual_seed(42)
    hidden_op = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    attn_o = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    o_w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    trace_path = "/workspace/trace_oproj.json"
    kernels.oproj_residual_forward_profiled(hidden_op, attn_o, o_w, trace_path)
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            traces["oproj"] = json.load(f)
        print(f"[Op 5] OProj trace: {len(traces['oproj'].get('traceEvents', []))} events")

    # ===== Op 6: Upgate + SiLU (profiled) =====
    torch.manual_seed(42)
    hidden_mlp = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    mlp_ln = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    gate_w = torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    up_w = torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    trace_path = "/workspace/trace_upgate_silu.json"
    kernels.upgate_silu_forward_profiled(hidden_mlp, mlp_ln, gate_w, up_w, trace_path)
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            traces["upgate_silu"] = json.load(f)
        print(f"[Op 6] Upgate+SiLU trace: {len(traces['upgate_silu'].get('traceEvents', []))} events")

    # ===== Op 7: Down Proj + Residual (profiled) =====
    torch.manual_seed(42)
    hidden_dp = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    silu_in = torch.randn(INTER, device="cuda", dtype=torch.float32)
    down_w = torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.float32) * 0.01

    trace_path = "/workspace/trace_downproj.json"
    kernels.downproj_residual_forward_profiled(hidden_dp, silu_in, down_w, trace_path)
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            traces["downproj"] = json.load(f)
        print(f"[Op 7] DownProj trace: {len(traces['downproj'].get('traceEvents', []))} events")

    # ===== Op 8: RMS + LM Head (profiled) =====
    TEST_VOCAB = 1024
    torch.manual_seed(42)
    hidden_lm = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    lm_w = torch.randn(TEST_VOCAB, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    trace_path = "/workspace/trace_rms_lm_head.json"
    kernels.rms_lm_head_forward_profiled(hidden_lm, norm_w, lm_w, TEST_VOCAB, trace_path)
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            traces["rms_lm_head"] = json.load(f)
        print(f"[Op 8] RMS+LMHead trace: {len(traces['rms_lm_head'].get('traceEvents', []))} events")

    print(f"\nCollected {len(traces)} traces")
    return traces


@app.function(image=h100_image, gpu="H100", timeout=900)
def run_profiled_h100() -> dict:
    return _run_profiled("sm_90a")


@app.function(image=b200_image, gpu="B200", timeout=1800)
def run_profiled_b200() -> dict:
    return _run_profiled("sm_100a")


@app.local_entrypoint()
def main(gpu: str = "b200", label: str = "baseline"):
    import json

    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    print(f"Profiling all ops on {gpu.upper()} (label: {label})...\n")

    if gpu == "h100":
        traces = run_profiled_h100.remote()
    else:
        traces = run_profiled_b200.remote()

    # Save traces locally
    traces_dir = PROJECT_ROOT / "src" / "csrc" / "profiler" / "traces" / label
    traces_dir.mkdir(parents=True, exist_ok=True)

    for op_name, trace_data in traces.items():
        out_path = traces_dir / f"{op_name}_{gpu}.json"
        with open(out_path, "w") as f:
            json.dump(trace_data, f, indent=2)
        n_events = len(trace_data.get("traceEvents", []))
        print(f"  Saved {op_name}: {n_events} events -> {out_path}")

    print(f"\nAll traces saved to {traces_dir}/")
    print(f"Open https://ui.perfetto.dev and load any .json file to visualize")
