"""
Test all Qwen3 standalone CUDA ops on Modal.

Usage:
    modal run modal_tests/test_all_ops_modal.py --gpu b200
    modal run modal_tests/test_all_ops_modal.py --gpu h100

    # force rebuild:
    FORCE_REBUILD=1 modal run modal_tests/test_all_ops_modal.py --gpu b200
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

app = modal.App("cs450-all-ops-test")

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
    img = (
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
    )
    if gpu == "b200":
        # Clone CUTLASS for Blackwell SM100 FMHA kernel headers (FA4 attention)
        img = img.run_commands("git clone --depth 1 https://github.com/NVIDIA/cutlass.git /workspace/cutlass")
    return img.add_local_dir(str(PROJECT_ROOT / "src" / "csrc"), "/workspace/src/csrc")


_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    h100_image = _build_image("h100")
    b200_image = _placeholder
else:
    h100_image = _placeholder
    b200_image = _build_image("b200")


def _jit_compile_unified(arch):
    """Compile the unified qwen3_kernels.cu driver."""
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


def _jit_compile(name, source_file, arch):
    """Compile a standalone kernel (for attention ops that haven't been refactored)."""
    from torch.utils.cpp_extension import load

    print(f"\n  Compiling {name}...")
    mod = load(
        name=name,
        sources=[source_file],
        extra_include_paths=[
            "/workspace/src/csrc/profiler",
            "/workspace/src/csrc/kernels",
        ],
        extra_cuda_cflags=["-std=c++20", "-O2", f"-arch={arch}"],
        verbose=False,
    )
    print(f"  {name} compiled OK.")
    return mod


def _jit_compile_fa4(arch):
    """Compile FA4 attention kernel with CUTLASS SM100 FMHA headers (B200 only)."""
    from torch.utils.cpp_extension import load

    print("\n  Compiling fmha_attention (CUTLASS FMHA)...")
    mod = load(
        name="fmha_attention",
        sources=["/workspace/src/csrc/kernels/fmha_attention.cu"],
        extra_include_paths=[
            "/workspace/src/csrc/profiler",
            "/workspace/src/csrc/kernels",
            "/workspace/cutlass/include",
            "/workspace/cutlass/examples/77_blackwell_fmha",
            "/workspace/cutlass/tools/util/include",
        ],
        extra_cuda_cflags=["-std=c++17", "-O2", f"-arch={arch}"],
        verbose=False,
    )
    print("  fmha_attention compiled OK.")
    return mod


def _run_all_ops(arch: str) -> dict:
    import math

    import torch

    os.chdir("/workspace")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    results = {}

    # Compile the unified persistent kernel driver (SiLU + full-model decode)
    kernels = _jit_compile_unified(arch)

    HIDDEN = 2048
    NQ = 16
    NKV = 8
    HD = 128
    Q_DIM = NQ * HD
    K_DIM = NKV * HD
    V_DIM = NKV * HD
    QKV_DIM = Q_DIM + K_DIM + V_DIM
    INTER = 6144
    NUM_LAYERS = 28

    def rmsnorm_fn(x, w, eps=1e-6):
        return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    # ============================
    # SiLU-Multiply: correctness + speedup at Qwen3-relevant sizes
    # Fast PTX approx (ex2+rcp) vs PyTorch F.silu — tolerance 1e-3 due to approx.
    # ============================
    silu_shapes = {
        "silu_inter": INTER,  # 6144  — Qwen3-1.7B MLP intermediate
        "silu_batched": 28 * INTER,  # 28×6144 — full-layer batch
    }
    for label, N in silu_shapes.items():
        torch.manual_seed(42)
        gate_t = torch.randn(N, device="cuda", dtype=torch.float32)
        up_t = torch.randn(N, device="cuda", dtype=torch.float32)
        ref_s = torch.nn.functional.silu(gate_t) * up_t

        # Warmup
        for _ in range(3):
            kernels.silu_multiply(gate_t, up_t)
        torch.cuda.synchronize()

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        out_s = kernels.silu_multiply(gate_t, up_t)
        ev_end.record()
        torch.cuda.synchronize()
        t_custom_ms = ev_start.elapsed_time(ev_end)

        ev_start.record()
        _ = torch.nn.functional.silu(gate_t) * up_t
        ev_end.record()
        torch.cuda.synchronize()
        t_ref_ms = ev_start.elapsed_time(ev_end)

        diff_s = (out_s - ref_s).abs().max().item()
        speedup = t_ref_ms / t_custom_ms if t_custom_ms > 0 else float("inf")
        passed = diff_s < 1e-3
        results[label] = {"max_diff": diff_s, "pass": passed}
        status = "PASS" if passed else "FAIL"
        print(
            f"[SiLU/{label}] N={N}: max_diff={diff_s:.2e} "
            f"custom={t_custom_ms:.3f}ms ref={t_ref_ms:.3f}ms "
            f"speedup={speedup:.2f}x {status}"
        )

    # ============================
    # Persistent decode kernel — smoke test
    # Verifies the kernel launches without errors and returns a finite logit tensor.
    # Full correctness would require real model weights; here we use small random
    # weights and check for NaN/Inf and correct output shape.
    # ============================
    VOCAB = 1024
    MAX_SEQ_P = 8  # small seq for fast allocation
    POS_P = 4  # decode at position 4 (cache positions 0..4 already filled)

    torch.manual_seed(99)

    # RoPE cos/sin cache [MAX_SEQ_P, HD]
    inv_freq_p = 1.0 / (1e6 ** (torch.arange(0, HD, 2, dtype=torch.float32, device="cuda") / HD))
    cos_cache_p = torch.zeros(MAX_SEQ_P, HD, device="cuda", dtype=torch.float32)
    sin_cache_p = torch.zeros(MAX_SEQ_P, HD, device="cuda", dtype=torch.float32)
    for p in range(MAX_SEQ_P):
        emb_p = torch.cat([inv_freq_p * p, inv_freq_p * p])
        cos_cache_p[p] = emb_p.cos()
        sin_cache_p[p] = emb_p.sin()

    hidden_p = torch.randn(HIDDEN, device="cuda", dtype=torch.float32) * 0.1
    attn_ln_ws_p = torch.ones(NUM_LAYERS, HIDDEN, device="cuda", dtype=torch.float32)
    qkv_ws_p = torch.randn(NUM_LAYERS, QKV_DIM * HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    q_norm_ws_p = torch.ones(NUM_LAYERS, HD, device="cuda", dtype=torch.float32)
    k_norm_ws_p = torch.ones(NUM_LAYERS, HD, device="cuda", dtype=torch.float32)
    k_caches_p = torch.zeros(NUM_LAYERS, MAX_SEQ_P * K_DIM, device="cuda", dtype=torch.float32)
    v_caches_p = torch.zeros(NUM_LAYERS, MAX_SEQ_P * V_DIM, device="cuda", dtype=torch.float32)
    o_proj_ws_p = torch.randn(NUM_LAYERS, HIDDEN * HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    mlp_ln_ws_p = torch.ones(NUM_LAYERS, HIDDEN, device="cuda", dtype=torch.float32)
    gate_ws_p = torch.randn(NUM_LAYERS, INTER * HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    up_ws_p = torch.randn(NUM_LAYERS, INTER * HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    down_ws_p = torch.randn(NUM_LAYERS, HIDDEN * INTER, device="cuda", dtype=torch.float32) * 0.01
    norm_w_p = torch.ones(HIDDEN, device="cuda", dtype=torch.float32)
    lm_head_w_p = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    logits_p = kernels.qwen3_decode_persistent_forward(
        hidden_p,
        attn_ln_ws_p,
        qkv_ws_p,
        q_norm_ws_p,
        k_norm_ws_p,
        cos_cache_p,
        sin_cache_p,
        k_caches_p,
        v_caches_p,
        o_proj_ws_p,
        mlp_ln_ws_p,
        gate_ws_p,
        up_ws_p,
        down_ws_p,
        norm_w_p,
        lm_head_w_p,
        POS_P,
    )
    torch.cuda.synchronize()
    persist_ok = (
        list(logits_p.shape) == [VOCAB] and not logits_p.isnan().any().item() and not logits_p.isinf().any().item()
    )
    results["persistent_decode"] = {"max_diff": 0.0, "pass": persist_ok}
    status = "PASS" if persist_ok else "FAIL"
    print(
        f"[Persistent] Decode kernel: shape={list(logits_p.shape)} finite={not logits_p.isnan().any().item()} {status}"
    )

    # ============================
    # FA4 Attention (B200 / CUTLASS SM100 FMHA only)
    # ============================
    if arch == "sm_100a":
        fa4 = _jit_compile_fa4(arch)

        FA4_SEQ = 1024
        FA4_NQ = 16
        FA4_NKV = 8
        FA4_HD = 128
        FA4_SCALE = 1.0 / math.sqrt(FA4_HD)
        torch.manual_seed(42)
        Q_fa4 = torch.randn(FA4_NQ, FA4_SEQ, FA4_HD, device="cuda", dtype=torch.bfloat16)
        K_fa4 = torch.randn(FA4_NKV, FA4_SEQ, FA4_HD, device="cuda", dtype=torch.bfloat16)
        V_fa4 = torch.randn(FA4_NKV, FA4_SEQ, FA4_HD, device="cuda", dtype=torch.bfloat16)

        # Reference: PyTorch scaled_dot_product_attention (GQA, causal)
        # sdpa expects [batch, heads, seq, head_dim] — use batch=1
        Q_sdpa = Q_fa4.unsqueeze(0)  # [1, NQ, SEQ, HD]
        K_sdpa = K_fa4.unsqueeze(0)  # [1, NKV, SEQ, HD]
        V_sdpa = V_fa4.unsqueeze(0)  # [1, NKV, SEQ, HD]
        ref_out = torch.nn.functional.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa, is_causal=True).squeeze(
            0
        )  # [NQ, SEQ, HD]

        fa4_out = fa4.forward(Q_fa4, K_fa4, V_fa4, FA4_SCALE, True, False)[0]
        fa4_diff = (fa4_out.float() - ref_out.float()).abs().max().item()
        results["fmha_attention"] = {"max_diff": fa4_diff, "pass": fa4_diff < 0.01}
        print(f"[FA4] CUTLASS FMHA: max_diff={fa4_diff:.8f} {'PASS' if fa4_diff < 0.01 else 'FAIL'}")

    # ============================
    # Summary
    # ============================
    all_pass = all(r["pass"] for r in results.values())
    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*50}")

    return {"all_pass": all_pass, "results": results}


@app.function(image=h100_image, gpu="H100", timeout=900)
def run_all_h100() -> dict:
    return _run_all_ops("sm_90a")


@app.function(image=b200_image, gpu="B200", timeout=1800)
def run_all_b200() -> dict:
    return _run_all_ops("sm_100a")


@app.local_entrypoint()
def main(gpu: str = "b200"):
    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    print(f"Running all ops test on {gpu.upper()}...\n")

    if gpu == "h100":
        result = run_all_h100.remote()
    else:
        result = run_all_b200.remote()

    print("\nFinal Results:")
    for name, r in result["results"].items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {name:20s}: max_diff={r['max_diff']:.8f} [{status}]")

    if result["all_pass"]:
        print("\nAll ops PASSED!")
    else:
        print("\nSome ops FAILED!")
