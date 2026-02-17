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
        extra_include_paths=["/workspace/src/csrc/profiler"],
        extra_cuda_cflags=["-std=c++20", "-O2", f"-arch={arch}"],
        verbose=False,
    )
    print(f"  {name} compiled OK.")
    return mod


def _run_all_ops(arch: str) -> dict:
    import math
    import torch

    os.chdir("/workspace")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    results = {}

    # Compile the unified driver (Ops 2, 5, 6, 7, 8 + SiLU)
    kernels = _jit_compile_unified(arch)

    HIDDEN = 2048; NQ = 16; NKV = 8; HD = 128
    Q_DIM = NQ * HD; K_DIM = NKV * HD; V_DIM = NKV * HD; QKV_DIM = Q_DIM + K_DIM + V_DIM
    EPS = 1e-6; MAX_SEQ = 128; POS = 5; INTER = 6144

    def rmsnorm_fn(x, w, eps=1e-6):
        return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    # ============================
    # Op 2: QKV + Q/K Norm + RoPE
    # ============================
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

    ref_pln = rmsnorm_fn(hidden, attn_ln_w, EPS)
    ref_qkv = qkv_w @ ref_pln
    ref_q = ref_qkv[:Q_DIM].view(NQ, HD)
    ref_k = ref_qkv[Q_DIM:Q_DIM+K_DIM].view(NKV, HD)
    ref_v = ref_qkv[Q_DIM+K_DIM:].view(NKV, HD)
    for h in range(NQ):
        ref_q[h] = rmsnorm_fn(ref_q[h], q_norm_w, EPS)
    for h in range(NKV):
        ref_k[h] = rmsnorm_fn(ref_k[h], k_norm_w, EPS)
    ref_q = ref_q * cos_c + rotate_half(ref_q) * sin_c
    ref_k = ref_k * cos_c + rotate_half(ref_k) * sin_c

    q_out, k_out, v_out = kernels.qkv_rope_append_forward(
        hidden, attn_ln_w, qkv_w, q_norm_w, k_norm_w, cos_c, sin_c, k_cache, v_cache, POS
    )

    q_diff = (q_out - ref_q.view(-1)).abs().max().item()
    k_diff = (k_out - ref_k.view(-1)).abs().max().item()
    v_diff = (v_out - ref_v.view(-1)).abs().max().item()
    max_diff = max(q_diff, k_diff, v_diff)
    results["qkv_rope"] = {"q": q_diff, "k": k_diff, "v": v_diff, "max_diff": max_diff, "pass": max_diff < 1e-2}
    print(f"[Op 2] QKV+Norm+RoPE: q={q_diff:.6f} k={k_diff:.6f} v={v_diff:.6f} {'PASS' if max_diff < 1e-2 else 'FAIL'}")

    # ============================
    # Op 3: Partial Attention (still compiled standalone — not our responsibility)
    # ============================
    attn_cuda = _jit_compile("attn_partial_cuda", "/workspace/src/csrc/attention_partial.cu", arch)

    SEQ_LEN = 32
    GQA = 2
    torch.manual_seed(42)
    q_attn = torch.randn(GQA, HD, device="cuda", dtype=torch.float32)
    k_c = torch.randn(SEQ_LEN, HD, device="cuda", dtype=torch.float32)
    v_c = torch.randn(SEQ_LEN, HD, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(HD)

    scores = (q_attn @ k_c.T) * scale
    probs = torch.softmax(scores, dim=-1)
    ref_attn_out = probs @ v_c

    attn_out, lse = attn_cuda.attention_partial_forward(q_attn, k_c, v_c, SEQ_LEN, scale)
    attn_diff = (attn_out - ref_attn_out).abs().max().item()
    results["attention"] = {"max_diff": attn_diff, "pass": attn_diff < 1e-3}
    print(f"[Op 3] Attention: max_diff={attn_diff:.8f} {'PASS' if attn_diff < 1e-3 else 'FAIL'}")

    # ============================
    # Op 4: Attention Reduction (still compiled standalone — not our responsibility)
    # ============================
    attn_red = _jit_compile("attn_red_cuda", "/workspace/src/csrc/attention_reduction.cu", arch)

    N_PARTS = 4
    torch.manual_seed(42)
    p_outs = torch.randn(N_PARTS, GQA, HD, device="cuda", dtype=torch.float32)
    p_lses = torch.randn(N_PARTS, GQA, device="cuda", dtype=torch.float32)

    ref_reduced = torch.zeros(GQA, HD, device="cuda", dtype=torch.float32)
    for h in range(GQA):
        max_lse = p_lses[:, h].max()
        weights = torch.exp(p_lses[:, h] - max_lse)
        wsum = weights.sum()
        for p in range(N_PARTS):
            ref_reduced[h] += weights[p] * p_outs[p, h]
        ref_reduced[h] /= wsum

    red_out = attn_red.attention_reduction_forward(p_outs, p_lses, N_PARTS)
    red_diff = (red_out - ref_reduced).abs().max().item()
    results["attn_reduction"] = {"max_diff": red_diff, "pass": red_diff < 1e-3}
    print(f"[Op 4] Attn Reduction: max_diff={red_diff:.8f} {'PASS' if red_diff < 1e-3 else 'FAIL'}")

    # ============================
    # Op 5: O-Proj + Residual
    # ============================
    torch.manual_seed(42)
    hidden_op = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    attn_o = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    o_w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    ref_oproj = hidden_op + o_w @ attn_o
    out_oproj = kernels.oproj_residual_forward(hidden_op, attn_o, o_w)
    oproj_diff = (out_oproj - ref_oproj).abs().max().item()
    results["oproj"] = {"max_diff": oproj_diff, "pass": oproj_diff < 1e-2}
    print(f"[Op 5] O-Proj+Residual: max_diff={oproj_diff:.8f} {'PASS' if oproj_diff < 1e-2 else 'FAIL'}")

    # ============================
    # Op 6: Upgate + SiLU
    # ============================
    torch.manual_seed(42)
    hidden_mlp = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    mlp_ln = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    gate_w = torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.float32) * 0.01
    up_w = torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    ref_pln2 = rmsnorm_fn(hidden_mlp, mlp_ln, EPS)
    ref_gate = gate_w @ ref_pln2
    ref_up = up_w @ ref_pln2
    ref_silu = torch.nn.functional.silu(ref_gate) * ref_up

    out_silu = kernels.upgate_silu_forward(hidden_mlp, mlp_ln, gate_w, up_w)
    silu_diff = (out_silu - ref_silu).abs().max().item()
    results["upgate_silu"] = {"max_diff": silu_diff, "pass": silu_diff < 1e-2}
    print(f"[Op 6] Upgate+SiLU: max_diff={silu_diff:.8f} {'PASS' if silu_diff < 1e-2 else 'FAIL'}")

    # ============================
    # Op 7: Down Proj + Residual
    # ============================
    torch.manual_seed(42)
    hidden_dp = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    silu_in = torch.randn(INTER, device="cuda", dtype=torch.float32)
    down_w = torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.float32) * 0.01

    ref_dp = hidden_dp + down_w @ silu_in
    out_dp = kernels.downproj_residual_forward(hidden_dp, silu_in, down_w)
    dp_diff = (out_dp - ref_dp).abs().max().item()
    results["downproj"] = {"max_diff": dp_diff, "pass": dp_diff < 1e-2}
    print(f"[Op 7] DownProj+Residual: max_diff={dp_diff:.8f} {'PASS' if dp_diff < 1e-2 else 'FAIL'}")

    # ============================
    # Op 8: RMS + LM Head
    # ============================
    TEST_VOCAB = 1024
    torch.manual_seed(42)
    hidden_lm = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(HIDDEN, device="cuda", dtype=torch.float32)
    lm_w = torch.randn(TEST_VOCAB, HIDDEN, device="cuda", dtype=torch.float32) * 0.01

    ref_pln3 = rmsnorm_fn(hidden_lm, norm_w, EPS)
    ref_logits = lm_w @ ref_pln3
    out_logits = kernels.rms_lm_head_forward(hidden_lm, norm_w, lm_w, TEST_VOCAB)
    lm_diff = (out_logits - ref_logits).abs().max().item()
    results["rms_lm_head"] = {"max_diff": lm_diff, "pass": lm_diff < 1e-2}
    print(f"[Op 8] RMS+LMHead: max_diff={lm_diff:.8f} {'PASS' if lm_diff < 1e-2 else 'FAIL'}")

    # ============================
    # Standalone SiLU-Multiply test (Rishu's vectorized)
    # ============================
    torch.manual_seed(42)
    gate_test = torch.randn(INTER, device="cuda", dtype=torch.float32)
    up_test = torch.randn(INTER, device="cuda", dtype=torch.float32)
    ref_silu_standalone = torch.nn.functional.silu(gate_test) * up_test
    out_silu_standalone = kernels.silu_multiply(gate_test, up_test)
    silu_standalone_diff = (out_silu_standalone - ref_silu_standalone).abs().max().item()
    results["silu_multiply"] = {"max_diff": silu_standalone_diff, "pass": silu_standalone_diff < 1e-3}
    print(f"[SiLU] Standalone SiLU*Up: max_diff={silu_standalone_diff:.8f} {'PASS' if silu_standalone_diff < 1e-3 else 'FAIL'}")

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

    print(f"\nFinal Results:")
    for name, r in result["results"].items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {name:20s}: max_diff={r['max_diff']:.8f} [{status}]")

    if result["all_pass"]:
        print("\nAll ops PASSED!")
    else:
        print("\nSome ops FAILED!")
