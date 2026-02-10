"""
Model FLOPS Utilization (MFU) and memory bandwidth benchmark comparing HuggingFace (HF) baseline vs Megakernel.

Measures MFU and HBM bandwidth utilization using
theoretical FLOP/byte counts and precise GPU timing via capturing CUDA events.

Supports Qwen3-1.7B and Qwen3-8B model configs.

Usage:
    # Run benchmark for Qwen3-1.7B
    modal run modal_tests/test_benchmark_modal.py --gpu h100
    modal run modal_tests/test_benchmark_modal.py --gpu b200

    # Run benchmark for Qwen3-8B
    modal run modal_tests/test_benchmark_modal.py --gpu h100 --model qwen3-8b
    modal run modal_tests/test_benchmark_modal.py --gpu b200 --model qwen3-8b

    # Force rebuild the image
    FORCE_REBUILD=1 modal run modal_tests/test_benchmark_modal.py --gpu h100
    FORCE_REBUILD=1 modal run modal_tests/test_benchmark_modal.py --gpu b200
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

_target_gpu = "h100"
_target_model = "qwen3-1.7b"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _target_gpu = sys.argv[i + 1].lower()
    elif arg == "--model" and i + 1 < len(sys.argv):
        _target_model = sys.argv[i + 1].lower()

app = modal.App("cs450-benchmark")

GPU_CONFIGS = {
    "h100": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.h100",
        "modal_gpu": "H100",
        # H100 SXM peak specs
        "peak_bandwidth_tb_s": 3.35,
        "peak_tflops_bf16": 989.0,
    },
    "b200": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.b200",
        "modal_gpu": "B200",
        # B200 peak specs
        "peak_bandwidth_tb_s": 8.0,
        "peak_tflops_bf16": 2250.0,
    },
}

_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    bench_image = (
        modal.Image.from_dockerfile(GPU_CONFIGS["h100"]["dockerfile"], force_build=force_rebuild)
        .pip_install("transformers>=4.51.0", "accelerate", "sentencepiece")
        .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
    )
else:
    bench_image = (
        modal.Image.from_dockerfile(GPU_CONFIGS["b200"]["dockerfile"], force_build=force_rebuild)
        .pip_install("transformers>=4.51.0", "accelerate", "sentencepiece")
        .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
    )


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Architecture constants for a transformer model."""

    name: str
    hf_id: str
    hidden_size: int
    intermediate_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int
    bytes_per_param: int = 2  # bf16


QWEN3_1_7B = ModelConfig(
    name="Qwen3-1.7B",
    hf_id="Qwen/Qwen3-1.7B",
    hidden_size=2048,
    intermediate_size=6144,
    num_q_heads=16,
    num_kv_heads=8,
    head_dim=128,
    num_layers=28,
    vocab_size=151936,
)

QWEN3_8B = ModelConfig(
    name="Qwen3-8B",
    hf_id="Qwen/Qwen3-8B",
    hidden_size=4096,
    intermediate_size=12288,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    num_layers=36,
    vocab_size=151936,
)

MODEL_CONFIGS = {
    "qwen3-1.7b": QWEN3_1_7B,
    "qwen3-8b": QWEN3_8B,
}

WARMUP_ITERS = 5
BENCH_ITERS = 20
BENCH_SEQ_LENS = [1, 32, 128, 512, 1024]
BENCH_PROMPT = "The quick brown fox jumps over the lazy dog"


class TheoreticalPerf:
    def __init__(self, cfg: ModelConfig, seq_len: int):
        self.cfg = cfg
        self.seq_len = seq_len

    def flops_per_layer(self) -> int:
        c = self.cfg
        s = self.seq_len

        q_size = c.num_q_heads * c.head_dim
        kv_size = c.num_kv_heads * c.head_dim
        qkv_flops = 2 * c.hidden_size * (q_size + 2 * kv_size)
        attn_score_flops = 2 * c.num_q_heads * c.head_dim * s
        attn_value_flops = 2 * c.num_q_heads * s * c.head_dim
        o_proj_flops = 2 * (c.num_q_heads * c.head_dim) * c.hidden_size
        gate_flops = 2 * c.hidden_size * c.intermediate_size
        up_flops = 2 * c.hidden_size * c.intermediate_size
        down_flops = 2 * c.intermediate_size * c.hidden_size
        rmsnorm_flops = 2 * 3 * c.hidden_size
        silu_flops = 4 * c.intermediate_size
        rope_flops = 6 * (q_size + kv_size)

        return (
            qkv_flops
            + attn_score_flops
            + attn_value_flops
            + o_proj_flops
            + gate_flops
            + up_flops
            + down_flops
            + rmsnorm_flops
            + silu_flops
            + rope_flops
        )

    def total_flops(self) -> int:
        c = self.cfg
        per_layer = self.flops_per_layer()
        all_layers = per_layer * c.num_layers
        final_norm = 3 * c.hidden_size
        lm_head = 2 * c.hidden_size * c.vocab_size

        return all_layers + final_norm + lm_head

    def weight_bytes_per_layer(self) -> int:
        c = self.cfg
        q_size = c.num_q_heads * c.head_dim
        kv_size = c.num_kv_heads * c.head_dim

        qkv_weights = c.hidden_size * (q_size + 2 * kv_size)
        o_proj_weights = (c.num_q_heads * c.head_dim) * c.hidden_size
        gate_weights = c.hidden_size * c.intermediate_size
        up_weights = c.hidden_size * c.intermediate_size
        down_weights = c.intermediate_size * c.hidden_size
        # 2 RMSNorm weights per layer
        norm_weights = 2 * c.hidden_size

        total_params = qkv_weights + o_proj_weights + gate_weights + up_weights + down_weights + norm_weights
        return total_params * c.bytes_per_param

    def total_memory_bytes(self) -> int:
        c = self.cfg
        weight_bytes = self.weight_bytes_per_layer() * c.num_layers
        final_norm_bytes = c.hidden_size * c.bytes_per_param
        lm_head_bytes = c.hidden_size * c.vocab_size * c.bytes_per_param
        weight_bytes += final_norm_bytes + lm_head_bytes
        kv_bytes_per_layer = 2 * c.num_kv_heads * self.seq_len * c.head_dim * c.bytes_per_param
        kv_bytes = kv_bytes_per_layer * c.num_layers
        activation_bytes = c.hidden_size * c.bytes_per_param + c.num_layers * c.hidden_size * 4 * c.bytes_per_param

        return weight_bytes + kv_bytes + activation_bytes

    def summary(self) -> dict:
        total_flops = self.total_flops()
        total_bytes = self.total_memory_bytes()
        return {
            "model": self.cfg.name,
            "seq_len": self.seq_len,
            "total_flops": total_flops,
            "total_flops_gflop": total_flops / 1e9,
            "total_memory_bytes": total_bytes,
            "total_memory_gb": total_bytes / 1e9,
            "arithmetic_intensity": total_flops / total_bytes if total_bytes > 0 else 0,
            "flops_per_layer": self.flops_per_layer(),
            "weight_bytes_per_layer": self.weight_bytes_per_layer(),
        }


def _deepcopy_cache(past_kv):
    import copy

    return copy.deepcopy(past_kv)


def _run_hf_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int,
    bench_iters: int,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = ModelConfig(**model_cfg)

    print(f"Loading {cfg.hf_id} for benchmarking...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"CUDA version: {torch.version.cuda}")

    results = {}

    for seq_len in seq_lens:
        print(f"\n--- Benchmarking seq_len={seq_len} ---")

        inputs = tokenizer(BENCH_PROMPT, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]

        if seq_len > input_ids.shape[1]:
            with torch.no_grad():
                prefill_output = model.generate(
                    input_ids,
                    max_new_tokens=seq_len - input_ids.shape[1],
                    do_sample=False,
                    use_cache=True,
                )
            input_ids = prefill_output
        elif seq_len < input_ids.shape[1]:
            input_ids = input_ids[:, :seq_len]

        actual_seq_len = input_ids.shape[1]
        print(f"  Context length: {actual_seq_len}")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            cur_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        base_cache = _deepcopy_cache(past_kv)

        print(f"  Warming up ({warmup_iters} iters)...")
        for _ in range(warmup_iters):
            past_kv = _deepcopy_cache(base_cache)
            with torch.no_grad():
                out = model(input_ids=cur_token, past_key_values=past_kv, use_cache=True)
                cur_token = out.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()

        times_ms = []
        print(f"  Benchmarking ({bench_iters} iters)...")
        for _ in range(bench_iters):
            past_kv = _deepcopy_cache(base_cache)
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            with torch.no_grad():
                out = model(input_ids=cur_token, past_key_values=past_kv, use_cache=True)
            end_evt.record()
            torch.cuda.synchronize()

            elapsed_ms = start_evt.elapsed_time(end_evt)
            times_ms.append(elapsed_ms)

            cur_token = out.logits[:, -1:, :].argmax(dim=-1)

        times_ms_sorted = sorted(times_ms)
        # Trim top/bottom 10% for robust mean
        trim = max(1, len(times_ms_sorted) // 10)
        trimmed = times_ms_sorted[trim:-trim] if len(times_ms_sorted) > 2 * trim else times_ms_sorted
        mean_ms = sum(trimmed) / len(trimmed)
        median_ms = times_ms_sorted[len(times_ms_sorted) // 2]
        min_ms = times_ms_sorted[0]

        tokens_per_sec = 1000.0 / mean_ms
        theory = TheoreticalPerf(cfg, actual_seq_len)
        total_flops = theory.total_flops()
        total_bytes = theory.total_memory_bytes()

        achieved_tflops = total_flops / (mean_ms / 1000.0) / 1e12
        achieved_bw_gb_s = total_bytes / (mean_ms / 1000.0) / 1e9

        result = {
            "seq_len": actual_seq_len,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min_ms,
            "tokens_per_sec": tokens_per_sec,
            "theoretical_flops": total_flops,
            "theoretical_bytes": total_bytes,
            "achieved_tflops": achieved_tflops,
            "achieved_bw_gb_s": achieved_bw_gb_s,
            "all_times_ms": times_ms,
        }
        results[actual_seq_len] = result

        print(f"  Mean time: {mean_ms:.3f} ms | Median: {median_ms:.3f} ms | Min: {min_ms:.3f} ms")
        print(f"  Tokens/sec: {tokens_per_sec:.1f}")
        print(f"  Achieved: {achieved_tflops:.2f} TFLOPS, {achieved_bw_gb_s:.1f} GB/s")

    return {
        "device": device_name,
        "model": cfg.name,
        "model_id": cfg.hf_id,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "results": results,
    }


@app.function(
    image=bench_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_hf_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    return _run_hf_benchmark(model_cfg, seq_lens, warmup_iters, bench_iters)


@app.function(
    image=bench_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_megakernel_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    """
    Runs the megakernel benchmark

    NOTE:

    Currently a placeholder and returns empty results, because the megakernel
    is not yet implemented.
    """
    import torch

    cfg = ModelConfig(**model_cfg)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # TODO: Replace with actual megakernel benchmark
    # ==========================================================================
    # When the megakernel is ready:
    # 1. Load weights into megakernel format
    # 2. For each seq_len, run warmup + timed decode steps
    # 3. Measure with torch.cuda.Event or the GPU profiler (gpu_profiler.cuh)
    # 4. Return results in the same format as _run_hf_benchmark
    # ==========================================================================
    print("Megakernel benchmark not yet implemented. Returning empty results.")
    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model": cfg.name,
        "model_id": cfg.hf_id,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "results": {},
    }


def _print_comparison_table(
    model_cfg: ModelConfig,
    gpu_name: str,
    gpu_config: dict,
    hf_results: dict,
    mega_results: dict,
    seq_lens: list[int],
):
    peak_tflops = gpu_config["peak_tflops_bf16"]
    peak_bw_gb_s = gpu_config["peak_bandwidth_tb_s"] * 1000  # TB/s -> GB/s
    ridge_point = peak_tflops / gpu_config["peak_bandwidth_tb_s"]  # FLOP/byte
    hf_data = hf_results.get("results", {})
    mega_data = mega_results.get("results", {})

    for seq_len in seq_lens:
        hf = hf_data.get(seq_len) or hf_data.get(str(seq_len))
        mega = mega_data.get(seq_len) or mega_data.get(str(seq_len))

        theory = TheoreticalPerf(model_cfg, seq_len)
        theory_summary = theory.summary()

        print(f"\n{model_cfg.name} Benchmark ({gpu_name}, seq_len={seq_len})")
        print(
            f"  Theoretical: {theory_summary['total_flops_gflop']:.2f} GFLOP, "
            f"{theory_summary['total_memory_gb']:.3f} GB, "
            f"AI={theory_summary['arithmetic_intensity']:.1f} FLOP/byte"
        )
        print("=" * 70)
        print(f"{'Metric':<28} {'HF Baseline':>18} {'Megakernel':>18}")
        print("-" * 70)

        def _fmt(val, fmt_str):
            if val is None:
                return "N/A"
            return f"{val:{fmt_str}}"

        hf_tps = hf["tokens_per_sec"] if hf else None
        mega_tps = mega["tokens_per_sec"] if mega else None
        print(f"{'Tokens/sec':<28} {_fmt(hf_tps, '.1f'):>18} {_fmt(mega_tps, '.1f'):>18}")

        hf_ms = hf["mean_ms"] if hf else None
        mega_ms = mega["mean_ms"] if mega else None
        hf_us = hf_ms * 1000 if hf_ms else None
        mega_us = mega_ms * 1000 if mega_ms else None
        print(f"{'Time/token (us)':<28} {_fmt(hf_us, '.0f'):>18} {_fmt(mega_us, '.0f'):>18}")

        hf_tf = hf["achieved_tflops"] if hf else None
        mega_tf = mega["achieved_tflops"] if mega else None
        print(f"{'Achieved TFLOPS':<28} {_fmt(hf_tf, '.2f'):>18} {_fmt(mega_tf, '.2f'):>18}")

        hf_mfu = (hf_tf / peak_tflops * 100) if hf_tf else None
        mega_mfu = (mega_tf / peak_tflops * 100) if mega_tf else None
        hf_mfu_s = f"{hf_mfu:.1f}%" if hf_mfu else "N/A"
        mega_mfu_s = f"{mega_mfu:.1f}%" if mega_mfu else "N/A"
        print(f"{'MFU':<28} {hf_mfu_s:>18} {mega_mfu_s:>18}")

        hf_bw = hf["achieved_bw_gb_s"] if hf else None
        mega_bw = mega["achieved_bw_gb_s"] if mega else None
        print(f"{'Achieved BW (GB/s)':<28} {_fmt(hf_bw, '.1f'):>18} {_fmt(mega_bw, '.1f'):>18}")

        hf_bw_util = (hf_bw / peak_bw_gb_s * 100) if hf_bw else None
        mega_bw_util = (mega_bw / peak_bw_gb_s * 100) if mega_bw else None
        hf_bw_util_s = f"{hf_bw_util:.1f}%" if hf_bw_util else "N/A"
        mega_bw_util_s = f"{mega_bw_util:.1f}%" if mega_bw_util else "N/A"
        print(f"{'BW Utilization':<28} {hf_bw_util_s:>18} {mega_bw_util_s:>18}")
        print("=" * 70)

        ai = theory_summary["arithmetic_intensity"]
        if hf_mfu is not None and hf_bw_util is not None:
            if ai < ridge_point and hf_bw_util > 30:
                print("  -> Memory-bound")
            elif ai < ridge_point and hf_bw_util <= 30:
                print(f"  -> Launch-bound {hf_bw_util:.1f}% BW util.\n Kernel dispatch overhead dominates")
            elif ai >= ridge_point and hf_mfu > 20:
                print("  -> Compute-bound")
            elif ai >= ridge_point:
                print(f"  -> Latency-bound {hf_mfu:.1f}% MFU.\n Fixed overhead dominates")
            else:
                print("  -> Underutilized")


def _generate_roofline_plot(
    model_cfg: ModelConfig,
    gpu_configs: dict,
    seq_lens: list[int],
    hf_results: dict | None = None,
    mega_results: dict | None = None,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 8))

    ai_range = np.logspace(-1, 4, 500)

    gpu_colors = {"H100": "#2196F3", "B200": "#4CAF50"}
    gpu_styles = {"H100": "-", "B200": "--"}

    for gpu_key, gcfg in gpu_configs.items():
        gpu_label = gpu_key.upper()
        peak_tflops = gcfg["peak_tflops_bf16"]
        peak_bw_tb_s = gcfg["peak_bandwidth_tb_s"]
        peak_bw_tflop_per_byte = peak_bw_tb_s
        ridge = peak_tflops / peak_bw_tflop_per_byte
        color = gpu_colors.get(gpu_label, "#FF9800")
        style = gpu_styles.get(gpu_label, "-.")

        roofline = np.minimum(peak_tflops, peak_bw_tflop_per_byte * ai_range)
        ax.plot(
            ai_range,
            roofline,
            style,
            color=color,
            linewidth=2.5,
            label=f"{gpu_label} roofline ({peak_tflops:.0f} TFLOPS, {peak_bw_tb_s:.1f} TB/s)",
        )

        ax.axvline(x=ridge, color=color, linestyle=":", alpha=0.4, linewidth=1)
        ax.annotate(
            f"{gpu_label} ridge\n({ridge:.0f} F/B)",
            xy=(ridge, peak_tflops * 0.7),
            fontsize=8,
            color=color,
            ha="center",
            alpha=0.7,
        )

    theory_ais = []
    theory_labels = []
    for sl in seq_lens:
        t = TheoreticalPerf(model_cfg, sl)
        s = t.summary()
        theory_ais.append(s["arithmetic_intensity"])
        theory_labels.append(f"s={sl}")

    for gpu_key, gcfg in gpu_configs.items():
        gpu_label = gpu_key.upper()
        peak_tflops = gcfg["peak_tflops_bf16"]
        peak_bw_tb_s = gcfg["peak_bandwidth_tb_s"]
        color = gpu_colors.get(gpu_label, "#FF9800")

        theory_perfs = [min(peak_tflops, peak_bw_tb_s * ai) for ai in theory_ais]
        ax.scatter(
            theory_ais,
            theory_perfs,
            marker="o",
            s=60,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            alpha=0.6,
        )

    for idx, (ai, label) in enumerate(zip(theory_ais, theory_labels)):
        min_bw = min(g["peak_bandwidth_tb_s"] for g in gpu_configs.values())
        y_pos = min_bw * ai * 0.55
        x_offset = 15 + idx * 25
        ax.annotate(
            label,
            xy=(ai, y_pos),
            xytext=(x_offset, -10),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            color="gray",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
        )

    if hf_results:
        hf_data = hf_results.get("results", {})
        measured_ais = []
        measured_tflops = []
        for sl in seq_lens:
            hf = hf_data.get(sl) or hf_data.get(str(sl))
            if hf:
                t = TheoreticalPerf(model_cfg, sl)
                s = t.summary()
                measured_ais.append(s["arithmetic_intensity"])
                measured_tflops.append(hf["achieved_tflops"])
        if measured_ais:
            ax.scatter(
                measured_ais,
                measured_tflops,
                marker="^",
                s=100,
                color="#F44336",
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
                label=f"HF baseline measured ({model_cfg.name})",
            )

    if mega_results:
        mega_data = mega_results.get("results", {})
        measured_ais = []
        measured_tflops = []
        for sl in seq_lens:
            mega = mega_data.get(sl) or mega_data.get(str(sl))
            if mega and mega.get("achieved_tflops"):
                t = TheoreticalPerf(model_cfg, sl)
                s = t.summary()
                measured_ais.append(s["arithmetic_intensity"])
                measured_tflops.append(mega["achieved_tflops"])
        if measured_ais:
            ax.scatter(
                measured_ais,
                measured_tflops,
                marker="s",
                s=100,
                color="#9C27B0",
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
                label=f"Megakernel measured ({model_cfg.name})",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (TFLOPS)", fontsize=12)
    ax.set_title(
        f"Roofline Model: {model_cfg.name} Decode (batch=1, bf16)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="-")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.1, 1e4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Roofline plot saved to {save_path}")
    plt.close(fig)


@app.local_entrypoint()
def main(gpu: str = "h100", model: str = "qwen3-1.7b"):
    import json
    import time
    from dataclasses import asdict

    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    model = model.lower()
    if model not in MODEL_CONFIGS:
        print(f"Unknown model '{model}'. Choose from: {', '.join(MODEL_CONFIGS)}")
        return

    model_cfg = MODEL_CONFIGS[model]
    gpu_config = GPU_CONFIGS[gpu]
    gpu_name = gpu.upper()

    print(f"Running MFU & bandwidth benchmark: {model_cfg.name} on {gpu_name}")
    print(f"  Peak BF16 TFLOPS: {gpu_config['peak_tflops_bf16']}")
    print(f"  Peak HBM BW: {gpu_config['peak_bandwidth_tb_s']} TB/s")
    print(f"  Sequence lengths: {BENCH_SEQ_LENS}")
    print(f"  Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS} iterations")

    print(f"\n{'='*70}")
    print(f"Theoretical Analysis: {model_cfg.name} (per decode step, batch=1, bf16)")
    print(f"{'='*70}")
    print(f"{'seq_len':>8} {'GFLOP':>10} {'GB traffic':>12} {'AI (F/B)':>10} {'Regime':>15}")
    print("-" * 70)
    ridge_point = gpu_config["peak_tflops_bf16"] * 1e12 / (gpu_config["peak_bandwidth_tb_s"] * 1e12)
    for sl in BENCH_SEQ_LENS:
        t = TheoreticalPerf(model_cfg, sl)
        s = t.summary()
        regime = "memory-bound" if s["arithmetic_intensity"] < ridge_point else "compute-bound"
        print(
            f"{sl:>8} {s['total_flops_gflop']:>10.2f} {s['total_memory_gb']:>12.4f} "
            f"{s['arithmetic_intensity']:>10.1f} {regime:>15}"
        )
    print(f"  Ridge point (FLOP/byte): {ridge_point:.1f}")

    model_cfg_dict = asdict(model_cfg)
    print(f"\n{'='*70}")
    print("Running HF Baseline Benchmark...")
    print(f"{'='*70}")
    hf_results = run_hf_benchmark.remote(model_cfg_dict, BENCH_SEQ_LENS, WARMUP_ITERS, BENCH_ITERS)

    print(f"\n{'='*70}")
    print("Running Megakernel Benchmark...")
    print(f"{'='*70}")
    mega_results = run_megakernel_benchmark.remote(model_cfg_dict, BENCH_SEQ_LENS, WARMUP_ITERS, BENCH_ITERS)

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")

    actual_seq_lens = []
    hf_data = hf_results.get("results", {})
    for sl in BENCH_SEQ_LENS:
        if sl in hf_data or str(sl) in hf_data:
            actual_seq_lens.append(sl)

    if actual_seq_lens:
        _print_comparison_table(model_cfg, gpu_name, gpu_config, hf_results, mega_results, actual_seq_lens)
    else:
        print("No benchmark results to display.")

    ref_dir = PROJECT_ROOT / "modal_tests" / "reference_data"
    ref_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("-", "_").replace(".", "_")
    results_path = ref_dir / f"benchmark_{model_slug}_{gpu}_{timestamp}.json"

    save_data = {
        "gpu": gpu_name,
        "model": model_cfg.name,
        "model_config": model_cfg_dict,
        "gpu_config": {
            "peak_bandwidth_tb_s": gpu_config["peak_bandwidth_tb_s"],
            "peak_tflops_bf16": gpu_config["peak_tflops_bf16"],
        },
        "hf_baseline": hf_results,
        "megakernel": mega_results,
        "theoretical": {str(sl): TheoreticalPerf(model_cfg, sl).summary() for sl in BENCH_SEQ_LENS},
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    plot_path = ref_dir / f"roofline_{model_slug}_{gpu}_{timestamp}.png"
    _generate_roofline_plot(
        model_cfg=model_cfg,
        gpu_configs=GPU_CONFIGS,
        seq_lens=actual_seq_lens if actual_seq_lens else BENCH_SEQ_LENS,
        hf_results=hf_results if actual_seq_lens else None,
        mega_results=mega_results if actual_seq_lens else None,
        save_path=str(plot_path),
    )
