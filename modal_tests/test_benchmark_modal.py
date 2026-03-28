"""MFU / bandwidth / TTFT benchmark on Modal (H100 / B200).

Usage:
    modal run modal_tests/test_benchmark_modal.py --gpu b200
    modal run modal_tests/test_benchmark_modal.py --gpu b200 --mode ttft
    modal run modal_tests/test_benchmark_modal.py --gpu b200 --mode kernels
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

_target_gpu = "h100"
_target_model = "qwen3-1.7b"
_target_mode = "model"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _target_gpu = sys.argv[i + 1].lower()
    elif arg == "--model" and i + 1 < len(sys.argv):
        _target_model = sys.argv[i + 1].lower()
    elif arg == "--mode" and i + 1 < len(sys.argv):
        _target_mode = sys.argv[i + 1].lower()

if _target_gpu not in ("h100", "b200"):
    _target_gpu = "h100"

app = modal.App("cs450-benchmark")

GPU_CONFIGS = {
    "h100": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.h100",
        "modal_gpu": "H100",
        "peak_bandwidth_tb_s": 3.35,
        "peak_tflops_bf16": 989.0,
    },
    "b200": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.b200",
        "modal_gpu": "B200",
        "peak_bandwidth_tb_s": 8.0,
        "peak_tflops_bf16": 2250.0,
    },
}

_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    bench_image = (
        modal.Image.from_dockerfile(GPU_CONFIGS["h100"]["dockerfile"], force_build=force_rebuild)
        .pip_install("transformers>=4.51.0,<5.0", "accelerate", "sentencepiece")
        .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
    )
else:
    bench_image = (
        modal.Image.from_dockerfile(GPU_CONFIGS["b200"]["dockerfile"], force_build=force_rebuild)
        .pip_install("transformers>=4.51.0,<5.0", "accelerate", "sentencepiece")
        .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
    )

if _target_mode == "ttft":
    _ttft_dockerfile = GPU_CONFIGS.get(_target_gpu, GPU_CONFIGS["b200"])["dockerfile"]
    vllm_image = modal.Image.from_dockerfile(_ttft_dockerfile, force_build=force_rebuild).pip_install(
        "vllm>=0.8.0", "transformers>=4.51.0,<5.0", "sentencepiece"
    )
    sglang_image = (
        modal.Image.from_dockerfile(_ttft_dockerfile, force_build=force_rebuild)
        .apt_install("libnuma-dev")
        .pip_install(
            "sglang[srt]>=0.4.0",
            "flashinfer-python",
            "transformers>=4.51.0,<5.0",
            "sentencepiece",
        )
    )
else:
    vllm_image = _placeholder
    sglang_image = _placeholder


@dataclass(frozen=True)
class ModelConfig:
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
KERNEL_BENCH_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
KERNEL_DEFAULTS = ("fmha", "fmha_lse", "sdpa", "sdpa_math", "zigzag_local")
TTFT_PROMPT_LENS = [16, 64, 128, 256, 512, 1024]
_TTFT_BASE = "The quick brown fox jumps over the lazy dog. "


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
                    eos_token_id=[],
                )
            input_ids = prefill_output
        elif seq_len < input_ids.shape[1]:
            input_ids = input_ids[:, :seq_len]

        actual_seq_len = input_ids.shape[1]
        if actual_seq_len != seq_len:
            print(f"  Warning: requested seq_len={seq_len} but got {actual_seq_len}")
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
            "requested_seq_len": seq_len,
            "actual_seq_len": actual_seq_len,
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
        results[seq_len] = result

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


def _build_attention_kernel_modules(gpu: str, kernels: list[str]):
    import os

    import torch
    from torch.utils.cpp_extension import load

    kernels_dir = "/workspace/src/csrc/kernels"
    need_fmha = any(k in kernels for k in ("fmha", "fmha_lse", "fmha_profile", "fmha_nocausal"))
    need_zigzag = "zigzag_local" in kernels

    if not (need_fmha or need_zigzag):
        return None, None

    arch_flags = ["-arch=sm_90a"] if gpu.lower() == "h100" else ["-gencode=arch=compute_100a,code=sm_100a"]
    common_cuda_flags = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++17",
        "-lineinfo",
    ] + arch_flags
    common_cflags = ["-O3", "-std=c++20"]
    include_paths = [kernels_dir]
    if gpu.lower() == "b200":
        include_paths.extend(
            [
                "/workspace/cutlass/include",
                "/workspace/cutlass/examples/77_blackwell_fmha",
                "/workspace/cutlass/tools/util/include",
            ]
        )

    print(f"Building attention kernels with torch cpp_extension.load (GPU={gpu.upper()})")
    print(f"PyTorch version: {torch.__version__}")

    fmha_attention = None
    zigzag_attention = None

    if need_fmha:
        fmha_build_dir = "/tmp/torch_ext_fmha_attention_bench"
        os.makedirs(fmha_build_dir, exist_ok=True)
        fmha_attention = load(
            name=f"fmha_attention_bench_{gpu.lower()}",
            sources=[os.path.join(kernels_dir, "fmha_attention.cu")],
            extra_include_paths=include_paths,
            extra_cflags=common_cflags,
            extra_cuda_cflags=common_cuda_flags,
            extra_ldflags=["-lcuda"],
            verbose=True,
            with_cuda=True,
            build_directory=fmha_build_dir,
        )

    if need_zigzag:
        zigzag_build_dir = "/tmp/torch_ext_zigzag_attention_bench"
        os.makedirs(zigzag_build_dir, exist_ok=True)
        zigzag_attention = load(
            name=f"zigzag_attention_bench_{gpu.lower()}",
            sources=[os.path.join(kernels_dir, "zigzag_attention.cu")],
            extra_include_paths=include_paths,
            extra_cflags=common_cflags,
            extra_cuda_cflags=common_cuda_flags,
            extra_ldflags=["-lnccl"],
            verbose=True,
            with_cuda=True,
            build_directory=zigzag_build_dir,
        )

    return fmha_attention, zigzag_attention


def _attention_theoretical_metrics(cfg: ModelConfig, seq_len: int, causal: bool = True) -> dict:
    pairs = seq_len * (seq_len + 1) // 2 if causal else seq_len * seq_len
    flops_qk = 2 * cfg.num_q_heads * cfg.head_dim * pairs
    flops_pv = 2 * cfg.num_q_heads * cfg.head_dim * pairs
    total_flops = flops_qk + flops_pv

    bytes_per_elem = 2  # bf16
    q_bytes = cfg.num_q_heads * seq_len * cfg.head_dim * bytes_per_elem
    kv_bytes = 2 * cfg.num_kv_heads * seq_len * cfg.head_dim * bytes_per_elem
    o_bytes = cfg.num_q_heads * seq_len * cfg.head_dim * bytes_per_elem
    total_bytes = q_bytes + kv_bytes + o_bytes

    return {
        "flops": total_flops,
        "bytes": total_bytes,
    }


def _run_attention_kernels_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    kernels: list[str],
    gpu: str,
    warmup_iters: int,
    bench_iters: int,
    fmha_block_q: int = 0,
    fmha_dual_cta: bool = False,
) -> dict:
    import json
    import os

    import torch
    import torch.nn.functional as F

    cfg = ModelConfig(**model_cfg)
    print(f"CUDA available: {torch.cuda.is_available()}")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"Benchmark kernels: {kernels}")

    torch.manual_seed(0)
    scale = 1.0 / (cfg.head_dim**0.5)

    valid_kernels = set(KERNEL_DEFAULTS) | {"fmha_profile", "fmha_nocausal"}
    invalid = [k for k in kernels if k not in valid_kernels]
    if invalid:
        raise ValueError(f"Unknown kernels requested: {invalid}. Valid: {sorted(valid_kernels)}")
    if any(k in kernels for k in ("fmha", "fmha_lse", "fmha_profile", "fmha_nocausal")):
        if fmha_block_q not in (0, 64, 128):
            raise ValueError(f"fmha_block_q must be 0 (auto), 64, or 128, got {fmha_block_q}")
        os.environ["FMHA_BLOCK_Q"] = str(fmha_block_q)
        os.environ["FMHA_DUAL_CTA"] = "1" if fmha_dual_cta else "0"

    fmha_attention, zigzag_attention = _build_attention_kernel_modules(gpu, kernels)

    def _sdpa_math_context():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(backends=[SDPBackend.MATH])
        except Exception:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
                enable_cudnn=False,
            )

    results: dict[int, dict] = {}
    for seq_len in seq_lens:
        print(f"\n--- Kernel Benchmark seq_len={seq_len} ---")

        Q = torch.randn((cfg.num_q_heads, seq_len, cfg.head_dim), dtype=torch.bfloat16, device="cuda")
        K = torch.randn((cfg.num_kv_heads, seq_len, cfg.head_dim), dtype=torch.bfloat16, device="cuda")
        V = torch.randn((cfg.num_kv_heads, seq_len, cfg.head_dim), dtype=torch.bfloat16, device="cuda")
        Q_b = Q.unsqueeze(0)
        K_b = K.unsqueeze(0)
        V_b = V.unsqueeze(0)

        with torch.no_grad():
            ref_out = F.scaled_dot_product_attention(Q_b, K_b, V_b, is_causal=True, enable_gqa=True).squeeze(0)

        seq_res = {}

        for kernel_name in kernels:
            print(f"  -> {kernel_name}")

            ctx = nullcontext()
            if kernel_name == "sdpa_math":
                ctx = _sdpa_math_context()

            trace_path = None
            local_warmup_iters = warmup_iters
            local_bench_iters = bench_iters
            if kernel_name == "fmha_profile":
                trace_path = f"/tmp/fmha_profile_{gpu.lower()}_seq{seq_len}.json"
                local_warmup_iters = 0
                local_bench_iters = 1

            def _run_once():
                if kernel_name == "fmha":
                    if fmha_attention is None:
                        raise RuntimeError("fmha module not built")
                    return fmha_attention.forward(Q, K, V, scale, True, False, 0, 0, False, "")[0]
                if kernel_name == "fmha_lse":
                    if fmha_attention is None:
                        raise RuntimeError("fmha module not built")
                    return fmha_attention.forward(Q, K, V, scale, True, True, 0, 0, False, "")[0]
                if kernel_name == "fmha_nocausal":
                    if fmha_attention is None:
                        raise RuntimeError("fmha module not built")
                    return fmha_attention.forward(Q, K, V, scale, False, False, 0, 0, False, "")[0]
                if kernel_name == "fmha_profile":
                    if fmha_attention is None:
                        raise RuntimeError("fmha module not built")
                    return fmha_attention.forward(Q, K, V, scale, True, True, 0, 0, True, trace_path)[0]
                if kernel_name == "sdpa":
                    return F.scaled_dot_product_attention(Q_b, K_b, V_b, is_causal=True, enable_gqa=True).squeeze(0)
                if kernel_name == "sdpa_math":
                    return F.scaled_dot_product_attention(Q_b, K_b, V_b, is_causal=True, enable_gqa=True).squeeze(0)
                if kernel_name == "zigzag_local":
                    if zigzag_attention is None:
                        raise RuntimeError("zigzag module not built")
                    return zigzag_attention.zigzag_attention_local(Q_b, K_b, V_b, scale, True).squeeze(0)
                raise RuntimeError(f"Unhandled kernel: {kernel_name}")

            with ctx:
                for _ in range(local_warmup_iters):
                    _run_once()
                torch.cuda.synchronize()

                times_ms = []
                out = None
                for _ in range(local_bench_iters):
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    out = _run_once()
                    end_evt.record()
                    torch.cuda.synchronize()
                    times_ms.append(start_evt.elapsed_time(end_evt))

            times_ms_sorted = sorted(times_ms)
            trim = max(1, len(times_ms_sorted) // 10)
            trimmed = times_ms_sorted[trim:-trim] if len(times_ms_sorted) > 2 * trim else times_ms_sorted
            mean_ms = sum(trimmed) / len(trimmed)
            median_ms = times_ms_sorted[len(times_ms_sorted) // 2]
            min_ms = times_ms_sorted[0]

            is_causal_kernel = kernel_name not in ("fmha_nocausal",)
            theory = _attention_theoretical_metrics(cfg, seq_len, causal=is_causal_kernel)
            achieved_tflops = theory["flops"] / (mean_ms / 1000.0) / 1e12
            achieved_bw_gb_s = theory["bytes"] / (mean_ms / 1000.0) / 1e9
            qps = (cfg.num_q_heads * seq_len) / (mean_ms / 1000.0)

            max_abs = None
            mean_abs = None
            if kernel_name not in ("sdpa", "fmha_nocausal") and out is not None:
                diff = (out.float() - ref_out.float()).abs()
                max_abs = diff.max().item()
                mean_abs = diff.mean().item()

            trace_json = None
            trace_event_count = None
            if kernel_name == "fmha_profile" and trace_path and os.path.exists(trace_path):
                with open(trace_path) as f:
                    trace_json = json.load(f)
                trace_event_count = len(trace_json.get("traceEvents", []))

            seq_res[kernel_name] = {
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "q_rows_per_sec": qps,
                "achieved_tflops": achieved_tflops,
                "achieved_bw_gb_s": achieved_bw_gb_s,
                "max_abs_err_vs_sdpa": max_abs,
                "mean_abs_err_vs_sdpa": mean_abs,
                "all_times_ms": times_ms,
                "trace_event_count": trace_event_count,
                "trace_json": trace_json,
            }

            err_info = ""
            if max_abs is not None and mean_abs is not None:
                err_info = f" | max_err={max_abs:.4e}, mean_err={mean_abs:.4e}"
            trace_info = ""
            if trace_event_count is not None:
                trace_info = f" | trace_events={trace_event_count}"
            print(
                f"     mean={mean_ms:.3f} ms, median={median_ms:.3f} ms, "
                f"TFLOPS={achieved_tflops:.2f}, BW={achieved_bw_gb_s:.1f} GB/s{err_info}{trace_info}"
            )

        results[seq_len] = seq_res

    return {
        "device": device_name,
        "model": cfg.name,
        "model_id": cfg.hf_id,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "kernels": kernels,
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


def _run_megakernel_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int,
    bench_iters: int,
) -> dict:
    import os
    import sys

    import torch

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    cfg = ModelConfig(**model_cfg)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    print("Loading megakernel Decoder (JIT compile + weight load)...")
    from src.python.Qwen3.decoder import Decoder

    dec = Decoder(verbose=True)
    print("Decoder ready.")

    results = {}

    for seq_len in seq_lens:
        print(f"\n--- Megakernel benchmarking seq_len={seq_len} ---")

        dec.reset()
        dec._position = seq_len
        dummy_token = 1

        def _run_once():
            dec._position = seq_len
            return dec.step(dummy_token)

        print(f"  Warming up ({warmup_iters} iters)...")
        for _ in range(warmup_iters):
            _run_once()
        torch.cuda.synchronize()

        times_ms = []
        print(f"  Benchmarking ({bench_iters} iters)...")
        for _ in range(bench_iters):
            dec._position = seq_len
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            dec.step(dummy_token)
            end_evt.record()
            torch.cuda.synchronize()
            times_ms.append(start_evt.elapsed_time(end_evt))

        times_ms_sorted = sorted(times_ms)
        trim = max(1, len(times_ms_sorted) // 10)
        trimmed = times_ms_sorted[trim:-trim] if len(times_ms_sorted) > 2 * trim else times_ms_sorted
        mean_ms = sum(trimmed) / len(trimmed)
        median_ms = times_ms_sorted[len(times_ms_sorted) // 2]
        min_ms = times_ms_sorted[0]

        tokens_per_sec = 1000.0 / mean_ms
        theory = TheoreticalPerf(cfg, seq_len)
        total_flops = theory.total_flops()
        total_bytes = theory.total_memory_bytes()

        achieved_tflops = total_flops / (mean_ms / 1000.0) / 1e12
        achieved_bw_gb_s = total_bytes / (mean_ms / 1000.0) / 1e9

        result = {
            "requested_seq_len": seq_len,
            "actual_seq_len": seq_len,
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
        results[seq_len] = result

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


def _run_vm_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int,
    bench_iters: int,
) -> dict:
    import os
    import sys

    import torch

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    cfg = ModelConfig(**model_cfg)
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    print("Loading Decoder (JIT compile + weight load)...")
    from src.python.Qwen3.decoder import Decoder

    dec = Decoder(verbose=True)
    print("Decoder ready.")

    print("\n=== Correctness check: 20 tokens, baseline vs VM ===")
    dec.reset()
    dec._vm_enabled = False
    baseline_toks = []
    tok = dec.step(1)  # arbitrary start token
    baseline_toks.append(tok)
    for _ in range(19):
        tok = dec.step(tok)
        baseline_toks.append(tok)

    dec.reset()
    dec._vm_enabled = True
    vm_toks = []
    tok = dec.step(1)
    vm_toks.append(tok)
    for _ in range(19):
        tok = dec.step(tok)
        vm_toks.append(tok)

    match = baseline_toks == vm_toks
    print(f"  Baseline tokens: {baseline_toks[:10]}...")
    print(f"  VM tokens:       {vm_toks[:10]}...")
    print(f"  Match: {match}")
    if not match:
        mismatches = [(i, b, v) for i, (b, v) in enumerate(zip(baseline_toks, vm_toks)) if b != v]
        print(f"  First mismatches: {mismatches[:5]}")

    results: dict[int, dict] = {}
    for seq_len in seq_lens:
        print(f"\n--- Benchmarking seq_len={seq_len} ---")
        dummy_token = 1

        row: dict = {"seq_len": seq_len}
        theory = TheoreticalPerf(cfg, seq_len)
        total_flops = theory.total_flops()
        total_bytes = theory.total_memory_bytes()

        for label, use_vm in [("baseline", False), ("vm", True)]:
            dec.reset()
            dec._position = seq_len
            dec._vm_enabled = use_vm

            for _ in range(warmup_iters):
                dec._position = seq_len
                dec.step(dummy_token)
            torch.cuda.synchronize()

            times_ms = []
            for _ in range(bench_iters):
                dec._position = seq_len
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                dec.step(dummy_token)
                e.record()
                torch.cuda.synchronize()
                times_ms.append(s.elapsed_time(e))

            times_ms.sort()
            trim = max(1, len(times_ms) // 10)
            trimmed = times_ms[trim:-trim] if len(times_ms) > 2 * trim else times_ms
            mean_ms = sum(trimmed) / len(trimmed)
            median_ms = times_ms[len(times_ms) // 2]
            min_ms = times_ms[0]
            tok_s = 1000.0 / mean_ms
            tflops = total_flops / (mean_ms / 1000.0) / 1e12
            bw_gb_s = total_bytes / (mean_ms / 1000.0) / 1e9

            row[f"{label}_mean_ms"] = mean_ms
            row[f"{label}_median_ms"] = median_ms
            row[f"{label}_min_ms"] = min_ms
            row[f"{label}_tok_s"] = tok_s
            row[f"{label}_tflops"] = tflops
            row[f"{label}_bw_gb_s"] = bw_gb_s
            print(
                f"  [{label:>8}] mean={mean_ms:.3f}ms  median={median_ms:.3f}ms"
                f"  min={min_ms:.3f}ms  {tok_s:.0f} tok/s  {bw_gb_s:.0f} GB/s"
            )

        speedup = row["baseline_mean_ms"] / row["vm_mean_ms"] if row["vm_mean_ms"] > 0 else 0
        row["speedup"] = speedup
        print(f"  → VM speedup: {speedup:.2f}x")

        results[seq_len] = row

    return {
        "device": device_name,
        "model": cfg.name,
        "correctness_match": match,
        "results": results,
    }


def _prompt_ids_ttft(tokenizer, n_tokens: int) -> list[int]:
    base_ids = tokenizer.encode(_TTFT_BASE, add_special_tokens=False)
    reps = (n_tokens // len(base_ids)) + 2
    return (base_ids * reps)[:n_tokens]


def _run_ttft_benchmark(
    model_cfg: dict,
    prompt_lens: list[int],
    warmup_iters: int,
    bench_iters: int,
) -> dict:
    import time as _time

    import torch

    cfg = ModelConfig(**model_cfg)
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    results: dict[str, dict] = {}

    print("\n[megakernel] CUDA graph prefill (cuBLAS GEMM + Flash SDPA)")
    from src.python.Qwen3.decoder import Decoder

    dec = Decoder(verbose=False)
    dec._fused_prefill_max_n = 0
    tokenizer = dec.tokenizer
    mk_results: dict[int, float] = {}

    for n in prompt_lens:
        ids = _prompt_ids_ttft(tokenizer, n)
        ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")  # pre-create GPU tensor
        for _ in range(warmup_iters):
            dec.reset()
            dec.prefill(ids_t)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(bench_iters):
            dec.reset()
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            dec.prefill(ids_t)
            torch.cuda.synchronize()
            times_ms.append((_time.perf_counter() - t0) * 1000)

        mean_ms = sum(times_ms) / len(times_ms)
        mk_results[n] = mean_ms
        print(f"  n={n:5d} → {mean_ms:8.1f} ms  ({n / (mean_ms / 1000):.0f} tok/s prefill)")

    results["megakernel"] = mk_results

    print("\n[fused] persistent cooperative kernel prefill (thin-GEMM, single launch)")
    dec._fused_prefill_max_n = 32
    fused_results: dict[int, float] = {}

    for n in prompt_lens:
        if n > 32:
            continue
        ids = _prompt_ids_ttft(tokenizer, n)
        for _ in range(warmup_iters):
            dec.reset()
            dec.prefill_fused(ids)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(bench_iters):
            dec.reset()
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            dec.prefill_fused(ids)
            torch.cuda.synchronize()
            times_ms.append((_time.perf_counter() - t0) * 1000)

        mean_ms = sum(times_ms) / len(times_ms)
        fused_results[n] = mean_ms
        print(f"  n={n:5d} → {mean_ms:8.1f} ms  ({n / (mean_ms / 1000):.0f} tok/s prefill)")

    if fused_results:
        results["fused"] = fused_results

    del dec

    print("\n[HuggingFace] batched prefill")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        hf_model.eval()
        hf_results: dict[int, float] = {}

        for n in prompt_lens:
            ids = _prompt_ids_ttft(hf_tokenizer, n)
            input_ids = torch.tensor([ids], device="cuda")

            for _ in range(warmup_iters):
                with torch.no_grad():
                    hf_model.generate(
                        input_ids,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=hf_tokenizer.pad_token_id,
                    )
            torch.cuda.synchronize()

            times_ms = []
            for _ in range(bench_iters):
                torch.cuda.synchronize()
                t0 = _time.perf_counter()
                with torch.no_grad():
                    hf_model.generate(
                        input_ids,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=hf_tokenizer.pad_token_id,
                    )
                torch.cuda.synchronize()
                times_ms.append((_time.perf_counter() - t0) * 1000)

            mean_ms = sum(times_ms) / len(times_ms)
            hf_results[n] = mean_ms
            print(f"  n={n:5d} → {mean_ms:8.1f} ms")

        results["HuggingFace"] = hf_results
        del hf_model
    except Exception as e:
        print(f"  HuggingFace benchmark failed: {e}")

    print("\n[vLLM] optimized prefill")
    try:
        from vllm import LLM, SamplingParams

        vllm_tokenizer = tokenizer
        llm = LLM(model=cfg.hf_id, dtype="bfloat16", max_model_len=max(prompt_lens) + 64)
        vllm_params = SamplingParams(temperature=0.0, max_tokens=1)
        vllm_results: dict[int, float] = {}

        for n in prompt_lens:
            ids = _prompt_ids_ttft(vllm_tokenizer, n)
            prompt_text = vllm_tokenizer.decode(ids, skip_special_tokens=True)
            for _ in range(warmup_iters):
                llm.generate([prompt_text], vllm_params)
            times_ms = []
            for _ in range(bench_iters):
                t0 = _time.perf_counter()
                llm.generate([prompt_text], vllm_params)
                times_ms.append((_time.perf_counter() - t0) * 1000)
            mean_ms = sum(times_ms) / len(times_ms)
            vllm_results[n] = mean_ms
            print(f"  n={n:5d} → {mean_ms:8.1f} ms")

        results["vLLM"] = vllm_results
        del llm
    except ImportError:
        print("  vLLM not installed — skipping.")
    except Exception as e:
        print(f"  vLLM benchmark failed: {e}")

    print("\n[SGLang] optimized prefill")
    try:
        import sglang as sgl
        from sglang.srt.sampling.sampling_params import SamplingParams as SGLParams

        sgl_tokenizer = tokenizer
        engine = sgl.Engine(model_path=cfg.hf_id, dtype="bfloat16", tp_size=1)
        sgl_params = SGLParams(max_new_tokens=1, temperature=0.0)
        sgl_results: dict[int, float] = {}

        for n in prompt_lens:
            ids = _prompt_ids_ttft(sgl_tokenizer, n)
            prompt_text = sgl_tokenizer.decode(ids, skip_special_tokens=True)
            for _ in range(warmup_iters):
                engine.generate(prompts=[prompt_text], sampling_params=sgl_params)
            times_ms = []
            for _ in range(bench_iters):
                t0 = _time.perf_counter()
                engine.generate(prompts=[prompt_text], sampling_params=sgl_params)
                times_ms.append((_time.perf_counter() - t0) * 1000)
            mean_ms = sum(times_ms) / len(times_ms)
            sgl_results[n] = mean_ms
            print(f"  n={n:5d} → {mean_ms:8.1f} ms")

        results["SGLang"] = sgl_results
        engine.shutdown()
    except ImportError:
        print("  SGLang not installed — skipping.")
    except Exception as e:
        print(f"  SGLang benchmark failed: {e}")

    return {
        "device": device_name,
        "model": cfg.name,
        "prompt_lens": prompt_lens,
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
def run_megakernel_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return _run_megakernel_benchmark(model_cfg, seq_lens, warmup_iters, bench_iters)


@app.function(
    image=bench_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_ttft_benchmark(
    model_cfg: dict,
    prompt_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return _run_ttft_benchmark(model_cfg, prompt_lens, warmup_iters, bench_iters)


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_ttft_vllm(
    model_cfg: dict,
    prompt_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    import time as _time

    import torch
    from transformers import AutoTokenizer

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    print(f"[vLLM runner] Device: {device_name}")

    cfg = ModelConfig(**model_cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
    vllm_results: dict[int, float] = {}

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=cfg.hf_id, dtype="bfloat16", max_model_len=max(prompt_lens) + 64)
        vllm_params = SamplingParams(temperature=0.0, max_tokens=1)

        for n in prompt_lens:
            ids = _prompt_ids_ttft(tokenizer, n)
            prompt_text = tokenizer.decode(ids, skip_special_tokens=True)
            for _ in range(warmup_iters):
                llm.generate([prompt_text], vllm_params)
            times_ms = []
            for _ in range(bench_iters):
                t0 = _time.perf_counter()
                llm.generate([prompt_text], vllm_params)
                times_ms.append((_time.perf_counter() - t0) * 1000)
            mean_ms = sum(times_ms) / len(times_ms)
            vllm_results[n] = mean_ms
            print(f"  n={n:5d} → {mean_ms:8.1f} ms")

        del llm
    except Exception as e:
        print(f"  vLLM benchmark failed: {e}")

    return {
        "device": device_name,
        "model": cfg.name,
        "prompt_lens": prompt_lens,
        "results": {"vLLM": vllm_results},
    }


@app.function(
    image=sglang_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_ttft_sglang(
    model_cfg: dict,
    prompt_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    import time as _time

    import torch
    from transformers import AutoTokenizer

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    print(f"[SGLang runner] Device: {device_name}")

    cfg = ModelConfig(**model_cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
    sgl_results: dict[int, float] = {}

    try:
        import sglang as sgl

        engine = sgl.Engine(model_path=cfg.hf_id, dtype="bfloat16", tp_size=1)
        sgl_params = {"max_new_tokens": 1, "temperature": 0.0}

        for n in prompt_lens:
            ids = _prompt_ids_ttft(tokenizer, n)
            prompt_text = tokenizer.decode(ids, skip_special_tokens=True)
            for _ in range(warmup_iters):
                engine.generate([prompt_text], sgl_params)
            times_ms = []
            for _ in range(bench_iters):
                t0 = _time.perf_counter()
                engine.generate([prompt_text], sgl_params)
                times_ms.append((_time.perf_counter() - t0) * 1000)
            mean_ms = sum(times_ms) / len(times_ms)
            sgl_results[n] = mean_ms
            print(f"  n={n:5d} → {mean_ms:8.1f} ms")

        engine.shutdown()
    except Exception as e:
        print(f"  SGLang benchmark failed: {e}")

    return {
        "device": device_name,
        "model": cfg.name,
        "prompt_lens": prompt_lens,
        "results": {"SGLang": sgl_results},
    }


@app.function(
    image=bench_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_vm_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
) -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return _run_vm_benchmark(model_cfg, seq_lens, warmup_iters, bench_iters)


@app.function(
    image=bench_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=3600,
)
def run_attention_kernels_benchmark(
    model_cfg: dict,
    seq_lens: list[int],
    kernels: list[str],
    gpu: str,
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
    fmha_block_q: int = 0,
    fmha_dual_cta: bool = False,
) -> dict:
    return _run_attention_kernels_benchmark(
        model_cfg, seq_lens, kernels, gpu, warmup_iters, bench_iters, fmha_block_q, fmha_dual_cta
    )


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


def _print_kernel_table(
    model_cfg: ModelConfig,
    gpu_name: str,
    gpu_config: dict,
    kernel_results: dict,
    seq_lens: list[int],
):
    peak_tflops = gpu_config["peak_tflops_bf16"]
    peak_bw_gb_s = gpu_config["peak_bandwidth_tb_s"] * 1000
    data = kernel_results.get("results", {})
    kernels = kernel_results.get("kernels", list(KERNEL_DEFAULTS))

    print(f"\n{'=' * 90}")
    print(f"KERNEL MICROBENCH RESULTS ({model_cfg.name} on {gpu_name})")
    print(f"{'=' * 90}")
    print(f"Kernels: {kernels}")
    for seq_len in seq_lens:
        rows = data.get(seq_len) or data.get(str(seq_len))
        if not rows:
            continue
        print(f"\nseq_len={seq_len}")
        print("-" * 90)
        print(
            f"{'Kernel':<14} {'Mean ms':>9} {'Median ms':>10} {'Q-rows/s':>12} "
            f"{'TFLOPS':>9} {'MFU':>8} {'BW GB/s':>10} {'BW Util':>9} {'Max Abs Err':>12}"
        )
        print("-" * 90)
        for k in kernels:
            r = rows.get(k)
            if not r:
                continue
            mfu = (r["achieved_tflops"] / peak_tflops * 100) if r["achieved_tflops"] else 0.0
            bw_util = (r["achieved_bw_gb_s"] / peak_bw_gb_s * 100) if r["achieved_bw_gb_s"] else 0.0
            max_abs = r["max_abs_err_vs_sdpa"]
            max_abs_str = "N/A" if max_abs is None else f"{max_abs:.2e}"
            print(
                f"{k:<14} {r['mean_ms']:>9.3f} {r['median_ms']:>10.3f} {r['q_rows_per_sec']:>12.0f} "
                f"{r['achieved_tflops']:>9.2f} {mfu:>7.1f}% {r['achieved_bw_gb_s']:>10.1f} {bw_util:>8.1f}% \
                    {max_abs_str:>12}"
            )


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
def main(
    gpu: str = "h100",
    model: str = "qwen3-1.7b",
    mode: str = "model",
    kernels: str = ",".join(KERNEL_DEFAULTS),
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
    kernel_seq_lens: str = "",
    fmha_block_q: int = 0,
    fmha_dual_cta: bool = False,
):
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

    mode = mode.lower()
    if mode not in ("model", "kernels", "megakernel", "ttft", "vm"):
        print("Unknown mode. Use 'model', 'kernels', 'megakernel', 'ttft', or 'vm'.")
        return

    requested_kernels = [k.strip() for k in kernels.split(",") if k.strip()]
    if not requested_kernels:
        requested_kernels = list(KERNEL_DEFAULTS)

    selected_kernel_seq_lens = KERNEL_BENCH_SEQ_LENS
    if kernel_seq_lens.strip():
        selected_kernel_seq_lens = [int(x.strip()) for x in kernel_seq_lens.split(",") if x.strip()]
        if not selected_kernel_seq_lens:
            raise ValueError("kernel_seq_lens was provided but no valid lengths were parsed")
        if any(sl <= 0 for sl in selected_kernel_seq_lens):
            raise ValueError("kernel_seq_lens must contain positive integers")

    if warmup_iters < 0 or bench_iters <= 0:
        raise ValueError("warmup_iters must be >= 0 and bench_iters must be > 0")

    model_cfg = MODEL_CONFIGS[model]
    gpu_config = GPU_CONFIGS[gpu]
    gpu_name = gpu.upper()

    print(f"Running benchmark mode='{mode}': {model_cfg.name} on {gpu_name}")
    print(f"  Peak BF16 TFLOPS: {gpu_config['peak_tflops_bf16']}")
    print(f"  Peak HBM BW: {gpu_config['peak_bandwidth_tb_s']} TB/s")
    if mode == "kernels":
        print(f"  Kernels: {requested_kernels}")
        print(f"  Sequence lengths: {selected_kernel_seq_lens}")
        if any(k in requested_kernels for k in ("fmha", "fmha_lse", "fmha_profile")):
            block_q_label = "auto" if fmha_block_q == 0 else str(fmha_block_q)
            print(f"  FMHA block_q: {block_q_label}")
            print(f"  FMHA dual_cta: {fmha_dual_cta}")
    else:
        print(f"  Sequence lengths: {BENCH_SEQ_LENS}")
    print(f"  Warmup: {warmup_iters}, Bench: {bench_iters} iterations")

    ref_dir = PROJECT_ROOT / "modal_tests" / "reference_data"
    ref_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("-", "_").replace(".", "_")

    if mode == "kernels":
        print(f"\n{'=' * 70}")
        print("Running Attention Kernel Microbenchmarks...")
        print(f"{'=' * 70}")
        kernel_results = run_attention_kernels_benchmark.remote(
            asdict(model_cfg),
            selected_kernel_seq_lens,
            requested_kernels,
            gpu,
            warmup_iters,
            bench_iters,
            fmha_block_q,
            fmha_dual_cta,
        )
        _print_kernel_table(model_cfg, gpu_name, gpu_config, kernel_results, selected_kernel_seq_lens)

        kernel_results_for_save = dict(kernel_results)
        kernel_results_for_save["results"] = {}
        for seq_len, seq_rows in kernel_results.get("results", {}).items():
            cleaned_rows = {}
            for kernel_name, row in seq_rows.items():
                cleaned = dict(row)
                if "trace_json" in cleaned:
                    cleaned["trace_json"] = None
                cleaned_rows[kernel_name] = cleaned
            kernel_results_for_save["results"][seq_len] = cleaned_rows

        results_path = ref_dir / f"kernel_benchmark_{model_slug}_{gpu}_{timestamp}.json"
        save_data = {
            "gpu": gpu_name,
            "mode": mode,
            "model": model_cfg.name,
            "model_config": asdict(model_cfg),
            "gpu_config": {
                "peak_bandwidth_tb_s": gpu_config["peak_bandwidth_tb_s"],
                "peak_tflops_bf16": gpu_config["peak_tflops_bf16"],
            },
            "kernel_results": kernel_results_for_save,
        }
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nKernel benchmark results saved to {results_path}")

        trace_dir = PROJECT_ROOT / "src" / "csrc" / "profiler" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        for seq_len, seq_rows in kernel_results.get("results", {}).items():
            row = seq_rows.get("fmha_profile")
            if not row:
                continue
            trace_json = row.get("trace_json")
            if not trace_json:
                continue
            trace_path = trace_dir / f"trace_fmha_{gpu}_seq{seq_len}.json"
            with open(trace_path, "w") as f:
                json.dump(trace_json, f, indent=2)
            n_events = len(trace_json.get("traceEvents", []))
            print(f"Saved FMHA pipeline trace ({n_events} events) to {trace_path}")

        return

    model_cfg_dict = asdict(model_cfg)

    if mode == "megakernel":
        print(f"\n{'=' * 70}")
        print("Running Megakernel Benchmark (compile + decode only)...")
        print(f"{'=' * 70}")
        mega_results = run_megakernel_benchmark.remote(model_cfg_dict, BENCH_SEQ_LENS, warmup_iters, bench_iters)
        print(f"\n{'=' * 70}")
        print("MEGAKERNEL RESULTS")
        print(f"{'=' * 70}")
        mega_data = mega_results.get("results", {})
        print(f"{'seq_len':>8} {'mean_ms':>10} {'tok/s':>8} {'TFLOPS':>9} {'BW GB/s':>10}")
        print("-" * 55)
        for sl in BENCH_SEQ_LENS:
            r = mega_data.get(sl) or mega_data.get(str(sl))
            if r:
                print(
                    f"{sl:>8} {r['mean_ms']:>10.3f} {r['tokens_per_sec']:>8.1f} "
                    f"{r['achieved_tflops']:>9.3f} {r['achieved_bw_gb_s']:>10.1f}"
                )
        return

    if mode == "vm":
        vm_seq_lens = [1, 64, 256, 1024, 2048]
        print(f"\n{'=' * 70}")
        print(f"VM Kernel A/B Benchmark: {model_cfg.name} on {gpu_name}")
        print(f"Seq lengths: {vm_seq_lens}")
        print(f"{'=' * 70}")
        vm_results = run_vm_benchmark.remote(model_cfg_dict, vm_seq_lens, warmup_iters, bench_iters)

        correct = vm_results.get("correctness_match", "N/A")
        print(f"\nCorrectness match: {correct}")

        vm_data = vm_results.get("results", {})
        peak_bw_gb_s = gpu_config["peak_bandwidth_tb_s"] * 1000

        print(f"\n{'=' * 90}")
        print(
            f"{'seq_len':>8} {'baseline':>10} {'VM':>10} {'speedup':>8}"
            f" {'BL BW':>10} {'VM BW':>10} {'BL util':>8} {'VM util':>8}"
        )
        print("-" * 90)
        for sl in vm_seq_lens:
            r = vm_data.get(sl) or vm_data.get(str(sl))
            if not r:
                continue
            bl_ms = r["baseline_mean_ms"]
            vm_ms = r["vm_mean_ms"]
            sp = r["speedup"]
            bl_bw = r["baseline_bw_gb_s"]
            vm_bw = r["vm_bw_gb_s"]
            bl_util = bl_bw / peak_bw_gb_s * 100
            vm_util = vm_bw / peak_bw_gb_s * 100
            print(
                f"{sl:>8} {bl_ms:>9.3f}ms {vm_ms:>9.3f}ms {sp:>7.2f}x"
                f" {bl_bw:>9.0f} {vm_bw:>9.0f} {bl_util:>7.1f}% {vm_util:>7.1f}%"
            )
        print(f"{'=' * 90}")

        results_path = ref_dir / f"vm_benchmark_{model_slug}_{gpu}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump({"gpu": gpu_name, "model": model_cfg.name, **vm_results}, f, indent=2, default=str)
        print(f"\nVM benchmark results saved to {results_path}")
        return

    if mode == "ttft":
        print(f"\n{'=' * 70}")
        print(f"TTFT Benchmark: {model_cfg.name} on {gpu_name}")
        print(f"Prompt lengths: {TTFT_PROMPT_LENS}")
        print(f"{'=' * 70}")

        ttft_handle = run_ttft_benchmark.spawn(model_cfg_dict, TTFT_PROMPT_LENS, warmup_iters, bench_iters)
        vllm_handle = None
        sglang_handle = None
        try:
            vllm_handle = run_ttft_vllm.spawn(model_cfg_dict, TTFT_PROMPT_LENS, warmup_iters, bench_iters)
        except Exception as e:
            print(f"[WARN] Failed to spawn vLLM benchmark: {e}")
        try:
            sglang_handle = run_ttft_sglang.spawn(model_cfg_dict, TTFT_PROMPT_LENS, warmup_iters, bench_iters)
        except Exception as e:
            print(f"[WARN] Failed to spawn SGLang benchmark: {e}")

        ttft_results = ttft_handle.get()
        vllm_results: dict = {}
        sglang_results: dict = {}
        if vllm_handle is not None:
            try:
                vllm_results = vllm_handle.get()
            except Exception as e:
                print(f"[WARN] vLLM benchmark failed: {e}")
        if sglang_handle is not None:
            try:
                sglang_results = sglang_handle.get()
            except Exception as e:
                print(f"[WARN] SGLang benchmark failed: {e}")

        backends_data: dict = ttft_results.get("results", {})
        for ext_results in (vllm_results, sglang_results):
            for backend, timings in ext_results.get("results", {}).items():
                backends_data[backend] = timings
        active_backends = [b for b in backends_data if backends_data[b]]

        print(f"\n{'=' * 70}")
        print("TTFT RESULTS (mean ms, lower is better)")
        print(f"{'=' * 70}")

        print(f"{'prompt_len':>10}", end="")
        for b in active_backends:
            print(f"  {b:>14}", end="")
        mk_data = backends_data.get("megakernel")
        others = [b for b in active_backends if b != "megakernel"]
        if mk_data and others:
            for b in others:
                print(f"  {'MK/' + b:>12}", end="")
        print()
        print("-" * (10 + 16 * len(active_backends) + (13 * len(others) if mk_data else 0)))

        for n in TTFT_PROMPT_LENS:
            print(f"{n:>10}", end="")
            for b in active_backends:
                val = backends_data[b].get(n) if backends_data[b] else None
                print(f"  {(f'{val:.1f}' if val is not None else 'N/A'):>14}", end="")
            if mk_data and others:
                for b in others:
                    val = backends_data[b].get(n) if backends_data[b] else None
                    mk_val = mk_data.get(n) if mk_data else None
                    if val and mk_val:
                        print(f"  {mk_val / val:>11.2f}x", end="")
                    else:
                        print(f"  {'N/A':>12}", end="")
            print()

        print(f"{'=' * 70}")

        results_path = ref_dir / f"ttft_{model_slug}_{gpu}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump({"gpu": gpu_name, "model": model_cfg.name, **ttft_results}, f, indent=2, default=str)
        print(f"\nTTFT results saved to {results_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"Theoretical Analysis: {model_cfg.name} (per decode step, batch=1, bf16)")
    print(f"{'=' * 70}")
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

    print(f"\n{'=' * 70}")
    print("Running HF Baseline Benchmark...")
    print(f"{'=' * 70}")
    hf_results = run_hf_benchmark.remote(model_cfg_dict, BENCH_SEQ_LENS, WARMUP_ITERS, BENCH_ITERS)

    print(f"\n{'=' * 70}")
    print("Running Megakernel Benchmark...")
    print(f"{'=' * 70}")
    mega_results = run_megakernel_benchmark.remote(model_cfg_dict, BENCH_SEQ_LENS, WARMUP_ITERS, BENCH_ITERS)

    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 70}")

    actual_seq_lens = []
    hf_data = hf_results.get("results", {})
    for sl in BENCH_SEQ_LENS:
        if sl in hf_data or str(sl) in hf_data:
            actual_seq_lens.append(sl)

    if actual_seq_lens:
        _print_comparison_table(model_cfg, gpu_name, gpu_config, hf_results, mega_results, actual_seq_lens)
    else:
        print("No benchmark results to display.")

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
