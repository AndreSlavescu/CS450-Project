"""Full-model TTFT benchmark: Ours vs vLLM vs SGLang (Qwen3-Coder-480B on 8xB200).

Compares time-to-first-token across three inference backends for the
Qwen3-Coder-480B-A35B-Instruct MoE model on 8 NVIDIA B200 GPUs.

Each backend loads the model, then measures TTFT across multiple prompt
lengths. Model loading time is excluded from TTFT measurements.

The backends run sequentially so results are directly comparable on
the same hardware allocation.

Usage:
    # Step 1: Pre-download weights to the volume (cheap CPU, one-time)
    modal run modal_tests/test_moe_ttft_benchmark.py --download-only

    # Step 2: Run the benchmark (uses cached weights from volume)
    modal run modal_tests/test_moe_ttft_benchmark.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent
force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

app = modal.App("cs450-moe-ttft-benchmark")

# Persistent volume so ~960GB of weights are downloaded once and reused
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
WARMUP = 2
RUNS = 5
GEN_TOKENS = 16
MAX_SEQ_LEN = 4096
PROMPT_LENS = [128, 256, 512, 1024, 2048, 4096]
_PROMPT_BASE = "The quick brown fox jumps over the lazy dog. "

# ── Docker images ──
# Our engine uses Dockerfile.b200 (torch nightly cu130 + CUTLASS/ThunderKittens).
# vLLM and SGLang use their official Docker images which include CUDA toolkit
# (nvcc required for torch.compile/CUDAGraph), correct torch, and B200 support.
# IMPORTANT: We must clear the ENTRYPOINT from these images because their default
# entrypoints (e.g. "vllm serve") conflict with Modal's container entrypoint.

ours_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.b200", force_build=force_rebuild)
    .pip_install("transformers>=4.51.0,<5.0", "accelerate", "sentencepiece", "huggingface_hub")
    .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
)

vllm_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:latest",
        setup_dockerfile_commands=[
            "RUN ln -sf $(which python3) /usr/bin/python",
            "ENTRYPOINT []",
            "CMD []",
        ],
    )
    .pip_install("huggingface_hub", "sentencepiece", "transformers>=4.51.0,<5.0")
    .run_commands("pip install 'typing_extensions>=4.13.0'")  # Modal downgrades this; vLLM needs Sentinel
)

sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:latest",
        setup_dockerfile_commands=[
            "RUN ln -sf $(which python3) /usr/bin/python",
            "ENTRYPOINT []",
            "CMD []",
        ],
    )
    .pip_install("huggingface_hub", "sentencepiece", "transformers>=4.51.0,<5.0")
    .run_commands("pip install 'typing_extensions>=4.13.0'")  # Modal downgrades this; SGLang needs Sentinel
)


# Lightweight image for downloading weights (no GPU needed)
download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub[hf_transfer]", "transformers>=4.51.0,<5.0", "sentencepiece"
)


# ══════════════════════════════════════════════════════════════════
# Weight download (one-time, runs on cheap CPU instance)
# ══════════════════════════════════════════════════════════════════


@app.function(
    image=download_image,
    timeout=7200,  # 2 hours for ~960GB
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
    memory=8192,
)
def download_model():
    """Download all model weights to the persistent volume."""
    import huggingface_hub

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"Downloading {MODEL_NAME} to volume...", flush=True)
    path = huggingface_hub.snapshot_download(
        MODEL_NAME,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
    )
    print(f"Download complete: {path}", flush=True)

    # Commit volume so weights persist
    hf_cache.commit()
    print("Volume committed.", flush=True)


# ── Prompt construction ──


def _make_prompt(tokenizer, n_tokens: int) -> str:
    """Build a prompt of approximately n_tokens tokens."""
    base_ids = tokenizer.encode(_PROMPT_BASE, add_special_tokens=False)
    reps = (n_tokens // len(base_ids)) + 2
    ids = (base_ids * reps)[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
# Backend 1: Our custom engine (TP=8 + EP=8)
# ══════════════════════════════════════════════════════════════════


def _ours_worker(rank: int, world_size: int, results_dict: dict, prompt_lens: list):
    import time

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

    tp_group = dist.new_group(list(range(world_size)))
    ep_group = dist.new_group(list(range(world_size)))

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    # Build MoE kernel on rank 0
    if rank == 0:
        import subprocess

        torch_incs = subprocess.check_output(
            [
                "python3",
                "-c",
                "from torch.utils.cpp_extension import include_paths; "
                "print(' '.join('-I' + p for p in include_paths()))",
            ],
            text=True,
        ).strip()
        torch_libs = subprocess.check_output(
            [
                "python3",
                "-c",
                "from torch.utils.cpp_extension import library_paths; "
                "print(' '.join('-L' + p for p in library_paths()))",
            ],
            text=True,
        ).strip()
        extra_flags = f"{torch_incs} {torch_libs} -ltorch -ltorch_cpu -lc10 -ltorch_python"

        print("[Ours] Building MoE kernel...", flush=True)
        result = subprocess.run(
            [
                "make",
                "-C",
                "/workspace/src/csrc/kernels",
                f"EXTRA_NVCCFLAGS={extra_flags}",
                "moe_expert.cpython-312-x86_64-linux-gnu.so",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MoE kernel build failed:\n{result.stderr[:1000]}")
        print("[Ours] MoE kernel built successfully", flush=True)
    dist.barrier()

    sys.path.insert(0, "/workspace/src/csrc/kernels")

    from src.python.Qwen3.qwen import ExtraConfig, Qwen3ForCausalLM

    extra = ExtraConfig(
        tp_size=world_size,
        tp_rank=rank,
        tp_group=tp_group,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=ep_group,
        max_batch_size=1,
        max_len_override=MAX_SEQ_LEN,
    )

    if rank == 0:
        print("[Ours] Loading model...", flush=True)

    t_load = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_NAME,
        extra_config=extra,
        device=f"cuda:{rank}",
        dtype=torch.bfloat16,
    )
    dist.barrier()
    t_load = time.perf_counter() - t_load

    if rank == 0:
        print(f"[Ours] Model loaded in {t_load:.1f}s", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── TTFT measurement (model loading excluded) ──
    ttft_results = {}
    decode_results = {}

    for n_tok in prompt_lens:
        prompt_text = _make_prompt(tokenizer, n_tok)
        ids = tokenizer.encode(prompt_text, add_special_tokens=True)[:n_tok]
        input_ids = torch.tensor([ids], dtype=torch.long, device=f"cuda:{rank}")
        pos = torch.arange(len(ids), device=f"cuda:{rank}").unsqueeze(0)

        # Warmup
        for _ in range(WARMUP):
            model.stacked_kv_cache[0].zero_()
            model.stacked_kv_cache[1].zero_()
            model(input_ids, pos, len(ids))
        torch.cuda.synchronize()
        dist.barrier()

        # Measure TTFT (prefill only)
        ttft_times = []
        for _ in range(RUNS):
            model.stacked_kv_cache[0].zero_()
            model.stacked_kv_cache[1].zero_()
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(input_ids, pos, len(ids))
            torch.cuda.synchronize()
            ttft_times.append((time.perf_counter() - t0) * 1000.0)

        ttft_times.sort()
        median_ttft = ttft_times[len(ttft_times) // 2]
        ttft_results[n_tok] = median_ttft

        if rank == 0:
            tps = len(ids) / (median_ttft / 1000.0)
            print(f"  [Ours] n={n_tok:>5} → TTFT {median_ttft:8.1f} ms  ({tps:.0f} tok/s prefill)", flush=True)

        # Measure decode throughput
        model.stacked_kv_cache[0].zero_()
        model.stacked_kv_cache[1].zero_()
        logits = model(input_ids, pos, len(ids))
        first_tok = logits[0, -1].argmax().item()

        for _ in range(WARMUP):
            kv_k_saved = model.stacked_kv_cache[0].clone()
            kv_v_saved = model.stacked_kv_cache[1].clone()
            tok = first_tok
            seq = len(ids) + 1
            for _ in range(GEN_TOKENS):
                tok_t = torch.tensor([[tok]], dtype=torch.long, device=f"cuda:{rank}")
                pos_t = torch.tensor([[seq - 1]], dtype=torch.long, device=f"cuda:{rank}")
                out = model(tok_t, pos_t, seq)
                tok = out[0, -1].argmax().item()
                seq += 1
            model.stacked_kv_cache[0].copy_(kv_k_saved)
            model.stacked_kv_cache[1].copy_(kv_v_saved)
        torch.cuda.synchronize()
        dist.barrier()

        decode_times = []
        for _ in range(RUNS):
            kv_k_saved = model.stacked_kv_cache[0].clone()
            kv_v_saved = model.stacked_kv_cache[1].clone()
            tok = first_tok
            seq = len(ids) + 1
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(GEN_TOKENS):
                tok_t = torch.tensor([[tok]], dtype=torch.long, device=f"cuda:{rank}")
                pos_t = torch.tensor([[seq - 1]], dtype=torch.long, device=f"cuda:{rank}")
                out = model(tok_t, pos_t, seq)
                tok = out[0, -1].argmax().item()
                seq += 1
            torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - t0) * 1000.0)
            model.stacked_kv_cache[0].copy_(kv_k_saved)
            model.stacked_kv_cache[1].copy_(kv_v_saved)

        decode_times.sort()
        median_decode = decode_times[len(decode_times) // 2]
        ms_per_tok = median_decode / GEN_TOKENS
        decode_results[n_tok] = ms_per_tok

        if rank == 0:
            print(f"  [Ours] n={n_tok:>5} → Decode {ms_per_tok:.2f} ms/tok", flush=True)

    if rank == 0:
        results_dict["ttft"] = dict(ttft_results)
        results_dict["decode_ms_per_tok"] = dict(decode_results)

    dist.barrier()
    dist.destroy_process_group()


@app.function(
    image=ours_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
)
def run_ours(prompt_lens: list[int]) -> dict:
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_ours_worker, args=(8, results, prompt_lens), nprocs=8, join=True)
    return dict(results)


# ══════════════════════════════════════════════════════════════════
# Backend 2: vLLM (TP=8)
# ══════════════════════════════════════════════════════════════════


@app.function(
    image=vllm_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
)
def run_vllm(prompt_lens: list[int]) -> dict:
    import subprocess
    import time as _time

    # Modal's runtime downgrades typing_extensions; pydantic_core needs Sentinel (>=4.13)
    subprocess.check_call(
        ["pip", "install", "-q", "typing_extensions>=4.13.0"]
    )

    import torch
    from transformers import AutoTokenizer

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    print(f"[vLLM] Device: {device_name} x {torch.cuda.device_count()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ttft_results = {}
    decode_results = {}

    try:
        from vllm import LLM, SamplingParams

        print("[vLLM] Loading model...", flush=True)
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=8,
            dtype="bfloat16",
            max_model_len=max(prompt_lens) + GEN_TOKENS + 64,
            trust_remote_code=True,
            enable_expert_parallel=True,
        )
        print("[vLLM] Model loaded, starting TTFT measurements", flush=True)

        params = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS)
        params_1tok = SamplingParams(temperature=0.0, max_tokens=1)

        for n_tok in prompt_lens:
            prompt_text = _make_prompt(tokenizer, n_tok)

            # Warmup
            for _ in range(WARMUP):
                llm.generate([prompt_text], params)

            # TTFT: try vLLM metrics first, fall back to wall-clock
            ttft_times = []
            decode_ms_per_tok_list = []
            for _ in range(RUNS):
                outputs = llm.generate([prompt_text], params)
                output = outputs[0]
                m = output.metrics

                if hasattr(m, "first_token_time") and m.first_token_time is not None:
                    ttft_times.append((m.first_token_time - m.arrival_time) * 1000.0)

                n_gen = len(output.outputs[0].token_ids)
                if (
                    n_gen > 1
                    and hasattr(m, "last_token_time")
                    and m.last_token_time
                    and hasattr(m, "first_token_time")
                    and m.first_token_time
                ):
                    decode_total = (m.last_token_time - m.first_token_time) * 1000.0
                    decode_ms_per_tok_list.append(decode_total / (n_gen - 1))

            if not ttft_times:
                # Wall-clock fallback
                for _ in range(RUNS):
                    t0 = _time.perf_counter()
                    llm.generate([prompt_text], params_1tok)
                    ttft_times.append((_time.perf_counter() - t0) * 1000.0)

            ttft_times.sort()
            median_ttft = ttft_times[len(ttft_times) // 2]
            ttft_results[n_tok] = median_ttft
            print(f"  [vLLM] n={n_tok:>5} → TTFT {median_ttft:8.1f} ms", flush=True)

            if decode_ms_per_tok_list:
                decode_ms_per_tok_list.sort()
                decode_results[n_tok] = decode_ms_per_tok_list[len(decode_ms_per_tok_list) // 2]
                print(f"  [vLLM] n={n_tok:>5} → Decode {decode_results[n_tok]:.2f} ms/tok", flush=True)

        del llm

    except Exception as e:
        import traceback

        print(f"[vLLM] FAILED: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "ttft": {}, "decode_ms_per_tok": {}}

    return {"ttft": ttft_results, "decode_ms_per_tok": decode_results}


# ══════════════════════════════════════════════════════════════════
# Backend 3: SGLang (TP=8)
# ══════════════════════════════════════════════════════════════════


@app.function(
    image=sglang_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
)
def run_sglang(prompt_lens: list[int]) -> dict:
    import subprocess
    import time as _time

    # Modal's runtime downgrades typing_extensions; pydantic_core needs Sentinel (>=4.13)
    subprocess.check_call(
        ["pip", "install", "-q", "typing_extensions>=4.13.0"]
    )

    import torch
    from transformers import AutoTokenizer

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    print(f"[SGLang] Device: {device_name} x {torch.cuda.device_count()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ttft_results = {}
    decode_results = {}

    try:
        import sglang as sgl

        print("[SGLang] Loading model...", flush=True)
        engine = sgl.Engine(
            model_path=MODEL_NAME,
            tp_size=8,
            dtype="bfloat16",
        )
        print("[SGLang] Model loaded, starting TTFT measurements", flush=True)

        for n_tok in prompt_lens:
            prompt_text = _make_prompt(tokenizer, n_tok)

            # Warmup
            for _ in range(WARMUP):
                engine.generate(prompt_text, {"max_new_tokens": GEN_TOKENS, "temperature": 0.0})

            # TTFT (generate 1 token = prefill only)
            ttft_times = []
            for _ in range(RUNS):
                t0 = _time.perf_counter()
                engine.generate(prompt_text, {"max_new_tokens": 1, "temperature": 0.0})
                ttft_times.append((_time.perf_counter() - t0) * 1000.0)

            ttft_times.sort()
            median_ttft = ttft_times[len(ttft_times) // 2]
            ttft_results[n_tok] = median_ttft
            print(f"  [SGLang] n={n_tok:>5} → TTFT {median_ttft:8.1f} ms", flush=True)

            # Decode
            total_times = []
            for _ in range(RUNS):
                t0 = _time.perf_counter()
                engine.generate(prompt_text, {"max_new_tokens": GEN_TOKENS, "temperature": 0.0})
                total_times.append((_time.perf_counter() - t0) * 1000.0)

            total_times.sort()
            median_total = total_times[len(total_times) // 2]
            decode_total = max(median_total - median_ttft, 0.1)
            ms_per_tok = decode_total / (GEN_TOKENS - 1) if GEN_TOKENS > 1 else decode_total
            decode_results[n_tok] = ms_per_tok
            print(f"  [SGLang] n={n_tok:>5} → Decode {ms_per_tok:.2f} ms/tok", flush=True)

        engine.shutdown()

    except Exception as e:
        import traceback

        print(f"[SGLang] FAILED: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "ttft": {}, "decode_ms_per_tok": {}}

    return {"ttft": ttft_results, "decode_ms_per_tok": decode_results}


# ══════════════════════════════════════════════════════════════════
# Entrypoint: run sequentially, collect & compare
# ══════════════════════════════════════════════════════════════════


def _print_table(title: str, all_results: dict[str, dict], prompt_lens: list[int], key: str, fmt: str = ".1f"):
    """Print a comparison table for one metric across backends."""
    backends = list(all_results.keys())

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    header = f"{'prompt_len':>10}"
    for b in backends:
        header += f"  {b:>12}"
    # Speedup ratios vs Ours
    for b in backends:
        if b != "Ours":
            header += f"  {'Ours/' + b:>12}"
    print(header)
    print("-" * len(header))

    for n in prompt_lens:
        row = f"{n:>10}"
        vals = {}
        for b in backends:
            v = all_results[b].get(key, {}).get(n)
            vals[b] = v
            row += f"  {(f'{v:{fmt}}' if v is not None else 'N/A'):>12}"
        for b in backends:
            if b != "Ours":
                ours_v = vals.get("Ours")
                other_v = vals.get(b)
                if ours_v is not None and other_v is not None and other_v > 0:
                    row += f"  {ours_v / other_v:>11.2f}x"
                else:
                    row += f"  {'N/A':>12}"
        print(row)


@app.function(
    image=download_image,
    timeout=10800,  # 3 hours total (sum of all backends)
    memory=2048,
)
def orchestrate(prompt_lens: list[int]) -> dict:
    """Run all backends sequentially on Modal servers.

    This runs as a remote function so that the .remote() calls to each
    backend happen server-to-server, avoiding local heartbeat timeouts
    that killed previous runs.
    """
    all_results = {}

    print(f"\n{'='*70}")
    print(f"TTFT Benchmark: {MODEL_NAME}")
    print("Hardware: 8x B200 per backend")
    print(f"Prompt lengths: {prompt_lens}")
    print(f"Decode tokens: {GEN_TOKENS}")
    print("Note: Model loading time is NOT included in TTFT measurements")
    print(f"{'='*70}")

    # Run backends sequentially
    print("\n--- Running: Our custom engine (TP=8, EP=8) ---", flush=True)
    try:
        all_results["Ours"] = run_ours.remote(prompt_lens)
    except Exception as e:
        print(f"[WARN] Our engine failed: {e}")
        all_results["Ours"] = {"ttft": {}, "decode_ms_per_tok": {}}

    print("\n--- Running: vLLM (TP=8, EP=8) ---", flush=True)
    try:
        all_results["vLLM"] = run_vllm.remote(prompt_lens)
    except Exception as e:
        print(f"[WARN] vLLM failed: {e}")
        all_results["vLLM"] = {"ttft": {}, "decode_ms_per_tok": {}}

    print("\n--- Running: SGLang (TP=8) ---", flush=True)
    try:
        all_results["SGLang"] = run_sglang.remote(prompt_lens)
    except Exception as e:
        print(f"[WARN] SGLang failed: {e}")
        all_results["SGLang"] = {"ttft": {}, "decode_ms_per_tok": {}}

    # Print comparison tables
    _print_table(
        "TTFT (median ms, lower is better — model loading excluded)", all_results, prompt_lens, "ttft", fmt=".1f"
    )

    _print_table("DECODE (ms/tok, lower is better)", all_results, prompt_lens, "decode_ms_per_tok", fmt=".2f")

    # Errors
    for name, res in all_results.items():
        if "error" in res:
            print(f"\n[{name}] Error: {res['error']}")

    print(f"\n{'='*70}")
    print("Benchmark complete.")
    print(f"{'='*70}")

    return all_results


@app.local_entrypoint()
def main(download_only: bool = False):
    if download_only:
        print(f"Downloading {MODEL_NAME} weights to volume...")
        download_model.remote()
        print("Done. Weights are cached in the 'hf-model-cache' volume.")
        return

    # Dispatch to remote orchestrator — avoids local heartbeat timeouts
    print("Dispatching benchmark to remote orchestrator...")
    results = orchestrate.remote(PROMPT_LENS)

    # Reprint the comparison tables locally for convenience
    _print_table("TTFT (median ms, lower is better — model loading excluded)", results, PROMPT_LENS, "ttft", fmt=".1f")
    _print_table("DECODE (ms/tok, lower is better)", results, PROMPT_LENS, "decode_ms_per_tok", fmt=".2f")

    for name, res in results.items():
        if "error" in res:
            print(f"\n[{name}] Error: {res['error']}")

    print("\nBenchmark complete.")
