"""Full-model TTFT benchmark: Ours vs vLLM vs SGLang (Qwen3-Coder-480B on 8xB200).

Compares time-to-first-token across three inference backends for the
Qwen3-Coder-480B-A35B-Instruct MoE model on 8 NVIDIA B200 GPUs.

Each backend loads the model ONCE via @modal.enter() and keeps it in GPU
memory. The benchmark method can be called repeatedly without reloading.
With keep_warm=1, containers stay alive between `modal run` invocations.

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
# vLLM and SGLang use thin Dockerfiles (Dockerfile.vllm, Dockerfile.sglang) that
# pull the official images but clear ENTRYPOINT/CMD so Modal can bootstrap.

ours_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.b200", force_build=force_rebuild)
    .pip_install(
        "transformers>=4.51.0,<5.0",
        "accelerate",
        "sentencepiece",
        "huggingface_hub",
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
)

vllm_image = (
    modal.Image.from_dockerfile(
        PROJECT_ROOT / "modal_tests" / "Dockerfile.vllm",
        force_build=force_rebuild,
    )
    .pip_install(
        "huggingface_hub",
        "sentencepiece",
        "transformers>=4.51.0,<5.0",
    )
    .run_commands("pip install 'typing_extensions>=4.13.0'")
)

sglang_image = (
    modal.Image.from_dockerfile(
        PROJECT_ROOT / "modal_tests" / "Dockerfile.sglang",
        force_build=force_rebuild,
    )
    .pip_install(
        "huggingface_hub",
        "sentencepiece",
        "transformers>=4.51.0,<5.0",
    )
    .run_commands("pip install 'typing_extensions>=4.13.0'")
)

# Lightweight image for downloading weights (no GPU needed)
download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub[hf_transfer]",
    "transformers>=4.51.0,<5.0",
    "sentencepiece",
)


# ══════════════════════════════════════════════════════════════════
# Weight download (one-time, runs on cheap CPU instance)
# ══════════════════════════════════════════════════════════════════


@app.function(
    image=download_image,
    timeout=7200,
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
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.model",
            "*.tiktoken",
        ],
    )
    print(f"Download complete: {path}", flush=True)

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
# Model loaded once via persistent workers, kept in GPU memory.
# ══════════════════════════════════════════════════════════════════


def _ours_persistent_worker(rank, world_size, barrier, exit_flag, prompt_lens_ref, results_dict):
    """Persistent worker: loads model once, benchmarks on demand via barrier."""
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
    # Use cuBLAS GEMM path (tcgen05 persistent kernel has driver compat issues)
    os.environ.setdefault("MOE_FORCE_CUBLAS", "1")

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
        extra_flags = f"{torch_incs} {torch_libs}" " -ltorch -ltorch_cpu -lc10 -ltorch_python"

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

    # ── Signal ready to main process ──
    barrier.wait()

    # ── Task loop: wait for benchmark requests ──
    while True:
        barrier.wait()  # Wait for task from main
        if exit_flag.value:
            break

        prompt_lens = list(prompt_lens_ref)
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
                model(input_ids, pos, len(ids))
                torch.cuda.synchronize()
                ttft_times.append((time.perf_counter() - t0) * 1000.0)

            ttft_times.sort()
            median_ttft = ttft_times[len(ttft_times) // 2]
            ttft_results[n_tok] = median_ttft

            if rank == 0:
                tps = len(ids) / (median_ttft / 1000.0)
                print(
                    f"  [Ours] n={n_tok:>5} → " f"TTFT {median_ttft:8.1f} ms  " f"({tps:.0f} tok/s prefill)",
                    flush=True,
                )

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
                print(
                    f"  [Ours] n={n_tok:>5} → " f"Decode {ms_per_tok:.2f} ms/tok",
                    flush=True,
                )

        if rank == 0:
            results_dict["ttft"] = dict(ttft_results)
            results_dict["decode_ms_per_tok"] = dict(decode_results)

        dist.barrier()
        barrier.wait()  # Signal done to main

    dist.barrier()
    dist.destroy_process_group()


@app.cls(
    image=ours_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
    min_containers=1,
)
@modal.concurrent(max_inputs=1)
class OursEngine:
    """Our engine with persistent 8-GPU worker pool. Model loads once."""

    @modal.enter()
    def setup(self):
        import multiprocessing as mp

        self._manager = mp.Manager()
        self._barrier = mp.Barrier(9)  # 8 workers + 1 main
        self._exit_flag = mp.Value("i", 0)
        self._prompt_lens = self._manager.list()
        self._results = self._manager.dict()

        self._workers = []
        for rank in range(8):
            p = mp.Process(
                target=_ours_persistent_worker,
                args=(
                    rank,
                    8,
                    self._barrier,
                    self._exit_flag,
                    self._prompt_lens,
                    self._results,
                ),
            )
            p.start()
            self._workers.append(p)

        print("[Ours] Waiting for workers to load model...", flush=True)
        self._barrier.wait()  # Wait for all workers to finish loading
        print("[Ours] All 8 workers ready, model in GPU memory", flush=True)

    @modal.method()
    def benchmark(self, prompt_lens: list[int]) -> dict:
        self._prompt_lens[:] = prompt_lens
        self._barrier.wait()  # Signal workers to start benchmark
        self._barrier.wait()  # Wait for workers to finish
        return dict(self._results)

    @modal.exit()
    def teardown(self):
        self._exit_flag.value = 1
        self._barrier.wait()  # Release workers to check exit flag
        for w in self._workers:
            w.join(timeout=30)


# ══════════════════════════════════════════════════════════════════
# Backend 2: vLLM (TP=8) — model loaded once in @modal.enter()
# ══════════════════════════════════════════════════════════════════


@app.cls(
    image=vllm_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
    min_containers=1,
)
@modal.concurrent(max_inputs=1)
class VllmEngine:
    """vLLM engine. Model loads once and stays in GPU memory."""

    @modal.enter()
    def load(self):
        import subprocess

        # Modal runtime may downgrade typing_extensions; pydantic needs Sentinel
        subprocess.check_call(["pip", "install", "-q", "typing_extensions>=4.13.0"])

        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        print(
            f"[vLLM] Device: {device} x {torch.cuda.device_count()}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        print("[vLLM] Loading model...", flush=True)
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=8,
            dtype="bfloat16",
            max_model_len=MAX_SEQ_LEN + GEN_TOKENS + 64,
            trust_remote_code=True,
            enable_expert_parallel=True,
        )
        print("[vLLM] Model loaded, ready for benchmarks", flush=True)

    @modal.method()
    def benchmark(self, prompt_lens: list[int]) -> dict:
        import concurrent.futures
        import time as _time

        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS)
        params_1tok = SamplingParams(temperature=0.0, max_tokens=1)
        ttft_results = {}
        decode_results = {}

        # Timeout wrapper — vLLM generate() can hang on B200 EP mode
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        PER_CALL_TIMEOUT = 180  # seconds

        def _gen(prompt, sp):
            fut = _executor.submit(self.llm.generate, [prompt], sp)
            return fut.result(timeout=PER_CALL_TIMEOUT)

        try:
            for n_tok in prompt_lens:
                prompt_text = _make_prompt(self.tokenizer, n_tok)
                print(f"  [vLLM] n={n_tok:>5} starting warmup...", flush=True)

                # Warmup
                for wi in range(WARMUP):
                    print(f"  [vLLM]   warmup {wi+1}/{WARMUP}...", flush=True)
                    _gen(prompt_text, params)

                print(f"  [vLLM] n={n_tok:>5} measuring...", flush=True)

                # TTFT: try vLLM metrics, fall back to wall-clock
                ttft_times = []
                decode_ms_list = []
                for ri in range(RUNS):
                    outputs = _gen(prompt_text, params)
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
                        dt = (m.last_token_time - m.first_token_time) * 1000.0
                        decode_ms_list.append(dt / (n_gen - 1))

                if not ttft_times:
                    # Wall-clock fallback (generate 1 token only)
                    for _ in range(RUNS):
                        t0 = _time.perf_counter()
                        _gen(prompt_text, params_1tok)
                        ttft_times.append((_time.perf_counter() - t0) * 1000.0)

                ttft_times.sort()
                median_ttft = ttft_times[len(ttft_times) // 2]
                ttft_results[n_tok] = median_ttft
                print(
                    f"  [vLLM] n={n_tok:>5} → TTFT {median_ttft:8.1f} ms",
                    flush=True,
                )

                if decode_ms_list:
                    decode_ms_list.sort()
                    decode_results[n_tok] = decode_ms_list[len(decode_ms_list) // 2]
                    print(
                        f"  [vLLM] n={n_tok:>5} → " f"Decode {decode_results[n_tok]:.2f} ms/tok",
                        flush=True,
                    )

        except concurrent.futures.TimeoutError:
            print("[vLLM] TIMEOUT: generate() hung for > 180s, skipping remaining", flush=True)
            return {"error": "timeout", "ttft": ttft_results, "decode_ms_per_tok": decode_results}
        except Exception as e:
            import traceback

            print(f"[vLLM] FAILED: {e}", flush=True)
            traceback.print_exc()
            return {"error": str(e), "ttft": ttft_results, "decode_ms_per_tok": decode_results}
        finally:
            _executor.shutdown(wait=False)

        return {"ttft": ttft_results, "decode_ms_per_tok": decode_results}

    @modal.exit()
    def unload(self):
        if hasattr(self, "llm"):
            del self.llm


# ══════════════════════════════════════════════════════════════════
# Backend 3: SGLang (TP=8) — model loaded once in @modal.enter()
# ══════════════════════════════════════════════════════════════════


@app.cls(
    image=sglang_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
    min_containers=1,
)
@modal.concurrent(max_inputs=1)
class SglangEngine:
    """SGLang engine. Model loads once and stays in GPU memory."""

    @modal.enter()
    def load(self):
        import subprocess

        subprocess.check_call(["pip", "install", "-q", "typing_extensions>=4.13.0"])

        import torch
        from transformers import AutoTokenizer

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        print(
            f"[SGLang] Device: {device} x {torch.cuda.device_count()}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        import sglang as sgl

        print("[SGLang] Loading model...", flush=True)
        self.engine = sgl.Engine(
            model_path=MODEL_NAME,
            tp_size=8,
            dtype="bfloat16",
        )
        print("[SGLang] Model loaded, ready for benchmarks", flush=True)

    @modal.method()
    def benchmark(self, prompt_lens: list[int]) -> dict:
        import time as _time

        ttft_results = {}
        decode_results = {}

        try:
            for n_tok in prompt_lens:
                prompt_text = _make_prompt(self.tokenizer, n_tok)

                # Warmup
                for _ in range(WARMUP):
                    self.engine.generate(
                        prompt_text,
                        {"max_new_tokens": GEN_TOKENS, "temperature": 0.0},
                    )

                # TTFT (generate 1 token = prefill only)
                ttft_times = []
                for _ in range(RUNS):
                    t0 = _time.perf_counter()
                    self.engine.generate(
                        prompt_text,
                        {"max_new_tokens": 1, "temperature": 0.0},
                    )
                    ttft_times.append((_time.perf_counter() - t0) * 1000.0)

                ttft_times.sort()
                median_ttft = ttft_times[len(ttft_times) // 2]
                ttft_results[n_tok] = median_ttft
                print(
                    f"  [SGLang] n={n_tok:>5} → " f"TTFT {median_ttft:8.1f} ms",
                    flush=True,
                )

                # Decode
                total_times = []
                for _ in range(RUNS):
                    t0 = _time.perf_counter()
                    self.engine.generate(
                        prompt_text,
                        {"max_new_tokens": GEN_TOKENS, "temperature": 0.0},
                    )
                    total_times.append((_time.perf_counter() - t0) * 1000.0)

                total_times.sort()
                median_total = total_times[len(total_times) // 2]
                decode_total = max(median_total - median_ttft, 0.1)
                ms_per_tok = decode_total / (GEN_TOKENS - 1) if GEN_TOKENS > 1 else decode_total
                decode_results[n_tok] = ms_per_tok
                print(
                    f"  [SGLang] n={n_tok:>5} → " f"Decode {ms_per_tok:.2f} ms/tok",
                    flush=True,
                )

        except Exception as e:
            import traceback

            print(f"[SGLang] FAILED: {e}", flush=True)
            traceback.print_exc()
            return {"error": str(e), "ttft": {}, "decode_ms_per_tok": {}}

        return {"ttft": ttft_results, "decode_ms_per_tok": decode_results}

    @modal.exit()
    def shutdown(self):
        if hasattr(self, "engine"):
            self.engine.shutdown()


# ══════════════════════════════════════════════════════════════════
# Entrypoint: run sequentially, collect & compare
# ══════════════════════════════════════════════════════════════════


def _print_table(
    title: str,
    all_results: dict[str, dict],
    prompt_lens: list[int],
    key: str,
    fmt: str = ".1f",
):
    """Print a comparison table for one metric across backends."""
    backends = list(all_results.keys())

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    header = f"{'prompt_len':>10}"
    for b in backends:
        header += f"  {b:>12}"
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
                if ours_v and other_v and other_v > 0:
                    row += f"  {ours_v / other_v:>11.2f}x"
                else:
                    row += f"  {'N/A':>12}"
        print(row)


@app.function(
    image=download_image,
    timeout=10800,
    memory=2048,
)
def orchestrate(prompt_lens: list[int]) -> dict:
    """Run all backends sequentially. Models stay loaded via keep_warm."""
    all_results = {}

    print(f"\n{'='*70}")
    print(f"TTFT Benchmark: {MODEL_NAME}")
    print("Hardware: 8x B200 per backend")
    print(f"Prompt lengths: {prompt_lens}")
    print(f"Decode tokens: {GEN_TOKENS}")
    print(f"{'='*70}")

    print("\n--- Running: Our engine (TP=8, EP=8) ---", flush=True)
    try:
        ours = OursEngine()
        all_results["Ours"] = ours.benchmark.remote(prompt_lens)
    except Exception as e:
        print(f"[WARN] Our engine failed: {e}")
        all_results["Ours"] = {"ttft": {}, "decode_ms_per_tok": {}}

    print("\n--- Running: vLLM (TP=8, EP=8) ---", flush=True)
    try:
        vllm_engine = VllmEngine()
        all_results["vLLM"] = vllm_engine.benchmark.remote(prompt_lens)
    except Exception as e:
        print(f"[WARN] vLLM failed: {e}")
        all_results["vLLM"] = {"ttft": {}, "decode_ms_per_tok": {}}

    print("\n--- Running: SGLang (TP=8) ---", flush=True)
    try:
        sglang_engine = SglangEngine()
        all_results["SGLang"] = sglang_engine.benchmark.remote(prompt_lens)
    except Exception as e:
        print(f"[WARN] SGLang failed: {e}")
        all_results["SGLang"] = {"ttft": {}, "decode_ms_per_tok": {}}

    # Print comparison tables
    _print_table("TTFT (median ms)", all_results, prompt_lens, "ttft", fmt=".1f")
    _print_table(
        "DECODE (ms/tok)",
        all_results,
        prompt_lens,
        "decode_ms_per_tok",
        fmt=".2f",
    )

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

    print("Dispatching benchmark to remote orchestrator...")
    results = orchestrate.remote(PROMPT_LENS)

    _print_table(
        "TTFT (median ms, lower is better — model loading excluded)",
        results,
        PROMPT_LENS,
        "ttft",
        fmt=".1f",
    )
    _print_table(
        "DECODE (ms/tok, lower is better)",
        results,
        PROMPT_LENS,
        "decode_ms_per_tok",
        fmt=".2f",
    )

    for name, res in results.items():
        if "error" in res:
            print(f"\n[{name}] Error: {res['error']}")

    print("\nBenchmark complete.")
