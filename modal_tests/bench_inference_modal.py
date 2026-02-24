"""
Batch-size-1 greedy decode latency: megakernel vs HuggingFace vs vLLM vs SGLang.
Targets B200 on Modal.

Usage:
    modal run modal_tests/bench_inference_modal.py

    # force image rebuild:
    FORCE_REBUILD=1 modal run modal_tests/bench_inference_modal.py

    # custom options:
    modal run modal_tests/bench_inference_modal.py --tokens 200 --backends megakernel,hf
"""

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent
force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

app = modal.App("cs450-inference-bench")

# Persistent volume so the 3.4 GB model is downloaded once and reused.
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# ── Image ─────────────────────────────────────────────────────────────────────
# Build on top of the same Dockerfile.b200 used everywhere else in the project.

bench_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.b200", force_build=force_rebuild)
    .pip_install(
        "transformers>=4.51.0,<5.0",
        "accelerate",
        "sentencepiece",
        "huggingface_hub",
    )
    # vLLM: install latest stable; gracefully skipped inside bench if unsupported.
    .pip_install("vllm", extra_options="--extra-index-url https://download.pytorch.org/whl/nightly/cu130")
    # SGLang: optional, silently skipped if not importable.
    .pip_install("sglang[all]", extra_options="--find-links https://flashinfer.ai/whl/cu130/torch2.7/")
    .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
)


# ── Benchmark function ────────────────────────────────────────────────────────


@app.function(
    image=bench_image,
    gpu="B200",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache},
)
def run_bench(
    tokens: int = 100,
    warmup: int = 3,
    runs: int = 5,
    prompt: str = "Hello",
    backends: str = "megakernel,hf,vllm,sglang",
    no_correctness: bool = False,
) -> dict:
    import gc
    import time
    import warnings

    warnings.filterwarnings("ignore")

    # Make workspace importable so `from src.python.Qwen3.xxx import ...` works.
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backends: {backends}")
    print(f"Tokens: {tokens}  Warmup: {warmup}  Runs: {runs}")
    print(f"Prompt: {prompt!r}")

    backend_set = {b.strip().lower() for b in backends.split(",")}
    results = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _timed(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    # ── correctness check ─────────────────────────────────────────────────────

    if not no_correctness and "megakernel" in backend_set:
        print("\n--- Correctness check (8 tokens) ---")
        try:
            from src.python.Qwen3.bench import correctness_check

            correctness_check(prompt, n=8)
        except Exception as e:
            print(f"  Correctness check failed: {e}")

    # ── megakernel ────────────────────────────────────────────────────────────

    if "megakernel" in backend_set:
        print("\n--- Megakernel ---")
        try:
            from src.python.Qwen3.decoder import Decoder

            print("  Loading decoder...")
            dec = Decoder(verbose=True)

            def _mk_run():
                dec.reset()
                ids = dec.tokenizer.encode(prompt, add_special_tokens=True)
                for tid in ids[:-1]:
                    dec.step(tid)
                tok = ids[-1]
                eos = dec.tokenizer.eos_token_id
                for _ in range(tokens):
                    tok = dec.step(tok)
                    if tok == eos:
                        break

            avg_s = _timed(_mk_run)
            tps = tokens / avg_s
            mpt = avg_s * 1000 / tokens
            results["megakernel"] = {"tok_per_s": round(tps, 1), "ms_per_tok": round(mpt, 3)}
            print(f"  {tps:.1f} tok/s  |  {mpt:.3f} ms/tok")

            del dec
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results["megakernel"] = {"error": str(e)}

    # ── HuggingFace ───────────────────────────────────────────────────────────

    if "hf" in backend_set:
        print("\n--- HuggingFace PyTorch ---")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            MODEL = "Qwen/Qwen3-1.7B"
            print("  Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
            model.eval()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            def _hf_run():
                with torch.no_grad():
                    model.generate(
                        input_ids,
                        max_new_tokens=tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

            avg_s = _timed(_hf_run)
            tps = tokens / avg_s
            mpt = avg_s * 1000 / tokens
            results["hf"] = {"tok_per_s": round(tps, 1), "ms_per_tok": round(mpt, 3)}
            print(f"  {tps:.1f} tok/s  |  {mpt:.3f} ms/tok")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results["hf"] = {"error": str(e)}

    # ── vLLM ──────────────────────────────────────────────────────────────────

    if "vllm" in backend_set:
        print("\n--- vLLM ---")
        try:
            from vllm import LLM, SamplingParams

            MODEL = "Qwen/Qwen3-1.7B"
            print("  Loading vLLM engine...")
            llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096)
            params = SamplingParams(temperature=0.0, max_tokens=tokens)

            def _vllm_run():
                llm.generate([prompt], params)

            avg_s = _timed(_vllm_run)
            tps = tokens / avg_s
            mpt = avg_s * 1000 / tokens
            results["vllm"] = {"tok_per_s": round(tps, 1), "ms_per_tok": round(mpt, 3)}
            print(f"  {tps:.1f} tok/s  |  {mpt:.3f} ms/tok")

            del llm
            gc.collect()
            torch.cuda.empty_cache()

        except ImportError:
            print("  vLLM not installed — skipping.")
            results["vllm"] = {"error": "not installed"}
        except Exception as e:
            print(f"  FAILED: {e}")
            results["vllm"] = {"error": str(e)}

    # ── SGLang ────────────────────────────────────────────────────────────────

    if "sglang" in backend_set:
        print("\n--- SGLang ---")
        try:
            import sglang as sgl
            from sglang.srt.sampling.sampling_params import SamplingParams as SGLParams

            MODEL = "Qwen/Qwen3-1.7B"
            print("  Loading SGLang engine...")
            engine = sgl.Engine(model_path=MODEL, dtype="bfloat16", tp_size=1)
            params = SGLParams(max_new_tokens=tokens, temperature=0.0)

            def _sgl_run():
                engine.generate(prompts=[prompt], sampling_params=params)

            avg_s = _timed(_sgl_run)
            tps = tokens / avg_s
            mpt = avg_s * 1000 / tokens
            results["sglang"] = {"tok_per_s": round(tps, 1), "ms_per_tok": round(mpt, 3)}
            print(f"  {tps:.1f} tok/s  |  {mpt:.3f} ms/tok")

            engine.shutdown()
            gc.collect()
            torch.cuda.empty_cache()

        except ImportError:
            print("  SGLang not installed — skipping.")
            results["sglang"] = {"error": "not installed"}
        except Exception as e:
            print(f"  FAILED: {e}")
            results["sglang"] = {"error": str(e)}

    return results


# ── local entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    tokens: int = 100,
    warmup: int = 3,
    runs: int = 5,
    prompt: str = "Hello",
    backends: str = "megakernel,hf,vllm,sglang",
    no_correctness: bool = False,
):
    print("Running inference benchmark on B200...")

    results = run_bench.remote(
        tokens=tokens,
        warmup=warmup,
        runs=runs,
        prompt=prompt,
        backends=backends,
        no_correctness=no_correctness,
    )

    mk_tps = results.get("megakernel", {}).get("tok_per_s")

    print("\n" + "=" * 60)
    print(f"{'Backend':<16} {'tok/s':>10} {'ms/tok':>10}  {'vs megakernel':>14}")
    print("-" * 60)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<14} {'— ' + r['error']}")
            continue
        tps = r["tok_per_s"]
        mpt = r["ms_per_tok"]
        speedup = f"{mk_tps / tps:.2f}x slower" if mk_tps and name != "megakernel" else "baseline"
        print(f"  {name:<14} {tps:>10.1f} {mpt:>10.3f}  {speedup:>14}")
    print("=" * 60)
