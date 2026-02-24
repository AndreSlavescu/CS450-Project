"""Qwen3-1.7B batch-size-1 decode latency benchmark.

Compares the persistent megakernel against:
  - HuggingFace PyTorch (always available)
  - vLLM             (skipped if not installed)
  - SGLang           (skipped if not installed)

Usage:
    # from repo root:
    python -m src.python.Qwen3.bench

    # options:
    python -m src.python.Qwen3.bench --tokens 100 --warmup 3 --runs 5
    python -m src.python.Qwen3.bench --prompt "Explain quantum entanglement"
    python -m src.python.Qwen3.bench --no-correctness   # skip token comparison
"""

import argparse
import gc
import time
import warnings

import torch

warnings.filterwarnings("ignore")

MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_TOKENS = 100
DEFAULT_WARMUP = 3
DEFAULT_RUNS = 5
DEFAULT_PROMPT = "Hello"
CORRECTNESS_TOKENS = 8  # tokens to compare for correctness check


# ── timing helper ─────────────────────────────────────────────────────────────


def _timed(fn, warmup: int, runs: int) -> float:
    """Return average seconds per call over `runs` measured calls."""
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


# ── HuggingFace baseline ──────────────────────────────────────────────────────


def bench_hf(prompt: str, tokens: int, warmup: int, runs: int) -> tuple:
    """
    Returns (tok_per_sec, ms_per_tok, generated_ids_list).
    generated_ids_list is from the last run (for correctness comparison).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    gen_ids_out = []

    def run():
        with torch.no_grad():
            out = model.generate(  # noqa: F821
                input_ids,
                max_new_tokens=tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids_out.clear()
        gen_ids_out.extend(out[0, -tokens:].tolist())

    avg_s = _timed(run, warmup, runs)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return tokens / avg_s, avg_s * 1000 / tokens, gen_ids_out


# ── megakernel ────────────────────────────────────────────────────────────────


def bench_megakernel(prompt: str, tokens: int, warmup: int, runs: int) -> tuple:
    """Returns (tok_per_sec, ms_per_tok, generated_ids_list)."""
    from .decoder import Decoder

    print("  Loading megakernel decoder...")
    dec = Decoder(verbose=False)
    gen_ids_out = []

    def run():
        dec.reset()  # noqa: F821
        ids = dec.tokenizer.encode(prompt, add_special_tokens=True)  # noqa: F821
        for tid in ids[:-1]:
            dec.step(tid)  # noqa: F821

        out = []
        tok = ids[-1]
        eos = dec.tokenizer.eos_token_id  # noqa: F821
        for _ in range(tokens):
            tok = dec.step(tok)  # noqa: F821
            if tok == eos:
                break
            out.append(tok)
        gen_ids_out.clear()
        gen_ids_out.extend(out)

    avg_s = _timed(run, warmup, runs)

    tokenizer = dec.tokenizer
    del dec
    gc.collect()
    torch.cuda.empty_cache()

    return tokens / avg_s, avg_s * 1000 / tokens, gen_ids_out, tokenizer


# ── vLLM ──────────────────────────────────────────────────────────────────────


def bench_vllm(prompt: str, tokens: int, warmup: int, runs: int) -> tuple | None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not installed — skipping.")
        return None

    print("  Loading vLLM engine...")
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096)
    params = SamplingParams(temperature=0.0, max_tokens=tokens)

    def run():
        llm.generate([prompt], params)  # noqa: F821

    avg_s = _timed(run, warmup, runs)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return tokens / avg_s, avg_s * 1000 / tokens


# ── SGLang ────────────────────────────────────────────────────────────────────


def bench_sglang(prompt: str, tokens: int, warmup: int, runs: int) -> tuple | None:
    try:
        import sglang as sgl
        from sglang.srt.sampling.sampling_params import SamplingParams as SGLParams
    except ImportError:
        print("  SGLang not installed — skipping.")
        return None

    print("  Loading SGLang engine...")
    engine = sgl.Engine(model_path=MODEL, dtype="bfloat16", tp_size=1)
    params = SGLParams(max_new_tokens=tokens, temperature=0.0)

    def run():
        engine.generate(prompts=[prompt], sampling_params=params)

    avg_s = _timed(run, warmup, runs)

    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    return tokens / avg_s, avg_s * 1000 / tokens


# ── correctness check ─────────────────────────────────────────────────────────


def correctness_check(prompt: str, n: int = CORRECTNESS_TOKENS):
    """Compare first n generated tokens between HF and the megakernel."""
    print(f"\nCorrectness check ({n} tokens, prompt={prompt!r})")
    print("-" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=n,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    hf_ids = out[0, -n:].tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    from .decoder import Decoder

    dec = Decoder(verbose=False)
    dec.reset()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    for tid in prompt_ids[:-1]:
        dec.step(tid)

    mk_ids = []
    tok = prompt_ids[-1]
    for _ in range(n):
        tok = dec.step(tok)
        mk_ids.append(tok)
    del dec
    gc.collect()
    torch.cuda.empty_cache()

    match = hf_ids == mk_ids
    print(f"  HF ids : {hf_ids}")
    print(f"  MK ids : {mk_ids}")
    print(f"  HF text: {tokenizer.decode(hf_ids, skip_special_tokens=True)!r}")
    print(f"  MK text: {tokenizer.decode(mk_ids, skip_special_tokens=True)!r}")
    print(f"  Match  : {'YES' if match else 'NO (numerical divergence expected for long runs)'}")
    return match


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B batch-size-1 benchmark")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument(
        "--backends",
        type=str,
        default="megakernel,hf,vllm,sglang",
        help="Comma-separated list of backends to run",
    )
    args = parser.parse_args()

    backends = {b.strip().lower() for b in args.backends.split(",")}

    print("=" * 60)
    print(f"Qwen3-1.7B  batch=1  greedy  {args.tokens} tokens")
    print(f"Prompt: {args.prompt!r}")
    print("=" * 60)

    if not args.no_correctness and "megakernel" in backends:
        correctness_check(args.prompt)

    results = {}

    if "megakernel" in backends:
        print("\n[megakernel]")
        tps, mpt, _, _ = bench_megakernel(args.prompt, args.tokens, args.warmup, args.runs)
        results["megakernel"] = (tps, mpt)

    if "hf" in backends:
        print("\n[HuggingFace PyTorch]")
        tps, mpt, _ = bench_hf(args.prompt, args.tokens, args.warmup, args.runs)
        results["HuggingFace"] = (tps, mpt)

    if "vllm" in backends:
        print("\n[vLLM]")
        r = bench_vllm(args.prompt, args.tokens, args.warmup, args.runs)
        if r:
            results["vLLM"] = r

    if "sglang" in backends:
        print("\n[SGLang]")
        r = bench_sglang(args.prompt, args.tokens, args.warmup, args.runs)
        if r:
            results["SGLang"] = r

    # ── summary table ─────────────────────────────────────────────────────────
    if not results:
        print("\nNo backends ran — check --backends flag.")
        return

    print("\n" + "=" * 60)
    print(f"{'Backend':<20} {'tok/s':>10} {'ms/tok':>10}", end="")
    if "megakernel" in results and len(results) > 1:
        print(f"  {'speedup vs MK':>14}", end="")
    print()
    print("-" * 60)

    mk_tps = results.get("megakernel", (None,))[0]
    for name, (tps, mpt) in results.items():
        line = f"  {name:<18} {tps:>10.1f} {mpt:>10.2f}"
        if mk_tps is not None and name != "megakernel":
            line += f"  {'x':>4}{tps / mk_tps:>9.2f}"
        print(line)

    print("=" * 60)


if __name__ == "__main__":
    main()
