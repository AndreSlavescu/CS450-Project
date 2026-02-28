"""Distributed Qwen3-1.7B TTFT benchmark on Modal (B200 multi-GPU).

Runs TP-sharded inference across GPUs with multimem allreduce between layers.
Compares single-GPU megakernel TTFT vs distributed TTFT.

Usage:
    modal run modal_tests/test_distributed_modal.py
    modal run modal_tests/test_distributed_modal.py --tp 2
    modal run modal_tests/test_distributed_modal.py --tp 4
    modal run modal_tests/test_distributed_modal.py --tp 8
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent
force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

app = modal.App("cs450-distributed-qwen3")

dist_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.b200", force_build=force_rebuild)
    .pip_install("transformers>=4.51.0,<5.0", "accelerate", "sentencepiece")
    .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
)

WARMUP = 3
RUNS = 10
PROMPT = "The quick brown fox jumps over the lazy dog"
TTFT_PROMPT_LENS = [1, 16, 64, 128, 256, 512]
GEN_TOKENS = 32
CORRECTNESS_TOKENS = 8


def _worker(rank: int, world_size: int, results_dict: dict):
    import time

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    from src.python.Qwen3.distributed_decoder import DistributedDecoder, load_weights_for_rank

    weights = tokenizer = None
    for r in range(world_size):
        if rank == r:
            weights, tokenizer = load_weights_for_rank(rank, verbose=(rank == 0))
        dist.barrier()

    dec = DistributedDecoder(
        tp_size=world_size,
        tp_rank=rank,
        weights=weights,
        tokenizer=tokenizer,
        verbose=(rank == 0),
    )

    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"Distributed Qwen3-1.7B  TP={world_size}  batch=1  greedy", flush=True)
        print(f"{'='*60}", flush=True)

    # ── correctness check ──
    if rank == 0:
        print(f"\nCorrectness check ({CORRECTNESS_TOKENS} tokens)...", flush=True)

    dec.reset()
    ids = dec.tokenizer.encode(PROMPT, add_special_tokens=True)
    if len(ids) > 1:
        dec.prefill(ids[:-1])

    gen_ids = []
    tok = ids[-1]
    for _ in range(CORRECTNESS_TOKENS):
        tok = dec.step(tok)
        gen_ids.append(tok)

    if rank == 0:
        text = dec.tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"  Generated ids: {gen_ids}")
        print(f"  Generated text: {text!r}")

    dist.barrier()

    # ── TTFT benchmark: batched prefill, measuring time to first token ──
    if rank == 0:
        print(f"\n{'─'*60}")
        print(f"{'Prompt len':>12} {'TTFT (ms)':>12} {'Prefill tok/s':>14}")
        print(f"{'─'*60}")

    base_text = "The quick brown fox jumps over the lazy dog. "
    ttft_results = {}

    for plen in TTFT_PROMPT_LENS:
        prompt_text = (base_text * ((plen // 10) + 1))[:plen * 5]
        prompt_ids = dec.tokenizer.encode(prompt_text, add_special_tokens=True)[:plen]
        if len(prompt_ids) < plen:
            prompt_ids = prompt_ids + [prompt_ids[-1]] * (plen - len(prompt_ids))
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")

        for _ in range(WARMUP):
            dec.reset()
            dec.prefill(prompt_tensor)
        torch.cuda.synchronize()
        dist.barrier()

        times_ms = []
        for _ in range(RUNS):
            dec.reset()
            dist.barrier()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            dec.prefill(prompt_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0)

        times_ms.sort()
        trim = max(1, len(times_ms) // 5)
        trimmed = times_ms[trim:-trim] if len(times_ms) > 2 * trim else times_ms
        mean_ms = sum(trimmed) / len(trimmed)
        tps = plen / (mean_ms / 1000.0)

        ttft_results[plen] = {"mean_ms": mean_ms, "tok_per_s": tps, "all_ms": times_ms}

        if rank == 0:
            print(f"{plen:>12} {mean_ms:>12.2f} {tps:>14.1f}")

    # ── decode throughput: tokens/sec after prefill ──
    if rank == 0:
        print(f"\n{'─'*60}")
        print(f"Decode throughput ({GEN_TOKENS} tokens, prefill={len(ids)-1})")

    dec.reset()
    if len(ids) > 1:
        dec.prefill(ids[:-1])
    tok = ids[-1]

    for _ in range(WARMUP):
        saved_pos = dec._position
        saved_kc = dec._k_cache.clone()
        saved_vc = dec._v_cache.clone()
        for __ in range(GEN_TOKENS):
            tok = dec.step(tok)
        dec._position = saved_pos
        dec._k_cache.copy_(saved_kc)
        dec._v_cache.copy_(saved_vc)
        tok = ids[-1]

    torch.cuda.synchronize()
    dist.barrier()

    decode_times = []
    for _ in range(RUNS):
        saved_pos = dec._position
        saved_kc = dec._k_cache.clone()
        saved_vc = dec._v_cache.clone()
        tok_local = ids[-1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for __ in range(GEN_TOKENS):
            tok_local = dec.step(tok_local)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        decode_times.append((t1 - t0) * 1000.0)
        dec._position = saved_pos
        dec._k_cache.copy_(saved_kc)
        dec._v_cache.copy_(saved_vc)

    decode_times.sort()
    trim = max(1, len(decode_times) // 5)
    trimmed = decode_times[trim:-trim] if len(decode_times) > 2 * trim else decode_times
    mean_decode_ms = sum(trimmed) / len(trimmed)
    ms_per_tok = mean_decode_ms / GEN_TOKENS
    decode_tps = GEN_TOKENS / (mean_decode_ms / 1000.0)

    if rank == 0:
        print(f"  Mean total: {mean_decode_ms:.2f} ms")
        print(f"  Per token:  {ms_per_tok:.2f} ms/tok")
        print(f"  Throughput: {decode_tps:.1f} tok/s")

    # ── single-GPU megakernel baseline (rank 0 only) ──
    mk_results = {}
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Single-GPU megakernel baseline (rank 0)")
        print(f"{'='*60}")

        from src.python.Qwen3.decoder import Decoder as SingleDecoder
        single_dec = SingleDecoder(verbose=False)

        print(f"\n{'Prompt len':>12} {'TTFT (ms)':>12} {'Prefill tok/s':>14}")
        print(f"{'─'*60}")

        for plen in TTFT_PROMPT_LENS:
            prompt_text = (base_text * ((plen // 10) + 1))[:plen * 5]
            prompt_ids = single_dec.tokenizer.encode(prompt_text, add_special_tokens=True)[:plen]
            if len(prompt_ids) < plen:
                prompt_ids = prompt_ids + [prompt_ids[-1]] * (plen - len(prompt_ids))
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")

            # Warmup (also captures CUDA graph on first call per length)
            for _ in range(WARMUP):
                single_dec.reset()
                single_dec.prefill(prompt_tensor)
            torch.cuda.synchronize()

            stimes = []
            for _ in range(RUNS):
                single_dec.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                single_dec.prefill(prompt_tensor)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                stimes.append((t1 - t0) * 1000.0)

            stimes.sort()
            trim = max(1, len(stimes) // 5)
            trimmed_s = stimes[trim:-trim] if len(stimes) > 2 * trim else stimes
            mean_s = sum(trimmed_s) / len(trimmed_s)
            tps_s = plen / (mean_s / 1000.0)
            mk_results[plen] = {"mean_ms": mean_s, "tok_per_s": tps_s}
            print(f"{plen:>12} {mean_s:>12.2f} {tps_s:>14.1f}")

        # Decode throughput for single-GPU
        single_dec.reset()
        s_ids = single_dec.tokenizer.encode(PROMPT, add_special_tokens=True)
        s_tensor = torch.tensor(s_ids[:-1], dtype=torch.long, device="cuda")
        single_dec.prefill(s_tensor)
        stok = s_ids[-1]

        for _ in range(WARMUP):
            sp = single_dec._position
            skc = single_dec._k_cache.clone()
            svc = single_dec._v_cache.clone()
            skc_pf = single_dec._k_cache_pf.clone()
            svc_pf = single_dec._v_cache_pf.clone()
            for __ in range(GEN_TOKENS):
                stok = single_dec.step(stok)
            single_dec._position = sp
            single_dec._k_cache.copy_(skc)
            single_dec._v_cache.copy_(svc)
            single_dec._k_cache_pf.copy_(skc_pf)
            single_dec._v_cache_pf.copy_(svc_pf)
            stok = s_ids[-1]

        torch.cuda.synchronize()
        sdtimes = []
        for _ in range(RUNS):
            sp = single_dec._position
            skc = single_dec._k_cache.clone()
            svc = single_dec._v_cache.clone()
            skc_pf = single_dec._k_cache_pf.clone()
            svc_pf = single_dec._v_cache_pf.clone()
            stok_l = s_ids[-1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for __ in range(GEN_TOKENS):
                stok_l = single_dec.step(stok_l)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            sdtimes.append((t1 - t0) * 1000.0)
            single_dec._position = sp
            single_dec._k_cache.copy_(skc)
            single_dec._v_cache.copy_(svc)
            single_dec._k_cache_pf.copy_(skc_pf)
            single_dec._v_cache_pf.copy_(svc_pf)

        sdtimes.sort()
        trim = max(1, len(sdtimes) // 5)
        trimmed_sd = sdtimes[trim:-trim] if len(sdtimes) > 2 * trim else sdtimes
        mean_sd = sum(trimmed_sd) / len(trimmed_sd)

        del single_dec
        torch.cuda.empty_cache()

        # ── summary ──
        print(f"\n{'='*60}")
        print(f"Summary: TP={world_size} distributed vs single-GPU megakernel")
        print(f"{'='*60}")
        print(f"{'Prompt':>8} {'1-GPU (ms)':>12} {'TP{} (ms)'.format(world_size):>12} {'Speedup':>10}")
        print(f"{'─'*60}")
        for plen in TTFT_PROMPT_LENS:
            if plen in mk_results and plen in ttft_results:
                sg = mk_results[plen]["mean_ms"]
                tp = ttft_results[plen]["mean_ms"]
                speedup = sg / tp if tp > 0 else 0
                print(f"{plen:>8} {sg:>12.2f} {tp:>12.2f} {speedup:>9.2f}x")

        print(f"\nDecode throughput:")
        mk_tps = GEN_TOKENS / (mean_sd / 1000.0)
        print(f"  1-GPU megakernel: {mk_tps:.1f} tok/s ({mean_sd / GEN_TOKENS:.2f} ms/tok)")
        print(f"  TP={world_size} distributed: {decode_tps:.1f} tok/s ({ms_per_tok:.2f} ms/tok)")
        if decode_tps > mk_tps:
            print(f"  Speedup: {decode_tps / mk_tps:.2f}x (distributed faster)")
        elif mk_tps > 0:
            print(f"  Speedup: {mk_tps / decode_tps:.2f}x (single-GPU faster)")

    if rank == 0:
        results_dict["ttft"] = ttft_results
        results_dict["decode_tps"] = decode_tps
        results_dict["decode_ms_per_tok"] = ms_per_tok
        results_dict["mk_results"] = mk_results

    dist.barrier()
    dist.destroy_process_group()


@app.function(image=dist_image, gpu="B200:2", timeout=1200)
def run_distributed_benchmark_2():
    return _run_benchmark(2)


@app.function(image=dist_image, gpu="B200:4", timeout=1200)
def run_distributed_benchmark_4():
    return _run_benchmark(4)


@app.function(image=dist_image, gpu="B200:8", timeout=1200)
def run_distributed_benchmark_8():
    return _run_benchmark(8)


def _run_benchmark(tp_size: int):
    import torch.multiprocessing as mp
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(tp_size, results), nprocs=tp_size, join=True)
    return dict(results)


@app.local_entrypoint()
def main(tp: int = 2):
    fn = {2: run_distributed_benchmark_2,
          4: run_distributed_benchmark_4,
          8: run_distributed_benchmark_8}.get(tp)
    if fn is None:
        print(f"Unsupported --tp {tp}. Use 2, 4, or 8.")
        return
    results = fn.remote()
    print("\nBenchmark complete.")
