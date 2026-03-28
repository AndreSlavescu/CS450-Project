"""Qwen3-Coder-480B-A35B-Instruct MoE inference on Modal (8×B200).

Runs TP=8 (attention) + EP=8 (experts) sharded inference across 8 GPUs.
The model is ~960GB in bf16; each GPU holds ~127GB of weights.

Usage:
    modal run modal_tests/test_moe_modal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent
force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

app = modal.App("cs450-moe-qwen3-480b")

# Persistent volume so ~960GB of weights are downloaded once and reused
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

moe_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.b200", force_build=force_rebuild)
    .pip_install("transformers>=4.51.0,<5.0", "accelerate", "sentencepiece", "huggingface_hub")
    .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
)

MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
WARMUP = 2
RUNS = 5
PROMPT = "Write a Python function that computes the Fibonacci sequence"
CORRECTNESS_TOKENS = 8
GEN_TOKENS = 16
MAX_SEQ_LEN = 4096  # limit RoPE precomputation and KV cache


def _worker(rank: int, world_size: int, results_dict: dict):
    import time

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # NCCL tuning for NVLink
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

    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"Qwen3-Coder-480B-A35B-Instruct  TP={world_size}  EP={world_size}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # NCCL latency microbenchmark
    if rank == 0:
        print("NCCL allreduce latency:", flush=True)
    dist.barrier()
    for msg_elems, label in [(6144, "12KB"), (6144 * 16, "192KB")]:
        buf = torch.randn(msg_elems, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(50):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if rank == 0:
            print(f"  {label:>6}: {(t1 - t0) / 50 * 1e6:.0f} us/call", flush=True)
    dist.barrier()

    # TP and EP use the same group (all 8 ranks)
    tp_group = dist.new_group(list(range(world_size)))
    ep_group = dist.new_group(list(range(world_size)))

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    # Build tcgen05 MoE kernel on rank 0, barrier for others
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

        print("Building tcgen05 MoE kernel...", flush=True)
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
        if result.returncode == 0:
            print("  MoE kernel built successfully", flush=True)
        else:
            raise RuntimeError(f"MoE kernel build failed:\n{result.stderr[:1000]}")
    dist.barrier()

    # Add kernel dir to Python path so moe_expert can be imported
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
        print(f"Loading {MODEL_NAME} (EP={world_size}, TP={world_size})...", flush=True)
    dist.barrier()

    t_load_start = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_NAME,
        extra_config=extra,
        device=f"cuda:{rank}",
        dtype=torch.bfloat16,
    )
    dist.barrier()
    t_load = time.perf_counter() - t_load_start

    if rank == 0:
        mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9
        print(f"Model loaded in {t_load:.1f}s  (peak GPU mem: {mem_gb:.1f} GB)", flush=True)

    # Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── helper: greedy generate ──
    def generate(prompt_text: str, max_tokens: int) -> tuple[list[int], float]:
        ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=f"cuda:{rank}")

        # Prefill
        pos = torch.arange(len(ids), device=f"cuda:{rank}").unsqueeze(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(input_ids, pos, len(ids))
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        next_tok = logits[0, -1].argmax().item()
        gen_ids = [next_tok]

        seq_len = len(ids) + 1
        eos = tokenizer.eos_token_id
        for _ in range(max_tokens - 1):
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=f"cuda:{rank}")
            pos_tensor = torch.tensor([[seq_len - 1]], dtype=torch.long, device=f"cuda:{rank}")
            logits = model(tok_tensor, pos_tensor, seq_len)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == eos:
                break
            gen_ids.append(next_tok)
            seq_len += 1

        return gen_ids, ttft_ms

    # ── correctness check ──
    if rank == 0:
        print(f"\nCorrectness check ({CORRECTNESS_TOKENS} tokens)...", flush=True)

    gen_ids, ttft = generate(PROMPT, CORRECTNESS_TOKENS)

    if rank == 0:
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"  Prompt: {PROMPT!r}")
        print(f"  Output: {text!r}")
        print(f"  Token IDs: {gen_ids}")
        print(f"  TTFT: {ttft:.1f} ms")
    dist.barrier()

    # ── verify all ranks produce identical output ──
    gen_tensor = torch.tensor(gen_ids[:CORRECTNESS_TOKENS], dtype=torch.long, device=f"cuda:{rank}")
    gathered = [torch.zeros_like(gen_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, gen_tensor)
    if rank == 0:
        all_match = all(torch.equal(gathered[0], gathered[i]) for i in range(1, world_size))
        print(f"  All ranks agree: {all_match}")
        if not all_match:
            for i in range(world_size):
                print(f"    rank {i}: {gathered[i].tolist()}")
    dist.barrier()

    # ── TTFT benchmark ──
    if rank == 0:
        print(f"\n{'─'*60}")
        print("TTFT benchmark (prefill)")
        print(f"{'─'*60}")

    ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=f"cuda:{rank}")
    pos = torch.arange(len(ids), device=f"cuda:{rank}").unsqueeze(0)

    # Reset KV cache between runs
    for _ in range(WARMUP):
        model.stacked_kv_cache[0].zero_()
        model.stacked_kv_cache[1].zero_()
        model(input_ids, pos, len(ids))
    torch.cuda.synchronize()
    dist.barrier()

    ttft_times = []
    for _ in range(RUNS):
        model.stacked_kv_cache[0].zero_()
        model.stacked_kv_cache[1].zero_()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids, pos, len(ids))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ttft_times.append((t1 - t0) * 1000.0)

    ttft_times.sort()
    median_ttft = ttft_times[len(ttft_times) // 2]
    tps = len(ids) / (median_ttft / 1000.0)

    if rank == 0:
        print(f"  Prompt length: {len(ids)} tokens")
        print(f"  Median TTFT: {median_ttft:.2f} ms")
        print(f"  Prefill throughput: {tps:.1f} tok/s")
        print(f"  All runs (ms): {[f'{t:.1f}' for t in ttft_times]}")

    # ── decode throughput ──
    if rank == 0:
        print(f"\n{'─'*60}")
        print(f"Decode throughput ({GEN_TOKENS} tokens)")
        print(f"{'─'*60}")

    # Prefill once, then measure decode
    model.stacked_kv_cache[0].zero_()
    model.stacked_kv_cache[1].zero_()
    logits = model(input_ids, pos, len(ids))
    first_tok = logits[0, -1].argmax().item()

    # Warmup decode
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
        t1 = time.perf_counter()

        decode_times.append((t1 - t0) * 1000.0)
        model.stacked_kv_cache[0].copy_(kv_k_saved)
        model.stacked_kv_cache[1].copy_(kv_v_saved)

    decode_times.sort()
    median_decode = decode_times[len(decode_times) // 2]
    ms_per_tok = median_decode / GEN_TOKENS
    decode_tps = GEN_TOKENS / (median_decode / 1000.0)

    if rank == 0:
        print(f"  Median total: {median_decode:.2f} ms")
        print(f"  Per token: {ms_per_tok:.2f} ms/tok")
        print(f"  Throughput: {decode_tps:.1f} tok/s")

        mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9
        print(f"\n{'='*60}")
        print("Summary: Qwen3-Coder-480B  8×B200  TP=8 EP=8")
        print(f"  Model load: {t_load:.1f}s")
        print(f"  Peak GPU memory: {mem_gb:.1f} GB / 192 GB")
        print(f"  Prefill ({len(ids)} tok): {median_ttft:.1f} ms ({tps:.0f} tok/s)")
        print(f"  Decode: {ms_per_tok:.1f} ms/tok ({decode_tps:.0f} tok/s)")
        print(f"{'='*60}")

    if rank == 0:
        results_dict["ttft_ms"] = median_ttft
        results_dict["prefill_tps"] = tps
        results_dict["decode_ms_per_tok"] = ms_per_tok
        results_dict["decode_tps"] = decode_tps
        results_dict["load_time_s"] = t_load
        results_dict["peak_mem_gb"] = mem_gb

    dist.barrier()
    dist.destroy_process_group()


@app.function(
    image=moe_image,
    gpu="B200:8",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
)
def run_moe_benchmark():
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(8, results), nprocs=8, join=True)
    return dict(results)


@app.local_entrypoint()
def main():
    result = run_moe_benchmark.remote()
    print("\nBenchmark complete.")
    if result:
        print(f"  TTFT: {result.get('ttft_ms', 'N/A')} ms")
        print(f"  Decode: {result.get('decode_tps', 'N/A')} tok/s")
