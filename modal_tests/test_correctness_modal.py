"""
Correctness to compare megakernel logits against HuggingFace Qwen3-1.7B reference.

Generates reference logits (greedy, temp=0) from the huggingface model and compares them
against the megakernel implementation.

NOTE:

Since the megakernel is not developed yet, this script
generates and saves the reference data only.

Usage:
    # Generate reference logits on H100
    modal run modal_tests/test_correctness_modal.py --gpu h100

    # Generate reference logits on B200
    modal run modal_tests/test_correctness_modal.py --gpu b200

    # Force rebuild the image and run on H100
    FORCE_REBUILD=1 modal run modal_tests/test_correctness_modal.py --gpu h100

    # Force rebuild the image and run on B200
    FORCE_REBUILD=1 modal run modal_tests/test_correctness_modal.py --gpu b200
"""

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

_target_gpu = "h100"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _target_gpu = sys.argv[i + 1].lower()
        break

app = modal.App("cs450-correctness-test")

GPU_CONFIGS = {
    "h100": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.h100",
        "modal_gpu": "H100",
    },
    "b200": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.b200",
        "modal_gpu": "B200",
    },
}

_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    test_image = (
        modal.Image.from_dockerfile(GPU_CONFIGS["h100"]["dockerfile"], force_build=force_rebuild)
        .pip_install("transformers>=4.51.0", "accelerate", "sentencepiece")
        .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
    )
else:
    test_image = (
        modal.Image.from_dockerfile(GPU_CONFIGS["b200"]["dockerfile"], force_build=force_rebuild)
        .pip_install("transformers>=4.51.0", "accelerate", "sentencepiece")
        .add_local_dir(str(PROJECT_ROOT / "src"), "/workspace/src")
    )

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_hidden_layers": 28,
    "head_dim": 128,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "max_position_embeddings": 40960,
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
}

TEST_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def fibonacci(n):",
    "In the year 2025,",
    "The quick brown fox",
]

NUM_GENERATE_TOKENS = 32


def _generate_reference(prompts: list[str], num_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded. Device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    results = {}

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            prefill_out = model(input_ids=input_ids)
            prefill_logits = prefill_out.logits  # [1, seq_len, vocab_size]

        prefill_last_logits = prefill_logits[0, -1, :].float().cpu()

        decode_logits_list = []
        decode_tokens = []
        generated_ids = input_ids.clone()

        for step in range(num_tokens):
            with torch.no_grad():
                out = model(input_ids=generated_ids)
                step_logits = out.logits[0, -1, :].float()  # [vocab_size]

            # For greedy strategy, we pick argmax
            next_token = step_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            decode_logits_list.append(step_logits.cpu())
            decode_tokens.append(next_token.item())

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"  EOS at step {step}")
                break

        generated_text = tokenizer.decode(decode_tokens, skip_special_tokens=True)
        print(f"  Generated: {generated_text!r}")
        print(f"  Tokens: {decode_tokens[:10]}...")
        print(
            f"  Prefill last logit stats: min={prefill_last_logits.min():.4f}, "
            f"max={prefill_last_logits.max():.4f}, mean={prefill_last_logits.mean():.4f}"
        )

        results[prompt] = {
            "input_ids": input_ids[0].cpu().tolist(),
            "prompt_len": prompt_len,
            "generated_tokens": decode_tokens,
            "generated_text": generated_text,
            # Store top-k logits and their indices for comparison
            "prefill_top100_values": prefill_last_logits.topk(100).values.tolist(),
            "prefill_top100_indices": prefill_last_logits.topk(100).indices.tolist(),
            "prefill_argmax": prefill_last_logits.argmax().item(),
            "prefill_logit_stats": {
                "min": prefill_last_logits.min().item(),
                "max": prefill_last_logits.max().item(),
                "mean": prefill_last_logits.mean().item(),
                "std": prefill_last_logits.std().item(),
            },
            "decode_steps": [],
        }

        for i, (logits_i, token_i) in enumerate(zip(decode_logits_list, decode_tokens)):
            results[prompt]["decode_steps"].append(
                {
                    "step": i,
                    "token": token_i,
                    "token_str": tokenizer.decode([token_i]),
                    "top100_values": logits_i.topk(100).values.tolist(),
                    "top100_indices": logits_i.topk(100).indices.tolist(),
                    "argmax": logits_i.argmax().item(),
                    "logit_stats": {
                        "min": logits_i.min().item(),
                        "max": logits_i.max().item(),
                        "mean": logits_i.mean().item(),
                        "std": logits_i.std().item(),
                    },
                }
            )

    return {
        "model_id": MODEL_ID,
        "model_config": MODEL_CONFIG,
        "num_generate_tokens": num_tokens,
        "results": results,
    }


def _compare_logits(reference: dict, candidate: dict) -> dict:
    report = {"prompts": {}, "aggregate": {}}
    total_argmax_matches = 0
    total_steps = 0
    all_top1_diffs = []
    all_topk_overlaps = []

    for prompt, ref_data in reference["results"].items():
        if prompt not in candidate:
            report["prompts"][prompt] = {"status": "MISSING"}
            continue

        cand_data = candidate[prompt]
        prompt_report = {"prefill": {}, "decode_steps": []}

        ref_argmax = ref_data["prefill_argmax"]
        cand_argmax = cand_data["prefill_argmax"]
        prompt_report["prefill"]["argmax_match"] = ref_argmax == cand_argmax
        prompt_report["prefill"]["ref_argmax"] = ref_argmax
        prompt_report["prefill"]["cand_argmax"] = cand_argmax

        ref_top100 = set(ref_data["prefill_top100_indices"])
        cand_top100 = set(cand_data.get("prefill_top100_indices", []))
        overlap = len(ref_top100 & cand_top100)
        prompt_report["prefill"]["top100_overlap"] = overlap

        if ref_data["prefill_top100_values"] and cand_data.get("prefill_top100_values"):
            diff = abs(ref_data["prefill_top100_values"][0] - cand_data["prefill_top100_values"][0])
            prompt_report["prefill"]["top1_logit_diff"] = diff
            all_top1_diffs.append(diff)

        all_topk_overlaps.append(overlap)
        if ref_argmax == cand_argmax:
            total_argmax_matches += 1
        total_steps += 1

        ref_steps = ref_data.get("decode_steps", [])
        cand_steps = cand_data.get("decode_steps", [])
        min_steps = min(len(ref_steps), len(cand_steps))

        for i in range(min_steps):
            ref_s = ref_steps[i]
            cand_s = cand_steps[i]
            step_match = ref_s["argmax"] == cand_s["argmax"]
            step_report = {
                "step": i,
                "argmax_match": step_match,
                "ref_token": ref_s["argmax"],
                "cand_token": cand_s["argmax"],
            }

            ref_topk = set(ref_s["top100_indices"])
            cand_topk = set(cand_s.get("top100_indices", []))
            step_overlap = len(ref_topk & cand_topk)
            step_report["top100_overlap"] = step_overlap
            all_topk_overlaps.append(step_overlap)

            if ref_s["top100_values"] and cand_s.get("top100_values"):
                diff = abs(ref_s["top100_values"][0] - cand_s["top100_values"][0])
                step_report["top1_logit_diff"] = diff
                all_top1_diffs.append(diff)

            if step_match:
                total_argmax_matches += 1
            total_steps += 1

            prompt_report["decode_steps"].append(step_report)

        report["prompts"][prompt] = prompt_report

    report["aggregate"] = {
        "total_steps": total_steps,
        "argmax_match_rate": total_argmax_matches / total_steps if total_steps > 0 else 0,
        "mean_top1_logit_diff": sum(all_top1_diffs) / len(all_top1_diffs) if all_top1_diffs else None,
        "max_top1_logit_diff": max(all_top1_diffs) if all_top1_diffs else None,
        "mean_top100_overlap": sum(all_topk_overlaps) / len(all_topk_overlaps) if all_topk_overlaps else None,
    }

    # Allow small tolerance which is expected for bf16.
    # In our case, we allow up to a 2.5% failure rate, which is possible for aggregation
    #   of floating point errors at a large scale.
    argmax_rate = report["aggregate"]["argmax_match_rate"]
    report["aggregate"]["passed"] = argmax_rate >= 0.975

    return report


@app.function(
    image=test_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_reference() -> dict:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    return _generate_reference(TEST_PROMPTS, NUM_GENERATE_TOKENS)


@app.function(
    image=test_image,
    gpu=GPU_CONFIGS[_target_gpu]["modal_gpu"],
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_correctness_test(reference_data: dict) -> dict:
    """
    Run megakernel and compare against reference logits.

    Currently a placeholder that returns the reference data comparison format.
    Replace the candidate generation with actual megakernel calls once implemented.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # TODO: Replace this section with megakernel inference
    # ==========================================================================
    # When the megakernel is ready, it should:
    # 1. Load the same Qwen3-1.7B weights
    # 2. Run each prompt through the megakernel
    # 3. Collect top-100 logits per step in the same format as reference
    # ==========================================================================
    candidate_data = _generate_reference(TEST_PROMPTS, NUM_GENERATE_TOKENS)

    candidate = {}
    for prompt, data in candidate_data["results"].items():
        candidate[prompt] = {
            "prefill_top100_values": data["prefill_top100_values"],
            "prefill_top100_indices": data["prefill_top100_indices"],
            "prefill_argmax": data["prefill_argmax"],
            "decode_steps": data["decode_steps"],
        }

    report = _compare_logits(reference_data, candidate)
    return report


@app.local_entrypoint()
def main(gpu: str = "h100", mode: str = "reference"):
    import json

    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    ref_dir = PROJECT_ROOT / "modal_tests" / "reference_data"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / f"qwen3_1.7b_reference_{gpu}.json"

    if mode == "reference":
        print(f"Generating reference logits on {gpu.upper()}...")
        reference = generate_reference.remote()

        with open(ref_path, "w") as f:
            json.dump(reference, f, indent=2)

        n_prompts = len(reference["results"])
        print(f"\nSaved reference data for {n_prompts} prompts to {ref_path}")

        for prompt, data in reference["results"].items():
            print(f"\n  Prompt: {prompt!r}")
            print(f"    Generated: {data['generated_text']!r}")
            print(f"    Tokens: {data['generated_tokens'][:8]}...")
            stats = data["prefill_logit_stats"]
            print(
                f"    Prefill logit stats: min={stats['min']:.2f} max={stats['max']:.2f} "
                f"mean={stats['mean']:.2f} std={stats['std']:.2f}"
            )

    elif mode == "selfcheck":
        print(f"Running self-check on {gpu.upper()} (reference vs reference)...")
        reference = generate_reference.remote()

        with open(ref_path, "w") as f:
            json.dump(reference, f, indent=2)
        print(f"Saved reference to {ref_path}")

        report = run_correctness_test.remote(reference)

        agg = report["aggregate"]
        print(f"\n{'=' * 60}")
        print("SELF-CHECK RESULTS")
        print(f"{'=' * 60}")
        print(f"  Argmax match rate: {agg['argmax_match_rate'] * 100:.1f}%")
        print(f"  Mean top-1 logit diff: {agg['mean_top1_logit_diff']:.6f}")
        print(f"  Max  top-1 logit diff: {agg['max_top1_logit_diff']:.6f}")
        print(f"  Mean top-100 overlap:  {agg['mean_top100_overlap']:.1f}/100")
        print(f"  PASSED: {agg['passed']}")

        report_path = ref_dir / f"selfcheck_report_{gpu}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {report_path}")

    elif mode == "compare":
        if not ref_path.exists():
            print(f"No reference data found at {ref_path}")
            print("Run with --mode reference first.")
            return

        with open(ref_path) as f:
            reference = json.load(f)

        print(f"Loaded reference data from {ref_path}")
        print(f"Running megakernel comparison on {gpu.upper()}...")

        report = run_correctness_test.remote(reference)

        agg = report["aggregate"]
        passed = agg["passed"]
        status = "PASSED" if passed else "FAILED"

        print(f"\n{'=' * 60}")
        print(f"CORRECTNESS TEST: {status}")
        print(f"{'=' * 60}")
        print(f"  Argmax match rate: {agg['argmax_match_rate'] * 100:.1f}%")
        if agg["mean_top1_logit_diff"] is not None:
            print(f"  Mean top-1 logit diff: {agg['mean_top1_logit_diff']:.6f}")
            print(f"  Max  top-1 logit diff: {agg['max_top1_logit_diff']:.6f}")
        if agg["mean_top100_overlap"] is not None:
            print(f"  Mean top-100 overlap:  {agg['mean_top100_overlap']:.1f}/100")

        for prompt, pdata in report["prompts"].items():
            if pdata.get("status") == "MISSING":
                print(f"\n  [{prompt!r}] MISSING from candidate")
                continue
            pfx = pdata["prefill"]
            match_str = "MATCH" if pfx["argmax_match"] else "MISMATCH"
            print(f"\n  [{prompt!r}]")
            print(f"    Prefill argmax: {match_str} (ref={pfx['ref_argmax']}, cand={pfx['cand_argmax']})")
            print(f"    Prefill top-100 overlap: {pfx['top100_overlap']}/100")
            decode_matches = sum(1 for s in pdata["decode_steps"] if s["argmax_match"])
            total = len(pdata["decode_steps"])
            print(f"    Decode argmax matches: {decode_matches}/{total}")

        report_path = ref_dir / f"comparison_report_{gpu}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {report_path}")

    else:
        print(f"Unknown mode '{mode}'. Choose from: reference, selfcheck, compare")
