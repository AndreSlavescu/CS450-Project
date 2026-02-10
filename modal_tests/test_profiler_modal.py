"""
Test the GPU profiler on Modal.

Compiles and runs the CUDA profiler test binary, then saves the
resulting Perfetto trace JSON locally.

Usage:
    # Run profiler and save trace (default mode)
    modal run modal_tests/test_profiler_modal.py --gpu h100
    modal run modal_tests/test_profiler_modal.py --gpu b200

    # Run profiler twice and check timing reproducibility
    modal run modal_tests/test_profiler_modal.py --gpu h100 --mode selfcheck
    modal run modal_tests/test_profiler_modal.py --gpu b200 --mode selfcheck

    # Compare new profiler run against a previously saved trace
    modal run modal_tests/test_profiler_modal.py --gpu h100 --mode compare
    modal run modal_tests/test_profiler_modal.py --gpu b200 --mode compare

    # Force rebuild the image:
    FORCE_REBUILD=1 modal run modal_tests/test_profiler_modal.py --gpu h100
    FORCE_REBUILD=1 modal run modal_tests/test_profiler_modal.py --gpu b200
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

app = modal.App("cs450-profiler-test")

GPU_CONFIGS = {
    "h100": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.h100",
        "modal_gpu": "H100",
        "arch": "sm_90a",
    },
    "b200": {
        "dockerfile": PROJECT_ROOT / "Dockerfile.b200",
        "modal_gpu": "B200",
        "arch": "sm_100a",
    },
}

_placeholder = modal.Image.debian_slim()

if _target_gpu == "h100":
    h100_image = modal.Image.from_dockerfile(
        GPU_CONFIGS["h100"]["dockerfile"], force_build=force_rebuild
    ).add_local_dir(str(PROJECT_ROOT / "src" / "csrc"), "/workspace/src/csrc")
    b200_image = _placeholder
else:
    h100_image = _placeholder
    b200_image = modal.Image.from_dockerfile(
        GPU_CONFIGS["b200"]["dockerfile"], force_build=force_rebuild
    ).add_local_dir(str(PROJECT_ROOT / "src" / "csrc"), "/workspace/src/csrc")


def _run_profiler(arch: str) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/workspace")

    compile_cmd = [
        "nvcc",
        "-std=c++20",
        "-O2",
        f"-arch={arch}",
        "-I/workspace/src/csrc/profiler",
        "/workspace/src/csrc/profiler/test_profiler.cu",
        "-o",
        "/workspace/test_profiler",
    ]
    print(f"Compiling: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "success": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "trace_json": None,
        }

    print("Compilation succeeded.\n")

    run_result = subprocess.run(
        ["/workspace/test_profiler"],
        capture_output=True,
        text=True,
        cwd="/workspace",
    )

    print(run_result.stdout)
    if run_result.stderr:
        print(f"stderr: {run_result.stderr}")

    trace_json = None
    trace_path = "/workspace/trace.json"
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            trace_json = json.load(f)
        print(f"Trace JSON: {len(trace_json.get('traceEvents', []))} events")

    return {
        "success": run_result.returncode == 0,
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
        "trace_json": trace_json,
    }


@app.function(image=h100_image, gpu="H100", timeout=600)
def run_profiler_h100() -> dict:
    return _run_profiler("sm_90a")


@app.function(image=b200_image, gpu="B200", timeout=600)
def run_profiler_b200() -> dict:
    return _run_profiler("sm_100a")


def _dispatch_profiler(gpu: str) -> dict:
    if gpu == "h100":
        return run_profiler_h100.remote()
    else:
        return run_profiler_b200.remote()


def _save_trace(result: dict, out_path: Path) -> bool:
    import json

    if not result["trace_json"]:
        print("\nNo trace JSON produced.")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result["trace_json"], f, indent=2)
    n_events = len(result["trace_json"].get("traceEvents", []))
    print(f"\nSaved {n_events}-event Perfetto trace to {out_path.resolve()}")
    print(f"View: python src/csrc/profiler/view_trace.py {out_path}")
    return True


def _extract_event_durations(trace_json: dict) -> dict:
    """Extract per-event-name durations from a Perfetto trace."""
    events = trace_json.get("traceEvents", [])
    durations = {}
    for evt in events:
        name = evt.get("name")
        dur = evt.get("dur")
        if name and dur is not None:
            durations.setdefault(name, []).append(dur)
    return durations


def _compare_traces(ref_trace: dict, new_trace: dict, label: str) -> bool:
    """Compare two traces, print summary. Returns True if they look consistent."""
    ref_durations = _extract_event_durations(ref_trace)
    new_durations = _extract_event_durations(new_trace)

    ref_events = set(ref_durations.keys())
    new_events = set(new_durations.keys())

    print(f"\n{'='*60}")
    print(f"TRACE COMPARISON: {label}")
    print(f"{'='*60}")
    print(f"  Reference events: {len(ref_events)}")
    print(f"  New events:       {len(new_events)}")

    missing = ref_events - new_events
    extra = new_events - ref_events
    if missing:
        print(f"  MISSING events: {missing}")
    if extra:
        print(f"  NEW events: {extra}")

    common = ref_events & new_events
    passed = True
    if not common:
        print("  No common events to compare.")
        return len(missing) == 0

    print(f"\n  {'Event':<30} {'Ref (us)':>12} {'New (us)':>12} {'Delta':>10}")
    print(f"  {'-'*64}")

    for name in sorted(common):
        ref_mean = sum(ref_durations[name]) / len(ref_durations[name])
        new_mean = sum(new_durations[name]) / len(new_durations[name])
        if ref_mean > 0:
            delta_pct = (new_mean - ref_mean) / ref_mean * 100
        else:
            delta_pct = 0.0
        flag = " (!)" if abs(delta_pct) > 50 else ""
        print(f"  {name:<30} {ref_mean:>12.1f} {new_mean:>12.1f} {delta_pct:>+9.1f}%{flag}")
        if abs(delta_pct) > 100:
            passed = False

    print(f"\n  Result: {'PASS' if passed else 'FAIL (>100% timing drift detected)'}")
    return passed


@app.local_entrypoint()
def main(gpu: str = "h100", mode: str = "profile"):
    import json

    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    traces_dir = PROJECT_ROOT / "src" / "csrc" / "profiler" / "traces"
    trace_path = traces_dir / f"trace_{gpu}.json"

    if mode == "profile":
        print(f"Launching profiler test on {gpu.upper()}...\n")
        result = _dispatch_profiler(gpu)

        if not result["success"]:
            print("FAILED!")
            print(result["stderr"])
            return

        _save_trace(result, trace_path)

    elif mode == "selfcheck":
        print(f"Running profiler self-check on {gpu.upper()} (two runs)...\n")

        print("--- Run 1 ---")
        result1 = _dispatch_profiler(gpu)
        if not result1["success"]:
            print("Run 1 FAILED!")
            print(result1["stderr"])
            return

        print("\n--- Run 2 ---")
        result2 = _dispatch_profiler(gpu)
        if not result2["success"]:
            print("Run 2 FAILED!")
            print(result2["stderr"])
            return

        _save_trace(result2, trace_path)

        if result1["trace_json"] and result2["trace_json"]:
            _compare_traces(result1["trace_json"], result2["trace_json"], "Self-check (run1 vs run2)")
        else:
            print("Cannot compare — one or both runs produced no trace.")

    elif mode == "compare":
        if not trace_path.exists():
            print(f"No saved trace found at {trace_path}")
            print("Run with --mode profile first to generate a reference trace.")
            return

        with open(trace_path) as f:
            ref_trace = json.load(f)
        n_ref = len(ref_trace.get("traceEvents", []))
        print(f"Loaded reference trace ({n_ref} events) from {trace_path}")

        print(f"\nRunning new profiler test on {gpu.upper()}...")
        result = _dispatch_profiler(gpu)
        if not result["success"]:
            print("FAILED!")
            print(result["stderr"])
            return

        if result["trace_json"]:
            _compare_traces(ref_trace, result["trace_json"], f"Regression check ({gpu.upper()})")
        else:
            print("New run produced no trace — cannot compare.")

    else:
        print(f"Unknown mode '{mode}'. Choose from: profile, selfcheck, compare")
