"""
Test the GPU profiler on Modal.

Usage:
    # runs --mode reference by default. This generates reference logits
    modal run modal_tests/test_profiler_modal.py --gpu h100
    modal run modal_tests/test_profiler_modal.py --gpu b200

    # runs --mode selfcheck which loads the megakernel implementation and the HF implementation
    #   and then compares the results
    modal run modal_tests/test_profiler_modal.py --gpu h100 --mode selfcheck
    modal run modal_tests/test_profiler_modal.py --gpu b200 --mode selfcheck

    # runs --mode compare which loads saved reference logits and compares against the megakernel implementation
    modal run modal_tests/test_profiler_modal.py --gpu h100 --mode compare
    modal run modal_tests/test_profiler_modal.py --gpu b200 --mode compare

    # if you want to force rebuild the image:
    FORCE_REBUILD=1 modal run modal_tests/test_profiler_modal.py --gpu h100
    FORCE_REBUILD=1 modal run modal_tests/test_profiler_modal.py --gpu b200
"""

import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

# Parse --gpu from sys.argv at module level to only build the needed image.
# Modal builds all images at import time, so we avoid building the unused one.
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


@app.local_entrypoint()
def main(gpu: str = "h100"):
    import json

    gpu = gpu.lower()
    if gpu not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_CONFIGS)}")
        return

    print(f"Launching profiler test on {gpu.upper()}...\n")

    if gpu == "h100":
        result = run_profiler_h100.remote()
    else:
        result = run_profiler_b200.remote()

    if not result["success"]:
        print("FAILED!")
        print(result["stderr"])
        return

    if result["trace_json"]:
        traces_dir = PROJECT_ROOT / "src" / "csrc" / "profiler" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        out_path = traces_dir / f"trace_{gpu}.json"
        with open(out_path, "w") as f:
            json.dump(result["trace_json"], f, indent=2)
        n_events = len(result["trace_json"].get("traceEvents", []))
        print(f"\nSaved {n_events}-event Perfetto trace to {out_path.resolve()}")
        print(f"View: python src/csrc/profiler/view_trace.py {out_path}")
    else:
        print("\nNo trace JSON produced.")
