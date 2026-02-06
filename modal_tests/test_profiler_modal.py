"""
Test the GPU profiler on a Modal H100.

Usage:
    modal run modal_tests/test_profiler_modal.py

Produces:
    - stdout: profiler text report
    - src/csrc/profiler/traces/trace.json: Perfetto-compatible trace (saved locally)
"""

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("cs450-profiler-test")

# Extend the existing H100 image with our csrc directory.
profiler_image = (
    modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile.h100")
    .add_local_dir(str(PROJECT_ROOT / "src" / "csrc"), "/workspace/src/csrc")
)


@app.function(image=profiler_image, gpu="H100", timeout=600)
def run_profiler_test() -> dict:
    import json
    import os
    import subprocess

    os.chdir("/workspace")

    # Compile
    compile_cmd = [
        "nvcc",
        "-std=c++20",
        "-O2",
        "-arch=sm_90a",
        "-I/workspace/src/csrc/profiler",
        "/workspace/src/csrc/profiler/test_profiler.cu",
        "-o", "/workspace/test_profiler",
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

    # Run
    run_result = subprocess.run(
        ["/workspace/test_profiler"],
        capture_output=True,
        text=True,
        cwd="/workspace",
    )

    print(run_result.stdout)
    if run_result.stderr:
        print(f"stderr: {run_result.stderr}")

    # Read trace JSON
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


@app.local_entrypoint()
def main():
    import json

    print("Launching profiler test on H100...\n")
    result = run_profiler_test.remote()

    if not result["success"]:
        print("FAILED!")
        print(result["stderr"])
        return

    # Save trace locally into src/csrc/profiler/traces/
    if result["trace_json"]:
        traces_dir = PROJECT_ROOT / "src" / "csrc" / "profiler" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        out_path = traces_dir / "trace.json"
        with open(out_path, "w") as f:
            json.dump(result["trace_json"], f, indent=2)
        n_events = len(result["trace_json"].get("traceEvents", []))
        print(f"\nSaved {n_events}-event Perfetto trace to {out_path.resolve()}")
        print(f"View: python src/csrc/profiler/view_trace.py {out_path}")
    else:
        print("\nNo trace JSON produced.")
