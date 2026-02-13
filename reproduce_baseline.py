import subprocess
import time
from pathlib import Path

import modal

app = modal.App("cs450-reproduction")

# libnuma1 required by sgl_kernel (SGLang) native .so on Linux
image = (
    modal.Image.from_dockerfile(Path(__file__).parent / "Dockerfile.h100")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*"
    )
    .pip_install(
        "sglang[all]",
        "pandas",
        "tabulate",
    )
)

PROJECT_ROOT = "/workspace/Megakernels"


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_megakernel_h100():
    """
    Reproduces the Megakernel bar graph as per Figure 1.
    """
    print("=== Benchmarking Megakernel (H100) ===")

    # Using the same parameters as the paper (approx 32 token prompt, 128 gen)
    cmd = (
        f"python {PROJECT_ROOT}/megakernels/scripts/generate.py "
        "mode=mk "
        "prompt='This is a dummy prompt that is roughly thirty two tokens "
        "long to match the paper benchmark settings exactly.' "
        "ntok=128"
    )

    # We change dir to ensure relative paths in the script work
    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, check=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_vllm_baseline():
    """
    Reproduces the VLLM baseline bar graph as per Figure 1.
    """

    import requests

    print("=== Benchmarking vLLM Baseline (H100) ===")

    # 1. Start the vLLM server in the background
    server_cmd = [
        "vllm",
        "serve",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--port",
        "10210",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.9",
    ]

    print(f"Launching server: {' '.join(server_cmd)}")
    # We capture stdout/stderr to debug crashes if they occur
    server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        # 2. Wait for the server to actually start (Health Check)
        print("Waiting for vLLM to load model weights (this takes time)...")
        health_url = "http://localhost:10210/v1/models"
        start_time = time.time()
        server_ready = False

        while time.time() - start_time < 600:  # 10 minute timeout
            # Check if server crashed
            if server_process.poll() is not None:
                print("\nCRITICAL: Server crashed during startup!")
                print("Server Logs:")
                print(server_process.stdout.read())
                raise RuntimeError("vLLM server exited unexpectedly.")

            # Check if server is ready
            try:
                requests.get(health_url)
                print("\nServer is ready! Starting benchmark...")
                server_ready = True
                break
            except requests.exceptions.ConnectionError:
                # Still loading...
                time.sleep(5)
                print(".", end="", flush=True)

        if not server_ready:
            raise RuntimeError("Timed out waiting for vLLM to start.")

        # 3. Run the benchmarking script
        bench_cmd = (
            f"python {PROJECT_ROOT}/megakernels/scripts/bench_engines.py "
            "port=10210 "
            "prompt_len=32 "
            "output_len=128 "
            "model='meta-llama/Llama-3.2-1B-Instruct'"
        )

        # We allow stdout to print to the console so you can see the results
        subprocess.run(bench_cmd, shell=True, cwd=PROJECT_ROOT, check=True)

    finally:
        print("\nTerminating vLLM server...")
        server_process.terminate()
        server_process.wait()


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_sglang_baseline():
    """
    Reproduces the SGLang baseline bar graph as per Figure 1.
    """
    import requests

    print("=== Benchmarking SGLang Baseline (H100) ===")

    # 1. Start the SGLang server in the background
    server_cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--port",
        "10210",
        "--mem-fraction-static",
        "0.9",
    ]

    print(f"Launching server: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        # 2. Wait for the server to start (health check)
        print("Waiting for SGLang to load model weights (this takes time)...")
        health_url = "http://localhost:10210/v1/models"
        start_time = time.time()
        server_ready = False

        while time.time() - start_time < 600:  # 10 minute timeout
            if server_process.poll() is not None:
                print("\nCRITICAL: Server crashed during startup!")
                print("Server Logs:")
                print(server_process.stdout.read())
                raise RuntimeError("SGLang server exited unexpectedly.")

            try:
                requests.get(health_url)
                print("\nServer is ready! Starting benchmark...")
                server_ready = True
                break
            except requests.exceptions.ConnectionError:
                time.sleep(5)
                print(".", end="", flush=True)

        if not server_ready:
            raise RuntimeError("Timed out waiting for SGLang to start.")

        # 3. Run the same benchmarking script (OpenAI-compatible API)
        bench_cmd = (
            f"python {PROJECT_ROOT}/megakernels/scripts/bench_engines.py "
            "port=10210 "
            "prompt_len=32 "
            "output_len=128 "
            "model='meta-llama/Llama-3.2-1B-Instruct'"
        )

        subprocess.run(bench_cmd, shell=True, cwd=PROJECT_ROOT, check=True)

    finally:
        print("\nTerminating SGLang server...")
        server_process.terminate()
        server_process.wait()


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_profiler_trace():
    """
    Reproduces the 'Bubble Analysis': Generates a Chrome trace.
    Returns the file content bytes so we can save it locally.
    """
    print("=== Generating PyTorch Profiler Trace ===")

    trace_filename = "llama_profile.json"

    cmd = (
        f"python {PROJECT_ROOT}/megakernels/scripts/make_torch_profile.py " f"outfile={trace_filename} " "compile=True"
    )

    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, check=True)

    # Read the file and return content to local
    trace_path = Path(PROJECT_ROOT) / trace_filename
    if trace_path.exists():
        with open(trace_path, "rb") as f:
            return f.read()
    else:
        raise FileNotFoundError("Profiler trace was not generated successfully.")


@app.local_entrypoint()
def main(action: str = "megakernel"):
    if action == "megakernel":
        benchmark_megakernel_h100.remote()
    elif action == "vllm":
        benchmark_vllm_baseline.remote()
    elif action == "sglang":
        benchmark_sglang_baseline.remote()
    elif action == "profile":
        print("Running profiler... (this may take a few minutes)")
        trace_bytes = generate_profiler_trace.remote()

        local_filename = "reproduction_trace.json"
        with open(local_filename, "wb") as f:
            f.write(trace_bytes)
        print(f"Trace saved to {local_filename}. Open in chrome://tracing to see the bubbles (or lack thereof).")
    else:
        print("Invalid action. Choose from: megakernel, vllm, sglang, profile")
