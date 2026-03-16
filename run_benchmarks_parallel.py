import concurrent.futures
import pprint
import re
import subprocess
import threading

# Lock for thread-safe printing
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def run_benchmark(gpu, engine, run_idx, num_runs):
    script = f"reproduce_baseline_{gpu.lower()}.py"
    cmd = ["modal", "run", script, "--action", engine]

    prefix = f"[{gpu}|{engine}|{run_idx + 1}/{num_runs}]"
    safe_print(f"{prefix} Starting: {' '.join(cmd)}", flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    throughput = None

    for line in iter(process.stdout.readline, ""):
        if engine in ["vllm", "sglang"]:
            match = re.search(r"Throughput:\s*([\d.]+)\s*tokens/s", line)
            if match:
                throughput = float(match.group(1))
        else:
            match = re.search(r"Tokens per second:\s*([\d.]+)", line)
            if match:
                throughput = float(match.group(1))

    process.wait()

    if process.returncode != 0:
        safe_print(
            f"{prefix} Error (return code {process.returncode})",
            flush=True,
        )
        return gpu, engine, throughput

    if throughput is None:
        safe_print(f"{prefix} Could not parse throughput.", flush=True)
    else:
        safe_print(
            f"{prefix} Finished. Parsed Throughput: {throughput}",
            flush=True,
        )

    return gpu, engine, throughput


def main():
    gpus = ["H100", "B200"]
    engines = ["megakernel", "vllm", "sglang"]
    num_runs = 3

    name_map = {
        "megakernel": "Megakernel",
        "vllm": "vLLM",
        "sglang": "SgLang",
    }

    results = {gpu: {name_map[e]: [] for e in engines} for gpu in gpus}

    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as executor:
        futures = []
        for gpu in gpus:
            for engine in engines:
                for run_idx in range(num_runs):
                    futures.append(
                        executor.submit(
                            run_benchmark,
                            gpu,
                            engine,
                            run_idx,
                            num_runs,
                        )
                    )

        for future in concurrent.futures.as_completed(futures):
            gpu, engine, throughput = future.result()
            if throughput is not None:
                results[gpu][name_map[engine]].append(throughput)

    safe_print("\n\n======== FINAL DATA DICTIONARY ========\n", flush=True)
    with print_lock:
        pprint.pprint(results)


if __name__ == "__main__":
    main()
