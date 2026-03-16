import re
import subprocess


def run_benchmark(gpu, engine):
    script = f"reproduce_baseline_{gpu.lower()}.py"
    cmd = ["modal", "run", script, "--action", engine]
    print(f"Running: {' '.join(cmd)}", flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    throughput = None

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

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
        print(
            f"Error running {engine} on {gpu} (return code {process.returncode})",
            flush=True,
        )
        return None

    if throughput is None:
        print(
            f"Could not parse throughput from output for {engine} on {gpu}.",
            flush=True,
        )

    return throughput


def main():
    # gpus = ["H100", "B200"]
    gpus = ["H100"]
    # gpus = ["B200"]
    engines = [
        "megakernel",
        "vllm",
        "sglang",
    ]
    num_runs = 3

    name_map = {
        "megakernel": "Megakernel",
        "vllm": "vLLM",
        "sglang": "SgLang",
    }

    results = {gpu: {name_map[e]: [] for e in engines} for gpu in gpus}

    import pprint

    for gpu in gpus:
        for engine in engines:
            for run_idx in range(num_runs):
                print(
                    f"--- {gpu} | {engine} | Run {run_idx + 1}/{num_runs} ---",
                    flush=True,
                )
                throughput = run_benchmark(gpu, engine)
                if throughput is not None:
                    print(f"Parsed Throughput: {throughput}", flush=True)
                    results[gpu][name_map[engine]].append(throughput)
                else:
                    print(
                        "Parsed Throughput: FAILED",
                        flush=True,
                    )

    print("\n\n======== FINAL DATA DICTIONARY ========\n", flush=True)
    pprint.pprint(results)


if __name__ == "__main__":
    main()
