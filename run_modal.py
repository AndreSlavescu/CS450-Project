import argparse
import modal
from pathlib import Path

app = modal.App("cs450-project")

h100_image = modal.Image.from_dockerfile(
    Path(__file__).parent / "Dockerfile.h100",
    add_python="3.11",
)

b200_image = modal.Image.from_dockerfile(
    Path(__file__).parent / "Dockerfile.b200",
    add_python="3.11",
)

@app.function(image=h100_image, gpu="H100", timeout=3600)
def run_hazy_h100():
    import torch
    import sys
    sys.path.insert(0, '/workspace/Megakernels')
    
    print("Running Hazy baseline on H100.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("Running Hazy megakernel demo...")
    return "Hazy H100 job completed"


@app.function(image=b200_image, gpu="B200", timeout=3600)
def run_hazy_b200():
    import torch
    import sys
    sys.path.insert(0, '/workspace/Megakernels')
    
    print("Running Hazy baseline on B200.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("Running Hazy megakernel demo...")
    return "Hazy B200 job completed"


@app.function(image=h100_image, gpu="H100", timeout=3600)
def run_waterloo_h100():
    import torch
    
    print("Running Waterloo implementation on H100.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    raise NotImplementedError("Waterloo megakernel implementation not yet available.")


@app.function(image=b200_image, gpu="B200", timeout=3600)
def run_waterloo_b200():
    import torch
    
    print("Running Waterloo implementation on B200.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    raise NotImplementedError("Waterloo megakernel implementation not yet available.")


@app.local_entrypoint()
def main(implementation: str = "waterloo", gpu: str = "b200"):
    if implementation == "hazy":
        if gpu == "h100":
            result = run_hazy_h100.remote()
        elif gpu == "b200":
            result = run_hazy_b200.remote()
    elif implementation == "waterloo":
        if gpu == "h100":
            result = run_waterloo_h100.remote()
        elif gpu == "b200":
            result = run_waterloo_b200.remote()
    else:
        raise ValueError(f"Unsupported implementation: {implementation}")
    
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    implementation_choice = parser.add_mutually_exclusive_group() # can't select both at same time. This prevents that.
    implementation_choice.add_argument(
        "--hazy-megakernel",
        action="store_const",
        const="hazy",
        dest="implementation",
        help="Run HazyResearch baseline implementation",
    )
    implementation_choice.add_argument(
        "--waterloo-megakernel",
        action="store_const",
        const="waterloo",
        dest="implementation",
        help="Run Waterloo custom implementation (default)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=["h100", "b200"],
        default="b200",
        help="Select between H100 80GB or B200 80GB",
    )
    args = parser.parse_args()

    with app.run():
        if args.implementation == "hazy":
            if args.gpu == "h100":
                result = run_hazy_h100.remote()
            else:
                result = run_hazy_b200.remote()
        else:
            if args.gpu == "h100":
                result = run_waterloo_h100.remote()
            else:
                result = run_waterloo_b200.remote()
        
        print(result)
