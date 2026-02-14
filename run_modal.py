import os
from pathlib import Path

import modal

force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"

app = modal.App("cs450-project")

h100_image = modal.Image.from_dockerfile(
    Path(__file__).parent / "Dockerfile.h100",
    force_build=force_rebuild,
)

b200_image = modal.Image.from_dockerfile(
    Path(__file__).parent / "Dockerfile.b200",
    force_build=force_rebuild,
)


@app.function(
    image=h100_image,
    gpu="H100",
    timeout=3600,
)
def run_hazy_h100():
    import os
    import sys

    import torch

    if "HF_TOKEN" not in os.environ:
        print("MISSING HF_TOKEN!")

    sys.path.insert(0, "/workspace/Megakernels")

    print("Running Hazy baseline on H100.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    print("Running Hazy megakernel demo...")

    # cd into /workspace/Megakernels and run the demo script
    import os

    os.chdir("/workspace/Megakernels")
    os.system('python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100')

    return "Hazy H100 job completed"


@app.function(
    image=b200_image,
    gpu="B200",
    timeout=3600,
)
def run_hazy_b200():
    import os
    import sys
    import torch

    if "HF_TOKEN" not in os.environ:
        print("MISSING HF_TOKEN!")

    sys.path.insert(0, "/workspace/Megakernels")

    print("Running Hazy baseline on B200.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    print("Running Hazy megakernel demo...")

    os.chdir("/workspace/Megakernels")
    os.system(
        'python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100'
    )
    return "Hazy B200 job completed"


@app.function(image=h100_image, gpu="H100", timeout=3600)
def run_waterloo_h100():
    import sys
    import torch

    print("Running Waterloo SiLU implementation on H100.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Add kernel directory to path
    sys.path.insert(0, "/workspace/src/csrc/kernels")
    
    from silu_torch import test_against_pytorch
    
    print("\nTesting SiLU kernel on H100...")
    results = test_against_pytorch(shape=(28, 6144), verbose=True)
    
    if results['pass_fused']:
        print("\n✓ Waterloo SiLU kernel test PASSED on H100")
        return "Waterloo H100 test passed"
    else:
        print("\n✗ Waterloo SiLU kernel test FAILED on H100")
        return "Waterloo H100 test failed"


@app.function(image=b200_image, gpu="B200", timeout=3600)
def run_waterloo_b200():
    import sys
    import torch

    print("Running Waterloo SiLU implementation on B200.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Add kernel directory to path
    sys.path.insert(0, "/workspace/src/csrc/kernels")
    
    from silu_torch import test_against_pytorch
    
    print("\nTesting SiLU kernel on B200...")
    results = test_against_pytorch(shape=(28, 6144), verbose=True)
    
    if results['pass_fused']:
        print("\n✓ Waterloo SiLU kernel test PASSED on B200")
        return "Waterloo B200 test passed"
    else:
        print("\n✗ Waterloo SiLU kernel test FAILED on B200")
        return "Waterloo B200 test failed"


@app.local_entrypoint()
def main(hazy_megakernel: bool = False, waterloo_megakernel: bool = False, gpu: str = "b200"):
    # Determine implementation logic
    implementation = "waterloo"  # Default
    if hazy_megakernel:
        implementation = "hazy"
    elif waterloo_megakernel:
        implementation = "waterloo"

    if implementation == "hazy":
        if gpu == "h100":
            run_hazy_h100.remote()
        elif gpu == "b200":
            run_hazy_b200.remote()
    elif implementation == "waterloo":
        if gpu == "h100":
            run_waterloo_h100.remote()
        elif gpu == "b200":
            run_waterloo_b200.remote()
