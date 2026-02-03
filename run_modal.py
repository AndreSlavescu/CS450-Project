import argparse
from pathlib import Path

import modal

app = modal.App("cs450-project")

h100_image = modal.Image.from_dockerfile(
    Path(__file__).parent / "Dockerfile.h100",
)

# b200_image = modal.Image.from_dockerfile(
#     Path(__file__).parent / "Dockerfile.b200",
#     add_python="3.11",
# )


@app.function(image=h100_image, gpu="H100", timeout=3600)
def run_hazy_h100():
    import sys

    import torch

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
    os.system(
        'python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100'
    )

    return "Hazy H100 job completed"


# @app.function(image=b200_image, gpu="B200", timeout=3600)
# def run_hazy_b200():
#     import sys

#     import torch

#     sys.path.insert(0, "/workspace/Megakernels")

#     print("Running Hazy baseline on B200.")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"Device: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA version: {torch.version.cuda}")

#     print("Running Hazy megakernel demo...")
#     import os

#     os.chdir("/workspace/Megakernels")
#     os.system(
#         'python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100'
#     )
#     return "Hazy B200 job completed"


@app.function(image=h100_image, gpu="H100", timeout=3600)
def run_waterloo_h100():
    import torch

    print("Running Waterloo implementation on H100.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    raise NotImplementedError("Waterloo megakernel implementation not yet available.")


# @app.function(image=b200_image, gpu="B200", timeout=3600)
# def run_waterloo_b200():
#     import torch

#     print("Running Waterloo implementation on B200.")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"Device: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA version: {torch.version.cuda}")

#     raise NotImplementedError("Waterloo megakernel implementation not yet available.")


@app.local_entrypoint()
def main(
    hazy_megakernel: bool = False, waterloo_megakernel: bool = False, gpu: str = "b200"
):
    # Determine implementation logic
    implementation = "waterloo"  # Default
    if hazy_megakernel:
        implementation = "hazy"
    elif waterloo_megakernel:
        implementation = "waterloo"

    if implementation == "hazy":
        if gpu == "h100":
            result = run_hazy_h100.remote()
        elif gpu == "b200":
            pass
            # result = run_hazy_b200.remote()
    elif implementation == "waterloo":
        if gpu == "h100":
            result = run_waterloo_h100.remote()
        elif gpu == "b200":
            pass
            # result = run_waterloo_b200.remote()

    # print(result)
