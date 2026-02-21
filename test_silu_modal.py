# deployment script for Modal cloud

import modal

app = modal.App("waterloo-silu-test")

# Build image with kernels copied in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get -y install cuda-toolkit-12-4",
    )
    .pip_install("torch==2.4.0", "ninja")
    .env({"PATH": "/usr/local/cuda-12.4/bin:$PATH"})
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"})
    .copy_local_dir("src/csrc/kernels", "/workspace/kernels")
)


@app.function(image=image, gpu="H100", timeout=600)
def test_silu():
    """Test SiLU kernel on H100"""
    import sys

    import torch

    print("=" * 70)
    print("Waterloo SiLU Kernel Test on Modal")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Add kernel path
    sys.path.insert(0, "/workspace/kernels")

    from silu_torch import test_against_pytorch

    print("\n" + "=" * 70)
    print("Running SiLU kernel tests...")
    print("=" * 70)

    results = test_against_pytorch(shape=(28, 6144), verbose=True)

    if results["pass_vectorized"] and results["pass_fused"]:
        print("\n✓ ALL TESTS PASSED!")
        print(f"  Vectorized: {results['speedup_vectorized']:.2f}x speedup")
        print(f"  Fused: {results['speedup_fused']:.2f}x speedup")
        return {"status": "success", "results": results}
    else:
        print("\n✗ TESTS FAILED")
        return {"status": "failed"}


@app.local_entrypoint()
def main():
    """Run SiLU test"""
    print("Deploying Waterloo SiLU kernel to Modal H100...")
    print()

    result = test_silu.remote()

    if result["status"] == "success":
        print("\n" + "=" * 70)
        print("✓ Modal deployment successful!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ Modal deployment failed")
        print("=" * 70)
        return 1
