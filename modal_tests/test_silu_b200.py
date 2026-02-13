"""
CS450 Project - Waterloo Team
Modal B200 Deployment Test for SiLU CUDA Kernel

This script deploys and tests the custom SiLU kernel on Modal B200 GPU.
"""

import modal

app = modal.App("cs450-silu-test")

# Build image with CUDA toolkit and PyTorch
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # Install CUDA toolkit for nvcc
    .run_commands(
        "apt-get update",
        "apt-get install -y wget",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get -y install cuda-toolkit-12-4",
    )
    .pip_install(
        "torch==2.4.0",
        "ninja",
    )
    .env({"PATH": "/usr/local/cuda-12.4/bin:$PATH"})
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"})
)


@app.function(
    image=image,
    gpu="H100",  # Start with H100 (change to B200 when available)
    timeout=600,
)
def test_silu_kernel():
    """Test SiLU CUDA kernel on Modal GPU"""
    import sys
    import torch
    import time
    from pathlib import Path
    
    print("="*70)
    print("CS450 Project - SiLU CUDA Kernel Test on Modal")
    print("="*70)
    
    # GPU info
    print(f"\n{'GPU Information':-^70}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # Import the kernel wrapper
    # We'll mount the kernel directory in the next step
    kernel_dir = Path("/root/kernels")
    sys.path.insert(0, str(kernel_dir))
    
    from silu_torch import silu_naive, silu_vectorized, silu_multiply, test_against_pytorch
    
    print(f"\n{'Running Tests':-^70}")
    
    # Test 1: Small tensor (warm-up)
    print("\n[1/4] Testing small tensor (1024 elements)...")
    results_small = test_against_pytorch(shape=(1, 1024), verbose=False)
    print(f"  ✓ Vectorized: {results_small['speedup_vectorized']:.2f}x speedup, max_diff={results_small['max_diff_vectorized']:.2e}")
    print(f"  ✓ Fused multiply: {results_small['speedup_fused']:.2f}x speedup, max_diff={results_small['max_diff_fused']:.2e}")
    
    # Test 2: Qwen3 hidden size
    print("\n[2/4] Testing Qwen3 hidden size (2048 elements)...")
    results_hidden = test_against_pytorch(shape=(1, 2048), verbose=False)
    print(f"  ✓ Vectorized: {results_hidden['speedup_vectorized']:.2f}x speedup, max_diff={results_hidden['max_diff_vectorized']:.2e}")
    print(f"  ✓ Fused multiply: {results_hidden['speedup_fused']:.2f}x speedup, max_diff={results_hidden['max_diff_fused']:.2e}")
    
    # Test 3: Qwen3 intermediate size (main use case)
    print("\n[3/4] Testing Qwen3 intermediate size (6144 elements)...")
    results_inter = test_against_pytorch(shape=(1, 6144), verbose=False)
    print(f"  ✓ Vectorized: {results_inter['speedup_vectorized']:.2f}x speedup, max_diff={results_inter['max_diff_vectorized']:.2e}")
    print(f"  ✓ Fused multiply: {results_inter['speedup_fused']:.2f}x speedup, max_diff={results_inter['max_diff_fused']:.2e}")
    
    # Test 4: Batched (28 tokens)
    print("\n[4/4] Testing batched (28 tokens × 6144)...")
    results_batch = test_against_pytorch(shape=(28, 6144), verbose=False)
    print(f"  ✓ Vectorized: {results_batch['speedup_vectorized']:.2f}x speedup, max_diff={results_batch['max_diff_vectorized']:.2e}")
    print(f"  ✓ Fused multiply: {results_batch['speedup_fused']:.2f}x speedup, max_diff={results_batch['max_diff_fused']:.2e}")
    
    # Summary
    print(f"\n{'Test Summary':-^70}")
    all_passed = all([
        results_small['pass_vectorized'], results_small['pass_fused'],
        results_hidden['pass_vectorized'], results_hidden['pass_fused'],
        results_inter['pass_vectorized'], results_inter['pass_fused'],
        results_batch['pass_vectorized'], results_batch['pass_fused'],
    ])
    
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
        print(f"  Average speedup (vectorized): {(results_inter['speedup_vectorized'] + results_batch['speedup_vectorized']) / 2:.2f}x")
        print(f"  Average speedup (fused): {(results_inter['speedup_fused'] + results_batch['speedup_fused']) / 2:.2f}x")
    else:
        print("  ✗ SOME TESTS FAILED")
        return {"status": "failed"}
    
    print("="*70)
    
    return {
        "status": "success",
        "device": torch.cuda.get_device_name(0),
        "results": {
            "small": results_small,
            "hidden": results_hidden,
            "intermediate": results_inter,
            "batched": results_batch,
        }
    }


# Mount the kernel directory
kernel_mount = modal.Mount.from_local_dir(
    local_path="src/csrc/kernels",
    remote_path="/root/kernels",
)

# Update the function to include the mount
@app.function(
    image=image,
    gpu="H100",  # Change to "B200" when you have access
    timeout=600,
    mounts=[kernel_mount],
)
def test_silu_with_mount():
    """Test with kernel files mounted"""
    return test_silu_kernel.local()


@app.local_entrypoint()
def main():
    """Run the SiLU test on Modal"""
    print("Deploying SiLU test to Modal...")
    print("Note: Currently using H100. Change to B200 in code when available.")
    print()
    
    result = test_silu_with_mount.remote()
    
    if result["status"] == "success":
        print("\n✓ Modal deployment successful!")
        print(f"Tested on: {result['device']}")
    else:
        print("\n✗ Modal deployment failed!")
        return 1
    
    return 0
