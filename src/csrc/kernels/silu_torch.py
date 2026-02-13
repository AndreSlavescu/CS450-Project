# binding for the Cuda file and Python
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from pathlib import Path

# Lazy load the CUDA extension
_silu_cuda = None

def _get_silu_cuda():
    """Load CUDA extension on first use"""
    global _silu_cuda
    if _silu_cuda is None:
        kernel_dir = Path(__file__).parent
        _silu_cuda = load(
            name='silu_cuda',
            sources=[str(kernel_dir / 'silu.cu')],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-std=c++17',
            ],
            verbose=True,
        )
    return _silu_cuda


def silu_naive(input: torch.Tensor) -> torch.Tensor:
    """
    Apply SiLU activation using naive CUDA kernel
    
    Args:
        input: Input tensor of shape [..., N]
    
    Returns:
        Output tensor of same shape as input
    """
    assert input.is_cuda, "Input must be on CUDA device"
    assert input.dtype == torch.float32, "Only float32 supported"
    
    cuda_module = _get_silu_cuda()
    original_shape = input.shape
    input_flat = input.reshape(-1).contiguous()
    output = cuda_module.silu_naive(input_flat)
    return output.reshape(original_shape)


def silu_vectorized(input: torch.Tensor) -> torch.Tensor:
    """
    Apply SiLU activation using vectorized CUDA kernel (float4 optimization)
    
    Args:
        input: Input tensor of shape [..., N]
    
    Returns:
        Output tensor of same shape as input
    """
    assert input.is_cuda, "Input must be on CUDA device"
    assert input.dtype == torch.float32, "Only float32 supported"
    
    cuda_module = _get_silu_cuda()
    original_shape = input.shape
    input_flat = input.reshape(-1).contiguous()
    output = cuda_module.silu_vectorized(input_flat)
    return output.reshape(original_shape)


def silu_multiply(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Apply fused SiLU-Multiply: output = SiLU(gate) * up
    
    This is the common pattern in Qwen3 MLP blocks:
        gate_out = gate_proj(x)
        up_out = up_proj(x)
        result = silu_multiply(gate_out, up_out)  # <- This function
    
    Args:
        gate: Gate tensor of shape [..., N]
        up: Up tensor of same shape as gate
    
    Returns:
        Output tensor of same shape as inputs
    """
    assert gate.is_cuda and up.is_cuda, "Inputs must be on CUDA device"
    assert gate.dtype == torch.float32 and up.dtype == torch.float32, "Only float32 supported"
    assert gate.shape == up.shape, "Gate and up must have same shape"
    
    cuda_module = _get_silu_cuda()
    original_shape = gate.shape
    gate_flat = gate.reshape(-1).contiguous()
    up_flat = up.reshape(-1).contiguous()
    output = cuda_module.silu_multiply(gate_flat, up_flat)
    return output.reshape(original_shape)


def test_against_pytorch(shape=(1, 6144), device='cuda', verbose=True):
    """
    Test custom SiLU kernels against PyTorch baseline
    
    Args:
        shape: Shape of test tensor (default: Qwen3 intermediate size)
        device: CUDA device (default: 'cuda')
        verbose: Print detailed results
    
    Returns:
        Dictionary with test results and timings
    """
    torch.manual_seed(42)
    
    # Generate test input
    x = torch.randn(shape, device=device, dtype=torch.float32)
    
    # PyTorch baseline
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    y_torch = F.silu(x)
    end.record()
    torch.cuda.synchronize()
    time_torch = start.elapsed_time(end)
    
    # Test naive kernel
    start.record()
    y_naive = silu_naive(x)
    end.record()
    torch.cuda.synchronize()
    time_naive = start.elapsed_time(end)
    
    diff_naive = torch.max(torch.abs(y_torch - y_naive)).item()
    
    # Test vectorized kernel
    start.record()
    y_vec = silu_vectorized(x)
    end.record()
    torch.cuda.synchronize()
    time_vec = start.elapsed_time(end)
    
    diff_vec = torch.max(torch.abs(y_torch - y_vec)).item()
    
    # Test fused multiply
    x_gate = torch.randn(shape, device=device, dtype=torch.float32)
    x_up = torch.randn(shape, device=device, dtype=torch.float32)
    
    start.record()
    y_torch_fused = F.silu(x_gate) * x_up
    end.record()
    torch.cuda.synchronize()
    time_torch_fused = start.elapsed_time(end)
    
    start.record()
    y_fused = silu_multiply(x_gate, x_up)
    end.record()
    torch.cuda.synchronize()
    time_fused = start.elapsed_time(end)
    
    diff_fused = torch.max(torch.abs(y_torch_fused - y_fused)).item()
    
    results = {
        'shape': shape,
        'max_diff_naive': diff_naive,
        'max_diff_vectorized': diff_vec,
        'max_diff_fused': diff_fused,
        'time_pytorch_ms': time_torch,
        'time_naive_ms': time_naive,
        'time_vectorized_ms': time_vec,
        'time_pytorch_fused_ms': time_torch_fused,
        'time_fused_ms': time_fused,
        'speedup_vectorized': time_torch / time_vec,
        'speedup_fused': time_torch_fused / time_fused,
        'pass_naive': diff_naive < 1e-5,
        'pass_vectorized': diff_vec < 1e-5,
        'pass_fused': diff_fused < 1e-5,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SiLU CUDA Kernel Test Results")
        print(f"{'='*60}")
        print(f"Shape: {shape}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"\n{'Correctness':-^60}")
        print(f"  Naive kernel     : {'✓ PASS' if results['pass_naive'] else '✗ FAIL'} (max_diff={diff_naive:.2e})")
        print(f"  Vectorized kernel: {'✓ PASS' if results['pass_vectorized'] else '✗ FAIL'} (max_diff={diff_vec:.2e})")
        print(f"  Fused multiply   : {'✓ PASS' if results['pass_fused'] else '✗ FAIL'} (max_diff={diff_fused:.2e})")
        print(f"\n{'Performance':-^60}")
        print(f"  PyTorch SiLU           : {time_torch:.3f} ms")
        print(f"  Naive kernel           : {time_naive:.3f} ms ({time_naive/time_torch:.2f}x)")
        print(f"  Vectorized kernel      : {time_vec:.3f} ms ({results['speedup_vectorized']:.2f}x speedup)")
        print(f"  PyTorch SiLU+Multiply  : {time_torch_fused:.3f} ms")
        print(f"  Fused SiLU-Multiply    : {time_fused:.3f} ms ({results['speedup_fused']:.2f}x speedup)")
        print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    # Test with various sizes
    print("Testing SiLU CUDA kernels...")
    
    test_against_pytorch(shape=(1, 6144), verbose=True)  # Qwen3 intermediate size
    test_against_pytorch(shape=(1, 2048), verbose=True)  # Qwen3 hidden size
    test_against_pytorch(shape=(28, 6144), verbose=True)  # Batch of tokens
