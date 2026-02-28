from pathlib import Path

from torch.utils.cpp_extension import load

_module = None
_tp_module = None

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # CS450-Project/
_KERNELS_DIR = _REPO_ROOT / "src" / "csrc" / "kernels"
_PROFILER_DIR = _REPO_ROOT / "src" / "csrc" / "profiler"


def _detect_arch() -> str:
    """Return nvcc -arch flag for the current GPU."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    if major == 10:
        return "sm_100a"  # Blackwell (B200)
    elif major == 9:
        return "sm_90a"  # Hopper   (H100)
    elif major == 8:
        return "sm_80"  # Ampere   (A100)
    else:
        raise RuntimeError(
            f"Unsupported GPU compute capability {major}.{minor}. " "Add the arch flag for your GPU in build.py."
        )


def get_kernels():
    """Build (or return cached) the qwen3_kernels PyTorch extension."""
    global _module
    if _module is not None:
        return _module

    arch = _detect_arch()
    print(f"Compiling qwen3_kernels for {arch}...")

    _module = load(
        name="qwen3_kernels",
        sources=[str(_KERNELS_DIR / "qwen3_kernels.cu")],
        extra_include_paths=[str(_KERNELS_DIR), str(_PROFILER_DIR)],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            f"-arch={arch}",
        ],
        verbose=False,
    )
    print("qwen3_kernels compiled OK.")
    return _module


def get_tp_kernels():
    """Build (or return cached) the tp_sp distributed kernels extension."""
    global _tp_module
    if _tp_module is not None:
        return _tp_module

    arch = _detect_arch()
    print(f"Compiling tp_sp kernels for {arch}...")

    _tp_module = load(
        name="tp_sp",
        sources=[str(_KERNELS_DIR / "tp_sp.cu")],
        extra_include_paths=[str(_KERNELS_DIR)],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            f"-arch={arch}",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    print("tp_sp kernels compiled OK.")
    return _tp_module
