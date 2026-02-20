from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm

DeviceType = torch.device | str


def assert_div(a: int, b: int) -> int:
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def compute_shard_bounds(tensor_shape: list[int], dim: int, num_shards: int, shard_index: int) -> slice:
    dim_size = tensor_shape[dim]
    base_shard_size = dim_size // num_shards
    remainder = dim_size % num_shards

    start_idx = shard_index * base_shard_size + min(shard_index, remainder)

    if shard_index < remainder:
        end_idx = start_idx + base_shard_size + 1
    else:
        end_idx = start_idx + base_shard_size

    return slice(start_idx, end_idx)


def load_safetensors_repo(
    repo_path: Path,
    include_parameters: set[str],
    device: DeviceType,
    tp_rank: int = 0,
    tp_size: int = 1,
    tp_map: dict[str, int] | None = None,
) -> dict[str, Tensor]:
    if tp_map is None:
        tp_map = {}

    single_file = repo_path / "model.safetensors"
    if single_file.exists():
        files_to_load = [single_file]
    else:
        safetensors_index = repo_path / "model.safetensors.index.json"

        if not safetensors_index.exists():
            raise FileNotFoundError(f"Could not find model.safetensors or model.safetensors.index.json in {repo_path}")

        with open(safetensors_index) as f:
            index = json.load(f)

        param_to_path = index["weight_map"]

        files_to_load_set = set()
        for param_name, path in param_to_path.items():
            if param_name in include_parameters:
                files_to_load_set.add(repo_path / path)

        files_to_load = list(sorted(files_to_load_set))

    state_dict: dict[str, Tensor] = {}

    for file in tqdm(files_to_load, desc="Loading safetensors files"):
        with safe_open(file, framework="pt", device=str(device)) as f:
            for k in f.keys():
                if k in include_parameters:
                    if tp_size > 1 and (split_dim := tp_map.get(k)) is not None:
                        tensor_slice = f.get_slice(k)
                        shard_bounds = compute_shard_bounds(tensor_slice.get_shape(), split_dim, tp_size, tp_rank)
                        if split_dim == 0:
                            state_dict[k] = tensor_slice[shard_bounds]
                        elif split_dim == 1:
                            state_dict[k] = tensor_slice[:, shard_bounds]
                        else:
                            raise ValueError(f"Unsupported split dimension: {split_dim}")
                    else:
                        state_dict[k] = f.get_tensor(k)

    return state_dict


def trepr(t: Tensor) -> str:
    return f"shape={t.shape}, dtype={t.dtype}, device={t.device}, sum={t.sum().item():.4f}"


def get_sm_count(device: DeviceType = "cuda") -> int:
    device_props = torch.cuda.get_device_properties(device)
    return device_props.multi_processor_count
