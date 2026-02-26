from __future__ import annotations

import torch
from torch import Tensor


def merge_partial_attention(
    O_prev: Tensor,
    lse_prev: Tensor,
    O_new: Tensor,
    lse_new: Tensor,
) -> tuple[Tensor, Tensor]:
    O_prev_f = O_prev.float()
    O_new_f = O_new.float()

    max_lse = torch.maximum(lse_prev, lse_new)
    exp_prev = torch.exp(lse_prev - max_lse)
    exp_new = torch.exp(lse_new - max_lse)

    denom = exp_prev + exp_new
    O_merged = (O_prev_f * exp_prev.unsqueeze(-1) + O_new_f * exp_new.unsqueeze(-1)) / denom.unsqueeze(-1)

    lse_merged = max_lse + torch.log(denom)

    return O_merged.bfloat16(), lse_merged


def get_zigzag_assignment(
    rank: int,
    world_size: int,
    seq_len: int,
    block_size: int,
) -> dict[str, int]:
    total_blocks = (seq_len + block_size - 1) // block_size
    padded_blocks = 2 * world_size

    block_a = rank
    block_b = padded_blocks - 1 - rank

    return {
        "block_a": block_a if block_a < total_blocks else -1,
        "block_b": block_b if block_b < total_blocks else -1,
        "pos_a": block_a * block_size,
        "pos_b": block_b * block_size,
    }
