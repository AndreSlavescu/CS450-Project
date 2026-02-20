from __future__ import annotations

import torch
from torch import Tensor


def merge_partial_attention(
    O_prev: Tensor,
    lse_prev: Tensor,
    O_new: Tensor,
    lse_new: Tensor,
) -> tuple[Tensor, Tensor]:
    """Merge two partial attention outputs using log-sum-exp.

    Used by zigzag ring attention to combine results from different KV blocks
    across ring iterations.  The merge is numerically stable via the LSE trick.

    Args:
        O_prev:   [num_heads, seq_q, head_dim]  bf16 — previous merged output
        lse_prev: [num_heads, seq_q]             fp32 — previous log-sum-exp
        O_new:    [num_heads, seq_q, head_dim]  bf16 — new partial output
        lse_new:  [num_heads, seq_q]             fp32 — new partial log-sum-exp

    Returns:
        O_merged:   [num_heads, seq_q, head_dim]  bf16
        lse_merged: [num_heads, seq_q]             fp32
    """
    # Work in fp32 for numerical accuracy
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
    """Compute zigzag block assignment for a given rank.

    ZigZag splits the sequence into 2*world_size blocks and assigns mirrored
    pairs to each GPU for load-balanced causal attention:
        GPU k gets blocks (k) and (2*world_size - 1 - k)

    Args:
        rank:       GPU rank index
        world_size: total number of GPUs
        seq_len:    full sequence length
        block_size: tokens per block

    Returns:
        dict with keys:
            block_a: forward block index (-1 if out of bounds)
            block_b: mirrored block index (-1 if out of bounds)
            pos_a:   starting token position of block_a
            pos_b:   starting token position of block_b
    """
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
