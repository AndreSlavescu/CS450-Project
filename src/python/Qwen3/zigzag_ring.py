from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

from .attention_utils import get_zigzag_assignment, merge_partial_attention


def zigzag_ring_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    comm_group: dist.ProcessGroup,
    scale: float,
    total_seq_len: int,
) -> Tensor:
    """Zigzag ring attention with FA4 kernel.

    Distributes long-context prefill across GPUs using the zigzag ring
    topology.  Each rank holds a zigzag-assigned chunk of the sequence.
    KV blocks rotate through a ring, and each step computes local attention
    using the FA4 CUDA kernel, then merges partial results via online
    softmax (LSE-based accumulation).

    Communication is overlapped with compute via async send/recv.

    Args:
        Q:  [num_q_heads, local_seq_len, head_dim]  bf16
        K:  [num_kv_heads, local_seq_len, head_dim]  bf16
        V:  [num_kv_heads, local_seq_len, head_dim]  bf16
        comm_group: torch.distributed process group for ring communication
        scale: attention scale (1/sqrt(head_dim))
        total_seq_len: full sequence length across all ranks

    Returns:
        O:  [num_q_heads, local_seq_len, head_dim]  bf16
    """
    import fmha_attention  # compiled CUDA module

    rank = dist.get_rank(comm_group)
    world_size = dist.get_world_size(comm_group)

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    head_dim = K.size(2)
    local_seq = K.size(1)

    # Zigzag assignment for this rank — determines Q position offset
    assignment = get_zigzag_assignment(rank, world_size, total_seq_len, local_seq)
    q_offset = assignment["pos_a"]

    # Pack K and V into a single buffer for ring communication
    # Shape: [num_kv_heads, local_seq, head_dim * 2]
    kv_buf_0 = torch.cat([K, V], dim=-1).contiguous()
    kv_buf_1 = torch.empty_like(kv_buf_0)

    current_kv = kv_buf_0
    next_kv = kv_buf_1
    current_kv_rank = rank

    O_acc: Tensor | None = None
    lse_acc: Tensor | None = None

    for step in range(world_size):
        # Determine KV position offset for the current source rank
        kv_assignment = get_zigzag_assignment(current_kv_rank, world_size, total_seq_len, local_seq)
        kv_offset = kv_assignment["pos_a"]

        # Unpack K and V from the ring buffer
        K_tile = current_kv[..., :head_dim].contiguous()
        V_tile = current_kv[..., head_dim:].contiguous()

        # Start async send/recv for the next step
        ops: list[dist.Work] = []
        if step < world_size - 1:
            send_op = dist.isend(current_kv, dst=next_rank, group=comm_group)
            recv_op = dist.irecv(next_kv, src=prev_rank, group=comm_group)
            ops = [send_op, recv_op]

        # FMHA kernel: compute partial attention with LSE
        causal = step == 0  # causal only for self-block
        result = fmha_attention.forward(
            Q,
            K_tile,
            V_tile,
            scale,
            causal,
            True,  # return_lse
            q_offset,
            kv_offset,
        )
        O_partial, lse_partial = result[0], result[1]

        # Merge partial results via online softmax
        if O_acc is None:
            O_acc = O_partial
            lse_acc = lse_partial
        else:
            assert lse_acc is not None
            O_acc, lse_acc = merge_partial_attention(O_acc, lse_acc, O_partial, lse_partial)

        # Wait for communication before swapping buffers
        for op in ops:
            op.wait()

        # Swap double buffers and advance ring
        current_kv, next_kv = next_kv, current_kv
        current_kv_rank = (current_kv_rank - 1 + world_size) % world_size

    assert O_acc is not None
    return O_acc
