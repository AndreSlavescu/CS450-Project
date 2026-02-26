"""Instruction scheduler for the Qwen3-1.7B CUDA VM megakernel.

This single file provides the complete scheduling stack:

  Base infrastructure
  -------------------
  Instruction           — base class for all opcodes
  NoOp                  — padding instruction (opcode 0)
  BaseGlobals           — base model state dataclass
  DAG_Node              — node in the computation DAG
  Schedule              — container + assignment helpers
  ScheduleBuilder       — abstract base for schedule construction
  assign_to_sms()       — dispatcher for SM-assignment strategies
  tensorize_instructions() — serialize to GPU tensors for the CUDA VM

  Qwen3-1.7B opcodes (1-7)
  -------------------------
  1  QKV_LayerNorm_RoPE_Append   — RMSNorm + QKV GEMV + Q/K RMSNorm + RoPE + KV cache
  2  FlashDecode_GQA              — Partial flash-decode attention (one KV head)
  3  AttentionReduction           — Reduce partial outputs (log-sum-exp merge)
  4  O_Proj_Residual              — O-proj GEMV + residual add
  5  MLP_LayerNorm_UpGate_SiLU   — MLP RMSNorm + gate+up GEMV + fused SiLU-multiply
  6  DownProj_Residual            — Down-proj GEMV + residual add
  7  RMSNorm_LM_Head              — Final RMSNorm + LM-head GEMV

  Qwen3-1.7B scheduling
  ----------------------
  Qwen3Globals                 — extends BaseGlobals with per-head Q/K norms + buffers
  Qwen3LatencyScheduleBuilder  — builds the full 28-layer DAG for BS=1 latency path
  Qwen3PyVM                    — pure-PyTorch reference interpreter for diff-testing

B200 SM budget (148 SMs)
  QKV:       4096  / 16  = 256  blocks → 1.73/SM
  O-proj:    2048  / 8   = 256  blocks → 1.73/SM
  Up+Gate:   6144  / 16  = 384  blocks → 2.59/SM
  Down-proj: 2048  / 8   = 256  blocks → 1.73/SM  (x3 col-splits = 768, 5.19/SM)
  LM head:   151936/ 16  = 9496 blocks → 64.2/SM
  Attention: 8 KV heads  = 8    tasks  (coarser, bounded by GQA ratio)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

import torch
from torch import Tensor

from .utils import DeviceType, get_sm_count

INTS_PER_INSTRUCTION = 32
TIMING_SLOTS = 128


# ---------------------------------------------------------------------------
# Base instruction classes
# ---------------------------------------------------------------------------


@dataclass
class Instruction:
    @classmethod
    def opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def prev_opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def tags(cls) -> dict[str, Any]:
        return {}

    def cost(self, globs: BaseGlobals) -> float:
        return 1.0

    def serialize(self) -> list[int]:
        words: list[int] = [self.opcode()]
        for f in fields(self):
            if f.name == "global_idx":
                continue
            attr = getattr(self, f.name)
            if isinstance(attr, int):
                words.append(attr)
            elif isinstance(attr, (tuple, list)):
                words.append(len(attr))
                words.extend(attr)
            elif attr is None:
                words.append(0)
            else:
                raise ValueError(f"Unsupported field type: {type(attr)}")
        return words


@dataclass
class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0


# ---------------------------------------------------------------------------
# BaseGlobals — holds all model state needed by the VM
# ---------------------------------------------------------------------------


@dataclass
class BaseGlobals:
    # stacked model parameters
    qkv_proj_weights: Tensor
    attn_ln_weights: Tensor
    o_proj_weights: Tensor
    mlp_ln_weights: Tensor
    up_proj_weights: Tensor
    gate_proj_weights: Tensor
    down_proj_weights: Tensor
    lm_head_norm_weights: Tensor
    lm_head_weights: Tensor
    k_cache: Tensor
    v_cache: Tensor

    # RoPE embeddings (not stacked per layer)
    rope_cos: Tensor
    rope_sin: Tensor

    # model constants
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    attn_scale: float
    rms_norm_eps: float
    device: DeviceType

    hidden_states: Tensor
    barriers: Tensor

    pos_id: int

    def __post_init__(self):
        self.instructions: Tensor | None = None
        self.timings: Tensor | None = None

    def sm_count(self) -> int:
        return get_sm_count(self.device)

    def num_total_heads(self) -> int:
        return self.num_attention_heads + self.num_kv_heads * 2


# ---------------------------------------------------------------------------
# DAG node for instruction scheduling
# ---------------------------------------------------------------------------


@dataclass
class DAG_Node:
    instruction: Instruction
    dependencies: list[DAG_Node]

    children: set[DAG_Node] = field(default_factory=set)
    start_time: float = float("inf")
    end_time: float = float("inf")
    remaining_dependencies: set[DAG_Node] = field(default_factory=set)
    priority: float = 0

    def __hash__(self):
        return hash(tuple(self.instruction.serialize()))

    def earliest_ready_time(self, globs: BaseGlobals) -> float:
        if len(self.dependencies) == 0:
            return 0
        return max(dep.end_time for dep in self.dependencies)

    def register_with_parents(self):
        for dep in self.dependencies:
            dep.children.add(self)

    def calc_priority(self, globs: BaseGlobals):
        cur_cost = self.priority
        for dep in self.dependencies:
            pri = cur_cost + dep.instruction.cost(globs)
            dep.priority = max(pri, dep.priority)
            dep.calc_priority(globs)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass
class Schedule:
    globs: BaseGlobals
    dag_nodes: list[DAG_Node]
    end_node: DAG_Node

    def get_linear_instructions(self) -> list[Instruction]:
        return [node.instruction for node in self.dag_nodes]

    def smart_assign_to_sms(self) -> list[list[Instruction]]:
        return assign_dag_to_sms(self)

    def round_robin_assign_to_sms(self) -> list[list[Instruction]]:
        instructions = self.get_linear_instructions()
        return round_robin_assign_to_sms(instructions, self.globs.sm_count())


# ---------------------------------------------------------------------------
# ScheduleBuilder — abstract base for model-specific schedule construction
# ---------------------------------------------------------------------------


class ScheduleBuilder:
    @classmethod
    def make_globals(cls, model) -> BaseGlobals:
        raise NotImplementedError

    @classmethod
    def make_dag(
        cls,
        globs: BaseGlobals,
        stop_after_op: str | None = None,
        layer_limit: int | None = None,
    ) -> tuple[list[DAG_Node], DAG_Node]:
        raise NotImplementedError

    @classmethod
    def build(
        cls,
        model,
        stop_after_op: str | None = None,
        layer_limit: int | None = None,
    ) -> Schedule:
        globs = cls.make_globals(model)
        dag_nodes, end_node = cls.make_dag(globs, stop_after_op, layer_limit)
        return Schedule(globs, dag_nodes, end_node)

    @classmethod
    def with_new_globals(cls, schedule: Schedule, model) -> Schedule:
        return replace(schedule, globs=cls.make_globals(model))


# ---------------------------------------------------------------------------
# SM assignment strategies
# ---------------------------------------------------------------------------


def assign_dag_to_sms(schedule: Schedule) -> list[list[Instruction]]:
    nodes = schedule.dag_nodes
    globs = schedule.globs

    for node in nodes:
        node.register_with_parents()
    for node in nodes:
        node.remaining_dependencies = set(node.dependencies)

    sm_count = globs.sm_count()
    sm_queues: list[list[Instruction]] = [[] for _ in range(sm_count)]

    sm_heap: list[tuple[float, int]] = [(0, i) for i in range(sm_count)]
    heapq.heapify(sm_heap)

    idx_to_node = dict(enumerate(nodes))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    ready_heap: list[tuple[float, int]] = []
    for node in nodes:
        if len(node.dependencies) == 0:
            idx = node_to_idx[node]
            ready_heap.append((-node.instruction.cost(globs), idx))
    heapq.heapify(ready_heap)

    while ready_heap:
        _ready_cost, idx = heapq.heappop(ready_heap)
        node = idx_to_node[idx]

        sm_time, sm_idx = heapq.heappop(sm_heap)
        end_time = sm_time + node.instruction.cost(globs)

        node.start_time = sm_time
        node.end_time = end_time
        sm_queues[sm_idx].append(node.instruction)
        heapq.heappush(sm_heap, (end_time, sm_idx))

        for child in node.children:
            child.remaining_dependencies.discard(node)
            if len(child.remaining_dependencies) == 0:
                cidx = node_to_idx[child]
                heapq.heappush(ready_heap, (-child.instruction.cost(globs), cidx))

    return sm_queues


def round_robin_assign_to_sms(instructions: list[Instruction], sm_count: int) -> list[list[Instruction]]:
    sm_queues: list[list[Instruction]] = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        sm_queues[i % sm_count].append(instruction)
    return sm_queues


def zig_zag_assign_to_sms(instructions: list[Instruction], sm_count: int) -> list[list[Instruction]]:
    sm_queues: list[list[Instruction]] = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        base_id = i % (sm_count * 2)
        if base_id < sm_count:
            sm_queues[base_id].append(instruction)
        else:
            sm_queues[sm_count - 1 - (base_id - sm_count)].append(instruction)
    return sm_queues


def collect_into_waves(
    instructions: list[Instruction],
) -> list[list[Instruction]]:
    waves: list[list[Instruction]] = []
    cur: list[Instruction] = []
    for instruction in instructions:
        if not cur or cur[-1].opcode() == instruction.opcode():
            cur.append(instruction)
        else:
            waves.append(cur)
            cur = [instruction]
    if cur:
        waves.append(cur)
    return waves


def wave_assign_to_sms(schedule: Schedule) -> list[list[Instruction]]:
    instructions = schedule.get_linear_instructions()
    globs = schedule.globs
    sm_count = globs.sm_count()

    waves = collect_into_waves(instructions)

    sm_queues: list[list[Instruction]] = [[] for _ in range(sm_count)]
    sm_heap: list[tuple[float, int]] = [(0, i) for i in range(sm_count)]
    heapq.heapify(sm_heap)

    for wave in waves:
        sorted_by_cost = sorted(wave, key=lambda x: x.cost(globs), reverse=True)
        for ins in sorted_by_cost:
            sm_cost, sm_idx = heapq.heappop(sm_heap)
            sm_cost += ins.cost(globs)
            heapq.heappush(sm_heap, (sm_cost, sm_idx))
            sm_queues[sm_idx].append(ins)

    return sm_queues


def pool_assign_to_sms(
    instructions: list[Instruction], sm_count: int, memory_fraction: float
) -> list[list[Instruction]]:
    memory_instructions = []
    compute_instructions = []

    for ins in instructions:
        pool = ins.tags().get("pool")
        if pool == "memory":
            memory_instructions.append(ins)
        elif pool == "compute":
            compute_instructions.append(ins)
        else:
            raise ValueError(f"Unknown pool: {pool}")

    mem_sms = round(sm_count * memory_fraction)
    compute_sms = sm_count - mem_sms

    memory_queues = round_robin_assign_to_sms(memory_instructions, mem_sms)
    compute_queues = round_robin_assign_to_sms(compute_instructions, compute_sms)

    return memory_queues + compute_queues


def assign_to_sms(
    mode: str,
    schedule: Schedule | None = None,
    instructions: list[Instruction] | None = None,
    sm_count: int | None = None,
    memory_fraction: float | None = None,
) -> list[list[Instruction]]:
    if schedule is not None:
        instructions = schedule.get_linear_instructions()
        sm_count = schedule.globs.sm_count()

    assert instructions is not None
    assert sm_count is not None

    if mode == "rr":
        return round_robin_assign_to_sms(instructions, sm_count)
    elif mode == "zz":
        return zig_zag_assign_to_sms(instructions, sm_count)
    elif mode == "wave":
        assert schedule is not None
        return wave_assign_to_sms(schedule)
    elif mode == "dag":
        assert schedule is not None
        return assign_dag_to_sms(schedule)
    elif mode == "pool":
        assert memory_fraction is not None
        return pool_assign_to_sms(instructions, sm_count, memory_fraction=memory_fraction)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Tensorization — serialize instructions for the CUDA VM
# ---------------------------------------------------------------------------


def serialize_and_pad(instruction: Instruction) -> list[int]:
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0, f"Instruction too large: {len(serialized)} > {INTS_PER_INSTRUCTION}"
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: BaseGlobals,
    instruction_queues: list[list[Instruction]],
):
    num_sms = globs.sm_count()

    max_queue_len = max(len(queue) for queue in instruction_queues)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))

    flattened = []
    for queue in instruction_queues:
        flattened.extend(serialize_and_pad(instruction) for instruction in queue)

    device = globs.device

    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(num_sms, -1, INTS_PER_INSTRUCTION)

    timings = torch.zeros(
        [num_sms, max_queue_len, TIMING_SLOTS],
        dtype=torch.int32,
        device=device,
    )

    globs.instructions = serialized
    globs.timings = timings


# ===========================================================================
# Qwen3-1.7B — instruction types, globals, DAG builder, Python VM
# ===========================================================================

# ---------------------------------------------------------------------------
# Architecture constants (must match qwen3.cuh / QWEN3_1_7B)
# ---------------------------------------------------------------------------

NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 2
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6

# QKV concatenated output dim: (16 + 2×8) × 128 = 4096
QKV_OUT_DIM = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM  # 4096

# Block sizes: rows of the weight matrix each SM instruction handles.
QKV_BLOCK_SIZE = 16  # 4096/16  = 256 blocks → 1.73/SM
O_PROJ_BLOCK_SIZE = 8  # 2048/8   = 256 blocks → 1.73/SM
UPGATE_BLOCK_SIZE = 16  # 6144/16  = 384 blocks → 2.59/SM
DOWN_BLOCK_SIZE = 8  # 2048/8   = 256 blocks → 1.73/SM (×3 col-splits → 768 jobs)
LM_HEAD_BLOCK_SIZE = 16  # 151936/16 = 9496 blocks → 64.2/SM


def _pick_attn_partitions(seq_len: int) -> int:
    """Number of KV-sequence partitions for flash-decode on B200.

    Each partition covers ≥256 KV tokens.  Capped at 24 (reduction tree limit).
    """
    min_chunk = 256
    return min(24, max(1, math.ceil(seq_len / min_chunk)))


# ---------------------------------------------------------------------------
# Qwen3Globals
# ---------------------------------------------------------------------------


@dataclass
class Qwen3Globals(BaseGlobals):
    """All model state needed by the Qwen3 CUDA VM.

    Extends BaseGlobals with:
    - Per-head Q/K RMSNorm weights (Qwen3-specific, no equivalent in LLaMA)
    - Runtime activation buffers for the decode step
    - Block-size knobs controlling SM workload granularity
    """

    # Qwen3-specific per-head RMSNorm weights (stacked across layers)
    q_norm_weights: Tensor  # [num_layers, head_dim]   float32
    k_norm_weights: Tensor  # [num_layers, head_dim]   float32

    # Runtime activation buffers (pre-allocated, reused every decode step)
    post_qk_norm_q: Tensor  # [hidden_size]
    attn_out_intermediates: Tensor  # [num_q_heads, max_attn_parts, head_dim]
    attn_lse_intermediates: Tensor  # [num_q_heads, max_attn_parts]
    attn_out: Tensor  # [hidden_size]
    silu_out: Tensor  # [intermediate_size]
    logits: Tensor  # [vocab_size]

    # Block-size knobs (constant after construction)
    qkv_block_size: int = QKV_BLOCK_SIZE
    o_proj_block_size: int = O_PROJ_BLOCK_SIZE
    up_gate_block_size: int = UPGATE_BLOCK_SIZE
    down_proj_block_size: int = DOWN_BLOCK_SIZE
    lm_head_block_size: int = LM_HEAD_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Qwen3 instruction types (opcodes 1–8)
# ---------------------------------------------------------------------------


@dataclass
class QKV_LayerNorm_RoPE_Append(Instruction):
    """RMSNorm(hidden) → QKV GEMV → Q-RMSNorm + K-RMSNorm → RoPE → KV-cache append."""

    layer_idx: int
    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProj_Residual.opcode()

    def block_indices(self) -> list[int]:
        return list(range(self.start_output_block_idx, self.end_output_block_idx))

    def cost(self, globs: Qwen3Globals) -> float:
        n_blocks = self.end_output_block_idx - self.start_output_block_idx
        return n_blocks * globs.qkv_block_size * globs.hidden_size


@dataclass
class FlashDecode_GQA(Instruction):
    """Partial flash-decode attention for one KV head.

    GQA_RATIO=2 Q-heads are computed per KV head.
    When num_partials=1 this is the terminal instruction (no reduction needed).
    """

    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return QKV_LayerNorm_RoPE_Append.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        seq_len = globs.pos_id + 1
        chunk_len = seq_len / self.num_partials
        return chunk_len * globs.head_dim * 2  # K + V loads


@dataclass
class AttentionReduction(Instruction):
    """Log-sum-exp merge of partial flash-decode outputs."""

    layer_idx: int
    head_start_idx: int
    num_partials: int
    is_terminal: bool
    reduction_list: list[int]
    output_partial_idx: Optional[int] = None

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> int:
        return FlashDecode_GQA.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        return len(self.reduction_list) * globs.head_dim * GQA_RATIO


@dataclass
class O_Proj_Residual(Instruction):
    """O-projection GEMV + residual add."""

    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> int:
        return AttentionReduction.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        n_blocks = self.end_block_idx - self.start_block_idx
        return n_blocks * globs.o_proj_block_size * globs.hidden_size


@dataclass
class MLP_LayerNorm_UpGate_SiLU(Instruction):
    """MLP RMSNorm → gate GEMV + up GEMV → fused SiLU-multiply."""

    layer_idx: int
    block_idxs: list[int]

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> int:
        return O_Proj_Residual.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        return len(self.block_idxs) * globs.up_gate_block_size * globs.hidden_size * 2


@dataclass
class DownProj_Residual(Instruction):
    """Down-projection GEMV + residual add (col-split partitioned)."""

    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> int:
        return MLP_LayerNorm_UpGate_SiLU.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        n_blocks = self.end_block_idx - self.start_block_idx
        return n_blocks * globs.down_proj_block_size * globs.hidden_size


@dataclass
class RMSNorm_LM_Head(Instruction):
    """Final RMSNorm + LM-head GEMV."""

    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProj_Residual.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        n_blocks = self.end_output_block_idx - self.start_output_block_idx
        return n_blocks * globs.lm_head_block_size * globs.hidden_size


@dataclass
class QK_Norm_RoPE_Cache(Instruction):
    """Per-head Q/K RMSNorm + RoPE + KV-cache append.

    Runs on a single block after ALL QKV blocks complete.
    Applies per-head normalization, rotary embeddings, and writes KV cache.
    """

    layer_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> int:
        return QKV_LayerNorm_RoPE_Append.opcode()

    def cost(self, globs: Qwen3Globals) -> float:
        # Relatively cheap: per-head norms + RoPE + cache write
        return globs.hidden_size * 4


# ---------------------------------------------------------------------------
# Globals factory
# ---------------------------------------------------------------------------


def make_qwen3_globals(weights: dict, device: torch.device, max_seq_len: int = 4096) -> Qwen3Globals:
    """Build Qwen3Globals from the weight dict produced by Decoder.load_weights().

    Parameters
    ----------
    weights : dict
        Keys: attn_ln_ws, qkv_ws, q_norm_ws, k_norm_ws, o_proj_ws, mlp_ln_ws,
              gate_ws, up_ws, down_ws, k_caches, v_caches, cos_cache, sin_cache,
              norm_w, lm_head_w.
    device : torch.device
    max_seq_len : int  — for attention-partition sizing.
    """
    max_attn_parts = _pick_attn_partitions(max_seq_len)

    def buf(shape, dtype=torch.float32) -> Tensor:
        return torch.zeros(shape, device=device, dtype=dtype)

    barriers = buf([NUM_LAYERS, 10, NUM_Q_HEADS + NUM_KV_HEADS * 2], dtype=torch.int32)

    return Qwen3Globals(
        # BaseGlobals weight tensors
        qkv_proj_weights=weights["qkv_ws"],
        attn_ln_weights=weights["attn_ln_ws"],
        o_proj_weights=weights["o_proj_ws"],
        mlp_ln_weights=weights["mlp_ln_ws"],
        up_proj_weights=weights["up_ws"],
        gate_proj_weights=weights["gate_ws"],
        down_proj_weights=weights["down_ws"],
        lm_head_norm_weights=weights["norm_w"],
        lm_head_weights=weights["lm_head_w"],
        k_cache=weights["k_caches"],
        v_cache=weights["v_caches"],
        rope_cos=weights["cos_cache"],
        rope_sin=weights["sin_cache"],
        # Qwen3-specific Q/K-norm weights
        q_norm_weights=weights["q_norm_ws"],
        k_norm_weights=weights["k_norm_ws"],
        # activation buffers
        hidden_states=buf(HIDDEN_SIZE),
        post_qk_norm_q=buf(HIDDEN_SIZE),
        attn_out_intermediates=buf([NUM_Q_HEADS, max_attn_parts, HEAD_DIM]),
        attn_lse_intermediates=buf([NUM_Q_HEADS, max_attn_parts]),
        attn_out=buf(HIDDEN_SIZE),
        silu_out=buf(INTERMEDIATE_SIZE),
        logits=buf(VOCAB_SIZE),
        barriers=barriers,
        # model constants
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        vocab_size=VOCAB_SIZE,
        attn_scale=1.0 / math.sqrt(HEAD_DIM),
        rms_norm_eps=RMS_NORM_EPS,
        device=device,
        pos_id=0,
    )


# ---------------------------------------------------------------------------
# Per-operation scheduling helpers
# ---------------------------------------------------------------------------


def _distribute_blocks(num_blocks: int, sm_count: int) -> list[tuple[int, int]]:
    """Evenly distribute num_blocks across sm_count SMs; omit SMs that get 0."""
    return [
        (start, end)
        for sm_idx in range(sm_count)
        if (start := round(sm_idx * num_blocks / sm_count)) < (end := round((sm_idx + 1) * num_blocks / sm_count))
    ]


def _schedule_qkv(globs: Qwen3Globals, layer_idx: int) -> list[QKV_LayerNorm_RoPE_Append]:
    num_blocks = QKV_OUT_DIM // globs.qkv_block_size
    return [
        QKV_LayerNorm_RoPE_Append(layer_idx=layer_idx, start_output_block_idx=s, end_output_block_idx=e)
        for s, e in _distribute_blocks(num_blocks, globs.sm_count())
    ]


def _schedule_attention(
    globs: Qwen3Globals, layer_idx: int, num_partials: int
) -> tuple[list[FlashDecode_GQA], list[AttentionReduction]]:
    nkv = globs.num_kv_heads
    gqa = globs.num_attention_heads // nkv
    partials = [
        FlashDecode_GQA(layer_idx=layer_idx, kv_head_idx=kv_h, num_partials=num_partials, partial_idx=p)
        for kv_h in range(nkv)
        for p in range(num_partials)
    ]
    reductions = (
        [
            AttentionReduction(
                layer_idx=layer_idx,
                head_start_idx=kv_h * gqa,
                num_partials=num_partials,
                is_terminal=True,
                reduction_list=list(range(num_partials)),
            )
            for kv_h in range(nkv)
        ]
        if num_partials > 1
        else []
    )
    return partials, reductions


def _schedule_o_proj(globs: Qwen3Globals, layer_idx: int) -> list[O_Proj_Residual]:
    num_blocks = globs.hidden_size // globs.o_proj_block_size
    return [
        O_Proj_Residual(layer_idx=layer_idx, start_block_idx=s, end_block_idx=e, reduction_block_idx=0)
        for s, e in _distribute_blocks(num_blocks, globs.sm_count())
    ]


def _schedule_upgate(globs: Qwen3Globals, layer_idx: int) -> list[MLP_LayerNorm_UpGate_SiLU]:
    num_blocks = globs.intermediate_size // globs.up_gate_block_size
    sm_count = globs.sm_count()
    return [
        MLP_LayerNorm_UpGate_SiLU(layer_idx=layer_idx, block_idxs=list(range(sm_idx, num_blocks, sm_count)))
        for sm_idx in range(min(sm_count, num_blocks))
    ]


def _schedule_downproj(globs: Qwen3Globals, layer_idx: int) -> list[DownProj_Residual]:
    """col-split aware down-proj scheduling: 3 × 256 = 768 jobs on B200."""
    num_row_blocks = globs.hidden_size // globs.down_proj_block_size
    col_splits = globs.intermediate_size // globs.hidden_size
    sm_count = globs.sm_count()

    all_jobs = [(col, row) for col in range(col_splits) for row in range(num_row_blocks)]

    instructions: list[DownProj_Residual] = []
    num_assigned = 0
    for sm_idx in range(sm_count):
        jobs_left = len(all_jobs) - num_assigned
        sms_left = sm_count - sm_idx
        raw = all_jobs[num_assigned : num_assigned + round(jobs_left / sms_left)]
        if not raw:
            break
        col = raw[0][0]
        same = [j for j in raw if j[0] == col]
        instructions.append(
            DownProj_Residual(
                layer_idx=layer_idx,
                start_block_idx=same[0][1],
                end_block_idx=same[-1][1] + 1,
                reduction_block_idx=col,
            )
        )
        num_assigned += len(same)

    return instructions


def _schedule_downproj_vm(globs: Qwen3Globals, layer_idx: int) -> list[DownProj_Residual]:
    """Simple down-proj scheduling without col-splits for the VM kernel.

    Unlike _schedule_downproj (which creates 3× col-split jobs for the grid-strided
    kernel), this version creates one job per row block.  The VM kernel's
    vm_exec_downproj_residual always does the full GEMV across all columns, so
    col-splitting would triple-count the residual.
    """
    num_blocks = globs.hidden_size // globs.down_proj_block_size
    return [
        DownProj_Residual(
            layer_idx=layer_idx,
            start_block_idx=s,
            end_block_idx=e,
            reduction_block_idx=0,
        )
        for s, e in _distribute_blocks(num_blocks, globs.sm_count())
    ]


def _schedule_lm_head(globs: Qwen3Globals) -> list[RMSNorm_LM_Head]:
    num_blocks = globs.vocab_size // globs.lm_head_block_size
    return [
        RMSNorm_LM_Head(start_output_block_idx=s, end_output_block_idx=e)
        for s, e in _distribute_blocks(num_blocks, globs.sm_count())
    ]


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------


def _make_dag_layer(
    globs: Qwen3Globals,
    layer_idx: int,
    prev_layer_outputs: list[DAG_Node],
    seq_len: int,
    stop_after_op: str | None = None,
) -> tuple[list[DAG_Node], list[DAG_Node]]:
    """Build the computation DAG for one Qwen3 transformer layer.

    Dependency graph:
        prev_layer → QKV
            ↓  (per-KV-head fine-grained: waits only for that head's K+V blocks)
        FlashDecode_GQA
            ↓  (if num_partials > 1)
        AttentionReduction
            ↓
        O_Proj_Residual
            ↓
        MLP_LayerNorm_UpGate_SiLU
            ↓
        DownProj_Residual  →  next layer's prev_layer_outputs
    """
    num_partials = _pick_attn_partitions(seq_len)
    new_nodes: list[DAG_Node] = []

    # QKV
    qkv_insts = _schedule_qkv(globs, layer_idx)
    qkv_nodes = [DAG_Node(ins, prev_layer_outputs) for ins in qkv_insts]
    new_nodes.extend(qkv_nodes)

    # block → node map for fine-grained attention deps
    block_to_node: dict[int, DAG_Node] = {}
    for node in qkv_nodes:
        ins: QKV_LayerNorm_RoPE_Append = node.instruction  # type: ignore[assignment]
        for b in ins.block_indices():
            block_to_node[b] = node

    if stop_after_op == "qkv":
        return new_nodes, qkv_nodes

    # Attention partials
    partial_insts, reduction_insts = _schedule_attention(globs, layer_idx, num_partials)
    partial_nodes: list[DAG_Node] = []
    bps = globs.qkv_block_size
    for ins in partial_insts:
        fd: FlashDecode_GQA = ins  # type: ignore[assignment]
        kv_h = fd.kv_head_idx
        k_b0 = (NUM_Q_HEADS + kv_h) * HEAD_DIM // bps
        v_b0 = (NUM_Q_HEADS + NUM_KV_HEADS + kv_h) * HEAD_DIM // bps
        n_b = HEAD_DIM // bps
        deps = list(
            {block_to_node[b] for b in range(k_b0, k_b0 + n_b) if b in block_to_node}
            | {block_to_node[b] for b in range(v_b0, v_b0 + n_b) if b in block_to_node}
        )
        partial_nodes.append(DAG_Node(ins, deps))
    new_nodes.extend(partial_nodes)

    if stop_after_op == "partial":
        return new_nodes, partial_nodes

    # Reduction (or pass-through when num_partials==1)
    if reduction_insts:
        red_nodes = [DAG_Node(ins, partial_nodes) for ins in reduction_insts]
        new_nodes.extend(red_nodes)
        attn_out_nodes = red_nodes
    else:
        attn_out_nodes = partial_nodes

    if stop_after_op == "attn":
        return new_nodes, attn_out_nodes

    # O-projection
    o_nodes = [DAG_Node(ins, attn_out_nodes) for ins in _schedule_o_proj(globs, layer_idx)]
    new_nodes.extend(o_nodes)
    if stop_after_op == "oproj":
        return new_nodes, o_nodes

    # Up+Gate+SiLU
    ug_nodes = [DAG_Node(ins, o_nodes) for ins in _schedule_upgate(globs, layer_idx)]
    new_nodes.extend(ug_nodes)
    if stop_after_op == "upgate":
        return new_nodes, ug_nodes

    # Down-proj
    dp_nodes = [DAG_Node(ins, ug_nodes) for ins in _schedule_downproj(globs, layer_idx)]
    new_nodes.extend(dp_nodes)

    return new_nodes, dp_nodes


def make_qwen3_dag(
    globs: Qwen3Globals,
    seq_len: int = 1,
    stop_after_op: str | None = None,
    layer_limit: int | None = None,
) -> tuple[list[DAG_Node], DAG_Node]:
    """Build the full Qwen3-1.7B computation DAG.

    Parameters
    ----------
    seq_len : int
        Current KV-cache length; controls attention partition count
        (more partitions at longer context → lower per-partition cost).
    stop_after_op : str | None
        Truncate after 'qkv', 'partial', 'attn', 'oproj', 'upgate', or 'downproj'.
    layer_limit : int | None
        Run fewer than 28 layers (for profiling).
    """
    nlayers = layer_limit if layer_limit is not None else NUM_LAYERS
    all_nodes: list[DAG_Node] = []
    prev_outputs: list[DAG_Node] = []

    for layer_idx in range(nlayers):
        layer_nodes, prev_outputs = _make_dag_layer(globs, layer_idx, prev_outputs, seq_len, stop_after_op)
        all_nodes.extend(layer_nodes)

    if nlayers == NUM_LAYERS and stop_after_op is None:
        lm_nodes = [DAG_Node(ins, prev_outputs) for ins in _schedule_lm_head(globs)]
        all_nodes.extend(lm_nodes)
        prev_outputs = lm_nodes

    return all_nodes, DAG_Node(NoOp(), prev_outputs)


# ---------------------------------------------------------------------------
# VM-specific DAG — splits QKV into GEMV + separate QK_Norm_RoPE_Cache
# ---------------------------------------------------------------------------


def _make_vm_dag_layer(
    globs: Qwen3Globals,
    layer_idx: int,
    prev_layer_outputs: list[DAG_Node],
    seq_len: int,
) -> tuple[list[DAG_Node], list[DAG_Node]]:
    """Build one layer's DAG for the VM kernel.

    Key difference from _make_dag_layer: QKV GEMV and Q/K norm+RoPE+cache
    are separate instructions (opcodes 1 and 8), allowing the norm to run
    on a single block after all QKV blocks complete.
    """
    num_partials = _pick_attn_partitions(seq_len)
    new_nodes: list[DAG_Node] = []

    # QKV GEMV blocks
    qkv_insts = _schedule_qkv(globs, layer_idx)
    qkv_nodes = [DAG_Node(ins, prev_layer_outputs) for ins in qkv_insts]
    new_nodes.extend(qkv_nodes)

    # QK norm + RoPE + KV cache (single instruction, depends on ALL QKV blocks)
    norm_inst = QK_Norm_RoPE_Cache(layer_idx=layer_idx)
    norm_node = DAG_Node(norm_inst, qkv_nodes)
    new_nodes.append(norm_node)

    # FlashDecode (depends on norm/rope/cache, not just QKV GEMV)
    partial_insts, reduction_insts = _schedule_attention(globs, layer_idx, num_partials)
    partial_nodes = [DAG_Node(ins, [norm_node]) for ins in partial_insts]
    new_nodes.extend(partial_nodes)

    if reduction_insts:
        red_nodes = [DAG_Node(ins, partial_nodes) for ins in reduction_insts]
        new_nodes.extend(red_nodes)
        attn_out_nodes = red_nodes
    else:
        attn_out_nodes = partial_nodes

    # O-proj
    o_nodes = [DAG_Node(ins, attn_out_nodes) for ins in _schedule_o_proj(globs, layer_idx)]
    new_nodes.extend(o_nodes)

    # Up+Gate+SiLU
    ug_nodes = [DAG_Node(ins, o_nodes) for ins in _schedule_upgate(globs, layer_idx)]
    new_nodes.extend(ug_nodes)

    # Down-proj (no col-splits — VM handler does full GEMV per row)
    dp_nodes = [DAG_Node(ins, ug_nodes) for ins in _schedule_downproj_vm(globs, layer_idx)]
    new_nodes.extend(dp_nodes)

    return new_nodes, dp_nodes


def make_qwen3_vm_dag(
    globs: Qwen3Globals,
    seq_len: int = 1,
    layer_limit: int | None = None,
) -> tuple[list[DAG_Node], DAG_Node]:
    """Build the full Qwen3-1.7B VM computation DAG."""
    nlayers = layer_limit if layer_limit is not None else NUM_LAYERS
    all_nodes: list[DAG_Node] = []
    prev_outputs: list[DAG_Node] = []

    for layer_idx in range(nlayers):
        layer_nodes, prev_outputs = _make_vm_dag_layer(globs, layer_idx, prev_outputs, seq_len)
        all_nodes.extend(layer_nodes)

    if nlayers == NUM_LAYERS:
        lm_nodes = [DAG_Node(ins, prev_outputs) for ins in _schedule_lm_head(globs)]
        all_nodes.extend(lm_nodes)
        prev_outputs = lm_nodes

    return all_nodes, DAG_Node(NoOp(), prev_outputs)


# ---------------------------------------------------------------------------
# Barrier ID assignment for VM kernel
# ---------------------------------------------------------------------------

# Barrier layout per layer (7 barriers):
#   layer * 7 + 0 = qkv_done
#   layer * 7 + 1 = qknorm_done
#   layer * 7 + 2 = flash_done      (FlashDecode signals here)
#   layer * 7 + 3 = attn_done       (AttnReduction signals here, or OProj waits on flash_done if no reduction)
#   layer * 7 + 4 = oproj_done
#   layer * 7 + 5 = upgate_done
#   layer * 7 + 6 = downproj_done
# Total: NUM_LAYERS * 7 (no extra for LM head — it just waits on last downproj)

BARRIERS_PER_LAYER = 7


def _barrier_id(layer: int, phase: int) -> int:
    return layer * BARRIERS_PER_LAYER + phase


def assign_vm_barriers(
    sm_queues: list[list[Instruction]],
    globs: Qwen3Globals,
    seq_len: int = 1,
) -> tuple[list[list[list[int]]], int]:
    """Assign barrier IDs to serialized VM instructions.

    For each instruction in each SM queue, computes [wait_barrier_id, wait_count, signal_barrier_id].
    Returns (barrier_meta[sm][pc] -> [wait_bar, wait_count, signal_bar], total_barriers).

    Convention: barrier_id = -1 means no barrier.
    """
    # Count instructions per barrier group for expected arrival counts
    qkv_counts: dict[int, int] = {}  # layer -> count of QKV instructions
    flash_counts: dict[int, int] = {}  # layer -> count of FlashDecode only
    reduction_counts: dict[int, int] = {}  # layer -> count of AttnReduction only
    oproj_counts: dict[int, int] = {}  # layer -> count of O-proj
    upgate_counts: dict[int, int] = {}  # layer -> count of UpGate
    downproj_counts: dict[int, int] = {}  # layer -> count of DownProj

    for queue in sm_queues:
        for ins in queue:
            if isinstance(ins, NoOp):
                continue
            if isinstance(ins, QKV_LayerNorm_RoPE_Append):
                qkv_counts[ins.layer_idx] = qkv_counts.get(ins.layer_idx, 0) + 1
            elif isinstance(ins, FlashDecode_GQA):
                flash_counts[ins.layer_idx] = flash_counts.get(ins.layer_idx, 0) + 1
            elif isinstance(ins, AttentionReduction):
                reduction_counts[ins.layer_idx] = reduction_counts.get(ins.layer_idx, 0) + 1
            elif isinstance(ins, O_Proj_Residual):
                oproj_counts[ins.layer_idx] = oproj_counts.get(ins.layer_idx, 0) + 1
            elif isinstance(ins, MLP_LayerNorm_UpGate_SiLU):
                upgate_counts[ins.layer_idx] = upgate_counts.get(ins.layer_idx, 0) + 1
            elif isinstance(ins, DownProj_Residual):
                downproj_counts[ins.layer_idx] = downproj_counts.get(ins.layer_idx, 0) + 1

    total_barriers = NUM_LAYERS * BARRIERS_PER_LAYER
    barrier_meta: list[list[list[int]]] = []

    for queue in sm_queues:
        queue_meta: list[list[int]] = []
        for ins in queue:
            if isinstance(ins, NoOp):
                queue_meta.append([-1, 0, -1])
            elif isinstance(ins, QKV_LayerNorm_RoPE_Append):
                L = ins.layer_idx
                wait_bar = _barrier_id(L - 1, 6) if L > 0 else -1
                wait_count = downproj_counts.get(L - 1, 0) if L > 0 else 0
                signal_bar = _barrier_id(L, 0)
                queue_meta.append([wait_bar, wait_count, signal_bar])
            elif isinstance(ins, QK_Norm_RoPE_Cache):
                L = ins.layer_idx
                queue_meta.append([_barrier_id(L, 0), qkv_counts.get(L, 0), _barrier_id(L, 1)])
            elif isinstance(ins, FlashDecode_GQA):
                L = ins.layer_idx
                # FlashDecode signals flash_done (phase 2)
                queue_meta.append([_barrier_id(L, 1), 1, _barrier_id(L, 2)])
            elif isinstance(ins, AttentionReduction):
                L = ins.layer_idx
                # AttnReduction waits on flash_done (phase 2), signals attn_done (phase 3)
                queue_meta.append([_barrier_id(L, 2), flash_counts.get(L, 0), _barrier_id(L, 3)])
            elif isinstance(ins, O_Proj_Residual):
                L = ins.layer_idx
                # OProj waits on attn_done if reductions exist, else flash_done
                n_red = reduction_counts.get(L, 0)
                if n_red > 0:
                    queue_meta.append([_barrier_id(L, 3), n_red, _barrier_id(L, 4)])
                else:
                    queue_meta.append([_barrier_id(L, 2), flash_counts.get(L, 0), _barrier_id(L, 4)])
            elif isinstance(ins, MLP_LayerNorm_UpGate_SiLU):
                L = ins.layer_idx
                queue_meta.append([_barrier_id(L, 4), oproj_counts.get(L, 0), _barrier_id(L, 5)])
            elif isinstance(ins, DownProj_Residual):
                L = ins.layer_idx
                queue_meta.append([_barrier_id(L, 5), upgate_counts.get(L, 0), _barrier_id(L, 6)])
            elif isinstance(ins, RMSNorm_LM_Head):
                last_L = NUM_LAYERS - 1
                queue_meta.append([_barrier_id(last_L, 6), downproj_counts.get(last_L, 0), -1])
            else:
                queue_meta.append([-1, 0, -1])
        barrier_meta.append(queue_meta)

    return barrier_meta, total_barriers


def tensorize_vm_instructions(
    globs: Qwen3Globals,
    sm_queues: list[list[Instruction]],
    seq_len: int = 1,
) -> tuple[Tensor, Tensor, int]:
    """Serialize VM instructions with barrier metadata to GPU tensors.

    Returns (instructions, barrier_buf, queue_len).
    instructions: [num_sms, max_queue_len, 32] int32
    barrier_buf:  [num_barriers] int32 (zeroed)
    """
    barrier_meta, num_barriers = assign_vm_barriers(sm_queues, globs, seq_len)
    num_sms = len(sm_queues)

    max_queue_len = max(len(queue) for queue in sm_queues)
    # Pad all queues to same length
    for queue in sm_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))
    # Pad barrier_meta too
    for meta in barrier_meta:
        meta.extend([[-1, 0, -1]] * (max_queue_len - len(meta)))

    flattened = []
    for sm_idx, queue in enumerate(sm_queues):
        for pc, instruction in enumerate(queue):
            words = serialize_and_pad(instruction)
            # Embed barrier info at words 29-31
            bm = barrier_meta[sm_idx][pc]
            words[29] = bm[0]  # wait_barrier_id
            words[30] = bm[1]  # wait_count
            words[31] = bm[2]  # signal_barrier_id
            flattened.append(words)

    device = globs.device
    instructions = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, max_queue_len, INTS_PER_INSTRUCTION
    )
    barrier_buf = torch.zeros(num_barriers, dtype=torch.int32, device=device)

    return instructions, barrier_buf, max_queue_len


# ---------------------------------------------------------------------------
# ScheduleBuilder
# ---------------------------------------------------------------------------


class Qwen3LatencyScheduleBuilder(ScheduleBuilder):
    """Latency-optimised schedule builder for Qwen3-1.7B at BS=1.

    Usage
    -----
    ::

        from src.python.scheduler import Qwen3LatencyScheduleBuilder, tensorize_instructions

        schedule = Qwen3LatencyScheduleBuilder.build(weights, device=device, seq_len=1)

        # DAG-critical-path-aware assignment (best for latency)
        sm_queues = schedule.smart_assign_to_sms()

        # Serialize to GPU tensors for the CUDA VM
        tensorize_instructions(schedule.globs, sm_queues)
        # → schedule.globs.instructions : [148, max_q_len, 32]  int32
        # → schedule.globs.timings      : [148, max_q_len, 128] int32
    """

    @classmethod
    def make_globals(cls, model_or_weights, device=None, max_seq_len: int = 4096) -> Qwen3Globals:
        if isinstance(model_or_weights, dict):
            assert device is not None, "device required when passing a weights dict"
            return make_qwen3_globals(model_or_weights, device, max_seq_len)
        dec = model_or_weights
        return make_qwen3_globals(dec.weights, dec.device, max_seq_len)

    @classmethod
    def make_dag(
        cls,
        globs: Qwen3Globals,
        stop_after_op: str | None = None,
        layer_limit: int | None = None,
        seq_len: int = 1,
    ) -> tuple[list[DAG_Node], DAG_Node]:
        return make_qwen3_dag(globs, seq_len=seq_len, stop_after_op=stop_after_op, layer_limit=layer_limit)

    @classmethod
    def build(cls, model_or_weights, device=None, seq_len: int = 1, **kwargs) -> Schedule:
        globs = cls.make_globals(model_or_weights, device=device)
        dag_nodes, end_node = cls.make_dag(globs, seq_len=seq_len, **kwargs)
        return Schedule(globs, dag_nodes, end_node)


# ---------------------------------------------------------------------------
# Python VM — pure-PyTorch reference for correctness testing
# ---------------------------------------------------------------------------


class Qwen3PyVM:
    """Interprets Qwen3 instructions in Python/PyTorch.

    Allows diff-testing the scheduled VM path against the CUDA megakernel.

    Usage
    -----
    ::

        vm = Qwen3PyVM(globs)
        vm.run(schedule.get_linear_instructions())
        logits = globs.logits
    """

    def __init__(self, globs: Qwen3Globals):
        self.g = globs

    @staticmethod
    def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        return x.float() * rms * weight.float()

    def _exec_qkv(self, ins: QKV_LayerNorm_RoPE_Append):
        g = self.g
        normed = self._rms_norm(g.hidden_states, g.attn_ln_weights[ins.layer_idx], g.rms_norm_eps)
        start = ins.start_output_block_idx * g.qkv_block_size
        end = ins.end_output_block_idx * g.qkv_block_size
        W = g.qkv_proj_weights[ins.layer_idx].view(QKV_OUT_DIM, HIDDEN_SIZE)
        qkv_slice = W[start:end] @ normed

        q_rows = NUM_Q_HEADS * HEAD_DIM
        k_rows = NUM_KV_HEADS * HEAD_DIM
        pos = g.pos_id

        if start < q_rows:
            q_end = min(end, q_rows)
            g.post_qk_norm_q[start:q_end] = qkv_slice[: q_end - start]

        if start < q_rows + k_rows and end > q_rows:
            k_start = max(start, q_rows)
            k_end = min(end, q_rows + k_rows)
            g.k_cache[ins.layer_idx, pos, k_start - q_rows : k_end - q_rows] = qkv_slice[
                max(0, q_rows - start) : max(0, q_rows - start) + (k_end - k_start)
            ]

        if end > q_rows + k_rows:
            v_start = max(start, q_rows + k_rows)
            v_off = v_start - q_rows - k_rows
            g.v_cache[ins.layer_idx, pos, v_off : v_off + (end - v_start)] = qkv_slice[v_start - start :]

    def _exec_flash_decode(self, ins: FlashDecode_GQA):
        g = self.g
        seq_len = g.pos_id + 1
        chunk = math.ceil(seq_len / ins.num_partials)
        kv_start = ins.partial_idx * chunk
        kv_end = min(kv_start + chunk, seq_len)
        kv_off = ins.kv_head_idx * HEAD_DIM

        for gqa_i in range(GQA_RATIO):
            q_head = ins.kv_head_idx * GQA_RATIO + gqa_i
            q = g.post_qk_norm_q[q_head * HEAD_DIM : (q_head + 1) * HEAD_DIM]
            m, d = float("-inf"), 0.0
            o = torch.zeros(HEAD_DIM, device=g.device)

            for pos in range(kv_start, kv_end):
                k = g.k_cache[ins.layer_idx, pos, kv_off : kv_off + HEAD_DIM]
                score = (q * k).sum().item() * g.attn_scale
                m_new = max(m, score)
                alpha, beta = math.exp(m - m_new), math.exp(score - m_new)
                d = alpha * d + beta
                o = alpha * o + beta * g.v_cache[ins.layer_idx, pos, kv_off : kv_off + HEAD_DIM]
                m = m_new

            if ins.num_partials == 1:
                g.attn_out[q_head * HEAD_DIM : (q_head + 1) * HEAD_DIM] = o / d
            else:
                g.attn_out_intermediates[q_head, ins.partial_idx] = o
                g.attn_lse_intermediates[q_head, ins.partial_idx] = m + math.log(d)

    def _exec_attn_reduction(self, ins: AttentionReduction):
        g = self.g
        for gqa_i in range(GQA_RATIO):
            qh = ins.head_start_idx + gqa_i
            lses = g.attn_lse_intermediates[qh, ins.reduction_list]
            outs = g.attn_out_intermediates[qh, ins.reduction_list]
            w = (lses - lses.max()).exp()
            w /= w.sum()
            g.attn_out[qh * HEAD_DIM : (qh + 1) * HEAD_DIM] = (outs * w.unsqueeze(-1)).sum(0)

    def _exec_o_proj(self, ins: O_Proj_Residual):
        g = self.g
        start = ins.start_block_idx * g.o_proj_block_size
        end = ins.end_block_idx * g.o_proj_block_size
        W = g.o_proj_weights[ins.layer_idx].view(HIDDEN_SIZE, HIDDEN_SIZE)
        g.hidden_states[start:end] += W[start:end] @ g.attn_out

    def _exec_upgate_silu(self, ins: MLP_LayerNorm_UpGate_SiLU):
        g = self.g
        normed = self._rms_norm(g.hidden_states, g.mlp_ln_weights[ins.layer_idx], g.rms_norm_eps)
        bs = g.up_gate_block_size
        G = g.gate_proj_weights[ins.layer_idx].view(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        U = g.up_proj_weights[ins.layer_idx].view(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        for b in ins.block_idxs:
            rows = slice(b * bs, (b + 1) * bs)
            gate = G[rows] @ normed
            g.silu_out[rows] = gate * torch.sigmoid(gate) * (U[rows] @ normed)

    def _exec_downproj(self, ins: DownProj_Residual):
        g = self.g
        bs = g.down_proj_block_size
        col = ins.reduction_block_idx
        col_size = HIDDEN_SIZE
        D = g.down_proj_weights[ins.layer_idx].view(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        start = ins.start_block_idx * bs
        end = ins.end_block_idx * bs
        g.hidden_states[start:end] += (
            D[start:end, col * col_size : (col + 1) * col_size] @ g.silu_out[col * col_size : (col + 1) * col_size]
        )

    def _exec_lm_head(self, ins: RMSNorm_LM_Head):
        g = self.g
        normed = self._rms_norm(g.hidden_states, g.lm_head_norm_weights, g.rms_norm_eps)
        start = ins.start_output_block_idx * g.lm_head_block_size
        end = ins.end_output_block_idx * g.lm_head_block_size
        W = g.lm_head_weights.view(VOCAB_SIZE, HIDDEN_SIZE)
        g.logits[start:end] = W[start:end].float() @ normed.float()

    def _exec_qknorm_rope_cache(self, ins: QK_Norm_RoPE_Cache):
        """Apply per-head Q/K RMSNorm + RoPE + KV cache write."""
        g = self.g
        pos = g.pos_id

        # Q per-head RMSNorm
        for h in range(NUM_Q_HEADS):
            off = h * HEAD_DIM
            q_head = g.post_qk_norm_q[off : off + HEAD_DIM]
            g.post_qk_norm_q[off : off + HEAD_DIM] = self._rms_norm(
                q_head, g.q_norm_weights[ins.layer_idx], g.rms_norm_eps
            )

        # K per-head RMSNorm
        for h in range(NUM_KV_HEADS):
            off = h * HEAD_DIM
            k_head = g.k_cache[ins.layer_idx, pos, off : off + HEAD_DIM]
            g.k_cache[ins.layer_idx, pos, off : off + HEAD_DIM] = self._rms_norm(
                k_head, g.k_norm_weights[ins.layer_idx], g.rms_norm_eps
            )

        # RoPE (in-place on post_qk_norm_q and k_cache)
        half = HEAD_DIM // 2
        cos_pos = g.rope_cos[pos, :half]
        sin_pos = g.rope_sin[pos, :half]

        for h in range(NUM_Q_HEADS):
            off = h * HEAD_DIM
            x0 = g.post_qk_norm_q[off : off + half].clone()
            x1 = g.post_qk_norm_q[off + half : off + HEAD_DIM].clone()
            g.post_qk_norm_q[off : off + half] = x0 * cos_pos - x1 * sin_pos
            g.post_qk_norm_q[off + half : off + HEAD_DIM] = x1 * cos_pos + x0 * sin_pos

        for h in range(NUM_KV_HEADS):
            off = h * HEAD_DIM
            x0 = g.k_cache[ins.layer_idx, pos, off : off + half].clone()
            x1 = g.k_cache[ins.layer_idx, pos, off + half : off + HEAD_DIM].clone()
            g.k_cache[ins.layer_idx, pos, off : off + half] = x0 * cos_pos - x1 * sin_pos
            g.k_cache[ins.layer_idx, pos, off + half : off + HEAD_DIM] = x1 * cos_pos + x0 * sin_pos

    _HANDLERS = {
        QKV_LayerNorm_RoPE_Append: _exec_qkv,
        QK_Norm_RoPE_Cache: _exec_qknorm_rope_cache,
        FlashDecode_GQA: _exec_flash_decode,
        AttentionReduction: _exec_attn_reduction,
        O_Proj_Residual: _exec_o_proj,
        MLP_LayerNorm_UpGate_SiLU: _exec_upgate_silu,
        DownProj_Residual: _exec_downproj,
        RMSNorm_LM_Head: _exec_lm_head,
        NoOp: lambda self, ins: None,
    }

    def step(self, instruction: Instruction):
        handler = self._HANDLERS.get(type(instruction))
        if handler is None:
            raise ValueError(f"Unknown instruction: {type(instruction)}")
        handler(self, instruction)

    def run(self, instructions: list[Instruction]):
        for ins in instructions:
            self.step(ins)
