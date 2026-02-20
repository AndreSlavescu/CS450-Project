from __future__ import annotations

import heapq
from dataclasses import dataclass, field, fields, replace
from typing import Any

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
