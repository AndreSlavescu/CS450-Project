from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol

from ..utils import DeviceType, load_safetensors_repo

KV_Cache = tuple[Tensor, Tensor]


# ---------------------------------------------------------------------------
# Qwen3 config
# ---------------------------------------------------------------------------


@dataclass
class Qwen3Config:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    use_qk_norm: bool = False
    # MoE fields (None = dense model)
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    norm_topk_prob: bool = True

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None


# Pre-defined model configurations
QWEN3_1_7B = Qwen3Config()  # defaults are 1.7B

QWEN3_8B = Qwen3Config(
    hidden_size=4096,
    intermediate_size=12288,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    num_hidden_layers=36,
    vocab_size=151936,
    max_position_embeddings=40960,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)

QWEN3_CODER_480B = Qwen3Config(
    hidden_size=6144,
    intermediate_size=8192,
    num_attention_heads=96,
    num_key_value_heads=8,
    head_dim=128,
    num_hidden_layers=62,
    vocab_size=151936,
    max_position_embeddings=262144,
    rope_theta=10000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    use_qk_norm=True,
    num_experts=160,
    num_experts_per_tok=8,
    moe_intermediate_size=2560,
    norm_topk_prob=True,
)


@dataclass
class ExtraConfig:
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: dist.ProcessGroup | None = None
    ep_size: int = 1
    ep_rank: int = 0
    ep_group: dist.ProcessGroup | None = None
    torch_compile: bool = False
    max_batch_size: int = 1
    max_len_override: int | None = None


# ---------------------------------------------------------------------------
# Stacked parameters for efficient kernel access
# ---------------------------------------------------------------------------


@dataclass
class StackedParams:
    qkv_proj: Tensor
    o_proj: Tensor
    attn_ln_weight: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor


# ---------------------------------------------------------------------------
# TP + SP communication primitives
#
# For low-latency BS=1 decode, we use the Megatron-LM TP+SP pattern:
#   all_gather  before column-parallel linears  (SP layout -> full hidden)
#   reduce_scatter after row-parallel linears   (full hidden -> SP layout)
#
# Between TP regions (layernorm, residual add), activations stay in SP
# layout — each rank holds a 1/tp_size slice.  This halves activation
# memory vs. pure-TP all_reduce, and — critically — the split collectives
# can be pipelined with compute inside the megakernel scheduler.
#
# When tp_size == 1, both functions are identity (zero overhead).
# ---------------------------------------------------------------------------


def tp_all_gather(x: Tensor, extra: ExtraConfig) -> Tensor:
    if extra.tp_size == 1:
        return x
    assert extra.tp_group is not None
    if extra.torch_compile:
        return funcol.all_gather_tensor(x, gather_dim=0, group=extra.tp_group)
    out = torch.empty(
        (extra.tp_size * x.shape[0], *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    dist.all_gather_into_tensor(out, x, group=extra.tp_group)
    return out


def tp_reduce_scatter(x: Tensor, extra: ExtraConfig) -> Tensor:
    if extra.tp_size == 1:
        return x
    assert extra.tp_group is not None
    if extra.torch_compile:
        return funcol.reduce_scatter_tensor(x, reduceOp="sum", scatter_dim=0, group=extra.tp_group)
    out = torch.empty(
        (x.shape[0] // extra.tp_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    dist.reduce_scatter_tensor(out, x, group=extra.tp_group)
    return out


def tp_allreduce(x: Tensor, extra: ExtraConfig) -> Tensor:
    """In-place allreduce for row-parallel linears (no batch dim expansion)."""
    if extra.tp_size == 1:
        return x
    assert extra.tp_group is not None
    dist.all_reduce(x, group=extra.tp_group)
    return x


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, head_dim: int, max_position_embeddings: int, rope_theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # position_ids: (batch, seq_len) or (1, max_pos)
        # inv_freq: (head_dim // 2,)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, seq_len, head_dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 2
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    kv_cache: KV_Cache,
    position_ids: Tensor,
    seq_len: int,
) -> Tensor:
    k_cache, v_cache = kv_cache

    k_cache[:, position_ids] = key_states
    v_cache[:, position_ids] = value_states

    bsz, new_tok_len = query_states.shape[:2]

    # (b, l, h, d) -> (b, h, l, d) for SDPA
    q = query_states.transpose(1, 2)

    if new_tok_len > 1:
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    else:
        k = k_cache[:, :seq_len].transpose(1, 2)
        v = v_cache[:, :seq_len].transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)

    # (b, h, l, d) -> (b, l, h, d)
    return attn_output.transpose(1, 2)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

        tp_size = extra_config.tp_size
        assert config.num_attention_heads % tp_size == 0
        assert config.num_key_value_heads % tp_size == 0

        self.num_attention_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

        self.kv_cache: KV_Cache | None = None

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        seq_len: int,
    ) -> Tensor:
        assert self.kv_cache is not None

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Each rank already has the full hidden state (replicated for BS=1).
        # Column-parallel QKV: each rank projects to its local heads.
        bsz, slen = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states).view(bsz, slen, self.num_attention_heads, -1)
        key_states = self.k_proj(hidden_states).view(bsz, slen, self.num_kv_heads, -1)
        value_states = self.v_proj(hidden_states).view(bsz, slen, self.num_kv_heads, -1)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        dtype = query_states.dtype
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)

        attn_output = _attention(query_states, key_states, value_states, self.kv_cache, position_ids, seq_len)
        attn_output = attn_output.reshape(bsz, slen, -1)

        # Row-parallel O projection: each rank produces a partial sum, allreduce to combine
        o_proj = self.o_proj(attn_output)
        o_proj = tp_allreduce(o_proj, self.extra_config)

        return residual + o_proj


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        tp_size = extra_config.tp_size
        assert config.intermediate_size % tp_size == 0
        intermediate_size = config.intermediate_size // tp_size

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Column-parallel gate/up projections (each rank computes local slice)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        # Row-parallel down projection: partial sum, allreduce to combine
        down = self.down_proj(F.silu(gate) * up)
        down = tp_allreduce(down, self.extra_config)

        return residual + down


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts)
# ---------------------------------------------------------------------------


class Qwen3MoEExpert(nn.Module):
    """Single MoE expert — SwiGLU MLP with moe_intermediate_size."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        assert config.moe_intermediate_size is not None
        intermediate = config.moe_intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoELayer(nn.Module):
    """Replaces the dense MLP with a routed MoE layer.

    Each EP rank holds ``num_experts // ep_size`` local experts.
    After local expert execution the partial outputs are summed via
    ``all_reduce`` across the EP group.

    Uses tcgen05 persistent GEMM with sorted-expert-block dispatch
    for maximum TTFT on SM100 (B200).
    """

    _moe_kernel = None  # loaded once on first use

    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        assert config.num_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.num_experts % extra_config.ep_size == 0

        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Router (replicated on all ranks)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Only instantiate local experts for this EP rank
        self.experts_per_rank = config.num_experts // extra_config.ep_size
        self.local_expert_offset = extra_config.ep_rank * self.experts_per_rank
        self.experts = nn.ModuleList(
            [Qwen3MoEExpert(config, layer_idx) for _ in range(self.experts_per_rank)]
        )

        # Packed weight buffers for tcgen05 kernel (built lazily)
        self._gate_up_weights_packed: Tensor | None = None
        self._down_weights_packed: Tensor | None = None

    @classmethod
    def _load_moe_kernel(cls):
        """Load the compiled tcgen05 MoE kernel module."""
        if cls._moe_kernel is not None:
            return cls._moe_kernel
        import importlib
        cls._moe_kernel = importlib.import_module("moe_expert")
        assert cls._moe_kernel.has_tcgen05(), \
            "moe_expert module loaded but tcgen05 (SM100) not available"
        return cls._moe_kernel

    def _build_packed_weights(self) -> None:
        """Pack expert weights into contiguous buffers for the tcgen05 kernel.

        gate_up_weights: [experts_per_rank, 2*intermediate, hidden]
        down_weights:    [experts_per_rank, hidden, intermediate]
        """
        if self._gate_up_weights_packed is not None:
            return

        intermediate = self.config.moe_intermediate_size
        hidden = self.config.hidden_size

        gate_up = torch.empty(
            self.experts_per_rank, 2 * intermediate, hidden,
            dtype=torch.bfloat16,
            device=self.experts[0].gate_proj.weight.device,
        )
        down = torch.empty(
            self.experts_per_rank, hidden, intermediate,
            dtype=torch.bfloat16,
            device=self.experts[0].down_proj.weight.device,
        )

        for i, expert in enumerate(self.experts):
            gate_up[i, :intermediate] = expert.gate_proj.weight.data
            gate_up[i, intermediate:] = expert.up_proj.weight.data
            down[i] = expert.down_proj.weight.data

        self._gate_up_weights_packed = gate_up
        self._down_weights_packed = down

        # Free original expert weight tensors — they've been copied into the
        # packed buffers and are no longer needed.  Without this, expert
        # weights are stored twice (~109 GiB duplicate) causing OOM on B200.
        for expert in self.experts:
            expert.gate_proj = None
            expert.up_proj = None
            expert.down_proj = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        bsz, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        num_tokens = flat.shape[0]

        moe = self._load_moe_kernel()
        hidden_size = self.config.hidden_size
        intermediate = self.config.moe_intermediate_size

        self._build_packed_weights()

        # 1. Router + top-k + sort
        sorted_token_ids, sorted_weights, expert_offsets = moe.moe_router_topk(
            flat,
            self.gate.weight,
            self.num_experts_per_tok,
            self.norm_topk_prob,
            self.local_expert_offset,
            self.experts_per_rank,
        )

        total_assignments = sorted_token_ids.shape[0]
        if total_assignments == 0:
            expert_output = torch.zeros_like(flat)
        else:
            # 2. Gather sorted activations
            sorted_hidden = moe.moe_gather_tokens(flat, sorted_token_ids, hidden_size)

            # 3. Gate+Up GEMM (persistent tcgen05)
            gate_up_out = moe.moe_gate_up_gemm(
                sorted_hidden,
                self._gate_up_weights_packed,
                expert_offsets,
                hidden_size,
                2 * intermediate,
            )

            # 4. SiLU fusion
            intermediate_out = moe.moe_silu_fusion(gate_up_out, intermediate)

            # 5. Down GEMM (persistent tcgen05)
            down_out = moe.moe_down_gemm(
                intermediate_out,
                self._down_weights_packed,
                expert_offsets,
                hidden_size,
                intermediate,
            )

            # 6. Scatter-accumulate to original token order
            expert_output = moe.moe_scatter_accumulate(
                down_out, sorted_token_ids, sorted_weights,
                num_tokens, hidden_size,
            )

        # EP all-reduce: combine partial expert outputs across ranks
        if self.extra_config.ep_size > 1 and self.extra_config.ep_group is not None:
            dist.all_reduce(expert_output, group=self.extra_config.ep_group)

        return residual + expert_output.view(bsz, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class Qwen3MoEBlock(nn.Module):
    """Transformer block with MoE layer instead of dense MLP."""

    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, extra_config, layer_idx)
        self.mlp = Qwen3MoELayer(config, extra_config, layer_idx)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        seq_len: int,
    ) -> Tensor:
        hidden_states = self.self_attn(hidden_states, position_embeddings, position_ids, seq_len)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, extra_config, layer_idx)
        self.mlp = Qwen3MLP(config, extra_config, layer_idx)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        seq_len: int,
    ) -> Tensor:
        hidden_states = self.self_attn(hidden_states, position_embeddings, position_ids, seq_len)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Qwen3Model(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        BlockCls = Qwen3MoEBlock if config.is_moe else Qwen3Block
        self.layers = nn.ModuleList([BlockCls(config, extra_config, i) for i in range(config.num_hidden_layers)])

        rope_max_pos = extra_config.max_len_override or config.max_position_embeddings
        self.rope = Qwen3RotaryEmbedding(config.head_dim, rope_max_pos, config.rope_theta)

        # Precompute RoPE embeddings for all positions
        position_ids = torch.arange(rope_max_pos).unsqueeze(0)
        cos, sin = self.rope(position_ids)
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_len: int,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)

        cos = self.rope_cos[position_ids]
        sin = self.rope_sin[position_ids]
        position_embeddings = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, position_ids, seq_len)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device_: DeviceType = torch.get_default_device()
        self.dtype_ = torch.get_default_dtype()

        self.model = Qwen3Model(config, extra_config)

        self.final_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, position_ids: Tensor, seq_len: int) -> Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        hidden_states = self.model(input_ids, position_ids, seq_len)

        # Each rank has the full hidden state (allreduce pattern, not SP)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def setup_caches(self):
        max_len = self.extra_config.max_len_override or self.config.max_position_embeddings
        tp_size = self.extra_config.tp_size
        num_kv_heads = self.config.num_key_value_heads // tp_size

        k_cache = torch.zeros(
            (
                self.config.num_hidden_layers,
                self.extra_config.max_batch_size,
                max_len,
                num_kv_heads,
                self.config.head_dim,
            ),
            device=self.device_,
            dtype=self.dtype_,
        )
        v_cache = k_cache.clone()

        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(self.config.num_hidden_layers):
            layer: Qwen3Block = self.model.layers[layer_idx]  # type: ignore
            layer.self_attn.kv_cache = (
                self.stacked_kv_cache[0][layer_idx],
                self.stacked_kv_cache[1][layer_idx],
            )

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):  # type: ignore
        if device is not None:
            self.device_ = device
        if dtype is not None:
            self.dtype_ = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ) -> Qwen3ForCausalLM:
        if extra_config is None:
            extra_config = ExtraConfig()
        if dtype is None:
            dtype = torch.bfloat16

        config = Qwen3Config()

        # Load config.json from HF to get actual model dimensions
        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot_path_str = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )
            model_path = Path(snapshot_path_str)

        config_json_path = model_path / "config.json"
        if config_json_path.exists():
            import json

            with open(config_json_path) as f:
                hf_config = json.load(f)
            config.hidden_size = hf_config.get("hidden_size", config.hidden_size)
            config.intermediate_size = hf_config.get("intermediate_size", config.intermediate_size)
            config.num_attention_heads = hf_config.get("num_attention_heads", config.num_attention_heads)
            config.num_key_value_heads = hf_config.get("num_key_value_heads", config.num_key_value_heads)
            config.head_dim = hf_config.get("head_dim", config.head_dim)
            config.num_hidden_layers = hf_config.get("num_hidden_layers", config.num_hidden_layers)
            config.vocab_size = hf_config.get("vocab_size", config.vocab_size)
            config.max_position_embeddings = hf_config.get("max_position_embeddings", config.max_position_embeddings)
            config.rope_theta = hf_config.get("rope_theta", config.rope_theta)
            config.rms_norm_eps = hf_config.get("rms_norm_eps", config.rms_norm_eps)
            config.tie_word_embeddings = hf_config.get("tie_word_embeddings", config.tie_word_embeddings)
            config.use_qk_norm = hf_config.get("use_qk_norm", config.use_qk_norm)
            # MoE fields
            if "num_experts" in hf_config:
                config.num_experts = hf_config["num_experts"]
                config.num_experts_per_tok = hf_config.get("num_experts_per_tok", config.num_experts_per_tok)
                config.moe_intermediate_size = hf_config.get("moe_intermediate_size", config.moe_intermediate_size)
                config.norm_topk_prob = hf_config.get("norm_topk_prob", config.norm_topk_prob)

        with init_empty_weights(include_buffers=False):
            model = cls(config, extra_config)
        model.dtype_ = dtype
        model.device_ = device

        model.load_from_safetensors(model_path)

        # Move to device without converting RoPE inv_freq to fp16
        model.to(device=device)

        model.requires_grad_(False)
        if not config.is_moe:
            model.stack_params()
        else:
            # Eagerly pack MoE expert weights and free the originals so we
            # don't hold both copies in GPU memory simultaneously.
            for layer in model.model.layers:
                layer.mlp._build_packed_weights()
        model.setup_caches()

        return model

    def make_name_to_hf_name(self) -> dict[str, str]:
        keys = self.state_dict().keys()
        name_to_hf_name = {k: k for k in keys}

        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[f"model.layers.{layer_idx}.self_attn.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.input_layernorm.weight"
            )
            name_to_hf_name[f"model.layers.{layer_idx}.mlp.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )

            # MoE: map local expert indices to global HF expert indices
            if self.config.is_moe:
                experts_per_rank = self.config.num_experts // self.extra_config.ep_size
                local_offset = self.extra_config.ep_rank * experts_per_rank
                for local_idx in range(experts_per_rank):
                    global_idx = local_offset + local_idx
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        our = f"model.layers.{layer_idx}.mlp.experts.{local_idx}.{proj}.weight"
                        hf = f"model.layers.{layer_idx}.mlp.experts.{global_idx}.{proj}.weight"
                        name_to_hf_name[our] = hf

        name_to_hf_name["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        name_to_hf_name["final_norm.weight"] = "model.norm.weight"

        if self.config.tie_word_embeddings:
            name_to_hf_name["lm_head.weight"] = "model.embed_tokens.weight"
        else:
            name_to_hf_name["lm_head.weight"] = "lm_head.weight"

        return name_to_hf_name

    def make_tp_map(self) -> dict[str, int]:
        tp_map: dict[str, int] = {}
        for param_name, _ in self.named_parameters():
            # MoE expert weights are EP-sharded, not TP-sharded
            if ".mlp.experts." in param_name:
                continue
            # MoE router is replicated (small), no sharding
            if ".mlp.gate.weight" in param_name and self.config.is_moe:
                continue
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
                ]
            ):
                tp_map[param_name] = 0
            elif any(param_name.endswith(suffix) for suffix in ["o_proj.weight", "down_proj.weight"]):
                tp_map[param_name] = 1
        return tp_map

    def load_from_safetensors(self, model_path: Path):
        name_to_hf_name = self.make_name_to_hf_name()
        all_hf_names = set(name_to_hf_name.values())

        hf_tp_map: dict[str, int] = {}
        our_tp_map = self.make_tp_map()
        for our_name, hf_name in name_to_hf_name.items():
            if our_name in our_tp_map:
                hf_tp_map[hf_name] = our_tp_map[our_name]

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device_,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=hf_tp_map,
        )

        state_dict = {k: hf_state_dict[v] for k, v in name_to_hf_name.items()}
        self.load_state_dict(state_dict, assign=True, strict=True)

    def stack_params(self):
        def stack_and_reassign(modules, prop: str) -> Tensor:
            params = [getattr(m, prop) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, prop)[:] = stacked[i]
            return stacked

        layers: list[Qwen3Block] = list(self.model.layers)  # type: ignore
        self_attns = [x.self_attn for x in layers]
        mlps = [x.mlp for x in layers]

        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]
        mlp_lns = [x.input_layernorm for x in mlps]
        up_projs = [x.up_proj for x in mlps]
        gate_projs = [x.gate_proj for x in mlps]
        down_projs = [x.down_proj for x in mlps]

        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [self_attn.q_proj.weight, self_attn.k_proj.weight, self_attn.v_proj.weight],
                dim=0,
            )
            qkv_weights.append(cat_weight)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

        num_q = self.config.num_attention_heads * self.config.head_dim
        num_kv = self.config.num_key_value_heads * self.config.head_dim
        for i, self_attn in enumerate(self_attns):
            q_weight, k_weight, v_weight = stacked_qkv_weights[i].split([num_q, num_kv, num_kv], dim=0)
            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight

        self.stacked_params = StackedParams(
            qkv_proj=stacked_qkv_weights,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            gate_proj=stacked_gate_proj,
            down_proj=stacked_down_proj,
        )
