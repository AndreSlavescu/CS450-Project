"""Stateful single-token decoder wrapping the Qwen3-1.7B persistent megakernel.

Architecture mirrors the AlpinDale qwen_megakernel pattern (model.py) but
targets Qwen3-1.7B and our qwen3_decode_persistent CUDA kernel.

Weight loading
--------------
Weights are loaded from HuggingFace in bfloat16, then cast to float32 (our
kernel works in fp32).  Stacking across layers matches the layout expected
by Qwen3LayerWeights in qwen3_kernels.cu:

    attn_ln_ws   [num_layers, hidden_size]
    qkv_ws       [num_layers, qkv_dim * hidden_size]   (Q|K|V concat, flattened)
    q_norm_ws    [num_layers, head_dim]
    k_norm_ws    [num_layers, head_dim]
    o_proj_ws    [num_layers, hidden_size * hidden_size]
    mlp_ln_ws    [num_layers, hidden_size]
    gate_ws      [num_layers, intermediate_size * hidden_size]
    up_ws        [num_layers, intermediate_size * hidden_size]
    down_ws      [num_layers, hidden_size * intermediate_size]
    k_caches     [num_layers, max_seq * kv_dim]
    v_caches     [num_layers, max_seq * kv_dim]

RoPE tables
-----------
The qkv_rope_append.cuh kernel uses only cos/sin[:half_head] from each row, so
we build tables of shape [max_seq, head_dim] where [:, :half_head] holds the
precomputed values and [:, half_head:] is zero-padded (unused by the kernel).

Usage
-----
    from src.python.Qwen3.decoder import Decoder

    dec = Decoder()                     # loads Qwen3-1.7B from HuggingFace
    dec.generate("Hello!", max_tokens=50)

    # or step by step:
    dec.reset()
    tok = dec.step(tokenizer.encode("Hello")[0])
"""

import torch

# ── Qwen3-1.7B constants (must match qwen3.cuh / QWEN3_1_7B struct) ──────────
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 2048
QKV_DIM = Q_DIM + 2 * KV_DIM  # 4096
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000.0  # Qwen3 uses 1M, not 10K
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_MAX_SEQ = 4096


# ── RoPE precomputation ───────────────────────────────────────────────────────


def _precompute_rope(max_seq_len: int, rope_theta: float = ROPE_THETA) -> tuple:
    """
    Returns (cos_table, sin_table) each of shape [max_seq_len, HEAD_DIM] float32.

    The kernel reads cos_cached[i] for i in [0, HEAD_DIM//2); the second half
    of each row is zero-padded and never accessed.
    """
    half = HEAD_DIM // 2  # 64
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))  # [64]
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq, 64]

    cos_table = torch.zeros(max_seq_len, HEAD_DIM, dtype=torch.float32)
    sin_table = torch.zeros(max_seq_len, HEAD_DIM, dtype=torch.float32)
    cos_table[:, :half] = torch.cos(freqs)
    sin_table[:, :half] = torch.sin(freqs)

    return cos_table.cuda().contiguous(), sin_table.cuda().contiguous()


# ── Weight loading ────────────────────────────────────────────────────────────


def load_weights(
    model_name: str = DEFAULT_MODEL,
    max_seq_len: int = DEFAULT_MAX_SEQ,
    verbose: bool = True,
) -> tuple:
    """
    Load Qwen3-1.7B weights from HuggingFace and return (weights_dict, tokenizer).

    All tensors are float32 on CUDA.  The HuggingFace model is deleted after
    weight extraction to free GPU memory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if verbose:
        print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    rope_theta = float(getattr(model.config, "rope_theta", ROPE_THETA))
    tie_embeddings = getattr(model.config, "tie_word_embeddings", True)

    def f32(key: str) -> torch.Tensor:
        """Load weight as float32 (used for small norm weights and embeddings)."""
        return state[key].float().contiguous().cuda()  # noqa: F821

    def bf16(key: str) -> torch.Tensor:
        """Load weight as bfloat16 (used for large GEMV weight matrices)."""
        return state[key].to(torch.bfloat16).contiguous().cuda()  # noqa: F821

    if verbose:
        print("Stacking per-layer weights...")

    # ── attention layer-norm (float32 — small, used in rmsnorm) ──────────────
    attn_ln_ws = torch.stack(
        [f32(f"model.layers.{i}.input_layernorm.weight") for i in range(NUM_LAYERS)]
    )  # [28, 2048] float32

    # ── QKV projection (bfloat16 — large GEMV matrix, halves HBM traffic) ────
    qkv_ws = torch.stack(
        [
            torch.cat(
                [
                    bf16(f"model.layers.{i}.self_attn.q_proj.weight"),  # [2048, 2048]
                    bf16(f"model.layers.{i}.self_attn.k_proj.weight"),  # [1024, 2048]
                    bf16(f"model.layers.{i}.self_attn.v_proj.weight"),  # [1024, 2048]
                ],
                dim=0,
            )
            for i in range(NUM_LAYERS)
        ]
    ).view(
        NUM_LAYERS, -1
    )  # [28, 4096 * 2048] bfloat16

    # ── Q/K per-head norms (float32 — small, used in rmsnorm) ────────────────
    q_norm_ws = torch.stack(
        [f32(f"model.layers.{i}.self_attn.q_norm.weight") for i in range(NUM_LAYERS)]
    )  # [28, 128] float32
    k_norm_ws = torch.stack(
        [f32(f"model.layers.{i}.self_attn.k_norm.weight") for i in range(NUM_LAYERS)]
    )  # [28, 128] float32

    # ── output projection (bfloat16) ──────────────────────────────────────────
    o_proj_ws = torch.stack([bf16(f"model.layers.{i}.self_attn.o_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )  # [28, 2048 * 2048] bfloat16

    # ── MLP layer-norm (float32) + projections (bfloat16) ────────────────────
    mlp_ln_ws = torch.stack(
        [f32(f"model.layers.{i}.post_attention_layernorm.weight") for i in range(NUM_LAYERS)]
    )  # [28, 2048] float32
    gate_ws = torch.stack([bf16(f"model.layers.{i}.mlp.gate_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )  # [28, 6144 * 2048] bfloat16
    up_ws = torch.stack([bf16(f"model.layers.{i}.mlp.up_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )  # [28, 6144 * 2048] bfloat16
    down_ws = torch.stack([bf16(f"model.layers.{i}.mlp.down_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )  # [28, 2048 * 6144] bfloat16

    # ── final norm (float32) + LM head (bfloat16) ────────────────────────────
    norm_w = f32("model.norm.weight")  # [2048] float32
    embed_w = f32("model.embed_tokens.weight")  # [151936, 2048] float32 (for embedding lookup)
    # LM head uses bfloat16 for GEMV; separate copy even when embeddings are tied
    lm_head_w = (
        (state["model.embed_tokens.weight"] if tie_embeddings else state["lm_head.weight"])
        .to(torch.bfloat16)
        .contiguous()
        .cuda()
    )

    # ── RoPE tables ───────────────────────────────────────────────────────────
    cos_table, sin_table = _precompute_rope(max_seq_len, rope_theta)

    del model, state
    torch.cuda.empty_cache()

    if verbose:
        print("Weights loaded.")

    weights = dict(
        embed_w=embed_w,
        attn_ln_ws=attn_ln_ws,
        qkv_ws=qkv_ws,
        q_norm_ws=q_norm_ws,
        k_norm_ws=k_norm_ws,
        o_proj_ws=o_proj_ws,
        mlp_ln_ws=mlp_ln_ws,
        gate_ws=gate_ws,
        up_ws=up_ws,
        down_ws=down_ws,
        norm_w=norm_w,
        lm_head_w=lm_head_w,
        cos_table=cos_table,
        sin_table=sin_table,
    )
    return weights, tokenizer


# ── Decoder ───────────────────────────────────────────────────────────────────


class Decoder:
    """
    Stateful single-token decoder for Qwen3-1.7B using the persistent megakernel.

    Follows the AlpinDale Decoder pattern:
      - step(token_id)  → next token id
      - generate(prompt) → decoded string
      - reset()          → clear KV cache and position counter
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        verbose: bool = True,
        weights: dict | None = None,
        tokenizer=None,
    ):
        from .build import get_kernels

        self._kernels = get_kernels()

        if weights is None:
            weights, tokenizer = load_weights(model_name, max_seq_len, verbose)
        self.tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._position = 0

        # Keep references so tensors are not GC'd
        self._weights = weights
        self._embed_w = weights["embed_w"]
        self._attn_ln_ws = weights["attn_ln_ws"]
        self._qkv_ws = weights["qkv_ws"]
        self._q_norm_ws = weights["q_norm_ws"]
        self._k_norm_ws = weights["k_norm_ws"]
        self._o_proj_ws = weights["o_proj_ws"]
        self._mlp_ln_ws = weights["mlp_ln_ws"]
        self._gate_ws = weights["gate_ws"]
        self._up_ws = weights["up_ws"]
        self._down_ws = weights["down_ws"]
        self._norm_w = weights["norm_w"]
        self._lm_head_w = weights["lm_head_w"]
        self._cos = weights["cos_table"]
        self._sin = weights["sin_table"]

        # KV caches — [num_layers, max_seq * kv_dim] float32
        self._k_cache = torch.zeros(NUM_LAYERS, max_seq_len * KV_DIM, dtype=torch.float32, device="cuda")
        self._v_cache = torch.zeros_like(self._k_cache)

    # ── single-token step ─────────────────────────────────────────────────────

    def step(self, token_id: int) -> int:
        """
        Process one token and return the next predicted token id.

        The embedding lookup happens in Python; the megakernel handles all 28
        transformer layers in a single CUDA kernel launch.
        """
        hidden = self._embed_w[token_id].clone()  # [hidden_size] float32

        logits = self._kernels.qwen3_decode_persistent_forward(
            hidden,
            self._attn_ln_ws,
            self._qkv_ws,
            self._q_norm_ws,
            self._k_norm_ws,
            self._cos,
            self._sin,
            self._k_cache,
            self._v_cache,
            self._o_proj_ws,
            self._mlp_ln_ws,
            self._gate_ws,
            self._up_ws,
            self._down_ws,
            self._norm_w,
            self._lm_head_w,
            self._position,
        )  # → [vocab_size] float32

        self._position += 1
        return int(logits.argmax().item())

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset(self):
        """Clear KV cache and position counter for a new generation."""
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Encode prompt, prefill KV cache token-by-token, then generate up to
        max_tokens new tokens.  Returns the generated text (not including the
        prompt).
        """
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Prefill: run all prompt tokens except the last through the kernel so
        # their KV entries are cached.  We discard the output logits.
        for tid in ids[:-1]:
            self.step(tid)

        # Autoregressive generation starting from the last prompt token
        output_ids = []
        tok = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            tok = self.step(tok)
            if tok == eos:
                break
            output_ids.append(tok)

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
