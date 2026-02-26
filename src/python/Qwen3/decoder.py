import torch

NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
KV_DIM = NUM_KV_HEADS * HEAD_DIM
Q_DIM = NUM_Q_HEADS * HEAD_DIM
QKV_DIM = Q_DIM + 2 * KV_DIM
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000.0
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_MAX_SEQ = 4096


def _precompute_rope(max_seq_len: int, rope_theta: float = ROPE_THETA) -> tuple:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    cos_table = torch.zeros(max_seq_len, HEAD_DIM, dtype=torch.float32)
    sin_table = torch.zeros(max_seq_len, HEAD_DIM, dtype=torch.float32)
    cos_table[:, :half] = torch.cos(freqs)
    sin_table[:, :half] = torch.sin(freqs)

    return cos_table.cuda().contiguous(), sin_table.cuda().contiguous()


def load_weights(
    model_name: str = DEFAULT_MODEL,
    max_seq_len: int = DEFAULT_MAX_SEQ,
    verbose: bool = True,
) -> tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if verbose:
        print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    rope_theta = float(getattr(model.config, "rope_theta", ROPE_THETA))
    tie_embeddings = getattr(model.config, "tie_word_embeddings", True)

    def f32(key: str) -> torch.Tensor:
        return state[key].float().contiguous().cuda()  # noqa: F821

    def bf16(key: str) -> torch.Tensor:
        return state[key].to(torch.bfloat16).contiguous().cuda()  # noqa: F821

    if verbose:
        print("Stacking per-layer weights...")

    attn_ln_ws = torch.stack([f32(f"model.layers.{i}.input_layernorm.weight") for i in range(NUM_LAYERS)])

    qkv_ws = torch.stack(
        [
            torch.cat(
                [
                    bf16(f"model.layers.{i}.self_attn.q_proj.weight"),
                    bf16(f"model.layers.{i}.self_attn.k_proj.weight"),
                    bf16(f"model.layers.{i}.self_attn.v_proj.weight"),
                ],
                dim=0,
            )
            for i in range(NUM_LAYERS)
        ]
    ).view(NUM_LAYERS, -1)

    q_norm_ws = torch.stack([f32(f"model.layers.{i}.self_attn.q_norm.weight") for i in range(NUM_LAYERS)])
    k_norm_ws = torch.stack([f32(f"model.layers.{i}.self_attn.k_norm.weight") for i in range(NUM_LAYERS)])

    o_proj_ws = torch.stack([bf16(f"model.layers.{i}.self_attn.o_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )

    mlp_ln_ws = torch.stack([f32(f"model.layers.{i}.post_attention_layernorm.weight") for i in range(NUM_LAYERS)])
    gate_ws = torch.stack([bf16(f"model.layers.{i}.mlp.gate_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )
    up_ws = torch.stack([bf16(f"model.layers.{i}.mlp.up_proj.weight") for i in range(NUM_LAYERS)]).view(NUM_LAYERS, -1)
    down_ws = torch.stack([bf16(f"model.layers.{i}.mlp.down_proj.weight") for i in range(NUM_LAYERS)]).view(
        NUM_LAYERS, -1
    )

    norm_w = f32("model.norm.weight")
    embed_w = f32("model.embed_tokens.weight")
    lm_head_w = (
        (state["model.embed_tokens.weight"] if tie_embeddings else state["lm_head.weight"])
        .to(torch.bfloat16)
        .contiguous()
        .cuda()
    )

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


class Decoder:
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

        self._k_cache = torch.zeros(NUM_LAYERS, max_seq_len * KV_DIM, dtype=torch.float32, device="cuda")
        self._v_cache = torch.zeros_like(self._k_cache)

    def step(self, token_id: int) -> int:
        hidden = self._embed_w[token_id].clone()

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
        )

        self._position += 1
        return int(logits.argmax().item())

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        for tid in ids[:-1]:
            self.step(tid)

        output_ids = []
        tok = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            tok = self.step(tok)
            if tok == eos:
                break
            output_ids.append(tok)

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
