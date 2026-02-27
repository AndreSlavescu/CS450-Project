import torch
import torch.nn.functional as F

NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
KV_DIM = NUM_KV_HEADS * HEAD_DIM
Q_DIM = NUM_Q_HEADS * HEAD_DIM
QKV_DIM = Q_DIM + 2 * KV_DIM
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS
RMS_EPS = 1e-6
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

        self._qkv_ws_3d = self._qkv_ws.view(NUM_LAYERS, QKV_DIM, HIDDEN_SIZE)
        self._o_proj_ws_3d = self._o_proj_ws.view(NUM_LAYERS, HIDDEN_SIZE, HIDDEN_SIZE)
        self._down_ws_3d = self._down_ws.view(NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self._gate_up_ws_3d = torch.cat(
            [
                self._gate_ws.view(NUM_LAYERS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                self._up_ws.view(NUM_LAYERS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ],
            dim=1,
        )

        self._attn_ln_ws_bf16 = self._attn_ln_ws.to(torch.bfloat16)
        self._mlp_ln_ws_bf16 = self._mlp_ln_ws.to(torch.bfloat16)
        self._q_norm_ws_bf16 = self._q_norm_ws.to(torch.bfloat16)
        self._k_norm_ws_bf16 = self._k_norm_ws.to(torch.bfloat16)
        self._norm_w_bf16 = self._norm_w.to(torch.bfloat16)
        self._embed_w_bf16 = self._embed_w.to(torch.bfloat16)
        self._cos_bf16 = self._cos.to(torch.bfloat16)
        self._sin_bf16 = self._sin.to(torch.bfloat16)

        half = HEAD_DIM // 2
        self._cos_dup = torch.empty(max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._cos_dup[:, :half] = self._cos_bf16[:, :half]
        self._cos_dup[:, half:] = self._cos_bf16[:, :half]
        self._sin_dup = torch.empty(max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._sin_dup[:, :half] = self._sin_bf16[:, :half]
        self._sin_dup[:, half:] = self._sin_bf16[:, :half]

        self._k_cache_pf = torch.zeros(NUM_LAYERS, max_seq_len, KV_DIM, dtype=torch.bfloat16, device="cuda")
        self._v_cache_pf = torch.zeros_like(self._k_cache_pf)

    @staticmethod
    def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return x * rms * w

    @staticmethod
    def _head_rmsnorm(x: torch.Tensor, w: torch.Tensor, num_heads: int) -> torch.Tensor:
        x = x.view(x.size(0), num_heads, HEAD_DIM)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return (x * rms * w).view(x.size(0), -1)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, num_heads: int) -> torch.Tensor:
        half = HEAD_DIM // 2
        x = x.view(x.size(0), num_heads, HEAD_DIM)
        x0, x1 = x[..., :half], x[..., half:]
        cos_v, sin_v = cos[:, :half].unsqueeze(1), sin[:, :half].unsqueeze(1)
        out = torch.cat([x0 * cos_v - x1 * sin_v, x1 * cos_v + x0 * sin_v], dim=-1)
        return out.view(x.size(0), -1)

    def _apply_rope_fast(
        self, x: torch.Tensor, cos_dup: torch.Tensor, sin_dup: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        half = HEAD_DIM // 2
        x = x.view(x.size(0), num_heads, HEAD_DIM)
        x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        cos_v = cos_dup.unsqueeze(1)
        sin_v = sin_dup.unsqueeze(1)
        return torch.addcmul(x * cos_v, x_rot, sin_v).view(x.size(0), -1)

    def _prefill_forward(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        hidden = self._embed_w_bf16[token_ids]

        cos_dup = self._cos_dup[:n]
        sin_dup = self._sin_dup[:n]

        for layer in range(NUM_LAYERS):
            normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self._attn_ln_ws_bf16[layer], eps=RMS_EPS)

            qkv = torch.mm(normed, self._qkv_ws_3d[layer].t())
            q = qkv[:, :Q_DIM]
            k = qkv[:, Q_DIM : Q_DIM + KV_DIM]
            v = qkv[:, Q_DIM + KV_DIM :]

            q = F.rms_norm(
                q.view(n, NUM_Q_HEADS, HEAD_DIM), (HEAD_DIM,), self._q_norm_ws_bf16[layer], eps=RMS_EPS
            ).view(n, -1)
            k = F.rms_norm(
                k.view(n, NUM_KV_HEADS, HEAD_DIM), (HEAD_DIM,), self._k_norm_ws_bf16[layer], eps=RMS_EPS
            ).view(n, -1)

            q = self._apply_rope_fast(q, cos_dup, sin_dup, NUM_Q_HEADS)
            k = self._apply_rope_fast(k, cos_dup, sin_dup, NUM_KV_HEADS)

            self._k_cache_pf[layer, :n] = k.view(n, KV_DIM)
            self._v_cache_pf[layer, :n] = v.view(n, KV_DIM)

            q = q.view(1, n, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
            k_all = self._k_cache_pf[layer, :n].view(1, n, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v_all = self._v_cache_pf[layer, :n].view(1, n, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            attn = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True, enable_gqa=True)
            attn_out = attn.transpose(1, 2).reshape(n, Q_DIM)

            hidden = torch.addmm(hidden, attn_out, self._o_proj_ws_3d[layer].t())

            normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self._mlp_ln_ws_bf16[layer], eps=RMS_EPS)
            gate_up = torch.mm(normed, self._gate_up_ws_3d[layer].t())
            gate = gate_up[:, :INTERMEDIATE_SIZE]
            up = gate_up[:, INTERMEDIATE_SIZE:]
            hidden = torch.addmm(hidden, F.silu(gate) * up, self._down_ws_3d[layer].t())

        last_hidden = F.rms_norm(hidden[-1:], (HIDDEN_SIZE,), self._norm_w_bf16, eps=RMS_EPS)
        logits = torch.mm(last_hidden, self._lm_head_w.t())
        return logits

    def _capture_prefill_graph(self, n: int) -> dict:
        static_ids = torch.zeros(n, dtype=torch.long, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._prefill_forward(static_ids, n)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits = self._prefill_forward(static_ids, n)

        return {"graph": g, "input_ids": static_ids, "logits": logits}

    def prefill(self, token_ids: torch.Tensor) -> int:
        if token_ids.dim() == 1:
            token_ids = token_ids.long()
        n = token_ids.size(0)

        if not hasattr(self, "_prefill_graphs"):
            self._prefill_graphs = {}

        if n not in self._prefill_graphs:
            self._prefill_graphs[n] = self._capture_prefill_graph(n)

        gd = self._prefill_graphs[n]
        gd["input_ids"].copy_(token_ids)
        gd["graph"].replay()

        self._position += n
        return int(gd["logits"].argmax().item())

    def prefill_fused(self, token_ids: list[int]) -> int:
        for tid in token_ids[:-1]:
            self.step(tid)
        return self.step(token_ids[-1])

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
        self._k_cache_pf.zero_()
        self._v_cache_pf.zero_()

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
