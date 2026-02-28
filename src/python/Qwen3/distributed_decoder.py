import torch
import torch.distributed as dist
import torch.nn.functional as F

from .decoder import (
    DEFAULT_MAX_SEQ,
    DEFAULT_MODEL,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    Q_DIM,
    RMS_EPS,
    ROPE_THETA,
    _precompute_rope,
)

MULTIMEM_BF16_ALIGN = 8  # multimem ops work on 8 bf16 elements at a time


def _align_to_multimem(n: int) -> int:
    """Round up to nearest multiple of MULTIMEM_BF16_ALIGN."""
    return ((n + MULTIMEM_BF16_ALIGN - 1) // MULTIMEM_BF16_ALIGN) * MULTIMEM_BF16_ALIGN


def load_weights_for_rank(rank, model_name=DEFAULT_MODEL, max_seq_len=DEFAULT_MAX_SEQ, verbose=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{rank}"
    if verbose:
        print(f"Loading {model_name} to {device}...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    rope_theta = float(getattr(model.config, "rope_theta", ROPE_THETA))
    tie_embeddings = getattr(model.config, "tie_word_embeddings", True)

    def f32(key):
        return state[key].float().contiguous()  # noqa: F821

    def bf16(key):
        return state[key].to(torch.bfloat16).contiguous()  # noqa: F821

    if verbose:
        print("Stacking per-layer weights...", flush=True)

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
    )

    cos_table, sin_table = _precompute_rope(max_seq_len, rope_theta)

    del model, state
    torch.cuda.empty_cache()

    if verbose:
        print("Weights loaded.", flush=True)

    return (
        dict(
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
        ),
        tokenizer,
    )


class DistributedDecoder:
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        model_name: str = DEFAULT_MODEL,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        verbose: bool = True,
        weights: dict | None = None,
        tokenizer=None,
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self._max_seq_len = max_seq_len
        self._position = 0
        self._use_multimem = False

        if weights is None:
            weights, tokenizer = load_weights_for_rank(tp_rank, model_name, max_seq_len, verbose)
        self.tokenizer = tokenizer

        self._embed_w = weights["embed_w"]
        self._norm_w = weights["norm_w"]
        self._lm_head_w = weights["lm_head_w"]
        self._cos, self._sin = weights["cos_table"], weights["sin_table"]

        q_heads_per_rank = NUM_Q_HEADS // tp_size
        kv_heads_per_rank = NUM_KV_HEADS // tp_size
        self._q_heads_local = q_heads_per_rank
        self._kv_heads_local = kv_heads_per_rank
        self._q_dim_local = q_heads_per_rank * HEAD_DIM
        self._kv_dim_local = kv_heads_per_rank * HEAD_DIM

        q_start = tp_rank * q_heads_per_rank * HEAD_DIM
        q_end = q_start + q_heads_per_rank * HEAD_DIM
        kv_start = tp_rank * kv_heads_per_rank * HEAD_DIM
        kv_end = kv_start + kv_heads_per_rank * HEAD_DIM

        self._attn_ln_ws = weights["attn_ln_ws"]
        self._mlp_ln_ws = weights["mlp_ln_ws"]

        qkv_ws_3d = weights["qkv_ws"].view(NUM_LAYERS, Q_DIM + 2 * KV_DIM, HIDDEN_SIZE)
        q_w = qkv_ws_3d[:, :Q_DIM, :][:, q_start:q_end, :]
        k_w = qkv_ws_3d[:, Q_DIM : Q_DIM + KV_DIM, :][:, kv_start:kv_end, :]
        v_w = qkv_ws_3d[:, Q_DIM + KV_DIM :, :][:, kv_start:kv_end, :]
        self._qkv_ws = torch.cat([q_w, k_w, v_w], dim=1).contiguous()

        self._q_norm_ws = weights["q_norm_ws"]
        self._k_norm_ws = weights["k_norm_ws"]

        o_proj_ws_3d = weights["o_proj_ws"].view(NUM_LAYERS, HIDDEN_SIZE, HIDDEN_SIZE)
        self._o_proj_ws = o_proj_ws_3d[:, :, q_start:q_end].contiguous()

        inter_per_rank = INTERMEDIATE_SIZE // tp_size
        inter_start = tp_rank * inter_per_rank
        inter_end = inter_start + inter_per_rank
        self._inter_local = inter_per_rank

        gate_ws_3d = weights["gate_ws"].view(NUM_LAYERS, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self._gate_ws = gate_ws_3d[:, inter_start:inter_end, :].contiguous()
        up_ws_3d = weights["up_ws"].view(NUM_LAYERS, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self._up_ws = up_ws_3d[:, inter_start:inter_end, :].contiguous()
        down_ws_3d = weights["down_ws"].view(NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self._down_ws = down_ws_3d[:, :, inter_start:inter_end].contiguous()

        # Fused gate+up for both prefill and decode
        self._gate_up_ws = torch.cat(
            [
                self._gate_ws.view(NUM_LAYERS, inter_per_rank, HIDDEN_SIZE),
                self._up_ws.view(NUM_LAYERS, inter_per_rank, HIDDEN_SIZE),
            ],
            dim=1,
        )

        self._k_cache = torch.zeros(
            NUM_LAYERS,
            max_seq_len,
            self._kv_dim_local,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        half = HEAD_DIM // 2
        self._cos_dup = torch.empty(max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        cos_bf16 = self._cos.to(torch.bfloat16)
        sin_bf16 = self._sin.to(torch.bfloat16)
        self._cos_dup[:, :half] = cos_bf16[:, :half]
        self._cos_dup[:, half:] = cos_bf16[:, :half]
        self._sin_dup = torch.empty(max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._sin_dup[:, :half] = sin_bf16[:, :half]
        self._sin_dup[:, half:] = sin_bf16[:, :half]

        self._attn_ln_ws_bf16 = self._attn_ln_ws.to(torch.bfloat16)
        self._mlp_ln_ws_bf16 = self._mlp_ln_ws.to(torch.bfloat16)
        self._q_norm_ws_bf16 = self._q_norm_ws.to(torch.bfloat16)
        self._k_norm_ws_bf16 = self._k_norm_ws.to(torch.bfloat16)
        self._norm_w_bf16 = self._norm_w.to(torch.bfloat16)
        self._embed_w_bf16 = self._embed_w.to(torch.bfloat16)

        # Pre-allocate decode scratch buffers (reused every step)
        self._decode_hidden = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._decode_o_buf = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._decode_down_buf = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        # Pre-compile tp_sp kernels + init allreduce
        self._tp_kernels = None
        if tp_size > 1:
            self._init_multimem(max_seq_len, verbose)

        if verbose:
            q_params = self._qkv_ws[0].numel()
            o_params = self._o_proj_ws[0].numel()
            g_params = self._gate_ws[0].numel() + self._up_ws[0].numel() + self._down_ws[0].numel()
            total_sharded = (q_params + o_params + g_params) * NUM_LAYERS * 2
            print(
                f"[rank {tp_rank}] Sharded weight bytes: {total_sharded / 1e6:.1f} MB "
                f"({self._q_heads_local}Q/{self._kv_heads_local}KV heads, "
                f"{self._inter_local} intermediate)"
            )

    def _init_multimem(self, max_seq_len: int, verbose: bool):
        """Pre-compile tp_sp kernels (serialized across ranks to avoid race)."""
        try:
            from .build import get_tp_kernels

            # Serialize JIT compilation across ranks to avoid race condition
            # where both processes compile simultaneously and clobber the .so
            for r in range(self.tp_size):
                if self.tp_rank == r:
                    self._tp_kernels = get_tp_kernels()
                dist.barrier()

            if verbose:
                print(f"[rank {self.tp_rank}] tp_sp kernels compiled OK")

            # NOTE: Multicast (NVLS) allreduce requires all processes to share
            # the SAME multicast handle via IPC fd export/import. With mp.spawn
            # each process creates independent multicast objects, so the
            # multimem barrier deadlocks. Using NCCL allreduce instead.
            self._use_multimem = False
        except Exception as e:
            if verbose:
                print(f"[rank {self.tp_rank}] tp_sp compilation failed: {e}")
            self._use_multimem = False

    def _allreduce(self, tensor: torch.Tensor):
        """All-reduce a tensor across TP ranks via NCCL."""
        if self.tp_size <= 1:
            return
        dist.all_reduce(tensor)

    def _allreduce_mm(self, mat: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
        """Fused matmul -> allreduce: writes matmul result into out, then allreduces in-place."""
        torch.mm(mat, weight, out=out)
        self._allreduce(out)

    def _apply_rope_fast(self, x, cos_dup, sin_dup, num_heads):
        half = HEAD_DIM // 2
        x = x.view(x.size(0), num_heads, HEAD_DIM)
        x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return torch.addcmul(x * cos_dup.unsqueeze(1), x_rot, sin_dup.unsqueeze(1)).view(x.size(0), -1)

    def step(self, token_id: int) -> int:
        pos = self._position
        # Copy embedding into pre-allocated buffer (avoids allocation + enables in-place)
        self._decode_hidden[0].copy_(self._embed_w_bf16[token_id])
        hidden = self._decode_hidden

        cos_dup = self._cos_dup[pos : pos + 1]
        sin_dup = self._sin_dup[pos : pos + 1]

        o_buf = self._decode_o_buf
        down_buf = self._decode_down_buf

        for layer in range(NUM_LAYERS):
            normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self._attn_ln_ws_bf16[layer], eps=RMS_EPS)

            qkv = torch.mm(normed, self._qkv_ws[layer].t())
            q = qkv[:, : self._q_dim_local]
            k = qkv[:, self._q_dim_local : self._q_dim_local + self._kv_dim_local]
            v = qkv[:, self._q_dim_local + self._kv_dim_local :]

            q = F.rms_norm(
                q.view(1, self._q_heads_local, HEAD_DIM),
                (HEAD_DIM,),
                self._q_norm_ws_bf16[layer],
                eps=RMS_EPS,
            ).view(1, -1)
            k = F.rms_norm(
                k.view(1, self._kv_heads_local, HEAD_DIM),
                (HEAD_DIM,),
                self._k_norm_ws_bf16[layer],
                eps=RMS_EPS,
            ).view(1, -1)

            q = self._apply_rope_fast(q, cos_dup, sin_dup, self._q_heads_local)
            k = self._apply_rope_fast(k, cos_dup, sin_dup, self._kv_heads_local)

            self._k_cache[layer, pos] = k.view(self._kv_dim_local)
            self._v_cache[layer, pos] = v.view(self._kv_dim_local)

            seq_len = pos + 1
            q_4d = q.view(1, 1, self._q_heads_local, HEAD_DIM).transpose(1, 2)
            k_all = self._k_cache[layer, :seq_len].view(1, seq_len, self._kv_heads_local, HEAD_DIM).transpose(1, 2)
            v_all = self._v_cache[layer, :seq_len].view(1, seq_len, self._kv_heads_local, HEAD_DIM).transpose(1, 2)

            attn = F.scaled_dot_product_attention(q_4d, k_all, v_all, is_causal=False, enable_gqa=True)
            attn_out = attn.transpose(1, 2).reshape(1, self._q_dim_local)

            # matmul -> allreduce -> in-place residual add
            self._allreduce_mm(attn_out, self._o_proj_ws[layer].t(), o_buf)
            hidden.add_(o_buf)

            normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self._mlp_ln_ws_bf16[layer], eps=RMS_EPS)
            # Fused gate+up (single matmul instead of two)
            gate_up = torch.mm(normed, self._gate_up_ws[layer].t())
            gate = gate_up[:, : self._inter_local]
            up = gate_up[:, self._inter_local :]

            self._allreduce_mm(F.silu(gate) * up, self._down_ws[layer].t(), down_buf)
            hidden.add_(down_buf)

        last_hidden = F.rms_norm(hidden, (HIDDEN_SIZE,), self._norm_w_bf16, eps=RMS_EPS)
        logits = torch.mm(last_hidden, self._lm_head_w.t())

        self._position += 1
        return int(logits.argmax().item())

    def _prefill_forward(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        """Batched prefill: process all n tokens at once through each layer."""
        hidden = self._embed_w_bf16[token_ids]  # (n, HIDDEN_SIZE) — advanced indexing = new tensor

        cos_dup = self._cos_dup[:n]
        sin_dup = self._sin_dup[:n]

        # Pre-allocate reusable output buffers for this prefill size
        o_buf = torch.empty(n, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        down_buf = torch.empty(n, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        for layer in range(NUM_LAYERS):
            normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self._attn_ln_ws_bf16[layer], eps=RMS_EPS)

            qkv = torch.mm(normed, self._qkv_ws[layer].t())
            q = qkv[:, : self._q_dim_local]
            k = qkv[:, self._q_dim_local : self._q_dim_local + self._kv_dim_local]
            v = qkv[:, self._q_dim_local + self._kv_dim_local :]

            q = F.rms_norm(
                q.view(n, self._q_heads_local, HEAD_DIM),
                (HEAD_DIM,),
                self._q_norm_ws_bf16[layer],
                eps=RMS_EPS,
            ).view(n, -1)
            k = F.rms_norm(
                k.view(n, self._kv_heads_local, HEAD_DIM),
                (HEAD_DIM,),
                self._k_norm_ws_bf16[layer],
                eps=RMS_EPS,
            ).view(n, -1)

            q = self._apply_rope_fast(q, cos_dup, sin_dup, self._q_heads_local)
            k = self._apply_rope_fast(k, cos_dup, sin_dup, self._kv_heads_local)

            self._k_cache[layer, :n] = k.view(n, self._kv_dim_local)
            self._v_cache[layer, :n] = v.view(n, self._kv_dim_local)

            q_4d = q.view(1, n, self._q_heads_local, HEAD_DIM).transpose(1, 2)
            k_all = self._k_cache[layer, :n].view(1, n, self._kv_heads_local, HEAD_DIM).transpose(1, 2)
            v_all = self._v_cache[layer, :n].view(1, n, self._kv_heads_local, HEAD_DIM).transpose(1, 2)

            attn = F.scaled_dot_product_attention(q_4d, k_all, v_all, is_causal=True, enable_gqa=True)
            attn_out = attn.transpose(1, 2).reshape(n, self._q_dim_local)

            # matmul -> allreduce -> in-place residual add
            self._allreduce_mm(attn_out, self._o_proj_ws[layer].t(), o_buf)
            hidden.add_(o_buf)

            normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self._mlp_ln_ws_bf16[layer], eps=RMS_EPS)
            gate_up = torch.mm(normed, self._gate_up_ws[layer].t())
            gate = gate_up[:, : self._inter_local]
            up = gate_up[:, self._inter_local :]

            self._allreduce_mm(F.silu(gate) * up, self._down_ws[layer].t(), down_buf)
            hidden.add_(down_buf)

        last_hidden = F.rms_norm(hidden[-1:], (HIDDEN_SIZE,), self._norm_w_bf16, eps=RMS_EPS)
        logits = torch.mm(last_hidden, self._lm_head_w.t())
        return logits

    def prefill(self, token_ids: list[int] | torch.Tensor) -> int:
        """Batched prefill: all prompt tokens processed in one forward pass."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device="cuda")
        if token_ids.dim() == 1:
            token_ids = token_ids.long()
        n = token_ids.size(0)

        logits = self._prefill_forward(token_ids, n)
        self._position += n
        return int(logits.argmax().item())

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        self.prefill(ids[:-1])

        output_ids = []
        tok = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            tok = self.step(tok)
            if tok == eos:
                break
            output_ids.append(tok)

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
