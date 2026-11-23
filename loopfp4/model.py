"""Looped decoder-only transformer with 4-bit (NVFP4) forward passes.

Layout, following the Inkling 2 Mini description:

    prelude  (n_prelude layers)  runs once
    loop     (n_loop layers)     runs n_loops times with shared weights
    coda     (n_coda layers)     runs once

Every pass through the loop starts again from the prelude's output. Passes after
the first also read the previous pass's result (the "message") through learned
gates that start at zero, so at initialisation every pass computes the same thing
and the model behaves exactly like its non-looped counterpart. Training opens the
gates. Layers inside the loop can also re-read the prelude output through their
own zero-initialised gate.

This is a simplified take on the published design: the gates here are learned
scalars per layer, not content-dependent attention over earlier block outputs,
and the feed-forward layers are dense rather than mixture-of-experts.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nvfp4 import QLinear


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 192
    n_head: int = 6
    n_prelude: int = 3
    n_loop: int = 6            # stored layers in the looped section
    n_coda: int = 3
    n_loops: int = 3           # passes through the looped section
    block_size: int = 256      # context length
    quant: str = "nvfp4"       # "none" | "nvfp4"
    quant_scope: str = "all"   # "all" linear layers in a block, or only the "mlp"
    hp_tail_frac: float = 1 / 6  # last fraction of stored layers kept in MXFP8 instead of NVFP4
    four_over_six: bool = True   # 4/6 scale selection when quantizing activations
    backend: str = "sim"       # "sim" (portable simulation) | "te" (Transformer Engine, Blackwell)
    grad_checkpoint: bool = False  # recompute activations in backward to save memory

    @property
    def n_layer(self) -> int:
        return self.n_prelude + self.n_loop + self.n_coda

    def applications(self, n_loops=None) -> int:
        return self.n_prelude + self.n_loop * (n_loops or self.n_loops) + self.n_coda

    @property
    def mlp_hidden(self) -> int:
        return int(round(8 * self.d_model / 3 / 64)) * 64


PRESETS = {
    # ~5M params, byte-level vocab. Only for checking the code on a laptop.
    "smoke": Config(vocab_size=256, d_model=192, n_head=6, n_prelude=3, n_loop=6, n_coda=3, block_size=256),
    # ~1B params. First real run on a single B200 node.
    "1b": Config(vocab_size=50304, d_model=1792, n_head=28, n_prelude=6, n_loop=12, n_coda=6, block_size=2048),
    # ~5B stored params, 48 layer applications per token at 3 loops.
    "5b": Config(vocab_size=50304, d_model=4096, n_head=32, n_prelude=6, n_loop=12, n_coda=6, block_size=2048,
                 grad_checkpoint=True),
}


def make_linear(cfg: Config, n_in: int, n_out: int, mode: str) -> nn.Module:
    if cfg.backend == "te" and mode == "nvfp4":
        # Native NVFP4 GEMMs. Requires a Blackwell GPU (SM100+) and NVIDIA Transformer Engine,
        # and the forward pass must run inside te.autocast(recipe=NVFP4BlockScaling()).
        import transformer_engine.pytorch as te
        return te.Linear(n_in, n_out, bias=False)
    if cfg.backend == "te":
        return nn.Linear(n_in, n_out, bias=False)  # high-precision tail stays in BF16 under autocast
    return QLinear(n_in, n_out, mode=mode, four_over_six=cfg.four_over_six)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_tables(seq_len: int, head_dim: int, device):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return ang.cos()[None, None], ang.sin()[None, None]  # [1, 1, T, head_dim/2]


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config, mode: str):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = make_linear(cfg, cfg.d_model, 3 * cfg.d_model, mode)
        self.proj = make_linear(cfg, cfg.d_model, cfg.d_model, mode)

    def forward(self, x, rope):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        cos, sin = rope
        q = apply_rope(q, cos[:, :, :T].to(q.dtype), sin[:, :, :T].to(q.dtype))
        k = apply_rope(k, cos[:, :, :T].to(k.dtype), sin[:, :, :T].to(k.dtype))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, cfg: Config, mode: str):
        super().__init__()
        self.w1 = make_linear(cfg, cfg.d_model, 2 * cfg.mlp_hidden, mode)
        self.w2 = make_linear(cfg, cfg.mlp_hidden, cfg.d_model, mode)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: Config, mode: str):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, mode if cfg.quant_scope == "all" else "none")
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg, mode)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), (cos, sin))
        return x + self.mlp(self.n2(x))


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0 and (cfg.d_model // cfg.n_head) % 2 == 0
        self.cfg = cfg
        n_hp = round(cfg.n_layer * cfg.hp_tail_frac) if cfg.quant == "nvfp4" else 0

        def mode(i):  # stored layer index -> numeric format
            if cfg.quant == "none":
                return "none"
            return "mxfp8" if i >= cfg.n_layer - n_hp else "nvfp4"

        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        idx = iter(range(cfg.n_layer))
        self.prelude = nn.ModuleList(Block(cfg, mode(next(idx))) for _ in range(cfg.n_prelude))
        self.loop = nn.ModuleList(Block(cfg, mode(next(idx))) for _ in range(cfg.n_loop))
        self.coda = nn.ModuleList(Block(cfg, mode(next(idx))) for _ in range(cfg.n_coda))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tied; embeddings and head stay high precision

        # Loop plumbing. Both gates start closed, so loops begin as exact repeats.
        self.msg_norm = RMSNorm(cfg.d_model)
        self.g_msg = nn.Parameter(torch.zeros(cfg.n_loop))  # gain on the previous pass's message
        self.g_pre = nn.Parameter(torch.zeros(cfg.n_loop))  # gain on re-reading the prelude output

        self.apply(self._init)
        resid_std = 0.02 / math.sqrt(2 * cfg.applications())
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=resid_std)

        # Loop-section weights are used n_loops times per step; quantize them once per step.
        self._cached = [m for m in self.loop.modules() if isinstance(m, QLinear)]
        for m in self._cached:
            m.cache_weight = not cfg.grad_checkpoint
        self._rope = None

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Embedding) or (hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 2):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _run(self, blk, x, cos, sin):
        if self.cfg.grad_checkpoint and self.training:
            ckpt_fn = checkpoint
            if self.cfg.backend == "te":
                try:
                    from transformer_engine.pytorch.distributed import checkpoint as ckpt_fn
                except ImportError:
                    pass
            return ckpt_fn(blk, x, cos, sin, use_reentrant=False)
        return blk(x, cos, sin)

    def forward(self, idx, targets=None, n_loops=None, return_logits: bool = True):
        cfg = self.cfg
        n_loops = n_loops or cfg.n_loops
        T = idx.shape[1]
        if self._rope is None or self._rope[0].shape[2] < T or self._rope[0].device != idx.device:
            self._rope = rope_tables(max(T, cfg.block_size), cfg.d_model // cfg.n_head, idx.device)
        cos, sin = self._rope

        x = self.emb(idx)
        if torch.is_autocast_enabled(idx.device.type):
            x = x.to(torch.get_autocast_dtype(idx.device.type))
        for blk in self.prelude:
            x = self._run(blk, x, cos, sin)

        pre, msg = x, None
        for p in range(n_loops):
            x = pre
            for i, blk in enumerate(self.loop):
                if i > 0:
                    x = x + self.g_pre[i] * pre
                if msg is not None:
                    x = x + self.g_msg[i] * msg
                x = self._run(blk, x, cos, sin)
            if p < n_loops - 1:
                msg = self.msg_norm(x - pre)  # what the looped section added on this pass
        for m in self._cached:
            m._wq = None
        if n_loops == 1:  # keep the loop plumbing in the graph so DDP sees a gradient for it
            x = x + 0.0 * (self.g_msg.sum() + self.msg_norm.weight.sum())
        if cfg.n_loop == 1:
            x = x + 0.0 * self.g_pre.sum()

        for blk in self.coda:
            x = self._run(blk, x, cos, sin)
        hidden = self.norm_f(x)
        if targets is not None and not return_logits:
            h_flat = hidden.reshape(-1, hidden.size(-1))
            t_flat = targets.reshape(-1)
            chunk = 1024

            def _ce_chunk(h_c, t_c):
                return F.cross_entropy(self.head(h_c).float(), t_c, reduction="sum")

            total = h_flat.new_zeros((), dtype=torch.float32)
            for s in range(0, h_flat.size(0), chunk):
                h_c, t_c = h_flat[s : s + chunk], t_flat[s : s + chunk]
                if self.training and h_c.requires_grad:
                    total = total + checkpoint(_ce_chunk, h_c, t_c, use_reentrant=False)
                else:
                    total = total + _ce_chunk(h_c, t_c)
            return None, total / t_flat.numel()
        logits = self.head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, n_loops=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:], n_loops=n_loops)
            if temperature == 0:
                idx = torch.cat([idx, logits[:, -1:].argmax(dim=-1)], dim=1)
                continue
            logits = logits[:, -1].float() / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def config_dict(self) -> dict:
        return asdict(self.cfg)
