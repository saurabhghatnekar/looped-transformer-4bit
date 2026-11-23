"""Inkling 2 Mini text decoder, written for training from scratch with 4-bit forward passes.

The architecture follows the published reference implementation (Hugging Face transformers,
`models/inkling2_loop` and `models/inkling2`) and the released `config.json`. Module and
parameter names match the reference so a state dict can be copied across;
`tests/test_reference.py` checks that both produce the same logits from the same weights.

    embeddings -> RMSNorm
    24 layers in 4 blocks of 6. Layers 6..17 (the two middle blocks) run `loop_num_passes` times.
    each layer:  attention (sliding window, or global on every 6th layer) -> short causal conv
                 -> mixture of experts (8 of 128, softcapped SwiGLU) -> short causal conv
    attention residuals: a layer does not read one running sum. It reads a learned softmax mix
        of the finished blocks' outputs and the current block's partial sum.
    looping: every pass restarts from the prelude output. Passes after the first also get a
        "message" (a learned mix of the previous pass's block outputs), merged through
        low-rank gates that start closed.

What is different from the reference: no KV cache (training and simple sampling only), no
vision tower, no multi-token-prediction head, and the expert GEMMs are batched across experts
and can run quantized (NVFP4, or MXFP8 in the last layers) with MXFP8 backward GEMMs.
"""

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nvfp4 import qgemm, quantize_act, quantize_expert_weight


@dataclass
class InklingConfig:
    # --- architecture: names and defaults are the released Inkling 2 Mini text_config ---
    vocab_size: int = 201024                 # embedding rows (padded)
    unpadded_vocab_size: int = 200058        # output head rows
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24            # global-attention layers
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 24        # sliding-window layers
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    global_every: int = 6                    # layer i is global when (i + 1) % global_every == 0
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    conv_kernel_size: int = 4
    n_layers_per_attn_res_block: int = 6
    n_routed_experts: int = 128              # 384 before pruning
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1152
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    final_logit_softcapping: float = 30.0
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64
    initializer_range: float = 0.02
    # --- training numerics (not part of the released config) ---
    block_size: int = 2048                   # training context length
    quant: str = "nvfp4"                     # "none" | "nvfp4": numeric format of the expert GEMMs
    hp_tail_layers: int = 4                  # last layers whose experts run in MXFP8 instead
    four_over_six: bool = True               # 4/6 scale selection on expert activations
    backward: str = "mxfp8"                  # "mxfp8" (stochastic rounding) | "exact" backward GEMMs
    grad_checkpoint: bool = False
    # How the expert GEMMs are computed. "sim": 4-bit numerics in ordinary float maths, runs
    # anywhere, no speedup. "te": NVIDIA Transformer Engine's GroupedLinear, native NVFP4 / MXFP8
    # GEMMs on Blackwell. A runtime choice: checkpoints are the same for both.
    backend: str = "sim"
    te_recipe: str = "tml"                   # "tml" | "rowscaled" | "nvidia", see te_recipe() below

    def is_global(self, i: int) -> bool:
        return (i + 1) % self.global_every == 0

    def is_loop(self, i: int) -> bool:
        return self.loop_layer_index_start <= i < self.loop_layer_index_end

    def applications(self, n_loops=None) -> int:
        n = self.loop_layer_index_end - self.loop_layer_index_start
        return self.num_hidden_layers + ((n_loops or self.loop_num_passes) - 1) * n

    def expert_mode(self, i: int) -> str:
        if self.quant == "none":
            return "none"
        return "mxfp8" if i >= self.num_hidden_layers - self.hp_tail_layers else "nvfp4"

    @classmethod
    def from_hf(cls, config: dict, **overrides) -> "InklingConfig":
        """Build from a released config.json (the whole file or its text_config)."""
        t = dict(config.get("text_config", config))
        loop = t.pop("loop_config", {})
        renames = {"intermediate_size": "moe_intermediate_size", "sconv_kernel_size": "conv_kernel_size"}
        t = {renames.get(k, k): v for k, v in t.items()}
        t.update(loop_layer_index_start=loop.get("core_start", 6), loop_layer_index_end=loop.get("core_end", 18),
                 loop_num_passes=loop.get("num_passes", 3), loop_gate_rank=loop.get("gate_rank", 64))
        if "local_layer_ids" in t:
            glob = sorted(set(range(t["num_hidden_layers"])) - set(t["local_layer_ids"]))
            t["global_every"] = glob[0] + 1
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in t.items() if k in known}, **overrides)


def _scaled(hidden, heads, kv_heads, experts=128, **kw) -> InklingConfig:
    """Inkling 2 Mini's layout at another width: same depth, head size, expert count and ratios."""
    return InklingConfig(hidden_size=hidden, num_attention_heads=heads, num_key_value_heads=kv_heads,
                         swa_num_attention_heads=heads, swa_num_key_value_heads=kv_heads,
                         moe_intermediate_size=hidden // 2, n_routed_experts=experts, **kw)


PRESETS = {
    # ~10M parameters, byte vocabulary, 12 layers in 4 blocks of 3. Laptop check only.
    "smoke": InklingConfig(
        vocab_size=256, unpadded_vocab_size=256, hidden_size=128, num_hidden_layers=12,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=32,
        sliding_window_size=64, global_every=3, d_rel=8, rel_extent=64,
        qk_l2_clip_threshold=128**0.5, rel_l2_clip_threshold=32**0.5,
        n_layers_per_attn_res_block=3, n_routed_experts=32, num_experts_per_tok=4, moe_intermediate_size=64,
        loop_layer_index_start=3, loop_layer_index_end=9, loop_gate_rank=16,
        block_size=256, hp_tail_layers=2),
    # Inkling 2 Mini's layout scaled to ~5B stored parameters (after pruning to 128 experts).
    "5b": _scaled(1024, 12, 4, grad_checkpoint=True),
    # The same model before pruning: 384 experts per layer.
    "5b-pretrain": _scaled(1024, 12, 4, experts=384, grad_checkpoint=True),
    # Inkling 2 Mini as released (text decoder): 26.1B stored, 128 experts per layer.
    "mini": InklingConfig(grad_checkpoint=True),
    # Inkling 2 Mini as pretrained, before pruning: 384 experts per layer.
    "mini-pretrain": InklingConfig(n_routed_experts=384, grad_checkpoint=True),
}


# ----------------------------------------------------------------------------------------
# small pieces
# ----------------------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (self.weight * rms_norm(x.float(), self.eps)).to(x.dtype)


class NormedEmbedding(nn.Embedding):
    def __init__(self, n: int, dim: int, eps: float):
        super().__init__(n, dim)
        self.embed_norm = RMSNorm(dim, eps)

    def forward(self, idx):
        return self.embed_norm(super().forward(idx))


class ScaledL2Clip(nn.Module):
    """Learned per-head scale, then shrink any head vector longer than `threshold` back to it.
    With log_scaling, positions past n_floor are scaled up by 1 + alpha * log(position / n_floor)."""

    def __init__(self, cfg: InklingConfig, heads: int, dim: int, threshold: float, log_scaling: bool):
        super().__init__()
        self.threshold, self.log_scaling = threshold, log_scaling
        self.alpha, self.n_floor = cfg.log_scaling_alpha, cfg.log_scaling_n_floor
        self.scale = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x, pos):  # x [B, T, heads, dim]
        xf = x.float() * self.scale.float()
        norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
        xf = xf * (self.threshold / (norm + 1e-6)).clamp(max=1.0)
        if self.log_scaling:
            tau = 1.0 + self.alpha * torch.log(((pos + 1).float() / self.n_floor).clamp(min=1.0))
            xf = xf * tau[None, :, None, None]
        return xf.to(x.dtype)


class RelativePositionBias(nn.Module):
    """Content-dependent relative position bias: each token mixes a trained bank of
    bias-versus-distance profiles. Zero outside 0 <= distance < rel_extent."""

    def __init__(self, cfg: InklingConfig, heads: int, rel_extent: int, log_scaling: bool):
        super().__init__()
        self.heads, self.rel_extent = heads, rel_extent
        self.linear = nn.Linear(cfg.hidden_size, heads * cfg.d_rel, bias=False)
        self.norm = ScaledL2Clip(cfg, heads, cfg.d_rel, cfg.rel_l2_clip_threshold, log_scaling)
        self.proj = nn.Parameter(torch.empty(cfg.d_rel, rel_extent))

    def forward(self, h, dist, pos):  # h [B, T, H], dist [T, T]
        B, T, _ = h.shape
        rel = self.norm(self.linear(h).view(B, T, self.heads, -1), pos)
        rel_logits = (rel.float() @ self.proj.float()).transpose(1, 2)              # [B, heads, T, extent]
        index = dist.clamp(0, self.rel_extent - 1).expand(B, self.heads, -1, -1)
        bias = rel_logits.gather(-1, index)
        return bias.masked_fill((dist < 0) | (dist >= self.rel_extent), 0.0)


class ShortConvolution(nn.Module):
    """Depthwise causal convolution over the last few tokens, plus a residual."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv1d = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1, bias=False)

    def forward(self, x, dynamic_input=None):
        # Same result as the depthwise F.conv1d of the reference, written as a sum of shifted
        # copies: tap K-1 multiplies the current token, tap 0 the token K-1 steps back.
        xf = x.float()
        T, K = xf.shape[1], self.kernel
        w = self.conv1d.weight.float()[:, 0, :]                                     # [dim, K]
        padded = F.pad(xf, (0, 0, K - 1, 0))
        y = sum(padded[:, k : k + T] * w[:, k] for k in range(K))
        return ((y + xf) / math.sqrt(K + 1)).to(x.dtype)


class DynamicShortConvolution(nn.Module):
    """Short causal convolution whose taps are static per-channel weights plus per-token weights
    projected from the layer input (shared within a key/value head). Keys and values of the
    global-attention layers."""

    def __init__(self, dim: int, input_dim: int, heads: int, kernel: int):
        super().__init__()
        self.kernel, self.heads = kernel, heads
        self.static_weight = nn.Parameter(torch.zeros(kernel, dim))
        self.weight_proj = nn.Linear(input_dim, heads * kernel, bias=False)

    def forward(self, x, dynamic_input):
        B, T, C = x.shape
        K = self.kernel
        dyn = self.weight_proj(dynamic_input).float().view(B, T, self.heads, 1, K)
        weights = self.static_weight.float().T.reshape(self.heads, -1, K) + dyn
        xf = x.float()
        ctx = F.pad(xf.transpose(1, 2), (K - 1, 0))                                  # [B, C, T + K - 1]
        windows = ctx.unfold(-1, K, 1).flip(-1).transpose(1, 2)                     # [B, T, C, K], tap 0 = current
        windows = windows.reshape(B, T, self.heads, -1, K)
        y = (windows * weights).sum(-1).reshape(B, T, C)
        return ((y + xf) / math.sqrt(2 * K + 1)).to(x.dtype)


class QueryProjection(nn.Module):
    def __init__(self, cfg, heads, head_dim, log_scaling):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(cfg.hidden_size, heads * head_dim, bias=False)
        self.norm = ScaledL2Clip(cfg, heads, head_dim, cfg.qk_l2_clip_threshold, log_scaling)

    def forward(self, h, pos):
        s = self.linear(h)
        return self.norm(s.view(*s.shape[:-1], -1, self.head_dim), pos).transpose(1, 2)  # [B, heads, T, hd]


class KeyValueProjection(nn.Module):
    def __init__(self, cfg, heads, head_dim, is_key, dynamic):
        super().__init__()
        self.head_dim, self.is_key, self.eps = head_dim, is_key, cfg.rms_norm_eps
        self.threshold = cfg.qk_l2_clip_threshold
        self.linear = nn.Linear(cfg.hidden_size, heads * head_dim, bias=False)
        self.sconv = (DynamicShortConvolution(heads * head_dim, cfg.hidden_size, heads, cfg.conv_kernel_size)
                      if dynamic else ShortConvolution(heads * head_dim, cfg.conv_kernel_size))

    def forward(self, h):
        s = self.sconv(self.linear(h), h)
        s = s.view(*s.shape[:-1], -1, self.head_dim)
        if self.is_key:  # keys: clip the L2 norm. values: RMS-normalise.
            sf = s.float()
            norm = torch.linalg.vector_norm(sf, dim=-1, keepdim=True)
            s = (sf * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(s.dtype)
        else:
            s = rms_norm(s, self.eps)
        return s.transpose(1, 2)


class GatedOutputProjection(nn.Module):
    def __init__(self, cfg, heads, head_dim):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, heads * head_dim, bias=False)
        self.linear = nn.Linear(heads * head_dim, cfg.hidden_size, bias=False)

    def forward(self, attn_out, h):
        gate = torch.sigmoid(self.gate_proj(h).float())
        return self.linear((attn_out.float() * gate).to(attn_out.dtype))


class Attention(nn.Module):
    """Grouped-query attention with a content-dependent relative position bias, an attention
    sink (a learned logit a head can dump weight on instead of any token) and a gated output."""

    def __init__(self, cfg: InklingConfig, layer_idx: int):
        super().__init__()
        g = cfg.is_global(layer_idx)
        self.heads = cfg.num_attention_heads if g else cfg.swa_num_attention_heads
        self.kv_heads = cfg.num_key_value_heads if g else cfg.swa_num_key_value_heads
        self.head_dim = cfg.head_dim if g else cfg.swa_head_dim
        rel_extent = cfg.rel_extent if g else cfg.sliding_window_size
        self.q_proj = QueryProjection(cfg, self.heads, self.head_dim, log_scaling=g)
        self.k_proj = KeyValueProjection(cfg, self.kv_heads, self.head_dim, is_key=True, dynamic=g)
        self.v_proj = KeyValueProjection(cfg, self.kv_heads, self.head_dim, is_key=False, dynamic=g)
        self.o_proj = GatedOutputProjection(cfg, self.heads, self.head_dim)
        self.sink_proj = nn.Linear(cfg.hidden_size, self.heads, bias=False)
        self.relative_position_bias = RelativePositionBias(cfg, self.heads, rel_extent, log_scaling=g)

    def forward(self, q_in, k_in, v_in, mask, dist, pos):
        B, T, _ = q_in.shape
        q, k, v = self.q_proj(q_in, pos), self.k_proj(k_in), self.v_proj(v_in)
        group = self.heads // self.kv_heads
        q = q.reshape(B, self.kv_heads, group, T, self.head_dim)
        scores = torch.matmul(q, k.unsqueeze(2).transpose(-1, -2)).reshape(B, self.heads, T, T)
        scores = scores * self.head_dim**-0.5 + self.relative_position_bias(q_in, dist, pos) + mask
        sinks = self.sink_proj(q_in).transpose(1, 2).unsqueeze(-1)                   # [B, heads, T, 1]
        attn = F.softmax(torch.cat([scores, sinks], dim=-1).float(), dim=-1)[..., :-1].to(v.dtype)
        out = torch.matmul(attn.reshape(B, self.kv_heads, group, T, T), v.unsqueeze(2))
        out = out.reshape(B, self.heads, T, self.head_dim).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out, q_in)


# ----------------------------------------------------------------------------------------
# mixture of experts
# ----------------------------------------------------------------------------------------

class Router(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        # Added to the scores only when picking experts. Not trained by gradient.
        self.register_buffer("e_score_correction_bias", torch.zeros(cfg.n_routed_experts))


def tml_quantizer_factory(role):
    """Transformer Engine quantizer per tensor slot, following the published 4-bit recipe:

        forward GEMM   input (NVFP4, one outer scale per token, Four-Over-Six) x weight (NVFP4, 1x16 scales)
        backward GEMMs MXFP8, on operands re-quantized from the dequantized 4-bit forward values

    This is Transformer Engine's own `nvfp4_row_scaled_fwd_mxfp8_bwd_factory` plus Four-Over-Six on
    the activations. One piece of the published recipe is missing: Transformer Engine's MXFP8
    quantizer has no stochastic-rounding option, so the re-quantization rounds to nearest.
    """
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.custom_recipes.quantizer_factories import mxfp8_factory
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    plain = dict(with_rht=False, with_post_rht_amax=False, with_2d_quantization=False, stochastic_rounding=False)
    kind = role.tensor_type if role is not None else ""
    if kind == "input":
        # 256 instead of 448 as the block-scale bound leaves room for the 1.5x larger scale that
        # mapping a block to 4 instead of 6 needs; it is what the stock recipe uses with 4/6.
        forward = NVFP4Quantizer(**plain, row_scaled_nvfp4=True, nvfp4_use_4over6=True, nvfp4_e4m3_max=256,
                                 nvfp4_4over6_err_mode="MSE")
    elif kind == "weight":
        forward = NVFP4Quantizer(**plain)
    else:  # grad_output, and any slot this factory does not know: MXFP8
        return mxfp8_factory(role)
    return te.HybridQuantizer(rowwise_quantizer=forward, columnwise_quantizer=mxfp8_factory(role),
                              columnwise_source="rowwise_dequantized")


def te_recipe(mode: str, style: str):
    """Transformer Engine recipe for one expert layer (None = high precision).

    "tml"        tml_quantizer_factory above: NVFP4 forward, MXFP8 backward, Four-Over-Six.
    "rowscaled"  the same without Four-Over-Six, using only the factory Transformer Engine ships.
    "nvidia"     Transformer Engine's default NVFP4 recipe: 16x16 weight tiles, Hadamard
                 transforms, NVFP4 backward GEMMs with stochastic rounding.

    "tml" and "rowscaled" use Transformer Engine's fine-grained recipes, which it marks
    experimental. The last layers (mode "mxfp8") use the stock MXFP8 recipe in every case.
    """
    if mode == "none":
        return None
    from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
    if mode == "mxfp8":
        return MXFP8BlockScaling()
    if style == "nvidia":
        return NVFP4BlockScaling()
    try:
        from transformer_engine.common.recipe import CustomRecipe
        if style == "rowscaled":
            from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
                nvfp4_row_scaled_fwd_mxfp8_bwd_factory)
            return CustomRecipe(qfactory=nvfp4_row_scaled_fwd_mxfp8_bwd_factory)
        tml_quantizer_factory(None)  # fail here, not mid-training, if an option is missing
        return CustomRecipe(qfactory=tml_quantizer_factory)
    except (ImportError, TypeError) as err:
        raise RuntimeError(f"This Transformer Engine release cannot build the '{style}' recipe ({err}). "
                           "Upgrade it, or run with --te-recipe nvidia.") from err


class Experts(nn.Module):
    """Expert weights. "sim": two stacked parameters, gate_up_proj [E, 2*inner, H] and
    down_proj [E, H, inner]. "te": two GroupedLinear modules holding one weight per expert
    (weight0 .. weight{E-1}); Inkling.canonical_state_dict() stacks them back."""

    def __init__(self, cfg: InklingConfig):
        super().__init__()
        E, H, inner = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        if cfg.backend == "te":
            import transformer_engine.pytorch as te
            init = lambda w: nn.init.normal_(w, mean=0.0, std=cfg.initializer_range)
            self.gate_up = te.GroupedLinear(E, H, 2 * inner, bias=False, init_method=init)
            self.down = te.GroupedLinear(E, inner, H, bias=False, init_method=init)
        else:
            self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * inner, H))
            self.down_proj = nn.Parameter(torch.empty(E, H, inner))


class SparseMoeBlock(nn.Module):
    """Each token goes to its top-k experts by sigmoid router score. The expert GEMMs are the
    4-bit part of the model: their weights (one outer scale per expert) and their input
    activations (one outer scale per token) are quantized on every forward pass."""

    def __init__(self, cfg: InklingConfig, mode: str):
        super().__init__()
        self.experts, self.gate = Experts(cfg), Router(cfg)
        self.mode, self.four_over_six = mode, cfg.four_over_six
        self.backward_mode = cfg.backward if mode != "none" else "exact"
        self.backend = cfg.backend
        self.recipe = te_recipe(mode, cfg.te_recipe) if cfg.backend == "te" else None
        self.top_k, self.route_scale = cfg.num_experts_per_tok, cfg.route_scale
        self.gate_cap, self.up_cap = cfg.mlp_gate_softcap, cfg.mlp_up_softcap
        self.register_buffer("usage", torch.zeros(cfg.n_routed_experts))            # running share of load
        self.register_buffer("step_counts", torch.zeros(cfg.n_routed_experts), persistent=False)
        self.cache_weights = False  # set by the model for looped layers: quantize once per step
        self._wq = None

    def quantized_weights(self):
        if self._wq is not None:
            return self._wq
        wq = (quantize_expert_weight(self.experts.gate_up_proj, self.mode),
              quantize_expert_weight(self.experts.down_proj, self.mode))
        if self.cache_weights:
            self._wq = wq
        return wq

    def route(self, flat):
        scores = torch.sigmoid(F.linear(flat.float(), self.gate.weight.float()))
        index = torch.topk(scores + self.gate.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, index)
        return index, weights / (weights.sum(-1, keepdim=True) + 1e-20) * self.route_scale

    def _experts_te(self, flat, token, sorted_expert, counts):
        """Native path: Transformer Engine quantizes the operands and runs one GEMM per expert on
        consecutive rows of the input, m_splits[e] rows for expert e."""
        import transformer_engine.pytorch as te
        x = flat[token]
        with te.autocast(enabled=self.recipe is not None, recipe=self.recipe):
            gate, up = self.experts.gate_up(x, counts).float().chunk(2, dim=-1)
            act = softcap(gate, self.gate_cap) * torch.sigmoid(gate) * softcap(up, self.up_cap)
            return self.experts.down(act.to(x.dtype), counts).float()

    def _experts_sim(self, flat, token, sorted_expert, counts):
        """Simulated path: quantize-dequantize the operands, then ordinary batched GEMMs."""
        # Cut each expert's group of pairs into chunks of at most C pairs (C = the mean load).
        # All chunks run as one batched GEMM, [G, C, H] x [G, out, H], with each chunk using its
        # expert's weights. A busy expert gets several chunks and an idle one none, so the work
        # stays within 2x of the pairs that exist however uneven the routing is.
        P, E, H, x = token.numel(), counts.numel(), flat.shape[-1], flat
        C = -(-P // E)
        c = counts if x.is_cuda else counts.cpu()                 # small bookkeeping; cheap on the CPU
        start = torch.cumsum(c, 0) - c                            # first sorted position of each expert
        n_chunks = (c + C - 1) // C
        chunk_start = torch.cumsum(n_chunks, 0) - n_chunks
        chunk_expert = torch.repeat_interleave(torch.arange(E, device=c.device), n_chunks)        # [G]
        G = chunk_expert.numel()
        first = start[chunk_expert] + (torch.arange(G, device=c.device) - chunk_start[chunk_expert]) * C
        source = first[:, None] + torch.arange(C, device=c.device)[None, :]                       # [G, C]
        valid = source < (start + c)[chunk_expert][:, None]
        source = source.clamp(max=P - 1)
        rank = torch.arange(P, device=x.device) - start.to(x.device)[sorted_expert]
        slot = (chunk_start.to(x.device)[sorted_expert] + torch.div(rank, C, rounding_mode="floor")) * C + rank % C
        chunk_expert, source, valid = chunk_expert.to(x.device), source.to(x.device), valid.to(x.device)

        xq = quantize_act(flat, self.mode, self.four_over_six)
        padded = xq[token[source]] * valid.unsqueeze(-1)                                        # [G, C, H]
        w_gate_up, w_down = self.quantized_weights()
        if G != E or not bool((n_chunks == 1).all()):
            w_gate_up, w_down = w_gate_up[chunk_expert], w_down[chunk_expert]
        gate, up = qgemm(padded, w_gate_up.to(padded.dtype), self.backward_mode).float().chunk(2, dim=-1)
        act = softcap(gate, self.gate_cap) * torch.sigmoid(gate) * softcap(up, self.up_cap)
        act = quantize_act(act.to(padded.dtype), self.mode, self.four_over_six)
        return qgemm(act, w_down.to(padded.dtype), self.backward_mode).reshape(G * C, H)[slot].float()

    def forward(self, x):
        B, T, H = x.shape
        N, k, E = B * T, self.top_k, self.gate.weight.shape[0]
        flat = x.reshape(N, H)
        index, weights = self.route(flat)

        # Sort the N*k (token, expert) pairs by expert.
        expert = index.reshape(-1)
        counts = torch.zeros(E, dtype=torch.long, device=x.device).scatter_add_(0, expert, torch.ones_like(expert))
        if self.training:
            self.step_counts += counts
        sorted_expert, order = torch.sort(expert)
        token = torch.div(order, k, rounding_mode="floor")
        run = self._experts_te if self.backend == "te" else self._experts_sim
        y = run(flat, token, sorted_expert, counts)                                 # [N*k, H], in sorted order
        y = y * weights.reshape(-1)[order].unsqueeze(-1)
        out = torch.zeros(N, H, dtype=torch.float32, device=x.device).index_add(0, token, y)
        return out.to(x.dtype).view(B, T, H)


# ----------------------------------------------------------------------------------------
# attention residuals and loop messages
# ----------------------------------------------------------------------------------------

class BlockAttentionResidual(nn.Module):
    """A layer input: learned queries attend over the RMS-normalised outputs of the finished
    blocks plus the current block's partial sum. Zero-initialised queries give a plain average."""

    def __init__(self, cfg: InklingConfig, num_queries: int = 1):
        super().__init__()
        self.eps = cfg.rms_norm_eps
        shape = (cfg.hidden_size,) if num_queries == 1 else (num_queries, cfg.hidden_size)
        self.query = nn.Parameter(torch.zeros(*shape))

    def mix(self, query, block_states, partial):
        states = rms_norm(torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float(), self.eps)
        H = states.shape[-1]
        attn = F.softmax(torch.einsum("qh,btsh->btqs", query.float().view(-1, H), states) * H**-0.5, dim=-1)
        return attn @ states                                                         # [B, T, queries, H]

    def forward(self, block_states, partial):
        return self.mix(self.query, block_states, partial)


class MessageAttentionResidual(BlockAttentionResidual):
    """Messages for one looped layer: four queries (for its query, key, value and MLP inputs)
    attend over the previous pass's block outputs."""

    def __init__(self, cfg: InklingConfig):
        super().__init__(cfg)
        n = cfg.loop_layer_index_end - cfg.loop_layer_index_start
        self.query = nn.Parameter(torch.zeros(n, 4, cfg.hidden_size))

    def forward(self, block_states, partial, loop_layer_idx: int):
        return self.mix(self.query[loop_layer_idx], block_states, partial)


class MessageGate(nn.Module):
    """(1 - forget) * hidden + gain * message, both gates low-rank functions of the two inputs.
    Both gate projections start at zero, so a looped pass starts out identical to the first."""

    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.eps = cfg.rms_norm_eps
        self.down_proj = nn.Linear(2 * cfg.hidden_size, cfg.loop_gate_rank, bias=False)
        self.forget_gate_proj = nn.Linear(cfg.loop_gate_rank, cfg.hidden_size, bias=False)
        self.message_gate_proj = nn.Linear(cfg.loop_gate_rank, cfg.hidden_size, bias=False)

    def forward(self, hidden, message):
        hidden, message = hidden.float(), message.float()
        z = torch.cat([rms_norm(hidden, self.eps), rms_norm(message, self.eps)], dim=-1)
        z = self.down_proj(z.to(self.down_proj.weight.dtype)).float()
        forget = torch.tanh(self.forget_gate_proj(z))
        gain = torch.tanh(self.message_gate_proj(z))
        return (1 - forget) * hidden + gain * message


class DecoderLayer(nn.Module):
    def __init__(self, cfg: InklingConfig, layer_idx: int):
        super().__init__()
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.is_global = cfg.is_global(layer_idx)
        self.is_loop = cfg.is_loop(layer_idx)
        self.is_block_boundary = layer_idx % cfg.n_layers_per_attn_res_block == 0
        self.attention_input = (BlockAttentionResidual(cfg, 3 if self.is_global else 1) if layer_idx > 0 else None)
        self.mlp_input = BlockAttentionResidual(cfg)
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = SparseMoeBlock(cfg, cfg.expert_mode(layer_idx))
        if self.is_global:
            self.q_input_layernorm = RMSNorm(H, eps)
            self.k_input_layernorm = RMSNorm(H, eps)
            self.v_input_layernorm = RMSNorm(H, eps)
        else:
            self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.post_attention_convolution = ShortConvolution(H, cfg.conv_kernel_size)
        self.post_mlp_convolution = ShortConvolution(H, cfg.conv_kernel_size)
        if self.is_loop:
            self.q_message_gate = MessageGate(cfg)
            self.k_message_gate = MessageGate(cfg)
            self.v_message_gate = MessageGate(cfg)
            self.mlp_message_gate = MessageGate(cfg)

    def forward(self, hidden, block_states, messages, mask, dist, pos):
        dtype = hidden.dtype
        block_residual = hidden
        if messages is not None:
            q_msg, k_msg, v_msg, mlp_msg = messages.unbind(-2)

        if self.is_global:  # separate mixes for query, key and value, each with its own norm
            if self.attention_input is None:
                q = k = v = hidden
            else:
                q, k, v = self.attention_input(block_states, hidden).unbind(-2)
            if messages is not None:
                q, k, v = self.q_message_gate(q, q_msg), self.k_message_gate(k, k_msg), self.v_message_gate(v, v_msg)
            q, k, v = self.q_input_layernorm(q), self.k_input_layernorm(k), self.v_input_layernorm(v)
        else:
            x = hidden if self.attention_input is None else self.attention_input(block_states, hidden)[..., 0, :]
            if messages is None:
                q = k = v = self.input_layernorm(x)
            else:
                q = self.input_layernorm(self.q_message_gate(x, q_msg))
                k = self.input_layernorm(self.k_message_gate(x, k_msg))
                v = self.input_layernorm(self.v_message_gate(x, v_msg))

        attn = self.self_attn(q.to(dtype), k.to(dtype), v.to(dtype), mask, dist, pos)
        attn = self.post_attention_convolution(attn).float()
        if self.is_block_boundary:  # a new block starts: the incoming sum becomes a finished block
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
            residual = attn
        else:
            residual = attn = block_residual.float() + attn

        x = self.mlp_input(block_states, attn)[..., 0, :]
        if messages is not None:
            x = self.mlp_message_gate(x, mlp_msg)
        x = self.mlp(self.post_attention_layernorm(x).to(dtype))
        x = self.post_mlp_convolution(x)
        return (residual + x.float()).to(dtype)


# ----------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------

class Inkling(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.loop_layer_index_start % cfg.n_layers_per_attn_res_block == 0
        assert cfg.loop_layer_index_end % cfg.n_layers_per_attn_res_block == 0
        self.embed_tokens = NormedEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.message_input = MessageAttentionResidual(cfg)
        self.final_input = BlockAttentionResidual(cfg)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.unpadded_vocab_size, bias=False)
        self.apply(self._init)
        for layer in self.layers:
            layer.mlp.cache_weights = layer.is_loop and not cfg.grad_checkpoint

    def _init(self, m):
        std = self.cfg.initializer_range
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, RelativePositionBias):
            nn.init.normal_(m.proj, mean=0.0, std=std)
        elif isinstance(m, Experts) and self.cfg.backend != "te":
            nn.init.normal_(m.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(m.down_proj, mean=0.0, std=std)
        elif isinstance(m, Router):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, MessageGate):  # closed gates: a looped pass starts as an exact repeat
            nn.init.normal_(m.down_proj.weight, mean=0.0, std=std)
            nn.init.zeros_(m.forget_gate_proj.weight)
            nn.init.zeros_(m.message_gate_proj.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def moe_blocks(self):
        return [layer.mlp for layer in self.layers]

    def loop_gate_strength(self) -> float:
        """Mean |weight| of the message gain projections. Zero at initialisation."""
        w = [layer.q_message_gate.message_gate_proj.weight.abs().mean() for layer in self.layers if layer.is_loop]
        w += [layer.mlp_message_gate.message_gate_proj.weight.abs().mean() for layer in self.layers if layer.is_loop]
        return torch.stack(w).mean().item()

    @torch.no_grad()
    def balance_step(self, rate: float, usage_decay: float = 0.99, sync=None):
        """Call once per optimizer step. Folds this step's expert loads into the running usage
        (used later to choose which experts to prune) and nudges each router's selection bias
        towards an even load: +rate for under-used experts, -rate for over-used ones.
        `sync` sums the counts across processes (pass torch.distributed.all_reduce under DDP)."""
        for moe in self.moe_blocks():
            counts = moe.step_counts.clone()
            if sync is not None:
                sync(counts)
            total = counts.sum()
            if total > 0:
                moe.usage.mul_(usage_decay).add_(counts / total, alpha=1 - usage_decay)
                if rate:
                    moe.gate.e_score_correction_bias.add_(rate * torch.sign(counts.mean() - counts))
            moe.step_counts.zero_()

    def _run(self, layer, hidden, block_states, messages, masks, dist, pos):
        mask = masks[layer.is_global]
        if self.cfg.grad_checkpoint and self.training:
            out = checkpoint(layer, hidden, block_states, messages, mask, dist, pos, use_reentrant=False)
        else:
            out = layer(hidden, block_states, messages, mask, dist, pos)
        if layer.is_block_boundary:
            block_states = torch.cat([block_states, hidden.unsqueeze(-2)], dim=-2)
        return out, block_states

    def forward(self, idx, targets=None, n_loops=None):
        cfg = self.cfg
        n_loops = n_loops or cfg.loop_num_passes
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        dist = pos[:, None] - pos[None, :]
        full = torch.zeros(T, T, device=idx.device).masked_fill(dist < 0, float("-inf"))
        masks = {True: full, False: full.masked_fill(dist >= cfg.sliding_window_size, float("-inf"))}
        ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        args = (masks, dist, pos)

        hidden = self.embed_tokens(idx)
        block_states = hidden.new_zeros(B, T, 0, cfg.hidden_size)
        for layer in self.layers[:ls]:
            hidden, block_states = self._run(layer, hidden, block_states, None, *args)

        pre_hidden, pre_blocks = hidden, block_states      # the prelude output feeds every pass
        n_pre = pre_blocks.shape[-2]
        for layer in self.layers[ls:le]:
            hidden, block_states = self._run(layer, hidden, block_states, None, *args)
        for _ in range(1, n_loops):
            prev_blocks, prev_hidden = block_states[..., n_pre + 1:, :], hidden
            hidden, block_states = pre_hidden, pre_blocks
            for i, layer in enumerate(self.layers[ls:le]):
                messages = self.message_input(prev_blocks, prev_hidden, i)
                hidden, block_states = self._run(layer, hidden, block_states, messages, *args)
        for layer in self.layers[ls:le]:
            layer.mlp._wq = None
        if n_loops == 1:  # keep the loop plumbing in the graph so DDP sees a gradient for it
            unused = [self.message_input.query] + [p for layer in self.layers[ls:le] for n, p in
                                                   layer.named_parameters() if "message_gate" in n]
            hidden = hidden + 0.0 * sum(p.sum() for p in unused)

        for layer in self.layers[le:]:
            hidden, block_states = self._run(layer, hidden, block_states, None, *args)
        hidden = self.norm(self.final_input(block_states, hidden)[..., 0, :].to(hidden.dtype))
        logits = softcap(self.lm_head(hidden).float(), cfg.final_logit_softcapping)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, n_loops=None):
        for _ in range(max_new_tokens):  # no KV cache: recomputes the context every token
            logits, _ = self(idx[:, -self.cfg.block_size:], n_loops=n_loops)
            logits = logits[:, -1] / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    _TE_PARTS = (("gate_up", "gate_up_proj"), ("down", "down_proj"))

    def canonical_state_dict(self) -> dict:
        """State dict in the "sim" layout whatever the backend, so checkpoints, pruning and
        export do not depend on how the expert GEMMs were run."""
        sd = self.state_dict()
        if self.cfg.backend != "te":
            return sd
        E = self.cfg.n_routed_experts
        for i in range(self.cfg.num_hidden_layers):
            pre = f"layers.{i}.mlp.experts."
            for module, stacked in self._TE_PARTS:
                sd[pre + stacked] = torch.stack([sd[f"{pre}{module}.weight{e}"].float() for e in range(E)])
            for key in [k for k in sd if k.startswith((pre + "gate_up.", pre + "down."))]:
                del sd[key]  # per-expert weights and any Transformer Engine bookkeeping
        return sd

    def load_canonical(self, sd: dict):
        if self.cfg.backend != "te":
            return self.load_state_dict(sd)
        sd = dict(sd)
        for i in range(self.cfg.num_hidden_layers):
            pre = f"layers.{i}.mlp.experts."
            for module, stacked in self._TE_PARTS:
                for e, w in enumerate(sd.pop(pre + stacked)):
                    sd[f"{pre}{module}.weight{e}"] = w
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if ".experts." not in k]  # Transformer Engine's own buffers
        assert not missing and not unexpected, (missing, unexpected)


def param_counts(cfg: InklingConfig, n_loops=None) -> dict:
    """Parameter and per-token weight counts computed from the config alone, so the 26B and
    75B configurations can be sized without building them.

    stored   every trainable parameter
    active   what one token touches in a single pass: 8 experts per layer, not all of them
    applied  matrix weights a token is multiplied through, counting looped layers once per pass
             (the embedding is a lookup, so it is left out; the output head is included)
    """
    H, K, E = cfg.hidden_size, cfg.conv_kernel_size, cfg.n_routed_experts
    expert = 3 * H * cfg.moe_intermediate_size
    gates = 4 * (2 * H * cfg.loop_gate_rank + 2 * cfg.loop_gate_rank * H)
    stored = active = applied = 0
    for i in range(cfg.num_hidden_layers):
        g = cfg.is_global(i)
        nh = cfg.num_attention_heads if g else cfg.swa_num_attention_heads
        nkv = cfg.num_key_value_heads if g else cfg.swa_num_key_value_heads
        hd = cfg.head_dim if g else cfg.swa_head_dim
        ext = cfg.rel_extent if g else cfg.sliding_window_size
        qd, kvd = nh * hd, nkv * hd
        gemm = H * qd + 2 * H * kvd + 2 * H * qd + H * nh + H * nh * cfg.d_rel + E * H   # q, k, v, gate+out, sink, rel, router
        gemm += 2 * H * nkv * K if g else 0                                              # dynamic conv projections
        small = nh * hd + nh * cfg.d_rel + cfg.d_rel * ext + 2 * H * K                   # scales, rel bank, post convs
        small += 2 * K * kvd                                                             # k/v conv taps
        small += (4 if g else 2) * H + ((3 if g else 1) * H if i > 0 else 0) + H         # norms, input queries
        loop = gates if cfg.is_loop(i) else 0
        passes = (n_loops or cfg.loop_num_passes) if cfg.is_loop(i) else 1
        stored += gemm + small + loop + E * expert
        active += gemm + small + loop + cfg.num_experts_per_tok * expert
        applied += passes * (gemm + cfg.num_experts_per_tok * expert) + (passes - 1) * loop
    n_loop = cfg.loop_layer_index_end - cfg.loop_layer_index_start
    glob = cfg.vocab_size * H + H + n_loop * 4 * H + H + H + cfg.unpadded_vocab_size * H
    return {"stored": stored + glob, "active": active + glob, "applied": applied + cfg.unpadded_vocab_size * H,
            "experts": cfg.num_hidden_layers * E * expert, "expert_one": expert,
            "embeddings": cfg.vocab_size * H + cfg.unpadded_vocab_size * H}
