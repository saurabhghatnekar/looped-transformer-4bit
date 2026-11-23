"""Simulated NVFP4 / MXFP8 quantization for training with 4-bit forward passes.

Formats (per NVIDIA Transformer Engine docs and the NVFP4 pretraining paper):

  NVFP4   x = x_e2m1 * s_block * s_global
          x_e2m1   4-bit E2M1 value, one of +/-{0, .5, 1, 1.5, 2, 3, 4, 6}
          s_block  FP8 E4M3 scale shared by 16 consecutive elements
          s_global FP32 scale shared by a whole tensor (or one row of it)
          -> (4*16 + 8) / 16 = 4.5 bits per value

  MXFP8   x = x_e4m3 * 2^k, with one power-of-two scale per 32 elements
          -> (8*32 + 8) / 32 = 8.25 bits per value

Everything here runs in ordinary float32 maths, so it works on CPU, MPS and any
CUDA GPU. It reproduces the *values* a 4-bit forward pass computes with, not the
speed: the speedup needs native NVFP4 GEMMs on Blackwell (see backend="te").

Blocks run along the last dimension, which is the dimension a linear layer
contracts over (in_features) for both the activations and the weight matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FP4_MAX = 6.0
E4M3_MAX = 448.0
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
_TINY = 1e-30


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid +/-{0, .5, 1, 1.5, 2, 3, 4, 6}; ties to even, saturating."""
    a = x.abs().clamp(max=FP4_MAX)
    q = torch.where(
        a <= 2.0,
        torch.round(a * 2.0) / 2.0,                                  # step 0.5
        torch.where(a <= 4.0, torch.round(a), torch.round(a / 2.0) * 2.0),  # step 1, step 2
    )
    return torch.copysign(q, x)


def round_e4m3(v: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Round non-negative values to FP8 E4M3 (3 mantissa bits, max 448, subnormals to 2^-9).

    stochastic: round up with probability equal to the fractional part, so the result is
    unbiased. Used for the operands of backward GEMMs.
    """
    v = v.clamp(min=0.0, max=E4M3_MAX)
    e = torch.floor(torch.log2(v.clamp(min=2.0**-9))).clamp(min=-6.0, max=8.0)
    step = torch.pow(2.0, e - 3.0)
    r = v / step
    r = torch.floor(r + torch.rand_like(r)) if stochastic else torch.round(r)
    return (r * step).clamp(max=E4M3_MAX)


def _to_blocks(x: torch.Tensor, block: int):
    pad = (-x.shape[-1]) % block
    if pad:
        x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, block), pad


def _from_blocks(xb: torch.Tensor, pad: int) -> torch.Tensor:
    x = xb.reshape(*xb.shape[:-2], -1)
    return x[..., : x.shape[-1] - pad] if pad else x


def nvfp4_scales(xb: torch.Tensor, outer: str, target: float, headroom: float):
    """Return (s_block [..., nb, 1], s_global) for blocked input xb [..., nb, 16]."""
    block_amax = xb.abs().amax(dim=-1, keepdim=True)
    if outer == "tensor":
        g_amax = block_amax.amax()
    elif outer == "row":  # one outer scale per row: per token for activations
        g_amax = block_amax.amax(dim=(-2, -1), keepdim=True)
    elif outer == "lead":  # one outer scale per leading slice: per expert for [E, out, in] weights
        g_amax = block_amax.amax(dim=tuple(range(1, block_amax.dim())), keepdim=True)
    else:
        raise ValueError(outer)
    s_global = (g_amax / (E4M3_MAX * headroom)).clamp(min=_TINY)
    s_block = round_e4m3(block_amax / target / s_global)
    return s_block, s_global


def nvfp4_quantize(x: torch.Tensor, outer: str = "tensor", four_over_six: bool = False) -> torch.Tensor:
    """Quantize-dequantize x to NVFP4. Returns a float tensor holding representable values.

    four_over_six: per block, also try mapping the block max to 4 instead of 6 and keep
    whichever has lower squared error ("Four Over Six"). Used for activations.
    """
    xb, pad = _to_blocks(x, NVFP4_BLOCK)
    # With 4/6 the block scale can be 1.5x larger, so leave room for it in the outer scale.
    headroom = 4.0 if four_over_six else FP4_MAX

    def quant(target: float) -> torch.Tensor:
        s_block, s_global = nvfp4_scales(xb, outer, target, headroom)
        scale = s_block * s_global
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return round_e2m1(xb / safe) * scale  # a block whose scale underflows flushes to zero

    q = quant(FP4_MAX)
    if four_over_six:
        q4 = quant(4.0)
        err6 = (q - xb).square().sum(dim=-1, keepdim=True)
        err4 = (q4 - xb).square().sum(dim=-1, keepdim=True)
        q = torch.where(err4 < err6, q4, q)
    return _from_blocks(q, pad)


def mxfp8_quantize(x: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Quantize-dequantize x to MXFP8: E4M3 elements, power-of-two scale per 32 elements."""
    xb, pad = _to_blocks(x, MXFP8_BLOCK)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=_TINY)
    k = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(min=-127.0, max=127.0)
    scale = torch.pow(2.0, k)
    v = xb / scale
    q = torch.copysign(round_e4m3(v.abs(), stochastic), v) * scale
    return _from_blocks(q, pad)


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward value exactly q, gradient of identity."""
    return q.detach() + (x - x.detach())


class _GemmMXFP8Backward(torch.autograd.Function):
    """y = x @ w^T per leading slice, x [E, M, in], w [E, out, in].

    The forward product is exact on whatever (already quantized) operands it is given. The two
    backward products run on MXFP8 operands: each operand is transposed so that its blocks of
    32 lie along the dimension that product contracts over, then re-quantized with stochastic
    rounding. This is the "dequantize -> transpose -> re-quantize" step of 4-bit training.
    """

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return torch.bmm(x, w.transpose(1, 2))

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g, gx, gw = g.float(), None, None
        if ctx.needs_input_grad[0]:  # dL/dx = g @ w, contracts over `out`
            gq = mxfp8_quantize(g, stochastic=True)
            wt = mxfp8_quantize(w.float().transpose(1, 2), stochastic=True)  # [E, in, out]
            gx = torch.bmm(gq, wt.transpose(1, 2)).to(x.dtype)
        if ctx.needs_input_grad[1]:  # dL/dw = g^T @ x, contracts over the tokens
            gt = mxfp8_quantize(g.transpose(1, 2), stochastic=True)          # [E, out, M]
            xt = mxfp8_quantize(x.float().transpose(1, 2), stochastic=True)  # [E, in, M]
            gw = torch.bmm(gt, xt.transpose(1, 2)).to(w.dtype)
        return gx, gw


def qgemm(x: torch.Tensor, w: torch.Tensor, backward: str = "exact") -> torch.Tensor:
    """x [E, M, in] times w [E, out, in]^T. backward: "exact" (autograd) or "mxfp8"."""
    if backward == "mxfp8":
        return _GemmMXFP8Backward.apply(x, w)
    return torch.bmm(x, w.transpose(1, 2))


def quantize_act(x: torch.Tensor, mode: str, four_over_six: bool = True) -> torch.Tensor:
    """Quantize activations feeding a GEMM (one outer scale per token), straight-through."""
    if mode == "none":
        return x
    xf = x.float()
    q = nvfp4_quantize(xf, outer="row", four_over_six=four_over_six) if mode == "nvfp4" else mxfp8_quantize(xf)
    return ste(xf, q).to(x.dtype)


def quantize_expert_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    """Quantize stacked expert weights [E, out, in] (one outer scale per expert), straight-through."""
    if mode == "none":
        return w
    q = nvfp4_quantize(w, outer="lead") if mode == "nvfp4" else mxfp8_quantize(w)
    return ste(w, q)


class QLinear(nn.Linear):
    """Bias-free linear layer whose forward pass multiplies quantized activations by
    quantized weights. Master weights stay in float32 for the optimizer.

    The gradient that reaches the weights is g^T x_q and the gradient passed back is
    g w_q, i.e. the backward pass sees the same quantized operands the forward used.
    (The backward GEMMs themselves are full precision here; on Blackwell they would be
    MXFP8 with stochastic rounding.)

    mode: "none" | "nvfp4" | "mxfp8"
    """

    def __init__(self, in_features: int, out_features: int, mode: str = "nvfp4", four_over_six: bool = True):
        super().__init__(in_features, out_features, bias=False)
        assert mode in ("none", "nvfp4", "mxfp8"), mode
        self.mode = mode
        self.four_over_six = four_over_six
        self.cache_weight = False  # set by the model for layers applied several times per step
        self._wq = None

    def quantized_weight(self) -> torch.Tensor:
        if self.cache_weight and self._wq is not None:
            return self._wq
        w = self.weight
        q = nvfp4_quantize(w, outer="tensor") if self.mode == "nvfp4" else mxfp8_quantize(w)
        wq = ste(w, q)
        if self.cache_weight:
            self._wq = wq
        return wq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return F.linear(x, self.weight)
        xf = x.float()
        if self.mode == "nvfp4":
            xq = nvfp4_quantize(xf, outer="row", four_over_six=self.four_over_six)
        else:
            xq = mxfp8_quantize(xf)
        return F.linear(ste(xf, xq).to(x.dtype), self.quantized_weight().to(x.dtype))

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, mode={self.mode}"


# ---------------------------------------------------------------------------
# Packing: turn a trained float weight into real 4-bit codes + scales and back.
# This is what a deployed checkpoint stores, and dequantizing it gives exactly
# the weights the model trained its forward pass against.
# ---------------------------------------------------------------------------

_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4_pack(w: torch.Tensor) -> dict:
    """Pack a 2-D weight into NVFP4: uint8 nibble pairs, float32-held E4M3 block scales, FP32 global."""
    w = w.detach().float().cpu()
    xb, pad = _to_blocks(w, NVFP4_BLOCK)
    s_block, s_global = nvfp4_scales(xb, "tensor", FP4_MAX, FP4_MAX)
    scale = s_block * s_global
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = round_e2m1(xb / safe)
    mag = torch.bucketize(q.abs(), _E2M1_GRID)  # exact grid values -> index 0..7
    codes = (mag | ((q < 0).long() << 3)).to(torch.uint8).reshape(-1)
    if codes.numel() % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    packed = codes[0::2] | (codes[1::2] << 4)
    return {
        "codes": packed,
        "s_block": s_block.squeeze(-1).contiguous(),  # E4M3-representable; 1 byte each on disk
        "s_global": s_global.reshape(1),
        "shape": torch.tensor(w.shape),
    }


def nvfp4_unpack(p: dict) -> torch.Tensor:
    shape = [int(s) for s in p["shape"]]
    padded = shape[-1] + ((-shape[-1]) % NVFP4_BLOCK)
    n = shape[0] * padded
    codes = torch.stack([p["codes"] & 0xF, p["codes"] >> 4], dim=1).reshape(-1)[:n].long()
    mag = _E2M1_GRID[codes & 0x7]
    q = torch.where((codes >> 3) == 1, -mag, mag).reshape(shape[0], -1, NVFP4_BLOCK)
    w = q * (p["s_block"].unsqueeze(-1) * p["s_global"])
    return w.reshape(shape[0], padded)[:, : shape[-1]]
