"""Checks for the quantizer and the loop wiring. Run: python tests/test_core.py"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopfp4 import Config, LoopedTransformer  # noqa: E402
from loopfp4.nvfp4 import (QLinear, mxfp8_quantize, nvfp4_pack, nvfp4_quantize, nvfp4_scales,  # noqa: E402
                           nvfp4_unpack, round_e2m1, round_e4m3, _to_blocks)

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
torch.manual_seed(0)


def e4m3_values():
    vals = [m * 2.0**-9 for m in range(8)]                                   # zero + subnormals
    vals += [(1 + m / 8) * 2.0**e for e in range(-6, 9) for m in range(8)]   # normals
    return torch.tensor(sorted(v for v in set(vals) if v <= 448.0))


def test_e2m1_grid_and_ties():
    x = torch.linspace(-8, 8, 20001)
    q = round_e2m1(x)
    assert torch.isin(q.abs(), GRID).all()
    nearest = GRID[(x.abs().clamp(max=6).unsqueeze(-1) - GRID).abs().argmin(-1)]
    off = (q.abs() - nearest).abs() > 0
    # the only disagreements allowed are exact ties, which must resolve to an even mantissa
    assert ((x.abs()[off].unsqueeze(-1) - GRID).abs().sort(-1).values[:, :2].diff().abs() < 1e-6).all()
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert torch.equal(round_e2m1(ties), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]))
    assert torch.equal(round_e2m1(-ties), -torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]))


def test_e4m3_representable():
    table = e4m3_values()
    v = torch.cat([torch.rand(5000) * 448, torch.rand(5000) * 1.0, torch.rand(5000) * 2.0**-5])
    q = round_e4m3(v)
    assert torch.isin(q, table).all()
    nearest = (v.unsqueeze(-1) - table).abs().min(-1).values
    assert ((q - v).abs() <= nearest + 1e-9).all()  # it is the nearest representable value


def test_nvfp4_structure():
    # Rows span 9 decades (more than the ~6.4 decades one FP4 x FP8 scale pair covers); width not /16.
    x = torch.randn(64, 200) * torch.logspace(-6, 3, 64).unsqueeze(-1)
    for outer in ("tensor", "row"):
        q = nvfp4_quantize(x, outer=outer)
        assert q.shape == x.shape
        xb, _ = _to_blocks(x, 16)
        qb, _ = _to_blocks(q, 16)
        s_block, s_global = nvfp4_scales(xb, outer, 6.0, 6.0)
        scale = s_block * s_global
        ratio = qb / torch.where(scale > 0, scale, torch.ones_like(scale))
        assert (ratio.abs().unsqueeze(-1) - GRID).abs().min(-1).values.max() < 1e-4  # every value is grid * scale
    # per-row outer scales keep small-magnitude rows alive; a single tensor scale flushes them
    rel = lambda q: ((q - x).norm(dim=-1) / x.norm(dim=-1))
    assert rel(nvfp4_quantize(x, "row")).max() < 0.2
    assert rel(nvfp4_quantize(x, "tensor")).max() > 0.9


def test_four_over_six_never_worse():
    x = torch.randn(256, 512)
    e6 = (nvfp4_quantize(x, "row", four_over_six=False) - x).square().mean()
    e46 = (nvfp4_quantize(x, "row", four_over_six=True) - x).square().mean()
    print(f"    gaussian rel. RMS error: nvfp4 {e6.sqrt() / x.std():.4f}, with 4/6 {e46.sqrt() / x.std():.4f}, "
          f"mxfp8 {(mxfp8_quantize(x) - x).square().mean().sqrt() / x.std():.4f}")
    assert e46 < e6
    assert (mxfp8_quantize(x) - x).square().mean() < e46 / 5  # 3 mantissa bits vs 1


def test_ste_gradients():
    lin = QLinear(64, 32, mode="nvfp4")
    x = torch.randn(8, 64, requires_grad=True)
    g = torch.randn(8, 32)
    (lin(x) * g).sum().backward()
    with torch.no_grad():
        xq = nvfp4_quantize(x, "row", four_over_six=True)
        wq = nvfp4_quantize(lin.weight, "tensor")
    assert torch.allclose(lin.weight.grad, g.T @ xq, atol=1e-5)  # wgrad uses quantized activations
    assert torch.allclose(x.grad, g @ wq, atol=1e-5)             # dgrad uses quantized weights


def test_pack_roundtrip():
    w = torch.randn(48, 200) * 0.02
    p = nvfp4_pack(w)
    assert p["codes"].dtype == torch.uint8 and p["codes"].numel() == 48 * 208 // 2
    assert torch.equal(nvfp4_unpack(p), nvfp4_quantize(w, "tensor"))  # deployed weights == trained-against weights


def test_loops_start_as_identity_and_then_differ():
    cfg = Config(vocab_size=64, d_model=64, n_head=4, n_prelude=1, n_loop=2, n_coda=1, block_size=32)
    m = LoopedTransformer(cfg).eval()
    idx = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        one, three = m(idx, n_loops=1)[0], m(idx, n_loops=3)[0]
        assert torch.allclose(one, three, atol=1e-5)      # gates closed: looping changes nothing
        m.g_msg.fill_(0.5)
        assert not torch.allclose(one, m(idx, n_loops=3)[0], atol=1e-3)  # gates open: it does
    assert all(q._wq is None for q in m._cached)          # no stale quantized weights left behind


def test_gates_and_all_weights_get_gradients():
    for ckpt in (False, True):
        cfg = Config(vocab_size=64, d_model=64, n_head=4, n_prelude=1, n_loop=2, n_coda=1, block_size=32,
                     grad_checkpoint=ckpt)
        m = LoopedTransformer(cfg)
        idx = torch.randint(0, 64, (2, 33))
        m(idx[:, :-1], idx[:, 1:])[1].backward()
        assert m.g_msg.grad.abs().sum() > 0 and m.g_pre.grad[1:].abs().sum() > 0
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert missing == ["msg_norm.weight"] or missing == [], missing


def test_formats_by_layer():
    m = LoopedTransformer(Config(n_prelude=6, n_loop=12, n_coda=6, d_model=64, n_head=4))
    modes = [b.mlp.w1.mode for b in list(m.prelude) + list(m.loop) + list(m.coda)]
    assert modes == ["nvfp4"] * 20 + ["mxfp8"] * 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
