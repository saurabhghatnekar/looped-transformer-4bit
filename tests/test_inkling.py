"""Checks for the Inkling model and its 4-bit expert path. Run: python tests/test_inkling.py

(tests/test_reference.py separately checks the architecture against the published reference.)
"""

import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopfp4 import PRESETS, Inkling, InklingConfig, param_counts  # noqa: E402
from loopfp4.inkling import (BlockAttentionResidual, DynamicShortConvolution,  # noqa: E402
                             ShortConvolution, rms_norm, softcap)
from loopfp4.nvfp4 import (mxfp8_quantize, nvfp4_quantize, qgemm, quantize_act, quantize_expert_weight,  # noqa: E402
                           round_e4m3)

torch.manual_seed(0)
TINY = dict(vocab_size=64, unpadded_vocab_size=64, hidden_size=64, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
            sliding_window_size=8, global_every=2, d_rel=4, rel_extent=16,
            n_layers_per_attn_res_block=2, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
            loop_layer_index_start=2, loop_layer_index_end=6, loop_gate_rank=8, block_size=24, hp_tail_layers=2)


def randomise(model):
    """Move every zero-initialised piece off zero so the loop and residual wiring is exercised."""
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().max() == 0:
                p.add_(torch.randn_like(p) * 0.3)
    return model


def test_per_expert_outer_scale():
    # Experts spanning 9 decades: more than the ~6.4 decades one FP4 x FP8 scale pair covers.
    w = torch.randn(6, 24, 40) * torch.logspace(-7, 2, 6).view(6, 1, 1)
    stacked = nvfp4_quantize(w, outer="lead")
    for e in range(6):
        assert torch.equal(stacked[e], nvfp4_quantize(w[e], outer="tensor"))
    shared = nvfp4_quantize(w, outer="tensor")
    err = lambda q: ((q[0] - w[0]).square().mean() / w[0].square().mean()).item()
    assert err(stacked) < 0.05 and err(shared) > 10 * err(stacked)             # one shared scale starves the small expert


def test_stochastic_rounding_is_unbiased_and_representable():
    v = torch.rand(4000) * 300
    table = round_e4m3(torch.linspace(0, 448, 200001)).unique()
    q = torch.stack([round_e4m3(v, stochastic=True) for _ in range(400)])
    assert torch.isin(q, table).all()
    assert ((q.mean(0) - v).abs() / v.clamp(min=1)).max() < 0.02               # mean converges to the input
    x = torch.randn(64, 96)
    mean = torch.stack([mxfp8_quantize(x, stochastic=True) for _ in range(400)]).mean(0)
    assert (mean - x).abs().max() < 0.02
    assert (mxfp8_quantize(x) - x).abs().max() < 0.2


def test_mxfp8_backward_gemm():
    x = torch.randn(3, 50, 32, requires_grad=True)
    w = torch.randn(3, 20, 32, requires_grad=True)
    g = torch.randn(3, 50, 20)
    exact = torch.autograd.grad((qgemm(x, w, "exact") * g).sum(), [x, w])
    assert torch.equal(qgemm(x, w, "mxfp8"), qgemm(x, w, "exact"))             # forward untouched
    runs = [torch.autograd.grad((qgemm(x, w, "mxfp8") * g).sum(), [x, w]) for _ in range(200)]
    for i in range(2):
        one = runs[0][i]
        rel_one = ((one - exact[i]).norm() / exact[i].norm()).item()
        rel_mean = ((torch.stack([r[i] for r in runs]).mean(0) - exact[i]).norm() / exact[i].norm()).item()
        assert 0 < rel_one < 0.06, rel_one                                      # a real, small quantization error
        assert rel_mean < rel_one / 5, (rel_mean, rel_one)                      # that averages out: unbiased


def test_formats_by_layer():
    cfg = PRESETS["mini"]
    assert [cfg.expert_mode(i) for i in range(24)] == ["nvfp4"] * 20 + ["mxfp8"] * 4
    assert [cfg.is_global(i) for i in range(24)] == [(i + 1) % 6 == 0 for i in range(24)]
    assert [i for i in range(24) if cfg.is_loop(i)] == list(range(6, 18)) and cfg.applications() == 48


def test_param_counts_match_built_models_and_the_model_card():
    for cfg in (PRESETS["smoke"], InklingConfig(**TINY)):
        assert param_counts(cfg)["stored"] == Inkling(cfg).num_params()
    mini = param_counts(PRESETS["mini"])
    # Model card: 26.4B total, 3.4B active, including a ~0.3B vision tower that is not built here.
    assert abs(mini["stored"] / 1e9 - 26.07) < 0.01 and abs(mini["active"] / 1e9 - 3.14) < 0.01
    assert abs(param_counts(PRESETS["mini-pretrain"])["stored"] / 1e9 - 75.0) < 0.1


def test_released_config_loads():
    released = {"text_config": {
        "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024, "num_attention_heads": 24,
        "num_key_value_heads": 8, "head_dim": 128, "d_rel": 16, "rel_extent": 1024, "log_scaling_n_floor": 8192,
        "local_layer_ids": [i for i in range(24) if (i + 1) % 6], "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058, "sliding_window_size": 1024, "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 128, "num_experts_per_tok": 8, "intermediate_size": 1152, "route_scale": 8.0,
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}}}
    cfg = InklingConfig.from_hf(released, grad_checkpoint=True)
    assert cfg == PRESETS["mini"]


def test_expert_dispatch_matches_a_per_token_loop():
    for quant in ("none", "nvfp4"):
        model = Inkling(InklingConfig(**TINY, quant=quant, backward="exact")).eval()
        for moe in (model.layers[0].mlp, model.layers[7].mlp):                  # an NVFP4 layer and an MXFP8 layer
            x = torch.randn(2, 12, 64)
            got = moe(x)
            flat = x.reshape(-1, 64)
            index, weights = moe.route(flat)
            w13, w2 = moe.quantized_weights()
            want = torch.zeros_like(flat)
            for n in range(flat.shape[0]):
                xq = quantize_act(flat[n : n + 1], moe.mode)
                for j in range(moe.top_k):
                    e = index[n, j]
                    gate, up = F.linear(xq, w13[e]).chunk(2, dim=-1)
                    act = quantize_act(softcap(gate, 4.0) * torch.sigmoid(gate) * softcap(up, 25.0), moe.mode)
                    want[n] += weights[n, j] * F.linear(act, w2[e])[0]
            assert torch.allclose(got.reshape(-1, 64), want, atol=1e-5), (quant, moe.mode)


def test_model_is_causal():
    model = randomise(Inkling(InklingConfig(**TINY))).eval()
    a = torch.randint(0, 64, (2, 24))
    b = a.clone()
    b[:, 15:] = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :15], lb[:, :15], atol=1e-5)                    # the past ignores the future
    assert (la[:, 15:] - lb[:, 15:]).abs().max() > 1e-3


def test_loops_start_closed_then_open():
    model = Inkling(InklingConfig(**TINY)).eval()
    idx = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        assert torch.equal(model(idx, n_loops=1)[0], model(idx, n_loops=3)[0])  # gates closed at initialisation
        assert model.loop_gate_strength() == 0
        randomise(model)
        assert (model(idx, n_loops=1)[0] - model(idx, n_loops=3)[0]).abs().max() > 1e-4


def test_every_parameter_gets_a_gradient():
    idx = torch.randint(0, 64, (2, 25))
    for ckpt, n_loops in ((False, 3), (True, 3), (False, 1)):
        model = Inkling(InklingConfig(**TINY, grad_checkpoint=ckpt)).train()
        # Parameters whose gradient is exactly zero at the default initialisation because the piece
        # they feed starts at zero. One optimizer step moves the other factor and they come alive.
        dormant = ("message_input.query", "message_gate.down_proj")
        _, loss = model(idx[:, :-1], idx[:, 1:], n_loops=n_loops)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (ckpt, n_loops, name)
            if n_loops > 1 and not any(d in name for d in dormant):
                assert p.grad.abs().max() > 0, (ckpt, name)
        randomise(model).zero_grad()
        model(idx[:, :-1], idx[:, 1:], n_loops=3)[1].backward()
        dead = [n for n, p in model.named_parameters() if p.grad.abs().max() == 0]
        assert not dead, dead


def test_balancing_and_pruning_statistics():
    model = Inkling(InklingConfig(**TINY)).train()
    idx = torch.randint(0, 64, (4, 24))
    moe = model.layers[3].mlp                                                   # a looped layer
    model(idx)
    assert moe.step_counts.sum() == 4 * 24 * 2 * 3                              # tokens x top-k x passes
    busiest = moe.step_counts.argmax().item()
    model.balance_step(rate=0.01)
    assert moe.step_counts.sum() == 0 and abs(moe.usage.sum().item() - 0.01) < 1e-6
    assert moe.usage.argmax().item() == busiest
    assert moe.gate.e_score_correction_bias[busiest] == -0.01                   # over-used expert is discouraged


def test_quantized_training_step_reduces_loss():
    torch.manual_seed(1)
    model = Inkling(dataclasses.replace(InklingConfig(**TINY), quant="nvfp4", backward="mxfp8")).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 64, (8, 25))
    first = None
    for _ in range(30):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.balance_step(1e-3)
    assert loss.item() < 0.7 * first, (first, loss.item())



# ----------------------------------------------------------------------------------------
# Native backend (NVIDIA Transformer Engine). On a machine with Transformer Engine and a
# Blackwell GPU these tests run against the real library. Elsewhere they run against the
# stand-in below, which copies the documented GroupedLinear interface (one weight per GEMM
# named weight0.., forward(inp, m_splits) on consecutive rows) in plain PyTorch. The stand-in
# checks this project's side of the interface, not Transformer Engine itself.
# ----------------------------------------------------------------------------------------

_TE = []


def _transformer_engine():
    """Returns (device, calls). calls is None with the real library, else the stand-in's autocast log."""
    if not _TE:
        _TE.append(_find_transformer_engine())
    return _TE[0]


def _find_transformer_engine():
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        return "cuda", None
    except ImportError:
        pass
    import contextlib
    import types

    calls = []

    class GroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, in_features, out_features, bias=True, init_method=None, **kwargs):
            super().__init__()
            assert not bias
            self.num_gemms = num_gemms
            for i in range(num_gemms):
                w = torch.nn.Parameter(torch.empty(out_features, in_features))
                init_method(w)
                self.register_parameter(f"weight{i}", w)

        def forward(self, inp, m_splits):
            assert inp.dim() == 2 and tuple(m_splits.shape) == (self.num_gemms,)
            assert int(m_splits.sum()) == inp.shape[0]
            parts = torch.split(inp, m_splits.tolist())
            return torch.cat([F.linear(part, getattr(self, f"weight{i}")) for i, part in enumerate(parts)])

    @contextlib.contextmanager
    def autocast(enabled=True, recipe=None):
        calls.append((enabled, recipe))
        yield

    class _Options:  # records the keyword arguments it was built with
        def __init__(self, *args, **options):
            self.args, self.options = args, options

    names = ["NVFP4BlockScaling", "MXFP8BlockScaling", "CustomRecipe", "HybridQuantizer", "NVFP4Quantizer",
             "MXFP8Quantizer"]
    kinds = {name: type(name, (_Options,), {}) for name in names}

    def module(name, **members):
        mod = types.ModuleType(name)
        for key, value in members.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    module("transformer_engine")
    module("transformer_engine.common")
    module("transformer_engine.common.recipe", **{k: kinds[k] for k in names[:3]})
    module("transformer_engine.pytorch", GroupedLinear=GroupedLinear, autocast=autocast,
           HybridQuantizer=kinds["HybridQuantizer"])
    module("transformer_engine.pytorch.tensor")
    module("transformer_engine.pytorch.tensor.nvfp4_tensor", NVFP4Quantizer=kinds["NVFP4Quantizer"])
    module("transformer_engine.pytorch.custom_recipes")
    module("transformer_engine.pytorch.custom_recipes.quantizer_factories",
           mxfp8_factory=lambda role: kinds["MXFP8Quantizer"]())
    module("transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo",
           nvfp4_row_scaled_fwd_mxfp8_bwd_factory=lambda role: None)
    return "cpu", calls


def test_native_backend_matches_simulated_in_full_precision():
    device, calls = _transformer_engine()
    sim = randomise(Inkling(InklingConfig(**TINY, quant="none"))).to(device)
    native = Inkling(InklingConfig(**TINY, quant="none", backend="te")).to(device)
    native.load_canonical(sim.state_dict())
    assert sim.num_params() == native.num_params()
    idx = torch.randint(0, 64, (2, 25), device=device)
    (ls, loss_s), (ln, loss_n) = sim(idx[:, :-1], idx[:, 1:]), native(idx[:, :-1], idx[:, 1:])
    tol = 1e-4 if device == "cpu" else 5e-2                                     # the real library computes in BF16
    assert (ls - ln).abs().max() < tol, (ls - ln).abs().max()
    loss_s.backward()
    loss_n.backward()
    moe_s, moe_n = sim.layers[3].mlp.experts, native.layers[3].mlp.experts
    grad_n = torch.stack([getattr(moe_n.gate_up, f"weight{e}").grad for e in range(8)]).float()
    assert (moe_s.gate_up_proj.grad - grad_n).abs().max() < tol * moe_s.gate_up_proj.grad.abs().max().clamp(min=1)
    if calls is not None:
        assert calls and all(enabled is False and recipe is None for enabled, recipe in calls)


def test_checkpoints_do_not_depend_on_the_backend():
    device, _ = _transformer_engine()
    native = randomise(Inkling(InklingConfig(**TINY, backend="te"))).to(device)
    state = native.canonical_state_dict()
    sim = Inkling(InklingConfig(**TINY))
    assert set(state) == set(sim.state_dict())                                  # same keys as the simulated layout
    sim.load_state_dict({k: v.cpu() for k, v in state.items()})
    again = Inkling(InklingConfig(**TINY, backend="te")).to(device)
    again.load_canonical(sim.canonical_state_dict())
    for (name, a), (_, b) in zip(native.named_parameters(), again.named_parameters()):
        assert torch.equal(a.float().cpu(), b.float().cpu()), name


def test_native_recipes_by_layer():
    device, calls = _transformer_engine()
    if calls is None:
        return  # real library: recipe objects are not inspectable the same way
    model = Inkling(InklingConfig(**TINY, backend="te"))
    kinds = [type(layer.mlp.recipe).__name__ for layer in model.layers]
    assert kinds == ["CustomRecipe"] * 6 + ["MXFP8BlockScaling"] * 2            # last hp_tail_layers in MXFP8

    # The published recipe, slot by slot: 4-bit forward operands, 8-bit backward operands made
    # from the dequantized forward values, 8-bit gradients.
    import types
    factory = model.layers[0].mlp.recipe.options["qfactory"]
    role = lambda kind: types.SimpleNamespace(module_type="grouped_linear", tensor_type=kind, name="")
    plain = dict(with_rht=False, with_post_rht_amax=False, with_2d_quantization=False, stochastic_rounding=False)
    for kind, extra in (("input", dict(row_scaled_nvfp4=True, nvfp4_use_4over6=True, nvfp4_e4m3_max=256,
                                       nvfp4_4over6_err_mode="MSE")), ("weight", {})):
        hybrid = factory(role(kind)).options
        assert type(hybrid["rowwise_quantizer"]).__name__ == "NVFP4Quantizer"
        assert hybrid["rowwise_quantizer"].options == {**plain, **extra}
        assert type(hybrid["columnwise_quantizer"]).__name__ == "MXFP8Quantizer"
        assert hybrid["columnwise_source"] == "rowwise_dequantized"
    assert type(factory(role("grad_output"))).__name__ == "MXFP8Quantizer"
    assert type(factory(None)).__name__ == "MXFP8Quantizer"

    shipped = Inkling(InklingConfig(**TINY, backend="te", te_recipe="rowscaled"))
    assert type(shipped.layers[0].mlp.recipe).__name__ == "CustomRecipe"
    stock = Inkling(InklingConfig(**TINY, backend="te", te_recipe="nvidia"))
    assert type(stock.layers[0].mlp.recipe).__name__ == "NVFP4BlockScaling" and stock.layers[0].mlp.recipe.options == {}
    del calls[:]
    model(torch.randint(0, 64, (1, 8)))
    assert all(enabled for enabled, _ in calls) and len(calls) == 8 + 2 * 4      # 8 layers, 4 looped twice more


def test_bfloat16_autocast_and_export_roundtrip():
    import tempfile
    import export_nvfp4
    from loopfp4 import get_state, load_checkpoint

    model = randomise(Inkling(InklingConfig(**TINY, grad_checkpoint=True))).train()
    seen = []
    orig = model.layers[0].mlp.forward
    model.layers[0].mlp.forward = lambda inp: (seen.append(inp.dtype), orig(inp))[1]
    idx = torch.randint(0, 64, (2, 17))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss = model(idx[:, :-1], idx[:, 1:])
    loss.backward()
    assert seen == [torch.bfloat16, torch.bfloat16]  # forward + checkpoint recomputation
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    model.layers[0].mlp.forward = orig
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        ck_path, exp_path = os.path.join(tmp, "ckpt.pt"), os.path.join(tmp, "nvfp4.pt")
        torch.save({"model": get_state(model), "config": model.config_dict(), "arch": "inkling"}, ck_path)
        export_nvfp4.main(ck_path, exp_path)
        unpacked, _ = load_checkpoint(exp_path)
        unpacked.eval()
        assert torch.equal(quantize_expert_weight(model.layers[0].mlp.experts.gate_up_proj, "nvfp4"),
                           unpacked.layers[0].mlp.experts.gate_up_proj)
        sd_bf16 = {k: (v.bfloat16().float() if v.is_floating_point() and not (".mlp.experts." in k and any(f"layers.{i}." in k for i in range(6))) else v)
                   for k, v in model.state_dict().items()}
        with torch.no_grad():
            raw_logits = model(idx[:, :-1])[0]
            unpacked_logits = unpacked(idx[:, :-1])[0]
            assert torch.allclose(raw_logits, unpacked_logits, atol=1e-1)  # BF16 rounding noise on non-FP4 weights
            model.load_state_dict(sd_bf16)
            assert torch.allclose(model(idx[:, :-1])[0], unpacked_logits, atol=1e-5)  # zero packing error
        broken = torch.load(exp_path, map_location="cpu")
        del broken["tensors"]["lm_head.weight"]
        torch.save(broken, exp_path)
        try:
            load_checkpoint(exp_path)
            assert False, "expected missing weight in exported file to fail"
        except RuntimeError:
            pass


def test_fast_convs_and_attn_res_exact():
    import math
    torch.manual_seed(7)
    sc = ShortConvolution(32, 4)
    with torch.no_grad():
        sc.conv1d.weight.normal_(std=0.2)
    x1 = torch.randn(2, 16, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    g = torch.randn_like(x1)
    y_got = sc(x1)
    (y_got * g).sum().backward()
    dw_got = sc.conv1d.weight.grad.clone()
    sc.zero_grad()

    y_ref = F.conv1d(x2.transpose(1, 2), sc.conv1d.weight.float(), padding=3, groups=32)[..., :16].transpose(1, 2)
    y_ref = (y_ref + x2) / math.sqrt(5)
    (y_ref * g).sum().backward()
    assert torch.allclose(y_got, y_ref, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(dw_got, sc.conv1d.weight.grad, atol=1e-5)

    dsc = DynamicShortConvolution(32, 48, heads=4, kernel=4)
    with torch.no_grad():
        dsc.static_weight.normal_(std=0.2)
        dsc.weight_proj.weight.normal_(std=0.2)
    dx1 = torch.randn(2, 16, 32, requires_grad=True)
    dyn1 = torch.randn(2, 16, 48, requires_grad=True)
    dx2, dyn2 = dx1.detach().clone().requires_grad_(True), dyn1.detach().clone().requires_grad_(True)
    dy_got = dsc(dx1, dyn1)
    (dy_got * g).sum().backward()
    g_dx1, g_dyn1 = dx1.grad.clone(), dyn1.grad.clone()
    g_sw1, g_wp1 = dsc.static_weight.grad.clone(), dsc.weight_proj.weight.grad.clone()
    dsc.zero_grad()

    dyn = dsc.weight_proj(dyn2).float().view(2, 16, 4, 1, 4)
    weights = dsc.static_weight.float().T.reshape(4, -1, 4) + dyn
    ctx = F.pad(dx2.transpose(1, 2), (3, 0))
    windows = ctx.unfold(-1, 4, 1).flip(-1).transpose(1, 2).reshape(2, 16, 4, -1, 4)
    dy_ref = ((windows * weights).sum(-1).reshape(2, 16, 32) + dx2) / math.sqrt(9)
    (dy_ref * g).sum().backward()
    assert torch.allclose(dy_got, dy_ref, atol=1e-5)
    assert torch.allclose(g_dx1, dx2.grad, atol=1e-5)
    assert torch.allclose(g_dyn1, dyn2.grad, atol=1e-5)
    assert torch.allclose(g_sw1, dsc.static_weight.grad, atol=1e-5)
    assert torch.allclose(g_wp1, dsc.weight_proj.weight.grad, atol=1e-5)

    for nq in (1, 3):
        bar = BlockAttentionResidual(InklingConfig(**TINY), num_queries=nq)
        with torch.no_grad():
            bar.query.normal_(std=0.3)
        bs1 = torch.randn(2, 12, 3, 64, requires_grad=True)
        pt1 = torch.randn(2, 12, 64, requires_grad=True)
        bs2, pt2 = bs1.detach().clone().requires_grad_(True), pt1.detach().clone().requires_grad_(True)
        gm = torch.randn(2, 12, nq, 64)
        out_got = bar(bs1, pt1)
        (out_got * gm).sum().backward()
        g_q1, g_bs1, g_pt1 = bar.query.grad.clone(), bs1.grad.clone(), pt1.grad.clone()
        bar.zero_grad()

        states = rms_norm(torch.cat([bs2, pt2.unsqueeze(-2)], dim=-2).float(), bar.eps)
        attn = F.softmax(torch.einsum("qh,btsh->btqs", bar.query.float().view(-1, 64), states) * 64**-0.5, dim=-1)
        out_ref = attn @ states
        (out_ref * gm).sum().backward()
        assert torch.allclose(out_got, out_ref, atol=1e-5), nq
        assert torch.allclose(g_q1, bar.query.grad, atol=1e-5), nq
        assert torch.allclose(g_bs1, bs2.grad, atol=1e-5), nq
        assert torch.allclose(g_pt1, pt2.grad, atol=1e-5), nq


def test_grad_checkpoint_mlp_mode():
    torch.manual_seed(11)
    ref = randomise(Inkling(InklingConfig(**TINY, backward="exact", grad_checkpoint=False))).train()
    idx = torch.randint(0, 64, (2, 25))
    _, loss_ref = ref(idx[:, :-1], idx[:, 1:], return_logits=False)
    loss_ref.backward()
    for mode in ("mlp", "half", "sac", "sac_all"):
        ckpt_m = Inkling(
            InklingConfig(**TINY, backward="exact", grad_checkpoint=True, grad_checkpoint_mode=mode)
        ).train()
        ckpt_m.load_canonical(ref.canonical_state_dict())
        _, loss_got = ckpt_m(idx[:, :-1], idx[:, 1:], return_logits=False)
        assert torch.allclose(loss_got, loss_ref, atol=1e-5), mode
        loss_got.backward()
        for (n, p_ref), (_, p_got) in zip(ref.named_parameters(), ckpt_m.named_parameters()):
            assert p_got.grad is not None and torch.isfinite(p_got.grad).all(), (mode, n)
            assert torch.allclose(p_got.grad, p_ref.grad, atol=1e-5), (mode, n)


def test_grouped_canonical_roundtrip():
    sim = randomise(Inkling(InklingConfig(**TINY, backward="exact", backend="sim"))).eval()
    grp = Inkling(InklingConfig(**TINY, backward="exact", backend="grouped")).eval()
    assert set(grp.canonical_state_dict()) == set(sim.state_dict())
    grp.load_canonical(sim.canonical_state_dict())
    idx = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        assert torch.allclose(grp(idx)[0], sim(idx)[0], atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
