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
from loopfp4.inkling import softcap  # noqa: E402
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
