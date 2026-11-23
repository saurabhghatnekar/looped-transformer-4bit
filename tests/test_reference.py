"""Check loopfp4.inkling against the published reference implementation.

Builds a small random Inkling2Loop model with Hugging Face transformers, copies its weights
into loopfp4.inkling.Inkling (full precision) and compares the logits. Needs a transformers
release that includes `inkling2_loop` (5.24 or later):

    python tests/test_reference.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopfp4.inkling import Inkling, InklingConfig  # noqa: E402

from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig  # noqa: E402
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM  # noqa: E402

SMALL = dict(
    vocab_size=100, unpadded_vocab_size=90, hidden_size=64, num_hidden_layers=12,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    swa_num_attention_heads=8, swa_num_key_value_heads=2, swa_head_dim=8,
    sliding_window_size=8, d_rel=4, rel_extent=16, log_scaling_n_floor=4,
    n_layers_per_attn_res_block=3, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    loop_layer_index_start=3, loop_layer_index_end=9, loop_gate_rank=8,
)


def build(passes: int):
    torch.manual_seed(0)
    hf_cfg = Inkling2LoopTextConfig(
        **SMALL, loop_num_passes=passes,
        layer_types=["hybrid" if (i + 1) % 3 == 0 else "hybrid_sliding" for i in range(12)])
    ref = Inkling2LoopForCausalLM(hf_cfg).float().eval()
    with torch.no_grad():  # zero-initialised pieces would hide wiring mistakes: randomise everything
        for name, p in ref.named_parameters():
            if p.abs().max() == 0 or "scale" in name or "norm" in name:
                p.add_(torch.randn_like(p) * 0.5)
        for name, b in ref.named_buffers():
            if "e_score_correction_bias" in name:
                b.add_(torch.randn_like(b) * 0.1)

    mine = Inkling(InklingConfig(**SMALL, loop_num_passes=passes, global_every=3, quant="none")).eval()
    state = {k.removeprefix("model."): v for k, v in ref.state_dict().items()}
    missing, unexpected = mine.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(("usage", "step_counts"))]
    assert not missing and not unexpected, (missing, unexpected)
    return ref, mine


def test_logits_match_reference():
    idx = torch.randint(0, 90, (2, 24))
    for passes in (1, 2, 3):
        ref, mine = build(passes)
        with torch.no_grad():
            want = ref(input_ids=idx).logits
            got, _ = mine(idx)
        err = (want - got).abs().max().item()
        print(f"passes={passes}: max |logit difference| = {err:.2e} (logit scale {want.abs().max().item():.2f})")
        assert err < 1e-5, err
    # the same weights run with a different number of passes than the config default
    ref5, _ = build(5)
    _, mine3 = build(3)
    with torch.no_grad():
        err = (ref5(input_ids=idx).logits - mine3(idx, n_loops=5)[0]).abs().max().item()
    print(f"passes=5 via n_loops override: max |logit difference| = {err:.2e}")
    assert err < 1e-5, err


if __name__ == "__main__":
    test_logits_match_reference()
    print("reference check passed")
