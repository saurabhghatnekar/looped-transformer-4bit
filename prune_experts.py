"""Prune a trained checkpoint to its most-used experts.

    python prune_experts.py runs/5b-pretrain/ckpt.pt runs/5b-pretrain/pruned.pt --keep 128

Inkling 2 Mini was pretrained with 384 experts per layer and, at one point in training, cut
to the 128 experts each layer used most. Training then continued ("healing") so the remaining
experts took over the deleted ones' work:

    python train.py --init runs/5b-pretrain/pruned.pt --data ... --out runs/5b-healed

"Most used" here is the running share of routed tokens that train.py keeps per expert
(the `usage` buffer of each layer).
"""

import argparse

import torch

PER_EXPERT = ["experts.gate_up_proj", "experts.down_proj", "gate.weight", "gate.e_score_correction_bias", "usage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep", type=int, default=128)
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu")
    assert ck.get("arch") == "inkling", "pruning applies to the Inkling presets"
    cfg, sd = dict(ck["config"]), dict(ck["model"])
    before = cfg["n_routed_experts"]
    assert cfg["num_experts_per_tok"] <= args.keep < before, (args.keep, before)
    kept_share = []
    for i in range(cfg["num_hidden_layers"]):
        pre = f"layers.{i}.mlp."
        usage = sd[pre + "usage"]
        assert usage.sum() > 0, "no usage statistics: train before pruning"
        keep = usage.topk(args.keep).indices.sort().values
        kept_share.append((usage[keep].sum() / usage.sum()).item())
        for key in PER_EXPERT:
            sd[pre + key] = sd[pre + key][keep].clone()
        sd[pre + "usage"] /= sd[pre + "usage"].sum()
    cfg["n_routed_experts"] = args.keep
    torch.save({"model": sd, "config": cfg, "arch": "inkling", "step": ck.get("step", 0),
                "pruned_from": before}, args.dst)
    n = sum(v.numel() for k, v in sd.items() if not k.endswith(("usage", "e_score_correction_bias")))
    print(f"experts per layer {before} -> {args.keep}; parameters now {n / 1e6:.2f}M")
    print(f"share of routed tokens that went to the kept experts: "
          f"min {min(kept_share):.1%}, mean {sum(kept_share) / len(kept_share):.1%} across layers")


if __name__ == "__main__":
    main()
