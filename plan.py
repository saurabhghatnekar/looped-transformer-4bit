"""Size a run: parameters, compute per token, memory, GPU-hours and cost.

    python plan.py --preset 5b
    python plan.py --preset mini-pretrain --gpus 512
    python plan.py --preset 5b --tokens 300e9 --gpus 64 --tok-per-s-per-gpu 21000

Throughput is the weak point of any estimate like this. The default derives it from an
assumed peak BF16 rate and utilisation; once train.py has run on real hardware, pass the
tok_per_s it logs (divided by GPU count) via --tok-per-s-per-gpu and the guess goes away.
"""

import argparse
import dataclasses

from loopfp4 import InklingConfig, PRESETS, param_counts


def size_dense(cfg):
    d, h = cfg.d_model, cfg.mlp_hidden
    per_layer = 4 * d * d + 3 * d * h          # attention (qkv + proj) + SwiGLU
    emb = cfg.vocab_size * d                   # tied with the output head
    n_hp = round(cfg.n_layer * cfg.hp_tail_frac)
    stored = per_layer * cfg.n_layer + emb
    return {
        "desc": f"dense, d_model={cfg.d_model} layers={cfg.n_prelude}+{cfg.n_loop}+{cfg.n_coda} loops={cfg.n_loops}",
        "stored": stored, "active": stored, "applied": per_layer * cfg.applications() + emb,
        "applications": cfg.applications(), "layers": cfg.n_layer,
        "ckpt": (per_layer * (cfg.n_layer - n_hp) * 4.5 + per_layer * n_hp * 8.25 + emb * 16) / 8,
        "formats": f"{cfg.n_layer - n_hp} layers NVFP4, {n_hp} MXFP8, embeddings BF16",
    }


def size_inkling(cfg):
    c = param_counts(cfg)
    L, tail = cfg.num_hidden_layers, cfg.hp_tail_layers
    per_layer_experts = c["experts"] / L
    other = c["stored"] - c["experts"]
    return {
        "desc": (f"mixture of experts {cfg.num_experts_per_tok} of {cfg.n_routed_experts}, hidden={cfg.hidden_size} "
                 f"layers={L} (looped {cfg.loop_layer_index_start}-{cfg.loop_layer_index_end - 1}) "
                 f"loops={cfg.loop_num_passes}"),
        "stored": c["stored"], "active": c["active"], "applied": c["applied"],
        "applications": cfg.applications(), "layers": L,
        "ckpt": (per_layer_experts * (L - tail) * 4.5 + per_layer_experts * tail * 8.25 + other * 16) / 8,
        "formats": f"experts of {L - tail} layers NVFP4, of {tail} MXFP8, everything else BF16",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="5b")
    ap.add_argument("--loops", type=int)
    ap.add_argument("--experts", type=int)
    ap.add_argument("--tokens", type=float, nargs="*", default=[100e9, 300e9, 1e12])
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-pflops", type=float, default=2.25, help="ASSUMED peak dense BF16 PFLOPS per B200")
    ap.add_argument("--util", type=float, default=0.35, help="ASSUMED fraction of peak achieved end to end")
    ap.add_argument("--gpu-gb", type=float, default=180, help="ASSUMED usable memory per GPU")
    ap.add_argument("--tok-per-s-per-gpu", type=float, default=0, help="measured throughput; overrides the two above")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=4.0, help="ASSUMED rental price")
    args = ap.parse_args()

    cfg = dataclasses.replace(PRESETS[args.preset])
    ink = isinstance(cfg, InklingConfig)
    if args.loops:
        setattr(cfg, "loop_num_passes" if ink else "n_loops", args.loops)
    if args.experts and ink:
        cfg.n_routed_experts = args.experts
    s = size_inkling(cfg) if ink else size_dense(cfg)
    flops_per_token = 6 * s["applied"]                  # 2 forward + 4 backward per weight applied
    state_gb = s["stored"] * 16 / 1e9

    print(f"preset {args.preset}: {s['desc']}")
    print(f"  stored parameters        {s['stored'] / 1e9:8.3f} B")
    print(f"  active per token         {s['active'] / 1e9:8.3f} B   (one pass; what the model card calls 'active')")
    print(f"  layer applications/token {s['applications']:8d}   (non-looped would be {s['layers']})")
    print(f"  weights applied/token    {s['applied'] / 1e9:8.3f} B   (looped layers counted once per pass)")
    print(f"  training FLOPs/token     {flops_per_token:8.2e}")
    print(f"  deployed checkpoint      {s['ckpt'] / 1e9:8.2f} GB  ({s['formats']})")
    print(f"  training weight state    {state_gb:8.1f} GB  (FP32 weights + grads + two Adam moments), before activations")
    if state_gb > 0.8 * args.gpu_gb:
        print(f"  !! does not fit one {args.gpu_gb:.0f} GB GPU: train.py keeps a full copy of the model on every GPU. "
              f"This size needs the experts sharded across GPUs, which is not implemented here.")

    if args.tok_per_s_per_gpu:
        tps, src = args.tok_per_s_per_gpu, "measured"
    else:
        tps = args.gpu_pflops * 1e15 * args.util / flops_per_token
        src = f"assumed {args.gpu_pflops} PFLOPS x {args.util:.0%}"
    print(f"\n  throughput {tps:,.0f} tokens/s per GPU ({src}); {args.gpus} GPUs; ${args.usd_per_gpu_hour}/GPU-hour (assumed)")
    print(f"  {'tokens':>10} {'tokens/param':>13} {'GPU-hours':>11} {'days':>7} {'cost':>10}")
    for t in args.tokens:
        gpu_h = t / tps / 3600
        print(f"  {t / 1e9:8.0f} B {t / s['stored']:13.0f} {gpu_h:11,.0f} {gpu_h / args.gpus / 24:7.1f} {gpu_h * args.usd_per_gpu_hour:10,.0f}")


if __name__ == "__main__":
    main()
