"""Train a looped transformer with 4-bit forward passes.

Laptop check (CPU / Apple MPS), Inkling 2 Mini's architecture at ~10M parameters:
    python train.py --preset smoke --data data/tinystories-bytes --out runs/ink-smoke

One Blackwell node, 8 GPUs, the same architecture at ~5B parameters:
    torchrun --nproc_per_node=8 train.py --preset 5b --data /data/fineweb-edu-100b-inkling --out runs/5b \
        --batch-size 4 --grad-accum 4 --steps 380000 --lr 2e-4 --warmup 3000

Continue from a pruned checkpoint (see prune_experts.py):
    python train.py --init runs/5b/pruned.pt --data ... --out runs/5b-healed
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from loopfp4 import Config, InklingConfig, PRESETS, build, get_state, set_state
from loopfp4.data import ShardData


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights and config (fresh optimizer)")
    # model overrides
    ap.add_argument("--quant", choices=["none", "nvfp4"])
    ap.add_argument("--loops", type=int, help="passes through the looped section")
    ap.add_argument("--var-loops", default="", help="train with sampled loop counts, e.g. '1:1,2:2,3:6,4:2,5:1'")
    ap.add_argument("--vocab-size", type=int, help="default: the data's meta.json, else the preset")
    # Inkling presets only
    ap.add_argument("--experts", type=int, help="experts per layer (384 before pruning, 128 after)")
    ap.add_argument("--backward", choices=["mxfp8", "exact"], help="precision of the expert backward GEMMs")
    ap.add_argument("--balance-rate", type=float, default=1e-3,
                    help="step size of the router's load-balancing bias; 0 turns balancing off")
    ap.add_argument("--backend", choices=["sim", "te"],
                    help="sim: 4-bit numerics simulated, runs anywhere. te: native NVFP4 GEMMs through "
                         "NVIDIA Transformer Engine, Blackwell only. Default sim.")
    ap.add_argument("--te-recipe", choices=["tml", "rowscaled", "nvidia"],
                    help="with --backend te on Inkling presets. tml (default): NVFP4 forward, MXFP8 backward, "
                         "Four-Over-Six, as published. rowscaled: the same without Four-Over-Six, using only "
                         "a factory Transformer Engine ships. nvidia: its default NVFP4 recipe.")
    ap.add_argument("--te-align", type=int, help="m_splits row alignment for TE GroupedLinear (1 = unpadded)")
    # dense-* presets only
    ap.add_argument("--quant-scope", choices=["all", "mlp"])
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    # optimisation
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per GPU per micro-step")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    # logging
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    return ap.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    args = parse()
    ddp = int(os.environ.get("RANK", -1)) != -1
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    master = rank == 0

    if torch.cuda.is_available():
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
    torch.manual_seed(args.seed + rank)

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    resuming = args.resume and os.path.exists(ckpt_path)
    ck = torch.load(ckpt_path if resuming else args.init, map_location="cpu") if (resuming or args.init) else None
    if ck is not None:
        cfg = InklingConfig(**ck["config"]) if ck.get("arch") == "inkling" else Config(**ck["config"])
    else:
        cfg = dataclasses.replace(PRESETS[args.preset])
    ink = isinstance(cfg, InklingConfig)
    train_probe = ShardData(args.data, "train", 1, 1, "cpu")  # opened early for its vocabulary size
    vocab = args.vocab_size or (None if ck is not None else train_probe.meta.get("vocab_size"))
    del train_probe
    overrides = {"quant": args.quant, "block_size": args.block_size, "grad_checkpoint": args.grad_checkpoint,
                 "backend": args.backend or "sim"}  # the backend is chosen per run
    if ink:
        padded_vocab = args.vocab_size or (cfg.vocab_size if vocab == cfg.unpadded_vocab_size else vocab)
        overrides.update(vocab_size=padded_vocab, unpadded_vocab_size=vocab, loop_num_passes=args.loops,
                         n_routed_experts=args.experts, backward=args.backward, te_recipe=args.te_recipe,
                         te_align=args.te_align)
    else:
        overrides.update(vocab_size=vocab, n_loops=args.loops, quant_scope=args.quant_scope)
    if overrides["backend"] == "te" and not torch.cuda.is_available():
        raise SystemExit("--backend te needs a Blackwell GPU and NVIDIA Transformer Engine; use --backend sim here")
    for field, val in overrides.items():
        if val is not None:
            setattr(cfg, field, val)
    passes = cfg.loop_num_passes if ink else cfg.n_loops
    loop_choices = None
    if args.var_loops:
        pairs = [p.split(":") for p in args.var_loops.split(",")]
        loop_choices = ([int(a) for a, _ in pairs], [float(b) for _, b in pairs])

    model = build(cfg)
    if ck is not None:
        set_state(model, ck["model"])
        del ck["model"]  # free CPU state_dict copy before moving model + optimizer to device
    model = model.to(device)
    raw = model
    if master:
        os.makedirs(args.out, exist_ok=True)
        detail = f"backend={cfg.backend}"
        if ink:
            detail += f" experts={cfg.num_experts_per_tok}/{cfg.n_routed_experts} " + (
                f"te_recipe={cfg.te_recipe}" if cfg.backend == "te" else f"backward={cfg.backward}")
        print(f"device={device} world={world} params={raw.num_params() / 1e6:.2f}M "
              f"applications={cfg.applications()} quant={cfg.quant} {detail}", flush=True)

    # Weight decay on matrices only: never on norm gains, learned scales, 1D depthwise convs or residual queries.
    skip = ("scale", "query", "norm", "static_weight", "conv1d")
    named = list(raw.named_parameters())
    decay = [p for n, p in named if p.dim() >= 2 and not any(k in n for k in skip)]
    no_decay = [p for n, p in named if not (p.dim() >= 2 and not any(k in n for k in skip))]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device.startswith("cuda"),
    )

    step = 0
    loop_rng = torch.Generator().manual_seed(args.seed)  # same stream on every rank
    train = ShardData(args.data, "train", cfg.block_size, args.batch_size, device, seed=args.seed * 1000 + rank)
    val = ShardData(args.data, "val", cfg.block_size, args.batch_size, device, seed=12345)
    if resuming:
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        rngs = ck.get("data_rngs")
        if rngs and rank < len(rngs):
            train.rng.bit_generator.state = rngs[rank]
        else:
            train.rng.bit_generator.advance(step * args.grad_accum * (args.batch_size + 2))
        if "loop_rng" in ck:
            loop_rng.set_state(ck["loop_rng"].cpu())
        if master:
            print(f"resumed from step {step}", flush=True)
    is_fresh_init = bool(args.init and not resuming)
    del ck

    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], broadcast_buffers=False)

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else contextlib.nullcontext()
    if not ink and cfg.backend == "te" and cfg.quant == "nvfp4":
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling
        fp4 = lambda: te.autocast(enabled=True, recipe=NVFP4BlockScaling())
    else:
        fp4 = contextlib.nullcontext

    @torch.no_grad()
    def evaluate(n_loops=None):
        raw.eval()
        tot = 0.0
        for _ in range(args.eval_batches):
            x, y = val.batch()
            with amp, fp4():
                _, loss = raw(x, y, n_loops=n_loops, return_logits=False)
            tot += loss.item()
        raw.train()
        return tot / args.eval_batches

    def save(data_rngs):
        torch.save({"model": get_state(raw), "opt": opt.state_dict(), "step": step,
                    "arch": "inkling" if ink else "dense", "config": raw.config_dict(), "args": vars(args),
                    "data_rngs": data_rngs, "loop_rng": loop_rng.get_state()},
                   ckpt_path)

    log = open(os.path.join(args.out, "log.jsonl"), "a") if master else None
    if master and is_fresh_init:  # where the loaded weights start from, e.g. the loss right after pruning
        rec = {"step": 0, "val_loss": round(evaluate(), 4), "init": args.init}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
    if ddp and is_fresh_init:
        torch.distributed.barrier()
    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum * world
    t0, last = time.time(), step
    model.train()
    while step < args.steps:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        n_loops = None
        if loop_choices:
            n_loops = loop_choices[0][torch.multinomial(torch.tensor(loop_choices[1]), 1, generator=loop_rng).item()]

        for micro in range(args.grad_accum):
            x, y = train.batch()
            sync_ctx = model.no_sync() if (ddp and micro < args.grad_accum - 1) else contextlib.nullcontext()
            with sync_ctx, amp, fp4():
                _, loss = model(x, y, n_loops=n_loops, return_logits=False)
            (loss / args.grad_accum).backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ink:
            raw.balance_step(args.balance_rate, sync=torch.distributed.all_reduce if ddp else None)
        step += 1

        if master and step % args.log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr,
                   "tok_per_s": round(tokens_per_step * (step - last) / dt)}
            if ink:  # gate: how far the loop message gates have opened. load: busiest expert's
                # share of tokens relative to an even split (1.0 = perfectly balanced), worst layer.
                raw_u = torch.stack([m.usage for m in raw.moe_blocks()])
                share = raw_u / raw_u.sum(dim=1, keepdim=True).clamp(min=1e-12)
                rec["gate"] = round(raw.loop_gate_strength(), 5)
                rec["load_max"] = round((share.max(dim=1).values * share.shape[1]).max().item(), 2)
                rec["idle_experts"] = int((share * share.shape[1] < 0.1).sum().item())
            else:
                rec["g_msg"] = round(raw.g_msg.abs().mean().item(), 4)
                rec["g_pre"] = round(raw.g_pre.abs().mean().item(), 4)
            t0, last = time.time(), step
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            if master:
                rec = {"step": step, "val_loss": round(evaluate(), 4)}
                if passes > 1:
                    rec["val_loss_1_loop"] = round(evaluate(n_loops=1), 4)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
            if ddp:
                torch.distributed.barrier()
            t0, last = time.time(), step
        if step % args.save_every == 0 or step == args.steps:
            data_rngs = [None] * world if ddp else [train.rng.bit_generator.state]
            if ddp:
                torch.distributed.all_gather_object(data_rngs, train.rng.bit_generator.state)
            if master:
                save(data_rngs)
            if ddp:
                torch.distributed.barrier()

    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
