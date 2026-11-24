"""Historical Part 1 verification harness (--backend te) on a single NVIDIA B200 via Modal.
For the current Part 2 harness (--backend grouped, Triton RPB+Sink attention, SAC), see modal_verify_part2.py.

    modal run modal_verify_b200.py
    modal run modal_verify_b200.py --full   # also re-run the 6-min TinyStories smoke e2e CLI pipeline

Stages run on 1x B200:
  1. Hardware & library check (B200 sm_100, CUDA, PyTorch, Transformer Engine, transformers)
  2. Unit & reference tests (test_core.py, test_inkling.py against real TE, test_reference.py)
  3. Determinism & m_splits alignment diagnostic (smoke eval repeat check + 5b-width SparseMoeBlock 32 vs 128)
  4. (Optional --full) End-to-end smoke pretrain -> prune -> heal -> export -> sample
  5. Real-scale 5b preset (5.4B params, block_size=2048, 2-step Adam-materialized batch probe) throughput & VRAM check
  6. Full mini-pretrain expert shape check (384 experts, H=2304, inner=1152) at te_align=32 vs 128
"""

import dataclasses
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
import modal

APP_NAME = "loopfp4-b200-verify"
ROOT = os.path.dirname(os.path.abspath(__file__))

# Pull NVIDIA's prebuilt Blackwell NGC PyTorch + Transformer Engine 2.20 container on Modal's
# CPU builder (layer-cached across runs) so zero B200 GPU seconds are spent compiling CUDA kernels.
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.10-py3")
    .pip_install("transformers>=5.0.0", "tokenizers", "tiktoken")
    .add_local_dir(os.path.join(ROOT, "loopfp4"), remote_path="/workspace/loopfp4")
    .add_local_dir(os.path.join(ROOT, "tests"), remote_path="/workspace/tests")
    .add_local_dir(os.path.join(ROOT, "data", "tinystories-bytes"), remote_path="/workspace/data/tinystories-bytes")
    .add_local_file(os.path.join(ROOT, "train.py"), remote_path="/workspace/train.py")
    .add_local_file(os.path.join(ROOT, "prune_experts.py"), remote_path="/workspace/prune_experts.py")
    .add_local_file(os.path.join(ROOT, "export_nvfp4.py"), remote_path="/workspace/export_nvfp4.py")
    .add_local_file(os.path.join(ROOT, "prepare_data.py"), remote_path="/workspace/prepare_data.py")
    .add_local_file(os.path.join(ROOT, "sample.py"), remote_path="/workspace/sample.py")
    .add_local_file(os.path.join(ROOT, "plan.py"), remote_path="/workspace/plan.py")
)

app = modal.App(APP_NAME)


def banner(title: str):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}", flush=True)


def run_cmd(cmd: list[str], cwd: str = "/workspace", check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    dt = time.time() - t0
    if res.stdout:
        print(res.stdout.rstrip(), flush=True)
    if res.stderr:
        print(res.stderr.rstrip(), flush=True)
    print(f"  [exit={res.returncode} in {dt:.1f}s]", flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")
    return res


@app.function(image=image, gpu="B200", timeout=1800)
def verify_b200(run_smoke_e2e: bool = False) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    t_start = time.time()
    warnings = []
    report = {"warnings": warnings, "stages": {}}

    # ------------------------------------------------------------------
    # Stage 1: Hardware & Software Environment
    # ------------------------------------------------------------------
    banner("STAGE 1: Hardware & Software Environment")
    import numpy as np
    import torch
    import transformer_engine
    import transformers

    props = torch.cuda.get_device_properties(0)
    has_ref = importlib.util.find_spec("transformers.models.inkling2_loop") is not None
    env_info = {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1e9, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_engine": transformer_engine.__version__,
        "transformers": transformers.__version__,
        "has_inkling2_loop_ref": has_ref,
    }
    for k, v in env_info.items():
        print(f"  {k:22s}: {v}", flush=True)
    assert props.major >= 10, f"Expected Blackwell (SM100+), got SM{props.major}{props.minor}"
    report["env"] = env_info

    # ------------------------------------------------------------------
    # Stage 2: Unit & Reference Tests (Real Transformer Engine on B200)
    # ------------------------------------------------------------------
    banner("STAGE 2: Unit & Reference Test Suite")
    run_cmd([sys.executable, "tests/test_core.py"])
    run_cmd([sys.executable, "tests/test_inkling.py"])
    if has_ref:
        run_cmd([sys.executable, "tests/test_reference.py"], check=True)
        ref_status = "pass"
    else:
        warnings.append(f"transformers {transformers.__version__} lacks models.inkling2_loop; test_reference not run")
        ref_status = "not_installed"
    report["stages"]["unit_tests"] = {
        "test_core": "pass",
        "test_inkling_real_te": "pass",
        "test_reference": ref_status,
    }

    # ------------------------------------------------------------------
    # Stage 3: Determinism & m_splits Alignment Diagnostic
    # ------------------------------------------------------------------
    banner("STAGE 3: Determinism Diagnostic & 5b-Width SparseMoeBlock (te_align 32 vs 128)")
    from loopfp4 import Inkling, InklingConfig, PRESETS
    from loopfp4.inkling import SparseMoeBlock

    torch.manual_seed(42)
    base_cfg = PRESETS["smoke"]
    sim_model = Inkling(InklingConfig(**{**base_cfg.__dict__, "backend": "sim", "quant": "nvfp4", "backward": "mxfp8"})).cuda().eval()
    idx = torch.randint(0, 256, (3, 114), device="cuda")
    x, y = idx[:, :-1], idx[:, 1:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits_sim, loss_sim = sim_model(x, y)
    sim_sd = sim_model.canonical_state_dict()
    del sim_model
    torch.cuda.empty_cache()

    # 3a. Same-model repeat check & cross-instance check on smoke (H=128, inner=64)
    smoke_repeat = {}
    for r in ("none", "tml", "rowscaled", "nvidia"):
        try:
            q_mode = "none" if r == "none" else "nvfp4"
            r_arg = "tml" if r == "none" else r
            cfg_a = InklingConfig(**{**base_cfg.__dict__, "backend": "te", "quant": q_mode, "te_recipe": r_arg, "te_align": 128})
            m_a = Inkling(cfg_a).cuda().eval()
            m_a.load_canonical(sim_sd)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                la1, _ = m_a(x, y)
                la2, _ = m_a(x, y)
            same_diff = ((la1 - la2).norm() / la1.norm()).item()

            m_b = Inkling(cfg_a).cuda().eval()
            m_b.load_canonical(sim_sd)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lb1, _ = m_b(x, y)
            cross_diff = ((la1 - lb1).norm() / la1.norm()).item()
            vs_sim = ((la1 - logits_sim).norm() / logits_sim.norm()).item()
            smoke_repeat[r] = {
                "status": "ok",
                "same_model_2_calls_rel_diff": round(same_diff, 6),
                "two_instances_rel_diff": round(cross_diff, 6),
                "rel_logit_diff_vs_sim": round(vs_sim, 5),
            }
            print(f"  smoke eval ({r:9s}): {smoke_repeat[r]}", flush=True)
            del m_a, m_b
        except Exception as err:
            smoke_repeat[r] = {"status": "error", "error": repr(err)}
            print(f"  smoke eval ({r:9s}): ERROR {err!r}", flush=True)

    # 3b. Single SparseMoeBlock at 5b width (H=1024, inner=512, 512 % 128 == 0 -> fused NVFP4 kernels)
    moe_diag = {}
    working_recipe = next((r for r in ("tml", "rowscaled", "nvidia") if smoke_repeat[r]["status"] == "ok"), None)
    assert working_recipe is not None, f"No TE recipe succeeded: {smoke_repeat}"
    if working_recipe != "tml":
        warnings.append(f"Primary recipe 'tml' failed ({smoke_repeat['tml'].get('error')}); Stages 4-6 used fallback '{working_recipe}'")

    for mode in ("nvfp4", "mxfp8"):
        for regime in ("dense", "sparse"):
            torch.manual_seed(7)
            cfg_128 = dataclasses.replace(PRESETS["5b"], n_routed_experts=32, num_experts_per_tok=4,
                                          backend="te", te_recipe=working_recipe, te_align=128)
            cfg_32 = dataclasses.replace(cfg_128, te_align=32)
            moe_128 = SparseMoeBlock(cfg_128, mode=mode).cuda().train()
            if regime == "sparse":
                with torch.no_grad():
                    moe_128.gate.e_score_correction_bias[16:].fill_(-50.0)  # guarantee 16+ idle 0-token experts
                inp = torch.randn(1, 32, cfg_128.hidden_size, device="cuda", dtype=torch.bfloat16)
            else:
                inp = torch.randn(4, 1024, cfg_128.hidden_size, device="cuda", dtype=torch.bfloat16)
            sd_moe = moe_128.state_dict()
            moe_32 = SparseMoeBlock(cfg_32, mode=mode).cuda().train()
            moe_32.load_state_dict(sd_moe, strict=False)

            x1 = inp.detach().clone().requires_grad_(True)
            x2 = inp.detach().clone().requires_grad_(True)
            x3 = inp.detach().clone().requires_grad_(True)
            g_out = torch.randn_like(inp)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_128 = moe_128(x1)
            (y_128 * g_out).sum().backward()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_32_a = moe_32(x2)
            (y_32_a * g_out).sum().backward()
            g_32 = {n: p.grad.detach().float().clone() for n, p in moe_32.named_parameters() if p.grad is not None}
            gx_32 = x2.grad.detach().float().clone()
            moe_32.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_32_b = moe_32(x3)
            (y_32_b * g_out).sum().backward()

            same_fwd = ((y_32_a - y_32_b).norm() / y_32_a.norm().clamp(min=1e-8)).item()
            vs128_fwd = ((y_32_a - y_128).norm() / y_128.norm().clamp(min=1e-8)).item()
            gx_diff = ((gx_32 - x1.grad.float()).norm() / x1.grad.float().norm().clamp(min=1e-8)).item()
            assert (torch.isfinite(y_128).all() and torch.isfinite(y_32_a).all() and torch.isfinite(y_32_b).all()
                    and torch.isfinite(x1.grad).all() and torch.isfinite(gx_32).all()), f"non-finite in {mode}_{regime}"
            w_diffs = []
            for n, p in moe_128.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all() and torch.isfinite(g_32[n]).all(), f"non-finite wgrad {n}"
                    if p.grad.norm() > 0:
                        w_diffs.append(((g_32[n] - p.grad.float()).norm() / p.grad.float().norm()).item())
            max_w_diff = max(w_diffs) if w_diffs else 0.0

            key = f"{mode}_{regime}"
            moe_diag[key] = {
                "same_32_2_calls_fwd_diff": round(same_fwd, 6),
                "align_32_vs_128_fwd_diff": round(vs128_fwd, 6),
                "align_32_vs_128_dgrad_diff": round(gx_diff, 6),
                "align_32_vs_128_max_wgrad_diff": round(max_w_diff, 6),
            }
            print(f"  5b-width MoE ({key:12s}): {moe_diag[key]}", flush=True)
            del moe_128, moe_32, x1, x2, x3, inp, g_out, g_32, gx_32, sd_moe

    align_32_ok = all(
        math.isfinite(v["align_32_vs_128_fwd_diff"])
        and v["align_32_vs_128_fwd_diff"] < 0.02
        and v["align_32_vs_128_dgrad_diff"] < 0.05
        and v["align_32_vs_128_max_wgrad_diff"] < 0.05
        for v in moe_diag.values()
    )
    best_align = 32 if align_32_ok else 128
    if not align_32_ok:
        warnings.append(f"5b-width MoE te_align=32 diverged from 128; using te_align=128")
    print(f"  => Verified m_splits alignment for B200: te_align={best_align}", flush=True)

    # 3c. 5-second 5b layer bottleneck breakdown (bs=8, T=2048): Attention vs MoE backends/recipes
    banner("STAGE 3c: 5b Single-Layer Component Timing Breakdown (bs=8, T=2048)")
    from loopfp4.inkling import Attention
    cfg_5b_base = PRESETS["5b"]
    T_prof, B_prof = 2048, 8
    pos_p = torch.arange(T_prof, device="cuda")
    dist_p = pos_p[:, None] - pos_p[None, :]
    mask_p = torch.zeros(T_prof, T_prof, device="cuda").masked_fill((dist_p < 0) | (dist_p >= cfg_5b_base.sliding_window_size), float("-inf"))
    x_p = torch.randn(B_prof, T_prof, cfg_5b_base.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def bench_comp(name, fn):
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 3 * 1000
        print(f"  {name:32s}: {ms:7.1f} ms / fwd+bwd", flush=True)
        return round(ms, 2)

    profile_ms = {}
    attn_mod = Attention(cfg_5b_base, 0).cuda().train()
    def run_attn():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = attn_mod(x_p, x_p, x_p, mask_p, dist_p, pos_p)
        o.sum().backward()
        attn_mod.zero_grad(set_to_none=True)
        x_p.grad = None
    profile_ms["self_attn_swa_dense_fp32"] = bench_comp("self_attn (swa dense fp32)", run_attn)
    del attn_mod

    for label, bk, rec, al in (
        ("moe_te_tml_align32", "te", "tml", 32),
        ("moe_te_tml_align128", "te", "tml", 128),
        ("moe_te_rowscaled_align32", "te", "rowscaled", 32),
        ("moe_te_nvidia_align32", "te", "nvidia", 32),
        ("moe_sim_nvfp4", "sim", "tml", 1),
    ):
        mod_c = None
        try:
            cfg_c = dataclasses.replace(cfg_5b_base, backend=bk, te_recipe=rec, te_align=al)
            mod_c = SparseMoeBlock(cfg_c, mode="nvfp4").cuda().train()
            def run_moe(m=mod_c):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(x_p)
                o.sum().backward()
                m.zero_grad(set_to_none=True)
                x_p.grad = None
            profile_ms[label] = bench_comp(label, run_moe)
        except Exception as err:
            print(f"  {label:32s}: ERROR {err!r}", flush=True)
            profile_ms[label] = {"error": repr(err)}
        finally:
            del mod_c
            x_p.grad = None
            torch.cuda.empty_cache()
    del pos_p, dist_p, mask_p, x_p

    report["stages"]["alignment_and_determinism"] = {
        "best_align": best_align,
        "smoke_repeat_eval": smoke_repeat,
        "moe_5b_width_32_vs_128": moe_diag,
    }
    report["stages"]["5b_layer_profile_ms"] = profile_ms
    del logits_sim, sim_sd
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 4: Optional End-to-End Smoke Pipeline (--full)
    # ------------------------------------------------------------------
    if run_smoke_e2e:
        banner(f"STAGE 4: End-to-End Pipeline on TinyStories (smoke, recipe={working_recipe}, align={best_align})")
        with tempfile.TemporaryDirectory() as tmp:
            te_dir = os.path.join(tmp, "smoke-te")
            sim_dir = os.path.join(tmp, "smoke-sim")
            heal_dir = os.path.join(tmp, "smoke-healed")
            pruned_pt = os.path.join(te_dir, "pruned.pt")
            exported_pt = os.path.join(heal_dir, "model-nvfp4.pt")

            run_cmd([sys.executable, "train.py", "--preset", "smoke", "--data", "data/tinystories-bytes",
                     "--out", te_dir, "--backend", "te", "--te-recipe", working_recipe, "--te-align", str(best_align),
                     "--batch-size", "16", "--steps", "150", "--warmup", "30",
                     "--log-every", "50", "--eval-every", "150", "--save-every", "150"])
            run_cmd([sys.executable, "train.py", "--preset", "smoke", "--data", "data/tinystories-bytes",
                     "--out", sim_dir, "--backend", "sim",
                     "--batch-size", "16", "--steps", "150", "--warmup", "30",
                     "--log-every", "50", "--eval-every", "150", "--save-every", "150"])
            run_cmd([sys.executable, "prune_experts.py", os.path.join(te_dir, "ckpt.pt"), pruned_pt, "--keep", "16"])
            run_cmd([sys.executable, "train.py", "--init", pruned_pt, "--data", "data/tinystories-bytes",
                     "--out", heal_dir, "--backend", "te", "--te-recipe", working_recipe, "--te-align", str(best_align),
                     "--batch-size", "16", "--steps", "50", "--warmup", "10",
                     "--log-every", "25", "--eval-every", "50", "--save-every", "50"])
            run_cmd([sys.executable, "export_nvfp4.py", os.path.join(heal_dir, "ckpt.pt"), exported_pt])
            sample_ck = run_cmd([sys.executable, "sample.py", exported_pt, "Once upon a time"])

            def read_jsonl(p):
                with open(p) as f:
                    return [json.loads(line) for line in f if line.strip()]

            report["stages"]["smoke_e2e"] = {
                "recipe_used": working_recipe,
                "te_align": best_align,
                "te_log": read_jsonl(os.path.join(te_dir, "log.jsonl")),
                "sim_log": read_jsonl(os.path.join(sim_dir, "log.jsonl")),
                "heal_log": read_jsonl(os.path.join(heal_dir, "log.jsonl")),
                "exported_sample_preview": sample_ck.stdout[:240],
            }
    else:
        print("\n[Skipping Stage 4 smoke e2e CLI pipeline (already verified in prior run; pass --full to re-run)]", flush=True)

    # ------------------------------------------------------------------
    # Stage 5: Real-Scale 5B Preset at Production Context Length (block_size=2048)
    # ------------------------------------------------------------------
    banner(f"STAGE 5: 5B Preset Training Pass on 1x B200 (recipe={working_recipe}, te_align={best_align})")
    with tempfile.TemporaryDirectory() as tmp:
        data_200k = os.path.join(tmp, "data-200k")
        os.makedirs(data_200k, exist_ok=True)
        rng = np.random.default_rng(0)
        rng.integers(0, 200058, size=2_000_000, dtype=np.uint32).tofile(os.path.join(data_200k, "train_00000.bin"))
        rng.integers(0, 200058, size=200_000, dtype=np.uint32).tofile(os.path.join(data_200k, "val_00000.bin"))
        with open(os.path.join(data_200k, "meta.json"), "w") as f:
            json.dump({"dtype": "uint32", "vocab_size": 200058, "tokenizer": "synthetic-200k"}, f)

        torch.cuda.empty_cache()
        cfg_5b = InklingConfig(**{
            **PRESETS["5b"].__dict__,
            "backend": "te",
            "te_recipe": working_recipe,
            "te_align": best_align,
            "block_size": 2048,
            "grad_checkpoint": True,
        })
        print(f"Building 5B model (vocab={cfg_5b.vocab_size}, unpadded={cfg_5b.unpadded_vocab_size}, "
              f"block_size={cfg_5b.block_size}, te_align={best_align}, recipe={working_recipe})...", flush=True)
        t_build = time.time()
        model_5b = Inkling(cfg_5b).cuda()
        print(f"Built 5B on GPU in {time.time() - t_build:.1f}s ({model_5b.num_params() / 1e9:.3f}B params)", flush=True)

        skip = ("scale", "query", "norm", "static_weight", "conv1d")
        named = list(model_5b.named_parameters())
        decay = [p for n, p in named if p.dim() >= 2 and not any(k in n for k in skip)]
        no_decay = [p for n, p in named if not (p.dim() >= 2 and not any(k in n for k in skip))]
        opt_5b = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=2e-4, betas=(0.9, 0.95), fused=True,
        )

        from loopfp4.data import ShardData
        # Run 2 full steps inside the probe so AdamW's 43.2 GB states exist before peak activation check
        chosen_bs = None
        for cand_bs in (8, 4, 2):
            xb = yb = loss = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                probe_loader = ShardData(data_200k, "train", cfg_5b.block_size, batch_size=cand_bs, device="cuda", seed=0)
                for _ in range(2):
                    xb, yb = probe_loader.batch()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model_5b(xb, yb)
                    loss.backward()
                    opt_5b.step()
                    opt_5b.zero_grad(set_to_none=True)
                    model_5b.balance_step(1e-3)
                    del xb, yb, loss
                    xb = yb = loss = None
                torch.cuda.synchronize()
                chosen_bs = cand_bs
                print(f"  Batch probe batch_size={cand_bs} x {cfg_5b.block_size} ({cand_bs * cfg_5b.block_size:,} tok/step) FIT w/ Adam states "
                      f"(peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"  Batch probe batch_size={cand_bs} x {cfg_5b.block_size} OOM; trying smaller...", flush=True)
                del xb, yb, loss
                xb = yb = loss = None
                opt_5b.zero_grad(set_to_none=True)
                for moe in model_5b.moe_blocks():
                    moe.step_counts.zero_()
                gc.collect()
                torch.cuda.empty_cache()
        assert chosen_bs is not None, "Even batch_size=2 x 2048 OOMed on 5B"

        torch.cuda.reset_peak_memory_stats()
        loader = ShardData(data_200k, "train", cfg_5b.block_size, batch_size=chosen_bs, device="cuda", seed=1)
        tokens_per_step = chosen_bs * cfg_5b.block_size
        losses_5b = []
        step_times = []
        model_5b.train()
        for s in range(1, 7):
            torch.cuda.synchronize()
            ts = time.time()
            xb, yb = loader.batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model_5b(xb, yb)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model_5b.parameters(), 1.0).item()
            opt_5b.step()
            opt_5b.zero_grad(set_to_none=True)
            model_5b.balance_step(1e-3)
            torch.cuda.synchronize()
            dt = time.time() - ts
            step_times.append(dt)
            losses_5b.append(round(loss.item(), 4))
            print(f"  5b TE step {s:2d}/6 (bs={chosen_bs}x2048) | loss={loss.item():.4f} | gnorm={gnorm:.3f} | "
                  f"dt={dt*1000:.0f}ms | tok/s={tokens_per_step / dt:,.0f} | "
                  f"gate={model_5b.loop_gate_strength():.6f}", flush=True)

        peak_gb_5b = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        steady_tps_te = round(tokens_per_step / (sum(step_times[2:]) / len(step_times[2:])))
        print(f"  5B TE (bs={chosen_bs}x2048, align={best_align}) peak VRAM: {peak_gb_5b} GB | "
              f"steady-state throughput: {steady_tps_te:,} tok/s/B200", flush=True)

        del opt_5b, model_5b, decay, no_decay, named
        gc.collect()
        torch.cuda.empty_cache()

        steady_tps_sim = None
        peak_gb_sim = None
        sim_bs = chosen_bs
        if run_smoke_e2e:
            cfg_5b_sim = InklingConfig(**{**PRESETS["5b"].__dict__, "backend": "sim", "block_size": 2048, "grad_checkpoint": True})
            model_5b_sim = Inkling(cfg_5b_sim).cuda()
            opt_sim = torch.optim.AdamW(model_5b_sim.parameters(), lr=2e-4, betas=(0.9, 0.95), fused=True)
            for cand_sim_bs in dict.fromkeys([chosen_bs, max(1, chosen_bs // 2)]):
                xb = yb = loss = None
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    sim_loader = ShardData(data_200k, "train", cfg_5b.block_size, batch_size=cand_sim_bs, device="cuda", seed=2)
                    sim_times = []
                    for s in range(1, 4):
                        torch.cuda.synchronize()
                        ts = time.time()
                        xb, yb = sim_loader.batch()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, loss = model_5b_sim(xb, yb)
                        loss.backward()
                        opt_sim.step()
                        opt_sim.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        sim_times.append(time.time() - ts)
                        del xb, yb, loss
                        xb = yb = loss = None
                    sim_bs = cand_sim_bs
                    steady_tps_sim = round((sim_bs * cfg_5b.block_size) / (sum(sim_times[1:]) / len(sim_times[1:])))
                    peak_gb_sim = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    break
                except torch.cuda.OutOfMemoryError:
                    del xb, yb, loss
                    xb = yb = loss = None
                    opt_sim.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            del opt_sim, model_5b_sim
            gc.collect()
            torch.cuda.empty_cache()

        plan_out = run_cmd([sys.executable, "plan.py", "--preset", "5b", "--tok-per-s-per-gpu", str(steady_tps_te)])
        report["stages"]["5b_scale"] = {
            "recipe": working_recipe,
            "te_align": best_align,
            "batch_shape": f"{chosen_bs}x{cfg_5b.block_size}",
            "tokens_per_step": tokens_per_step,
            "losses_6_steps": losses_5b,
            "peak_vram_gb_te": peak_gb_5b,
            "sim_batch_shape": f"{sim_bs}x{cfg_5b.block_size}" if steady_tps_sim else None,
            "peak_vram_gb_sim": peak_gb_sim,
            "tok_per_s_te": steady_tps_te,
            "tok_per_s_sim": steady_tps_sim,
            "te_speedup_vs_sim": round(steady_tps_te / max(steady_tps_sim, 1), 2) if steady_tps_sim else None,
            "plan_with_measured_tps": plan_out.stdout,
        }

    # ------------------------------------------------------------------
    # Stage 6: Full 75B mini-pretrain Single MoE Layer Kernel Check
    # ------------------------------------------------------------------
    banner(f"STAGE 6: 75B mini-pretrain Expert Shape Check (E=384, H=2304, inner=1152)")
    mini_configs = [
        (working_recipe, "nvfp4", 32),
        (working_recipe, "mxfp8", 32),
        (working_recipe, "nvfp4", 128),
        ("nvidia", "nvfp4", 32),
    ]
    for rec, mode, al in mini_configs:
        moe = x_mini = out = None
        try:
            mini_cfg = InklingConfig(**{**PRESETS["mini-pretrain"].__dict__, "backend": "te",
                                        "te_recipe": rec, "te_align": al})
            torch.cuda.reset_peak_memory_stats()
            moe = SparseMoeBlock(mini_cfg, mode=mode).cuda()
            x_mini = torch.randn(8, 2048, mini_cfg.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            torch.cuda.synchronize()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = moe(x_mini)
            out.sum().backward()
            moe.zero_grad(set_to_none=True)
            x_mini.grad = None
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = moe(x_mini)
                out.sum().backward()
                moe.zero_grad(set_to_none=True)
                x_mini.grad = None
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 5 * 1000
            print(f"  mini-pretrain 384-expert SparseMoeBlock ({rec:6s}/{mode:5s}, align={al:3d}, 16384 tok): {ms:.1f} ms/fwd+bwd "
                  f"(peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)", flush=True)
            report["stages"][f"mini_384_moe_{rec}_{mode}_align{al}"] = {"ms_per_fwd_bwd": round(ms, 2), "tokens": 16384}
        except Exception as err:
            print(f"  mini-pretrain 384-expert SparseMoeBlock ({rec:6s}/{mode:5s}, align={al:3d}): ERROR {err!r}", flush=True)
            report["stages"][f"mini_384_moe_{rec}_{mode}_align{al}"] = {"error": repr(err)}
        finally:
            del moe, x_mini, out
            gc.collect()
            torch.cuda.empty_cache()

    total_sec = round(time.time() - t_start, 1)
    est_usd = round(total_sec * 0.001736, 3)
    if warnings:
        banner(f"FINISHED WITH {len(warnings)} WARNING(S) in {total_sec}s (~${est_usd} GPU compute)")
        for w in warnings:
            print(f"  !! {w}", flush=True)
    else:
        banner(f"ALL B200 VERIFICATION STAGES PASSED (tml + reference + align={best_align}) in {total_sec}s (~${est_usd} GPU compute)")
    report["gpu_seconds"] = total_sec
    report["estimated_gpu_usd"] = est_usd
    return report


@app.local_entrypoint()
def main(full: bool = False):
    report = verify_b200.remote(run_smoke_e2e=full)
    out_path = os.path.join(ROOT, "runs", "b200-verify-report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full B200 verification report to {out_path}")
