"""Historical Part 1 verification harness for FlexAttention, chunked CE, TE GroupedLinear perf, and 2-GPU NCCL DDP.
For the current Part 2 harness (--backend grouped, Triton RPB+Sink attention, SAC, 2x B200 DDP), see modal_verify_part2.py.

Usage:
    modal run modal_verify_ddp_and_perf.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
import modal

ROOT = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "loopfp4-ddp-perf-verify"

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


def _time_cuda(fn, warmup: int = 2, iters: int = 5):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000.0
    gb = torch.cuda.max_memory_allocated() / 1e9
    return ms, gb


@app.function(image=image, gpu="B200", timeout=1800)
def verify_perf_1gpu() -> dict:
    import contextlib
    import gc
    import inspect
    from dataclasses import replace

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    import transformer_engine.pytorch as te

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    from loopfp4 import PRESETS, build
    from loopfp4.inkling import Attention, SparseMoeBlock, _flex_attn_compiled, _get_block_mask, softcap

    t_start = time.time()
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "exactness": {},
        "te_inspection": {},
        "single_layer_5b": {},
        "full_5b": {},
    }

    # ------------------------------------------------------------------
    # 1. Exactness: FlexAttention & Chunked CE on CUDA
    # ------------------------------------------------------------------
    banner("PERF-1A: FlexAttention + Sink + RPB & Chunked-CE Exactness (CUDA)")
    fp32_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    report["exactness"]["fp32_env"] = {
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "get_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_fp32_precision": fp32_prec,
    }
    print("  fp32_env:", report["exactness"]["fp32_env"], flush=True)
    torch.set_float32_matmul_precision("highest")
    if fp32_prec not in (None, "none"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"

    B, Hq, Hkv, T, D = 2, 12, 4, 512, 128
    pos = torch.arange(T, device="cuda")
    dist = pos[:, None] - pos[None, :]
    base_causal = torch.zeros(T, T, device="cuda").masked_fill(dist < 0, float("-inf"))

    for kind, window, ext in [("swa", 512, 512), ("global", None, 1024)]:
        mask = base_causal if window is None else base_causal.masked_fill(dist >= window, float("-inf"))
        bm = _get_block_mask(T, window, torch.device("cuda"))
        for dt_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            torch.manual_seed(42)
            q0 = torch.randn(B, Hq, T, D, device="cuda", dtype=dtype)
            k0 = torch.randn(B, Hkv, T, D, device="cuda", dtype=dtype)
            v0 = torch.randn(B, Hkv, T, D, device="cuda", dtype=dtype)
            r0 = torch.randn(B, Hq, T, ext, device="cuda", dtype=torch.float32) * 0.2
            s0 = torch.randn(B, Hq, T, device="cuda", dtype=torch.float32) * 0.5
            g_out = torch.randn(B, Hq, T, D, device="cuda", dtype=dtype)

            q1, k1, v1, r1, s1 = [x.detach().clone().requires_grad_(True) for x in (q0, k0, v0, r0, s0)]
            q2, k2, v2, r2, s2 = [x.detach().clone().requires_grad_(True) for x in (q0, k0, v0, r0, s0)]

            group = Hq // Hkv
            sc = torch.matmul(q1.reshape(B, Hkv, group, T, D), k1.unsqueeze(2).transpose(-1, -2)).reshape(B, Hq, T, T)
            index = dist.clamp(0, ext - 1).expand(B, Hq, -1, -1)
            rpb = r1.gather(-1, index).masked_fill((dist < 0) | (dist >= ext), 0.0)
            sc = sc * (D ** -0.5) + rpb + mask
            att = F.softmax(torch.cat([sc, s1.unsqueeze(-1)], dim=-1).float(), dim=-1)[..., :-1].to(dtype)
            out1 = torch.matmul(att.reshape(B, Hkv, group, T, T), v1.unsqueeze(2)).reshape(B, Hq, T, D)
            (out1 * g_out).sum().backward()

            out2 = _flex_attn_compiled(q2, k2, v2, r2, s2, bm, ext)
            (out2 * g_out).sum().backward()

            fwd_diff = (out1.float() - out2.float()).abs().max().item()
            fwd_rel = (out1.float() - out2.float()).norm().item() / max(out1.float().norm().item(), 1e-6)
            grad_diffs = {
                n: (a.grad.float() - b.grad.float()).abs().max().item()
                for n, a, b in [("q", q1, q2), ("k", k1, k2), ("v", v1, v2), ("rel_logits", r1, r2), ("sinks", s1, s2)]
            }
            grad_rel_l2 = {
                n: (a.grad.float() - b.grad.float()).norm().item() / max(a.grad.float().norm().item(), 1e-6)
                for n, a, b in [("q", q1, q2), ("k", k1, k2), ("v", v1, v2), ("rel_logits", r1, r2), ("sinks", s1, s2)]
            }
            rec = {
                "fwd_max_diff": fwd_diff,
                "fwd_rel_l2": fwd_rel,
                **{f"d_{k}_max_diff": v for k, v in grad_diffs.items()},
                **{f"d_{k}_rel_l2": v for k, v in grad_rel_l2.items()},
            }
            if dtype == torch.float32:
                rec["ieee_exact"] = fwd_diff < 5e-4 and max(grad_diffs.values()) < 2e-3
            report["exactness"][f"raw_flex_{kind}_{dt_name}"] = rec
            print(f"  raw flex {kind:6s} {dt_name}: fwd={fwd_diff:.2e}(rel={fwd_rel:.2e}) "
                  + " ".join(f"d_{k}={v:.2e}(rel={grad_rel_l2[k]:.2e})" for k, v in grad_diffs.items()), flush=True)
            tol_rel = 5e-3 if dtype == torch.float32 else 2e-2
            assert fwd_rel < tol_rel, f"raw flex {kind} {dt_name} fwd_rel={fwd_rel}"
            for k_name, v_rel in grad_rel_l2.items():
                assert v_rel < tol_rel, f"raw flex {kind} {dt_name} d_{k_name}_rel={v_rel}"

    # Full Attention module parity on 5b layer 0 (SWA) & layer 5 (global) under BF16 autocast
    cfg5 = PRESETS["5b"]
    for li, kind in [(0, "swa"), (5, "global")]:
        torch.manual_seed(7)
        a_eager = Attention(replace(cfg5, use_flex_attn=False), li).cuda()
        a_flex = Attention(replace(cfg5, use_flex_attn=True), li).cuda()
        a_flex.load_state_dict(a_eager.state_dict())
        h_in = torch.randn(B, T, cfg5.hidden_size, device="cuda", dtype=torch.bfloat16)
        h1 = h_in.detach().clone().requires_grad_(True)
        h2 = h_in.detach().clone().requires_grad_(True)
        mask = base_causal if li == 5 else base_causal.masked_fill(dist >= cfg5.sliding_window_size, float("-inf"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o1 = a_eager(h1, h1, h1, mask, dist, pos)
            o2 = a_flex(h2, h2, h2, None, None, pos)
        g_attn = torch.randn_like(o1)
        (o1 * g_attn).sum().backward()
        (o2 * g_attn).sum().backward()
        fdiff = (o1.float() - o2.float()).abs().max().item()
        hdiff = (h1.grad.float() - h2.grad.float()).abs().max().item()
        h_rel = (h1.grad.float() - h2.grad.float()).norm().item() / max(h1.grad.float().norm().item(), 1e-6)
        wdiff = max((p1.grad.float() - p2.grad.float()).abs().max().item()
                    for p1, p2 in zip(a_eager.parameters(), a_flex.parameters()))
        w_rel = max(
            (p1.grad.float() - p2.grad.float()).norm().item() / max(p1.grad.float().norm().item(), 1e-6)
            for p1, p2 in zip(a_eager.parameters(), a_flex.parameters())
        )
        report["exactness"][f"module_attn_5b_{kind}_bf16"] = {
            "fwd_max_diff": fdiff, "d_in_max_diff": hdiff, "d_in_rel_l2": h_rel,
            "d_param_max_diff": wdiff, "d_param_rel_l2": w_rel,
        }
        print(f"  5b Attention {kind:6s} BF16: fwd={fdiff:.2e} d_in={hdiff:.2e}(rel={h_rel:.2e}) d_param={wdiff:.2e}(rel={w_rel:.2e})", flush=True)
        assert fdiff < 5e-2 and h_rel < 2e-2 and w_rel < 2e-2
        del a_eager, a_flex

    # Full smoke Inkling end-to-end parity (FlexAttention + chunked CE + checkpointing) under BF16 autocast
    for qmode in ["none", "nvfp4"]:
        torch.manual_seed(11)
        c_e = replace(PRESETS["smoke"], grad_checkpoint=True, quant=qmode, backward="exact", use_flex_attn=False)
        c_f = replace(PRESETS["smoke"], grad_checkpoint=True, quant=qmode, backward="exact", use_flex_attn=True)
        m_e, m_f = build(c_e).cuda(), build(c_f).cuda()
        m_f.load_state_dict(m_e.state_dict())
        idx = torch.randint(0, c_e.unpadded_vocab_size, (2, 256), device="cuda")
        tgt = torch.randint(0, c_e.unpadded_vocab_size, (2, 256), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l_e = m_e(idx, tgt, return_logits=True)
            _, l_f = m_f(idx, tgt, return_logits=False)
        l_e.backward()
        l_f.backward()
        loss_diff = (l_e - l_f).abs().item()
        max_g = max((p1.grad.float() - p2.grad.float()).abs().max().item()
                    for p1, p2 in zip(m_e.parameters(), m_f.parameters()) if p1.grad is not None)
        report["exactness"][f"smoke_e2e_bf16_{qmode}"] = {"loss_diff": loss_diff, "max_grad_diff": max_g}
        print(f"  smoke e2e ({qmode:5s}, T=256, BF16): loss_diff={loss_diff:.2e} max_grad_diff={max_g:.2e}", flush=True)
        assert loss_diff < 5e-3
        del m_e, m_f

    # Chunked CE parity at 200k vocab on CUDA (N=2048 tokens = 2 chunks)
    torch.manual_seed(13)
    head = nn.Linear(cfg5.hidden_size, cfg5.unpadded_vocab_size, bias=False, device="cuda")
    h0 = torch.randn(2048, cfg5.hidden_size, device="cuda", dtype=torch.bfloat16)
    tgt_ce = torch.randint(0, cfg5.unpadded_vocab_size, (2048,), device="cuda")
    h_ref = h0.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l_ref = F.cross_entropy(softcap(head(h_ref).float(), cfg5.final_logit_softcapping), tgt_ce)
    l_ref.backward()
    g_w_ref = head.weight.grad.detach().clone()
    head.zero_grad(set_to_none=True)

    h_chk = h0.detach().clone().requires_grad_(True)
    def _ce_chunk(hc, tc):
        return F.cross_entropy(softcap(head(hc).float(), cfg5.final_logit_softcapping), tc, reduction="sum")
    tot = h_chk.new_zeros((), dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, 2048, 1024):
            tot = tot + checkpoint(_ce_chunk, h_chk[s : s + 1024], tgt_ce[s : s + 1024], use_reentrant=False)
        l_chk = tot / 2048
    l_chk.backward()
    ce_rec = {
        "loss_diff": (l_ref - l_chk).abs().item(),
        "d_hidden_max_diff": (h_ref.grad.float() - h_chk.grad.float()).abs().max().item(),
        "d_head_max_diff": (g_w_ref.float() - head.weight.grad.float()).abs().max().item(),
    }
    report["exactness"]["chunked_ce_200k_cuda_bf16"] = ce_rec
    print(f"  200k-vocab chunked CE CUDA BF16: loss_diff={ce_rec['loss_diff']:.2e} "
          f"d_h={ce_rec['d_hidden_max_diff']:.2e} d_w={ce_rec['d_head_max_diff']:.2e}", flush=True)
    assert ce_rec["loss_diff"] < 1e-4 and ce_rec["d_hidden_max_diff"] < 1e-4
    del head, h0, h_ref, h_chk, g_w_ref, tgt_ce
    if fp32_prec not in (None, "none"):
        torch.backends.cuda.matmul.fp32_precision = "none"
    torch.set_float32_matmul_precision("high")
    torch._dynamo.reset()
    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. TE GroupedLinear fast-path inspection
    # ------------------------------------------------------------------
    banner("PERF-1B: TE GroupedLinear Inspection & Single-Layer 5b Benchmarks (bs=8, T=2048)")
    try:
        from transformer_engine.pytorch.module import grouped_linear as gl_mod
        te_info = {"file": getattr(gl_mod, "__file__", None)}
        gl_src = inspect.getsource(gl_mod.GroupedLinear)
        fn_hits = [name for name in dir(gl_mod) if "grouped" in name.lower()]
        te_info["grouped_helpers"] = fn_hits
        check_fn = getattr(gl_mod, "is_module_grouped_tensor_path_supported", None)
        if check_fn is not None:
            te_info["is_module_grouped_tensor_path_supported_src"] = inspect.getsource(check_fn)
        snippet_lines = [line for line in gl_src.splitlines() if any(
            k in line for k in ("grouped", "Quantizer", "Hybrid", "params_dtype", "m_splits", "fallback", "Custom")
        )]
        te_info["grouped_linear_key_lines"] = snippet_lines[:40]
        report["te_inspection"] = te_info
        print("  TE grouped_linear helpers:", fn_hits, flush=True)
        if check_fn is not None:
            print(te_info["is_module_grouped_tensor_path_supported_src"], flush=True)
        for line in snippet_lines[:20]:
            print("   ", line, flush=True)
    except Exception as err:
        report["te_inspection"] = {"error": repr(err)}
        print(f"  TE inspection error: {err!r}", flush=True)

    # Single-layer 5b Attention benchmark: Eager vs FlexAttention at bs=8, T=2048
    B8, T2 = 8, 2048
    pos2 = torch.arange(T2, device="cuda")
    dist2 = pos2[:, None] - pos2[None, :]
    base2 = torch.zeros(T2, T2, device="cuda").masked_fill(dist2 < 0, float("-inf"))
    for li, kind in [(0, "swa"), (5, "global")]:
        m_full = base2 if li == 5 else base2.masked_fill(dist2 >= cfg5.sliding_window_size, float("-inf"))
        for use_flex in [False, True]:
            attn = Attention(replace(cfg5, use_flex_attn=use_flex), li).cuda()
            x_a = torch.randn(B8, T2, cfg5.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            def _step_attn():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = attn(x_a, x_a, x_a, None if use_flex else m_full, None if use_flex else dist2, pos2)
                o.sum().backward()
                attn.zero_grad(set_to_none=True)
                x_a.grad = None
            ms, gb = _time_cuda(_step_attn, warmup=2, iters=5)
            key = f"attn_{kind}_{'flex' if use_flex else 'eager'}"
            report["single_layer_5b"][key] = {"ms_fwd_bwd": round(ms, 2), "peak_gb": round(gb, 2)}
            print(f"  {key:20s}: {ms:7.2f} ms/fwd+bwd  peak={gb:5.2f} GB", flush=True)
            del attn, x_a
            gc.collect(); torch.cuda.empty_cache()

    del pos2, dist2, base2
    gc.collect(); torch.cuda.empty_cache()

    # Single-layer 5b SparseMoeBlock benchmark (bs=8, T=2048)
    def _bench_moe(label: str, moe_fn, x_in):
        ms, gb = _time_cuda(lambda: moe_fn(x_in), warmup=2, iters=5)
        report["single_layer_5b"][label] = {"ms_fwd_bwd": round(ms, 2), "peak_gb": round(gb, 2)}
        print(f"  {label:30s}: {ms:7.2f} ms/fwd+bwd  peak={gb:5.2f} GB", flush=True)

    for rec_style, rec_mode, p_dtype, use_list in [
        ("tml", "nvfp4", torch.float32, False),
        ("tml", "nvfp4", torch.bfloat16, False),
        ("tml", "nvfp4", torch.float32, True),
        ("tml", "nvfp4", torch.bfloat16, True),
        ("tml", "mxfp8", torch.float32, False),
        ("tml", "mxfp8", torch.bfloat16, True),
    ]:
        tag = f"moe_{rec_style}_{rec_mode}_{'bf16' if p_dtype == torch.bfloat16 else 'fp32'}_{'list' if use_list else 'tensor'}"
        try:
            mc = replace(cfg5, backend="te", te_recipe=rec_style, te_align=32)
            moe = SparseMoeBlock(mc, mode=rec_mode).cuda()
            if p_dtype == torch.bfloat16:
                moe.experts = nn.Module()
                moe.experts.gate_up = te.GroupedLinear(
                    mc.n_routed_experts, mc.hidden_size, 2 * mc.moe_intermediate_size, bias=False, params_dtype=torch.bfloat16
                ).cuda()
                moe.experts.down = te.GroupedLinear(
                    mc.n_routed_experts, mc.moe_intermediate_size, mc.hidden_size, bias=False, params_dtype=torch.bfloat16
                ).cuda()
            x_m = torch.randn(B8, T2, mc.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)

            def _run_moe(xin, mod=moe, as_list=use_list):
                if as_list:
                    B_, T_, H_ = xin.shape
                    flat = xin.reshape(B_ * T_, H_)
                    index, weights = mod.route(flat)
                    expert = index.reshape(-1)
                    counts = torch.zeros(mc.n_routed_experts, dtype=torch.long, device=xin.device).scatter_add_(
                        0, expert, torch.ones_like(expert))
                    order = torch.argsort(expert, stable=True)
                    token = order // mod.top_k
                    padded = ((counts + 31) // 32) * 32
                    splits = padded.tolist()
                    pos_m = torch.arange(token.numel(), device=flat.device) + (
                        (padded.cumsum(0) - padded) - (counts.cumsum(0) - counts))[expert[order]]
                    xb = flat.new_zeros((sum(splits), H_), dtype=torch.bfloat16)
                    xb[pos_m] = flat[token].to(torch.bfloat16)
                    with te.autocast(enabled=True, recipe=mod.recipe):
                        g, u = mod.experts.gate_up(xb, splits).float().chunk(2, dim=-1)
                        act = softcap(g, mod.gate_cap) * torch.sigmoid(g) * softcap(u, mod.up_cap)
                        out_e = mod.experts.down(act.to(torch.bfloat16), splits)[pos_m].float()
                    y_flat = out_e.new_zeros((B_ * T_ * mod.top_k, H_))
                    y_flat[order] = out_e
                    y = (y_flat.view(B_ * T_, mod.top_k, H_) * weights.unsqueeze(-1)).sum(dim=1).view(B_, T_, H_).to(xin.dtype)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = mod(xin)
                y.sum().backward()
                mod.zero_grad(set_to_none=True)
                xin.grad = None

            _bench_moe(tag, _run_moe, x_m)
            del moe, x_m
        except Exception as err:
            report["single_layer_5b"][tag] = {"error": repr(err)}
            print(f"  {tag:30s}: ERROR {err!r}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    # torch._grouped_mm BF16 baseline (same 5b MoE sizes + align=32)
    try:
        gmm = getattr(torch, "_grouped_mm", None)
        assert gmm is not None, "torch._grouped_mm missing"
        mc = replace(cfg5, backend="sim")
        moe_g = SparseMoeBlock(mc, mode="none").cuda()
        E_, H_, inner_ = mc.n_routed_experts, mc.hidden_size, mc.moe_intermediate_size
        w1 = nn.Parameter(torch.randn(E_, H_, 2 * inner_, device="cuda", dtype=torch.bfloat16) * 0.02)
        w2 = nn.Parameter(torch.randn(E_, inner_, H_, device="cuda", dtype=torch.bfloat16) * 0.02)
        x_m = torch.randn(B8, T2, H_, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def _run_gmm(xin):
            B_, T_, _ = xin.shape
            flat = xin.reshape(B_ * T_, H_)
            index, weights = moe_g.route(flat)
            expert = index.reshape(-1)
            counts = torch.zeros(E_, dtype=torch.long, device=xin.device).scatter_add_(0, expert, torch.ones_like(expert))
            order = torch.argsort(expert, stable=True)
            token = order // moe_g.top_k
            padded = ((counts + 31) // 32) * 32
            splits = padded.tolist()
            pos_m = torch.arange(token.numel(), device=flat.device) + (
                (padded.cumsum(0) - padded) - (counts.cumsum(0) - counts))[expert[order]]
            xb = flat.new_zeros((sum(splits), H_), dtype=torch.bfloat16)
            xb[pos_m] = flat[token].to(torch.bfloat16)
            offs = padded.cumsum(0).to(torch.int32)
            g, u = gmm(xb, w1, offs=offs).float().chunk(2, dim=-1)
            act = softcap(g, moe_g.gate_cap) * torch.sigmoid(g) * softcap(u, moe_g.up_cap)
            out_e = gmm(act.to(torch.bfloat16), w2, offs=offs)[pos_m].float()
            y_flat = out_e.new_zeros((B_ * T_ * moe_g.top_k, H_))
            y_flat[order] = out_e
            y = (y_flat.view(B_ * T_, moe_g.top_k, H_) * weights.unsqueeze(-1)).sum(dim=1).view(B_, T_, H_).to(xin.dtype)
            y.sum().backward()
            moe_g.zero_grad(set_to_none=True)
            w1.grad = None; w2.grad = None; xin.grad = None

        _bench_moe("moe_torch_grouped_mm_bf16", _run_gmm, x_m)
        del moe_g, w1, w2, x_m
    except Exception as err:
        report["single_layer_5b"]["moe_torch_grouped_mm_bf16"] = {"error": repr(err)}
        print(f"  {'moe_torch_grouped_mm_bf16':30s}: ERROR {err!r}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Full 5b benchmark (FlexAttention + chunked CE) + plan.py
    # ------------------------------------------------------------------
    banner("PERF-1C: Full 5B Benchmark (FlexAttention + Chunked CE) & Cost Plan")
    best_tps = 0.0
    for label, bs, flex, do_compile in [
        ("bs8x2048_flex", 8, True, False),
        ("bs8x2048_eager_attn", 8, False, False),
        ("bs16x2048_flex", 16, True, False),
        ("bs8x2048_compile", 8, True, True),
    ]:
        try:
            if do_compile:
                torch._dynamo.reset()
            cfg = replace(PRESETS["5b"], backend="te", te_recipe="tml", te_align=32, use_flex_attn=flex)
            model = build(cfg).cuda()
            if do_compile:
                model = torch.compile(model)
            raw = getattr(model, "_orig_mod", model)
            opt = torch.optim.AdamW(raw.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
            xb = torch.randint(0, cfg.unpadded_vocab_size, (bs, 2048), device="cuda")
            yb = torch.randint(0, cfg.unpadded_vocab_size, (bs, 2048), device="cuda")

            def _train_step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(xb, yb, return_logits=False)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                raw.balance_step(1e-3)

            ms, peak_gb = _time_cuda(_train_step, warmup=2, iters=3)
            tps = round(bs * 2048 / (ms / 1000.0))
            best_tps = max(best_tps, tps)
            report["full_5b"][label] = {"tok_per_s": tps, "ms_per_step": round(ms, 1), "peak_gb": round(peak_gb, 2)}
            print(f"  full 5b {label:18s}: {tps:,} tok/s ({ms:.1f} ms/step, peak {peak_gb:.2f} GB)", flush=True)
            del model, raw, opt, xb, yb
        except Exception as err:
            report["full_5b"][label] = {"error": repr(err)}
            print(f"  full 5b {label:18s}: ERROR {err!r}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    if best_tps > 0:
        res_plan = run_cmd([
            sys.executable, "plan.py", "--preset", "5b", "--gpus", "8", "--tok-per-s-per-gpu", str(int(best_tps)),
        ])
        report["plan_5b"] = res_plan.stdout
    report["gpu_seconds"] = round(time.time() - t_start, 1)
    return report


@app.function(image=image, gpu="B200:2", timeout=1800)
def verify_ddp_2gpu() -> dict:
    import numpy as np
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    t_start = time.time()
    n_gpus = torch.cuda.device_count()
    assert n_gpus == 2, f"Expected 2 GPUs, got {n_gpus}"
    du_tmp = shutil.disk_usage("/tmp")
    mem_avail_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail_gb = round(int(line.split()[1]) / 1e6, 1)
                    break
    except Exception:
        pass

    report = {
        "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
        "tmp_free_gb": round(du_tmp.free / 1e9, 1),
        "mem_avail_gb": mem_avail_gb,
    }
    banner(f"DDP-2GPU: {n_gpus}x {report['gpus'][0]} (tmp_free={report['tmp_free_gb']} GB, ram_avail={mem_avail_gb} GB)")

    # 1. Smoke 2-GPU NCCL DDP + grad_accum=2 + save + --resume
    banner("DDP-2A: smoke 2-GPU NCCL DDP + grad_accum=2 + save + --resume")
    smoke_out = "/tmp/ddp-smoke"
    shutil.rmtree(smoke_out, ignore_errors=True)
    try:
        r1 = run_cmd([
            "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
            "--preset", "smoke", "--data", "data/tinystories-bytes", "--out", smoke_out,
            "--backend", "te", "--te-recipe", "tml", "--te-align", "32",
            "--batch-size", "4", "--grad-accum", "2", "--steps", "6",
            "--log-every", "1", "--eval-every", "3", "--eval-batches", "2", "--save-every", "3",
        ])
        assert os.path.exists(os.path.join(smoke_out, "ckpt.pt")), "smoke ckpt.pt not written"
        r2 = run_cmd([
            "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
            "--preset", "smoke", "--data", "data/tinystories-bytes", "--out", smoke_out, "--resume",
            "--backend", "te", "--te-recipe", "tml", "--te-align", "32",
            "--batch-size", "4", "--grad-accum", "2", "--steps", "10",
            "--log-every", "1", "--eval-every", "4", "--eval-batches", "2", "--save-every", "10",
        ])
        assert "resumed from step 6" in r2.stdout, "resume banner missing"
        with open(os.path.join(smoke_out, "log.jsonl")) as f:
            logs = [json.loads(line) for line in f]
        steps_logged = [row["step"] for row in logs if "loss" in row]
        assert steps_logged == list(range(1, 11)), f"Unexpected smoke steps: {steps_logged}"
        report["smoke_ddp"] = {
            "passed": True,
            "steps_logged": steps_logged,
            "initial_loss": logs[0]["loss"],
            "final_loss": next(r["loss"] for r in reversed(logs) if "loss" in r),
            "final_val": next(r["val_loss"] for r in reversed(logs) if "val_loss" in r),
            "tok_per_s": [r["tok_per_s"] for r in logs if "tok_per_s" in r],
        }
    finally:
        shutil.rmtree(smoke_out, ignore_errors=True)

    # 2. 5b 2-GPU NCCL DDP with synthetic uint32 200k-vocab shards
    banner("DDP-2B: 5B 2-GPU NCCL DDP (bs=4, T=2048, grad_accum=2)")
    synth_dir = "/tmp/synth-5b-data"
    out_5b = "/tmp/ddp-5b"
    shutil.rmtree(synth_dir, ignore_errors=True)
    shutil.rmtree(out_5b, ignore_errors=True)
    try:
        os.makedirs(synth_dir, exist_ok=True)
        rng = np.random.default_rng(0)
        rng.integers(0, 200058, size=2_000_000, dtype=np.uint32).tofile(os.path.join(synth_dir, "train_00000.bin"))
        rng.integers(0, 200058, size=65_536, dtype=np.uint32).tofile(os.path.join(synth_dir, "val_00000.bin"))
        with open(os.path.join(synth_dir, "meta.json"), "w") as f:
            json.dump({"vocab_size": 200058, "dtype": "uint32"}, f)

        free_gb = shutil.disk_usage("/tmp").free / 1e9
        can_save_5b = free_gb >= 75.0
        can_resume_5b = free_gb >= 110.0 and (mem_avail_gb is None or mem_avail_gb >= 160.0)
        if not can_save_5b:
            raise RuntimeError(f"Insufficient /tmp disk ({free_gb:.1f} GB) for train.py final 5B checkpoint (~60 GB)")

        run_cmd([
            "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
            "--preset", "5b", "--data", synth_dir, "--out", out_5b,
            "--backend", "te", "--te-recipe", "tml", "--te-align", "32",
            "--block-size", "2048", "--batch-size", "4", "--grad-accum", "2",
            "--steps", "4", "--log-every", "1", "--eval-every", "4", "--eval-batches", "2",
            "--save-every", "4",
        ])
        resumed_5b = False
        if can_resume_5b:
            r_5b_2 = run_cmd([
                "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
                "--preset", "5b", "--data", synth_dir, "--out", out_5b, "--resume",
                "--backend", "te", "--te-recipe", "tml", "--te-align", "32",
                "--block-size", "2048", "--batch-size", "4", "--grad-accum", "2",
                "--steps", "6", "--log-every", "1", "--eval-every", "6", "--eval-batches", "2",
                "--save-every", "6",
            ])
            assert "resumed from step 4" in r_5b_2.stdout
            resumed_5b = True

        with open(os.path.join(out_5b, "log.jsonl")) as f:
            logs_5b = [json.loads(line) for line in f]
        report["fiveb_ddp"] = {
            "passed": True,
            "resumed": resumed_5b,
            "steps_logged": [r["step"] for r in logs_5b if "loss" in r],
            "tok_per_s_total": [r["tok_per_s"] for r in logs_5b if "tok_per_s" in r],
            "logs": logs_5b,
        }
    finally:
        shutil.rmtree(out_5b, ignore_errors=True)
        shutil.rmtree(synth_dir, ignore_errors=True)

    report["gpu_seconds"] = round(time.time() - t_start, 1)
    return report


@app.local_entrypoint()
def main(skip_perf: bool = False, skip_ddp: bool = False):
    out_path = os.path.join(ROOT, "runs", "b200-ddp-perf-report.json")
    combined = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            combined = json.load(f)
    if not skip_perf:
        combined["perf_1gpu"] = verify_perf_1gpu.remote()
    if not skip_ddp:
        combined["ddp_2gpu"] = verify_ddp_2gpu.remote()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved DDP + perf report to {out_path}")
