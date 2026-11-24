"""Verify Part 2 Triton RPB+Sink attention, grouped backend, full 5B speed, and 2-GPU NCCL DDP on B200.

Usage:
    modal run modal_verify_part2.py --skip-ddp
    modal run modal_verify_part2.py --skip-perf
    modal run modal_verify_part2.py
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
APP_NAME = "loopfp4-part2-verify"

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


def _time_cuda_events(fn, warmup: int = 2, iters: int = 5):
    import torch

    t_cold0 = time.time()
    for i in range(warmup):
        fn()
        if i == 0:
            torch.cuda.synchronize()
            first_s = time.time() - t_cold0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    t0 = time.time()
    start_ev.record()
    for _ in range(iters):
        fn()
    end_ev.record()
    torch.cuda.synchronize()
    wall_ms = (time.time() - t0) / iters * 1000.0
    cuda_ms = start_ev.elapsed_time(end_ev) / iters
    gb = torch.cuda.max_memory_allocated() / 1e9
    return round(cuda_ms, 2), round(wall_ms, 2), round(gb, 2), round(first_s, 1)


@app.function(image=image, gpu="B200", timeout=2400)
def verify_part2_1gpu() -> dict:
    import gc
    from dataclasses import replace

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    from loopfp4 import PRESETS, build
    from loopfp4.inkling import (
        DecoderLayer,
        SparseMoeBlock,
        _flex_attn_compiled,
        _get_block_mask,
    )
    from loopfp4.triton_attn import (
        _LOG2E,
        _rpb_sink_bwd_dkdv_kernel,
        _rpb_sink_fwd_kernel,
        triton_rpb_sink_attn,
    )

    t_start = time.time()
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "triton_attn_exactness": {},
        "triton_tile_sweep": {},
        "triton_vs_flex_speed": {},
        "grouped_mm_edge_and_parity": {},
        "single_layer_5b": {},
        "full_5b": {},
    }

    # ------------------------------------------------------------------
    # 1. Triton RPB+Sink Attention Exactness (FP32 IEEE + BF16 + smoke/odd T)
    # ------------------------------------------------------------------
    banner("PART2-1: Triton Flash-RPB+Sink Exactness & Speed vs FlexAttention")
    torch.set_float32_matmul_precision("highest")
    if getattr(torch.backends.cuda.matmul, "fp32_precision", None) not in (None, "none"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"

    def _eager_rpb_sink_ref(q, k, v, rel, proj, sinks, window, rel_extent):
        B, Hq, T, D = q.shape
        Hkv = k.shape[1]
        G = Hq // Hkv
        pos = torch.arange(T, device=q.device)
        dist = pos[:, None] - pos[None, :]
        mask = torch.zeros(T, T, device=q.device).masked_fill(dist < 0, float("-inf"))
        if window > 0:
            mask = mask.masked_fill(dist >= window, float("-inf"))
        sc = torch.matmul(q.reshape(B, Hkv, G, T, D), k.unsqueeze(2).transpose(-1, -2)).reshape(B, Hq, T, T)
        rel_logits = rel.float() @ proj.float()
        index = dist.clamp(0, rel_extent - 1).expand(B, Hq, -1, -1)
        rpb = rel_logits.gather(-1, index).masked_fill((dist < 0) | (dist >= rel_extent), 0.0)
        sc = sc * (D**-0.5) + rpb + mask
        att = F.softmax(torch.cat([sc, sinks.unsqueeze(-1)], dim=-1).float(), dim=-1)[..., :-1].to(v.dtype)
        out = torch.matmul(att.reshape(B, Hkv, G, T, T), v.unsqueeze(2)).reshape(B, Hq, T, D)
        return out

    exact_cases = [
        ("swa_5b_T2048", 2, 12, 4, 2048, 128, 16, 1024, 1024),
        ("global_5b_T2048", 2, 12, 4, 2048, 128, 16, 0, 1024),
        ("smoke_D32_T256", 2, 4, 2, 256, 32, 8, 64, 64),
        ("nonmult_D32_T300", 2, 4, 2, 300, 32, 8, 64, 64),
    ]
    for cname, B, Hq, Hkv, T, D, d_rel, window, ext in exact_cases:
        for dt_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            lbl = f"{cname}_{dt_name}"
            torch.manual_seed(17)
            # Build tensors with [B, T, H, ...] layout then .transpose(1, 2) to exercise zero-copy strides
            q_src = torch.randn(B, T, Hq, D, device="cuda", dtype=dtype) * 0.5
            k_src = torch.randn(B, T, Hkv, D, device="cuda", dtype=dtype) * 0.5
            v_src = torch.randn(B, T, Hkv, D, device="cuda", dtype=dtype) * 0.5
            rel_src = torch.randn(B, T, Hq, d_rel, device="cuda", dtype=dtype) * 0.2
            proj0 = torch.randn(d_rel, ext, device="cuda", dtype=torch.float32) * 0.1
            s_src = torch.randn(B, T, Hq, device="cuda", dtype=torch.float32) * 0.5
            g_out = (torch.randn(B, T, Hq, D, device="cuda", dtype=dtype)).transpose(1, 2)

            ref_inputs = [
                q_src.transpose(1, 2).detach().clone().float().requires_grad_(True),
                k_src.transpose(1, 2).detach().clone().float().requires_grad_(True),
                v_src.transpose(1, 2).detach().clone().float().requires_grad_(True),
                rel_src.transpose(1, 2).detach().clone().float().requires_grad_(True),
                proj0.detach().clone().requires_grad_(True),
                s_src.transpose(1, 2).detach().clone().requires_grad_(True),
            ]
            out_ref = _eager_rpb_sink_ref(*ref_inputs, window, ext)
            (out_ref * g_out.float()).sum().backward()

            q_leaf = q_src.detach().clone().requires_grad_(True)
            k_leaf = k_src.detach().clone().requires_grad_(True)
            v_leaf = v_src.detach().clone().requires_grad_(True)
            rel_leaf = rel_src.detach().clone().requires_grad_(True)
            proj_leaf = proj0.detach().clone().requires_grad_(True)
            s_leaf = s_src.detach().clone().requires_grad_(True)
            tri_leaves = [q_leaf, k_leaf, v_leaf, rel_leaf, proj_leaf, s_leaf]
            out_tri = triton_rpb_sink_attn(
                q_leaf.transpose(1, 2), k_leaf.transpose(1, 2), v_leaf.transpose(1, 2),
                rel_leaf.transpose(1, 2), proj_leaf, s_leaf.transpose(1, 2), window, ext,
            )
            (out_tri * g_out).sum().backward()

            names = ["q", "k", "v", "rel", "proj", "sinks"]
            tri_grads = [
                q_leaf.grad.transpose(1, 2),
                k_leaf.grad.transpose(1, 2),
                v_leaf.grad.transpose(1, 2),
                rel_leaf.grad.transpose(1, 2),
                proj_leaf.grad,
                s_leaf.grad.transpose(1, 2),
            ]
            fwd_max = (out_ref.float() - out_tri.float()).abs().max().item()
            fwd_rel = (out_ref.float() - out_tri.float()).norm().item() / max(out_ref.float().norm().item(), 1e-6)
            grad_max = {
                n: (a.grad.float() - bg.float()).abs().max().item()
                for n, a, bg in zip(names, ref_inputs, tri_grads)
            }
            grad_rel = {
                n: (a.grad.float() - bg.float()).norm().item() / max(a.grad.float().norm().item(), 1e-6)
                for n, a, bg in zip(names, ref_inputs, tri_grads)
            }
            rec = {
                "fwd_max_diff": round(fwd_max, 7),
                "fwd_rel_l2": round(fwd_rel, 7),
                "grad_max_diff": {k: round(v, 7) for k, v in grad_max.items()},
                "grad_rel_l2": {k: round(v, 7) for k, v in grad_rel.items()},
            }
            report["triton_attn_exactness"][lbl] = rec
            print(f"  {lbl:24s}: fwd_rel={fwd_rel:.2e} grads_rel={rec['grad_rel_l2']}", flush=True)
            if dt_name == "fp32":
                assert fwd_rel < 1e-5 and max(grad_rel.values()) < 1e-5, (lbl, rec)
                assert fwd_max < 1e-4 and max(grad_max.values()) < 1e-4, (lbl, rec)
            else:
                assert fwd_rel < 1e-2 and max(grad_rel.values()) < 1.5e-2, (lbl, rec)
            del ref_inputs, tri_leaves, tri_grads, out_ref, out_tri, q_src, k_src, v_src, rel_src, proj0, s_src, g_out
            gc.collect(); torch.cuda.empty_cache()

    # Quick Triton tile sweep on SWA bs=16x2048 BF16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    B_sp, Hq_sp, Hkv_sp, T_sp, D_sp, drel_sp, ext_sp = 16, 12, 4, 2048, 128, 16, 1024
    try:
        import triton
        qs = (torch.randn(B_sp, T_sp, Hq_sp, D_sp, device="cuda", dtype=torch.bfloat16) * 0.5).transpose(1, 2)
        ks = (torch.randn(B_sp, T_sp, Hkv_sp, D_sp, device="cuda", dtype=torch.bfloat16) * 0.5).transpose(1, 2)
        vs = (torch.randn(B_sp, T_sp, Hkv_sp, D_sp, device="cuda", dtype=torch.bfloat16) * 0.5).transpose(1, 2)
        rls = torch.randn(B_sp, Hq_sp, T_sp, ext_sp, device="cuda", dtype=torch.bfloat16) * 0.1
        sks = torch.zeros(B_sp, T_sp, Hq_sp, device="cuda", dtype=torch.float32).transpose(1, 2)
        dos = (torch.randn(B_sp, T_sp, Hq_sp, D_sp, device="cuda", dtype=torch.bfloat16) * 0.5).transpose(1, 2)
        out_s = torch.empty((B_sp, Hq_sp, T_sp, D_sp), device="cuda", dtype=torch.bfloat16)
        lse_s = torch.full((B_sp, Hq_sp, T_sp), 8.0, device="cuda", dtype=torch.float32)
        dlt_s = torch.zeros((B_sp, Hq_sp, T_sp), device="cuda", dtype=torch.float32)
        dks = torch.empty((B_sp, Hkv_sp, T_sp, D_sp), device="cuda", dtype=torch.bfloat16)
        dvs = torch.empty((B_sp, Hkv_sp, T_sp, D_sp), device="cuda", dtype=torch.bfloat16)
        sm_s = D_sp**-0.5
        sm_sl2 = sm_s * _LOG2E
        for bm, bn, warps, stages in [(64, 64, 4, 2), (128, 64, 4, 2), (128, 64, 8, 2)]:
            def _f(bm=bm, bn=bn, w=warps, st=stages):
                _rpb_sink_fwd_kernel[(triton.cdiv(T_sp, bm), B_sp * Hq_sp)](
                    qs, ks, vs, rls, sks, out_s, lse_s,
                    *qs.stride(), *ks.stride(), *vs.stride(), *rls.stride(), *sks.stride(),
                    *out_s.stride(), *lse_s.stride(), T_sp, Hq_sp, sm_sl2,
                    rel_extent=ext_sp, window=1024, HAS_WINDOW=True, G=3, HEAD_DIM=D_sp,
                    BM=bm, BN=bn, IS_FP32=False, LOG2E=_LOG2E, num_warps=w, num_stages=st,
                )
            cms, _, _, _ = _time_cuda_events(_f, warmup=3, iters=5)
            report["triton_tile_sweep"][f"fwd_{bm}x{bn}_w{warps}_s{stages}"] = cms
            print(f"  tile sweep fwd ({bm}x{bn}, w={warps}, s={stages}): {cms:.2f} ms", flush=True)
        for bm, bn, warps, stages in [(64, 64, 4, 2), (64, 128, 8, 2)]:
            def _dk(bm=bm, bn=bn, w=warps, st=stages):
                _rpb_sink_bwd_dkdv_kernel[(triton.cdiv(T_sp, bn), B_sp * Hkv_sp)](
                    qs, ks, vs, rls, dos, lse_s, dlt_s, dks, dvs,
                    *qs.stride(), *ks.stride(), *vs.stride(), *rls.stride(), *dos.stride(),
                    *lse_s.stride(), *dlt_s.stride(), *dks.stride(), *dvs.stride(),
                    T_sp, Hkv_sp, sm_s, sm_sl2, rel_extent=ext_sp, window=1024, HAS_WINDOW=True,
                    G=3, HEAD_DIM=D_sp, BM=bm, BN=bn, IS_FP32=False, LOG2E=_LOG2E, num_warps=w, num_stages=st,
                )
            cms, _, _, _ = _time_cuda_events(_dk, warmup=3, iters=5)
            report["triton_tile_sweep"][f"dkdv_{bm}x{bn}_w{warps}_s{stages}"] = cms
            print(f"  tile sweep dkdv ({bm}x{bn}, w={warps}, s={stages}): {cms:.2f} ms", flush=True)
        del qs, ks, vs, rls, sks, dos, out_s, lse_s, dlt_s, dks, dvs
        gc.collect(); torch.cuda.empty_cache()
    except Exception as err:
        report["triton_tile_sweep"]["error"] = repr(err)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    for kind, window in [("swa", 1024), ("global", 0)]:
        q_s = torch.randn(B_sp, Hq_sp, T_sp, D_sp, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k_s = torch.randn(B_sp, Hkv_sp, T_sp, D_sp, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v_s = torch.randn(B_sp, Hkv_sp, T_sp, D_sp, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        rel_s = torch.randn(B_sp, Hq_sp, T_sp, drel_sp, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        proj_s = torch.randn(drel_sp, ext_sp, device="cuda", dtype=torch.float32, requires_grad=True)
        sink_s = torch.randn(B_sp, Hq_sp, T_sp, device="cuda", dtype=torch.float32, requires_grad=True)
        g_s = torch.randn(B_sp, Hq_sp, T_sp, D_sp, device="cuda", dtype=torch.bfloat16)

        def _zero():
            for t in (q_s, k_s, v_s, rel_s, proj_s, sink_s):
                t.grad = None

        def _run_tri_fwd():
            return triton_rpb_sink_attn(q_s, k_s, v_s, rel_s, proj_s, sink_s, window, ext_sp)

        c_trif, _, _, _ = _time_cuda_events(_run_tri_fwd, warmup=3, iters=5)

        # Triton eager & compiled (fwd+bwd)
        def _run_tri_eager():
            out = triton_rpb_sink_attn(q_s, k_s, v_s, rel_s, proj_s, sink_s, window, ext_sp)
            out.backward(g_s)
            _zero()

        c_tri, w_tri, gb_tri, _ = _time_cuda_events(_run_tri_eager, warmup=3, iters=5)

        @torch.compile
        def _tri_comp_fn(q, k, v, rel, proj, sinks):
            return triton_rpb_sink_attn(q, k, v, rel, proj, sinks, window, ext_sp)

        def _run_tri_comp():
            out = _tri_comp_fn(q_s, k_s, v_s, rel_s, proj_s, sink_s)
            out.backward(g_s)
            _zero()

        c_tric, w_tric, gb_tric, _ = _time_cuda_events(_run_tri_comp, warmup=3, iters=5)

        # FlexAttention fwd & fwd+bwd
        bm = _get_block_mask(T_sp, None if window == 0 else window, torch.device("cuda"))

        def _run_flex_fwd():
            rl = (rel_s.float() @ proj_s.float()).contiguous()
            return _flex_attn_compiled(q_s, k_s, v_s, rl, sink_s, bm, ext_sp)

        c_flxf, _, _, _ = _time_cuda_events(_run_flex_fwd, warmup=3, iters=5)

        def _run_flex():
            rl = (rel_s.float() @ proj_s.float()).contiguous()
            out = _flex_attn_compiled(q_s, k_s, v_s, rl, sink_s, bm, ext_sp)
            out.backward(g_s)
            _zero()

        c_flx, w_flx, gb_flx, _ = _time_cuda_events(_run_flex, warmup=3, iters=5)

        report["triton_vs_flex_speed"][kind] = {
            "triton_fwd_cuda_ms": c_trif,
            "triton_bwd_approx_ms": round(max(0.0, c_tri - c_trif), 2),
            "triton_eager_cuda_ms": c_tri,
            "triton_eager_peak_gb": gb_tri,
            "triton_compile_cuda_ms": c_tric,
            "triton_compile_peak_gb": gb_tric,
            "flex_fwd_cuda_ms": c_flxf,
            "flex_bwd_approx_ms": round(max(0.0, c_flx - c_flxf), 2),
            "flex_cuda_ms": c_flx,
            "flex_peak_gb": gb_flx,
            "speedup": round(c_flx / max(c_tric, 1e-3), 2),
        }
        print(
            f"  attn bs16x2048 {kind:6s}: Flex={c_flx:.2f}ms (fwd {c_flxf:.2f}ms, {gb_flx:.2f}GB) -> "
            f"Triton={c_tric:.2f}ms (fwd {c_trif:.2f}ms, {gb_tric:.2f}GB) [{c_flx/c_tric:.2f}x]",
            flush=True,
        )
        del q_s, k_s, v_s, rel_s, proj_s, sink_s, g_s
        gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Unpadded torch._grouped_mm Edge Cases & Parity vs sim
    # ------------------------------------------------------------------
    banner("PART2-2: Unpadded torch._grouped_mm Edge Cases, Parity & torchao Diag")
    cfg5 = PRESETS["5b"]

    # Edge case: E=16, half empty experts + odd/unaligned group counts (1, 3, 7, 17, 33)
    try:
        edge_counts = torch.tensor([0, 1, 0, 3, 0, 7, 0, 17, 0, 33, 0, 2, 0, 5, 0, 11], device="cuda", dtype=torch.int32)
        N_edge = int(edge_counts.sum().item())  # 79
        E_edge, H_edge, in_edge = 16, 128, 64
        mc_edge = replace(
            PRESETS["smoke"], hidden_size=H_edge, moe_intermediate_size=in_edge,
            n_routed_experts=E_edge, num_experts_per_tok=1, quant="none", backward="exact",
        )
        moe_grp = SparseMoeBlock(replace(mc_edge, backend="grouped"), mode="none").cuda().eval()
        moe_sim = SparseMoeBlock(replace(mc_edge, backend="sim"), mode="none").cuda().eval()
        moe_sim.load_state_dict(moe_grp.state_dict())

        # Construct token assignments matching edge_counts
        expert_ids = torch.cat([torch.full((int(c),), i, device="cuda", dtype=torch.long) for i, c in enumerate(edge_counts.tolist())])
        perm = torch.randperm(N_edge, device="cuda")
        expert_unsorted = expert_ids[perm]
        order = torch.argsort(expert_unsorted.to(torch.int32), stable=False)
        token = order  # top_k=1
        x_edge = torch.randn(N_edge, H_edge, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_edge2 = x_edge.detach().clone().requires_grad_(True)
        g_e = torch.randn(N_edge, H_edge, device="cuda", dtype=torch.bfloat16)

        y_g = moe_grp._experts_grouped(x_edge, token, edge_counts)
        (y_g * g_e.float()).sum().backward()
        # Eager reference using PyTorch linear per expert
        offs_list = edge_counts.tolist()
        parts = torch.split(x_edge2[token].float(), offs_list)
        w1_parts = moe_sim.experts.gate_up_proj.float()
        w2_parts = moe_sim.experts.down_proj.float()
        outs_ref = []
        for idx_e, part in enumerate(parts):
            if part.shape[0] == 0:
                outs_ref.append(part.new_zeros((0, H_edge)))
            else:
                g_, u_ = F.linear(part, w1_parts[idx_e]).chunk(2, dim=-1)
                act_ = (torch.tanh(g_ / moe_sim.gate_cap) * moe_sim.gate_cap) * torch.sigmoid(g_) * (torch.tanh(u_ / moe_sim.up_cap) * moe_sim.up_cap)
                outs_ref.append(F.linear(act_, w2_parts[idx_e]))
        y_ref_e = torch.cat(outs_ref, dim=0)
        (y_ref_e * g_e.float()).sum().backward()

        edge_fwd_rel = (y_g - y_ref_e).norm().item() / max(y_ref_e.norm().item(), 1e-6)
        edge_dx_rel = (x_edge.grad.float() - x_edge2.grad.float()).norm().item() / max(x_edge2.grad.float().norm().item(), 1e-6)
        edge_dw1_rel = (moe_grp.experts.gate_up_proj.grad.float() - moe_sim.experts.gate_up_proj.grad.float()).norm().item() / max(moe_sim.experts.gate_up_proj.grad.float().norm().item(), 1e-6)
        report["grouped_mm_edge_and_parity"]["empty_and_odd_counts"] = {
            "counts": offs_list,
            "fwd_rel_l2": round(edge_fwd_rel, 6),
            "dx_rel_l2": round(edge_dx_rel, 6),
            "dw1_rel_l2": round(edge_dw1_rel, 6),
            "empty_expert_grad_absmax": moe_grp.experts.gate_up_proj.grad[0].abs().max().item(),
        }
        print(f"  grouped empty+odd counts parity: fwd={edge_fwd_rel:.2e} dx={edge_dx_rel:.2e} dw1={edge_dw1_rel:.2e}", flush=True)
        assert edge_fwd_rel < 2e-2 and edge_dx_rel < 2e-2 and edge_dw1_rel < 2e-2
        del moe_grp, moe_sim, x_edge, x_edge2
    except Exception as err:
        report["grouped_mm_edge_and_parity"]["empty_and_odd_counts_error"] = repr(err)
        print(f"  grouped empty+odd ERROR: {err!r}", flush=True)

    # Full SparseMoeBlock parity: grouped vs sim (quant="none" and quant="nvfp4")
    for qm in ("none", "nvfp4"):
        try:
            mc_p = replace(PRESETS["smoke"], quant=qm, backward="exact", n_routed_experts=32, hidden_size=128, moe_intermediate_size=64)
            m_sim = SparseMoeBlock(replace(mc_p, backend="sim"), mode=qm).cuda().train()
            m_grp = SparseMoeBlock(replace(mc_p, backend="grouped"), mode=qm).cuda().train()
            m_grp.load_state_dict(m_sim.state_dict())

            torch.manual_seed(7)
            xs = torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            xg = xs.detach().clone().requires_grad_(True)
            gs = torch.randn_like(xs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                ys = m_sim(xs)
                yg = m_grp(xg)
            (ys * gs).sum().backward()
            (yg * gs).sum().backward()

            f_rel = (ys.float() - yg.float()).norm().item() / max(ys.float().norm().item(), 1e-6)
            dx_rel = (xs.grad.float() - xg.grad.float()).norm().item() / max(xs.grad.float().norm().item(), 1e-6)
            dw_rel = (m_sim.experts.gate_up_proj.grad.float() - m_grp.experts.gate_up_proj.grad.float()).norm().item() / max(m_sim.experts.gate_up_proj.grad.float().norm().item(), 1e-6)
            report["grouped_mm_edge_and_parity"][f"full_moe_parity_{qm}"] = {
                "fwd_rel_l2": round(f_rel, 6),
                "dx_rel_l2": round(dx_rel, 6),
                "dw1_rel_l2": round(dw_rel, 6),
            }
            print(f"  full SparseMoeBlock parity ({qm:5s}): fwd={f_rel:.2e} dx={dx_rel:.2e} dw1={dw_rel:.2e}", flush=True)
            del m_sim, m_grp, xs, xg, gs
        except Exception as err:
            report["grouped_mm_edge_and_parity"][f"full_moe_parity_{qm}_error"] = repr(err)
            print(f"  full SparseMoeBlock parity ({qm}) ERROR: {err!r}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Single DecoderLayer 5B (SWA li=0 & Global li=5, uncheckpointed)
    # ------------------------------------------------------------------
    banner("PART2-3: Single Compiled DecoderLayer 5B (SWA & Global, bs=16x2048)")
    B16, T2 = 16, 2048
    pos_c = torch.arange(T2, device="cuda")

    def _prof_layer(fn, label_p: str):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            fn()
        torch.cuda.synchronize()
        avgs = prof.key_averages()
        top_cu = [
            {
                "key": e.key[:120],
                "count": e.count,
                "self_cuda_ms": round(e.self_device_time_total / 1000.0, 2),
                "cuda_ms": round(e.device_time_total / 1000.0, 2),
            }
            for e in sorted(avgs, key=lambda x: x.self_device_time_total, reverse=True)[:15]
        ]
        print(f"    [prof {label_p}] top CUDA ops:", flush=True)
        for row in top_cu[:10]:
            print(f"      {row['self_cuda_ms']:6.2f}ms ({row['count']:3d}x) {row['key']}", flush=True)
        return top_cu

    for qmode in ("none", "nvfp4"):
        for li, kind, n_blk in [(0, "swa", 0), (5, "global", 1)]:
            lbl = f"{kind}_{qmode}_compile"
            try:
                torch._dynamo.reset()
                mc_l = replace(cfg5, backend="grouped", quant=qmode, grad_checkpoint=False, use_flex_attn=True)
                layer = DecoderLayer(mc_l, li).cuda()
                layer.compile()
                h_l = torch.randn(B16, T2, mc_l.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                blk_st = (torch.randn(B16, T2, n_blk, mc_l.hidden_size, device="cuda", dtype=torch.bfloat16)
                          if n_blk else h_l.new_zeros(B16, T2, 0, mc_l.hidden_size))
                g_l = torch.randn(B16, T2, mc_l.hidden_size, device="cuda", dtype=torch.bfloat16)

                def _step_l():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = layer(h_l, blk_st, None, None, None, pos_c)
                    out.backward(g_l)
                    layer.zero_grad(set_to_none=True)
                    h_l.grad = None

                cms, wms, gb, _ = _time_cuda_events(_step_l, warmup=2, iters=5)
                rec_l = {"cuda_ms": cms, "wall_ms": wms, "peak_gb": gb}
                if kind == "swa":
                    rec_l["top_cuda_ops"] = _prof_layer(_step_l, lbl)
                report["single_layer_5b"][lbl] = rec_l
                print(f"  single layer {lbl:20s}: {cms:6.2f} ms (wall {wms:6.2f} ms, peak {gb:5.2f} GB)", flush=True)
                del layer, h_l, blk_st, g_l
            except Exception as err:
                report["single_layer_5b"][lbl] = {"error": repr(err)}
                print(f"  single layer {lbl}: ERROR {err!r}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Full 5B Training Benchmarks (FP32 master weights, fused AdamW, clip, balance_step)
    # ------------------------------------------------------------------
    banner("PART2-4: Full 5B Benchmarks (grouped backend, layer.compile())")
    runs_spec = [
        ("bs16_grouped_bf16_gc_sac", 16, "none", "sac"),
        ("bs16_grouped_nvfp4_gc_sac", 16, "nvfp4", "sac"),
        ("bs16_grouped_bf16_gc_sac_all", 16, "none", "sac_all"),
        ("bs16_grouped_nvfp4_gc_sac_all", 16, "nvfp4", "sac_all"),
        ("bs24_grouped_bf16_gc_sac", 24, "none", "sac"),
        ("bs24_grouped_nvfp4_gc_sac", 24, "nvfp4", "sac"),
    ]
    best_nvfp4_tps = 0
    best_bf16_tps = 0
    peak_bs24_sac = {}

    def _bench_full_5b(label: str, bs: int, qmode: str, gcmode: str):
        nonlocal best_nvfp4_tps, best_bf16_tps
        torch._dynamo.reset()
        cfg = replace(
            cfg5, backend="grouped", quant=qmode,
            grad_checkpoint=True, grad_checkpoint_mode=gcmode, use_flex_attn=True,
        )
        model = build(cfg).cuda()
        for layer in model.layers:
            layer.compile()
        skip = ("scale", "query", "norm", "static_weight", "conv1d")
        named = list(model.named_parameters())
        decay = [p for n, p in named if p.dim() >= 2 and not any(k in n for k in skip)]
        no_decay = [p for n, p in named if not (p.dim() >= 2 and not any(k in n for k in skip))]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=2e-4, betas=(0.9, 0.95), fused=True,
        )
        xb = torch.randint(0, cfg.unpadded_vocab_size, (bs, 2048), device="cuda")
        yb = torch.randint(0, cfg.unpadded_vocab_size, (bs, 2048), device="cuda")

        def _train_step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xb, yb, return_logits=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.balance_step(1e-3)

        cms, wms, peak_gb, cold_s = _time_cuda_events(_train_step, warmup=2, iters=5)
        tps = round(bs * 2048 / (wms / 1000.0))
        cost_100b = round(100e9 / tps / 3600.0 * 4.0)
        if qmode == "nvfp4":
            best_nvfp4_tps = max(best_nvfp4_tps, tps)
        else:
            best_bf16_tps = max(best_bf16_tps, tps)
        rec = {
            "tok_per_s": tps,
            "cuda_ms": cms,
            "wall_ms": wms,
            "peak_gb": peak_gb,
            "cold_compile_step0_s": cold_s,
            "cost_100b_at_4usd_hr": cost_100b,
        }
        if bs == 16:
            e0, e1, e2, e3, e4, e5 = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            torch.cuda.synchronize()
            e0.record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xb, yb, return_logits=False)
            e1.record()
            loss.backward()
            e2.record()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            e3.record()
            opt.step()
            opt.zero_grad(set_to_none=True)
            e4.record()
            model.balance_step(1e-3)
            e5.record()
            torch.cuda.synchronize()
            rec["phase_breakdown"] = {
                "fwd_ms": round(e0.elapsed_time(e1), 2),
                "bwd_ms": round(e1.elapsed_time(e2), 2),
                "clip_ms": round(e2.elapsed_time(e3), 2),
                "opt_ms": round(e3.elapsed_time(e4), 2),
                "balance_ms": round(e4.elapsed_time(e5), 2),
            }
        report["full_5b"][label] = rec
        print(
            f"  full 5b {label:28s}: {tps:,} tok/s ({wms:.1f} ms/step, peak {peak_gb:.2f} GB, cold {cold_s:.1f}s) -> ${cost_100b:,}/100B",
            flush=True,
        )
        if "phase_breakdown" in rec:
            print(f"    phases: {rec['phase_breakdown']}", flush=True)
        del model, opt, xb, yb
        return rec

    for label, bs, qmode, gcmode in runs_spec:
        try:
            rec = _bench_full_5b(label, bs, qmode, gcmode)
            if bs == 24 and gcmode == "sac":
                peak_bs24_sac[qmode] = rec["peak_gb"]
        except Exception as err:
            report["full_5b"][label] = {"error": repr(err)}
            print(f"  full 5b {label:28s}: ERROR {err!r}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    for qmode, tag in [("none", "bf16"), ("nvfp4", "nvfp4")]:
        if peak_bs24_sac.get(qmode, 999.0) < 148.0:
            lbl = f"bs32_grouped_{tag}_gc_sac"
            try:
                _bench_full_5b(lbl, 32, qmode, "sac")
            except Exception as err:
                report["full_5b"][lbl] = {"error": repr(err)}
                print(f"  full 5b {lbl:28s}: ERROR {err!r}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    if best_nvfp4_tps > 0:
        res_plan = run_cmd([
            sys.executable, "plan.py", "--preset", "5b", "--gpus", "8", "--tok-per-s-per-gpu", str(int(best_nvfp4_tps)),
        ])
        report["plan_5b_nvfp4"] = res_plan.stdout
    if best_bf16_tps > 0:
        res_plan_b = run_cmd([
            sys.executable, "plan.py", "--preset", "5b", "--gpus", "8", "--tok-per-s-per-gpu", str(int(best_bf16_tps)),
        ])
        report["plan_5b_bf16"] = res_plan_b.stdout

    report["gpu_seconds"] = round(time.time() - t_start, 1)
    return report


@app.function(image=image, gpu="B200:2", timeout=1800)
def verify_part2_ddp_2gpu() -> dict:
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
    banner(f"DDP-PART2: {n_gpus}x {report['gpus'][0]} (tmp_free={report['tmp_free_gb']} GB, ram_avail={mem_avail_gb} GB)")

    synth_dir = "/tmp/part2-5b-data"
    out_5b = "/tmp/ddp-part2-5b"
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
            raise RuntimeError(f"Insufficient /tmp disk ({free_gb:.1f} GB) for train.py final 5B checkpoint")

        run_cmd([
            "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
            "--preset", "5b", "--data", synth_dir, "--out", out_5b,
            "--backend", "grouped", "--quant", "nvfp4",
            "--grad-checkpoint", "--grad-checkpoint-mode", "sac", "--compile",
            "--block-size", "2048", "--batch-size", "24", "--grad-accum", "1",
            "--steps", "4", "--log-every", "1", "--eval-every", "4", "--eval-batches", "2",
            "--save-every", "4",
        ])
        resumed_5b = False
        if can_resume_5b:
            r2 = run_cmd([
                "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
                "--preset", "5b", "--data", synth_dir, "--out", out_5b, "--resume",
                "--backend", "grouped", "--quant", "nvfp4",
                "--grad-checkpoint", "--grad-checkpoint-mode", "sac", "--compile",
                "--block-size", "2048", "--batch-size", "24", "--grad-accum", "1",
                "--steps", "8", "--log-every", "1", "--eval-every", "8", "--eval-batches", "2",
                "--save-every", "8",
            ])
            assert "resumed from step 4" in r2.stdout
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
    out_path = os.path.join(ROOT, "runs", "b200-part2-verify-report.json")
    combined = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            combined = json.load(f)
    if not skip_perf:
        combined["perf_1gpu"] = verify_part2_1gpu.remote()
    if not skip_ddp:
        combined["ddp_2gpu"] = verify_part2_ddp_2gpu.remote()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved Part 2 B200 verification report to {out_path}")
