"""Fused Triton Flash-Attention with content-dependent relative position bias (RPB) and sinks.

Replaces PyTorch FlexAttention's score_mod backward (which scatters gradients into rel_logits via
tl.atomic_add) with a two-pass FlashAttention-2 backward:
  1. K/V-parallel pass computing (dK, dV) with zero atomics.
  2. Q-parallel pass computing dQ and writing d_rel_logits[b, h, q, q - k] with coalesced plain
     stores (zero atomics, since d = q - k is 1-to-1 per query row).

Forward saves only the small pre-projection RPB features `rel` ([B, H_q, T, d_rel]) instead of the
expanded `rel_logits` ([B, H_q, T, rel_extent]), cutting per-layer saved attention memory by >100x.
"""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_ATTN_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_ATTN_AVAILABLE = False

_LOG2E = 1.4426950408889634


def _tile_cfg(is_fp32: bool, head_dim: int):
    if not is_fp32 and head_dim >= 64:
        return {
            "fwd": (64, 64, 4, 2),
            "dkdv": (64, 64, 4, 2),
            "dq": (128, 64, 8, 2),
        }
    return {
        "fwd": (64, 64, 4, 2),
        "dkdv": (64, 64, 4, 2),
        "dq": (64, 64, 4, 2),
    }

if TRITON_ATTN_AVAILABLE:

    @triton.jit
    def _rpb_sink_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, REL_ptr, SINK_ptr, Out_ptr, LSE_ptr,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_rb, stride_rh, stride_rt, stride_re,
        stride_sb, stride_sh, stride_st,
        stride_ob, stride_oh, stride_ot, stride_od,
        stride_lb, stride_lh, stride_lt,
        T, H_q, sm_scale_log2,
        rel_extent: tl.constexpr,
        window: tl.constexpr,
        HAS_WINDOW: tl.constexpr,
        G: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        IS_FP32: tl.constexpr,
        LOG2E: tl.constexpr,
    ):
        pid_m = (tl.num_programs(0) - 1) - tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H_q
        h_q = pid_bh % H_q
        h_kv = h_q // G

        q_start = pid_m * BM
        q_offs = q_start + tl.arange(0, BM)
        k_sub = tl.arange(0, BN)
        d_offs = tl.arange(0, HEAD_DIM)
        q_mask = q_offs < T

        q_base = Q_ptr + b * stride_qb + h_q * stride_qh
        q_blk = tl.load(
            q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=q_mask[:, None],
            other=0.0,
        )

        sink_base = SINK_ptr + b * stride_sb + h_q * stride_sh
        m_i = tl.load(sink_base + q_offs * stride_st, mask=q_mask, other=0.0).to(tl.float32) * LOG2E
        l_i = tl.full([BM], 1.0, dtype=tl.float32)
        acc = tl.zeros([BM, HEAD_DIM], dtype=tl.float32)

        k_base = K_ptr + b * stride_kb + h_kv * stride_kh
        v_base = V_ptr + b * stride_vb + h_kv * stride_vh
        r_base = REL_ptr + b * stride_rb + h_q * stride_rh

        if HAS_WINDOW:
            k_lo_raw = tl.maximum(0, q_start - window + 1)
            k_lo = (k_lo_raw // BN) * BN
        else:
            k_lo = 0
        k_hi = tl.minimum(T, q_start + BM)

        for k_start in range(k_lo, k_hi, BN):
            k_offs = k_start + k_sub
            k_mask = k_offs < T
            k_blk = tl.load(
                k_base + k_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                mask=k_mask[:, None],
                other=0.0,
            )
            v_blk = tl.load(
                v_base + k_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
                mask=k_mask[:, None],
                other=0.0,
            )

            if IS_FP32:
                qk = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * sm_scale_log2
            else:
                qk = tl.dot(q_blk, tl.trans(k_blk)) * sm_scale_log2

            dist = q_offs[:, None] - k_offs[None, :]
            if q_start - (k_start + BN - 1) < rel_extent:
                rpb_mask = q_mask[:, None] & k_mask[None, :] & (dist >= 0) & (dist < rel_extent)
                safe_q = tl.where(rpb_mask, q_offs[:, None], 0)
                safe_d = tl.where(rpb_mask, dist, 0)
                bias = tl.load(
                    r_base + safe_q * stride_rt + safe_d * stride_re,
                    mask=rpb_mask,
                    other=0.0,
                ).to(tl.float32)
                qk = qk + bias * LOG2E

            if HAS_WINDOW:
                valid = q_mask[:, None] & k_mask[None, :] & (dist >= 0) & (dist < window)
            else:
                valid = q_mask[:, None] & k_mask[None, :] & (dist >= 0)
            qk = tl.where(valid, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if IS_FP32:
                pv = tl.dot(p.to(v_blk.dtype), v_blk, input_precision="ieee")
            else:
                pv = tl.dot(p.to(v_blk.dtype), v_blk)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        out_base = Out_ptr + b * stride_ob + h_q * stride_oh
        out_val = (acc / l_i[:, None]).to(Out_ptr.dtype.element_ty)
        tl.store(
            out_base + q_offs[:, None] * stride_ot + d_offs[None, :] * stride_od,
            out_val,
            mask=q_mask[:, None],
        )
        lse_base = LSE_ptr + b * stride_lb + h_q * stride_lh
        tl.store(lse_base + q_offs * stride_lt, m_i + tl.math.log2(l_i), mask=q_mask)

    @triton.jit
    def _rpb_sink_bwd_dkdv_kernel(
        Q_ptr, K_ptr, V_ptr, REL_ptr, DO_ptr, LSE_ptr, DELTA_ptr, DK_ptr, DV_ptr,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_rb, stride_rh, stride_rt, stride_re,
        stride_dob, stride_doh, stride_dot, stride_dod,
        stride_lb, stride_lh, stride_lt,
        stride_delb, stride_delh, stride_delt,
        stride_dkb, stride_dkh, stride_dkt, stride_dkd,
        stride_dvb, stride_dvh, stride_dvt, stride_dvd,
        T, H_kv, sm_scale, sm_scale_log2,
        rel_extent: tl.constexpr,
        window: tl.constexpr,
        HAS_WINDOW: tl.constexpr,
        G: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        IS_FP32: tl.constexpr,
        LOG2E: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bhkv = tl.program_id(1)
        b = pid_bhkv // H_kv
        h_kv = pid_bhkv % H_kv

        k_start = pid_n * BN
        k_offs = k_start + tl.arange(0, BN)
        q_sub = tl.arange(0, BM)
        d_offs = tl.arange(0, HEAD_DIM)
        k_mask = k_offs < T

        k_base = K_ptr + b * stride_kb + h_kv * stride_kh
        v_base = V_ptr + b * stride_vb + h_kv * stride_vh
        k_blk = tl.load(
            k_base + k_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
            mask=k_mask[:, None],
            other=0.0,
        )
        v_blk = tl.load(
            v_base + k_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
            mask=k_mask[:, None],
            other=0.0,
        )

        dk_acc = tl.zeros([BN, HEAD_DIM], dtype=tl.float32)
        dv_acc = tl.zeros([BN, HEAD_DIM], dtype=tl.float32)

        q_lo = (k_start // BM) * BM
        if HAS_WINDOW:
            q_hi = tl.minimum(T, k_start + BN + window - 1)
        else:
            q_hi = T

        for g in range(G):
            h_q = h_kv * G + g
            q_base = Q_ptr + b * stride_qb + h_q * stride_qh
            do_base = DO_ptr + b * stride_dob + h_q * stride_doh
            r_base = REL_ptr + b * stride_rb + h_q * stride_rh
            lse_base = LSE_ptr + b * stride_lb + h_q * stride_lh
            del_base = DELTA_ptr + b * stride_delb + h_q * stride_delh

            for q_start in range(q_lo, q_hi, BM):
                q_offs = q_start + q_sub
                q_mask = q_offs < T
                q_blk = tl.load(
                    q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                    mask=q_mask[:, None],
                    other=0.0,
                )
                do_blk = tl.load(
                    do_base + q_offs[:, None] * stride_dot + d_offs[None, :] * stride_dod,
                    mask=q_mask[:, None],
                    other=0.0,
                )
                lse_blk = tl.load(lse_base + q_offs * stride_lt, mask=q_mask, other=0.0)
                delta_blk = tl.load(del_base + q_offs * stride_delt, mask=q_mask, other=0.0)

                if IS_FP32:
                    qk = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * sm_scale_log2
                else:
                    qk = tl.dot(q_blk, tl.trans(k_blk)) * sm_scale_log2

                dist = q_offs[:, None] - k_offs[None, :]
                if q_start - (k_start + BN - 1) < rel_extent:
                    rpb_mask = q_mask[:, None] & k_mask[None, :] & (dist >= 0) & (dist < rel_extent)
                    safe_q = tl.where(rpb_mask, q_offs[:, None], 0)
                    safe_d = tl.where(rpb_mask, dist, 0)
                    bias = tl.load(
                        r_base + safe_q * stride_rt + safe_d * stride_re,
                        mask=rpb_mask,
                        other=0.0,
                    ).to(tl.float32)
                    qk = qk + bias * LOG2E

                if HAS_WINDOW:
                    valid = q_mask[:, None] & k_mask[None, :] & (dist >= 0) & (dist < window)
                else:
                    valid = q_mask[:, None] & k_mask[None, :] & (dist >= 0)

                p = tl.where(valid, tl.math.exp2(qk - lse_blk[:, None]), 0.0)
                if IS_FP32:
                    dv_acc += tl.dot(tl.trans(p.to(do_blk.dtype)), do_blk, input_precision="ieee")
                    dp = tl.dot(do_blk, tl.trans(v_blk), input_precision="ieee")
                    ds = p * (dp - delta_blk[:, None])
                    dk_acc += tl.dot(tl.trans(ds.to(q_blk.dtype)), q_blk, input_precision="ieee")
                else:
                    dv_acc += tl.dot(tl.trans(p.to(do_blk.dtype)), do_blk)
                    dp = tl.dot(do_blk, tl.trans(v_blk))
                    ds = p * (dp - delta_blk[:, None])
                    dk_acc += tl.dot(tl.trans(ds.to(q_blk.dtype)), q_blk)

        dk_base = DK_ptr + b * stride_dkb + h_kv * stride_dkh
        dv_base = DV_ptr + b * stride_dvb + h_kv * stride_dvh
        tl.store(
            dk_base + k_offs[:, None] * stride_dkt + d_offs[None, :] * stride_dkd,
            (dk_acc * sm_scale).to(DK_ptr.dtype.element_ty),
            mask=k_mask[:, None],
        )
        tl.store(
            dv_base + k_offs[:, None] * stride_dvt + d_offs[None, :] * stride_dvd,
            dv_acc.to(DV_ptr.dtype.element_ty),
            mask=k_mask[:, None],
        )

    @triton.jit
    def _rpb_sink_bwd_dq_drpb_kernel(
        Q_ptr, K_ptr, V_ptr, REL_ptr, DO_ptr, LSE_ptr, DELTA_ptr, DQ_ptr, DREL_ptr,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_rb, stride_rh, stride_rt, stride_re,
        stride_dob, stride_doh, stride_dot, stride_dod,
        stride_lb, stride_lh, stride_lt,
        stride_delb, stride_delh, stride_delt,
        stride_dqb, stride_dqh, stride_dqt, stride_dqd,
        stride_drb, stride_drh, stride_drt, stride_dre,
        T, H_q, sm_scale, sm_scale_log2,
        rel_extent: tl.constexpr,
        window: tl.constexpr,
        HAS_WINDOW: tl.constexpr,
        G: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        IS_FP32: tl.constexpr,
        LOG2E: tl.constexpr,
    ):
        pid_m = (tl.num_programs(0) - 1) - tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H_q
        h_q = pid_bh % H_q
        h_kv = h_q // G

        q_start = pid_m * BM
        q_offs = q_start + tl.arange(0, BM)
        k_sub = tl.arange(0, BN)
        d_offs = tl.arange(0, HEAD_DIM)
        q_mask = q_offs < T

        q_base = Q_ptr + b * stride_qb + h_q * stride_qh
        do_base = DO_ptr + b * stride_dob + h_q * stride_doh
        lse_base = LSE_ptr + b * stride_lb + h_q * stride_lh
        del_base = DELTA_ptr + b * stride_delb + h_q * stride_delh

        q_blk = tl.load(
            q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=q_mask[:, None],
            other=0.0,
        )
        do_blk = tl.load(
            do_base + q_offs[:, None] * stride_dot + d_offs[None, :] * stride_dod,
            mask=q_mask[:, None],
            other=0.0,
        )
        lse_blk = tl.load(lse_base + q_offs * stride_lt, mask=q_mask, other=0.0)
        delta_blk = tl.load(del_base + q_offs * stride_delt, mask=q_mask, other=0.0)

        dq_acc = tl.zeros([BM, HEAD_DIM], dtype=tl.float32)

        k_base = K_ptr + b * stride_kb + h_kv * stride_kh
        v_base = V_ptr + b * stride_vb + h_kv * stride_vh
        r_base = REL_ptr + b * stride_rb + h_q * stride_rh
        dr_base = DREL_ptr + b * stride_drb + h_q * stride_drh

        if HAS_WINDOW:
            k_lo_raw = tl.maximum(0, q_start - window + 1)
            k_lo = (k_lo_raw // BN) * BN
        else:
            k_lo = 0
        k_hi = tl.minimum(T, q_start + BM)

        for k_start in range(k_lo, k_hi, BN):
            k_offs = k_start + k_sub
            k_mask = k_offs < T
            k_blk = tl.load(
                k_base + k_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                mask=k_mask[:, None],
                other=0.0,
            )
            v_blk = tl.load(
                v_base + k_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
                mask=k_mask[:, None],
                other=0.0,
            )

            if IS_FP32:
                qk = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * sm_scale_log2
            else:
                qk = tl.dot(q_blk, tl.trans(k_blk)) * sm_scale_log2

            dist = q_offs[:, None] - k_offs[None, :]
            if q_start - (k_start + BN - 1) < rel_extent:
                rpb_mask = q_mask[:, None] & k_mask[None, :] & (dist >= 0) & (dist < rel_extent)
                safe_q = tl.where(rpb_mask, q_offs[:, None], 0)
                safe_d = tl.where(rpb_mask, dist, 0)
                bias = tl.load(
                    r_base + safe_q * stride_rt + safe_d * stride_re,
                    mask=rpb_mask,
                    other=0.0,
                ).to(tl.float32)
                qk = qk + bias * LOG2E

            if HAS_WINDOW:
                valid = q_mask[:, None] & k_mask[None, :] & (dist >= 0) & (dist < window)
            else:
                valid = q_mask[:, None] & k_mask[None, :] & (dist >= 0)

            p = tl.where(valid, tl.math.exp2(qk - lse_blk[:, None]), 0.0)
            if IS_FP32:
                dp = tl.dot(do_blk, tl.trans(v_blk), input_precision="ieee")
                ds = p * (dp - delta_blk[:, None])
                dq_acc += tl.dot(ds.to(k_blk.dtype), k_blk, input_precision="ieee")
            else:
                dp = tl.dot(do_blk, tl.trans(v_blk))
                ds = p * (dp - delta_blk[:, None])
                dq_acc += tl.dot(ds.to(k_blk.dtype), k_blk)

            if q_start - (k_start + BN - 1) < rel_extent:
                store_mask = valid & (dist >= 0) & (dist < rel_extent)
                safe_dq = tl.where(store_mask, q_offs[:, None], 0)
                safe_dd = tl.where(store_mask, dist, 0)
                tl.store(
                    dr_base + safe_dq * stride_drt + safe_dd * stride_dre,
                    ds.to(DREL_ptr.dtype.element_ty),
                    mask=store_mask,
                )

        dq_base = DQ_ptr + b * stride_dqb + h_q * stride_dqh
        tl.store(
            dq_base + q_offs[:, None] * stride_dqt + d_offs[None, :] * stride_dqd,
            (dq_acc * sm_scale).to(DQ_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )

    @torch.library.custom_op("loopfp4::rpb_sink_attn_fwd", mutates_args=())
    def _rpb_sink_attn_fwd_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel: torch.Tensor,
        proj: torch.Tensor,
        sinks: torch.Tensor,
        window: int,
        rel_extent: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H_q, T, D = q.shape
        H_kv = k.shape[1]
        G = H_q // H_kv
        is_fp32 = q.dtype == torch.float32
        rpb_dtype = torch.float32 if is_fp32 else q.dtype

        q_c = q
        k_c = k if k.dtype == q.dtype else k.to(q.dtype)
        v_c = v if v.dtype == q.dtype else v.to(q.dtype)
        sinks_c = sinks if sinks.dtype == torch.float32 else sinks.float()
        rel_rpb = rel if rel.dtype == rpb_dtype else rel.to(rpb_dtype)
        proj_rpb = proj if proj.dtype == rpb_dtype else proj.to(rpb_dtype)
        rel_logits = torch.matmul(rel_rpb, proj_rpb).contiguous()

        out = torch.empty((B, H_q, T, D), device=q.device, dtype=v.dtype)
        lse_log2 = torch.empty((B, H_q, T), device=q.device, dtype=torch.float32)

        sm_scale_log2 = (D**-0.5) * _LOG2E
        has_window = window > 0
        bm, bn, warps, stages = _tile_cfg(is_fp32, D)["fwd"]
        grid = (triton.cdiv(T, bm), B * H_q)

        _rpb_sink_fwd_kernel[grid](
            q_c, k_c, v_c, rel_logits, sinks_c, out, lse_log2,
            *q_c.stride(), *k_c.stride(), *v_c.stride(),
            *rel_logits.stride(), *sinks_c.stride(),
            *out.stride(), *lse_log2.stride(),
            T, H_q, sm_scale_log2,
            rel_extent=rel_extent,
            window=window if has_window else 1,
            HAS_WINDOW=has_window,
            G=G,
            HEAD_DIM=D,
            BM=bm,
            BN=bn,
            IS_FP32=is_fp32,
            LOG2E=_LOG2E,
            num_warps=warps,
            num_stages=stages,
        )
        return out, lse_log2

    @_rpb_sink_attn_fwd_op.register_fake
    def _rpb_sink_attn_fwd_fake(q, k, v, rel, proj, sinks, window, rel_extent):
        B, H_q, T, D = q.shape
        return (
            torch.empty((B, H_q, T, D), device=q.device, dtype=v.dtype),
            torch.empty((B, H_q, T), device=q.device, dtype=torch.float32),
        )

    @torch.library.custom_op("loopfp4::rpb_sink_attn_bwd", mutates_args=())
    def _rpb_sink_attn_bwd_op(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel: torch.Tensor,
        proj: torch.Tensor,
        sinks: torch.Tensor,
        out: torch.Tensor,
        lse_log2: torch.Tensor,
        window: int,
        rel_extent: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H_q, T, D = q.shape
        H_kv = k.shape[1]
        G = H_q // H_kv
        is_fp32 = q.dtype == torch.float32
        rpb_dtype = torch.float32 if is_fp32 else q.dtype

        q_c = q
        k_c = k if k.dtype == q.dtype else k.to(q.dtype)
        v_c = v if v.dtype == q.dtype else v.to(q.dtype)
        do_c = dout if dout.dtype == q.dtype else dout.to(q.dtype)
        out_c = out
        lse_c = lse_log2 if (lse_log2.dtype == torch.float32 and lse_log2.is_contiguous()) else lse_log2.float().contiguous()
        sinks_c = sinks if sinks.dtype == torch.float32 else sinks.float()

        rel_rpb = rel if rel.dtype == rpb_dtype else rel.to(rpb_dtype)
        proj_rpb = proj if proj.dtype == rpb_dtype else proj.to(rpb_dtype)
        rel_logits = torch.matmul(rel_rpb, proj_rpb).contiguous()

        delta = (do_c.float() * out_c.float()).sum(dim=-1).contiguous()
        d_sinks = (-torch.exp2(sinks_c * _LOG2E - lse_c) * delta).to(sinks.dtype).contiguous()

        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
        dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
        d_rel_logits = torch.zeros_like(rel_logits)

        sm_scale = D**-0.5
        sm_scale_log2 = sm_scale * _LOG2E
        has_window = window > 0
        tcfg = _tile_cfg(is_fp32, D)
        bm_k, bn_k, warps_k, stages_k = tcfg["dkdv"]
        bm_q, bn_q, warps_q, stages_q = tcfg["dq"]

        _rpb_sink_bwd_dkdv_kernel[(triton.cdiv(T, bn_k), B * H_kv)](
            q_c, k_c, v_c, rel_logits, do_c, lse_c, delta, dk, dv,
            *q_c.stride(), *k_c.stride(), *v_c.stride(),
            *rel_logits.stride(), *do_c.stride(), *lse_c.stride(),
            *delta.stride(), *dk.stride(), *dv.stride(),
            T, H_kv, sm_scale, sm_scale_log2,
            rel_extent=rel_extent,
            window=window if has_window else 1,
            HAS_WINDOW=has_window,
            G=G,
            HEAD_DIM=D,
            BM=bm_k,
            BN=bn_k,
            IS_FP32=is_fp32,
            LOG2E=_LOG2E,
            num_warps=warps_k,
            num_stages=stages_k,
        )

        _rpb_sink_bwd_dq_drpb_kernel[(triton.cdiv(T, bm_q), B * H_q)](
            q_c, k_c, v_c, rel_logits, do_c, lse_c, delta, dq, d_rel_logits,
            *q_c.stride(), *k_c.stride(), *v_c.stride(),
            *rel_logits.stride(), *do_c.stride(), *lse_c.stride(),
            *delta.stride(), *dq.stride(), *d_rel_logits.stride(),
            T, H_q, sm_scale, sm_scale_log2,
            rel_extent=rel_extent,
            window=window if has_window else 1,
            HAS_WINDOW=has_window,
            G=G,
            HEAD_DIM=D,
            BM=bm_q,
            BN=bn_q,
            IS_FP32=is_fp32,
            LOG2E=_LOG2E,
            num_warps=warps_q,
            num_stages=stages_q,
        )

        d_rel = torch.matmul(d_rel_logits, proj_rpb.T).to(rel.dtype)
        d_rel_dim = rel_rpb.shape[-1]
        d_proj = torch.matmul(
            rel_rpb.contiguous().reshape(-1, d_rel_dim).T,
            d_rel_logits.reshape(-1, rel_extent),
        ).to(proj.dtype)

        return (
            dq.contiguous(),
            dk.contiguous(),
            dv.contiguous(),
            d_rel.contiguous(),
            d_proj.contiguous(),
            d_sinks.contiguous(),
        )

    @_rpb_sink_attn_bwd_op.register_fake
    def _rpb_sink_attn_bwd_fake(dout, q, k, v, rel, proj, sinks, out, lse_log2, window, rel_extent):
        return (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
            torch.empty(v.shape, device=v.device, dtype=v.dtype),
            torch.empty(rel.shape, device=rel.device, dtype=rel.dtype),
            torch.empty(proj.shape, device=proj.device, dtype=proj.dtype),
            torch.empty(sinks.shape, device=sinks.device, dtype=sinks.dtype),
        )

    def _setup_context(ctx, inputs, output):
        q, k, v, rel, proj, sinks, window, rel_extent = inputs
        out, lse_log2 = output
        ctx.save_for_backward(q, k, v, rel, proj, sinks, out, lse_log2)
        ctx.window = window
        ctx.rel_extent = rel_extent

    def _backward(ctx, dout, _dlse):
        q, k, v, rel, proj, sinks, out, lse_log2 = ctx.saved_tensors
        dq, dk, dv, d_rel, d_proj, d_sinks = _rpb_sink_attn_bwd_op(
            dout, q, k, v, rel, proj, sinks, out, lse_log2, ctx.window, ctx.rel_extent
        )
        return dq, dk, dv, d_rel, d_proj, d_sinks, None, None

    torch.library.register_autograd("loopfp4::rpb_sink_attn_fwd", _backward, setup_context=_setup_context)


def triton_rpb_sink_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    proj: torch.Tensor,
    sinks: torch.Tensor,
    window: int,
    rel_extent: int,
) -> torch.Tensor:
    """Compute GQA attention with content-dependent RPB and attention sinks using Triton kernels."""
    if not TRITON_ATTN_AVAILABLE:
        raise RuntimeError("Triton is not available")
    out, _ = _rpb_sink_attn_fwd_op(q, k, v, rel, proj, sinks, int(window), int(rel_extent))
    return out
