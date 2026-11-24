# Cutting a 100B-Token 5B MoE Run from \$100k+ Down to \$4.8k–\$5.1k on NVIDIA B200s (And the Path to \$2k–\$4k)
### *Part 1 & Part 2: A Verified 16.4x End-to-End Speedup (1,412 → 23,175 tok/s/B200) and 41.5k tok/s on 2x B200 DDP*

> **Engineering honesty upfront:** Every throughput, memory, convergence, and parity number in this writeup was measured directly on real NVIDIA B200 GPUs (no synthetic extrapolations or mocked timings).
> - **Starting point:** Our correctness-verified `5b` Inkling looped MoE run (`5.403B` stored / `0.873B` active per pass / `48` layer-applications per token / `86.4 GB` FP32 master+grad+AdamW state) launched at **`1,412 tok/s/B200`** (`145.52 GB` peak VRAM → **\$78,691** at \$4/GPU-hr, crossing **\$100k+** at small batches or \$5–\$6/hr cloud rates).
> - **Part 1 (`3,756 tok/s/B200`, \$29,582, 2.66x):** Tightened TE's `m_splits` padding alignment (`128 → 32`), replaced the `36 GB` FP32 `200k`-vocab logit materialization with checkpointed chunked cross-entropy (`0.0` loss diff, `-43.6 GB` VRAM), and derived exact `FlexAttention` online-softmax sink rescaling (`1e-6` IEEE FP32 parity) — enabling `bs=16x2048` and profiling the exact culprit holding utilization at `1.51%` MFU: TE `2.20.2`'s Python loop over 128 experts (`484,457` CUDA kernel launches and `312` `cudaStreamSynchronize` calls per step).
> - **Part 2 (`21,719–23,175 tok/s/B200`, \$4,794–\$5,116, 15.4x–16.4x):** Replacing TE's loop with zero-sync unpadded `torch._grouped_mm` (`62.18 ms → 4.90 ms` MoE), writing a custom zero-atomic **Triton Flash-RPB+Sink** kernel (saving `12.6 MB` pre-projection features instead of `1.61 GB` `rel_logits`), eliminating transpose `.contiguous()` copies, rewriting depthwise `ShortConvolution` / `DynamicShortConvolution` and `BlockAttentionResidual` in contiguous `[B, T, C]` layouts, compiling chunked CE (`chunk=4096`), hoisting loop residual prep, and applying selective activation checkpointing (`gc_sac`) pushed single-GPU `5b` (`bs=32x2048`, `136.85 GB`) to **`23,175 tok/s/B200`** in BF16 (**\$4,794** / 100B tokens) and **`21,719 tok/s/B200`** with exact Four-Over-Six NVFP4 quantization (**\$5,116** / 100B tokens), plus **`41,510 tok/s` combined on `2x NVIDIA B200` NCCL DDP** (`97.4%` scaling efficiency) with `--resume`.
> - **Why \$4.8k–\$5.1k isn't yet \$2k–\$4k (`30k–60k tok/s`):** We tested TE `2.20.2` (`NVTE_GROUPED_LINEAR_SINGLE_PARAM=1`), PyTorch `2.15` `F.scaled_grouped_mm`, and `torchao` `prototype.moe_training` live on B200. Both TE (`<= 64` tensors) and `torchao`'s SM100 padding kernel (`<= 32` groups) hard-fail on `E=128`, and PyTorch `2.15` has no hardware NVFP4 grouped GEMM kernel (`ValueError: No gemm implementation was found`). Closing the last `1.3x–1.5x` to **`< \$3,700`** waits on SM100 grouped FP4/FP8 kernels supporting `E >= 128` plus a CUTLASS FlashAttention-4 RPB+Sink backward kernel.

---

## 1. The Scoreboard & Hardware Stack

All measurements below were run on **NVIDIA B200 (`191.51 GB` HBM3e, compute capability `10.0`)** GPUs via Modal (`nvcr.io/nvidia/pytorch:26.10-py3`):

- **PyTorch:** `2.15.0a0+875d815502.nv26.10`
- **CUDA:** `13.4`
- **Transformer Engine:** `2.20.2+6ea2a74a`
- **Transformers:** `5.24.0`
- **Target Model (`5b` preset):** `5.403 B` stored parameters, `8`-of-`128` routed experts (`hidden=1024`, `moe_intermediate_size=512`), `24` layers with layers `6..17` looped `3x` (`48` layer applications/token), `0.873 B` active per pass, `1.127 B` weights applied/token, `9.02e9` training FLOPs/token (with activation checkpointing `8 * applied`), `86.4 GB` training weight+grad+AdamW state before activations.

| Configuration (`5b`, FP32 master weights + AdamW) | Status | Tok/s per B200 | Peak VRAM | Effective TFLOPS (BF16 MFU vs 2.25 PF) | GPU-hours (100B tok) | Est. Cost @ \$4/GPU-hr (8 GPUs) |
|---|---|---:|---:|---:|---:|---:|
| Baseline (`te tml`, `bs=8x2048`, `align=32`, eager attn/CE) | **Measured** | `1,412` | `145.52 GB` | `12.74` (`0.57%`) | `19,673` | **\$78,691** (`102.5 d`) *(or \$100k+ @ small batch / \$5.50/hr)* |
| + Chunked CE (`200k` vocab) & Fused ShortConv (`bs=8`) | **Measured** | `2,170` | `101.91 GB` | `19.57` (`0.87%`) | `12,801` | **\$51,203** (`66.7 d`) |
| + CUDA `FlexAttention` w/ Sinks & RPB (`bs=8`) | **Measured** | `2,476` | `99.38 GB` | `22.33` (`0.99%`) | `11,219` | **\$44,875** (`58.4 d`) |
| + Double Micro-batch (`bs=16x2048`, Part 1 peak) | **Measured** | `3,756` | `115.71 GB` | `33.88` (`1.51%`) | `7,396` | **\$29,582** (`38.5 d`) |
| + Zero-sync `torch._grouped_mm` BF16 + Compiled Layer & CE (`bs=16`, `gc_layer`) | **Measured** | `15,686` | `104.81 GB` | `141.49` (`6.29%`) | `1,771` | **\$7,083** (`9.2 d`) |
| + **Triton Flash-RPB+Sink, Zero-Copy Convs/AttnRes, `gc_sac` (`bs=32`, exact 4/6 NVFP4)** | **Measured** | **`21,719`** | **`136.85 GB`** | **`195.91` (`8.71%`)** | **`1,279`** | **\$5,116** (`6.7 d`) |
| + **Triton Flash-RPB+Sink, Zero-Copy Convs/AttnRes, `gc_sac` (`bs=32`, BF16 `quant="none"`)** | **Measured** | **`23,175`** | **`136.85 GB`** | **`209.04` (`9.29%`)** | **`1,199`** | **\$4,794** (`6.2 d`) |
| Next Target: SM100 Hardware Grouped FP4/FP8 (`E >= 128`) + FA-4 RPB | *Projected* | *30,000–45,000* | *< 140 GB* | *270.6–405.9 (`12.0%–18.0%`)* | *617–926* | ***\$2,469–\$3,704*** *(3.2–4.8 d)* |

*Note on `plan.py` FLOPs basis:* Effective TFLOPS above use `plan.py`'s `9.02e9` training FLOPs/token basis (`8x` applied weights assuming full layer recompute; actual FLOPs/token under `gc_sac` is slightly lower because Triton attention forward is saved rather than recomputed).

---

## 2. Why First-Pass B200 Runs Cost \$78k–\$100k+

When our correctness-verified `loopfp4` training script first ran the `5b` preset cleanly on a real B200 (`runs/b200-verify-report.json`), `100B` tokens projected to **19,673 GPU-hours (\$78,691 at \$4/GPU-hr, crossing \$100k+ at small batches like `4x1024` or standard on-demand \$5–\$6/hr B200 rates)**:

- Throughput was only **`1,412 tok/s/B200`** at `bs=8x2048` (`16,384` tokens/step).
- Peak memory hit **`145.52 GB` / `191.51 GB`** — leaving no room to double micro-batch size (`bs=16`), because static FP32 weights + gradients + two Adam moments already occupy `86.4 GB`, leaving ~`59 GB` consumed by activations and a single huge output tensor.

An isolated single-layer profile (`bs=8x2048`) exposed where the wall clock was going:

```text
self_attn_swa_dense_fp32   :  24.55 ms / fwd+bwd
moe_te_tml_align32         :  85.37 ms / fwd+bwd   (TE NVFP4 CustomRecipe, 128 experts)
moe_te_tml_align128        :  83.79 ms / fwd+bwd
moe_te_rowscaled_align32   :  85.79 ms / fwd+bwd
moe_sim_nvfp4              :  95.90 ms / fwd+bwd
mini-pretrain (384 experts): 311.84 ms (TE tml/nvfp4, align=32) vs 116.01 ms (TE tml/mxfp8, align=32)
```

Three red flags jumped straight off the profiler output:
1. Why did `mini-pretrain`'s 384-expert MoE run **2.69x slower** in `tml/nvfp4` (`311.84 ms`) than in `tml/mxfp8` (`116.01 ms`) on the exact same `GroupedLinear` module?
2. Why was `5b` dense attention + eager `softcap` + RPB costing **`24.55 ms` per layer** (~1.2 seconds per step across 48 layer applications)?
3. Why was peak VRAM hitting **`145.52 GB`** at `bs=8` even with `grad_checkpoint=True`?

We tackled the clean, zero-approximation wins first, then dug into TE's C++/Python dispatch to solve #1.

---

## 3. Optimization 1 — Empirical `m_splits` Alignment (`128 → 32`)

TE's `GroupedLinear` requires each expert's row count `m_splits[e]` to satisfy block-scaling alignment rules. Padding every expert up to `128` rows wastes serious FLOPs at small token-per-expert counts (with `128` experts and `top_k=8`, `16,384 * 8 / 128 = 1,024` tokens/expert average, and far worse at `4x1024` where `32` tokens/expert padded to `128` adds **~25%–75%** dead compute).

Testing powers of two directly on B200 TE `2.20.2` (`nvfp4` and `mxfp8`, dense and sparse expert splits) pinpointed the exact hardware minimum:
- `align=1`: crashes in `quantizer.cpp:2776` (NVFP4 requires `dim % 16 == 0`).
- `align=16`: crashes in `quantizer.cpp:1861` (MXFP8 requires `dim % 32 == 0`).
- **`align=32`**: runs cleanly across both NVFP4 and MXFP8 layers, matching `align=128` on `5b` MoE to **`0.0` `dgrad` diff**, **`0.0` `max_wgrad` diff**, and **`<= 1.9e-5` forward diff** (`runs/b200-verify-report.json`).

Making `--te-align 32` the default in `InklingConfig` eliminated unnecessary row padding cleanly without changing numerics.

---

## 4. Optimization 2 — Eliminating the 36 GB Logit Wall (Chunked Cross-Entropy + Fused ShortConv)

Inkling uses an unpadded output vocabulary of **`200,058`** tokens and `final_logit_softcapping = 30.0`. At `bs=8x2048` (`16,384` tokens), materializing full `logits` in FP32 (`softcap(self.lm_head(hidden).float(), 30.0)`) creates:

- `[16384, 200058]` in FP32 = **`13.1 GB`** for `logits` alone,
- plus intermediate `tanh` and `cross_entropy` log-probabilities retained for backward = **`~24–36 GB`** of transient VRAM right at the peak of the compute graph.

### Fix: Checkpointed `1024`-Token Chunked Cross-Entropy (`return_logits=False`)
In `loopfp4/inkling.py` and `loopfp4/model.py`, we kept `return_logits=True` untouched for inference/generation/reference tests and added a training/eval fast path (`return_logits=False` used in `train.py`):

```python
        if targets is not None and not return_logits:
            h_flat = hidden.reshape(-1, hidden.size(-1))
            t_flat = targets.reshape(-1)
            chunk = 1024

            def _ce_chunk(h_c, t_c):
                logits_c = softcap(self.lm_head(h_c).float(), cfg.final_logit_softcapping)
                return F.cross_entropy(logits_c, t_c, reduction="sum")

            total = h_flat.new_zeros((), dtype=torch.float32)
            for s in range(0, h_flat.size(0), chunk):
                h_c, t_c = h_flat[s : s + chunk], t_flat[s : s + chunk]
                if self.training and h_c.requires_grad:
                    total = total + checkpoint(_ce_chunk, h_c, t_c, use_reentrant=False)
                else:
                    total = total + _ce_chunk(h_c, t_c)
            return None, total / t_flat.numel()
```

Along the way, we also replaced the 4-tap Python slice loop in `ShortConvolution.forward` (which executed 96 times per step across `q_proj`, `k_proj`, `v_proj`, and post-attn/post-MLP convs) with a single fused depthwise `F.conv1d`:

```python
        y = F.conv1d(
            xf.transpose(1, 2), self.conv1d.weight.float(), padding=K - 1, groups=xf.shape[-1]
        )[..., :T].transpose(1, 2)
```

### Measured Impact on B200
- **200k-vocab CUDA BF16 parity (`2048` tokens):** `loss_diff = 0.0`, `d_hidden_max_diff = 3.58e-7`, `d_head_max_diff = 1.53e-5`.
- **Peak VRAM (`bs=8x2048`):** **`145.52 GB → 101.91 GB`** (`-43.6 GB`!).
- **Throughput (`bs=8x2048`, eager attn):** **`1,412 → 2,170 tok/s/B200`** (`+53.7%`, mostly from memory pressure relief + fused conv, even before touching attention).
- **Bigger win:** freeing 43 GB immediately unlocked **`bs=16x2048`** within `191.5 GB`.

---

## 5. Optimization 3 — Exact CUDA `FlexAttention` for Content-Dependent RPB + Attention Sinks

Inkling's attention isn't standard SDPA. Every layer mixes two custom terms that normally defeat FlashAttention:
1. **Content-dependent Relative Position Bias (RPB):** Each token projects `rel_logits` of shape `[B, heads, T, rel_extent]` (`512` for sliding-window layers, `1024` for every 6th global layer), gathered at `dist = q_idx - k_idx` inside `0 <= dist < rel_extent`.
2. **Per-token, per-head Attention Sinks (`sink_proj`):** Softmax runs over `[score_1, ..., score_T, sink_logit]`, keeping only the first `T` weights so a head can dump probability mass onto the sink instead of any token.

### Deriving Exact Sink Rescaling from Online Softmax `LSE`
`flex_attention(..., score_mod=..., block_mask=..., enable_gqa=True)` can compute `score + rpb(q, k)` inside Triton without materializing an `[B, H, T, T]` score matrix, and it returns both:

$$
\text{out}_{\text{flex}} = \sum_{k \in \mathcal{A}(q)} \frac{e^{s_k}}{Z_{\text{tok}}} v_k, \qquad \text{LSE} = \ln\left(\sum_{k \in \mathcal{A}(q)} e^{s_k}\right) = \ln Z_{\text{tok}}
$$

Adding a virtual sink token with logit $c = \text{sink\_proj}(q_{\text{in}})$ multiplies every token's softmax weight by the exact scalar:

$$
\frac{Z_{\text{tok}}}{Z_{\text{tok}} + e^c} = \frac{1}{1 + e^{c - \text{LSE}}} = \sigma(\text{LSE} - c)
$$

Therefore:

$$
\text{out}_{\text{sink}} = \text{out}_{\text{flex}} \cdot \sigma(\text{LSE} - c)
$$

This is mathematically exact — no approximation, no extra `[T, T]` tensor, and `Inkling.forward` completely skips allocating `dist` and `masks` when `T >= 128` on CUDA (`BlockMask` is cached once per `(T, window, device)` behind `@torch.compiler.disable`). Inside `score_mod`, clamping `d = q_idx - k_idx` is essential because FlexAttention invokes `score_mod` on partial boundary tiles before masking inactive positions:

```python
def _flex_attn_impl(q, k, v, rel_logits, sinks, block_mask, rel_extent: int):
    def score_mod(score, b, h, q_idx, k_idx):
        d = q_idx - k_idx
        ok = (d >= 0) & (d < rel_extent)
        bias = torch.where(ok, rel_logits[b, h, q_idx, d.clamp(0, rel_extent - 1)], 0.0).to(score.dtype)
        return score + bias

    if _FLEX_USE_AUX:
        out, aux = flex_attention(
            q, k, v, score_mod=score_mod, block_mask=block_mask,
            enable_gqa=True, return_aux=AuxRequest(lse=True),
        )
        lse = aux.lse
    else:
        out, lse = flex_attention(...)
    return (out.float() * torch.sigmoid(lse.float() - sinks).unsqueeze(-1)).to(v.dtype)
```

### Two B200 Debugging War Stories Along the Way

#### War Story #1: The `6.15e-3` FP32 "Bug" That Was NGC's Default TF32
On our first B200 verification run, raw FlexAttention matched CPU to `1.19e-6`, yet on B200 FP32 it failed our `< 5e-4` assertion with `fwd = 6.15e-03` (`d_q rel_l2 = 2.10e-03`, `d_sinks rel_l2 = 6.92e-04`). All 5 gradients differed by ~`1e-3` relative error — the exact fingerprint of 10-bit TF32 (`2^-10 ≈ 9.8e-4`) vs 23-bit IEEE FP32.

Logging PyTorch's precision state inside NGC `26.10` confirmed why:
```text
fp32_env: {'NVIDIA_TF32_OVERRIDE': None, 'get_float32_matmul_precision': 'high', 'matmul_fp32_precision': 'tf32'}
```
Once `PERF-1A` explicitly set `torch.set_float32_matmul_precision("highest")` and `matmul.fp32_precision = "ieee"` before compiling, FlexAttention matched eager FP32 on B200 to **`1e-6`**:
- **SWA (`512`, FP32 IEEE):** `fwd = 1.67e-6` (`rel=2.99e-7`), max grad diff across `q, k, v, rel_logits, sinks` `<= 1.12e-5` (`rel_l2 <= 9.66e-7`).
- **Global (`1024`, FP32 IEEE):** `fwd = 7.75e-7` (`rel=2.98e-7`), max grad diff `<= 1.05e-5` (`rel_l2 <= 9.66e-7`).
- **BF16 (`5b` Attention modules):** `fwd_max_diff = 1.95e-3`, `d_in_rel_l2 <= 6.80e-3`, `d_param_rel_l2 <= 6.12e-3`.

#### War Story #2: The Silent 7.7x Slowdown From TorchDynamo's `recompile_limit = 8`
Right after `PERF-1A` restored TF32 for benchmarking, our first B200 single-layer `FlexAttention` benchmark printed a horror story: **FlexAttention ran in `183.28 ms` (`34.79 GB`) vs eager `23.91 ms` (`9.87 GB`)**, and full `5b` crashed in `WhereBackward0` with `aten._local_scalar_dense.default` inside `make_fx`!

Checking stderr revealed the silent killer:
```text
torch._dynamo hit config.recompile_limit (8)
function: '_flex_attn_impl'
last reason: 0/6: GLOBAL_STATE changed: cuda_matmul_precision
UserWarning: flex_attention called without torch.compile() - this will use an unfused implementation that materializes the full scores matrix instead of generating a fused kernel.
```
Our test harness had compiled `_flex_attn_impl` across >8 combinations of dtypes/shapes plus a `cuda_matmul_precision` toggle (`highest → high`). Once Dynamo hit its default `recompile_limit = 8`, PyTorch **silently bypassed `torch.compile`** and dropped back to unfused Python `flex_attention`!

Raising `torch._dynamo.config.cache_size_limit` and `recompile_limit` to `>= 64` in [loopfp4/inkling.py](../loopfp4/inkling.py) (and resetting Dynamo after precision switches in the test script) restored true fused Triton kernels:
- **Single-layer SWA (`bs=8x2048`):** `23.87 ms` (`9.87 GB`) → **`8.70 ms` (`4.29 GB`)** — **2.74x faster**, **2.30x less VRAM**.
- **Single-layer Global (`bs=8x2048`):** `25.66 ms` (`10.35 GB`) → **`11.68 ms` (`4.78 GB`)** — **2.20x faster**, **2.17x less VRAM**.
- **Full `5b` (`bs=8x2048` → `bs=16x2048`):** `2,170 → 2,476 tok/s` (`99.38 GB`) at `bs=8`, and **`3,756 tok/s` (`115.71 GB`) at `bs=16`**!

---

## 6. Why `torch.compile(model)` Didn't Save Us — and the 5.5x TE `CustomRecipe` MoE Wall

So why did full `5b` only reach `3,756 tok/s/B200` (`1.51%` MFU) even when attention took just `8.7–11.7 ms`, and why did `torch.compile(model)` (`2,366 tok/s` at `bs=8`) actually run *slower* than eager (`2,476 tok/s`)?

We inspected `transformer_engine/pytorch/module/grouped_linear.py` live inside the B200 container (`runs/b200-ddp-perf-report.json`) and found the exact gatekeeper function: **`is_module_grouped_tensor_path_supported(recipe, dtype)`**:

```python
    if recipe is None:
        return True
    if recipe.custom():
        return False
    if recipe.backward_override is not None:
        return False
    ...
    if recipe.mxfp8():
        return device_capability >= (10, 0)
    if recipe.nvfp4():
        return (
            device_capability >= (10, 0)
            and not recipe.disable_rht
            and not recipe.row_scaled_activation
        )
    return False
```

This single predicate explains every anomalous MoE number we measured on B200:
1. **`CustomRecipe` (`tml` and `rowscaled`) always hits the slow loop:** Because `recipe.custom()` returns `True`, `is_module_grouped_tensor_path_supported()` immediately returns `False`. `GroupedLinear` falls back to splitting and quantizing experts in a Python loop (`128` experts per GEMM, `2` GEMMs per layer, `48` layer applications = **`12,288` separate small GEMM dispatches per step**!).
2. **Stock `NVFP4BlockScaling()` crashes at `> 64` experts:** Enabling NVIDIA's built-in grouped NVFP4 path keeps Random Hadamard Transforms (`disable_rht=False`) on by default, which hits an internal kernel assertion on both `5b` (`E=128`) and `mini-pretrain` (`E=384`):
   `group_hadamard_transform.cu:408: Assertion failed: num_tensors <= kMaxTensorsPerKernel. Number of tensors should be less than or equal to 64`
3. **`torch.compile(model)` can't fuse across that boundary:** `_experts_te` calls `int(padded_counts.sum())` (triggering a CUDA stream sync / graph break) and dispatches opaque TE `GroupedLinear` custom ops, so full-model `torch.compile` adds trace/guard overhead without fusing the 128-expert loop (`2,366` vs `2,476 tok/s`).

Benchmarking a single `5b` MoE block (`bs=8x2048`, `16,384` tokens, `E=128, H=1024, inner=512`) side by side on B200 proves how massive that single MoE bottleneck is:

| MoE Kernel Implementation (`5b`, `16,384` tokens) | ms / fwd+bwd | Peak VRAM | vs TE `tml` NVFP4 |
|---|---:|---:|---:|
| TE `GroupedLinear` (`tml` NVFP4, FP32 params, `m_splits` Tensor) | `49.93 ms` | `7.73 GB` | `1.00x` (baseline) |
| TE `GroupedLinear` (`tml` NVFP4, BF16 params, `m_splits` Tensor) | `50.89 ms` | `7.33 GB` | `0.98x` (same Python loop) |
| TE `GroupedLinear` (`tml` NVFP4, BF16 params, `list[int]`) | `61.65 ms` | `8.52 GB` | `0.81x` |
| TE `GroupedLinear` (`tml` MXFP8, FP32 params, `m_splits` Tensor) | `30.76 ms` | `7.82 GB` | `1.62x` faster |
| **PyTorch `torch._grouped_mm` (BF16 fused grouped GEMM, `align=32`)** | **`9.01 ms`** | `9.33 GB` | **`5.54x` faster** |

At `49.93 ms` per MoE block across `48` layer-applications, TE `GroupedLinear` accounts for `~2.40 seconds` per pass (plus backward recomputation under `grad_checkpoint=True`). Replacing it with a true fused grouped GEMM immediately cuts MoE time by **5.54x** (`49.93 ms → 9.01 ms`) even *before* using Blackwell's 2x FP4 tensor core rate over BF16.

---

## 7. Multi-GPU `2x B200` NCCL DDP + Live Data Verification

Before putting any serious cluster budget on `5b`, we also verified end-to-end distributed training on **`2x NVIDIA B200`** (`torchrun --standalone --nproc_per_node=2` over NCCL, `runs/b200-ddp-perf-report.json`) and live Hugging Face data streaming:
- **Live FineWeb-Edu + Inkling-2-Mini data prep ([prepare_data.py](../prepare_data.py)):** Streamed `hf:HuggingFaceFW/fineweb-edu:sample-100BT` tokenized with `hf:thinkingmachines/Inkling-2-Mini` (`200,058` vocab, `uint32` shards), verified `ShardData` loads `[2, 2048]` training batches in `[0, 199999]`, and fixed a Python interpreter teardown hang after `from_hf` early return via `os._exit(0)`.
- **`smoke` 2-GPU DDP (`tml` NVFP4, `grad_accum=2`):** Trained steps `1..6`, saved `ckpt.pt`, resumed cleanly (`resumed from step 6`) to step `10` with synchronized per-rank `data_rngs` and router `balance_step` `all_reduce`. Loss dropped `5.5958 → 5.0209` (`val_loss = 4.9848`), `exit=0`.
- **`5b` 2-GPU DDP (`5.403B` params, `bs=4x2048`, `grad_accum=2`):** Trained steps `1..4`, saved the `~58.5 GB` checkpoint (`549.8 GB` free `/tmp`), resumed cleanly (`resumed from step 4`) through step `6`. Expert balance converged (`load_max` `2.11 → 1.46`, `0` idle experts) and combined throughput reached **`2,830 tok/s`** (`~1,415 tok/s/GPU` at micro-batch `4`, highlighting again why large per-GPU batch sizes matter for sparse MoE).

---

## 8. Part 2: Profiling the 8,483 ms Step & Mapping Hardware 4-Bit / 8-Bit Grouped GEMM Reality on B200

With Part 1 at `3,756 tok/s/B200` (**\$29,582**), we profiled a full `bs=16x2048` step on B200 (`runs/b200-part2-probe.json`) before writing a single new kernel:

```text
full_5b bs=16x2048 (te/tml, align=32): 8,483.1 ms/step (3,863 tok/s, 112.98 GB)
  forward_ms     : 3,132.6 ms
  backward_ms    : 5,304.8 ms
  adamw_step_ms  :    44.2 ms  (standalone fused AdamW: 25.48 ms across 6,765 TE tensors vs 23.52 ms across 669 stacked tensors)
  balance_step_ms:     1.5 ms
  CPU/launch view: _GroupedLinear 3,920 ms self CPU (6,533 ms cum), 312x cudaStreamSynchronize (2,406 ms), 484,457 cudaLaunchKernel calls!
```

Optimizer step overhead wasn't the culprit (`44 ms` out of `8,483 ms`). Nearly everything was TE `_GroupedLinear` Python loop overhead (`312` host syncs, `484k` launches) plus uncompiled `1024`-token chunked CE (`346.75 ms` per step).

### Testing Every Native SM100 Grouped FP4 / FP8 Path on B200
We tested every Blackwell grouped quantization API shipped in NGC `26.10` (PyTorch `2.15`, TE `2.20.2`, `torchao`) on our `E=128` (`5b`) and `E=384` (`pretrain`) shapes (`runs/b200-fp4-gmm-probe.json`, `runs/b200-torchao-layer-probe.json`):

1. **TE `2.20.2` with `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1` (`single_grouped_weight=True, use_grouped_tensor=True`):**
   - `BF16` works (`17.13 ms` single MoE vs `62.18 ms` `tml`).
   - `MXFP8BlockScaling()` hard-fails: `group_quantize_mxfp8.cuh:1174: num_tensors <= 64` (got `128`).
   - `NVFP4BlockScaling()` hard-fails: `graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu:1385: num_tensors <= 64`.
   - TE's internal `general_grouped_gemm_for_grouped_tensor` explicitly raises `NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")`.
2. **`torchao.prototype.moe_training`:**
   - `nvfp4_grouped_mm.py` is **emulated BF16 only** (`_emulated_nvfp4_scaled_grouped_mm_2d_3d` dequantizes to BF16 and calls `torch._grouped_mm`).
   - `mxfp8_grouped_mm.py` has real SM100 hardware kernels, but enabling `pad_token_groups_for_grouped_mm=True` hits `mxfp8_extension.cpp:299: fused_pad_token_groups: num_groups must be <= 32, got: 128`.
3. **PyTorch `2.15` `torch.nn.functional.scaled_grouped_mm` (raw 2-GEMM at `M=262,144, E=128, H=1024, inner=512`):**
   - **MXFP8 (`ScalingType.BlockWise1x32`, `SWIZZLE_32_4_4`):** runs on SM100 tensor cores in **`0.52 ms` forward** (`3.17 GB`, `rel_l2 = 3.33e-6` vs dequantized BF16) — **`3.2x` faster** than `torch._grouped_mm` BF16 (`1.66 ms` fwd, `4.22 ms` fwd+bwd)!
   - **NVFP4 (`ScalingType.BlockWise1x16`):** raises `ValueError('No gemm implementation was found')` across all tested swizzle/shape variants.

**Takeaway:** Neither TE `2.20.2`, `torchao`, nor PyTorch `2.15` can execute hardware NVFP4 grouped GEMMs for `128` experts today. To get immediate `5x–6x` production speedups *while preserving exact Four-Over-Six NVFP4 training numerics and identical checkpoints*, we added `--backend grouped` in [loopfp4/inkling.py](../loopfp4/inkling.py): zero-sync `torch.argsort` + **unpadded `torch._grouped_mm` BF16** over `canonical_state_dict`-identical stacked weights, with optional exact Four-Over-Six NVFP4 straight-through quantization compiled right into Inductor!

---

## 9. Zero-Sync `backend="grouped"` & Compiled Chunked CE (`3,756 → 15,686 tok/s`)

1. **Unpadded zero-sync MoE dispatch (`SparseMoeBlock._experts_grouped`):**
   - Replaces `int(padded_counts.sum())` (which broke `torch.compile` and forced host-device syncs) with `expert_i32 = expert.to(torch.int32)`, `torch.argsort(expert_i32, stable=False)`, `offs = counts.cumsum(0, dtype=torch.int32)`, unpadded `torch._grouped_mm`, and direct `torch.zeros(...).index_add_(0, token, y)`.
   - When `quant="nvfp4"`, `quantize_act(flat, ...)` quantizes `[N, H]` **once per token before gathering** (`8x` less quantization work than quantizing `[N*top_k, H]`), with identical Four-Over-Six STE values and gradients.
   - Verified on B200 even with half empty experts and odd/unaligned counts `[0, 1, 0, 3, 0, 7, 0, 17, 0, 33, 0, 2, 0, 5, 0, 11]` (`fwd_rel=4.42e-3`, `dx_rel=4.59e-3`, `dw1_rel=4.04e-3`, empty expert grad `0.0`) and full `SparseMoeBlock` parity vs `sim` (`quant="none"` `fwd=4.6e-5`, `quant="nvfp4"` `fwd=2.2e-5`).
   - Compiled `5b` MoE (`bs=16x2048`): drops from `62.18 ms` (TE `tml`) to **`4.90 ms`** (`3.93 GB`)! Four-Over-Six quantization adds just `+1.27 ms/layer` (`+6.8%`) when compiled (`17.76 ms → 19.05 ms` SWA layer).
2. **Compiled Chunked CE (`chunk = 4096` on CUDA):**
   - Extracting `_ce_chunk_impl` cleanly behind `@torch.compile` and bumping `chunk` from `1024 → 4096` cut `200k`-vocab cross-entropy (`32,768` tokens) from **`346.75 ms → 97.59 ms` (`3.55x` faster)** at just `5.70 GB` peak VRAM.
3. **Per-layer compilation (`layer.compile()`):**
   - Compiling each of the 24 `DecoderLayer` instances (`layer.compile()` in [train.py](../train.py)) gives clean TorchInductor fusion across `48` layer calls without unrolling a 48-layer giant graph — immediately lifting `5b` with FP32 master weights to **`15,686 tok/s/B200`** (**`4.18x`** over Part 1, **\$7,083**/100B tokens)!

---

## 10. Why Attention + ShortConv Cost 3x More Than MoE — and How We Fixed Both

Once MoE dropped to `~5 ms/layer`, profiling a compiled SWA layer (`23.89 ms`) and Global layer (`28.68 ms`) revealed an ironic inversion (`runs/b200-torchao-layer-probe.json`):

```text
Compiled ZeroSyncGroupedMMMoE alone (bs=16x2048)        :  5.56 ms
Compiled SWA Attention + Convs + Residuals ONLY (no MLP): 16.81 ms  (3.0x SLOWER than MoE!)
```

Three hidden kernels explained why:
1. **`FlexAttention` backward + `rel_logits` atomics (`6.41 ms` SWA / `8.64 ms` Global):** Capturing a 4D `[16, 12, 2048, 1024]` `rel_logits` tensor (`805 MB` BF16 / `1.61 GB` FP32) inside `score_mod` forces `flex_attention_backward` to scatter into `d_rel_logits` via `tl.atomic_add` inside its inner K-loop (`1.04 ms fwd → 6.41 ms bwd`, **`6.2x`** backward asymmetry!).
2. **`ShortConvolution` `F.conv1d(..., groups=dim)` under `torch.compile` (`~3.0 ms/SWA layer`):** Replacing the slice loop with `F.conv1d` helped eager mode in Part 1, but Inductor lowers depthwise `conv1d` to opaque `aten::_conv_depthwise2d` + `[B, T, C] ↔ [B, C, T]` transposes across 4 convs per SWA layer. Meanwhile `DynamicShortConvolution` used `unfold(-1, K, 1)` to materialize `[16, 2048, 1024, 4]` windows (`~4 ms` on Global layers).
3. **`BlockAttentionResidual.mix` `aten::bmm` (`1.12 ms` SWA / `4.18 ms` Global):** `attn @ states` invoked Tensor Core `bmm` on tiny `[1..3, 1..5] × [1..5, 1024]` slices (`cutlass_80_wmma` `16x16` padding waste). Plus `Inkling.forward` called `MessageAttentionResidual` 12 times per loop pass (`24x` total) on the same `(prev_blocks, prev_hidden)`.

### Fixes Implemented
- **Custom Triton Flash-RPB+Sink Attention ([loopfp4/triton_attn.py](../loopfp4/triton_attn.py)):**
  - Because `d = q_idx - k_idx` is **strictly 1-to-1 per query row**, splitting backward into K/V-parallel `_rpb_sink_bwd_dkdv_kernel` (`zero atomics`) + Q-parallel `_rpb_sink_bwd_dq_drpb_kernel` replaces every `tl.atomic_add` with plain coalesced stores into `d_rel_logits`, while forwarding skips saving `rel_logits` completely (`12.6 MB` `rel` instead of `1.61 GB`!).
  - Accepts non-contiguous `.transpose(1, 2)` inputs (`q, k, v, rel, sinks`) directly via strides — **zero `.contiguous()` transpose copies**, avoiding the `17` copies (`1.31 ms/layer`) we hit on our first draft! *(Tile-tuning lesson on B200: bumping forward to `BM=128, num_stages=3` spilled registers/SMEM and regressed fwd to `4.00 ms`, whereas `(BM=64, BN=64, num_warps=4, num_stages=2)` achieved **`3.44 ms`** kernel time cleanly!)*
  - **Exactness verified on B200 (`runs/b200-part2-verify-report.json`):** Matches eager FP32 IEEE across SWA `T=2048`, Global `T=2048`, `smoke` (`D=32, T=256`), and non-multiple `T=300` to **`fwd_rel_l2 <= 5e-7`** and **all 6 gradients (`q, k, v, rel, proj, sinks`) `<= 1.8e-6`** (`BF16 <= 2.93e-3`). At `bs=16x2048`: SWA `10.08 ms (3.90 GB) → 8.28 ms (2.21 GB)` (`1.22x`), Global `14.86 ms → 9.12 ms` (`1.63x`).
- **Contiguous Convs, Fused AttnRes, & Hoisted Loop Messages ([loopfp4/inkling.py](../loopfp4/inkling.py)):**
  - Rewrote `ShortConvolution` and `DynamicShortConvolution` as native contiguous `[B, T, C]` 4-tap shift-sums that Inductor fuses into single pointwise kernels (verified `atol=1e-5` exact forward + grads in `test_fast_convs_and_attn_res_exact`).
  - Replaced `attn @ states` in `BlockAttentionResidual` with fused elementwise reductions (`zero bmm`), and hoisted `MessageAttentionResidual.prepare_states(...)` to run once per loop pass (`2x` instead of `24x`).
  - Result: Compiled uncheckpointed `5b` layer latency dropped from **`23.89 ms → 17.76 ms`** (`SWA`) and **`28.68 ms → 19.87 ms`** (`Global`)!

---

## 11. Selective Activation Checkpointing (`gc_sac`), `bs=32` Scaling, & `2x B200` DDP + `--resume`

Full-layer checkpointing (`gc_layer`) left `~66 GB` of B200's `178.36 GiB` usable HBM unused (`112.08 GB` at `bs=16`) while paying recomputation on all 48 passes (`1,760 ms/step`). Simply disabling layer checkpointing on MLP-only (`gc_mlp`) OOM'd at `171.7 / 178.4 GiB` (`48` passes of uncheckpointed FP32 conv/residual intermediates cost `~85 GB`), and `gc_half` peaked near the limit (`173.9 GB`, `20,667 tok/s`).

Adding **Selective Activation Checkpointing (`--grad-checkpoint-mode sac`)** — saving `rpb_sink_attn_fwd` and dense `aten.mm` while recomputing only cheap elementwise/conv ops and MoE — unlocked the memory/compute sweet spot (`runs/b200-part2-verify-report.json`):

| `5b` Configuration (`backend="grouped"`, `layer.compile()`, FP32 master weights) | Tok/s / B200 | Step Time | Peak VRAM | Cost / 100B @ \$4/hr |
|---|---:|---:|---:|---:|
| `bs=16`, `bf16`, `gc_layer` | `18,617` | `1,760.1 ms` | `112.08 GB` | **\$5,968** |
| `bs=16`, `nvfp4`, `gc_layer` | `17,523` | `1,870.0 ms` | `112.08 GB` | **\$6,341** |
| `bs=16`, `bf16`, `gc_half` | `20,667` | `1,585.5 ms` | `173.85 GB` | **\$5,376** |
| `bs=16`, `nvfp4`, `gc_sac` (`fwd 513ms`, `bwd 1047ms`, `clip 10ms`, `opt 24ms`, `bal 3ms`) | `20,502` | `1,598.3 ms` | `104.19 GB` | **\$5,420** |
| `bs=16`, `bf16`, `gc_sac` (`fwd 455ms`, `bwd 993ms`, `clip 10ms`, `opt 24ms`, `bal 3ms`) | `22,056` | `1,485.7 ms` | `104.19 GB` | **\$5,038** |
| `bs=24`, `nvfp4`, `gc_sac` | `21,309` | `2,306.6 ms` | `120.51 GB` | **\$5,214** |
| `bs=24`, `bf16`, `gc_sac` | `22,800` | `2,155.8 ms` | `120.51 GB` | **\$4,873** |
| **`bs=32`, `nvfp4` (exact 4/6), `gc_sac`** | **`21,719`** | **`3,017.5 ms`** | **`136.85 GB`** | **\$5,116** |
| **`bs=32`, `bf16` (`quant="none"`), `gc_sac`** | **`23,175`** | **`2,827.9 ms`** | **`136.85 GB`** | **\$4,794** |

### Two-GPU `2x NVIDIA B200` NCCL DDP War Story & Verified Resume
Combining `layer.compile()`, `DDP`, and `SAC` on `2x B200` immediately hit a PyTorch `2.15` compiler bug: `DDPOptimizer` splits each compiled layer FX subgraph into multiple submodules across parameter buckets (`down_proj`), stripping `meta["ac_graph_id"]` and crashing in `cleanup_recompute_tags` with `TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'`.

Because `train.py` already compiles per-layer across `48` layer calls, setting `torch._dynamo.config.optimize_ddp = False` in [train.py](../train.py) when `args.compile and ink and ddp` keeps full inter-layer NCCL backward overlap while running cleanly under `DDP + SAC`. Verified end-to-end on **`2x NVIDIA B200`** (`--preset 5b --backend grouped --quant nvfp4 --grad-checkpoint-mode sac --compile --batch-size 24`):
- **Steps `1..4` + checkpoint save:** steady-state **`41,420–41,510 tok/s` combined** (`97.4%` scaling efficiency vs `21,309 tok/s` single-GPU at `bs=24`), `idle_experts = 0`, `exit=0`.
- **Resumed steps `5..8` (`--resume`):** `resumed from step 4`, **`41,473–41,508 tok/s` combined**, `load_max` tightened `1.70 → 1.32`, loop gates opened (`1e-5 → 0.00011`), `exit=0`.

---

## 12. Closing the Last `1.3x–1.5x` to \$2k–\$4k (`30k–45k tok/s`)

From our final `bs=16` SWA profiler (`17.76 ms` BF16 / `19.05 ms` NVFP4) and phase breakdown (`455 ms` fwd / `993 ms` bwd), two concrete hardware/kernel items separate **\$4,794–\$5,116** (`21.7k–23.2k tok/s`) from **\$2,469–\$3,704** (`30k–45k tok/s`):
1. **Warp-Specialized / TMA FlashAttention-4 RPB+Sink:** Triton `rpb_sink_attn` currently accounts for `3.54 ms fwd + 4.84 ms bwd = 8.38 ms` — **`47%` of the entire compiled SWA layer** — largely due to `805 MB` `rel_logits`/`d_rel_logits` materialization (`~1.15 ms`) and lack of SM100 UMMA/TMA warp specialization compared to `FlexAttention` fwd (`1.92 ms`). Cutting attention fwd+bwd from `~8.4 ms → ~3.5–4.0 ms` saves **`~235 ms/step`** at `bs=16`.
2. **SM100 Native Grouped NVFP4/MXFP8 GEMMs for `E >= 128`:** Bypassing the `<= 32` / `<= 64` tensor-list caps lets `F.scaled_grouped_mm` (`0.52 ms` raw 2-GEMM MXFP8 vs `1.66 ms` BF16) and SM100 NVFP4 tensor cores eliminate both the `+1.27 ms/layer` Four-Over-Six emulation tax and `2x–3x` of grouped GEMM compute/bandwidth.

---

## 13. Reproducibility Appendix

```bash
# 1. Local unit tests (9 core + 19 inkling tests, zero network/GPU needed)
python3 tests/test_core.py && python3 tests/test_inkling.py

# 2. Live Hugging Face FineWeb-Edu + Inkling-2-Mini tokenizer data prep
python3 prepare_data.py \
    --source hf:HuggingFaceFW/fineweb-edu:sample-100BT \
    --tokenizer hf:thinkingmachines/Inkling-2-Mini \
    --out /tmp/fineweb-check --max-tokens 1e6

# 3. Part 1 B200 baseline, alignment sweep, exactness & TE 2x DDP
modal run modal_verify_b200.py
modal run modal_verify_ddp_and_perf.py

# 4. Part 2 B200 Triton attention exactness, unpadded grouped parity, 5B throughput & 2x DDP
modal run modal_verify_part2.py

# 5. Recompute 100B-token cost table at any measured throughput
python3 plan.py --preset 5b --gpus 8 --tokens 100e9 --tok-per-s-per-gpu 21719
python3 plan.py --preset 5b --gpus 8 --tokens 100e9 --tok-per-s-per-gpu 23175
```
