# How to Cut a 100B-Token 5B MoE Run from \$100k+ Toward \$2k–\$4k on NVIDIA B200s
### *Part 1: Achieving a Verified 2.66x Speedup (1,412 → 3,756 tok/s/B200) and Root-Causing the 5.5x Transformer Engine MoE Wall*

> **Engineering honesty upfront:** This post splits cleanly into **what is already measured and verified on real NVIDIA B200s** versus **what is root-caused and benchmarked as the blueprint for Part 2**.
> - **Completed & verified on B200 hardware:** We took the `5b` Inkling-style looped MoE model (`5.403B` stored / `0.873B` active per pass / `48` layer-applications per token) from **`1,412 tok/s/B200`** (`145.52 GB` peak VRAM → **\$78,691** at \$4/GPU-hr, or **\$100k+** at smaller batches or \$5–\$6/hr cloud pricing) up to **`3,756 tok/s/B200`** (`115.71 GB` peak VRAM → **\$29,582**) — a **2.66x end-to-end speedup** and **-46 GB peak VRAM reduction** at `bs=8x2048` with zero numeric approximations (`1e-6` IEEE FP32 attention parity, `0.0` chunked cross-entropy loss diff). We also verified real multi-GPU `2x B200` NCCL DDP with checkpoint save/resume and live `FineWeb-Edu` streaming data prep.
> - **Root-caused & benchmarked blueprint (not yet `30k–60k tok/s`):** Even after making attention `2.74x` faster (`8.70 ms`), full-step utilization sits at **`1.51%` of B200 BF16 peak** (`33.88 TFLOPS` effective). Inspecting NVIDIA Transformer Engine `2.20.2`'s `GroupedLinear` source and running isolated MoE benchmarks on B200 revealed the exact culprit: TE's `is_module_grouped_tensor_path_supported()` explicitly disables its fused `GroupedTensor` GEMM path for *any* `CustomRecipe` (`tml`, `rowscaled`), forcing every layer into a per-expert Python loop (`49.93 ms`). Swapping just the expert GEMMs to PyTorch's native `torch._grouped_mm` in BF16 drops single-layer MoE time to **`9.01 ms` (`5.54x` faster)** — proving the path to **`30k–60k tok/s/B200`** (**\$1,852–\$3,704** for 100B tokens).

---

## 1. The Scoreboard & Hardware Stack

All measurements below were run on **NVIDIA B200 (`191.51 GB` HBM3e, compute capability `10.0`)** GPUs via Modal (`nvcr.io/nvidia/pytorch:26.10-py3`):

- **PyTorch:** `2.15.0a0+875d815502.nv26.10`
- **CUDA:** `13.4`
- **Transformer Engine:** `2.20.2+6ea2a74a`
- **Transformers:** `5.24.0`
- **Target Model (`5b` preset):** `5.403 B` stored parameters, `8`-of-`128` routed experts (`hidden=1024`, `moe_intermediate_size=512`), `24` layers with layers `6..17` looped `3x` (`48` layer applications/token), `0.873 B` active per pass, `1.127 B` weights applied/token, `9.02e9` training FLOPs/token (with activation checkpointing `8 * applied`), `86.4 GB` training weight+grad+AdamW state before activations.

| Configuration (`5b`, TE `tml` NVFP4/MXFP8) | Status | Tok/s per B200 | Peak VRAM | Effective TFLOPS (BF16 MFU vs 2.25 PF) | GPU-hours (100B tok) | Est. Cost @ \$4/GPU-hr (8 GPUs) |
|---|---|---:|---:|---:|---:|---:|
| Baseline (`bs=8x2048`, `align=32`, eager attn/CE) | **Measured** | `1,412` | `145.52 GB` | `12.74` (`0.57%`) | `19,673` | **\$78,691** (`102.5 d`) *(or \$100k+ @ small batch / \$5.50/hr)* |
| + Chunked CE (`200k` vocab) & Fused ShortConv (`bs=8`) | **Measured** | `2,170` | `101.91 GB` | `19.57` (`0.87%`) | `12,801` | **\$51,203** (`66.7 d`) |
| + CUDA `FlexAttention` w/ Sinks & RPB (`bs=8`) | **Measured** | `2,476` | `99.38 GB` | `22.33` (`0.99%`) | `11,219` | **\$44,875** (`58.4 d`) |
| + Double Micro-batch (`bs=16x2048`, enabled by CE mem drop) | **Measured** | **`3,756`** | **`115.71 GB`** | **`33.88` (`1.51%`)** | **`7,396`** | **\$29,582** (`38.5 d`) |
| Target Part 2: Fused Grouped NVFP4 + Epilogue Fusion | *Projected* | *30,000–60,000* | *< 130 GB* | *270.6–541.2 (`12.0%–24.1%`)* | *463–926* | ***\$1,852–\$3,704*** *(2.4–4.8 d)* |

*Notice the headroom:* hitting \$2k–\$4k per 100B tokens (`30k–60k tok/s/B200`) only requires **`12%–24%` BF16-equivalent MFU** (`3%–6%` of B200's 9 PF FP4 tensor core peak). The whole game is kernel fusion and eliminating host/Python launch overhead.

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

## 8. Part 2 Roadmap — Closing the Gap from \$29.6k (`3,756 tok/s`) to \$2k–\$4k (`30k–60k tok/s/B200`)

The arithmetic from `plan.py` is unforgiving and crystal clear: reaching **\$1,852–\$3,704** for a `100B`-token `5b` run (\$4/GPU-hr) means hitting **`30,000–60,000 tok/s/B200`** (`~271–541 TFLOPS` effective, `~12%–24%` of BF16 peak):

1. **Replace TE's `CustomRecipe` `GroupedLinear` loop with fused grouped NVFP4 / MXFP8 GEMMs (Biggest lever — `5.5x+` on the dominant bottleneck):**
   - *Quick diagnostic / BF16 baseline:* swap `GroupedLinear` to `torch._grouped_mm` (`9.01 ms` vs `49.93 ms`, **5.54x faster** on the MoE layer) inside `_experts_te` to confirm immediate end-to-end speedup before adding FP4 quantization kernels.
   - *Target 4-bit kernel:* dispatch directly to Blackwell SM100 CUTLASS / cuBLASLt block-scaled `GroupedGEMM` (`NVFP4` forward without the `<= 64` expert RHT limit + `MXFP8` backward via TE's `general_grouped_gemm_for_grouped_tensor` or custom CUTLASS extensions using `single_grouped_weight=True`).
2. **Fused Triton MoE epilogue & quantization (`softcap + SwiGLU + 4/6 NVFP4 quant` + CUDA routing):**
   - Today `_experts_te` runs `softcap(gate) * sigmoid(gate) * softcap(up)` as ~7 separate memory-bound elementwise passes in FP32 between `gate_up` and `down`. Fusing `softcap + SiLU + Four-Over-Six NVFP4 activation quantization + padding scatter` into two Triton kernels removes multiple intermediate HBM round-trips per MoE layer.
3. **Zero host-syncs & CUDA Graphs / regional `torch.compile`:**
   - Eliminate `int(padded_counts.sum())` in `SparseMoeBlock._experts_te` (use fixed worst-case padded buffer capacity or asynchronous device-side offsets) so CUDA Graph capture or `torch.compile` runs zero-break across non-MoE blocks.
4. **Selective activation checkpointing & scale:**
   - Full-layer checkpointing increases compute from `6x` to `8x` applied weights per token (`6.76e9 → 9.02e9` FLOPs/token). With fused MoE + FlexAttention reducing activation footprints dramatically (`115.7 GB` at `bs=16`), checkpointing only dense/MoE sub-blocks selectively or using larger micro-batches recaptures another **1.25x–1.33x** throughput (and for `26B` `mini`, sharding optimizer/expert states via FSDP2 / Expert Parallelism instead of pure DDP).

---

## 9. Reproducibility Appendix

```bash
# 1. Local unit tests (9 core + 16 inkling tests, zero network/GPU needed)
python3 tests/test_core.py && python3 tests/test_inkling.py

# 2. Live Hugging Face FineWeb-Edu + Inkling-2-Mini tokenizer data prep
python3 prepare_data.py \
    --source hf:HuggingFaceFW/fineweb-edu:sample-100BT \
    --tokenizer hf:thinkingmachines/Inkling-2-Mini \
    --out /tmp/fineweb-check --max-tokens 1e6

# 3. Initial B200 verification, alignment sweep & recipe checks (~$0.28 on Modal)
modal run modal_verify_b200.py

# 4. Exactness (IEEE FP32 + BF16 + 200k CE), single-layer + full 5B benchmarks, and 2x B200 DDP
modal run modal_verify_ddp_and_perf.py

# 5. Recompute 100B-token cost table at any measured throughput
python3 plan.py --preset 5b --gpus 8 --tokens 100e9 --tok-per-s-per-gpu 3756
python3 plan.py --preset 5b --gpus 8 --tokens 100e9 --tok-per-s-per-gpu 30000
```
