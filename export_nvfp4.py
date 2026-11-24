"""Export a trained checkpoint with NVFP4 (4-bit) and MXFP8 (8-bit) weights packed.

    python export_nvfp4.py runs/ink-smoke/ckpt.pt runs/ink-smoke/model-nvfp4.pt

NVFP4 weights become packed nibbles + one E4M3 scale per 16 values + one FP32 scale per
tensor (per expert for the Inkling presets). MXFP8 weights become E4M3 values + one int8
power-of-two exponent per 32 values. Dequantizing them reproduces exactly the weights the
forward pass trained against, so there is no post-training quantization step and no
conversion loss. Everything else is stored in BF16. The container is this project's own,
not the released Inkling 2 Mini safetensors layout.
"""

import sys

import torch

from loopfp4 import Inkling, load_checkpoint
from loopfp4.nvfp4 import (
    QLinear,
    mxfp8_pack,
    mxfp8_unpack,
    nvfp4_pack,
    nvfp4_unpack,
    quantize_expert_weight,
)


def pack_nvfp4(w, want):
    p = nvfp4_pack(w)
    assert torch.equal(nvfp4_unpack(p), want), "packed NVFP4 weights do not reproduce the training weights"
    nbytes = p["codes"].numel() + p["s_block"].numel() * p["s_block"].element_size() + p["s_global"].numel() * 4
    return p, nbytes


def pack_mxfp8(w, want):
    p = mxfp8_pack(w)
    assert torch.equal(mxfp8_unpack(p), want), "packed MXFP8 weights do not reproduce the training weights"
    nbytes = p["codes"].numel() * p["codes"].element_size() + p["exp"].numel() * p["exp"].element_size()
    return p, nbytes


def main(src, dst):
    model, ck = load_checkpoint(src)
    out, n_fp4, n_fp8, n_other, nbytes = {}, 0, 0, 0, 0
    if isinstance(model, Inkling):
        fp4 = {f"layers.{i}.mlp.experts.{k}" for i, layer in enumerate(model.layers) if layer.mlp.mode == "nvfp4"
               for k in ("gate_up_proj", "down_proj")}
        fp8 = {f"layers.{i}.mlp.experts.{k}" for i, layer in enumerate(model.layers) if layer.mlp.mode == "mxfp8"
               for k in ("gate_up_proj", "down_proj")}
        skip = ("usage",)
    else:
        fp4 = {n + ".weight" for n, m in model.named_modules() if isinstance(m, QLinear) and m.mode == "nvfp4"}
        fp8 = {n + ".weight" for n, m in model.named_modules() if isinstance(m, QLinear) and m.mode == "mxfp8"}
        skip = ("head.weight",)  # tied to emb.weight
    for name, w in model.state_dict().items():
        if name in fp4:
            want = quantize_expert_weight(w, "nvfp4") if w.dim() == 3 else model.get_submodule(name[:-7]).quantized_weight().detach()
            out[name], b = pack_nvfp4(w, want)
            nbytes += b
            n_fp4 += w.numel()
        elif name in fp8:
            want = quantize_expert_weight(w, "mxfp8") if w.dim() == 3 else model.get_submodule(name[:-7]).quantized_weight().detach()
            out[name], b = pack_mxfp8(w, want)
            nbytes += b
            n_fp8 += w.numel()
        elif not name.endswith(skip):
            out[name] = w.to(torch.bfloat16)
            n_other += w.numel()
            nbytes += w.numel() * 2
    torch.save({"config": ck["config"], "arch": ck.get("arch", "dense"), "tensors": out}, dst)
    total = n_fp4 + n_fp8 + n_other
    print(f"{n_fp4 / total:.1%} NVFP4, {n_fp8 / total:.1%} MXFP8 of {total / 1e6:.2f}M weights")
    print(f"payload {nbytes / 1e6:.2f} MB = {nbytes * 8 / total:.2f} bits/weight  (FP32 checkpoint: {total * 4 / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
