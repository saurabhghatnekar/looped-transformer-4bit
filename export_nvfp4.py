"""Export a trained checkpoint with its NVFP4 weights stored as real 4-bit codes.

    python export_nvfp4.py runs/ink-smoke/ckpt.pt runs/ink-smoke/model-nvfp4.pt

NVFP4 weights become packed nibbles + one E4M3 scale per 16 values + one FP32 scale per
tensor (per expert for the Inkling presets). Dequantizing them reproduces exactly the weights
the forward pass trained against, so there is no post-training quantization step and no
conversion loss. MXFP8 layers and everything else are stored in BF16 here (an MXFP8 packer
would shrink the tail further). The container is this project's own, not the released
Inkling 2 Mini safetensors layout.
"""

import sys

import torch

from loopfp4 import Inkling, load_checkpoint
from loopfp4.nvfp4 import QLinear, nvfp4_pack, nvfp4_unpack, quantize_expert_weight


def pack(w, want):
    p = nvfp4_pack(w)
    assert torch.equal(nvfp4_unpack(p), want), "packed weights do not reproduce the training weights"
    p["s_block_e4m3"] = p.pop("s_block")  # held as float32 here; one byte each in a real container
    return p, p["codes"].numel() + p["s_block_e4m3"].numel() + 4


def main(src, dst):
    model, ck = load_checkpoint(src)
    out, n_fp4, n_other, nbytes = {}, 0, 0, 0
    if isinstance(model, Inkling):
        fp4 = {f"layers.{i}.mlp.experts.{k}" for i, layer in enumerate(model.layers) if layer.mlp.mode == "nvfp4"
               for k in ("gate_up_proj", "down_proj")}
        skip = ("usage",)
    else:
        fp4 = {n + ".weight" for n, m in model.named_modules() if isinstance(m, QLinear) and m.mode == "nvfp4"}
        skip = ("head.weight",)  # tied to emb.weight
    for name, w in model.state_dict().items():
        if name in fp4 and w.dim() == 3:  # stacked experts: one outer scale each
            want = quantize_expert_weight(w, "nvfp4")
            packed = [pack(w[e], want[e]) for e in range(w.shape[0])]
            out[name] = [p for p, _ in packed]
            nbytes += sum(b for _, b in packed)
            n_fp4 += w.numel()
        elif name in fp4:
            out[name], b = pack(w, model.get_submodule(name[:-7]).quantized_weight().detach())
            nbytes += b
            n_fp4 += w.numel()
        elif not name.endswith(skip):
            out[name] = w.to(torch.bfloat16)
            n_other += w.numel()
            nbytes += w.numel() * 2
    torch.save({"config": ck["config"], "arch": ck.get("arch", "dense"), "tensors": out}, dst)
    total = n_fp4 + n_other
    print(f"{n_fp4 / total:.1%} of {total / 1e6:.2f}M weights packed as NVFP4")
    print(f"payload {nbytes / 1e6:.2f} MB = {nbytes * 8 / total:.2f} bits/weight  (FP32 checkpoint: {total * 4 / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
