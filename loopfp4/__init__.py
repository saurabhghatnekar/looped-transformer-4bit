"""Two models share this package.

inkling.py  the Inkling 2 Mini text decoder (mixture of experts, attention residuals, looping),
            checked against the published reference implementation. Presets: smoke, 5b,
            5b-pretrain, mini, mini-pretrain.
model.py    the earlier simplified dense looped transformer. Presets: dense-smoke, dense-1b, dense-5b.
"""

import torch

from .inkling import Inkling, InklingConfig, param_counts  # noqa: F401
from .inkling import PRESETS as INKLING_PRESETS
from .model import Config, LoopedTransformer  # noqa: F401
from .model import PRESETS as DENSE_PRESETS
from .nvfp4 import mxfp8_unpack, nvfp4_unpack

PRESETS = {**INKLING_PRESETS, **{f"dense-{k}": v for k, v in DENSE_PRESETS.items()}}


def build(cfg):
    return Inkling(cfg) if isinstance(cfg, InklingConfig) else LoopedTransformer(cfg)


def get_state(model) -> dict:
    """State dict for a checkpoint. The same layout whichever backend ran the expert GEMMs."""
    if isinstance(model, Inkling):
        return model.canonical_state_dict()
    return {k: v for k, v in model.state_dict().items() if not k.endswith("._extra_state")}


def set_state(model, state: dict):
    if isinstance(model, Inkling):
        return model.load_canonical(state)
    clean = {k: v for k, v in state.items() if not k.endswith("._extra_state")}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    missing = [k for k in missing if not k.endswith("._extra_state")]
    assert not missing and not unexpected, (missing, unexpected)


def load_checkpoint(path, map_location="cpu"):
    """Returns (model, checkpoint dict), with the model on the portable "sim" backend.
    Accepts both training checkpoints ("model") and 4-bit exported containers ("tensors")."""
    ck = torch.load(path, map_location=map_location)
    if ck.get("arch") == "inkling":
        cfg = InklingConfig(**{**ck["config"], "backend": "sim"})
    else:
        cfg = Config(**{**ck["config"], "backend": "sim"})
    model = build(cfg)
    if "tensors" in ck:
        sd = {}
        for k, v in ck["tensors"].items():
            if isinstance(v, list):
                sd[k] = torch.stack([nvfp4_unpack(p) for p in v])
            elif isinstance(v, dict) and v.get("format") == "mxfp8":
                sd[k] = mxfp8_unpack(v)
            elif isinstance(v, dict) and "codes" in v:
                sd[k] = nvfp4_unpack(v)
            else:
                sd[k] = v.float()
        if isinstance(model, Inkling):
            for k, b in model.named_buffers():
                if k.endswith(".mlp.usage"):
                    sd.setdefault(k, b)
        elif "emb.weight" in sd:
            sd.setdefault("head.weight", sd["emb.weight"])
        set_state(model, sd)
    else:
        set_state(model, ck["model"])
    return model, ck
