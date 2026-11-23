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

PRESETS = {**INKLING_PRESETS, **{f"dense-{k}": v for k, v in DENSE_PRESETS.items()}}


def build(cfg):
    return Inkling(cfg) if isinstance(cfg, InklingConfig) else LoopedTransformer(cfg)


def load_checkpoint(path, map_location="cpu"):
    """Returns (model, checkpoint dict). Checkpoints written before the Inkling model have no "arch"."""
    ck = torch.load(path, map_location=map_location)
    cfg = InklingConfig(**ck["config"]) if ck.get("arch") == "inkling" else Config(**ck["config"])
    model = build(cfg)
    model.load_state_dict(ck["model"])
    return model, ck
