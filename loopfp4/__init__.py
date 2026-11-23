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


def get_state(model) -> dict:
    """State dict for a checkpoint. The same layout whichever backend ran the expert GEMMs."""
    return model.canonical_state_dict() if isinstance(model, Inkling) else model.state_dict()


def set_state(model, state: dict):
    model.load_canonical(state) if isinstance(model, Inkling) else model.load_state_dict(state)


def load_checkpoint(path, map_location="cpu"):
    """Returns (model, checkpoint dict), with the model on the portable "sim" backend.
    Checkpoints written before the Inkling model have no "arch"."""
    ck = torch.load(path, map_location=map_location)
    if ck.get("arch") == "inkling":
        cfg = InklingConfig(**{**ck["config"], "backend": "sim"})
    else:
        cfg = Config(**{**ck["config"], "backend": "sim"})
    model = build(cfg)
    set_state(model, ck["model"])
    return model, ck
