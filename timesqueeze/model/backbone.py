"""The forecaster's weights on disk: build the model, save / load a backbone, and get-or-pretrain one.

A backbone is the forecaster after Time-300B pretraining (stage 2). Each patch rule has its own (the Transformer learned
on that rule's patches): outputs/backbone_<patcher>.pt. Both spacing variants share it - Time-300B has no timestamps, so
pretraining is identical for them; they differ only on ORCL (fine-tuning, and the ORCL CutRule). Every file carries
metadata (patch rule, CutRule fingerprint); get_backbone refuses a file written for another patch rule instead of
silently fine-tuning the wrong model, and pretrains one (training.pretrain.pretrain_backbone) when none exists yet.
"""
import os

import torch

from ..utils.paths import out
from .forecaster import HORIZONS, TimeSqueeze


def backbone_path(patcher):
  """Each patch rule gets its own backbone (the Transformer is pretrained on that rule's patches), shared by both
  spacing variants."""
  return out(f"backbone_{patcher}.pt")




def build_model(mamba_cls=None):
  """TimeSqueeze-base: 2+2 Mamba layers (d_model 384, d_state 128), 12-layer MoE Transformer, heads 1/8/32/64."""
  return TimeSqueeze(horizons=HORIZONS, mamba_cls=mamba_cls)


def save_backbone(path, model, extra=None):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  torch.save({"model": model.state_dict(), **(extra or {})}, tmp)
  os.replace(tmp, path)


def load_backbone(path, device=None, mamba_cls=None):
  """Build a TimeSqueeze and load pretrained weights from `path`. Returns (model, saved metadata)."""
  device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
  saved = torch.load(path, map_location="cpu")
  model = build_model(mamba_cls)
  model.load_state_dict(saved["model"])
  return model.to(device), {k: v for k, v in saved.items() if k != "model"}


def get_backbone(path=None, device=None, mamba_cls=None, patcher="power", **pretrain_kwargs):
  """Load the backbone pretrained with `patcher`'s patches (default path outputs/backbone_<patcher>.pt); if there
  is none, pretrain one on Time-300B first (pretraining the CutRule first when patcher="cutrule"), then load it.

  pretrain_kwargs go to pretrain.pretrain_backbone (steps, micro_batch, max_minutes, ...).
  """
  path = path or backbone_path(patcher)
  if not os.path.exists(path):
    from ..training.pretrain import pretrain_backbone
    print(f"no pretrained backbone at {path}; pretraining one on Time-300B with the {patcher} patcher")
    pretrain_backbone(out_path=path, mamba_cls=mamba_cls, patcher=patcher, **pretrain_kwargs)
  else:
    print(f"loading pretrained backbone from {path}")
  model, meta = load_backbone(path, device, mamba_cls)
  if meta.get("patcher", patcher) != patcher:                                   # never use another rule's backbone
    raise RuntimeError(f"{path} was pretrained with patcher={meta['patcher']!r}, but patcher={patcher!r} was requested")
  return model, meta
