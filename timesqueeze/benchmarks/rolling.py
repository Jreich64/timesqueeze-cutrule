"""Forecasting further than the longest head (64 steps) by feeding forecasts back in ("rolling" / autoregressive).

The paper forecasts 96 to 720 steps with heads of {1, 8, 32, 64} and "a simple scheduling strategy" (section 3.1.4, as in
Time-MoE): at each roll, use the longest head that does not overshoot the steps still needed, append its forecast to
the series, and forecast again from the new end. 720 steps = 11 rolls of the 64-head, then two of the 8-head.

Details that matter for the numbers:
  * normalisation - each series is z-scored once with the mean / std of its *original* context (as in training), and the
    whole rollout stays in those units; it is undone at the end. Forecasts are never used to re-estimate the scale.
  * steps -> levels - the heads forecast scaled one-step changes; each roll turns them into levels as the last value
    of the current window plus their running sum, times the original context's RMS change (step_targets).
  * patching - boundaries are recomputed on every roll for the current last `context` points (forecasts included),
    with the same rule the model was pretrained with.
  * every forecast comes from the last position, where the whole input is in the past.
"""
import numpy as np
import torch

from ..model.forecaster import run_model
from ..patching.batching import collate_patches


def _head_for(remaining, horizons):
  """The longest head that does not overshoot `remaining` steps (or the shortest head if all do)."""
  fitting = [h for h in horizons if h <= remaining]
  return max(fitting) if fitting else min(horizons)


@torch.no_grad()
def rolling_forecast(model, contexts, steps, patcher, device):
  """Forecast `steps` points after each row of `contexts` [B, T] (any units). Returns [B, steps] in the same units.

  patcher(z) -> patch start indices of one normalised window z [T] (the rule the model was trained with)."""
  x = np.asarray(contexts, dtype=np.float64)
  B, T = x.shape
  mean, std = x.mean(1, keepdims=True), x.std(1, keepdims=True)
  std = np.where(std > 1e-12, std, 1.0)
  z = ((x - mean) / std).astype(np.float32)
  step_scale = np.sqrt((np.diff(z, axis=1) ** 2).mean(1)).clip(min=1e-6)           # as step_targets, context only
  horizons = sorted(model.horizons)
  out = np.empty((B, 0), dtype=np.float32)
  was_training = model.training
  model.eval()
  while out.shape[1] < steps:
    window = np.concatenate([z, out], axis=1)[:, -T:]
    pos, mask, ids = collate_patches([patcher(w) for w in window], T)
    xb = torch.from_numpy(np.ascontiguousarray(window)).to(device)
    batch = {"x": xb, "patch_pos": torch.from_numpy(pos).to(device), "patch_mask": torch.from_numpy(mask).to(device),
             "patch_id": torch.from_numpy(ids).to(device)}
    h = _head_for(steps - out.shape[1], horizons)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
      preds, _ = run_model(model, batch)
    steps_h = preds[h][:, -1].float().cpu().numpy()
    level = window[:, -1:] + step_scale[:, None] * np.cumsum(steps_h, axis=1)      # levels_from_steps
    out = np.concatenate([out, level[:, :steps - out.shape[1]].astype(np.float32)], axis=1)
  model.train(was_training)
  return out * std + mean
