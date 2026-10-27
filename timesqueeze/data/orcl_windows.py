"""ORCL training windows for the forecaster: one sample = a fixed-length stretch of ticks inside one trading day.

The model sees the log price *path* of the window, p_j = r_1 + ... + r_j (cumulative log returns from the window's
start), a level series like the Time-300B pretraining data, and forecasts its next one-step changes, which are exactly
the next log returns (see forecaster.step_targets: changes scaled by the context's RMS log return).
A window is `context` input ticks followed by max(horizons) = 64 target ticks. It never crosses a day boundary or a
train / validation / test split, and it is z-normalised with the statistics of its context only (no future information in
the scale). Each sample also carries its patch boundaries, from the chosen patch rule:
  * "power"   - the paper's rule on the normalised context path,
  * "cutrule" - the trained CutRule on the raw log returns (its features are return volatilities).
With time_aware (the default) a sample also carries the real tick timing: `time` (elapsed time of each tick, used as the
rotary position of its patch) and `dtf` (log(1 + gap), an encoder input channel), and both rules use the timing too.
"""
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..model.forecaster import HORIZONS
from ..patching.batching import collate_patches
from ..patching.power_rule import elapsed_time, power_rule_boundaries, time_feature


class OrclWindows(Dataset):
  """Fixed-length windows [start, start + context + max_h) of the log price path, normalized by context stats.

  A window must lie inside [lo, hi) (one split) and inside a single trading day. With time_aware, each window also
  carries the real tick timing: `time` (elapsed time of each input tick, the rotary position of patches), `dtf`
  (log(1 + gap), an encoder input channel), and both patch rules use it. With time_aware=False ticks are treated
  as evenly spaced (the paper's setting).
  """

  def __init__(self, data, lo, hi, context, stride, patcher, horizons=HORIZONS, segmenter=None, time_aware=True, tau=0.3):
    self.r, self.dtu, self.context, self.max_h = data.returns, data.dtu, context, max(horizons)
    self.tau = tau                                    # power-rule threshold (the paper's 0.3; learned by power-bilevel)
    self.patcher, self.segmenter, self.time_aware = patcher, segmenter, time_aware
    if segmenter is not None:
      # features on the full series, restarted each day, as the CutRule was trained
      self.slow, self.fast = segmenter.ewma_features(self.r, data.day_starts, self.dtu if time_aware else None)
    day_of = np.searchsorted(data.day_starts, np.arange(len(self.r)), side="right")
    n = context + self.max_h
    self.starts = [s for s in range(lo, hi - n + 1, stride)
                   if day_of[s] == day_of[s + n - 1] and np.cumsum(self.r[s:s + context]).std() > 1e-12
                   and np.any(self.r[s + 1:s + context])]           # the path moves inside the context

  def __len__(self):
    return len(self.starts)

  def window(self, i):
    """Normalized window plus its timing; patching is added by __getitem__ (or sampled, in joint mode)."""
    s, T = self.starts[i], self.context
    raw = self.r[s:s + T + self.max_h]                                 # log returns (the CutRule's input)
    path = np.cumsum(raw)                                              # log price relative to the window start
    ctx = path[:T]
    y = ((path - ctx.mean()) / ctx.std()).astype(np.float32)
    dtu = self.dtu[s:s + T] if self.time_aware else None
    t = elapsed_time(dtu) if self.time_aware else None
    out = {"y": y, "n_valid": len(y), "n_in": T}
    if self.time_aware:
      out["time"], out["dtf"] = t.astype(np.float32), time_feature(dtu)
    return out, raw[:T], dtu, t, s

  def __getitem__(self, i):
    out, raw_ctx, dtu, t, s = self.window(i)
    T = self.context
    if self.patcher == "cutrule":
      out["bounds"] = self.segmenter.boundaries(raw_ctx, self.slow[s:s + T], self.fast[s:s + T], dtu)
    else:
      out["bounds"] = power_rule_boundaries(out["y"][:T], tau=self.tau, t=t)
    return out


def collate(samples, context):
  y = torch.tensor(np.stack([s["y"] for s in samples]))
  pos, mask, ids = collate_patches([s["bounds"] for s in samples], context)
  batch = {"y": y, "n_valid": torch.tensor([s["n_valid"] for s in samples]), "x": y[:, :context],
           "patch_pos": torch.tensor(pos), "patch_mask": torch.tensor(mask), "patch_id": torch.tensor(ids)}
  if "time" in samples[0]:
    batch["time"] = torch.tensor(np.stack([s["time"] for s in samples]))
    batch["dtf"] = torch.tensor(np.stack([s["dtf"] for s in samples]))
  return batch


def make_loader(ds, batch_size, shuffle):
  return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(collate, context=ds.context))
