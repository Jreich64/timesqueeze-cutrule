"""Scoring a forecaster on a benchmark (zero-shot) and fine-tuning it on the benchmark's train split (full-shot).

Zero-shot (paper section 4.1): the pretrained model, never trained on the benchmark, forecasts every test window of every
channel; MSE / MAE per horizon in the benchmark's scaled units (see datasets.py).
Full-shot (paper section 4.2): the same model first fine-tuned for one epoch at lr 1e-4 on the train split's windows,
then scored the same way.

One rollout to the longest horizon serves all horizons: the forecast for horizon H is the first H points of it (a
rollout does not depend on how far it will continue). Each horizon is scored on its own window set.
"""
import os
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..model.forecaster import HORIZONS
from ..patching.batching import collate_patches
from .rolling import rolling_forecast


def evaluate_benchmark(model, data, patcher, device, context=512, horizons=(96, 192, 336, 720), stride=1, batch_size=128,
                       split="test", max_windows=None, log=print, state_path=None, save_every=60):
  """MSE / MAE per horizon over every (window, channel) of `split`. Returns {H: {"mse", "mae", "n"}} and, for the
  figure, a few example forecasts.

  With state_path, the running sums are saved there about every save_every seconds and a later call with the same
  state_path continues after the series already scored (the caller removes the file once the scores are kept)."""
  H_max = max(horizons)
  starts = data.window_starts(split, context, min(horizons), stride)
  if max_windows is not None and len(starts) > max_windows:
    starts = starts[np.linspace(0, len(starts) - 1, max_windows).astype(int)]
  last_ok = {H: data.splits[split][1] - context - H for H in horizons}      # last start whose H targets are in the split
  rows, C = data.values.shape
  se = {H: 0.0 for H in horizons}; ae = {H: 0.0 for H in horizons}; n = {H: 0 for H in horizons}
  jobs = [(s, c) for s in starts for c in range(C)]                          # every window of every channel
  examples, first, saved_at = [], 0, time.time()
  if state_path and os.path.exists(state_path):
    st = torch.load(state_path, weights_only=False)
    if st["n_jobs"] == len(jobs):
      first, se, ae, n, examples = st["next"], st["se"], st["ae"], st["n"], st["examples"]
      log and log(f"  {data.name} {split}: continuing after {first}/{len(jobs)} series scored before an interruption")
  for k in range(first, len(jobs), batch_size):
    chunk = jobs[k:k + batch_size]
    ctx = np.stack([data.values[s:s + context, c] for s, c in chunk])
    pred = rolling_forecast(model, ctx, H_max, patcher, device)
    for (s, c), p in zip(chunk, pred):
      avail = min(H_max, rows - s - context)
      truth = data.values[s + context:s + context + avail, c]
      for H in horizons:
        if s <= last_ok[H]:
          err = p[:H] - truth[:H]
          se[H] += float((err ** 2).mean()); ae[H] += float(np.abs(err).mean()); n[H] += 1
      if len(examples) < 3 and c == C - 1 and s <= last_ok[H_max]:
        examples.append({"context": ctx[chunk.index((s, c))], "truth": truth, "pred": p, "channel": data.columns[c]})
    if log and (k // batch_size) % 50 == 0:
      log(f"  {data.name} {split}: {min(k + batch_size, len(jobs))}/{len(jobs)} series forecast")
    if state_path and time.time() - saved_at > save_every:
      torch.save({"n_jobs": len(jobs), "next": k + batch_size, "se": se, "ae": ae, "n": n, "examples": examples},
                 state_path + ".tmp")
      os.replace(state_path + ".tmp", state_path)                          # a crash mid-write keeps the old state
      saved_at = time.time()
  return {H: {"mse": se[H] / max(n[H], 1), "mae": ae[H] / max(n[H], 1), "n": n[H]} for H in horizons}, examples


class BenchmarkTrainWindows(Dataset):
  """Training windows of the train split for full-shot fine-tuning: (start, channel) pairs, normalised by context stats
  and patched with the pretraining rule, in the same sample format as the ORCL / Time-300B windows."""

  def __init__(self, data, patcher, context=512, stride=1, split="train"):
    self.values, self.patcher, self.context, self.max_h = data.values, patcher, context, max(HORIZONS)
    starts = data.window_starts(split, context, self.max_h, stride)
    self.jobs = [(s, c) for s in starts for c in range(data.values.shape[1])]

  def __len__(self):
    return len(self.jobs)

  def __getitem__(self, i):
    s, c = self.jobs[i]
    raw = self.values[s:s + self.context + self.max_h, c].astype(np.float64)
    ctx = raw[:self.context]
    sd = ctx.std() if ctx.std() > 1e-12 else 1.0
    y = ((raw - ctx.mean()) / sd).astype(np.float32)
    return {"y": y, "n_valid": len(y), "bounds": self.patcher(y[:self.context])}


def collate_benchmark(samples, context):
  y = torch.tensor(np.stack([s["y"] for s in samples]))
  pos, mask, ids = collate_patches([s["bounds"] for s in samples], context)
  return {"y": y, "n_valid": torch.tensor([s["n_valid"] for s in samples]), "x": y[:, :context],
          "patch_pos": torch.tensor(pos), "patch_mask": torch.tensor(mask), "patch_id": torch.tensor(ids)}


def benchmark_train_loader(data, patcher, context, batch_size, stride=1, num_workers=0, seed=0, skip_batches=0):
  """Shuffled training windows (the same order for the same seed). skip_batches leaves out the first batches of that
  order, to continue an epoch that was interrupted."""
  ds = BenchmarkTrainWindows(data, patcher, context, stride)
  g = torch.Generator().manual_seed(seed)
  kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=partial(collate_benchmark, context=context),
            persistent_workers=num_workers > 0)
  return DataLoader(ds, sampler=torch.randperm(len(ds), generator=g).tolist()[skip_batches * batch_size:], **kw)
