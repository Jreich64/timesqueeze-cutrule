"""Time-300B pretraining windows.

Windows are drawn inside a single series (never across a series boundary), with probability proportional to
series length, as in Time-MoE's sliding-window dataset. Each sample is `context` input points plus up to
max(horizons) extra points that serve as forecast targets. Series shorter than that are right-padded and masked.
The series is z-normalized with the mean/std of its context only, so no future information leaks into the scale.
"""
import json
import os
import zlib
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..model.forecaster import HORIZONS
from ..patching.batching import collate_patches
from ..patching.power_rule import power_rule_boundaries
from ..utils.paths import DATA_DIR
from .time300b_segments import build_segment_index

SUBSET_PREFIX = "Time-300B-subset"   # data/Time-300B-subset, -subset2, -subset3, ... (one folder per download)


def local_subsets(data_dir=DATA_DIR):
  """Every downloaded Time-300B folder under data_dir (Time-300B-subset*, with an index.json), in order. Pretraining uses
  all of them, so a new download is picked up without code changes."""
  found = [f for f in Path(data_dir).glob(SUBSET_PREFIX + "*") if f.is_dir() and (f / "index.json").exists()]
  return tuple(f.name for f in sorted(found, key=lambda f: (len(f.name), f.name)))
MIN_SEGMENT = 32          # series shorter than this carry no useful forecasting signal
VAL_BIN_MOD = 20          # source .bin files whose crc32 % 20 == 0 (about 5%) are held out for validation


class Time300BSegments:
  """All usable (window file, start, length) series segments across the local subset folders.

  Held-out (validation) data is whole source .bin files: the downloader fetches runs of adjacent windows from one file,
  so holding out single windows would leave the neighbouring training windows continuing the same series. Memory:
  one path per window file plus an int per series piece (an array of 1.2M full paths took hundreds of MB)."""

  def __init__(self, data_dir=DATA_DIR, subsets=None):
    subsets = local_subsets(data_dir) if subsets is None else subsets
    window_paths, piece_window, starts, lengths, is_val = [], [], [], [], []
    for name in subsets:
      folder = Path(data_dir) / name
      if not (folder / "index.json").exists():
        continue
      entries = json.load(open(folder / "index.json"))
      names = np.array([e["file"] for e in entries])                                      # one entry per window file
      win, start, length = build_segment_index(folder)                                   # one entry per series piece
      # Everything below is vectorised over windows/pieces. (It used to call Path.exists() once per piece, ~1.2M stat
      # calls in one thread, which took minutes across the WSL <-> Windows file boundary.)
      present = np.isin(names, os.listdir(folder))                                       # one directory listing
      keep = (length >= MIN_SEGMENT) & present[win]
      val_window = np.array([zlib.crc32(e["source"].encode()) % VAL_BIN_MOD == 0 for e in entries])
      if not val_window.any():
        val_window[::VAL_BIN_MOD] = True
      base = len(window_paths)
      window_paths.extend(str(folder / n) for n in names)
      piece_window.append(win[keep] + base)
      starts.append(start[keep])
      lengths.append(length[keep])
      is_val.append(val_window[win[keep]])
    if not window_paths:
      raise FileNotFoundError(f"no Time-300B windows found under {data_dir} ({', '.join(subsets) or SUBSET_PREFIX + '*'})")
    self.window_paths = np.array(window_paths)
    self.piece_window = np.concatenate(piece_window).astype(np.int32)
    self.starts = np.concatenate(starts).astype(np.int64)
    self.lengths = np.concatenate(lengths).astype(np.int64)
    self.is_val = np.concatenate(is_val)

  def path(self, i):
    return self.window_paths[self.piece_window[i]]

  @property
  def files(self):
    """Window file of every piece (built on demand; the loaders use path(i))."""
    return self.window_paths[self.piece_window]


class Time300BWindows(IterableDataset):
  """Infinite stream of training samples (or a fixed list when split='val')."""

  def __init__(self, segments, context=2048, horizons=HORIZONS, split="train", seed=0, patcher=power_rule_boundaries,
               val_samples=256):
    self.seg, self.context, self.max_h = segments, context, max(horizons)
    self.split, self.seed, self.patcher, self.val_samples = split, seed, patcher, val_samples
    keep = segments.is_val if split == "val" else ~segments.is_val
    self.ids = np.nonzero(keep)[0]
    self.cum = np.cumsum(segments.lengths[self.ids])
    self._maps = OrderedDict()

  def _load(self, path):
    m = self._maps.get(path)
    if m is None:
      if len(self._maps) >= 64:
        self._maps.popitem(last=False)
      m = self._maps[path] = np.load(path, mmap_mode="r")
    else:
      self._maps.move_to_end(path)
    return m

  def make_sample(self, y, n, n_in, rng):
    """Turn a drawn, normalized window into a sample dict (override to attach more, e.g. sampled patchings)."""
    return {"y": y, "n_valid": n, "bounds": self.patcher(y[:n_in])}

  def sample(self, rng):
    while True:
      i = self.ids[np.searchsorted(self.cum, rng.integers(self.cum[-1]), side="right")]
      L = int(self.seg.lengths[i])
      lo = int(rng.integers(0, max(L - self.context, 0) + 1))
      n = min(L - lo, self.context + self.max_h)
      a = self.seg.starts[i] + lo
      raw = np.array(self._load(self.seg.path(i))[a:a + n], dtype=np.float64)
      n_in = min(self.context, n)
      ctx = raw[:n_in]
      if not np.isfinite(raw).all() or ctx.std() < 1e-8:
        continue                                  # skip NaN/constant windows
      y = np.zeros(self.context + self.max_h, dtype=np.float32)
      y[:n] = (raw - ctx.mean()) / ctx.std()
      return self.make_sample(y, n, n_in, rng)

  def __iter__(self):
    info = get_worker_info()
    wid = info.id if info else 0
    if self.split == "val":
      rng = np.random.default_rng(12345)          # fixed validation set, same in every worker
      for k in range(self.val_samples):
        s = self.sample(rng)
        if k % (info.num_workers if info else 1) == wid:
          yield s
      return
    rng = np.random.default_rng([self.seed, wid])
    while True:
      yield self.sample(rng)


def collate(samples, context):  # module-level so it pickles for Windows dataloader workers
  y = torch.tensor(np.stack([s["y"] for s in samples]))
  n_valid = torch.tensor([s["n_valid"] for s in samples])
  pos, mask, ids = collate_patches([s["bounds"] for s in samples], context)
  return {"y": y, "n_valid": n_valid, "x": y[:, :context], "patch_pos": torch.tensor(pos),
          "patch_mask": torch.tensor(mask), "patch_id": torch.tensor(ids)}


def make_loader(dataset, batch_size, num_workers=2):
  ctx = dataset.context
  return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=partial(collate, context=ctx),
                    persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
                    pin_memory=torch.cuda.is_available())
