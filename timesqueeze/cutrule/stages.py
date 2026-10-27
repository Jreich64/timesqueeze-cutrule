"""Pretraining and ORCL training of the bilevel CutRule, in the same two stages as the backbone.

  1. pretrain_cutrule(): train the CutRule on Time-300B (the same series the backbone is pretrained on).
  2. train_orcl_cutrule(): continue training that CutRule on the ORCL training days, with real tick timing.

Time-300B series are values on arbitrary scales, not returns, and are regularly sampled. For the CutRule each series
is turned into "returns" = its increments, divided by their standard deviation and multiplied by `return_scale`
(the std of the ORCL training returns), so the CutRule's volatility thresholds see the scale they will see on
ORCL. Every series is its own "day": volatility estimates restart and a bar is closed at its end. Durations are tick
counts here and in the ORCL stage (bars of ~4 ticks, at most 8; see model.py).

CutRulePatcher applies a trained CutRule to a normalized Time-300B window with the same transform, so the backbone
can be pretrained on the patches the CutRule will produce.

Both spacing variants share stage 1 (and the backbones of stage 2): Time-300B has no timestamps - its files hold only
values, with each series evenly spaced at its dataset's sampling rate - so "real timing" and "evenly spaced" are the same
thing there. The CutRule is pretrained once with dt = 1 per tick (outputs/cutrule_pretrained.pt); the variants part ways
at the ORCL stage, where the time-aware variant's EWMA volatility features use the real ts_event gaps and the
evenly-spaced one's do not. Bar durations are tick counts in both.
"""
import hashlib
import os

import numpy as np
import torch

from ..data.orcl import load_orcl, returns_fingerprint
from ..data.time300b_windows import Time300BSegments, Time300BWindows
from ..patching.segmenter import CutRuleSegmenter
from ..utils.paths import DATA_DIR, out, variant_suffix
from ..utils.seed import ignore_deprecation_warnings, set_seed
from .bilevel import train
from .model import CutRule


def pretrained_cutrule_path():
  """Stage 1 is shared by both spacing variants (see the module docstring)."""
  return out("cutrule_pretrained.pt")


def orcl_cutrule_path(time_aware=True):
  return out(f"checkpoint{variant_suffix(time_aware)}.pt")


MIN_SERIES = 256
MIN_EPOCHS_TO_CHECK = 10     # shorter runs (quick tests, smoke runs) may legitimately end before the rule has learned to cut


def require_free_cuts(model, epochs, what):
  """A finished CutRule must cut on its own somewhere: the forced day/series-end cuts alone make every patch a whole
  day or series. Checked for real runs only (epochs >= MIN_EPOCHS_TO_CHECK); raises so callers can retry another seed."""
  if epochs >= MIN_EPOCHS_TO_CHECK and not model.has_free_cuts():
    raise RuntimeError(f"{what}: the CutRule finished {epochs} epochs without cutting on its own (only the forced "
                       f"day/series-end cuts), so every patch would be a whole day or series. Try another seed.")


def increments_as_returns(values, return_scale):
  """Per-tick 'returns' of a series: increments scaled to std `return_scale`, with 0 for the first tick."""
  d = np.diff(np.asarray(values, dtype=np.float64))
  s = d.std() if len(d) else 0.0
  scaled = d / s * return_scale if s > 1e-12 else np.zeros_like(d)
  return np.concatenate([[0.0], scaled])


class CutRulePatcher:
  """Patch boundaries of a Time-300B window from a trained CutRule (picklable for dataloader workers)."""

  def __init__(self, model, return_scale):
    self.segmenter = CutRuleSegmenter(model)
    self.return_scale = return_scale
    self.fingerprint = None                 # set by get_cutrule_patcher: which pretrained CutRule this is

  def __call__(self, y):
    r = increments_as_returns(y, self.return_scale)
    return self.segmenter.boundaries(r, None, None, np.ones(len(r)))       # Time-300B is regular: dt = 1 per tick


def state_fingerprint(state):
  """Hash of a CutRule state dict, so anything trained from it records exactly which pretrained model it began as."""
  h = hashlib.md5()
  for k in sorted(state):
    h.update(k.encode())
    h.update(state[k].detach().cpu().numpy().tobytes())
  return h.hexdigest()[:16]


def build_corpus(segments, n_segments, segment_len, return_scale, seed):
  """Concatenate `n_segments` random Time-300B series pieces into one 'returns' array plus the start of each."""
  ds = Time300BWindows(segments, context=segment_len, horizons=(1,), split="train", seed=seed,
                       patcher=lambda y: np.zeros(1, dtype=np.int64))
  rng = np.random.default_rng(seed)
  parts, starts, total = [], [], 0
  while len(parts) < n_segments:
    s = ds.sample(rng)
    n = min(s["n_valid"], segment_len)
    if n < MIN_SERIES:
      continue
    r = increments_as_returns(s["y"][:n], return_scale)[1:]          # drop the artificial first zero
    if not np.any(r):
      continue
    starts.append(total)
    parts.append(r)
    total += len(r)
  return np.concatenate(parts), np.array(starts, dtype=np.int64)


def empty_cutrule():
  return CutRule(np.ones(2), device="cpu")


def pretrain_cutrule(out_path=None, n_segments=800, segment_len=1024, epochs=60, seed=0,
                     data_dir=None, return_scale=None, max_attempts=5):
  """Train the bilevel CutRule on Time-300B (dt = 1 per tick) and save it. Returns (cpu CutRule, return_scale)."""
  ignore_deprecation_warnings()
  out_path = out_path or pretrained_cutrule_path()
  return_scale = float(return_scale or load_orcl().train_slice()[0].std())
  segments = Time300BSegments(data_dir or DATA_DIR)
  returns, day_starts = build_corpus(segments, n_segments, segment_len, return_scale, seed)
  print(f"pretraining CutRule on {len(returns)} points from {len(day_starts)} Time-300B series "
        f"(return scale {return_scale:.2e}, {epochs} epochs, tick-count durations)")
  for attempt in range(max_attempts):                              # some initialisations never cut; try the next
    set_seed(seed + attempt)
    try:
      model, trace = train(True, returns, epochs=epochs, day_starts=day_starts, dt=np.ones(len(returns)))
      require_free_cuts(model, epochs, "CutRule pretraining")
      break
    except RuntimeError as e:
      print(f"attempt {attempt + 1}: {e}")
  else:
    raise RuntimeError("CutRule pretraining collapsed on every seed")
  model = model.to("cpu")
  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  tmp = out_path + ".tmp"
  torch.save({"model": model.state_dict(), "return_scale": return_scale, "epochs": epochs, "n_points": len(returns),
              "n_series": len(day_starts), "final": trace[-1]}, tmp)
  os.replace(tmp, out_path)
  print(f"saved pretrained CutRule to {out_path}: {trace[-1]['bars']} bars over {len(returns)} points")
  return model, return_scale


def load_pretrained_cutrule(path=None):
  saved = torch.load(path or pretrained_cutrule_path(), map_location="cpu")
  model = empty_cutrule()
  model.load_state_dict(saved["model"])
  return model, saved["return_scale"], state_fingerprint(saved["model"])


def get_pretrained_cutrule(path=None, **pretrain_kwargs):
  """The CutRule pretrained on Time-300B: loaded from `path`, or pretrained first if it does not exist yet.
  Returns (cpu CutRule, return_scale, fingerprint)."""
  path = path or pretrained_cutrule_path()
  if not os.path.exists(path):
    print(f"no pretrained CutRule at {path}; pretraining one on Time-300B")
    pretrain_cutrule(out_path=path, **pretrain_kwargs)
  return load_pretrained_cutrule(path)


def get_cutrule_patcher(path=None, **pretrain_kwargs):
  model, scale, fp = get_pretrained_cutrule(path, **pretrain_kwargs)
  patcher = CutRulePatcher(model, scale)
  patcher.fingerprint = fp
  return patcher


def train_orcl_cutrule(data, time_aware=True, ckpt_path=None, epochs=30, from_scratch=False,
                       pretrained_path=None, resume=False, pretrain_kwargs=None):
  """Continue training the pretrained CutRule on the ORCL training days (or from scratch if from_scratch).

  The checkpoint is tagged with a fingerprint of the exact data, timing and starting weights."""
  ckpt_path = ckpt_path or orcl_cutrule_path(time_aware)
  pretrained_path = pretrained_path or pretrained_cutrule_path()
  returns, dtu, day_starts = data.train_slice()
  dtu = dtu if time_aware else None
  init_state, init_fp = None, "scratch"
  if not from_scratch:
    pre, _, init_fp = get_pretrained_cutrule(pretrained_path, **(pretrain_kwargs or {}))
    init_state = pre.state_dict()
  fp = returns_fingerprint(returns, dtu, extra=init_fp)
  print(f"training CutRule on {len(returns)} ORCL returns over {len(day_starts)} days "
        f"(tick-count durations, {'time-aware' if dtu is not None else 'evenly spaced'} volatility, "
        f"{'from scratch' if from_scratch else 'starting from the Time-300B pretrained CutRule'})")
  model, trace = train(True, returns, epochs=epochs, ckpt_path=ckpt_path, ckpt_every=10, resume=resume,
                       day_starts=day_starts, fingerprint=fp, dt=dtu, init_state=init_state)
  require_free_cuts(model, epochs, "ORCL CutRule training")
  return model, trace


def get_orcl_cutrule(data, time_aware=True, ckpt_path=None, epochs=30, from_scratch=False,
                     pretrained_path=None, pretrain_kwargs=None):
  """The CutRule to use on ORCL: loaded from ckpt_path if it was trained on exactly this data/timing/start, else
  trained now (pretrained on Time-300B first when needed). Returns a cpu CutRule."""
  ckpt_path = ckpt_path or orcl_cutrule_path(time_aware)
  pretrained_path = pretrained_path or pretrained_cutrule_path()
  returns, dtu, day_starts = data.train_slice()
  dtu = dtu if time_aware else None
  init_fp = "scratch" if from_scratch else get_pretrained_cutrule(pretrained_path, **(pretrain_kwargs or {}))[2]
  fp = returns_fingerprint(returns, dtu, extra=init_fp)
  if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("fingerprint") != fp:
      raise RuntimeError(f"{ckpt_path} was trained on different data, timing or starting weights than requested "
                         f"(time-aware={time_aware}, from_scratch={from_scratch}). Delete it, or run "
                         f"`python -m timesqueeze train-cutrule --full`, to retrain.")
    if ckpt.get("epoch", 0) < epochs:                                # an interrupted run: finish it, don't use it as is
      print(f"{ckpt_path} holds only {ckpt.get('epoch', 0)} of {epochs} epochs; continuing that run")
      model, _ = train_orcl_cutrule(data, time_aware, ckpt_path, epochs, from_scratch, pretrained_path, resume=True,
                                    pretrain_kwargs=pretrain_kwargs)
      return model.to("cpu")
    print(f"loading ORCL CutRule from {ckpt_path}")
    model = CutRule(returns, device="cpu", day_starts=day_starts, dt=dtu)
    model.load_state_dict(ckpt["model"])
    return model
  model, _ = train_orcl_cutrule(data, time_aware, ckpt_path, epochs, from_scratch, pretrained_path,
                                pretrain_kwargs=pretrain_kwargs)
  return model.to("cpu")


def main():
  import argparse

  from ..utils.paths import use_quick_outputs
  from ..utils.quick import CUTRULE_PRETRAIN, add_quick_flags, banner
  p = argparse.ArgumentParser(description="Pretrain the bilevel CutRule on Time-300B (shared by both spacing variants).")
  add_quick_flags(p)
  p.add_argument("--n-segments", type=int, default=800)
  p.add_argument("--segment-len", type=int, default=1024)
  p.add_argument("--epochs", type=int, default=60)
  p.add_argument("--out", default=None, help="default outputs/cutrule_pretrained.pt (outputs/quick/ in quick mode)")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  use_quick_outputs(a.quick)
  kw = dict(n_segments=a.n_segments, segment_len=a.segment_len, epochs=a.epochs)
  if a.quick:
    banner(True)
    kw.update(CUTRULE_PRETRAIN)
  pretrain_cutrule(a.out, seed=a.seed, **kw)


if __name__ == "__main__":
  main()
