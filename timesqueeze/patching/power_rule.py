"""The paper's patch rule ("relative deviation"), plus the tick-timing features shared by the patch rules.

A new patch starts at tick i when the jump |x_i - x_{i-1}| is large relative to the recent signal power
(|jump| > tau * sqrt(mean(x^2) over the last L ticks)), or when the current patch is already max_patch_length long. Each
patch is later represented by the encoder state at its first tick, so boundaries should mark where something changes.

Time-aware variant (pass `t`, the elapsed time of each tick): the power is averaged over the last L *time units* and a patch
is also closed once it spans max_patch_duration time units; with evenly spaced ticks it equals the paper's rule exactly.
The sequential part (walking patch by patch) is compiled in kernels.power_scan.
"""
import numpy as np

from .kernels import power_scan


def elapsed_time(dtu):
  """Elapsed time of each tick since the first tick of the window, in tick units (arange for regular spacing)."""
  dtu = np.asarray(dtu, dtype=np.float64)
  return np.cumsum(dtu) - dtu[0]


def time_feature(dtu):
  """Model input channel: log(1 + tick interval in tick units). Monotone, exactly 0 where ticks share a
  timestamp, and nothing is clipped, so the real gaps (sub-millisecond to hours) are all represented."""
  return np.log1p(np.asarray(dtu, dtype=np.float64)).astype(np.float32)


def power_rule_boundaries(x, L=32, tau=0.3, max_patch_length=8, eps=1e-12, t=None, max_patch_duration=None):
  """Start index of every patch of 1-D array `x`.

  A boundary is placed at i when |x_i - x_{i-1}| > tau * sqrt(power), or when the current patch already has
  max_patch_length ticks. Without `t` this is the paper's rule (power = mean(x[i-L:i]^2)) and gives the same result
  as RelativeDeviationPatch.calc_boundaries in tests/reference_patch.py.

  With `t` (elapsed time per tick in tick units, see elapsed_time) the rule becomes time-aware: the power is
  averaged over the ticks of the last L time units (at least the previous tick), and a patch is also closed once it
  spans max_patch_duration time units (default max_patch_length). For regular spacing (t = arange) both agree.
  """
  x = np.asarray(x, dtype=np.float64)
  T = len(x)
  if T < 2:
    return np.zeros(1, dtype=np.int64)
  csum = np.concatenate([[0.0], np.cumsum(x * x)])
  i = np.arange(1, T)
  if t is None:
    lo = np.maximum(i - L, 0)
  else:
    t = np.ascontiguousarray(t, dtype=np.float64)
    lo = np.minimum(np.searchsorted(t, t[i] - L, side="left"), i - 1)
  power = (csum[i] - csum[lo]) / (i - lo)
  cand = np.nonzero(np.abs(x[1:] - x[:-1]) > tau * np.sqrt(power + eps))[0] + 1
  out = np.empty(T, dtype=np.int64)
  n = power_scan(cand, np.zeros(1) if t is None else t, t is not None, T, max_patch_length,
                 float(max_patch_length if max_patch_duration is None else max_patch_duration), out)
  return out[:n].copy()
