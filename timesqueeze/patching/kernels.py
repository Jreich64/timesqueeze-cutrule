"""Compiled (numba) inner loops of the patch rules.

The cut decision of the CutRule is sequential (each bar depends on where the last one ended), the time-aware EWMA is
a recurrence with a time-varying coefficient, and the power rule walks patch by patch. In plain Python these loops were
the bottleneck of data loading and of every bilevel epoch; compiled they run at C speed. Results are the same as the
previous numpy implementations (same float32 MLP evaluation, same decisions), which the tests check.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _cut_scan(r2_cum, slow, fast, forced, W1, b1, W2, b2, c_sigma, c_n, max_len, sample, temp, u, feats, cuts,
              seg_start, seg_end, seg_free):
  """Walk the ticks once, deciding after each one whether the current bar ends.

  Features at tick t (bar began at s): [sqrt(sum r^2 over s..t)/c_sigma, (ticks s..t)/c_n, slow[t], fast[t]];
  a 4 -> H (tanh) -> 1 MLP gives the logit. Deterministic (sample=False): cut iff logit >= 0. Sampled: cut with
  probability sigmoid(logit/temp) against the pre-drawn uniforms u, recording features and decisions. forced[t] (end of
  a trading day) and a bar reaching max_len ticks end the bar regardless; at a max_len tick the sampled decision did not
  matter, so cuts[t] is set to -1 there (left out of the log-probability). Returns the number of bars; bar k spans
  seg_start[k]..seg_end[k], and seg_free[k] is 1 if the rule itself cut it (not a day end or the length cap)."""
  T = r2_cum.shape[0] - 1
  H = b1.shape[0]
  n = 0
  s = 0
  for t in range(T):
    d2 = r2_cum[t + 1] - r2_cum[s]
    if d2 < 0.0:
      d2 = 0.0
    x0 = np.float32(math.sqrt(d2) / c_sigma)
    x1 = np.float32((t - s + 1) / c_n)
    x2 = np.float32(slow[t])
    x3 = np.float32(fast[t])
    logit = np.float32(b2)
    for j in range(H):
      a = x0 * W1[0, j] + x1 * W1[1, j] + x2 * W1[2, j] + x3 * W1[3, j] + b1[j]
      logit += np.float32(math.tanh(a)) * W2[j]
    capped = max_len > 0 and t - s + 1 >= max_len
    if sample:
      feats[t, 0] = x0
      feats[t, 1] = x1
      feats[t, 2] = x2
      feats[t, 3] = x3
      z = logit / temp
      if z > 50.0:
        z = 50.0
      elif z < -50.0:
        z = -50.0
      cut = u[t] < 1.0 / (1.0 + math.exp(-z))
      if capped:
        cuts[t] = -1.0
      elif cut:
        cuts[t] = 1.0
    else:
      cut = logit >= 0.0
    if cut or forced[t] or capped:
      seg_start[n] = s
      seg_end[n] = t
      seg_free[n] = 1 if cut else 0
      n += 1
      s = t + 1
  return n


def cut_scan(r2_cum, slow, fast, forced, W1, b1, W2, b2, c_sigma, c_n, max_len=0, sample=False, temp=1.0, u=None):
  """Python wrapper of _cut_scan. W1 is [4, H] (float32), b1 [H], W2 [H]; max_len=0 means no length cap. Returns
  (starts, ends, feats, cuts, free); feats and cuts are only filled when sample=True."""
  T = len(r2_cum) - 1
  starts = np.empty(T, dtype=np.int64)
  ends = np.empty(T, dtype=np.int64)
  free = np.empty(T, dtype=np.bool_)
  feats = np.zeros((T, 4), dtype=np.float32) if sample else np.zeros((1, 4), dtype=np.float32)
  cuts = np.zeros(T, dtype=np.float32) if sample else np.zeros(1, dtype=np.float32)
  u = u if u is not None else np.zeros(1)
  n = _cut_scan(r2_cum, np.ascontiguousarray(slow, dtype=np.float64), np.ascontiguousarray(fast, dtype=np.float64),
                forced, W1, b1, W2, np.float32(b2), c_sigma, c_n, int(max_len), sample, temp, u, feats, cuts, starts,
                ends, free)
  return starts[:n].copy(), ends[:n].copy(), feats, cuts, free[:n].copy()


@njit(cache=True)
def ewma_timeaware(r2, decay):
  """Weighted mean of r2 where every earlier observation is multiplied by `decay[i]` at tick i."""
  out = np.empty(r2.shape[0])
  S = 0.0
  W = 0.0
  for i in range(r2.shape[0]):
    S = decay[i] * S + r2[i]
    W = decay[i] * W + 1.0
    out[i] = S / W
  return out


@njit(cache=True)
def power_scan(cand, t, use_t, T, max_len, max_dur, out):
  """Patch starts of the power rule: the next boundary after `last` is the first candidate cut, capped at max_len
  ticks and (time-aware) at max_dur time units. cand is sorted. Returns the number of starts written to out."""
  n = 1
  out[0] = 0
  last = 0
  j = 0
  m = cand.shape[0]
  while True:
    while j < m and cand[j] <= last:
      j += 1
    nxt = cand[j] if j < m else T
    if last + max_len < nxt:
      nxt = last + max_len
    if use_t:
      k = np.searchsorted(t, t[last] + max_dur)
      if k < nxt:
        nxt = k
    if nxt >= T:
      break
    out[n] = nxt
    n += 1
    last = nxt
  return n


def warm_up():
  """Compile (or load from numba's on-disk cache) every kernel once, in this process, through the same call paths the
  training code uses, so the argument types match. Call it before starting dataloader workers: otherwise each worker
  compiles the kernels itself on first use and they race to write the same cache files. Takes about a second when cached.
  """
  from ..cutrule.model import CutRule  # imported here: those modules import this one
  from .power_rule import power_rule_boundaries
  from .segmenter import CutRuleSegmenter

  rng = np.random.default_rng(0)
  r = rng.standard_normal(64) * 0.01
  dtu = rng.exponential(1.0, 64)
  t = np.cumsum(dtu)
  power_rule_boundaries(r)
  power_rule_boundaries(r, t=t, max_patch_duration=8.0)
  seg = CutRuleSegmenter(CutRule(r, device="cpu"))
  for d in (None, dtu):
    seg.boundaries(r, dtu=d)
    seg.sample_boundaries(r, None, None, d, rng)
