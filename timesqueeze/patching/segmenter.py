"""CutRuleSegmenter: applies a trained CutRule to a window of returns to get patch boundaries.

It copies the CutRule's tiny MLP weights (4 -> 6 -> 1) out of PyTorch so it can run in dataloader worker processes, on
the CPU, with the compiled scan in kernels.py. Two ways to use it:
  * boundaries()        - deterministic: a bar ends where the logit is >= 0 (used for patching at train/test time),
  * sample_boundaries() - stochastic: a bar ends with probability sigmoid(logit / temp); also returns the features and
                          decisions so the log-probability can be recomputed with gradients (REINFORCE, methods 3 and 4).
"""
import numpy as np

from .kernels import cut_scan
from .volatility import ewma_vol_by_day


class CutRuleSegmenter:
  """Applies a trained CutRule to a window of returns to get patch starts (no gradients, CPU)."""

  def __init__(self, model):
    self.W1 = np.ascontiguousarray(model.trunk.weight.detach().cpu().numpy().T, dtype=np.float32)     # [4, H]
    self.b1 = model.trunk.bias.detach().cpu().numpy().astype(np.float32)
    self.W2 = model.W2.detach().cpu().numpy()[:, 0].astype(np.float32)
    self.b2 = float(model.b2.detach().cpu().numpy()[0])
    self.c_sigma, self.c_n, self.c_e = model.c_sigma, model.c_n, model.c_e
    self.max_len = model.max_len
    self.slow_span, self.fast_span = model.slow_span, model.fast_span
    self.temp = model.temp

  def ewma_features(self, returns, day_starts=None, dtu=None):
    day_starts = [0] if day_starts is None else day_starts
    return (ewma_vol_by_day(returns, self.slow_span, day_starts, dtu) / self.c_e,
            ewma_vol_by_day(returns, self.fast_span, day_starts, dtu) / self.c_e)

  def _scan(self, returns, ewma_slow, ewma_fast, dtu, sample=False, rng=None):
    r = np.ascontiguousarray(returns, dtype=np.float64)
    T = len(r)
    if ewma_slow is None:                         # same estimate the CutRule was trained with (timing included)
      ewma_slow, ewma_fast = self.ewma_features(r, None, dtu)
    r2_cum = np.concatenate([[0.0], np.cumsum(r * r)])
    return cut_scan(r2_cum, ewma_slow, ewma_fast, np.zeros(T, dtype=np.bool_), self.W1, self.b1, self.W2, self.b2,
                    self.c_sigma, self.c_n, self.max_len, sample, self.temp, rng.random(T) if sample else None)[:4]

  @staticmethod
  def _starts(ends, T):
    """A cut at t ends a segment, so the next patch starts at t + 1 (if there is one)."""
    nxt = ends + 1
    return np.concatenate([[0], nxt[nxt < T]]).astype(np.int64)

  def sample_boundaries(self, returns, ewma_slow, ewma_fast, dtu, rng):
    """Like boundaries() but each tick cuts with probability sigmoid(logit / temp) instead of when logit >= 0.

    Returns (starts, feats [T, 4], cuts [T]): the patch starts, the features the policy saw at every tick and the
    0/1 cut decision it took, so the log-probability of the sampled patching can be recomputed (with gradient) from
    feats and cuts for a REINFORCE update of the CutRule. cuts is -1 where the length cap ended the bar (no decision)."""
    _, ends, feats, cuts = self._scan(returns, ewma_slow, ewma_fast, dtu, sample=True, rng=rng)
    return self._starts(ends, len(returns)), feats, cuts

  def boundaries(self, returns, ewma_slow=None, ewma_fast=None, dtu=None):
    """Patch starts for `returns`.

    ewma_slow / ewma_fast can be passed pre-computed on the full series (as CutRule was trained) and sliced
    to the window; otherwise they are computed from the start of the window. dtu (tick intervals in tick units) only
    enters the time-aware EWMA volatility features: bar duration is always counted in ticks, and no bar is longer than
    the CutRule's max_len ticks.
    """
    _, ends, _, _ = self._scan(returns, ewma_slow, ewma_fast, dtu)
    return self._starts(ends, len(returns))
