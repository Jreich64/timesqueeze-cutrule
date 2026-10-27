"""Volatility estimates used as CutRule inputs: exponentially weighted moving averages of squared returns.

  * ewma_vol            - the classic tick-based EWMA (one tick = one step), vectorised with a linear filter.
  * ewma_vol_timeaware  - the weight of past observations decays with *elapsed time*, so a long gap forgets the past
                          while ticks that share a timestamp (gap 0) still count in full.
  * ewma_vol_by_day     - either of the two, restarted at the start of every trading day / series.
"""
import numpy as np
from scipy.signal import lfilter

from .kernels import ewma_timeaware


def ewma_vol(returns, span):
  """sqrt of the EWMA of squared returns; identical to the notebook's loop (tests/reference_patch.EWMAVol) but vectorised."""
  r2 = np.asarray(returns, dtype=np.float64) ** 2
  lam = 2.0 / (span + 1)
  y, _ = lfilter([lam], [1.0, -(1.0 - lam)], r2, zi=[(1.0 - lam) * r2[0]])
  return np.sqrt(y)


def ewma_vol_timeaware(returns, span, dtu):
  """Time-decayed volatility: weighted mean of squared returns where an observation's weight decays by
  (1 - 2/(span+1)) per tick unit of elapsed time, so long gaps forget the past while ticks that share a timestamp
  (dtu = 0) still count in full. For dtu == 1 it matches ewma_vol up to its start-up transient."""
  r2 = np.ascontiguousarray(returns, dtype=np.float64) ** 2
  keep = 1.0 - 2.0 / (span + 1)
  decay = keep ** np.ascontiguousarray(dtu, dtype=np.float64)
  return np.sqrt(ewma_timeaware(r2, decay))


def ewma_vol_by_day(returns, span, day_starts, dtu=None):
  """ewma_vol (or ewma_vol_timeaware when dtu is given) restarted at every trading day."""
  returns = np.asarray(returns, dtype=np.float64)
  bounds = list(day_starts) + [len(returns)]
  parts = []
  for a, b in zip(bounds[:-1], bounds[1:]):
    parts.append(ewma_vol(returns[a:b], span) if dtu is None else ewma_vol_timeaware(returns[a:b], span, dtu[a:b]))
  return np.concatenate(parts)
