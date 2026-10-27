"""ORCL trade data: loading, per-day returns, tick timing and the train / validation / test split.

What it produces
  OrclData: the per-trade return series of 43 trading days, the time since the previous trade for every return (raw ts_event
  gaps, never capped), the index where each day starts, and the split points (whole days: 30 train / 7 validation / 6 test).

Why it looks like this
  * Only the regular session (09:30-16:00 New York time) is used, and returns are computed *within* each day, so no
    return spans the overnight gap or the pre-/after-market hours.
  * Tick timing comes from the exchange timestamp `ts_event` (the DataFrame index is ts_recv, the receive time).
  * `dtu` (OrclData.dtu) is the same timing divided by one constant, the mean training-day gap, so that rule
    hyperparameters written in "ticks" keep their meaning. It is a unit change only.
  * returns_fingerprint identifies the exact series a CutRule was trained on, so a checkpoint fitted to other data (or other
    timing) is never loaded by mistake.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

from ..utils.paths import DATA_DIR

ORCL_FILE = "ORCL_trades_2024-07-02_to_2024-08-30.dbn.zst"
SESSION_TZ = "America/New_York"
SESSION_OPEN_MIN, SESSION_CLOSE_MIN = 9 * 60 + 30, 16 * 60     # regular session [09:30, 16:00) New York time


def load_trades(data_dir=DATA_DIR):
  """Return the ORCL trades DataFrame from the local data/ folder."""
  file_path = Path(data_dir) / ORCL_FILE
  if not file_path.exists():
    raise FileNotFoundError(f"{file_path} not found; place the ORCL .dbn.zst file in data/.")
  return db.DBNStore.from_file(str(file_path)).to_df()


def trade_returns_by_day(trades):
  """Per-trade returns within each trading day's regular session plus the time between ticks.

  Uses the exchange timestamp `ts_event` (not the index, which is ts_recv). Only trades in the regular session,
  09:30 to 16:00 New York time, are kept, as in the bilevel tokenisation paper (the file also holds pre-market trades
  from 04:00 and after-hours trades to 20:00, ~1.8%). Returns are log returns, log(p_i / p_{i-1}), between consecutive
  kept trades of the same trading day, so none spans the overnight gap or the time outside the session. Returns
  (returns, day_starts, dt_seconds): day_starts[k] = index of day k's first return; dt_seconds[i] = the exact
  ts_event difference to the previous trade, unmodified.
  """
  ts = trades["ts_event"].astype("int64").values            # ns since epoch (UTC)
  price = trades["price"].values.astype(np.float64)
  et = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(SESSION_TZ)
  minute = et.hour * 60 + et.minute
  keep = (minute >= SESSION_OPEN_MIN) & (minute < SESSION_CLOSE_MIN)
  ts, price = ts[keep], price[keep]
  day = (et.year * 10_000 + et.month * 100 + et.day).values[keep]       # New York trading date
  same_day = day[1:] == day[:-1]
  returns = np.diff(np.log(price))[same_day]
  dt = np.diff(ts)[same_day] / 1e9
  day_of = day[1:][same_day]
  day_starts = np.nonzero(np.r_[True, day_of[1:] != day_of[:-1]])[0]
  return returns, day_starts, dt


def trade_returns(trades):
  """Per-trade returns as a flat float array (days concatenated); see trade_returns_by_day."""
  return trade_returns_by_day(trades)[0]


def returns_fingerprint(returns, dtu=None, extra=""):
  """Short hash identifying the exact series (returns and tick timing) a CutRule was trained on."""
  h = hashlib.md5(b"time-aware-v1" + extra.encode())
  h.update(np.ascontiguousarray(returns, dtype=np.float64).tobytes())
  if dtu is not None:
    h.update(np.ascontiguousarray(dtu, dtype=np.float64).tobytes())
  return h.hexdigest()[:16]


@dataclass
class OrclData:
  """ORCL trade returns with tick timing and a chronological split by whole trading days.

  train = [0, train_end), val = [train_end, val_end), test = [val_end, len). `dt` is the raw time in seconds since
  the previous tick. `dtu` is the same data divided by a constant `time_unit` (the mean raw tick interval of the
  training days) so that rule hyperparameters written in "ticks" (L=32, max patch 8) keep their
  meaning and positions stay on the scale the backbone was pretrained on. It is a unit choice only: nothing is
  capped, clipped or resampled, and dtu == 0 exactly where ts_event repeats.
  """
  returns: np.ndarray
  dt: np.ndarray
  day_starts: np.ndarray
  train_end: int
  val_end: int
  time_unit: float

  @property
  def dtu(self):
    return self.dt / self.time_unit

  def train_slice(self):
    """(returns, dtu, day_starts) restricted to the training days."""
    n = self.train_end
    return self.returns[:n], self.dtu[:n], self.day_starts[self.day_starts < n]


def load_orcl(data_dir=DATA_DIR, splits=(0.7, 0.15, 0.15)):
  returns, day_starts, dt = trade_returns_by_day(load_trades(data_dir))
  n_days = len(day_starts)
  d1 = max(1, int(round(splits[0] * n_days)))
  d2 = min(n_days - 1, max(d1 + 1, int(round((splits[0] + splits[1]) * n_days))))
  edges = list(day_starts) + [len(returns)]
  train_end, val_end = int(edges[d1]), int(edges[d2])
  return OrclData(returns, dt, day_starts, train_end, val_end, time_unit=float(dt[:train_end].mean()))
