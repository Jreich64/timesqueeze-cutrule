"""ORCL is fed as the log price path and forecast as per-step changes, which are the log returns themselves; reports
stay per-step log returns (not cumulative)."""
import numpy as np
import torch

from timesqueeze.data.orcl import load_orcl
from timesqueeze.data.orcl_windows import OrclWindows, collate
from timesqueeze.model.forecaster import HORIZONS, step_targets


def test_orcl_input_is_the_log_price_path_and_targets_are_scaled_log_returns(artifacts):
  d = load_orcl()
  ds = OrclWindows(d, 0, d.train_end, 512, 997, "power", time_aware=False)
  b = collate([ds[i] for i in range(4)], 512)
  T, H = 512, max(HORIZONS)
  steps, scale = step_targets(b["y"], b["n_valid"], T)
  worst = 0.0
  for j in range(4):
    s = ds.starts[j]
    r = d.returns[s:s + T + H]
    path = np.cumsum(r)
    z = (path - path[:T].mean()) / path[:T].std()
    assert np.allclose(b["y"][j].numpy(), z, atol=1e-5)                      # input = z-scored log price path
    rms = np.sqrt((r[1:T] ** 2).mean())
    back = steps[j, 1:].numpy() * rms                                           # scaled steps -> log returns
    worst = max(worst, float(np.abs(back - r[1:]).max()))
    assert np.allclose(back, r[1:], rtol=1e-3, atol=1e-8)
  artifacts.json("path_and_returns.json", {"windows": 4, "max_abs_log_return_error": worst})


def test_reported_forecasts_are_per_step_log_returns(device):
  """collect_forecasts must report per-step log returns: truth * std + mean reproduces the actual log returns."""
  from timesqueeze.training.evaluation import collect_forecasts

  class Zero(torch.nn.Module):
    horizons = HORIZONS

    def forward(self, x, *a):
      return {h: torch.zeros(x.shape[0], x.shape[1], h, device=x.device) for h in HORIZONS}, torch.zeros((), device=x.device)

  d = load_orcl()
  ds = OrclWindows(d, d.val_end, len(d.returns), 512, 499, "power", time_aware=False)
  f = collect_forecasts(Zero().to(device), ds, device, max_windows=5)
  for i, s in enumerate(f["starts"]):
    actual = d.returns[s + 512:s + 512 + 64]
    assert np.allclose(f["truth"][64][i] * f["std"][i] + f["mean"][i], actual, rtol=1e-3, atol=1e-8)
    assert np.allclose(f["ctx"][i] * f["std"][i], d.returns[s:s + 512], rtol=1e-5)
