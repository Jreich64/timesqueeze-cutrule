"""Evaluating a forecaster on ORCL windows and collecting its forecasts for the saved report.

The score is the forecast from the *last* position of each window (the whole context is in the past), for each horizon.
Everything is reported per step as **log returns**, not cumulative returns: the model forecasts the next h one-step
changes of the log price path, which are the next h log returns, in units of the context's RMS log return
(forecaster.step_targets). MSE and MAE are averaged over the h steps, next to two baselines:
  * zero forecast       - every future log return is 0 (no change: the random walk),
  * context-mean return - every future log return equals the context's mean log return (its drift continues).
"""
import numpy as np
import torch

from ..data.orcl_windows import collate
from ..model.forecaster import HORIZONS, forecasting_loss, run_model, step_targets
from .common import to_device


@torch.no_grad()
def evaluate_last_position(model, loader, device):
  """MSE / MAE per horizon of the per-step log-return forecasts from the end of each context window (units: the
  context's RMS log return), next to the zero forecast and the context-mean forecast. Also returns the loss over all
  positions."""
  model.eval()
  se = {h: 0.0 for h in HORIZONS}; ae = {h: 0.0 for h in HORIZONS}
  zse = {h: 0.0 for h in HORIZONS}; mse_mean = {h: 0.0 for h in HORIZONS}
  n, total_loss, patches, T = 0, 0.0, 0, 0
  for batch in loader:
    b = to_device(batch, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
      preds, aux = run_model(model, b)
    loss, _ = forecasting_loss(preds, b["y"], b["n_valid"], aux)
    B, T = b["x"].shape
    d, _ = step_targets(b["y"], b["n_valid"], T)
    drift = d[:, 1:T].mean(1, keepdim=True)                            # the context's mean scaled log return
    total_loss += loss.item() * B
    patches += b["patch_mask"].sum().item()
    for h in HORIZONS:
      tgt = d[:, T:T + h]
      err = preds[h][:, -1].float() - tgt
      se[h] += (err ** 2).mean(1).sum().item()
      ae[h] += err.abs().mean(1).sum().item()
      zse[h] += (tgt ** 2).mean(1).sum().item()
      mse_mean[h] += ((tgt - drift) ** 2).mean(1).sum().item()
    n += B
  model.train()
  return {"loss": total_loss / n, "compression": T * n / max(patches, 1),
          "mse": {h: se[h] / n for h in HORIZONS}, "mae": {h: ae[h] / n for h in HORIZONS},
          "zero_mse": {h: zse[h] / n for h in HORIZONS}, "mean_mse": {h: mse_mean[h] / n for h in HORIZONS}}


def print_test_scores(test):
  """The test-day table printed at the end of every ORCL run."""
  print(f"test compression {test['compression']:.1f}x tokens; per-step log-return forecasts from the last position "
        f"(units: the context's RMS log return):")
  for h in HORIZONS:
    print(f"  h={h:>2}: MSE {test['mse'][h]:.4f}  MAE {test['mae'][h]:.4f}  (zero forecast MSE {test['zero_mse'][h]:.4f}, "
          f"context-mean MSE {test['mean_mse'][h]:.4f})")


@torch.no_grad()
def collect_forecasts(model, ds, device, batch_size=8, max_windows=200, n_bounds=3):
  """Last-position forecasts of up to max_windows evenly spaced windows of `ds`, with what is needed to save and plot
  them: window starts, the scale back to log returns (mean 0, std = the context's RMS log return), the context's
  scaled log returns, per-step truth and prediction per horizon (scaled log returns), and the patch starts the model
  saw."""
  model.eval()
  idx = np.unique(np.linspace(0, len(ds) - 1, min(len(ds), max_windows)).astype(int))
  out = {"starts": [], "mean": [], "std": [], "ctx": [], "truth": {h: [] for h in HORIZONS},
         "pred": {h: [] for h in HORIZONS}, "bounds": {ds.patcher: []}}
  T = ds.context
  for k in range(0, len(idx), batch_size):
    chunk = idx[k:k + batch_size]
    samples = [ds[i] for i in chunk]
    b = to_device(collate(samples, T), device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
      preds, _ = run_model(model, b)
    d, _ = step_targets(b["y"], b["n_valid"], T)
    for h in HORIZONS:
      out["pred"][h].append(preds[h][:, -1].float().cpu().numpy())
      out["truth"][h].append(d[:, T:T + h].cpu().numpy())
    for i, smp in zip(chunk, samples):
      raw = ds.r[ds.starts[i]:ds.starts[i] + T]
      rms = float(np.sqrt((raw[1:] ** 2).mean()))                      # = the scale of step_targets, in log returns
      out["starts"].append(ds.starts[i]); out["mean"].append(0.0); out["std"].append(rms)
      out["ctx"].append((raw / rms).astype(np.float32))                 # the context's log returns, same units
      if len(out["bounds"][ds.patcher]) < n_bounds:
        out["bounds"][ds.patcher].append(smp["bounds"])
  for key in ("pred", "truth"):
    out[key] = {h: np.concatenate(out[key][h]) for h in HORIZONS}
  for key in ("starts", "mean", "std"):
    out[key] = np.array(out[key])
  model.train()
  return out
