"""Saving forecasting results (CSV / JSON) and graphics (PNG).

Colors are the first three slots of a palette validated for color-vision deficiency on the light surface
(blue, orange, aqua); series are also told apart by line style and direct labels, and every figure's numbers are
also written to CSV/JSON (the table view). Text uses ink colors, never series colors.
"""
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
STYLES = ("-", "--", ":")


def _fig(w=7.2, h=3.8):
  fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
  ax.set_facecolor(SURFACE)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(INK2)
  ax.tick_params(colors=INK2, labelsize=8)
  ax.grid(True, color=GRID, linewidth=0.6)
  ax.set_axisbelow(True)
  return fig, ax


def _write_durably(path, write, mode="wb", **open_kw):
  """write(f) into a temporary file next to path, force it to disk, then rename it over path. A crash (or power loss)
  leaves either the old file or the complete new one, never an empty or partial file."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  with open(tmp, mode, **open_kw) as f:
    write(f)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)
  if hasattr(os, "O_DIRECTORY"):                                       # make the rename itself durable (POSIX)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(fd)
    finally:
      os.close(fd)
  return str(path)


def _save_fig(fig, path, **kw):
  _write_durably(path, lambda f: fig.savefig(f, format=Path(path).suffix[1:] or "png", dpi=140, facecolor=SURFACE, **kw))
  plt.close(fig)
  return str(path)


def _finish(fig, ax, path, title, xlabel="", ylabel="", legend=True):
  ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
  ax.set_xlabel(xlabel, fontsize=8, color=INK2)
  ax.set_ylabel(ylabel, fontsize=8, color=INK2)
  if legend and ax.get_legend_handles_labels()[0]:
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    for h in leg.legend_handles:
      h.set_linewidth(2)
  fig.tight_layout()
  return _save_fig(fig, path)


def save_json(path, obj):
  return _write_durably(path, lambda f: json.dump(obj, f, indent=2,
                                                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                        "w", encoding="utf8")


def save_csv(path, header, rows):
  def write(f):
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
  return _write_durably(path, write, "w", newline="", encoding="utf8")


def plot_lines(path, curves, title, xlabel="", ylabel="", logy=False, vlines=None, markers=False):
  """curves: {label: (x, y)}. Thin lines, one per series, labelled."""
  fig, ax = _fig()
  for i, (label, (x, y)) in enumerate(curves.items()):
    ax.plot(x, y, color=SERIES[i % 3], linestyle=STYLES[i % 3], linewidth=1.6, label=label,
            marker="o" if markers else None, markersize=3)
  for v in vlines or []:
    ax.axvline(v, color=INK2, linewidth=0.8, linestyle=":")
  if logy:
    ax.set_yscale("log")
  return _finish(fig, ax, path, title, xlabel, ylabel, legend=len(curves) > 1)


def plot_bars(path, categories, groups, title, ylabel=""):
  """Grouped bars: groups = {label: values per category}."""
  fig, ax = _fig()
  k, x = len(groups), np.arange(len(categories))
  w = 0.8 / k
  for i, (label, vals) in enumerate(groups.items()):
    ax.bar(x + (i - (k - 1) / 2) * w, vals, w * 0.9, color=SERIES[i % 3], label=label)
  ax.set_xticks(x, [str(c) for c in categories])
  return _finish(fig, ax, path, title, "", ylabel, legend=k > 1)


def plot_hist(path, values, title, xlabel, bins=60, logx=False, logy=True, annotate=None):
  fig, ax = _fig()
  v = np.asarray(values, dtype=float)
  if logx:
    v = v[v > 0]
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), bins)
    ax.set_xscale("log")
  ax.hist(v, bins=bins, color=BLUE)
  if logy:
    ax.set_yscale("log")
  if annotate:
    ax.text(0.98, 0.95, annotate, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
  return _finish(fig, ax, path, title, xlabel, "count", legend=False)


def plot_patches(path, x, rules, title, xlabel="tick", ylabel="normalized value"):
  """A signal with the patch starts of one or more rules drawn as tick marks beneath it.

  rules: {label: boundary indices}. Each rule gets its own row so they never overlap."""
  n = len(rules)
  fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.8 + 0.3 * n), facecolor=SURFACE, sharex=True,
                                gridspec_kw={"height_ratios": [4, max(1, n)], "hspace": 0.08})
  for a in (ax, ax2):
    a.set_facecolor(SURFACE)
    for side in ("top", "right"):
      a.spines[side].set_visible(False)
    a.tick_params(colors=INK2, labelsize=8)
  ax.plot(np.arange(len(x)), x, color=INK2, linewidth=1.0)
  ax.grid(True, color=GRID, linewidth=0.6)
  for i, (label, b) in enumerate(rules.items()):
    ax2.vlines(b, i + 0.1, i + 0.9, color=SERIES[i % 3], linewidth=1.4)
    ax2.text(len(x) * 1.005, i + 0.5, f"{label} ({len(b)} patches)", va="center", fontsize=8, color=INK2)
  ax2.set_ylim(0, n)
  ax2.set_yticks([])
  ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
  ax.set_ylabel(ylabel, fontsize=8, color=INK2)
  ax2.set_xlabel(xlabel, fontsize=8, color=INK2)
  return _save_fig(fig, path, bbox_inches="tight")


def plot_forecasts(path, contexts, truths, preds, horizon, title, n=3, tail=96, xlabel="ticks relative to forecast origin"):
  """Small multiples: the end of the context, the realised future and the forecast, per example window."""
  n = min(n, len(contexts))
  fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.3), facecolor=SURFACE, squeeze=False)
  for j, ax in enumerate(axes[0]):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.6)
    c = contexts[j][-tail:]
    xc = np.arange(-len(c), 0)
    xf = np.arange(horizon)
    ax.plot(xc, c, color=INK2, linewidth=1.0, label="context")
    mk = dict(marker="o", markersize=3) if horizon <= 8 else {}     # a 1-step forecast is a single point
    ax.plot(xf, truths[j], color=BLUE, linewidth=1.6, label="actual", **mk)
    ax.plot(xf, preds[j], color=ORANGE, linewidth=1.6, linestyle="--", label="forecast", **mk)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
    if j == 0:
      ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    ax.set_xlabel(xlabel, fontsize=7, color=INK2)
  fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
  fig.tight_layout()
  return _save_fig(fig, path)


def write_forecast_report(results_dir, name, forecasts, horizons, history=None, extra=None):
  """Save forecasts.csv, metrics.json and the standard figures for one run.

  forecasts: dict with arrays starts [N], mean [N], std [N], ctx [N,T] (normalized), per-horizon truth {h:[N,h]},
  pred {h:[N,h]}, and bounds {rule label: list of N boundary arrays} (first windows are plotted). For ORCL these are
  per-step log returns in units of the context's RMS log return (value * std + mean = the log return itself).
  """
  d = Path(results_dir)
  d.mkdir(parents=True, exist_ok=True)
  rows = []
  for i, s in enumerate(forecasts["starts"]):
    for h in horizons:
      for k in range(h):
        t, p = float(forecasts["truth"][h][i][k]), float(forecasts["pred"][h][i][k])
        m, sd = float(forecasts["mean"][i]), float(forecasts["std"][i])
        rows.append([int(s), h, k + 1, t, p, t * sd + m, p * sd + m])
  save_csv(d / "forecasts.csv", ["window_start", "horizon", "step", "actual_norm", "forecast_norm",
                                 "actual_log_return", "forecast_log_return"], rows)
  mse = {h: float(np.mean((forecasts["pred"][h] - forecasts["truth"][h]) ** 2)) for h in horizons}
  mae = {h: float(np.mean(np.abs(forecasts["pred"][h] - forecasts["truth"][h]))) for h in horizons}
  zero = {h: float(np.mean(forecasts["truth"][h] ** 2)) for h in horizons}
  drift = np.array([c[1:].mean() for c in forecasts["ctx"]])[:, None] if len(forecasts["starts"]) else 0.0
  mean_f = {h: float(np.mean((forecasts["truth"][h] - drift) ** 2)) for h in horizons}
  metrics = {"name": name, "n_windows": len(forecasts["starts"]), "mse": mse, "mae": mae, "zero_forecast_mse": zero,
             "context_mean_forecast_mse": mean_f, **(extra or {})}
  save_json(d / "metrics.json", metrics)
  plot_bars(d / "mse_by_horizon.png", horizons, {"model": [mse[h] for h in horizons],
                                                 "zero forecast": [zero[h] for h in horizons],
                                                 "context mean": [mean_f[h] for h in horizons]},
            f"{name}: per-step log-return error by horizon (units of the context's RMS log return)", "MSE")
  for h in horizons:
    if len(forecasts["starts"]):
      plot_forecasts(d / f"forecast_examples_h{h}.png", forecasts["ctx"], forecasts["truth"][h], forecasts["pred"][h],
                     h, f"{name}: {h}-step log-return forecasts vs actual")
  if forecasts.get("bounds") and len(forecasts["starts"]):
    rules = {k: v[0] for k, v in forecasts["bounds"].items()}
    plot_patches(d / "patch_boundaries.png", forecasts["ctx"][0], rules,
                 f"{name}: patch starts on one test window")
    sizes = {k: np.concatenate([np.diff(np.append(b, len(forecasts["ctx"][0]))) for b in v])
             for k, v in forecasts["bounds"].items()}
    for k, v in sizes.items():
      plot_hist(d / f"patch_sizes_{k}.png", v, f"{name}: patch length (ticks), {k}", "ticks per patch", bins=40,
                logy=True, annotate=f"mean {v.mean():.1f}")
  if history:
    ep = [h["epoch"] for h in history]
    plot_lines(d / "loss_curves.png", {"train": (ep, [h["train_loss"] for h in history]),
                                       "validation": (ep, [h["val_loss"] for h in history])},
               f"{name}: loss per epoch", "epoch", "loss", markers=True)
    save_csv(d / "history.csv", ["epoch", "train_loss", "val_loss"],
             [[h["epoch"], h["train_loss"], h["val_loss"]] for h in history])
  return metrics
