"""The paper's long-term forecasting benchmarks: zero-shot and full-shot on ETTh1, ETTh2, ETTm1, ETTm2 and Weather.

  zero-shot - the Time-300B-pretrained backbone (stage 2), never trained on the benchmark, forecasts 96 / 192 / 336 / 720
              steps from a 512-point context (paper section 4.1, Table 1).
  full-shot - the same backbone fine-tuned for one epoch at lr 1e-4 on the benchmark's train split, then scored the same way
              (section 4.2, Table 2). A fresh copy of the backbone is fine-tuned for every dataset; the fine-tuned weights
              are not kept (about 0.5 GB each).

An interrupted run continues where it stopped: metrics.json keeps every finished (dataset, mode), and inside the one that
was running, the fine-tuning (model, optimizer, position) is saved about every 5 minutes and the running test scores
about every minute to
partial_<dataset>_<mode>_{finetune,scores}.pt, removed once that (dataset, mode) is scored.

The backbone is the one a full run trained: outputs/backbone_<patcher>.pt (pretrained first if missing; shared by both
spacing variants), or for --patcher joint / trilevel the Time-300B forecaster of method 3 / 4,
outputs/<joint|trilevel>_pretrained.pt. The benchmarks are evenly spaced series, so they are patched exactly as Time-300B
was during pretraining (power rule, the Time-300B CutRule of stage 1, or the CutRule that methods 3 / 4 learned alongside
their forecaster, cutting greedily as in their validation). Results go to outputs/results/benchmarks/<patcher>/: metrics.json, results.csv
(with the paper's TimeSqueeze-base numbers next to ours), mse_<dataset>.png and forecast examples for every horizon
(forecasts_<dataset>_<mode>_H<H>.png: three test windows, context, actual and forecast, in the benchmark's units).

Cost at stride 1 (the paper's protocol): every test window of every channel is rolled out to 720 steps, about 13 forward
passes each. Weather alone is ~220k series; use --stride (e.g. 4 or 8: every 4th / 8th window) for a quicker estimate.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from ..benchmarks.datasets import DATASETS, PAPER_HORIZONS, load_benchmark
from ..benchmarks.evaluate import benchmark_train_loader, collate_benchmark, evaluate_benchmark
from ..model.backbone import get_backbone
from ..patching.kernels import warm_up
from ..patching.power_rule import power_rule_boundaries
from ..utils.paths import output_dir, use_quick_outputs
from ..utils.quick import BACKBONE_PRETRAIN, BENCHMARK, CUTRULE_PRETRAIN, JOINT, TRILEVEL, add_quick_flags, banner
from ..utils.reporting import plot_bars, plot_forecasts, save_csv, save_json
from ..utils.seed import ignore_deprecation_warnings, set_seed
from .common import find_batch_size, make_adamw, to_device, train_one_epoch

PAPER_FULL_SHOT_LR = 1e-4            # the paper's full-shot protocol (section 4.2); ORCL fine-tuning uses less

# TimeSqueeze-base in the paper. Zero-shot: Table 1 (MSE, MAE per horizon). Full-shot: Table 2 (averages over horizons).
PAPER_ZERO_SHOT = {
    "ETTh1": {96: (0.359, 0.385), 192: (0.400, 0.410), 336: (0.420, 0.423), 720: (0.428, 0.446)},
    "ETTh2": {96: (0.282, 0.346), 192: (0.349, 0.394), 336: (0.379, 0.422), 720: (0.444, 0.471)},
    "ETTm1": {96: (0.312, 0.344), 192: (0.372, 0.385), 336: (0.435, 0.425), 720: (0.547, 0.494)},
    "ETTm2": {96: (0.181, 0.275), 192: (0.248, 0.323), 336: (0.310, 0.363), 720: (0.431, 0.437)},
    "Weather": {96: (0.167, 0.217), 192: (0.218, 0.269), 336: (0.278, 0.315), 720: (0.364, 0.372)},
}
PAPER_FULL_SHOT_AVG = {"ETTh1": (0.398, 0.419), "ETTh2": (0.350, 0.393), "ETTm1": (0.383, 0.386),
                       "ETTm2": (0.259, 0.321), "Weather": (0.243, 0.279)}

PATCHERS = ("power", "cutrule", "joint", "trilevel")
LEARNED = {"joint": "reinforce", "trilevel": "trilevel"}     # methods 3 / 4: backbone and CutRule from their phase A


def make_patcher(patcher, cutrule_kwargs=None):
  """The rule the backbone was pretrained with, applied to a normalised window."""
  if patcher == "power":
    return power_rule_boundaries
  from ..cutrule.stages import get_cutrule_patcher
  return get_cutrule_patcher(**(cutrule_kwargs or {}))


def learned_backbone(patcher, device, mamba_cls=None, pretrain_kwargs=None):
  """Methods 3 / 4: (patch_fn, load) for the phase-A forecaster and the CutRule it learned alongside (phase A is run
  first if missing). The CutRule cuts greedily, as in phase A's own validation; load() returns a fresh forecaster."""
  from ..cutrule.stages import CutRulePatcher
  from .joint import cutrule_on, get_joint_pretrained
  leader = LEARNED[patcher]
  model, cut_state, scale = get_joint_pretrained(leader, torch.device("cpu"), mamba_cls, **(pretrain_kwargs or {}))
  del model
  patch_fn = CutRulePatcher(cutrule_on(torch.device("cpu"), cut_state), scale)
  return patch_fn, lambda: get_joint_pretrained(leader, device, mamba_cls, **(pretrain_kwargs or {}))[0]


def full_shot_finetune(model, data, patch_fn, device, context, batch_size="auto", stride=1, lr=PAPER_FULL_SHOT_LR, epochs=1,
                       seed=0, ckpt_path=None, save_every=300):
  """Fine-tune on the benchmark's train split (paper: one epoch, lr 1e-4). Returns the mean training loss per epoch.

  With ckpt_path (one epoch only): model, optimizer and position are saved there about every save_every seconds and
  when the epoch ends, and a later call continues from it (or, if the epoch had finished, just loads the weights)."""
  state = torch.load(ckpt_path, map_location=device, weights_only=False) if ckpt_path and os.path.exists(ckpt_path) else None
  if state is not None and epochs == 1:
    model.load_state_dict(state["model"])
    if state["done"]:
      print(f"  full-shot fine-tuning on {data.name}: finished before an interruption, weights loaded", flush=True)
      return [state["loss"]]
    batch_size = state["batch_size"]
  else:
    state = None
  if batch_size == "auto":
    probe = benchmark_train_loader(data, patch_fn, context, 1, stride=max(stride, 97), seed=seed).dataset
    samples = sorted((probe[i] for i in range(0, len(probe), max(1, len(probe) // 16))), key=lambda s: -len(s["bounds"]))[:4]
    batch_size = find_batch_size(model, lambda b: to_device(collate_benchmark([samples[i % len(samples)] for i in range(b)],
                                                                              context), device), device, cap=64,
                                 reserve_bytes=8 * sum(p.numel() for p in model.parameters()), verbose=False)
  opt = make_adamw(model, lr=lr)
  if ckpt_path is None or epochs != 1:
    loader = benchmark_train_loader(data, patch_fn, context, batch_size, stride=stride, seed=seed)
    print(f"  full-shot fine-tuning on {data.name}: {len(loader.dataset)} windows, batch {batch_size}, {epochs} epoch(s)")
    return [train_one_epoch(model, opt, loader, device, log_every=max(1, len(loader) // 10), label=data.name)
            for _ in range(epochs)]

  skip = 0
  if state is not None:
    opt.load_state_dict(state["opt"])
    skip = state["step"]
  loader = benchmark_train_loader(data, patch_fn, context, batch_size, stride=stride, seed=seed, skip_batches=skip)
  print(f"  full-shot fine-tuning on {data.name}: {len(loader.dataset)} windows, batch {batch_size}, 1 epoch"
        + (f", continuing after batch {skip}" if skip else ""), flush=True)
  saved_at = [time.time()]

  def save(step, loss, done):
    torch.save({"model": model.state_dict(), "opt": None if done else opt.state_dict(), "step": step, "loss": loss,
                "batch_size": batch_size, "done": done}, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)                            # a crash mid-write keeps the old checkpoint
    saved_at[0] = time.time()

  def after_step(k, mean_loss):
    if time.time() - saved_at[0] > save_every:
      save(skip + k, mean_loss(), False)

  loss = train_one_epoch(model, opt, loader, device, log_every=max(1, len(loader) // 10), label=data.name,
                         after_step=after_step)
  save(skip + len(loader), loss, True)
  return [loss]


def run_benchmarks(patcher="power", datasets=tuple(DATASETS), modes=("zero-shot", "full-shot"),
                   context=512, horizons=PAPER_HORIZONS, stride=1, train_stride=1, batch_size=128, train_batch_size="auto",
                   max_windows=None, results_dir=None, mamba_cls=None, pretrain_kwargs=None, cutrule_kwargs=None, seed=0,
                   resume=True):
  """Score the backbone on each benchmark in each mode. Returns {dataset: {mode: {H: {"mse", "mae", "n"}}}}.

  metrics.json is rewritten after every (dataset, mode); resume=True (default) keeps the ones already scored with the
  same settings and backbone, so an interrupted benchmark continues with the next (dataset, mode)."""
  ignore_deprecation_warnings()
  set_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tag = patcher
  results_dir = results_dir or os.path.join(output_dir(), "results", "benchmarks", tag)
  os.makedirs(results_dir, exist_ok=True)
  pre = dict(pretrain_kwargs or {})
  if patcher in LEARNED:
    patch_fn, load = learned_backbone(patcher, device, mamba_cls, pre)
  else:
    patch_fn = make_patcher(patcher, cutrule_kwargs)
    if patcher == "cutrule":
      pre.setdefault("cutrule_kwargs", cutrule_kwargs)
    load = lambda: get_backbone(None, device, mamba_cls, patcher=patcher, **pre)[0]

  settings = {"patcher": patcher, "context": context, "stride": stride, "train_stride": train_stride,
              "max_windows": max_windows, "horizons": list(horizons)}
  metrics_path = os.path.join(results_dir, "metrics.json")
  results, t0 = {}, time.time()
  if resume and os.path.exists(metrics_path):
    saved = json.load(open(metrics_path))
    if saved.get("settings") == settings:
      results = {d: {m: {int(H): v for H, v in r.items()} for m, r in res.items()} for d, res in saved["results"].items()}
      for d, res in results.items():                           # empty figures: a crash before they reached the disk
        for m in list(res):
          figs = [os.path.join(results_dir, f"forecasts_{d}_{m}_H{H}.png") for H in horizons]
          if any(os.path.exists(f) and os.path.getsize(f) == 0 for f in figs):
            del res[m]
            print(f"{d} {m}: its forecast figures are empty (written just before a crash), scoring it again", flush=True)
      done = [f"{d} {m}" for d, res in results.items() for m in res]
      print(f"resuming: already scored {', '.join(done) or 'nothing'}", flush=True)
  # progress inside one (dataset, mode): fine-tuning checkpoint and running scores, removed once its scores are kept
  partial = lambda name, mode, what: os.path.join(results_dir, f"partial_{name}_{mode}_{what}.pt")
  if not results:                                              # starting over: drop any progress of an earlier run
    for f in os.listdir(results_dir):
      if f.startswith("partial_"):
        os.remove(os.path.join(results_dir, f))

  def save_all():
    rows = []
    for d, res in results.items():
      for m, scores in res.items():
        for H in horizons:
          paper = PAPER_ZERO_SHOT[d].get(H) if m == "zero-shot" else None
          rows.append([d, m, H, scores[H]["mse"], scores[H]["mae"], scores[H]["n"],
                       paper[0] if paper else "", paper[1] if paper else ""])
        avg = (np.mean([scores[H]["mse"] for H in horizons]), np.mean([scores[H]["mae"] for H in horizons]))
        paper_avg = (PAPER_FULL_SHOT_AVG[d] if m == "full-shot" else
                     tuple(np.mean([PAPER_ZERO_SHOT[d][H][i] for H in PAPER_HORIZONS]) for i in (0, 1)))
        rows.append([d, m, "avg", avg[0], avg[1], "", paper_avg[0], paper_avg[1]])
    save_json(metrics_path, {"settings": settings, "results": {d: {m: {str(H): v for H, v in r.items()}
                                                                   for m, r in res.items()} for d, res in results.items()}})
    save_csv(os.path.join(results_dir, "results.csv"),
             ["dataset", "mode", "horizon", "mse", "mae", "n_series", "paper_mse", "paper_mae"], rows)

  for name in datasets:
    data = load_benchmark(name)
    results.setdefault(name, {})
    for mode in modes:
      if mode in results[name]:
        continue                                               # scored before an interruption
      model = load()                                           # a fresh copy of the pretrained backbone every time
      print(f"{name} {mode} ({tag}): {data.values.shape[1]} channels, context {context}, stride {stride}", flush=True)
      if mode == "full-shot":
        full_shot_finetune(model, data, patch_fn, device, context, train_batch_size, train_stride, seed=seed,
                           ckpt_path=partial(name, mode, "finetune"))
      scores, examples = evaluate_benchmark(model, data, patch_fn, device, context, horizons, stride, batch_size,
                                            max_windows=max_windows, state_path=partial(name, mode, "scores"))
      # figures first, then the scores into metrics.json, then the saved progress is removed: a crash at any point
      # leaves either a finished part (figures and scores on disk) or the saves to finish it from
      for H in horizons if examples else ():                  # one figure per horizon, same example windows
        plot_forecasts(os.path.join(results_dir, f"forecasts_{name}_{mode}_H{H}.png"), [e["context"] for e in examples],
                       [e["truth"][:H] for e in examples], [e["pred"][:H] for e in examples], H,
                       f"{name} {mode}, H={H} ({tag}, {examples[0]['channel']})",
                       tail=min(512, 2 * H), xlabel="steps relative to forecast origin")
      results[name][mode] = scores
      save_all()
      for what in ("finetune", "scores"):
        if os.path.exists(partial(name, mode, what)):
          os.remove(partial(name, mode, what))
      for H in horizons:
        paper = PAPER_ZERO_SHOT[name].get(H) if mode == "zero-shot" else None
        print(f"  H={H:>3}: MSE {scores[H]['mse']:.3f}  MAE {scores[H]['mae']:.3f}"
              + (f"   (paper {paper[0]:.3f} / {paper[1]:.3f})" if paper else ""), flush=True)
      avg = (np.mean([scores[H]["mse"] for H in horizons]), np.mean([scores[H]["mae"] for H in horizons]))
      paper_avg = (PAPER_FULL_SHOT_AVG[name] if mode == "full-shot" else
                   tuple(np.mean([PAPER_ZERO_SHOT[name][H][i] for H in PAPER_HORIZONS]) for i in (0, 1)))
      print(f"  average: MSE {avg[0]:.3f}  MAE {avg[1]:.3f}   (paper {paper_avg[0]:.3f} / {paper_avg[1]:.3f})"
            f"  [{(time.time() - t0) / 60:.1f} min so far]", flush=True)
      del model
      torch.cuda.empty_cache()
    groups = {f"ours {m}": [results[name][m][H]["mse"] for H in horizons] for m in modes if m in results[name]}
    if all(H in PAPER_ZERO_SHOT[name] for H in horizons):
      groups["paper zero-shot"] = [PAPER_ZERO_SHOT[name][H][0] for H in horizons]
    plot_bars(os.path.join(results_dir, f"mse_{name}.png"), [str(H) for H in horizons], groups,
              f"{name}: test MSE by horizon ({tag})", "MSE (scaled units)")
    save_all()
  print(f"saved benchmark results and figures to {results_dir}")
  return results


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  add_quick_flags(p)
  p.add_argument("--patcher", choices=list(PATCHERS), default="power",
                 help="which pretrained backbone (and its patch rule) to evaluate; joint / trilevel = the Time-300B "
                      "forecaster of method 3 / 4 with its learned CutRule")
  p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
  p.add_argument("--mode", choices=["zero-shot", "full-shot", "both"], default="both")
  p.add_argument("--stride", type=int, default=1, help="score every n-th test window (1 = the paper's protocol)")
  p.add_argument("--train-stride", type=int, default=1, help="full-shot: use every n-th training window")
  p.add_argument("--batch-size", type=int, default=128, help="series forecast at once during scoring")
  p.add_argument("--train-batch-size", type=int, default=None,
                 help="full-shot fine-tuning batch (default: the largest that fits, up to 64)")
  p.add_argument("--context", type=int, default=512)
  p.add_argument("--pretrain-steps", type=int, default=5000, help="steps to pretrain if no backbone exists yet")
  p.add_argument("--restart", action="store_true", help="score every dataset again instead of continuing")
  a = p.parse_args()
  use_quick_outputs(a.quick)
  warm_up()
  modes = ("zero-shot", "full-shot") if a.mode == "both" else (a.mode,)
  kw = dict(datasets=a.datasets, modes=modes, context=a.context, stride=a.stride, train_stride=a.train_stride,
            batch_size=a.batch_size, train_batch_size=a.train_batch_size or "auto", pretrain_kwargs={"steps": a.pretrain_steps}, resume=not a.restart)
  if a.quick:
    banner(True)
    kw.update(BENCHMARK, pretrain_kwargs=dict(BACKBONE_PRETRAIN), cutrule_kwargs=dict(CUTRULE_PRETRAIN))
    if a.patcher in LEARNED:
      kw["pretrain_kwargs"] = dict((TRILEVEL if a.patcher == "trilevel" else JOINT)["pretrain"])
  run_benchmarks(a.patcher, **kw)


if __name__ == "__main__":
  main()
