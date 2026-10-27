"""Stage 4: fine-tune the pretrained forecaster on ORCL trade returns, then evaluate and save the report.

The one-command path: everything that is missing is done first, in order -
  1. CutRule pretrained on Time-300B          (cutrule.stages, only for patcher="cutrule"),
  2. forecaster pretrained on Time-300B       (training.pretrain, with that patch rule),
  3. CutRule continued on the ORCL train days (cutrule.stages, only for patcher="cutrule"),
then the forecaster is fine-tuned on the training days at lr 1e-5 (the paper uses 1e-4, which overfit ORCL within the
first epoch). Every 6,400 training windows (~200 steps, ~5 times per epoch) it is scored on the validation days, and the
weights of the best validation point are kept (the model before fine-tuning included; see common.BestOnValidation). Those are evaluated on the held-out test days, and the forecasts, metrics and figures are
written to outputs/results/<run>/.

Ticks are not evenly spaced, so by default everything uses the raw ts_event gaps; time_aware=False (or --spacing even) treats
ticks as evenly spaced instead, with its own models and weights ("_ticks").
"""
import argparse
import os
import time

import torch

from ..cutrule.stages import get_orcl_cutrule
from ..data.orcl import load_orcl
from ..data.orcl_windows import OrclWindows, collate, make_loader
from ..model.backbone import get_backbone
from ..model.forecaster import HORIZONS
from ..patching.kernels import warm_up
from ..patching.segmenter import CutRuleSegmenter
from ..utils.paths import output_dir, use_quick_outputs
from ..utils.quick import BACKBONE_PRETRAIN, CUTRULE_PRETRAIN, FINETUNE, add_quick_flags, add_spacing_flag, banner, spacings
from ..utils.reporting import write_forecast_report
from ..utils.seed import ignore_deprecation_warnings, set_seed
from .common import (
  FINETUNE_LR,
  ORCL_BATCH,
  VAL_EVERY_WINDOWS,
  BestOnValidation,
  find_batch_size,
  make_adamw,
  to_device,
  train_one_epoch,
)
from .evaluation import collect_forecasts, evaluate_last_position, print_test_scores


def finetune(patcher="cutrule", backbone_path=None, cutrule_ckpt=None, context=512, stride=16,
             batch_size=ORCL_BATCH, epochs=5, lr=FINETUNE_LR, out_dir=None, seed=0, mamba_cls=None, pretrain_kwargs=None,
             data=None, time_aware=True, cutrule_epochs=30, cutrule_pretrain_kwargs=None, cutrule_from_scratch=False,
             results_dir=None, resume=True):
  """Pretrain-if-missing, then fine-tune on ORCL with the chosen patch rule. Returns (model, results).

  patcher="cutrule": the CutRule is pretrained on Time-300B (if not already), the backbone is pretrained on its
  patches (if not already), the CutRule continues training on the ORCL train days, then the forecaster is
  fine-tuned on ORCL. patcher="power": only the backbone is pretrained (on power-rule patches).

  resume=True (default): an interrupted fine-tune continues after its last finished epoch, if the rolling checkpoint
  was written for exactly this run (same patch rule, spacing, backbone and settings); otherwise it starts over."""
  assert patcher in ("cutrule", "power"), "patcher must be 'cutrule' or 'power'"
  out_dir = out_dir or output_dir()
  ignore_deprecation_warnings()
  set_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pretrain_kwargs = dict(pretrain_kwargs or {})
  if patcher == "cutrule":
    pretrain_kwargs.setdefault("cutrule_kwargs", cutrule_pretrain_kwargs)
  model, meta = get_backbone(backbone_path, device, mamba_cls, patcher=patcher, **pretrain_kwargs)
  data = data or load_orcl()
  n, b1, b2 = len(data.returns), data.train_end, data.val_end

  segmenter = None
  if patcher == "cutrule":
    segmenter = CutRuleSegmenter(get_orcl_cutrule(data, time_aware, cutrule_ckpt, cutrule_epochs, cutrule_from_scratch,
                                                  pretrain_kwargs=cutrule_pretrain_kwargs))
  mk = lambda lo, hi, st: OrclWindows(data, lo, hi, context, st, patcher, segmenter=segmenter, time_aware=time_aware)
  train_ds, val_ds, test_ds = mk(0, b1, stride), mk(b1, b2, 64), mk(b2, n, 64)
  print(f"{patcher} patcher, time-aware={time_aware}; windows train/val/test = "
        f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} (context {context})")
  if batch_size == "auto":                         # fill the GPU: probe with real windows (the most patches = most memory)
    probes = sorted((train_ds[i] for i in range(0, len(train_ds), max(1, len(train_ds) // 16))),
                    key=lambda s: -len(s["bounds"]))[:4]
    print(f"finding the largest fine-tuning batch that fits (context {context})")
    batch_size = find_batch_size(model, lambda b: to_device(collate([probes[i % len(probes)] for i in range(b)], context),
                                                            device), device, cap=64,
                                 reserve_bytes=8 * sum(p.numel() for p in model.parameters()))
    print(f"batch size {batch_size}")
  train_loader = make_loader(train_ds, batch_size, True)
  val_loader, test_loader = make_loader(val_ds, batch_size, False), make_loader(test_ds, batch_size, False)

  opt = make_adamw(model, lr=lr)
  tag = f"{patcher}{'' if time_aware else '_ticks'}"
  ckpt = os.path.join(out_dir, f"finetune_{tag}_ckpt.pt")
  run_id = {"patcher": patcher, "time_aware": time_aware, "backbone_steps": meta.get("steps"),
            "cutrule_fp": meta.get("cutrule_fp"), "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "context": context, "stride": stride, "n_train": len(train_ds)}
  history, t0, best, start_ep = [], time.time(), BestOnValidation(), 0
  ck = torch.load(ckpt, map_location=device) if resume and os.path.exists(ckpt) else None
  if ck is not None and ck.get("run") == run_id and ck.get("best"):
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    history, start_ep, best = ck["history"], ck["epoch"], BestOnValidation.from_dict(ck["best"])
    val0 = {"loss": ck["val0"]}
    print(f"resumed from {ckpt} after epoch {start_ep} (best so far: epoch {best.epoch})", flush=True)
  else:
    if ck is not None:
      print(f"{ckpt} belongs to another run; starting this fine-tune over", flush=True)
    val0 = evaluate_last_position(model, val_loader, device)            # the pretrained model, before fine-tuning
    best.update(val0["loss"], 0, model)
    print(f"epoch 0 (pretrained, not fine-tuned): val loss {val0['loss']:.4f}", flush=True)
  steps_per_epoch = len(train_loader)
  val_every = max(1, VAL_EVERY_WINDOWS // batch_size)

  def validate(ep, k, train_loss):
    """Score on the validation days at step k of epoch ep (label: fractional epochs) and keep the best weights."""
    point = round(ep + k / steps_per_epoch, 3)
    val = evaluate_last_position(model, val_loader, device)
    history.append({"epoch": point, "train_loss": train_loss, "val_loss": val["loss"]})
    better = best.update(val["loss"], point, model)
    print(f"epoch {point:.2f} (step {ep * steps_per_epoch + k}) train loss {train_loss:.4f} val loss {val['loss']:.4f}"
          f"{'  (best so far)' if better else ''} ({time.time() - t0:.0f}s)", flush=True)

  for ep in range(start_ep, epochs):
    mid = lambda k, loss, ep=ep: validate(ep, k, loss()) if k % val_every == 0 and k < steps_per_epoch else None
    train_loss = train_one_epoch(model, opt, train_loader, device, after_step=mid)
    validate(ep, steps_per_epoch, train_loss)                           # and at the end of every epoch
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "history": history,
                "run": run_id, "best": best.to_dict(), "val0": val0["loss"]}, ckpt + ".tmp")
    os.replace(ckpt + ".tmp", ckpt)                    # single rolling checkpoint

  best.restore(model)
  test = evaluate_last_position(model, test_loader, device)
  print_test_scores(test)
  out = os.path.join(out_dir, f"finetuned_{tag}.pt")
  torch.save({"model": model.state_dict(), "patcher": patcher, "time_aware": time_aware, "history": history,
              "best_epoch": best.epoch, "pretrained_val_loss": val0["loss"], "test": test}, out)
  print(f"saved {out}")
  results_dir = results_dir or os.path.join(out_dir, "results", tag)
  forecasts = collect_forecasts(model, test_ds, device)
  write_forecast_report(results_dir, f"ORCL {tag}", forecasts, HORIZONS, history,
                        extra={"patcher": patcher, "time_aware": time_aware, "compression": test["compression"],
                               "test_windows_total": len(test_ds), "best_epoch": best.epoch,
                               "pretrained_val_loss": val0["loss"]})
  print(f"saved forecasts, metrics and figures to {results_dir}")
  return model, {"history": history, "test": test, "results_dir": results_dir, "best_epoch": best.epoch}


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  add_quick_flags(p)
  add_spacing_flag(p)
  p.add_argument("--patcher", choices=["cutrule", "power"], default="cutrule",
                 help="cutrule = pretrained bilevel cut rule, power = the paper's relative-deviation rule")
  p.add_argument("--backbone", default=None, help="default outputs/backbone_<patcher>.pt (shared by both spacings)")
  p.add_argument("--cutrule-ckpt", default=None)
  p.add_argument("--context", type=int, default=512)
  p.add_argument("--epochs", type=int, default=5)
  p.add_argument("--batch-size", default=str(ORCL_BATCH),
                 help=f"windows per step (default {ORCL_BATCH}, the same as methods 3/4's ORCL phase), or 'auto' (largest that fits)")
  p.add_argument("--lr", type=float, default=FINETUNE_LR, help=f"default {FINETUNE_LR} (the paper's 1e-4 overfit ORCL)")
  p.add_argument("--no-time-aware", action="store_true", help="same as --spacing even")
  p.add_argument("--cutrule-epochs", type=int, default=30, help="epochs of CutRule training on ORCL")
  p.add_argument("--cutrule-from-scratch", action="store_true", help="skip the Time-300B CutRule pretraining")
  p.add_argument("--pretrain-steps", type=int, default=5000, help="steps to pretrain if no backbone exists yet")
  p.add_argument("--pretrain-max-minutes", type=float, default=None)
  p.add_argument("--pretrain-micro-batch", default="auto")
  p.add_argument("--pretrain-global-batch", type=int, default=64, help="global batch for pretraining (paper: 256)")
  p.add_argument("--restart", action="store_true",
                 help="start an interrupted backbone pretraining / fine-tune over instead of continuing it")
  a = p.parse_args()
  use_quick_outputs(a.quick)
  warm_up()                                        # compile the patching kernels once, before dataloader workers start
  kw = dict(context=a.context, epochs=a.epochs, batch_size=a.batch_size if a.batch_size == "auto" else int(a.batch_size),
            cutrule_epochs=a.cutrule_epochs)
  pre = {"steps": a.pretrain_steps, "max_minutes": a.pretrain_max_minutes,
         "micro_batch": a.pretrain_micro_batch if a.pretrain_micro_batch == "auto" else int(a.pretrain_micro_batch),
         "global_batch": a.pretrain_global_batch, "resume": not a.restart}
  cut_pre = None
  if a.quick:
    banner(True)
    kw.update(FINETUNE)
    pre = dict(BACKBONE_PRETRAIN)
    cut_pre = dict(CUTRULE_PRETRAIN)
  variants = spacings("even" if a.no_time_aware else a.spacing)
  for time_aware in variants:
    finetune(a.patcher, a.backbone if len(variants) == 1 else None, a.cutrule_ckpt if len(variants) == 1 else None,
             lr=a.lr, time_aware=time_aware, cutrule_from_scratch=a.cutrule_from_scratch, pretrain_kwargs=dict(pre),
             cutrule_pretrain_kwargs=cut_pre, resume=not a.restart, **kw)


if __name__ == "__main__":
  main()
