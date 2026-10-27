"""Stage 2: pretrain the forecaster (Mamba encoder/decoder + MoE Transformer + heads) on Time-300B.

Follows the paper's recipe (section 3.2, appendix B): Huber loss averaged over horizons {1, 8, 32, 64} plus 0.02 x router
load-balancing loss; AdamW (lr 1e-3, weight decay 0.1, betas 0.9/0.95); linear warmup for the first 10% of steps then
cosine decay to 5e-5; bf16; context 2048. The paper uses a global batch of 256 for 100k steps on 2 A100s. Here the default
global batch is 64 (reached by gradient accumulation; pass 256 for the paper's): the GPU processes a fixed ~25k points/s, so
a smaller batch gives proportionally more optimizer updates in the same time. The step count is yours to choose.

Patch rule: "power" pretrains on the paper's rule; "cutrule" on the patches of the CutRule pretrained on Time-300B (stage 1,
trained first if missing). Fine-tune with the same rule. One backbone per rule serves both spacing variants: Time-300B
has no timestamps, so pretraining does not depend on the variant.

One rolling checkpoint is kept (each save replaces the previous one) together with what it was trained for (patch rule,
CutRule). A run continues from it automatically (resume=True, the default: an interrupted run loses at most
ckpt_every steps) and refuses a checkpoint written for another configuration; --restart starts over.
"""
import argparse
import os
import time

import numpy as np
import torch

from ..data.time300b_windows import Time300BSegments, Time300BWindows, make_loader
from ..model.backbone import backbone_path, build_model, save_backbone
from ..model.forecaster import HORIZONS, forecasting_loss, run_model
from ..patching.kernels import warm_up
from ..patching.power_rule import power_rule_boundaries
from ..utils.paths import out, use_quick_outputs
from ..utils.quick import BACKBONE_PRETRAIN, CUTRULE_PRETRAIN, add_quick_flags, banner
from ..utils.seed import ignore_deprecation_warnings, set_seed
from .common import default_workers, find_batch_size, lr_at, make_adamw, to_device


@torch.no_grad()
def evaluate(model, loader, device):
  model.eval()
  total, n = 0.0, 0
  for batch in loader:
    b = to_device(batch, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
      preds, aux = run_model(model, b)
    loss, _ = forecasting_loss(preds, b["y"], b["n_valid"], aux)
    total += loss.item() * len(b["x"])
    n += len(b["x"])
  model.train()
  return total / max(n, 1)


@torch.no_grad()
def no_change_loss(loader, device):
  """Validation loss of the "no change" forecast (every predicted step 0: the last value persists), the baseline the
  pretrained model should beat."""
  total, n = 0.0, 0
  for batch in loader:
    b = to_device(batch, device)
    T = b["x"].shape[1]
    zeros = {h: torch.zeros(len(b["x"]), T, h, device=device) for h in HORIZONS}
    total += forecasting_loss(zeros, b["y"], b["n_valid"])[0].item() * len(b["x"])
    n += len(b["x"])
  return total / max(n, 1)


def save_checkpoint(path, model, opt, step, history, meta=None):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "history": history, "meta": meta}, tmp)
  os.replace(tmp, path)                               # replaces (deletes) the previous checkpoint


def pretrain_backbone(out_path=None, ckpt_path=None, steps=5000, patcher="power", cutrule_kwargs=None,
                      context=2048, global_batch=64, micro_batch="auto", warmup_frac=0.1, log_every=10, val_every=250,
                      ckpt_every=100, resume=True, max_minutes=None, num_workers=None, seed=0, mamba_cls=None,
                      data_dir=None, benchmark=False):
  """Pretrain on the local Time-300B windows and write the backbone to out_path. Returns the model.

  patcher: "power" pretrains on the paper's power-rule patches; "cutrule" on the patches of the bilevel CutRule
  that was itself pretrained on Time-300B (cutrule_stages.get_cutrule_patcher; trained first if missing, with
  cutrule_kwargs). Use the same rule when fine-tuning on ORCL. The result serves both spacing variants.
  """
  ignore_deprecation_warnings()
  out_path = out_path or backbone_path(patcher)
  ckpt_path = ckpt_path or out(f"pretrain_ckpt_{patcher}.pt")
  if patcher == "cutrule":
    from ..cutrule.stages import get_cutrule_patcher
    patch_fn = get_cutrule_patcher(**(cutrule_kwargs or {}))
  else:
    patch_fn = power_rule_boundaries
  run_meta = {"patcher": patcher, "cutrule_fp": getattr(patch_fn, "fingerprint", None)}
  set_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_model(mamba_cls).to(device)
  opt = make_adamw(model)
  warmup = max(1, int(warmup_frac * steps))
  num_workers = default_workers() if num_workers is None else num_workers
  print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {device}")

  step, history = 0, []
  if resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("meta") not in (None, run_meta):                       # different patch rule / CutRule
      raise RuntimeError(f"{ckpt_path} was written for {ckpt['meta']}, this run is {run_meta}; delete it or change --out")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    step, history = ckpt["step"], ckpt["history"]
    print(f"resumed from {ckpt_path} at step {step}")

  from ..utils.paths import DATA_DIR
  segments = Time300BSegments(data_dir or DATA_DIR)
  print(f"{len(segments.lengths)} series segments ({int(segments.lengths.sum()) / 1e9:.2f}B points)")
  # reseed on resume so a resumed run does not replay the samples it already saw
  train_ds = Time300BWindows(segments, context, HORIZONS, "train", seed=seed * 1_000_003 + step, patcher=patch_fn)
  if micro_batch == "auto":                        # fill the GPU: probe with real windows (the most patches = most memory)
    from ..data.time300b_windows import collate
    rng = np.random.default_rng(seed)
    probes = sorted((train_ds.sample(rng) for _ in range(16)), key=lambda s: -len(s["bounds"]))[:4]
    cap = min(global_batch, 64)
    print(f"finding the largest micro-batch that fits (context {context}, up to {cap})")
    micro_batch = find_batch_size(model, lambda b: to_device(collate([probes[i % len(probes)] for i in range(b)], context),
                                                             device), device, cap,
                                  reserve_bytes=8 * sum(p.numel() for p in model.parameters()))
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()             # so "peak mem" in the log is training, not the batch-size probe
  accum = max(1, global_batch // micro_batch)
  print(f"effective batch {accum * micro_batch} = {accum} x {micro_batch}, context {context}, {num_workers} loader "
        f"workers, {steps} steps (~{steps * accum * micro_batch * context / 1e9:.2f}B points)")
  val_loader = make_loader(Time300BWindows(segments, context, HORIZONS, "val", patcher=patch_fn), micro_batch, num_workers=0)
  train_iter = iter(make_loader(train_ds, micro_batch, num_workers))
  baseline = no_change_loss(val_loader, device) if val_every else None
  if baseline is not None:
    print(f"validation loss of the no-change forecast: {baseline:.4f}", flush=True)

  start, t_last = time.time(), time.time()
  run_loss, run_n = 0.0, 0                         # run_loss is a GPU tensor: no host sync per micro-batch
  model.train()
  while step < steps:
    for g in opt.param_groups:
      g["lr"] = lr_at(step, steps, warmup)
    for _ in range(accum):
      b = to_device(next(train_iter), device)
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        preds, aux = run_model(model, b)
      loss, _ = forecasting_loss(preds, b["y"], b["n_valid"], aux)
      (loss / accum).backward()
      run_loss = run_loss + loss.detach()
      run_n += 1
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1

    if step % log_every == 0 or step == steps or benchmark:
      now = time.time()
      sec = (now - t_last) / (log_every if step % log_every == 0 else 1)
      eta = (steps - step) * sec / 3600
      mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
      mean_loss = float(run_loss) / run_n
      print(f"step {step}/{steps} loss {mean_loss:.4f} lr {opt.param_groups[0]['lr']:.2e} "
            f"gnorm {float(grad_norm):.2f} {sec:.2f}s/step ETA {eta:.1f}h peak mem {mem:.1f}GB", flush=True)
      history.append({"step": step, "loss": mean_loss})
      run_loss, run_n, t_last = 0.0, 0, now
    if benchmark and step >= 3:
      return model
    if val_every and step % val_every == 0:
      val = evaluate(model, val_loader, device)
      history.append({"step": step, "val_loss": val})
      print(f"step {step} val loss {val:.4f} (no-change forecast {baseline:.4f})", flush=True)
    if step % ckpt_every == 0:
      save_checkpoint(ckpt_path, model, opt, step, history, run_meta)
    if max_minutes and (time.time() - start) / 60 > max_minutes:
      print(f"stopping at step {step}: reached the {max_minutes}-minute budget")
      break

  save_checkpoint(ckpt_path, model, opt, step, history, run_meta)
  save_backbone(out_path, model, extra={"steps": step, "context": context, "history": history, "patcher": patcher,
                                         "cutrule_fp": run_meta["cutrule_fp"]})
  print(f"saved backbone to {out_path} after {step} steps ({(time.time() - start) / 60:.1f} min)")
  return model


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  add_quick_flags(p)
  p.add_argument("--steps", type=int, default=5000)
  p.add_argument("--context", type=int, default=2048)
  p.add_argument("--global-batch", type=int, default=64,
                 help="sequences per optimizer step (paper: 256; 64 gives 4x more updates per hour on one laptop GPU)")
  p.add_argument("--micro-batch", default="auto",
                 help="sequences per forward pass: 'auto' (default) finds the largest that fits the GPU, or give a number")
  p.add_argument("--max-minutes", type=float, default=None, help="stop (and save) after this many minutes")
  p.add_argument("--restart", action="store_true",
                 help="ignore the rolling checkpoint and start over (default: continue an interrupted run)")
  p.add_argument("--benchmark", action="store_true", help="time 3 optimizer steps, print speed and memory, exit")
  p.add_argument("--num-workers", type=int, default=None, help="dataloader processes (default: about half the cores, max 8)")
  p.add_argument("--patcher", choices=["power", "cutrule"], default="power",
                 help="patch rule to pretrain with (cutrule pretrains the bilevel CutRule on Time-300B first if needed)")
  p.add_argument("--out", default=None, help="default outputs/backbone_<patcher>.pt")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  use_quick_outputs(a.quick)
  warm_up()                                        # compile the patching kernels once, before dataloader workers start
  kw = dict(steps=a.steps, context=a.context, global_batch=a.global_batch,
            micro_batch=a.micro_batch if a.micro_batch == "auto" else int(a.micro_batch), num_workers=a.num_workers)
  if a.quick:
    banner(True)
    kw.update(BACKBONE_PRETRAIN)
    kw["cutrule_kwargs"] = dict(CUTRULE_PRETRAIN)
  pretrain_backbone(out_path=a.out, patcher=a.patcher, max_minutes=a.max_minutes, resume=not a.restart,
                    benchmark=a.benchmark, seed=a.seed, **kw)


if __name__ == "__main__":
  main()
