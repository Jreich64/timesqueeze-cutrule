"""Method 5: the paper's power rule with its threshold tau as the FOLLOWER and forecasting as the LEADER (bilevel).

The power rule (patching/power_rule.py) starts a patch where |x_i - x_{i-1}| > tau * sqrt(local power). The paper fixes
tau = 0.3 (about 4x compression). Here tau is chosen by a follower that reacts to the forecaster, and the forecaster
(the leader) is trained knowing that tau will re-tune to it - a Stackelberg game:

  follower (lower level), given the forecaster weights theta:
      tau*(theta) = argmin_tau  f(theta, tau) = L(theta, tau) + lam * (log c(tau) - log c_target)^2 + mu * (tau - tau_prev)^2
        L  - the forecasting loss of the current forecaster on a batch patched with threshold tau,
        c  - the compression (ticks per patch) that tau produces, held near c_target (the paper's 4x) by lam,
        mu - a proximal term that keeps tau from jumping between steps.
  leader (upper level):
      min_theta  F(theta) = L(theta, tau*(theta)),
      dF/dtheta = dL/dtheta  +  dL/dtau * dtau*/dtheta,
      dtau*/dtheta = -(d2f/dtau dtheta) / (d2f/dtau2)            (implicit function theorem at the follower's optimum)

tau is one scalar and patch boundaries are hard decisions, so every derivative in tau is a central finite difference on
ONE fixed batch (the same windows for every tau, so the differences measure tau and not sampling noise). The step e is a
fixed fraction of tau (default 25%), so it stays measurable at every scale of tau. On the ORCL log price path the
paper's 0.3 compresses ~5.3-5.6x and ~0.1 about 3.2x, so 4x lies in between; boundaries react to tau there and the
forecast loss pulls on it alongside the compression term:
      dL/dtau        ~ (L(tau+e) - L(tau-e)) / 2e                      two forward passes
      d2f/dtau2      ~ (f(tau+e) - 2 f(tau) + f(tau-e)) / e^2          (floored at `tik`, as in cutrule/bilevel.py)
      d2f/dtau dtheta ~ (grad_theta L(tau+e) - grad_theta L(tau-e)) / 2e   two backward passes (the penalties do not
                                                                       depend on theta)
A weight that takes no part in any of the three backward passes (an MoE expert no token was routed to, the timing input
of the evenly-spaced variant) gets no gradient (None), so AdamW skips it exactly as in the other training loops; a zero
would still move it through weight decay and Adam momentum.
The follower itself is solved by a bounded golden-section search around the previous tau (no derivatives needed).

Training runs on the ORCL training days, starting from the backbone pretrained with the power rule, and is evaluated on
the test days with the learned tau, like `finetune --patcher power`. Patch boundaries are computed in the main process at
batch time (tau changes every step; dataloader workers would patch with a stale tau).
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from ..data.orcl import load_orcl
from ..data.orcl_windows import OrclWindows, collate, make_loader
from ..model.backbone import get_backbone
from ..model.forecaster import HORIZONS, forecasting_loss, run_model
from ..patching.kernels import warm_up
from ..patching.power_rule import power_rule_boundaries
from ..utils.paths import output_dir, use_quick_outputs, variant_suffix
from ..utils.quick import BACKBONE_PRETRAIN, POWER_BILEVEL, add_quick_flags, add_spacing_flag, banner, spacings
from ..utils.reporting import plot_lines, save_csv, write_forecast_report
from ..utils.seed import ignore_deprecation_warnings, set_seed
from .common import FINETUNE_LR, VAL_EVERY_WINDOWS, BestOnValidation, make_adamw, split_batch, to_device
from .evaluation import collect_forecasts, evaluate_last_position, print_test_scores

GOLDEN = (math.sqrt(5) - 1) / 2


# ---------------------------------------------------------------------------------------------- the bilevel mathematics
def golden_section(fn, lo, hi, evals):
  """Minimise a 1-D function on [lo, hi] with `evals` evaluations (golden-section search; no derivatives, robust to
  the small steps of a piecewise-constant loss). Returns (best x, best f(x))."""
  a, b = lo, hi
  x1, x2 = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
  f1, f2 = fn(x1), fn(x2)
  best = min((f1, x1), (f2, x2))
  for _ in range(max(evals - 2, 0)):
    if f1 <= f2:
      b, x2, f2 = x2, x1, f1
      x1 = b - GOLDEN * (b - a)
      f1 = fn(x1)
      best = min(best, (f1, x1))
    else:
      a, x1, f1 = x1, x2, f2
      x2 = a + GOLDEN * (b - a)
      f2 = fn(x2)
      best = min(best, (f2, x2))
  return best[1], best[0]


def implicit_tau_gradient(grad_at, loss_at, objective_at, tau, eps, tik=1e-3):
  """Leader gradient dF/dtheta at the follower's solution tau (see the module docstring).

  grad_at(t)      -> list of tensors (None for a weight not used at t), grad_theta L(theta, t)
  loss_at(t)      -> float, L(theta, t)
  objective_at(t) -> float, the follower objective f(theta, t)
  Returns (total gradient [list; None = the weight was used at no tau], info dict with the pieces, for logging)."""
  direct = grad_at(tau)
  g_plus = grad_at(tau + eps)
  g_minus = grad_at(tau - eps)
  unused = [d is None and a is None and b is None for d, a, b in zip(direct, g_plus, g_minus)]
  ref = [None if u else next(g for g in trio if g is not None) for u, trio in zip(unused, zip(direct, g_plus, g_minus))]
  z = lambda g, r: torch.zeros_like(r) if g is None else g          # used at another tau but not at this one: 0
  direct = [None if u else z(d, r) for d, r, u in zip(direct, ref, unused)]
  mixed = [None if u else (z(a, r) - z(b, r)) / (2 * eps)            # d2f / dtau dtheta
           for a, b, r, u in zip(g_plus, g_minus, ref, unused)]
  del g_plus, g_minus, ref
  dL_dtau = (loss_at(tau + eps) - loss_at(tau - eps)) / (2 * eps)
  f0, fp, fm = objective_at(tau), objective_at(tau + eps), objective_at(tau - eps)
  curvature = max((fp - 2 * f0 + fm) / eps ** 2, tik)                             # d2f / dtau2, floored
  scale = -dL_dtau / curvature                                                    # dL/dtau * dtau*/d(.)
  total = [None if d is None else d + scale * m for d, m in zip(direct, mixed)]
  norm = lambda gs: math.sqrt(sum(float((g.float() ** 2).sum()) for g in gs if g is not None))
  info = {"dL_dtau": dL_dtau, "curvature": curvature, "eps": eps, "unused_tensors": sum(unused),
          "rel": norm([None if m is None else scale * m for m in mixed]) / max(norm(direct), 1e-12)}
  return total, info


class TauFollower:
  """The follower: picks tau for a batch given the forecaster (golden section in a window around the previous tau)."""

  def __init__(self, tau=0.3, target_compression=4.0, lam=1.0, mu=1.0, tau_range=(0.02, 3.0), radius=0.25, evals=8):
    self.tau, self.target, self.lam, self.mu = tau, target_compression, lam, mu
    self.tau_range, self.radius, self.evals = tau_range, radius, evals

  def objective(self, loss, compression, tau, tau_prev):
    return loss + self.lam * (math.log(max(compression, 1e-6)) - math.log(self.target)) ** 2 + self.mu * (tau - tau_prev) ** 2

  def solve(self, objective_at):
    """Move tau to the minimiser of objective_at near the previous tau. Returns the new tau."""
    lo = max(self.tau_range[0], self.tau - self.radius)
    hi = min(self.tau_range[1], self.tau + self.radius)
    self.tau, _ = golden_section(objective_at, lo, hi, self.evals)
    return self.tau


# ---------------------------------------------------------------------------------------------- batches patched with tau
class TauBatch:
  """One fixed set of ORCL windows that can be patched with any tau (the same windows for every tau evaluated)."""

  def __init__(self, ds, idx, device):
    self.items = [ds.window(i) for i in idx]           # (sample dict without bounds, raw ctx, dtu, t, start)
    self.T, self.device, self._cache = ds.context, device, {}

  def batch(self, tau):
    """Collated batch on the device with power-rule patches at threshold tau, and its compression."""
    key = round(float(tau), 9)
    if key not in self._cache:
      samples = []
      for out, _, _, t, _ in self.items:
        s = dict(out)
        s["bounds"] = power_rule_boundaries(out["y"][:self.T], tau=tau, t=t)
        samples.append(s)
      b = to_device(collate(samples, self.T), self.device)
      self._cache = {key: (b, self.T * len(samples) / sum(len(s["bounds"]) for s in samples))}   # keep only the last
    return self._cache[key]


def batch_loss(model, b, with_aux=False):
  with torch.autocast(b["x"].device.type, dtype=torch.bfloat16, enabled=b["x"].is_cuda):
    preds, aux = run_model(model, b)
  loss, _ = forecasting_loss(preds, b["y"], b["n_valid"], aux if with_aux else None)
  return loss


def power_bilevel_step(model, opt, follower, tb, eps_rel=0.25, tik=1e-3, micro=8):
  """One bilevel step on the fixed batch `tb`: solve the follower, then one leader update with the finite-difference
  step eps = eps_rel * tau. Returns a log dict."""
  params = [p for p in model.parameters() if p.requires_grad]
  tau_prev = follower.tau

  @torch.no_grad()
  def loss_at(t):
    model.eval()                                       # dropout-free, deterministic evaluation of L(theta, t)
    out = float(batch_loss(model, tb.batch(t)[0]))
    model.train()
    return out

  def objective_at(t):
    return follower.objective(loss_at(t), tb.batch(t)[1], t, tau_prev)

  def grad_at(t, with_aux=False):
    """grad_theta L at threshold t, accumulated over micro-batches of `micro` windows (bounded GPU memory)."""
    full = tb.batch(t)[0]
    total, acc = len(full["y"]), None
    for mb in split_batch(full, micro):
      loss = batch_loss(model, mb, with_aux) * (len(mb["y"]) / total)
      grads = torch.autograd.grad(loss, params, allow_unused=True)    # None: e.g. an MoE expert no token was routed to
      grads = [None if g is None else g.detach() for g in grads]
      acc = grads if acc is None else [g if a_ is None else (a_ if g is None else a_ + g) for a_, g in zip(acc, grads)]
    return acc

  tau = follower.solve(objective_at)
  total, info = implicit_tau_gradient(lambda t: grad_at(t, with_aux=(t == tau)), loss_at, objective_at, tau,
                                      eps_rel * tau, tik)
  opt.zero_grad(set_to_none=True)
  for p, g in zip(params, total):
    p.grad = g                                                        # None: AdamW skips this weight this step
  gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], 1.0))
  opt.step()
  return {"tau": tau, "compression": tb.batch(tau)[1], "loss": loss_at(tau), "grad_norm": gnorm, **info}


# ---------------------------------------------------------------------------------------------- the training program
def power_bilevel(time_aware=True, context=512, stride=16, batch_size=16, epochs=5, steps=None, lr=FINETUNE_LR,
                  tau0=0.3, target_compression=4.0, lam=1.0, mu=1.0, eps_rel=0.25, follower_evals=8, eval_every=None,
                  data=None, mamba_cls=None, pretrain_kwargs=None, out_dir=None, results_dir=None, seed=0,
                  resume=True):
  """Fine-tune the power-rule backbone on ORCL with tau as the follower. Returns (model, results).
  resume=True (default): continues an interrupted run from its last validation checkpoint (every epoch) when it was
  written for exactly this run; the window order comes from the seed, so the resumed run sees the same batches."""
  out_dir = out_dir or output_dir()
  ignore_deprecation_warnings()
  set_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, _ = get_backbone(None, device, mamba_cls, patcher="power", **(pretrain_kwargs or {}))
  data = data or load_orcl()
  n, b1, b2 = len(data.returns), data.train_end, data.val_end
  mk = lambda lo, hi, st: OrclWindows(data, lo, hi, context, st, "power", time_aware=time_aware, tau=tau0)
  train_ds, val_ds, test_ds = mk(0, b1, stride), mk(b1, b2, 64), mk(b2, n, 64)
  steps_per_epoch = max(1, len(train_ds) // batch_size)
  steps = steps or epochs * steps_per_epoch
  tag = f"power_bilevel{variant_suffix(time_aware)}"
  print(f"{tag}: windows train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}, {steps} steps of {batch_size} "
        f"windows; tau starts at {tau0}, target compression {target_compression}x", flush=True)

  follower = TauFollower(tau0, target_compression, lam, mu, evals=follower_evals)
  opt = make_adamw(model, lr=lr)
  rng = np.random.default_rng(seed)
  order = np.concatenate([rng.permutation(len(train_ds)) for _ in range(steps * batch_size // len(train_ds) + 1)])
  history, t0, best = [], time.time(), BestOnValidation()
  eval_every = eval_every if eval_every is not None else max(1, VAL_EVERY_WINDOWS // batch_size)   # as finetune

  def validate(step):
    val_ds.tau = follower.tau
    val = evaluate_last_position(model, make_loader(val_ds, batch_size, False), device)
    better = best.update(val["loss"], step, model, follower.tau)              # selection keeps the matching tau
    print(f"step {step} val loss {val['loss']:.4f} (tau {follower.tau:.3f}){'  (best so far)' if better else ''}",
          flush=True)
    return val["loss"]

  run_id = {"time_aware": time_aware, "steps": steps, "batch_size": batch_size, "lr": lr, "tau0": tau0,
            "target": target_compression, "lam": lam, "mu": mu, "eps_rel": eps_rel, "seed": seed, "context": context,
            "n_train": len(train_ds)}
  ckpt = os.path.join(out_dir, f"{tag}_ckpt.pt")
  ck = torch.load(ckpt, map_location=device) if resume and os.path.exists(ckpt) else None
  start = 0
  if ck is not None and ck.get("run") == run_id and ck.get("best"):
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    start, history, follower.tau = ck["step"], ck["history"], ck["tau"]
    best = BestOnValidation.from_dict(ck["best"])
    print(f"resumed from {ckpt} at step {start} (tau {follower.tau:.3f})", flush=True)
  else:
    validate(0)                                                                # the backbone before any ORCL training
  model.train()
  for step in range(start, steps):
    tb = TauBatch(train_ds, order[step * batch_size:(step + 1) * batch_size], device)
    log = power_bilevel_step(model, opt, follower, tb, eps_rel)
    history.append({"step": step + 1, **log})
    if (step + 1) % max(1, steps // 50) == 0 or step + 1 == steps:
      eta = (steps - step - 1) * (time.time() - t0) / (step + 1) / 60
      print(f"step {step + 1}/{steps} tau {log['tau']:.3f} compression {log['compression']:.2f}x loss {log['loss']:.4f} "
            f"indirect/direct {log['rel']:.3f} ETA {eta:.0f} min", flush=True)
    if (eval_every and (step + 1) % eval_every == 0) or step + 1 == steps:
      history[-1]["val_loss"] = validate(step + 1)
      model.train()
      os.makedirs(out_dir, exist_ok=True)
      torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "tau": follower.tau, "step": step + 1,
                  "history": history, "run": run_id, "best": best.to_dict()}, ckpt + ".tmp")
      os.replace(ckpt + ".tmp", ckpt)                                   # single rolling checkpoint

  follower.tau = best.restore(model)                                          # best validation step and its tau
  test_ds.tau = follower.tau
  test = evaluate_last_position(model, make_loader(test_ds, batch_size, False), device)
  print(f"learned tau {follower.tau:.3f}")
  print_test_scores(test)
  os.makedirs(out_dir, exist_ok=True)
  out = os.path.join(out_dir, f"finetuned_{tag}.pt")
  torch.save({"model": model.state_dict(), "tau": follower.tau, "time_aware": time_aware, "history": history,
              "best_step": best.epoch, "test": test}, out)
  results_dir = results_dir or os.path.join(out_dir, "results", tag)
  write_forecast_report(results_dir, f"ORCL {tag}", collect_forecasts(model, test_ds, device), HORIZONS, None,
                        extra={"patcher": "power", "tau": follower.tau, "best_step": best.epoch, "time_aware": time_aware,
                               "compression": test["compression"], "test_windows_total": len(test_ds)})
  save_csv(os.path.join(results_dir, "tau_history.csv"), ["step", "tau", "compression", "loss", "eps", "dL_dtau",
                                                          "curvature", "indirect_over_direct", "grad_norm", "unused_tensors"],
           [[h["step"], h["tau"], h["compression"], h["loss"], h["eps"], h["dL_dtau"], h["curvature"], h["rel"],
             h["grad_norm"], h["unused_tensors"]] for h in history])
  steps_x = [h["step"] for h in history]
  plot_lines(os.path.join(results_dir, "tau_and_compression.png"),
             {"tau": (steps_x, [h["tau"] for h in history]),
              "compression / 10": (steps_x, [h["compression"] / 10 for h in history])},
             f"{tag}: follower tau and compression", "step", "")
  curves = {"train (batch, at tau*)": (steps_x, [h["loss"] for h in history])}
  val = [(h["step"], h["val_loss"]) for h in history if "val_loss" in h]
  if val:
    curves["validation"] = ([v[0] for v in val], [v[1] for v in val])
  plot_lines(os.path.join(results_dir, "loss_curves.png"), curves, f"{tag}: forecasting loss", "step", "loss")
  print(f"saved {out} and the report in {results_dir}")
  return model, {"history": history, "test": test, "tau": follower.tau, "results_dir": results_dir}


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  add_quick_flags(p)
  add_spacing_flag(p)
  p.add_argument("--epochs", type=int, default=5, help="passes over the ORCL training windows (as finetune)")
  p.add_argument("--steps", type=int, default=None, help="overrides --epochs")
  p.add_argument("--batch-size", type=int, default=16, help="windows per step (each step runs ~14 forward/3 backward passes)")
  p.add_argument("--context", type=int, default=512)
  p.add_argument("--lr", type=float, default=FINETUNE_LR)
  p.add_argument("--tau0", type=float, default=0.3, help="starting threshold (the paper's)")
  p.add_argument("--target-compression", type=float, default=4.0)
  p.add_argument("--lam", type=float, default=1.0, help="weight of the compression term in the follower objective")
  p.add_argument("--mu", type=float, default=1.0, help="proximal weight keeping tau near its previous value")
  p.add_argument("--eps-rel", type=float, default=0.25, help="finite-difference step in tau, as a fraction of tau")
  p.add_argument("--follower-evals", type=int, default=8, help="golden-section evaluations per follower solve")
  p.add_argument("--eval-every", type=int, default=None,
                 help="validation every n steps (default: every 6,400 windows, as finetune)")
  p.add_argument("--pretrain-steps", type=int, default=5000, help="steps to pretrain the power backbone if missing")
  p.add_argument("--restart", action="store_true", help="start an interrupted run over instead of continuing it")
  a = p.parse_args()
  use_quick_outputs(a.quick)
  warm_up()
  kw = dict(epochs=a.epochs, steps=a.steps, batch_size=a.batch_size, context=a.context, lr=a.lr, tau0=a.tau0,
            target_compression=a.target_compression, lam=a.lam, mu=a.mu, eps_rel=a.eps_rel, follower_evals=a.follower_evals,
            eval_every=a.eval_every, pretrain_kwargs={"steps": a.pretrain_steps}, resume=not a.restart)
  if a.quick:
    banner(True)
    kw.update(POWER_BILEVEL, pretrain_kwargs=dict(BACKBONE_PRETRAIN))
  for time_aware in spacings(a.spacing):
    power_bilevel(time_aware, **kw)


if __name__ == "__main__":
  main()
