"""Bilevel training of the CutRule (stages 1 and 3): the leader is the MLP weights, the follower is the threshold b2.

The two levels
  * follower (inner problem): with the bars held fixed, b2 is moved to the minimum of follower_loss - the expected bar
    duration of the soft rule against a target, plus a proximal term lam * (b2 - b2_prev)^2 that keeps it near its last
    value. It is one-dimensional, so solve_follower uses damped Newton steps with a backtracking line search.
  * leader (outer problem): the remaining weights minimise leader_loss - how regular the volatility of each bar is, plus
    a penalty that keeps the cut rate near its budget - *taking into account that the follower will react*.

One epoch (epoch())
  (a) segment the series with the current hard rule and build the soft feature windows that start at every bar,
  (b) solve the follower,
  (c) hypergradient of the leader (implicit-function theorem, eq. 7):
        dF/dtheta = direct + indirect,   indirect = -(dF/db2) / (d2f/db2^2) * d2f/(db2 dtheta)
      (bilevel=False drops the indirect term: plain gradient descent on the leader, the ablation),
  (d) clip the gradient to unit norm and take one Adam step on the leader.

train() runs the epochs with a single rolling checkpoint and a rollback guard: an update that leaves no bars at all is
undone, the learning rate halved and the epoch retried.
"""
import copy
import math
import os

import torch

from .model import CutRule


def solve_follower(model, x, r2, mask, dt, b2_prev, lam, iters=30, max_step=1.0, tol=1e-5):
  """Move the follower b2 to the minimum of its (one-dimensional) loss with the segmentation held fixed.

  Damped Newton steps with a backtracking line search that only accepts steps that lower the loss. The duration
  loss varies over orders of magnitude when tick gaps are real (expected bar durations from ~0 to thousands of tick
  units), so the fixed-step gradient descent used before overshot and oscillated; here the loss never increases."""
  def loss():
    d_soft, _, _ = model.soft_stats(x, r2, mask, dt)
    return model.follower_loss(d_soft, b2_prev, lam)

  for _ in range(iters):
    f = loss()
    g, = torch.autograd.grad(f, model.b2, create_graph=True)
    h, = torch.autograd.grad(g.sum(), model.b2)
    if g.abs().max().item() < tol:
      break
    step = (g / (h.clamp(min=0.0) + 1.0)).detach()                  # damped Newton direction (>= gradient step / 1)
    step = step.clamp(-max_step, max_step)
    f0, b0, accepted = f.item(), model.b2.detach().clone(), False
    for _ in range(14):                                              # backtrack until the loss decreases
      with torch.no_grad():
        model.b2.copy_(b0 - step)
        if loss().item() <= f0:
          accepted = True
          break
      step = step * 0.5
    if not accepted:
      with torch.no_grad():
        model.b2.copy_(b0)
      break


def epoch(model, opt, bilevel=True, inner_steps=30, lam=0.5, tik=0.01):
  # (a) hard segmentation with the current weights, plus feature windows
  out = model.segment_data_helper(require_free=False)    # a start with only the forced day-end bars can still bootstrap
  if out is None:
    return None                                    # collapsed: 0 tokens
  starts, ends, x, r2, mask, in_hard, dt = out

  # (b) follower: move b2 toward its optimum, segmentation held fixed
  b2_prev = model.b2.detach().clone()
  solve_follower(model, x, r2, mask, dt, b2_prev, lam, iters=inner_steps)

  # (c) leader hypergradient, eq. (7)
  theta = model.leader_params()
  d_soft, var_soft, p_cut = model.soft_stats(x, r2, mask, dt)
  f = model.follower_loss(d_soft, b2_prev, lam)
  F = model.leader_loss(d_soft, var_soft, p_cut, in_hard, len(starts), dt)

  g_psi, = torch.autograd.grad(f, model.b2, create_graph=True)              # df/db2
  H_pp = torch.autograd.grad(g_psi.sum(), model.b2, retain_graph=True)[0]    # d2f/db2^2
  H_pp = (H_pp.detach() + tik).clamp(min=tik)
  H_pt = torch.autograd.grad(g_psi.sum(), theta, retain_graph=True)          # d2f/db2 dtheta
  *direct, dF_db2 = torch.autograd.grad(F, theta + [model.b2])              # dF/dtheta, dF/db2

  indirect = [-(dF_db2 / H_pp) * h for h in H_pt]
  total = [a + b for a, b in zip(direct, indirect)] if bilevel else direct

  # (d) clip to unit norm, one Adam step on the leader
  norm = torch.sqrt(sum((g**2).sum() for g in total)).item()
  rel = (torch.sqrt(sum((g**2).sum() for g in indirect)) /
         torch.sqrt(sum((g**2).sum() for g in direct))).item()
  if not math.isfinite(norm):
    raise FloatingPointError("non-finite gradient in the bilevel update: the objective diverged")
  scale = 1.0 / norm if norm > 1.0 else 1.0                          # clip to unit norm (a zero gradient is fine)
  opt.zero_grad()
  for p, g in zip(theta, total):
    p.grad = g * scale
  opt.step()

  return {"F": F.item(), "bars": len(starts), "grad_norm": norm, "rel": rel, "b2": model.b2.item()}


def save_checkpoint(path, model, opt, trace, bilevel, fingerprint=None):
  # Write to a temp file then swap it in, so only the latest checkpoint exists and a crash never leaves a partial one.
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "trace": trace,
              "epoch": len(trace), "bilevel": bilevel, "fingerprint": fingerprint}, tmp)
  os.replace(tmp, path)


def _snapshot(model, opt):
  return ({k: v.detach().clone() for k, v in model.state_dict().items()}, copy.deepcopy(opt.state_dict()))


def _restore(model, opt, snap):
  model.load_state_dict(snap[0])
  opt.load_state_dict(snap[1])


def train(bilevel, returns, epochs=60, verbose=True, ckpt_path=None, ckpt_every=10, resume=False, day_starts=None,
          fingerprint=None, dt=None, init_state=None, max_rollbacks=8):
  """Bilevel CutRule training. If an update leaves no bars at all (only possible without forced day-end cuts), the
  weights from before that epoch are restored, the learning rate is halved and the epoch is retried. Raises only if the
  starting weights give no bars, or after max_rollbacks failures of the same epoch.

  Note: a hard rule that momentarily cuts nothing on its own is normal here (the follower matches the *expected* duration
  of the soft rule, which pushes the logits negative) and is not treated as a collapse. Judge the finished rule with
  CutRule.has_free_cuts() (stages.require_free_cuts does this for real runs)."""
  model = CutRule(returns, day_starts=day_starts, dt=dt)
  if init_state is not None:                       # e.g. the CutRule pretrained on Time-300B
    model.load_state_dict(init_state)
  opt = torch.optim.Adam(model.leader_params(), lr=1e-2)   # leader only; b2 is moved in step (b)
  trace = []
  if resume and ckpt_path and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=model.device)
    if fingerprint and ckpt.get("fingerprint") != fingerprint:
      raise RuntimeError(f"{ckpt_path} was trained on different data; delete it or pass a different --ckpt")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    trace = ckpt["trace"]
    print(f"Resumed from {ckpt_path} at epoch {len(trace)}")
  before_last, rollbacks = None, {}                 # state before the last successful epoch's update; rollbacks per epoch
  while len(trace) < epochs:
    ep = len(trace)
    snap = _snapshot(model, opt)
    info = epoch(model, opt, bilevel=bilevel)       # None only if there are no bars at all (nothing to differentiate)
    if info is None:
      if before_last is None:
        raise RuntimeError(f"CutRule produced 0 bars at epoch {ep}; the initial weights never cut. Try a different --seed.")
      rollbacks[ep - 1] = rollbacks.get(ep - 1, 0) + 1   # counted per epoch index and never reset: guarantees termination
      if rollbacks[ep - 1] > max_rollbacks:
        raise RuntimeError(f"CutRule kept collapsing to 0 bars after epoch {ep - 1} despite {max_rollbacks} rollbacks")
      _restore(model, opt, before_last)
      for g in opt.param_groups:
        g["lr"] *= 0.5
      trace.pop()                                   # redo the epoch whose update collapsed, with a smaller step
      if verbose:
        print(f"  epoch {ep - 1}: its update collapsed the CutRule to 0 bars; rolled back, learning rate now "
              f"{opt.param_groups[0]['lr']:.1e}")
      continue
    before_last = snap
    trace.append(info)
    if verbose and (ep % 10 == 0 or ep == epochs - 1):
      print(f"{'Bilevel' if bilevel else 'Direct'} ep {ep}: {info}")
    if ckpt_path and ((ep + 1) % ckpt_every == 0 or ep == epochs - 1):
      save_checkpoint(ckpt_path, model, opt, trace, bilevel, fingerprint)
  return model, trace
