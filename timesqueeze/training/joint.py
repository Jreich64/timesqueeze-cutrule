"""Methods 3 and 4: the CutRule keeps learning from forecast feedback while the forecaster trains, in the same two phases
and with the same budgets as the staged pipeline (methods 1 and 2), so the methods are compared on equal training.

  phase A - Time-300B   (like `pretrain`)  5,000 steps x 64 windows, context 2048, lr 1e-3 with 10% warm-up and cosine
                        decay; the forecaster starts from random weights, the CutRule from the stage-1 CutRule pretrained
                        on Time-300B (the one method 2 uses). Shared by both spacing variants, like the backbones
                        (Time-300B has no timestamps): outputs/<joint|trilevel>_pretrained.pt.
  ORCL CutRule adaptation (like stage 3)   30 bilevel epochs on the ORCL training days, from the phase-A CutRule.
  phase B - ORCL        (like `finetune`)  5 epochs over the same training windows (context 512, every 16 ticks),
                        32 windows per step, lr 1e-5, validation every 6,400 windows; per spacing variant:
                        outputs/<joint|trilevel>_finetuned[_ticks].pt.
  evaluation            exactly as `finetune`: the best validation epoch (0 included), then last-position forecasts on
                        the held-out test days, the same report.

What differs from method 2 is only how the CutRule learns while the forecaster trains:

  Method 3 (`joint`, leader "reinforce"). Patch boundaries are hard decisions, so the forecast loss has no gradient into
  the CutRule. It gets forecast feedback by REINFORCE instead: it cuts stochastically, p = sigmoid(logit / temp) per tick;
  every window is patched K = 2 ways, both patchings go through the forecaster, and the one with the lower forecast loss
  has its cut decisions reinforced:  grad = E[(loss_k - mean_k loss) * grad log p(cuts_k)].  Comparing two patchings of
  the *same* window removes how hard the window is. Forecast loss alone always prefers more tokens, so every round
  also runs bilevel epochs of the original CutRule objective (target bar duration and volatility) on that phase's data.

  Method 4 (`trilevel`, leader "trilevel", cutrule/trilevel.py). The CutRule is a three-level problem: duration (b2)
  and volatility (W2) are lower levels re-solved at every step on a small corpus of that phase's data, and the leader
  (the hidden layer) follows the REINFORCE forecast gradient *through* those solutions by implicit differentiation, so
  it can only move cuts around under the duration and volatility conditions.

Cost: every window is forecast twice (two patchings), so each step does twice the forecaster work of `pretrain` /
`finetune`, and patchings are sampled in the main process from the *current* CutRule (on-policy, no stale workers).
"""
import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..cutrule.bilevel import train as train_bilevel
from ..cutrule.model import CutRule
from ..cutrule.stages import CutRulePatcher, build_corpus, get_pretrained_cutrule, increments_as_returns
from ..cutrule.trilevel import TrilevelCutRule, cut_grads
from ..data.orcl import load_orcl
from ..data.orcl_windows import OrclWindows, make_loader
from ..data.orcl_windows import collate as collate_single
from ..data.time300b_windows import Time300BSegments, Time300BWindows
from ..data.time300b_windows import make_loader as make_time_loader
from ..model.backbone import build_model
from ..model.forecaster import HORIZONS, per_sample_loss, run_model
from ..patching.batching import collate_patches
from ..patching.kernels import warm_up
from ..patching.power_rule import power_rule_boundaries
from ..patching.segmenter import CutRuleSegmenter
from ..utils.paths import DATA_DIR, out, output_dir, use_quick_outputs, variant_suffix
from ..utils.quick import JOINT, TRILEVEL, add_quick_flags, add_spacing_flag, banner, spacings
from ..utils.reporting import plot_lines, save_csv, write_forecast_report
from ..utils.seed import ignore_deprecation_warnings, set_seed
from .common import FINETUNE_LR, LR, ORCL_BATCH, VAL_EVERY_WINDOWS, BestOnValidation, lr_at, make_adamw, to_device
from .evaluation import collect_forecasts, evaluate_last_position, print_test_scores
from .pretrain import evaluate as evaluate_time300b

K = 2                                                    # sampled patchings per window


# ---------------------------------------------------------------------------------------------- datasets with patchings
class JointOrclWindows(OrclWindows):
  """ORCL windows, each patched `k` times by sampling the CutRule policy."""

  def __init__(self, *args, k=K, seed=0, **kw):
    super().__init__(*args, **kw)
    self.k, self.seed, self._draws = k, seed, 0

  def __getitem__(self, i):
    out_, raw_ctx, dtu, t, s = self.window(i)
    T = self.context
    rng = np.random.default_rng([self.seed, i, self._draws])
    self._draws += 1
    out_["variants"] = [self.segmenter.sample_boundaries(raw_ctx, self.slow[s:s + T], self.fast[s:s + T], dtu, rng)
                        for _ in range(self.k)]
    return out_


class JointTimeWindows(Time300BWindows):
  """Time-300B windows, each patched `k` times by sampling the CutRule policy on the window's increments (dt = 1)."""

  def __init__(self, segments, context, segmenter, return_scale, k=K, **kw):
    super().__init__(segments, context, HORIZONS, "train", patcher=power_rule_boundaries, **kw)
    self.segmenter, self.return_scale, self.k = segmenter, return_scale, k

  def make_sample(self, y, n, n_in, rng):
    r = increments_as_returns(y[:n_in], self.return_scale)
    ones = np.ones(n_in)                                     # Time-300B is regular: dt = 1, as in stage 1
    slow, fast = self.segmenter.ewma_features(r, None, ones)
    return {"y": y, "n_valid": n, "n_in": n_in,
            "variants": [self.segmenter.sample_boundaries(r, slow, fast, ones, rng) for _ in range(self.k)]}


def collate_joint(samples, context):
  """Batch of windows -> batch of window*k rows (consecutive rows are the k patchings of one window)."""
  if "variants" not in samples[0]:
    return collate_single(samples, context)
  rows = [(s, v) for s in samples for v in s["variants"]]
  y = torch.tensor(np.stack([s["y"] for s, _ in rows]))
  pos, mask, ids = collate_patches([v[0] for _, v in rows], context)
  N = len(rows)
  feats = np.zeros((N, context, 4), dtype=np.float32)
  cuts = np.zeros((N, context), dtype=np.float32)
  tick = np.zeros((N, context), dtype=bool)
  for j, (s, (_, f, c)) in enumerate(rows):
    feats[j, :len(f)], cuts[j, :len(c)], tick[j, :len(f)] = f, c, True
  batch = {"y": y, "n_valid": torch.tensor([s["n_valid"] for s, _ in rows]), "x": y[:, :context],
           "patch_pos": torch.tensor(pos), "patch_mask": torch.tensor(mask), "patch_id": torch.tensor(ids),
           "feats": torch.tensor(feats), "cuts": torch.tensor(cuts), "tick_mask": torch.tensor(tick)}
  if "time" in samples[0]:
    batch["time"] = torch.tensor(np.stack([s["time"] for s, _ in rows]))
    batch["dtf"] = torch.tensor(np.stack([s["dtf"] for s, _ in rows]))
  return batch


def _forever(ds, windows, context, shuffle):
  """Endless stream of collated batches of `windows` windows (main process: patchings are always on-policy)."""
  loader = DataLoader(ds, batch_size=windows, shuffle=shuffle, num_workers=0,
                      collate_fn=partial(collate_joint, context=context), drop_last=shuffle)
  while True:
    for batch in loader:
      yield batch


# ---------------------------------------------------------------------------------------------- REINFORCE
def cut_log_prob(cut, batch):
  """Mean per-tick log-probability, under the current CutRule, of the cuts that were actually sampled. [N]
  Ticks where the length cap ended the bar (cuts == -1) took no decision and are left out."""
  z = cut(batch["feats"]).squeeze(-1) / cut.temp
  c = batch["cuts"].clamp(min=0.0)
  logp = c * F.logsigmoid(z) + (1.0 - c) * F.logsigmoid(-z)
  m = batch["tick_mask"].float() * (batch["cuts"] >= 0).float()
  return (logp * m).sum(1) / m.sum(1).clamp(min=1.0)


def advantage_magnitude(window_losses, k):
  """Mean |loss - mean of its k sibling patchings| (a tensor, no host sync)."""
  adv = window_losses.detach().view(-1, k)
  return (adv - adv.mean(1, keepdim=True)).abs().mean().clamp(min=1e-8)


def reinforce_loss(logp, window_losses, k, scale=None):
  """Score-function loss for the CutRule. Rows are grouped in consecutive blocks of k patchings of the same window;
  a patching with a higher forecast loss than its siblings has its log-probability pushed down, a lower one up.
  `scale` normalises the advantages: pass a running average of advantage_magnitude (a micro-batch has only a few
  windows, and normalising by its own std would blow up pairs that differ only by noise); None = batch std."""
  adv = window_losses.detach().view(-1, k)
  adv = (adv - adv.mean(1, keepdim=True)).view(-1)
  adv = adv / (scale if scale is not None else adv.std() + 1e-8)
  return (adv * logp).mean()


# ---------------------------------------------------------------------------------------------- CutRule helpers
def run_bilevel(cut, returns, day_starts, dt, epochs, seed):
  """Train `cut` in place with the bilevel objective, keeping the previous weights if it collapses (0 bars)."""
  before = {k: v.detach().clone() for k, v in cut.state_dict().items()}
  try:
    set_seed(seed)
    model, trace = train_bilevel(True, returns, epochs=epochs, verbose=False, day_starts=day_starts, dt=dt,
                                 init_state=before)
    cut.load_state_dict(model.state_dict())
    return trace[-1] if trace else None
  except RuntimeError as e:
    print(f"  bilevel phase skipped: {e}")
    cut.load_state_dict(before)
    return None


def cutrule_on(device, state):
  """A CutRule on `device` with the given weights (the series it carries is irrelevant here: only weights are used)."""
  cut = CutRule(np.ones(2), device=device)
  cut.load_state_dict(state)
  return cut


def method_name(leader):
  return "trilevel" if leader == "trilevel" else "joint"


# ---------------------------------------------------------------------------------------------- one training phase
class JointPhase:
  """The shared step of both phases: forecaster update on K patchings per window + the CutRule's update.

  data_fn(rnd, segmenter) -> dataset of windows with sampled patchings for round `rnd`;
  bilevel_fn(rnd) -> (returns, day_starts, dt) corpus for method 3's per-round bilevel epochs (on the phase's data);
  lower_fn(rnd) -> the same, smaller, for method 4's lower levels."""

  def __init__(self, model, opt, cut, leader, device, context, windows_per_step, micro_windows, cut_lr=1e-3,
               bilevel_epochs=3, bars=2000, inner_iters=3, seed=0):
    self.model, self.opt, self.cut, self.device = model, opt, cut, device
    self.trilevel = leader == "trilevel"
    self.context, self.seed = context, seed
    self.micro, self.accum = micro_windows, max(1, windows_per_step // micro_windows)
    self.bilevel_epochs, self.bars, self.inner_iters = bilevel_epochs, bars, inner_iters
    # method 3 updates the hidden layer and readout by REINFORCE; method 4 only the hidden layer (theta): the
    # readout (volatility level) and bias (duration level) are re-solved by their own objectives
    params = list(cut.trunk.parameters()) if self.trilevel else cut.leader_params()
    self.cut_opt = torch.optim.Adam(params, lr=cut_lr)
    self.adv_scale = None                                # running scale of the REINFORCE advantages
    self.tri = None

  def start_round(self, rnd, data_fn, bilevel_fn, lower_fn, shuffle):
    if not self.trilevel and rnd > 0 and self.bilevel_epochs:          # keep the original CutRule objective in force
      info = run_bilevel(self.cut, *bilevel_fn(rnd), self.bilevel_epochs, self.seed + rnd)
      if info:
        print(f"  round {rnd}: bilevel epochs -> {info['bars']} bars", flush=True)
    if self.trilevel:
      self.tri = TrilevelCutRule(self.cut, *lower_fn(rnd), bars=self.bars, inner_iters=self.inner_iters,
                                 seed=self.seed + rnd)
    self.ds = data_fn(rnd, CutRuleSegmenter(self.cut))
    self.stream = _forever(self.ds, self.micro, self.context, shuffle)

  def step(self, lr):
    """One optimizer step (accum micro-batches of micro windows x K patchings). Returns the mean forecast loss."""
    for g in self.opt.param_groups:
      g["lr"] = lr
    solved = self.tri.solve_lower_levels() if self.trilevel else False
    self.ds.segmenter = CutRuleSegmenter(self.cut)                    # on-policy: the CutRule as it is now
    losses_log = []
    for _ in range(self.accum):
      b = to_device(next(self.stream), self.device)
      with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
        preds, aux = run_model(self.model, b)
      losses = per_sample_loss(preds, b["y"], b["n_valid"])
      ((losses.mean() + 0.02 * aux) / self.accum).backward()
      losses_log.append(losses.mean().detach())
      mag = advantage_magnitude(losses, K)
      self.adv_scale = mag if self.adv_scale is None else 0.9 * self.adv_scale + 0.1 * mag
      (reinforce_loss(cut_log_prob(self.cut, b), losses, K, self.adv_scale) / self.accum).backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.opt.step()
    self.opt.zero_grad(set_to_none=True)
    if self.trilevel:
      g_theta, g_phi, g_psi = cut_grads(self.cut)
      for p_ in self.cut.parameters():
        p_.grad = None
      if solved:
        d_theta = self.tri.leader_gradient(g_theta, g_phi, g_psi)     # through psi*(phi*(theta), theta), phi*(theta)
        h_ = self.cut.hidden
        self.cut.trunk.weight.grad = d_theta[:4 * h_].view(h_, 4).to(self.cut.trunk.weight.device)
        self.cut.trunk.bias.grad = d_theta[4 * h_:].to(self.cut.trunk.bias.device)
        torch.nn.utils.clip_grad_norm_(list(self.cut.trunk.parameters()), 1.0)
        self.cut_opt.step()
        self.cut_opt.zero_grad(set_to_none=True)
    else:
      torch.nn.utils.clip_grad_norm_(self.cut.leader_params(), 1.0)
      self.cut_opt.step()
      for p_ in self.cut.parameters():
        p_.grad = None
    return torch.stack(losses_log).mean()

  def state(self):
    return {"cut": self.cut.state_dict(), "cut_opt": self.cut_opt.state_dict(), "adv_scale": self.adv_scale}

  def load(self, st):
    self.cut.load_state_dict(st["cut"])
    self.cut_opt.load_state_dict(st["cut_opt"])
    self.adv_scale = st.get("adv_scale")


# ---------------------------------------------------------------------------------------------- phase A: Time-300B
def pretrained_joint_path(leader):
  return out(f"{method_name(leader)}_pretrained.pt")


def joint_pretrain(leader="reinforce", steps=5000, rounds=20, global_batch=64, micro_windows=2, context=2048, lr=LR,
                   cut_lr=1e-3, bilevel_epochs=3, corpus_series=300, corpus_len=1024, bars=2000, inner_iters=3,
                   val_every_round=True, resume=True, ckpt_every=50, seed=0, mamba_cls=None, data_dir=None,
                   cutrule_kwargs=None, out_path=None):
  """Phase A on Time-300B (see the module docstring). Saves and returns (model, cut, return_scale, history).
  A rolling checkpoint every ckpt_every steps; resume=True (default) continues an interrupted run from it."""
  ignore_deprecation_warnings()
  set_seed(seed)
  name = method_name(leader)
  out_path = out_path or pretrained_joint_path(leader)
  ckpt_path = out(f"{name}_pretrain_ckpt.pt")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pre_cut, scale, cut_fp = get_pretrained_cutrule(**(cutrule_kwargs or {}))       # the stage-1 CutRule (shared)
  cut = cutrule_on(device, pre_cut.state_dict())
  model = build_model(mamba_cls).to(device)
  opt = make_adamw(model, lr=lr)
  phase = JointPhase(model, opt, cut, leader, device, context, global_batch, micro_windows, cut_lr, bilevel_epochs,
                     bars, inner_iters, seed)
  segments = Time300BSegments(data_dir or DATA_DIR)
  step, history = 0, []
  if resume and os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); phase.load(ck)
    step, history = ck["step"], ck["history"]
    print(f"resumed from {ckpt_path} at step {step}")
  per_round = max(1, steps // rounds)
  print(f"{name} phase A (Time-300B): {steps} steps x {global_batch} windows x {K} patchings, context {context}, "
        f"{len(segments.lengths)} series pieces; starting from the stage-1 CutRule {cut_fp}", flush=True)

  def corpus(rnd, n_series):
    r, starts = build_corpus(segments, n_series, corpus_len, scale, seed + 1000 + rnd)
    return r, starts, np.ones(len(r))

  data_fn = lambda rnd, seg: JointTimeWindows(segments, context, seg, scale, seed=seed * 1000 + rnd)
  val_ds = lambda: Time300BWindows(segments, context, HORIZONS, "val", patcher=CutRulePatcher(cut, scale))  # greedy
  t0, step0 = time.time(), step
  model.train()

  def save_ckpt():
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "history": history,
                **phase.state()}, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)                         # single rolling checkpoint

  for rnd in range(step // per_round, rounds):
    phase.start_round(rnd, data_fn, lambda r: corpus(r, corpus_series), lambda r: corpus(r, max(1, corpus_series // 4)),
                      shuffle=False)
    while step < min((rnd + 1) * per_round, steps):
      loss = phase.step(lr_at(step, steps, max(1, steps // 10), lr))
      step += 1
      if step % 10 == 0 or step == steps:
        loss = float(loss)
        history.append({"step": step, "loss": loss})
        eta = (steps - step) * (time.time() - t0) / (step - step0) / 3600
        print(f"step {step}/{steps} loss {loss:.4f} lr {opt.param_groups[0]['lr']:.2e} ETA {eta:.1f}h", flush=True)
      if step % ckpt_every == 0:
        save_ckpt()
    if val_every_round:
      val = evaluate_time300b(model, make_time_loader(val_ds(), micro_windows * K, num_workers=0), device)
      history.append({"step": step, "val_loss": val})
      print(f"step {step} val loss {val:.4f} (greedy patches of the current CutRule)", flush=True)
    save_ckpt()
  torch.save({"model": model.state_dict(), "cut": cut.state_dict(), "return_scale": scale, "steps": step,
              "history": history, "leader": leader, "start_cutrule": cut_fp}, out_path + ".tmp")
  os.replace(out_path + ".tmp", out_path)
  print(f"saved {out_path} after {step} steps ({(time.time() - t0) / 3600:.1f} h)", flush=True)
  return model, cut, scale, history


def get_joint_pretrained(leader, device, mamba_cls=None, **pretrain_kwargs):
  """Phase A's result (forecaster + CutRule), loaded, or trained first if missing. Returns (model, cut_state, scale)."""
  path = pretrained_joint_path(leader)
  if not os.path.exists(path):
    print(f"no {path}; running phase A on Time-300B first")
    joint_pretrain(leader, mamba_cls=mamba_cls, **pretrain_kwargs)
  saved = torch.load(path, map_location="cpu")
  model = build_model(mamba_cls)
  model.load_state_dict(saved["model"])
  print(f"loaded {path} ({saved['steps']} Time-300B steps)")
  return model.to(device), saved["cut"], saved["return_scale"]


# ---------------------------------------------------------------------------------------------- phase B: ORCL
def joint_train(leader="reinforce", time_aware=True, epochs=5, batch_windows=ORCL_BATCH, micro_windows=8, context=512,
                stride=16, lr=FINETUNE_LR, cut_lr=1e-3, cutrule_epochs=30, bilevel_epochs=3, bars=2000,
                inner_iters=3, lower_days=2, eval_windows=200, data=None, mamba_cls=None, pretrain_kwargs=None,
                out_dir=None, results_dir=None, resume=True, seed=0):
  """Phase A if missing, the ORCL CutRule adaptation, phase B on ORCL, and the test evaluation (see the module
  docstring). Phase B checkpoints every epoch; resume=True (default) continues an interrupted run. Returns
  (model, cut, results)."""
  ignore_deprecation_warnings()
  set_seed(seed)
  out_dir = out_dir or output_dir()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  name = method_name(leader) + variant_suffix(time_aware)
  model, cut_state, scale = get_joint_pretrained(leader, device, mamba_cls, **(pretrain_kwargs or {}))
  cut = cutrule_on(device, cut_state)
  data = data or load_orcl()
  r_o, dtu_o, days_o = data.train_slice()
  dt_o = dtu_o if time_aware else None
  print(f"{name}: adapting the CutRule to the ORCL training days ({cutrule_epochs} bilevel epochs, as stage 3)")
  run_bilevel(cut, r_o, days_o, dt_o, cutrule_epochs, seed)

  def lower(rnd):
    n = int(data.day_starts[min(lower_days, len(days_o) - 1)]) if lower_days < len(days_o) else len(r_o)
    return r_o[:n], days_o[days_o < n], (dt_o[:n] if dt_o is not None else None)

  n, b1, b2 = len(data.returns), data.train_end, data.val_end
  data_fn = lambda rnd, seg: JointOrclWindows(data, 0, b1, context, stride, "cutrule", segmenter=seg, seed=seed + rnd,
                                              time_aware=time_aware)
  greedy = lambda lo, hi: OrclWindows(data, lo, hi, context, 64, "cutrule", segmenter=CutRuleSegmenter(cut),
                                      time_aware=time_aware)
  n_train = len(data_fn(0, CutRuleSegmenter(cut)))
  steps_per_epoch = max(1, n_train // batch_windows)
  opt = make_adamw(model, lr=lr)
  phase = JointPhase(model, opt, cut, leader, device, context, batch_windows, micro_windows, cut_lr, bilevel_epochs,
                     bars, inner_iters, seed)
  ckpt = os.path.join(out_dir, f"{name}_finetune_ckpt.pt")
  start_ep, history = 0, []
  if resume and os.path.exists(ckpt):
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); phase.load(ck)
    start_ep, history = ck["epoch"], ck["history"]
    print(f"resumed from {ckpt} at epoch {start_ep}")
  print(f"{name} phase B (ORCL): {epochs} epochs x {steps_per_epoch} steps of {batch_windows} windows x {K} "
        f"patchings, lr {lr}", flush=True)
  t0, best = time.time(), BestOnValidation()
  if start_ep == 0:                                                   # the phase-A model, before any ORCL training
    val0 = evaluate_last_position(model, make_loader(greedy(b1, b2), batch_windows, False), device)
    best.update(val0["loss"], 0, model, {k: v.detach().cpu().clone() for k, v in cut.state_dict().items()})
    print(f"epoch 0 (phase A only): val loss {val0['loss']:.4f}", flush=True)
  else:
    best = BestOnValidation.from_dict(ck["best"]) if ck.get("best") else best
  val_every = max(1, VAL_EVERY_WINDOWS // batch_windows)

  def validate(ep, k, train_loss):
    """Score on the validation days (greedy patches of the current CutRule); keep the best weights and CutRule."""
    point = round(ep + k / steps_per_epoch, 3)
    val = evaluate_last_position(model, make_loader(greedy(b1, b2), batch_windows, False), device)
    history.append({"epoch": point, "train_loss": train_loss, "val_loss": val["loss"]})
    better = best.update(val["loss"], point, model, {k_: v.detach().cpu().clone() for k_, v in cut.state_dict().items()})
    print(f"epoch {point:.2f} (step {ep * steps_per_epoch + k}) train loss {train_loss:.4f} val loss {val['loss']:.4f}"
          f"{'  (best so far)' if better else ''} ({time.time() - t0:.0f}s)", flush=True)

  for ep in range(start_ep, epochs):
    phase.start_round(ep, data_fn, lambda r: (r_o, days_o, dt_o), lower, shuffle=True)
    run = 0.0
    for i in range(steps_per_epoch):
      run = run + phase.step(lr)
      if (i + 1) % val_every == 0 and i + 1 < steps_per_epoch:
        validate(ep, i + 1, float(run) / (i + 1))
    validate(ep, steps_per_epoch, float(run) / steps_per_epoch)       # and at the end of every epoch
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "history": history,
                "best": best.to_dict(), **phase.state()}, ckpt + ".tmp")
    os.replace(ckpt + ".tmp", ckpt)                                   # single rolling checkpoint

  cut.load_state_dict(best.restore(model))                           # weights and CutRule of the best epoch
  test_ds = greedy(b2, n)
  test = evaluate_last_position(model, make_loader(test_ds, batch_windows, False), device)
  print_test_scores(test)
  torch.save({"model": model.state_dict(), "cut": cut.state_dict(), "return_scale": scale, "leader": leader,
              "time_aware": time_aware, "history": history, "best_epoch": best.epoch, "test": test},
             os.path.join(out_dir, f"{name}_finetuned.pt"))
  results_dir = results_dir or os.path.join(out_dir, "results", name)
  write_forecast_report(results_dir, f"ORCL {name}", collect_forecasts(model, test_ds, device, max_windows=eval_windows),
                        HORIZONS, history, extra={"method": name, "time_aware": time_aware, "best_epoch": best.epoch,
                                                  "compression": test["compression"], "test_windows_total": len(test_ds)})
  save_csv(os.path.join(results_dir, "history.csv"), ["epoch", "train_loss", "val_loss"],
           [[h["epoch"], h["train_loss"], h["val_loss"]] for h in history])
  plot_lines(os.path.join(results_dir, "training_loss.png"),
             {"train": ([h["epoch"] for h in history], [h["train_loss"] for h in history]),
              "validation": ([h["epoch"] for h in history], [h["val_loss"] for h in history])},
             f"ORCL {name}: loss per epoch", "epoch", "loss", markers=True)
  print(f"saved forecasts, metrics and figures to {results_dir}")
  return model, cut, {"history": history, "test": test, "results_dir": results_dir, "best_epoch": best.epoch}


def main(default_leader="reinforce"):
  p = argparse.ArgumentParser(description="Methods 3 / 4: the CutRule learns from forecast feedback during both phases.")
  add_quick_flags(p)
  add_spacing_flag(p)
  p.add_argument("--leader", choices=["reinforce", "trilevel"], default=default_leader,
                 help="reinforce = method 3, trilevel = method 4 (follower-follower-leader)")
  p.add_argument("--pretrain-steps", type=int, default=5000, help="phase A steps on Time-300B (as `pretrain`)")
  p.add_argument("--epochs", type=int, default=5, help="phase B epochs on ORCL (as `finetune`)")
  p.add_argument("--cut-lr", type=float, default=1e-3)
  p.add_argument("--restart", action="store_true",
                 help="ignore the rolling checkpoints and start phase A / B over (default: continue an interrupted run)")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  use_quick_outputs(a.quick)
  warm_up()
  pre = {"steps": a.pretrain_steps, "cut_lr": a.cut_lr, "resume": not a.restart, "seed": a.seed}
  kw = {"epochs": a.epochs, "cut_lr": a.cut_lr, "resume": not a.restart, "seed": a.seed}
  if a.quick:
    banner(True)
    q = dict(TRILEVEL if a.leader == "trilevel" else JOINT)
    pre.update(q.pop("pretrain"))
    kw.update(q)
  for time_aware in spacings(a.spacing):
    joint_train(a.leader, time_aware, pretrain_kwargs=dict(pre), **kw)


if __name__ == "__main__":
  main()
