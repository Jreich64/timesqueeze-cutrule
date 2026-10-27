"""Method 5 (power rule, tau = follower, forecasting = leader): the follower search, the implicit hypergradient against
finite differences on a smooth problem with a known solution, and a quick end-to-end run on ORCL."""
import math

import numpy as np
import pytest
import torch
from tests_support import small_orcl_data

from timesqueeze.training.power_bilevel import TauFollower, golden_section, implicit_tau_gradient

LAM, C = 0.7, 0.4


def toy_L(theta, tau):
  """A smooth stand-in for the forecast loss as a function of the forecaster (theta) and the threshold (tau)."""
  a, b = theta[0], theta[1]
  return (a * tau - 1) ** 2 + b ** 2 * tau + 0.5 * tau ** 2


def toy_f(theta, tau):
  return toy_L(theta, tau) + LAM * (tau - C) ** 2                     # follower: loss + a 'compression' penalty


def toy_tau_star(theta):
  a, b = float(theta[0]), float(theta[1])                              # df/dtau = 0, solved in closed form
  return (2 * a - b ** 2 + 2 * LAM * C) / (2 * a ** 2 + 1 + 2 * LAM)


def test_golden_section_finds_the_minimum():
  x, fx = golden_section(lambda t: (t - 0.37) ** 2 + 1.0, 0.0, 1.0, 30)
  assert abs(x - 0.37) < 1e-4 and abs(fx - 1.0) < 1e-6


@pytest.mark.parametrize("theta0", [[0.8, 0.3], [1.3, -0.5], [0.5, 0.9]])
def test_implicit_tau_gradient_matches_finite_differences(theta0, artifacts):
  """dF/dtheta of F(theta) = L(theta, tau*(theta)) from implicit_tau_gradient vs central differences of F."""
  theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
  tau = toy_tau_star(theta.detach())
  grad_at = lambda t: [torch.autograd.grad(toy_L(theta, t), theta)[0]]
  loss_at = lambda t: float(toy_L(theta.detach(), t))
  obj_at = lambda t: float(toy_f(theta.detach(), t))
  total, info = implicit_tau_gradient(grad_at, loss_at, obj_at, tau, eps=1e-4)
  F = lambda th: float(toy_L(th, toy_tau_star(th)))
  h = 1e-6
  fd = [(F(theta.detach() + h * e) - F(theta.detach() - h * e)) / (2 * h) for e in torch.eye(2, dtype=torch.float64)]
  direct = grad_at(tau)[0]
  artifacts.json("hypergradient.json", {"theta": theta0, "tau_star": tau, "implicit": total[0].tolist(), "fd": fd,
                                        "direct_only": direct.tolist(), "rel": info["rel"]})
  assert np.allclose(total[0].numpy(), fd, rtol=1e-4, atol=1e-6)
  assert not np.allclose(direct.numpy(), fd, rtol=1e-3)                  # the indirect term matters here


def test_follower_moves_tau_toward_its_objective_minimum():
  f = TauFollower(tau=0.3, target_compression=4.0, lam=0.0, mu=0.0, radius=0.25, evals=25)
  tau = f.solve(lambda t: (t - 0.42) ** 2)
  assert abs(tau - 0.42) < 1e-3
  f = TauFollower(tau=0.3, radius=0.1, evals=25)                         # bounded to a window around the last tau
  assert abs(f.solve(lambda t: (t - 2.0) ** 2) - 0.4) < 1e-3
  assert math.isclose(f.objective(1.0, 4.0, 0.5, 0.5), 1.0)              # on target, no move: just the loss


def test_unused_weights_get_no_gradient_and_partial_ones_count_as_zero():
  """None at every tau -> None (AdamW skips the weight, as with .backward()); None at only some tau -> 0 there."""
  w = torch.ones(2)
  grads = {0.4: [w, None, None], 0.5: [2 * w, 3 * w, None], 0.3: [w, None, None]}       # tau-eps, tau, tau+eps
  total, info = implicit_tau_gradient(lambda t: grads[round(t, 6)], lambda t: t, lambda t: t * t, 0.4, 0.1, tik=1e-9)
  assert total[2] is None and info["unused_tensors"] == 1
  # direct + scale * mixed, with dL/dtau = 1, curvature = 2, scale = -0.5
  assert torch.allclose(total[0], w + -0.5 * (2 * w - w) / 0.2)
  assert torch.allclose(total[1], torch.zeros(2) + -0.5 * (3 * w - 0) / 0.2)


def test_tau_steps_move_patches_and_loss_on_a_real_orcl_batch(artifacts, mamba_cls, device):
  """On a real 16 x 512 ORCL batch the relative step (25% of tau) moves many boundaries and changes the loss at the tau
  the follower works at (about 1 and above, where compression approaches 4x). Near the paper's 0.3 ORCL patches are almost
  flat in tau (boundaries come from jumps far above any such threshold); that is recorded, not asserted."""
  from timesqueeze.data.orcl import load_orcl
  from timesqueeze.data.orcl_windows import OrclWindows
  from timesqueeze.model.backbone import build_model
  from timesqueeze.training.power_bilevel import TauBatch, batch_loss
  data = load_orcl()
  ds = OrclWindows(data, 0, data.train_end, 512, 16, "power", time_aware=True)
  tb = TauBatch(ds, np.random.default_rng(0).choice(len(ds), 16, replace=False), device)
  torch.manual_seed(0)
  model = build_model(mamba_cls)
  for head in model.heads.values():           # untrained heads start at zero, which no patching could change
    torch.nn.init.normal_(head.weight, 0, 0.02)
  model = model.to(device).eval()
  rows = {}
  for tau in (0.3, 1.0):
    for t in (tau * 0.75, tau, tau * 1.25):
      b, c = tb.batch(t)
      with torch.no_grad():
        rows[round(t, 4)] = {"patches": int(b["patch_mask"].sum()), "compression": c, "loss": float(batch_loss(model, b))}
  artifacts.json("tau_sensitivity.json", rows)
  lo, hi = rows[0.75], rows[1.25]
  assert lo["patches"] - hi["patches"] >= 50                   # boundaries actually move across the step
  assert lo["loss"] != hi["loss"]                              # and the forecast loss sees it


@pytest.mark.slow
@pytest.mark.parametrize("time_aware", [True, False], ids=["time", "even"])
def test_power_bilevel_quick_end_to_end(time_aware, artifacts, tmp_path, monkeypatch, mamba_cls):
  from timesqueeze.training.power_bilevel import power_bilevel
  from timesqueeze.utils.quick import BACKBONE_PRETRAIN, POWER_BILEVEL
  monkeypatch.chdir(tmp_path)
  model, res = power_bilevel(time_aware, data=small_orcl_data(), mamba_cls=mamba_cls, results_dir=str(artifacts.dir),
                             pretrain_kwargs=dict(BACKBONE_PRETRAIN), **POWER_BILEVEL)
  sfx = "" if time_aware else "_ticks"
  assert (tmp_path / f"outputs/finetuned_power_bilevel{sfx}.pt").exists()
  assert 0.02 <= res["tau"] <= 3.0 and len(res["history"]) == POWER_BILEVEL["steps"]
  for h in res["history"]:
    assert all(np.isfinite([h["tau"], h["loss"], h["curvature"], h["rel"], h["grad_norm"]]))
  assert np.isfinite(list(res["test"]["mse"].values())).all()
  for f in ("tau_history.csv", "tau_and_compression.png", "metrics.json", "loss_curves.png"):
    assert (artifacts.dir / f).exists(), f
