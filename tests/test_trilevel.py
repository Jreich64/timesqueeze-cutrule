import pytest
import torch
from torch.autograd import grad

from timesqueeze.cutrule.model import CutRule
from timesqueeze.cutrule.trilevel import (
  TrilevelCutRule,
  hypergradient,
  make_objectives,
  pack,
  solve_phi,
  solve_psi,
  unpack_into,
)
from timesqueeze.data.orcl import load_orcl

D = torch.float64


# ---- a toy three-level problem (smooth, strongly convex lower levels, real couplings between all levels) ----

def f1(psi, phi, theta):                      # level 1: psi*(phi, theta) solves psi + 0.4 psi^3 = rhs
  rhs = torch.sin(phi[0]) + theta[0] * phi[1] + theta[1] ** 2
  return 0.5 * psi ** 2 + 0.1 * psi ** 4 - (psi * rhs).sum()


def f2(phi, psi, theta):                      # level 2: sees psi*, so its minimiser depends on level 1
  c = torch.stack([theta[0], torch.cos(theta[1])])
  return 0.5 * ((phi - c) ** 2).sum() + 0.3 * ((phi[0] * psi) ** 2).sum() + 0.05 * (phi ** 4).sum() \
      + 0.2 * (psi * theta[0] * phi[1]).sum()


def F(theta, phi, psi):                       # leader objective
  return torch.sin(theta[0]) + theta[1] * phi[1] ** 2 + (phi[0] * psi).sum() + (psi ** 2).sum() + 0.5 * (theta ** 2).sum()


def solve_lower(theta):
  phi, psi = solve_phi(f1, f2, torch.zeros(2, dtype=D), torch.zeros(1, dtype=D), theta, iters=30, steps=2,
                       damping=0.0, max_step=1.0, tol=0.0)   # tol=0: solve to machine precision, the checks need it
  return phi, solve_psi(f1, psi, phi, theta, iters=50)


def total_F(theta):
  phi, psi = solve_lower(theta)
  return float(F(theta, phi, psi))


def leader_partials(theta, phi, psi):
  t, p, s = (v.detach().clone().requires_grad_(True) for v in (theta, phi, psi))
  return grad(F(t, p, s), (t, p, s))


def test_lower_levels_converge_to_their_optima(artifacts):
  from timesqueeze.cutrule.trilevel import _reduced_level2
  theta = torch.tensor([0.4, -0.3], dtype=D)
  phi, psi = solve_lower(theta)
  p = psi.clone().requires_grad_(True)
  assert abs(float(grad(f1(p, phi, theta), p)[0])) < 1e-9                       # level 1 stationary
  ph = phi.clone().requires_grad_(True)
  s2, = grad(_reduced_level2(f1, f2, psi, ph, theta, 3, 0.0), ph)
  artifacts.json("stationarity.json", {"level1_gradient": float(grad(f1(p, phi, theta), p)[0]), "level2_gradient": s2})
  assert float(s2.abs().max()) < 1e-7                                           # level 2 (with level-1 response)


@pytest.mark.parametrize("theta0", [[0.4, -0.3], [-0.7, 0.5], [0.1, 0.9]])
def test_trilevel_hypergradient_matches_finite_differences(theta0, artifacts):
  theta = torch.tensor(theta0, dtype=D)
  phi, psi = solve_lower(theta)
  g_theta, g_phi, g_psi = leader_partials(theta, phi, psi)
  analytic = hypergradient(f1, f2, theta, phi, psi, g_theta, g_phi, g_psi, steps=2, damping=0.0)
  eps = 1e-5
  fd = torch.zeros(2, dtype=D)
  for i in range(2):
    e = torch.zeros(2, dtype=D)
    e[i] = eps
    fd[i] = (total_F(theta + e) - total_F(theta - e)) / (2 * eps)
  artifacts.json("hypergradient.json", {"theta": theta0, "analytic": analytic, "finite_difference": fd})
  artifacts.bars("analytic_vs_finite_difference.png", ["d/dtheta_1", "d/dtheta_2"],
                 {"implicit (analytic)": analytic.tolist(), "finite difference": fd.tolist()},
                 "Three-level hypergradient vs numerical derivative", "dF/dtheta")
  assert torch.allclose(analytic, fd, rtol=1e-4, atol=1e-6), (analytic, fd)


def test_ignoring_a_lower_level_gives_a_different_answer(artifacts):
  """Sanity: the answer depends on both followers, i.e. the check above is not trivially passed."""
  theta = torch.tensor([0.4, -0.3], dtype=D)
  phi, psi = solve_lower(theta)
  g_theta, g_phi, g_psi = leader_partials(theta, phi, psi)
  full = hypergradient(f1, f2, theta, phi, psi, g_theta, g_phi, g_psi)
  no_phi = hypergradient(f1, f2, theta, phi, psi, g_theta, torch.zeros_like(g_phi), g_psi)
  no_psi = hypergradient(f1, f2, theta, phi, psi, g_theta, g_phi, torch.zeros_like(g_psi))
  artifacts.json("ablation.json", {"full": full, "without_volatility_level": no_phi, "without_duration_level": no_psi})
  assert not torch.allclose(full, no_phi, atol=1e-4) and not torch.allclose(full, no_psi, atol=1e-4)


# ---- the CutRule objectives as functions agree with the CutRule itself ----

def small_cut():
  d = load_orcl()
  r, dtu = d.returns[:6000], d.dtu[:6000]
  torch.manual_seed(0)
  cut = CutRule(r, dt=dtu, device="cpu")
  return cut


def windows(cut):
  out = cut.segment_data_helper()
  assert out is not None
  starts, ends, x, r2, mask, in_hard, dt = out
  return starts, x, r2, mask, in_hard, dt


def test_functional_objectives_equal_cutrule_methods(artifacts):
  cut = small_cut()
  starts, x, r2, mask, in_hard, dt = windows(cut)
  psi_prev = cut.b2.detach().clone()
  f1_fn, f2_fn = make_objectives(cut, x, r2, mask, dt, psi_prev, lam=0.5)
  theta, phi, psi = pack(cut)
  d_soft, var_soft, p_cut = cut.soft_stats(x, r2, mask, dt)
  assert torch.allclose(f1_fn(psi, phi, theta), cut.follower_loss(d_soft, psi_prev, 0.5), rtol=1e-5)
  cut.beta = 0.0                                                    # leader_loss = volatility term only
  lead = cut.leader_loss(d_soft, var_soft, p_cut, in_hard, len(starts), dt)
  artifacts.json("objectives.json", {"duration_functional": f1_fn(psi, phi, theta),
                                     "duration_cutrule": cut.follower_loss(d_soft, psi_prev, 0.5),
                                     "volatility_functional": f2_fn(phi, psi, theta), "volatility_cutrule": lead})
  assert torch.allclose(f2_fn(phi, psi, theta), lead, rtol=1e-4)


def test_pack_unpack_roundtrip(artifacts):
  cut = small_cut()
  theta, phi, psi = pack(cut)
  other = small_cut()
  unpack_into(other, theta + 1.0, phi + 2.0, psi + 3.0)
  t2, p2, s2 = pack(other)
  assert torch.allclose(t2, theta + 1.0) and torch.allclose(p2, phi + 2.0) and torch.allclose(s2, psi + 3.0)


def test_lower_level_solve_decreases_both_objectives_and_hypergradient_is_finite(artifacts):
  d = load_orcl()
  n = int(d.day_starts[2])
  torch.manual_seed(0)
  cut = CutRule(d.returns[:n], day_starts=d.day_starts[:2], dt=d.dtu[:n], device="cpu")
  tri = TrilevelCutRule(cut, d.returns[:n], d.day_starts[:2], d.dtu[:n], bars=400, inner_iters=3)
  assert tri.refresh_windows()
  theta, phi, psi = pack(cut)
  f1_fn, f2_fn = make_objectives(cut, *tri.w, psi_prev=psi)
  g2 = lambda ph, ps: f2_fn(ph, ps, theta)
  psi_resp = lambda ph: solve_psi(f1_fn, psi, ph, theta)                  # level-1 response to a given phi
  before1, before2 = float(f1_fn(psi, phi, theta)), float(g2(phi, psi_resp(phi)))
  assert tri.solve_lower_levels(refresh=False)                            # same bars as the objectives above
  _, phi2, psi2 = pack(cut)
  after1, after2 = float(f1_fn(psi2, phi2, theta)), float(g2(phi2, solve_psi(f1_fn, psi2, phi2, theta)))
  stale1 = float(f1_fn(psi, phi2, theta))                                  # the old psi under the new phi
  artifacts.json("lower_levels.json", {"duration_before": before1, "duration_old_psi_new_phi": stale1,
                                       "duration_after": after1, "volatility_before": before2, "volatility_after": after2})
  artifacts.bars("objectives.png", ["duration", "volatility"], {"before": [before1, before2], "after": [after1, after2]},
                 "Lower-level objectives before/after the solve", "objective")
  # psi is the follower of phi: given the new phi it is at least as good as the old psi. (f1 itself may rise when phi
  # moves, since phi trades duration for volatility.)
  assert after1 <= stale1 + 1e-6
  assert after2 <= before2 + 1e-4                                          # reduced volatility objective
  g = [torch.randn_like(v) for v in (theta, phi, psi)]
  dtheta = tri.leader_gradient(*g)
  assert dtheta.shape == theta.shape and torch.isfinite(dtheta).all() and dtheta.abs().sum() > 0
