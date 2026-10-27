"""Method 4: the CutRule as a three-level (follower - follower - leader) problem.

    level 1 (innermost follower)  duration     f1(psi; phi, theta)   psi   = b2 (cut threshold bias)
    level 2 (follower)            volatility   f2(phi; theta)        phi   = W2 (readout weights)
    level 3 (leader)              forecasting  F(theta)              theta = hidden layer (W1, b1)

Each level optimises its own variables knowing that the levels below respond optimally:
    psi*(phi, theta) = argmin_psi f1(psi; phi, theta)
    phi*(theta)      = argmin_phi f2(phi, psi*(phi, theta); theta)
    leader minimises F(theta, phi*(theta), psi*(phi*(theta), theta)).
The duration level fixes the average bar length, the volatility level shapes bars given the features, and the leader
(forecast quality, estimated by REINFORCE because patch boundaries are hard decisions) only picks *where* to cut,
under both lower-level constraints. (The two-level CutRule objective in cutrule.py/bilevel.py is the special case with
no forecasting term and volatility in the leader.)

Hypergradient. With g = dF/d(theta, phi, psi) given (REINFORCE supplies it), the total derivative is
    dF/dtheta = g_theta + (dphi*/dtheta)^T g_phi + (dpsi*/dtheta)^T g_psi
and the solution maps are differentiated implicitly, not by unrolling the solvers:
  * psi*(phi, theta) is represented by `steps` Newton steps from its converged value, with the graph kept. Newton
    converges quadratically, so starting at the optimum k steps make the first k derivatives exact (verified against
    finite differences in tests/test_trilevel.py).
  * phi*(theta) is one Newton step on the reduced level-2 objective g2(phi, theta) = f2(phi, psi*(phi, theta), theta),
    whose gradient and Hessian already contain the level-1 response; this gives dphi*/dtheta = -H_phi^-1 d s2/dtheta.
Everything is differentiated with autograd on small parameter vectors (psi 1, phi 6, theta 30), so no Hessian
approximations are made beyond the damping term (0 in the tests).
"""
import numpy as np
import torch
from torch.autograd import grad

# ---------------------------------------------------------------- generic machinery

def _newton_psi(f1, psi0, phi, theta, steps, damping):
  """psi after `steps` Newton steps from psi0 on f1(., phi, theta), with a graph to phi and theta."""
  p = psi0.detach().clone().requires_grad_(True)
  for _ in range(steps):
    g, = grad(f1(p, phi, theta), p, create_graph=True)
    h, = grad(g.sum(), p, create_graph=True)
    p = p - g / (torch.clamp(h, min=1e-8) + damping)
  return p


def solve_psi(f1, psi, phi, theta, iters=8, damping=0.0, max_step=1.0):
  """Level 1: minimise f1 over the scalar psi by damped Newton (no graph)."""
  phi, theta = phi.detach(), theta.detach()
  p = psi.detach().clone()
  for _ in range(iters):
    p = p.requires_grad_(True)
    g, = grad(f1(p, phi, theta), p, create_graph=True)
    h, = grad(g.sum(), p)
    h = torch.clamp(h, min=1e-6) + damping
    p = (p - torch.clamp(g / h, -max_step, max_step)).detach()
  return p


def _reduced_level2(f1, f2, psi0, phi_, theta, steps, damping):
  return f2(phi_, _newton_psi(f1, psi0, phi_, theta, steps, damping), theta)


def solve_phi(f1, f2, phi, psi, theta, iters=8, steps=2, damping=1e-3, max_step=0.5, tol=1e-3):
  """Level 2: minimise g2(phi) = f2(phi, psi*(phi, theta), theta) by Newton with the exact level-1 response.
  Returns (phi, psi*) at the solution."""
  theta = theta.detach()
  phi = phi.detach().clone()
  psi = solve_psi(f1, psi, phi, theta)
  for _ in range(iters):
    ph = phi.clone().requires_grad_(True)
    val = _reduced_level2(f1, f2, psi, ph, theta, steps, 0.0)
    s, = grad(val, ph, create_graph=True)
    if s.norm().item() < tol:                                    # level 2 is stationary: done
      break
    H = torch.stack([grad(s[i], ph, retain_graph=True)[0] for i in range(len(phi))]).detach()
    H = 0.5 * (H + H.T)
    lam = max(0.0, -float(torch.linalg.eigvalsh(H).min())) + damping            # make the step a descent step
    step = torch.linalg.solve(H + lam * torch.eye(len(phi), dtype=H.dtype, device=H.device), s.detach())
    n = float(step.norm())
    if n > max_step:
      step = step * (max_step / n)
    phi = (phi - step).detach()
    psi = solve_psi(f1, psi, phi, theta)
  return phi, psi


def hypergradient(f1, f2, theta, phi, psi, g_theta, g_phi, g_psi, steps=2, damping=0.0):
  """dF/dtheta for the three-level problem at lower-level solutions (phi, psi); g_* are the partial derivatives of
  the leader objective F at that point (see the module docstring). theta [n], phi [m], psi [1]."""
  theta = theta.detach().clone().requires_grad_(True)
  phi_ = phi.detach().clone().requires_grad_(True)
  m = len(phi)
  # level 1 is differentiated with no damping (so the Newton steps are exact at its optimum, as solve_phi assumes); the
  # level-2 Hessian gets the same positive-definite shift as in solve_phi (indefinite H gives a meaningless solve)
  s2, = grad(_reduced_level2(f1, f2, psi, phi_, theta, steps, 0.0), phi_, create_graph=True)
  H = torch.stack([grad(s2[i], phi_, retain_graph=True)[0] for i in range(m)]).detach()
  H = 0.5 * (H + H.T)
  shift = max(0.0, -float(torch.linalg.eigvalsh(H).min())) + damping
  H = H + shift * torch.eye(m, dtype=H.dtype, device=H.device)
  phi_graph = phi_.detach() - torch.linalg.solve(H, s2)                           # dphi*/dtheta through s2
  psi_final = _newton_psi(f1, psi, phi_graph, theta, steps, 0.0)                  # dpsi*/dtheta incl. via phi*
  surrogate = (g_theta.detach() * theta).sum() + (g_phi.detach() * phi_graph).sum() + (g_psi.detach() * psi_final).sum()
  dtheta, = grad(surrogate, theta)
  return dtheta


# ---------------------------------------------------------------- the CutRule objectives as functions

def pack(cut):
  """(theta, phi, psi) vectors from a CutRule: theta = [W1 (hidden x 4), b1], phi = W2, psi = b2."""
  theta = torch.cat([cut.trunk.weight.detach().flatten(), cut.trunk.bias.detach()])
  return theta, cut.W2.detach().flatten().clone(), cut.b2.detach().clone()


def unpack_into(cut, theta=None, phi=None, psi=None):
  h = cut.hidden
  with torch.no_grad():
    if theta is not None:
      cut.trunk.weight.copy_(theta[:4 * h].view(h, 4))
      cut.trunk.bias.copy_(theta[4 * h:])
    if phi is not None:
      cut.W2.copy_(phi.view(h, 1))
    if psi is not None:
      cut.b2.copy_(psi)


def logits_fn(x, theta, phi, psi, hidden):
  W1, b1 = theta[:4 * hidden].view(hidden, 4), theta[4 * hidden:]
  return torch.tanh(x @ W1.T + b1) @ phi.view(hidden, 1) + psi            # same as CutRule.forward


def make_objectives(cut, x, r2, mask, dt, psi_prev, lam=0.5):
  """f1 (duration) and f2 (volatility) of the CutRule as functions of explicit (psi, phi, theta) tensors, on the
  soft-relaxed bars given by the windows (x, r2, mask, dt). Same formulas as CutRule.soft_stats / follower_loss /
  leader_loss's volatility term."""
  h, temp = cut.hidden, cut.temp

  def soft(theta, phi, psi):
    logit = logits_fn(x, theta, phi, psi, h).squeeze(-1)
    p_cut = torch.sigmoid(logit / temp)
    log_not_cut = torch.log(1 - p_cut.clamp(max=1 - 1e-6)) * mask
    cum = torch.cumsum(log_not_cut, dim=1)
    p_survive = torch.exp(torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], 1)) * mask
    return (p_survive * dt).sum(1), (p_survive * r2).sum(1)

  def f1(psi, phi, theta):                                                   # duration (follower)
    d_soft, _ = soft(theta, phi, psi)
    log_d = torch.log(d_soft.clamp(min=1e-3))
    return log_d.var(unbiased=False) + (log_d.mean() - np.log(cut.d_target)) ** 2 + 0.5 * lam * ((psi - psi_prev) ** 2).sum()

  def f2(phi, psi, theta):                                                   # volatility (follower)
    _, var_soft = soft(theta, phi, psi)
    log_sigma = 0.5 * torch.log(var_soft + 1e-12)
    return log_sigma.var(unbiased=False) + (log_sigma.mean() - np.log(cut.sigma_target)) ** 2

  return f1, f2


class TrilevelCutRule:
  """Lower-level solves and the leader hypergradient for one CutRule, on a fixed corpus of bars."""

  def __init__(self, cut, returns, day_starts, dt, bars=2000, inner_iters=8, steps=2, damping=1e-3, seed=0):
    from .model import CutRule
    self.cut, self.bars, self.inner_iters, self.steps, self.damping = cut, bars, inner_iters, steps, damping
    self.corpus = CutRule(returns, day_starts=day_starts, dt=dt, device=cut.device)
    self.rng = np.random.default_rng(seed)
    self.psi_prev = None                  # proximal centre of f1: must be the same in the solve and in the hypergradient

  def refresh_windows(self):
    """Segment the corpus with the current CutRule and keep a random subset of bars (x, r2, mask, dt windows)."""
    self.corpus.load_state_dict(self.cut.state_dict())
    out = self.corpus.segment_data_helper()
    if out is None:
      return False
    _, _, x, r2, mask, _, dt = out
    n = x.shape[0]
    idx = torch.as_tensor(self.rng.choice(n, size=min(self.bars, n), replace=False), device=x.device)
    self.w = (x[idx], r2[idx], mask[idx], dt[idx])
    return True

  def solve_lower_levels(self, refresh=True):
    """Move psi (duration) and phi (volatility) to their optima given theta; writes them into the CutRule.
    refresh=False reuses the bars of the previous refresh_windows() call."""
    if refresh and not self.refresh_windows():
      return False
    theta, phi, psi = pack(self.cut)
    self.psi_prev = psi.clone()
    f1, f2 = make_objectives(self.cut, *self.w, psi_prev=self.psi_prev.to(self.w[0].device))
    phi, psi = solve_phi(f1, f2, phi.to(self.w[0].device), psi.to(self.w[0].device), theta.to(self.w[0].device),
                         iters=self.inner_iters, steps=self.steps, damping=self.damping)
    unpack_into(self.cut, phi=phi.to(self.cut.W2.device), psi=psi.to(self.cut.b2.device))
    return True

  def leader_gradient(self, g_theta, g_phi, g_psi):
    """dF/dtheta from the REINFORCE partials, at the current lower-level solution. Windows are those of the last
    solve_lower_levels call."""
    theta, phi, psi = pack(self.cut)
    dev = self.w[0].device
    f1, f2 = make_objectives(self.cut, *self.w, psi_prev=(self.psi_prev if self.psi_prev is not None else psi).to(dev))
    return hypergradient(f1, f2, theta.to(dev), phi.to(dev), psi.to(dev), g_theta.to(dev), g_phi.to(dev),
                         g_psi.to(dev), steps=self.steps, damping=self.damping)


def cut_grads(cut):
  """(g_theta, g_phi, g_psi) vectors from the .grad fields of a CutRule (as filled by a REINFORCE backward)."""
  z = lambda p: torch.zeros_like(p) if p.grad is None else p.grad.detach()
  return (torch.cat([z(cut.trunk.weight).flatten(), z(cut.trunk.bias)]), z(cut.W2).flatten().clone(), z(cut.b2).clone())
