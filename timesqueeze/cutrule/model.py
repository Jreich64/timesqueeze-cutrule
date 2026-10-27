"""The CutRule: a learned rule that decides, after every tick, whether the current bar (patch) ends.

The rule is a tiny MLP, 4 -> 6 (tanh) -> 1, applied to four features of the bar so far:
  1. volatility accumulated since the bar began (scaled by c_sigma),
  2. the bar's duration in ticks (scaled by c_n),
  3. slow and 4. fast EWMA volatility of the series (time-aware when tick timing is given; scaled by c_e).
The bar ends where the output (logit) is >= 0, and in any case after max_len = 8 ticks (the power rule's cap) and at the
end of each trading day / series.

Targets: bars of d_target = 4 ticks on average, about the paper's 4x compression, so the CutRule and the power rule hand
the backbone the same number of tokens and differ only in where they cut. Durations are tick counts in both spacing
variants; tick timing only enters the EWMA volatility (and the forecaster's own timing inputs). The volatility constants
are in units of `vol_unit`, the standard deviation of the returns the rule is first built on (ORCL training returns, or
Time-300B increments scaled to them): bar volatility target sigma_target = c_sigma = vol_unit * sqrt(d_target) (the
volatility of a d_target-tick bar), EWMA scale c_e = vol_unit. vol_unit is saved with the weights (a buffer).

Training (bilevel.py) uses a *soft* version of the rule: each tick cuts with probability sigmoid(logit / temp), which makes
the expected bar duration and the expected bar volatility differentiable. soft_stats() computes them on windows that start
at every hard bar; leader_loss (volatility regularity + cut-rate budget) and follower_loss (bar duration) are the two
objectives of the bilevel problem. segment_data() is the hard segmentation (compiled scan, CPU); segment_data_helper()
builds the soft windows on the GPU.
"""
import numpy as np
import torch
from torch import nn

from ..patching.kernels import cut_scan
from ..patching.volatility import ewma_vol_by_day


class CutRule(nn.Module):
  def __init__(self, returns, d_target=4, max_len=8, beta=5, temp=0.5, vol_unit=None, slow_span=50, fast_span=10,
               hidden=6, device=None, day_starts=None, dt=None):
    super().__init__()
    self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    self.returns = np.ascontiguousarray(returns, dtype=np.float64)
    self.d_target = d_target
    self.max_len = max_len
    self.beta = beta
    self.temp = temp
    s = float(np.std(self.returns)) if vol_unit is None else float(vol_unit)
    self.register_buffer("_vol_unit", torch.tensor(s if s > 0 else 1.0, dtype=torch.float64))   # saved with the weights
    self.slow_span = slow_span
    self.fast_span = fast_span
    self.hidden = hidden
    # day_starts: index of the first return of every trading day. Volatility estimates restart each day and
    # segments are cut at the end of each day (except the last), so no bar spans the overnight gap.
    self.day_starts = np.array([0] if day_starts is None else day_starts, dtype=np.int64)
    # dt: raw time since the previous tick, in tick units; None = evenly spaced. Only the EWMA volatility uses it.
    self.forced_cut = np.zeros(len(returns), dtype=np.bool_)
    self.forced_cut[self.day_starts[1:] - 1] = True
    self._slow_raw = ewma_vol_by_day(self.returns, slow_span, self.day_starts, dt)
    self._fast_raw = ewma_vol_by_day(self.returns, fast_span, self.day_starts, dt)
    self._r2_cum = np.concatenate([[0.0], np.cumsum(self.returns ** 2)])    # for the compiled scan (computed once)
    self._gpu = None
    self.trunk = nn.Linear(4, hidden)
    self.W2 = nn.Parameter(0.1*torch.randn(hidden, 1))
    self.b2 = nn.Parameter(torch.tensor([-2*np.tanh(1.0)], dtype=torch.float32))
    with torch.no_grad():
      nn.init.normal_(self.trunk.weight, 0, 0.1)
      nn.init.normal_(self.trunk.bias, 0, 0.1)
      self.trunk.weight[0, :] = 0
      self.trunk.weight[0, 0] = 1.0
      self.trunk.bias[0] = 0
      self.trunk.weight[1, :] = 0
      self.trunk.weight[1, 1] = 1.0
      self.trunk.bias[1] = 0
      self.W2[:2] = 1.0
    self.to(self.device)

  # volatility constants, in units of vol_unit (see the module docstring); c_n = d_target, so the initial rule (which
  # cuts once both scaled features reach about 1) starts near the target duration and volatility
  @property
  def vol_unit(self):
    return float(self._vol_unit)

  @property
  def sigma_target(self):
    return self.vol_unit * np.sqrt(self.d_target)

  @property
  def c_sigma(self):
    return self.sigma_target

  @property
  def c_n(self):
    return float(self.d_target)

  @property
  def c_e(self):
    return self.vol_unit

  @property
  def EWMA_slow(self):
    return self._slow_raw / self.c_e

  @property
  def EWMA_fast(self):
    return self._fast_raw / self.c_e

  def leader_params(self):
    return list(self.trunk.parameters()) + [self.W2]

  def forward(self, x):
    return (torch.tanh(self.trunk(x)) @ self.W2) + self.b2

  @torch.no_grad()
  def _segment(self):
    W1 = np.ascontiguousarray(self.trunk.weight.detach().cpu().numpy().T, dtype=np.float32)
    starts, ends, _, _, free = cut_scan(self._r2_cum, self.EWMA_slow, self.EWMA_fast, self.forced_cut, W1,
                                        self.trunk.bias.detach().cpu().numpy().astype(np.float32),
                                        self.W2.detach().cpu().numpy()[:, 0].astype(np.float32),
                                        float(self.b2.detach().cpu().numpy()[0]), self.c_sigma, self.c_n, self.max_len)
    return starts, ends, free

  def segment_data(self):
    """Hard segmentation with the current weights: (starts, ends) of every bar. The decision is sequential (each bar
    depends on where the last one ended), so it is a compiled scan over a snapshot of the tiny MLP on the CPU."""
    return self._segment()[:2]

  def has_free_cuts(self):
    """True if the rule cuts by itself somewhere. Cuts forced at the end of each trading day / series or by the max_len
    cap do not count: a rule that never cuts on its own still gets bars from them, which would hide a collapse."""
    return bool(self._segment()[2].any())

  def _tensors(self, device):
    """Series as tensors on `device` (cached; rebuilt if the module moved or vol_unit changed)."""
    key = (device, self.vol_unit)
    if self._gpu is None or self._gpu[0] != key:
      f64 = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
      f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
      self._gpu = (key, f64(self.returns ** 2), f32(self.EWMA_slow), f32(self.EWMA_fast))
    return self._gpu[1:]

  @torch.no_grad()
  def segment_data_helper(self, H_min=100, require_free=True):
    """Hard bars plus, for each, a window of the following ticks with the features of the soft relaxation. A window
    covers at most max_len ticks (the soft bar ends there too) and stops at the next forced day / series end. The
    windows are gathered on the GPU from tensors held there, not built in numpy and copied."""
    starts, ends, free = self._segment()
    if len(ends) == 0 or (require_free and not free.any()):
      return None        # no bars at all, or (require_free) a rule that never cuts by itself = collapsed
    dev = self.trunk.weight.device
    r2_all, slow_all, fast_all = self._tensors(dev)
    T = len(self.returns)
    lens = torch.as_tensor(ends - starts + 1, device=dev)
    H = int(self.max_len) if self.max_len else max(H_min, int(1.5 * lens.max().item()))
    k = torch.arange(H, device=dev)
    idx = torch.as_tensor(starts, device=dev)[:, None] + k
    # a window stops at the next forced cut (end of the day / series): soft stats must not run into the next one
    forced_at = np.nonzero(self.forced_cut)[0]
    nxt = np.searchsorted(forced_at, starts)
    limit = np.where(nxt < len(forced_at), forced_at[np.minimum(nxt, max(len(forced_at) - 1, 0))] if len(forced_at) else 0,
                     T - 1)
    valid = idx <= torch.as_tensor(limit, device=dev)[:, None]
    idc = idx.clamp(max=T - 1)
    r2 = r2_all[idc] * valid
    dt_win = valid.double()                                         # every tick counts 1 toward the bar's duration
    x = torch.stack([torch.cumsum(r2, 1).sqrt() / self.c_sigma, torch.cumsum(dt_win, 1) / self.c_n,
                     slow_all[idc].double(), fast_all[idc].double()], -1).float()
    in_hard = valid & (k < lens[:, None])
    return starts, ends, x, r2.float(), valid.float(), in_hard.float(), dt_win.float()

  def soft_stats(self, x, r2, mask, dt):
    logit = self.forward(x).squeeze(-1)
    p_cut = torch.sigmoid(logit/self.temp)
    log_not_cut = torch.log(1-p_cut.clamp(max=1-1e-6)) * mask
    cum = torch.cumsum(log_not_cut, dim=1)
    p_survive = torch.exp(torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], 1))*mask
    d_soft = (p_survive*dt).sum(1)                                   # expected bar duration (ticks)
    var_soft = (p_survive*r2).sum(1)
    return d_soft, var_soft, p_cut

  def leader_loss(self, d_soft, var_soft, p_cut, in_hard, n_bars, dt):
    log_sigma = 0.5*torch.log(var_soft + 1e-12)
    vol_term = log_sigma.var(unbiased=False) + (log_sigma.mean() - np.log(self.sigma_target))**2
    # bars per tick inside the bars
    total_time = (dt*in_hard).sum().clamp(min=1.0)
    soft_rate = (p_cut*in_hard).sum() / total_time
    hard_rate = n_bars / total_time.item()
    log_rate = np.log(hard_rate) + torch.log(soft_rate) - torch.log(soft_rate).detach()
    budget_term = (log_rate - np.log(1.0/self.d_target))**2
    return vol_term + self.beta*budget_term

  def follower_loss(self, d_soft, b2_prev, lam=0.5):
    log_d = torch.log(d_soft.clamp(min=1e-3))
    f = log_d.var(unbiased=False) + (log_d.mean() - np.log(self.d_target))**2
    return f + 0.5*lam*(((self.b2 - b2_prev)**2).sum())
