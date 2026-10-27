"""The CutRule works at the power rule's scale: bars of ~4 ticks on average, never longer than 8, durations counted in
ticks in both spacing variants, volatility constants in units of the data's own return scale (saved with the weights)."""
import numpy as np
import torch

from timesqueeze.cutrule.model import CutRule
from timesqueeze.data.orcl import load_orcl
from timesqueeze.patching.segmenter import CutRuleSegmenter
from timesqueeze.training.joint import cut_log_prob


def orcl_cut(n_days=3, time_aware=True):
  d = load_orcl()
  n = int(d.day_starts[n_days])
  torch.manual_seed(0)
  return CutRule(d.returns[:n], day_starts=d.day_starts[:n_days], dt=d.dtu[:n] if time_aware else None, device="cpu"), d, n


def test_targets_and_volatility_units_follow_the_data_and_are_saved(artifacts):
  cut, d, n = orcl_cut()
  s = float(np.std(d.returns[:n]))
  assert cut.d_target == 4 and cut.max_len == 8 and cut.c_n == 4.0
  assert np.isclose(cut.vol_unit, s) and np.isclose(cut.sigma_target, 2 * s) and cut.c_e == cut.vol_unit
  blank = CutRule(np.ones(2), device="cpu")                           # how saved CutRules are rebuilt
  blank.load_state_dict(cut.state_dict())
  assert blank.vol_unit == cut.vol_unit and CutRuleSegmenter(blank).c_sigma == cut.c_sigma
  artifacts.json("units.json", {"vol_unit": cut.vol_unit, "sigma_target": cut.sigma_target, "c_n": cut.c_n})


def test_initial_rule_cuts_near_the_target_and_never_past_the_cap(artifacts):
  out = {}
  for time_aware in (True, False):
    cut, d, n = orcl_cut(time_aware=time_aware)
    starts, ends = cut.segment_data()
    lens = ends - starts + 1
    d_soft = cut.soft_stats(*[cut.segment_data_helper()[i] for i in (2, 3, 4, 6)])[0]
    out[str(time_aware)] = {"mean_bar": float(lens.mean()), "share_at_cap": float((lens == 8).mean()),
                            "max_soft_duration": float(d_soft.max())}
    assert lens.max() <= 8 and float(d_soft.max()) <= 8.0 + 1e-4
    assert 1.5 < lens.mean() < 8                                       # starts in the right range, not at 20-40 ticks
  artifacts.json("initial_bars.json", out)


def test_tick_durations_ignore_the_timing():
  """Durations are tick counts: the same returns with or without tick timing give the same duration feature."""
  timed, d, n = orcl_cut(time_aware=True)
  _, _, x, _, mask, _, dt = timed.segment_data_helper(require_free=False)
  assert torch.equal(dt, mask)                                         # every tick counts 1, whatever its gap
  ticks = x[..., 1] * timed.c_n
  assert torch.allclose(ticks, torch.cumsum(mask, 1), atol=1e-5)       # duration feature = ticks so far


def test_reinforce_ignores_ticks_where_the_cap_ended_the_bar():
  cut, d, n = orcl_cut()
  torch.manual_seed(1)
  feats = torch.randn(2, 16, 4)
  cuts = torch.zeros(2, 16)
  cuts[:, 7] = -1.0                                                    # capped: no decision taken there
  cuts[0, 3] = 1.0
  batch = {"feats": feats, "cuts": cuts, "tick_mask": torch.ones(2, 16, dtype=torch.bool)}
  a = cut_log_prob(cut, batch)
  feats2 = feats.clone()
  feats2[:, 7] += 10.0                                                 # changing a capped tick's features changes nothing
  b = cut_log_prob(cut, {**batch, "feats": feats2})
  assert torch.allclose(a, b) and torch.isfinite(a).all()
