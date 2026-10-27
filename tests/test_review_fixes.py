"""One test per confirmed finding of the independent code review (see README, "Review")."""
import json
import zlib

import numpy as np
import pytest
import torch
from tests_support import small_orcl_data
from torch.autograd import grad

import timesqueeze.model.backbone as bb
from timesqueeze.cutrule.model import CutRule
from timesqueeze.cutrule.stages import get_orcl_cutrule, train_orcl_cutrule
from timesqueeze.cutrule.trilevel import TrilevelCutRule, make_objectives, pack
from timesqueeze.data.orcl import load_orcl
from timesqueeze.data.time300b_windows import VAL_BIN_MOD, Time300BSegments, local_subsets
from timesqueeze.model.backbone import get_backbone
from timesqueeze.training.joint import advantage_magnitude, reinforce_loss
from timesqueeze.utils.paths import DATA_DIR


def three_day_cut():
  d = load_orcl()
  n = int(d.day_starts[3])
  torch.manual_seed(0)
  return CutRule(d.returns[:n], day_starts=d.day_starts[:3], dt=d.dtu[:n], device="cpu"), d, n


def test_collapse_is_detected_on_multi_day_data(artifacts):
  """A rule that never cuts by itself still gets bars from the forced day-end cuts and the 8-tick length cap; that must
  count as a collapse (it used to look like healthy bars, so the recovery logic could never fire)."""
  cut, d, n = three_day_cut()
  healthy = (len(cut.segment_data()[0]), cut.has_free_cuts(), cut.segment_data_helper() is not None)
  with torch.no_grad():
    cut.b2.fill_(-1e6)
  starts, ends = cut.segment_data()
  collapsed = (len(starts), cut.has_free_cuts(), cut.segment_data_helper() is not None)
  artifacts.json("collapse_detection.json", {"initial": healthy, "never_cuts": collapsed})
  assert collapsed[1:] == (False, False)                           # bars from the day ends and the cap only: collapsed
  assert (ends - starts + 1).max() == cut.max_len and len(starts) >= n // cut.max_len
  with torch.no_grad():
    cut.b2.fill_(5.0)                                              # cuts everywhere: certainly cuts by itself
  assert cut.has_free_cuts() and cut.segment_data_helper() is not None


def test_soft_windows_stop_at_the_end_of_the_day(artifacts):
  cut, d, n = three_day_cut()
  with torch.no_grad():
    cut.b2.fill_(-1.0)                                             # cuts, with bars short of the 8-tick windows,
  out = cut.segment_data_helper()                                  # so bars near a day's end have windows that would reach the next day
  assert out is not None
  starts, mask = out[0], out[4]
  day_of = np.searchsorted(d.day_starts[:3], np.arange(n), side="right")
  last_valid = (mask.cpu().numpy().sum(1) - 1).astype(int) + starts      # index of each window's last valid tick
  crosses = int((day_of[np.minimum(last_valid, n - 1)] != day_of[starts]).sum())
  artifacts.json("windows_and_days.json", {"bars": len(starts), "windows_crossing_a_day": crosses,
                                           "window_length": int(mask.shape[1])})
  assert crosses == 0


def test_hypergradient_uses_the_same_proximal_centre_as_the_solve(artifacts):
  """f1 contains 0.5*lam*(psi - psi_prev)^2. The hypergradient must differentiate the f1 that was solved (same
  psi_prev), at which psi is stationary; using the solved psi as the centre made the point non-stationary."""
  cut, d, n = three_day_cut()
  tri = TrilevelCutRule(cut, d.returns[:n], d.day_starts[:3], d.dtu[:n], bars=400)
  tri.refresh_windows()
  psi0 = pack(cut)[2]
  assert tri.solve_lower_levels(refresh=False)
  theta, phi, psi = pack(cut)

  def df1(center):
    f1, _ = make_objectives(cut, *tri.w, psi_prev=center)
    p = psi.clone().requires_grad_(True)
    return abs(float(grad(f1(p, phi, theta), p)[0]))

  used, wrong = df1(tri.psi_prev), df1(psi)
  artifacts.json("proximal_centre.json", {"psi_before": float(psi0), "psi_after": float(psi),
                                          "df1_with_solve_centre": used, "df1_with_solved_psi_as_centre": wrong})
  assert torch.equal(tri.psi_prev, psi0) and used < 1e-3 and wrong > 10 * used
  g = [torch.randn_like(v) for v in (theta, phi, psi)]
  assert torch.isfinite(tri.leader_gradient(*g)).all()


def test_validation_data_never_shares_a_source_file_with_training(artifacts):
  seg = Time300BSegments()
  names = {}
  for sub in local_subsets():                                  # every folder pretraining reads
    for e in json.load(open(DATA_DIR / sub / "index.json")):
      names[str(DATA_DIR / sub / e["file"])] = (sub, e["source"])
  by_source = {}
  for path, val in zip(seg.window_paths[seg.piece_window], seg.is_val):
    by_source.setdefault(names[path], set()).add(bool(val))
  mixed = [s for s, flags in by_source.items() if len(flags) > 1]
  artifacts.json("validation_split.json", {"sources": len(by_source), "sources_in_both_splits": len(mixed),
                                           "validation_pieces": int(seg.is_val.sum()), "pieces": len(seg.is_val)})
  assert not mixed and 0 < seg.is_val.sum() < len(seg.is_val)
  assert all((zlib.crc32(src.encode()) % VAL_BIN_MOD == 0) == (True in flags) for (_, src), flags in by_source.items())


def test_a_backbone_of_another_patch_rule_is_refused(artifacts, tmp_path, monkeypatch):
  """Backbones are per patch rule (shared by both spacings): another rule's backbone is refused."""
  f = tmp_path / "b.pt"
  f.write_bytes(b"x")
  monkeypatch.setattr(bb, "load_backbone", lambda *a, **k: (None, {"patcher": "power"}))
  with pytest.raises(RuntimeError, match="patcher"):
    get_backbone(str(f), patcher="cutrule")
  assert get_backbone(str(f), patcher="power")[1]["patcher"] == "power"
  artifacts.json("backbone_check.json", {"refused": ["patcher mismatch"], "accepted": "match"})


def test_an_interrupted_cutrule_checkpoint_is_continued_not_used(artifacts, tmp_path):
  data = small_orcl_data()
  path = str(tmp_path / "ck.pt")
  train_orcl_cutrule(data, True, path, epochs=1, from_scratch=True)                 # an "interrupted" run: 1 epoch
  assert torch.load(path, map_location="cpu")["epoch"] == 1
  get_orcl_cutrule(data, True, path, epochs=3, from_scratch=True)                  # asks for 3: must finish the job
  epoch_now = torch.load(path, map_location="cpu")["epoch"]
  artifacts.json("partial_checkpoint.json", {"epochs_in_file_after": epoch_now})
  assert epoch_now == 3


def test_reinforce_advantage_scale(artifacts):
  logp = torch.tensor([-1.0, -2.0, -1.5, -1.5], requires_grad=True)
  losses = torch.tensor([2.0, 1.0, 5.0, 5.0])
  mag = advantage_magnitude(losses, 2)
  assert torch.isclose(mag, torch.tensor(0.25))                          # mean(|+-0.5|, |+-0.5|, 0, 0)
  big = reinforce_loss(logp, losses, 2, scale=mag * 10)
  small = reinforce_loss(logp, losses, 2, scale=mag)
  artifacts.json("advantage_scale.json", {"magnitude": float(mag), "loss_scale_x10": float(big),
                                          "loss_scale_x1": float(small)})
  assert torch.isclose(small, big * 10)                                  # a given scale is used as is, not renormalised


def test_a_finished_real_run_must_cut_on_its_own(artifacts):
  """The check moved from 'every epoch' to 'the finished rule': a never-cutting rule is rejected after a real-length
  run, but a short smoke run (quick mode) is allowed to end before the rule has learned to cut."""
  from timesqueeze.cutrule.stages import MIN_EPOCHS_TO_CHECK, require_free_cuts
  cut, d, n = three_day_cut()
  with torch.no_grad():
    cut.b2.fill_(-1e6)
  require_free_cuts(cut, MIN_EPOCHS_TO_CHECK - 1, "quick run")                       # short run: accepted
  with pytest.raises(RuntimeError, match="without cutting on its own"):
    require_free_cuts(cut, MIN_EPOCHS_TO_CHECK, "real run")                         # real run: rejected
  with torch.no_grad():
    cut.b2.fill_(5.0)
  require_free_cuts(cut, MIN_EPOCHS_TO_CHECK, "real run")                           # a rule that cuts is fine
  artifacts.json("require_free_cuts.json", {"min_epochs_checked": MIN_EPOCHS_TO_CHECK})
