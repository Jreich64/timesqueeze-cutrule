"""Joint training (method 3) and the three-level CutRule (method 4): units and quick end-to-end runs."""
import argparse

import numpy as np
import pytest
import torch
from tests_support import small_orcl_data

from timesqueeze.cutrule.bilevel import train
from timesqueeze.data.orcl import load_orcl
from timesqueeze.patching.segmenter import CutRuleSegmenter
from timesqueeze.training.joint import JointOrclWindows, collate_joint, joint_train, reinforce_loss
from timesqueeze.utils.quick import JOINT, QUICK_DEFAULT, TRILEVEL, add_quick_flags


def trained_cut():
  d = load_orcl()
  torch.manual_seed(0)
  model, _ = train(True, d.returns[:11400], epochs=5, verbose=False, dt=d.dtu[:11400])
  return model, d


def test_quick_mode_is_on_by_default_and_full_turns_it_off(artifacts):
  p = add_quick_flags(argparse.ArgumentParser())
  res = {"default": p.parse_args([]).quick, "--full": p.parse_args(["--full"]).quick,
         "--no-quick": p.parse_args(["--no-quick"]).quick, "--quick": p.parse_args(["--quick"]).quick}
  artifacts.json("flags.json", {**res, "QUICK_DEFAULT": QUICK_DEFAULT, "joint_overrides": JOINT,
                                "trilevel_overrides": TRILEVEL})
  assert QUICK_DEFAULT is True and res == {"default": True, "--full": False, "--no-quick": False, "--quick": True}
  assert JOINT["pretrain"]["steps"] == 1 and JOINT["epochs"] == 1 and JOINT["cutrule_epochs"] == 1


def test_sampled_patching_matches_the_policy(artifacts):
  cut, d = trained_cut()
  seg = CutRuleSegmenter(cut)
  r, dtu = d.returns[:600], d.dtu[:600]
  slow, fast = seg.ewma_features(r, None, dtu)
  starts, feats, cuts = seg.sample_boundaries(r, slow, fast, dtu, np.random.default_rng(0))
  # the policy probabilities recomputed in torch (what the REINFORCE loss uses) equal those used while sampling
  with torch.no_grad():
    p_torch = torch.sigmoid(cut(torch.tensor(feats, device=cut.device)).squeeze(-1) / cut.temp).cpu().numpy()
  p_numpy = 1 / (1 + np.exp(-(np.tanh(feats @ seg.W1 + seg.b1) @ seg.W2 + seg.b2) / seg.temp))
  artifacts.json("sampled_patching.json", {"patches": len(starts), "cuts": int(cuts.sum()),
                                           "max_probability_difference": float(np.abs(p_torch - p_numpy).max())})
  assert np.allclose(p_torch, p_numpy, atol=1e-5)
  assert starts[0] == 0 and set(starts[1:].tolist()) == {int(t) + 1 for t in np.nonzero(cuts)[0] if t + 1 < len(r)}


def test_reinforce_pushes_down_the_patching_with_the_higher_loss(artifacts):
  """Two patchings of one window: the one with the worse forecast loss must become less likely, the better one more."""
  logit = torch.zeros(2, 5, requires_grad=True)                        # one parameter per (variant, tick)
  cuts = torch.tensor([[1., 0, 0, 1, 0], [0., 1, 0, 0, 1]])
  mean_logp = lambda z: (cuts * torch.nn.functional.logsigmoid(z) + (1 - cuts) * torch.nn.functional.logsigmoid(-z)).mean(1)
  losses = torch.tensor([2.0, 1.0])                                    # variant 0 forecasts worse
  reinforce_loss(mean_logp(logit), losses, 2).backward()
  before = mean_logp(logit.detach())
  after = mean_logp(logit.detach() - 0.5 * logit.grad)                 # one descent step on the REINFORCE loss
  artifacts.json("reinforce.json", {"logp_before": before, "logp_after": after, "forecast_losses": losses})
  assert after[0] < before[0] and after[1] > before[1]


def test_collate_joint_shapes_and_grouping(artifacts):
  cut, d = trained_cut()
  data = small_orcl_data()
  ds = JointOrclWindows(data, 0, data.train_end, 128, 64, "cutrule", segmenter=CutRuleSegmenter(cut), k=2, seed=0)
  batch = collate_joint([ds[0], ds[1], ds[2]], 128)
  artifacts.json("batch_shapes.json", {k: list(v.shape) for k, v in batch.items()})
  assert batch["y"].shape[0] == 6 and batch["feats"].shape == (6, 128, 4) and batch["cuts"].shape == (6, 128)
  assert torch.equal(batch["y"][0], batch["y"][1]) and not torch.equal(batch["y"][1], batch["y"][2])   # rows 0,1: one window
  assert batch["tick_mask"].all() and "time" in batch and "dtf" in batch


@pytest.mark.slow
@pytest.mark.parametrize("leader", ["reinforce", "trilevel"])
def test_joint_and_trilevel_quick_end_to_end_both_spacings(leader, artifacts, tmp_path, mamba_cls, monkeypatch):
  """Methods 3 and 4 with the quick-test settings (real Mamba layers when available): phase A on Time-300B runs once and
  is shared by both spacings, phase B on ORCL runs per spacing, losses are finite and the forecast report is saved."""
  monkeypatch.chdir(tmp_path)
  q = dict(TRILEVEL if leader == "trilevel" else JOINT)
  pre = q.pop("pretrain")
  name = "trilevel" if leader == "trilevel" else "joint"
  shared = tmp_path / f"outputs/{name}_pretrained.pt"
  stamps = []
  for time_aware in (True, False):
    sub = artifacts.dir / ("time" if time_aware else "even")
    model, cut, res = joint_train(leader, time_aware, data=small_orcl_data(), mamba_cls=mamba_cls, results_dir=str(sub),
                                  pretrain_kwargs=dict(pre), **q)
    stamps.append(shared.stat().st_mtime_ns)
    assert np.isfinite([h["train_loss"] for h in res["history"]]).all()
    assert np.isfinite(list(res["test"]["mse"].values())).all()
    assert all(torch.isfinite(p).all() for p in cut.parameters())
    for f in ("forecasts.csv", "metrics.json", "mse_by_horizon.png", "training_loss.png", "loss_curves.png"):
      assert (sub / f).exists(), f
  assert stamps[0] == stamps[1]                                       # phase A trained once, reused by the even run
  saved = sorted(f.name for f in (tmp_path / "outputs").glob(f"{name}*finetuned.pt"))
  assert saved == [f"{name}_finetuned.pt", f"{name}_ticks_finetuned.pt"]
  artifacts.json("files.json", {"shared": shared.name, "per_spacing": saved})
