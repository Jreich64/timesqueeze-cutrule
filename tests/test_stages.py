import numpy as np
import pytest
import torch

from timesqueeze.cutrule.bilevel import train
from timesqueeze.cutrule.stages import (
  CutRulePatcher,
  build_corpus,
  get_pretrained_cutrule,
  increments_as_returns,
  state_fingerprint,
)
from timesqueeze.data.orcl import OrclData, load_orcl
from timesqueeze.data.time300b_windows import Time300BSegments
from timesqueeze.utils.quick import BACKBONE_PRETRAIN, CUTRULE_PRETRAIN

TINY_CUTRULE = {"n_segments": 25, "segment_len": 512, "epochs": 3}


def test_increments_as_returns(artifacts):
  x = np.cumsum(np.random.default_rng(0).standard_normal(300))
  r = increments_as_returns(x, 1.5e-4)
  artifacts.lines("series_and_returns.png", {"series (z-scored)": (np.arange(300), (x - x.mean()) / x.std()),
                                               "increments, scaled to ORCL size x 1e3": (np.arange(300), r * 1e3 / 1.5e-4)},
                  "A Time-300B series turned into CutRule 'returns'", "tick", "value")
  artifacts.json("scaling.json", {"first": r[0], "std_of_rest": float(r[1:].std()), "target_std": 1.5e-4})
  assert r[0] == 0.0 and len(r) == 300 and np.isclose(r[1:].std(), 1.5e-4)
  assert not increments_as_returns(np.full(50, 3.0), 1.5e-4).any()        # flat series carry no increments


def test_corpus_marks_each_series_as_its_own_day(artifacts):
  seg = Time300BSegments()
  returns, starts = build_corpus(seg, n_segments=15, segment_len=512, return_scale=1.5e-4, seed=0)
  ends = list(starts[1:]) + [len(returns)]
  artifacts.lines("first_series.png", {f"series {i}": (np.arange(e - s), returns[s:e]) for i, (s, e) in
                                       enumerate(zip(starts[:3], ends[:3]))},
                  "Three corpus series (CutRule returns)", "tick", "return")
  artifacts.json("corpus.json", {"series": len(starts), "points": len(returns), "lengths": np.diff(starts)})
  assert len(starts) == 15 and starts[0] == 0 and np.all(np.diff(starts) >= 255)
  assert np.isfinite(returns).all()


def test_init_state_is_the_starting_point_and_patcher_is_valid(artifacts):
  r = load_orcl().returns[:11400]                                               # real day-1 returns (noise never cuts)
  torch.manual_seed(0)
  src, _ = train(True, r, epochs=5, verbose=False)
  state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
  start, _ = train(True, r, epochs=0, verbose=False, init_state=state)          # zero epochs: weights are the init
  assert all(torch.equal(start.state_dict()[k].cpu(), v) for k, v in state.items())
  patcher = CutRulePatcher(src.to("cpu"), 1.5e-4)
  w = np.random.default_rng(2).standard_normal(400)
  b = patcher(w)
  artifacts.patches("cutrule_patcher.png", w, {"CutRule patcher": b}, "CutRule patches on a Time-300B-style window")
  artifacts.json("patcher.json", {"patches": len(b), "starts": b})
  assert b[0] == 0 and np.all(np.diff(b) > 0) and b[-1] < 400


def test_pretrain_cutrule_saves_and_get_or_train(artifacts, tmp_path):
  path = str(tmp_path / "cutrule_pre.pt")
  model, scale, fp = get_pretrained_cutrule(path, **TINY_CUTRULE)                # missing -> pretrains
  assert (tmp_path / "cutrule_pre.pt").exists() and scale > 0
  model2, scale2, fp2 = get_pretrained_cutrule(path)                             # present -> just loads
  artifacts.json("pretrained_cutrule.json", {"fingerprint": fp, "return_scale": scale,
                                             "saved": torch.load(path, map_location="cpu")["final"]})
  assert fp == fp2 == state_fingerprint(model.state_dict()) and scale == scale2


def small_orcl(days=5):
  d = load_orcl()
  n = int(d.day_starts[days])
  starts = d.day_starts[:days]
  return OrclData(d.returns[:n], d.dt[:n], starts, int(starts[3]), int(starts[4]), d.time_unit)


@pytest.mark.slow
@pytest.mark.parametrize("time_aware", [True, False], ids=["time", "even"])
@pytest.mark.parametrize("patcher", ["power", "cutrule"])
def test_end_to_end_pretrain_then_finetune(patcher, time_aware, artifacts, tmp_path, monkeypatch, mamba_cls, mamba_kind):
  """Pretrain-if-missing (CutRule, then backbone), continue the CutRule on ORCL, fine-tune, evaluate - tiny settings.
  Runs with the real Mamba layers when available. Checkpoints go to a temp dir; the forecast report (CSV, metrics and
  figures) is written into this test's artifact folder."""
  from timesqueeze.training.finetune import finetune
  monkeypatch.chdir(tmp_path)                                                    # outputs/ goes to the temp dir
  sfx = "" if time_aware else "_ticks"
  model, res = finetune(
      patcher, context=128, stride=64, batch_size=4, epochs=2, mamba_cls=mamba_cls, data=small_orcl(), time_aware=time_aware,
      cutrule_epochs=2, cutrule_pretrain_kwargs=TINY_CUTRULE, results_dir=str(artifacts.dir),
      pretrain_kwargs={"steps": 2, "micro_batch": 2, "global_batch": 4, "context": 128, "num_workers": 0,
                       "val_every": 0, "ckpt_every": 1, "log_every": 1})
  assert (tmp_path / f"outputs/backbone_{patcher}.pt").exists()                    # shared by both spacings
  assert (tmp_path / f"outputs/finetuned_{patcher}{sfx}.pt").exists()
  assert (tmp_path / "outputs/cutrule_pretrained.pt").exists() == (patcher == "cutrule")
  test = res["test"]
  assert np.isfinite(list(test["mse"].values())).all() and test["compression"] >= 1.0
  assert 0 <= res["best_epoch"] <= 2                                 # model selection on the validation days
  for f in ("forecasts.csv", "metrics.json", "mse_by_horizon.png", "forecast_examples_h8.png",
            "forecast_examples_h64.png", "patch_boundaries.png", "loss_curves.png"):
    assert (artifacts.dir / f).exists(), f
  # a second call finds everything on disk and trains nothing new
  finetune(patcher, context=128, stride=64, batch_size=4, epochs=1, mamba_cls=mamba_cls, data=small_orcl(),
           time_aware=time_aware, cutrule_epochs=2, cutrule_pretrain_kwargs=TINY_CUTRULE,
           results_dir=str(tmp_path / "second"))


@pytest.mark.slow
@pytest.mark.parametrize("time_aware", [True, False], ids=["time", "even"])
@pytest.mark.parametrize("patcher", ["power", "cutrule"])
def test_quick_mode_settings_run_the_whole_pipeline(patcher, time_aware, artifacts, tmp_path, monkeypatch, mamba_cls):
  """The exact quick-test settings used by the command lines (1 epoch / step per stage) complete end to end."""
  from timesqueeze.training.finetune import finetune
  from timesqueeze.utils.quick import FINETUNE
  monkeypatch.chdir(tmp_path)
  kw = dict(FINETUNE)
  model, res = finetune(patcher, mamba_cls=mamba_cls, data=small_orcl(), time_aware=time_aware,
                        cutrule_pretrain_kwargs=dict(CUTRULE_PRETRAIN),
                        pretrain_kwargs=dict(BACKBONE_PRETRAIN), results_dir=str(artifacts.dir), **kw)
  assert (artifacts.dir / "metrics.json").exists() and np.isfinite(list(res["test"]["mse"].values())).all()


@pytest.mark.slow
def test_time_and_even_variants_share_pretraining_and_keep_separate_orcl_weights(artifacts, tmp_path, monkeypatch,
                                                                                 mamba_cls):
  """Both spacings use ONE Time-300B CutRule and ONE backbone (Time-300B has no timestamps; the second run reuses them
  untouched), and keep separate ORCL CutRules and fine-tuned models, which differ."""
  from timesqueeze.training.finetune import finetune
  from timesqueeze.utils.quick import FINETUNE
  monkeypatch.chdir(tmp_path)
  out_dir = tmp_path / "outputs"
  shared = ["cutrule_pretrained", "backbone_cutrule"]
  mtimes = {}
  for time_aware in (True, False):
    finetune("cutrule", mamba_cls=mamba_cls, data=small_orcl(), time_aware=time_aware,
             cutrule_pretrain_kwargs=dict(CUTRULE_PRETRAIN), pretrain_kwargs=dict(BACKBONE_PRETRAIN),
             results_dir=str(artifacts.dir / ("time" if time_aware else "even")), **dict(FINETUNE))
    mtimes[time_aware] = {n: (out_dir / f"{n}.pt").stat().st_mtime_ns for n in shared}
  assert mtimes[True] == mtimes[False]                                 # the even run reused them, did not retrain
  assert not any((out_dir / f"{n}_ticks.pt").exists() for n in shared)
  separate = ["checkpoint", "finetuned_cutrule"]
  differ = {}
  for n in separate:
    a, b = out_dir / f"{n}.pt", out_dir / f"{n}_ticks.pt"
    sa, sb = torch.load(a, map_location="cpu")["model"], torch.load(b, map_location="cpu")["model"]
    differ[n] = any(not torch.equal(sa[k], sb[k]) for k in sa)
  artifacts.json("shared_and_separate_weights.json", {"shared": shared, "separate": separate, "weights_differ": differ})
  assert all(differ.values())


def test_an_interrupted_pretraining_continues_from_its_checkpoint(artifacts, tmp_path, monkeypatch, mamba_cls):
  """Pretraining resumes by default (train-all relies on it after an interruption): a 2-step run, then the same call
  asking for 3 steps continues at step 2 instead of starting over; restart (resume=False) starts from step 0."""
  from timesqueeze.training.pretrain import pretrain_backbone
  monkeypatch.chdir(tmp_path)
  kw = dict(BACKBONE_PRETRAIN, ckpt_every=1, mamba_cls=mamba_cls, out_path=str(tmp_path / "b.pt"))
  kw.pop("steps")
  pretrain_backbone(steps=2, **kw)
  hist = lambda: torch.load(tmp_path / "b.pt", map_location="cpu")["history"]
  first = [h["step"] for h in hist() if "loss" in h]
  pretrain_backbone(steps=3, **kw)                                  # "interrupted" at 2 of 3: continues
  resumed = [h["step"] for h in hist() if "loss" in h]
  pretrain_backbone(steps=2, resume=False, **kw)                    # restart: from step 0 again
  restarted = [h["step"] for h in hist() if "loss" in h]
  artifacts.json("resume.json", {"first": first, "resumed": resumed, "restarted": restarted})
  assert first == [1, 2] and resumed == [1, 2, 3] and restarted == [1, 2]


@pytest.mark.slow
def test_an_interrupted_finetune_continues_after_its_last_epoch(artifacts, tmp_path, monkeypatch, mamba_cls):
  """A crash during epoch 2 of 2: the next call resumes after epoch 1 (same history and best epoch so far) instead of
  starting over - what train-all relies on after e.g. a Windows crash."""
  import timesqueeze.training.finetune as ft
  from timesqueeze.utils.quick import FINETUNE
  monkeypatch.chdir(tmp_path)
  kw = dict(FINETUNE, epochs=2)
  real, calls = ft.train_one_epoch, {"n": 0}

  def crash_in_epoch_2(*a, **k):
    calls["n"] += 1
    if calls["n"] == 2:
      raise RuntimeError("simulated crash")
    return real(*a, **k)

  monkeypatch.setattr(ft, "train_one_epoch", crash_in_epoch_2)
  with pytest.raises(RuntimeError, match="simulated crash"):
    ft.finetune("power", mamba_cls=mamba_cls, data=small_orcl(), pretrain_kwargs=dict(BACKBONE_PRETRAIN),
                results_dir=str(artifacts.dir / "first"), **kw)
  monkeypatch.setattr(ft, "train_one_epoch", real)
  epochs_run = []
  monkeypatch.setattr(ft, "train_one_epoch", lambda *a, **k: epochs_run.append(1) or real(*a, **k))
  model, res = ft.finetune("power", mamba_cls=mamba_cls, data=small_orcl(), pretrain_kwargs=dict(BACKBONE_PRETRAIN),
                           results_dir=str(artifacts.dir / "resumed"), **kw)
  artifacts.json("resume.json", {"epochs_trained_after_resume": len(epochs_run), "history": res["history"]})
  assert len(epochs_run) == 1 and [h["epoch"] for h in res["history"]] == [1, 2] and 0 <= res["best_epoch"] <= 2
