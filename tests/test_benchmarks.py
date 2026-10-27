"""The ETT / Weather benchmarks: standard splits, rolling forecasts past the 64-step head, and a quick end-to-end run."""
import numpy as np
import pytest
import torch
from torch import nn

from timesqueeze.benchmarks.datasets import BenchmarkData, load_benchmark, split_rows
from timesqueeze.benchmarks.rolling import _head_for, rolling_forecast
from timesqueeze.patching.power_rule import power_rule_boundaries


class Persistence(nn.Module):
  """Stand-in forecaster: every head forecasts no change (all steps 0), i.e. the last input value persists. Records
  the heads used."""

  def __init__(self, horizons=(1, 8, 32, 64)):
    super().__init__()
    self.horizons, self.used, self.w = tuple(horizons), [], nn.Parameter(torch.zeros(1))

  def forward(self, x, patch_pos, patch_mask, patch_id, time=None, dtf=None):
    preds = {h: torch.zeros(*x.shape, h) for h in self.horizons}
    self.used.append(None)
    return preds, torch.zeros(())


def test_split_rows_match_the_time_series_library():
  m = 30 * 24
  assert split_rows(17420, 1) == [(0, 12 * m), (12 * m, 16 * m), (16 * m, 20 * m)]          # ETTh: 8640 / 2880 / 2880
  assert split_rows(69680, 4) == [(0, 48 * m), (48 * m, 64 * m), (64 * m, 80 * m)]          # ETTm, 15-minute rows
  n = 52696                                                                                 # Weather
  assert split_rows(n, None) == [(0, int(n * 0.7)), (int(n * 0.7), n - int(n * 0.2)), (n - int(n * 0.2), n)]


def test_test_windows_are_the_paper_window_sets():
  """Same count and placement as Dataset_ETT_hour: test rows [11520-512, 14400), len - seq - pred + 1 windows."""
  data = BenchmarkData("toy", np.zeros((17420, 2), np.float32), ["a", "b"], dict(zip(("train", "val", "test"),
                                                                                     split_rows(17420, 1))))
  for H in (96, 192, 336, 720):
    s = data.window_starts("test", 512, H)
    assert s[0] == 11520 - 512 and s[-1] + 512 + H == 14400 and len(s) == 2880 - H + 1
  tr = data.window_starts("train", 512, 64)
  assert tr[0] == 0 and tr[-1] + 512 + 64 == 8640


def test_head_schedule_and_rolling_forecast(artifacts):
  assert [_head_for(r, (1, 8, 32, 64)) for r in (720, 64, 63, 16, 8, 7, 1)] == [64, 64, 32, 8, 8, 1, 1]
  model = Persistence()
  rng = np.random.default_rng(0)
  ctx = rng.standard_normal((5, 512)) * 3 + 10
  out = rolling_forecast(model, ctx, 720, power_rule_boundaries, torch.device("cpu"))
  artifacts.json("rolls.json", {"rolls": len(model.used)})
  assert out.shape == (5, 720) and len(model.used) == 13                     # 11 x 64 + 2 x 8
  assert np.allclose(out, ctx[:, -1:], atol=1e-4)                           # persistence, back in the original units


def test_etth1_loads_with_train_fitted_scaling(artifacts):
  try:
    data = load_benchmark("ETTh1")
  except Exception as e:                                                    # no network
    pytest.skip(f"ETTh1 not available: {type(e).__name__}")
  lo, hi = data.splits["train"]
  train = data.values[lo:hi]
  artifacts.json("etth1.json", {"shape": data.values.shape, "columns": data.columns,
                                "train_mean": train.mean(0).tolist(), "train_std": train.std(0).tolist()})
  assert data.values.shape == (17420, 7) and data.columns[-1] == "OT"
  assert np.allclose(train.mean(0), 0, atol=1e-4) and np.allclose(train.std(0), 1, atol=1e-3)


@pytest.mark.slow
def test_benchmark_quick_end_to_end(artifacts, tmp_path, monkeypatch, mamba_cls):
  """Zero-shot and full-shot on ETTh1 with the quick-mode settings: finite scores for every paper horizon and a report."""
  from timesqueeze.training.benchmark import run_benchmarks
  from timesqueeze.utils.quick import BACKBONE_PRETRAIN, BENCHMARK
  try:
    load_benchmark("ETTh1")
  except Exception as e:
    pytest.skip(f"ETTh1 not available: {type(e).__name__}")
  monkeypatch.chdir(tmp_path)
  res = run_benchmarks("power", datasets=("ETTh1",), mamba_cls=mamba_cls, results_dir=str(artifacts.dir),
                       pretrain_kwargs=dict(BACKBONE_PRETRAIN), **BENCHMARK)
  for mode in ("zero-shot", "full-shot"):
    for H in (96, 192, 336, 720):
      assert np.isfinite(res["ETTh1"][mode][H]["mse"]) and res["ETTh1"][mode][H]["n"] > 0
  figures = [f"forecasts_ETTh1_{m}_H{H}.png" for m in ("zero-shot", "full-shot") for H in (96, 192, 336, 720)]
  for f in ("metrics.json", "results.csv", "mse_ETTh1.png", *figures):
    assert (artifacts.dir / f).exists(), f


class CrashAfter(Persistence):
  """Persistence that raises on forward call number `after` + 1 (a crash part-way through scoring)."""

  def __init__(self, after):
    super().__init__()
    self.after = after

  def forward(self, *a, **k):
    if len(self.used) >= self.after:
      raise RuntimeError("simulated crash")
    return super().forward(*a, **k)


def test_interrupted_scoring_continues_with_the_same_scores(tmp_path, artifacts):
  """Running sums saved during scoring: a crash, then a second call, gives the same scores as one uninterrupted call."""
  from timesqueeze.benchmarks.evaluate import evaluate_benchmark
  rng = np.random.default_rng(0)
  vals = np.cumsum(rng.standard_normal((1400, 2)), 0).astype(np.float32)
  data = BenchmarkData("toy", vals, ["a", "b"], {"train": (0, 600), "val": (600, 800), "test": (800, 1400)})
  kw = dict(context=128, horizons=(8, 64), stride=4, batch_size=16, log=None)
  ref, _ = evaluate_benchmark(Persistence(), data, power_rule_boundaries, torch.device("cpu"), **kw)
  state = str(tmp_path / "scores.pt")
  with pytest.raises(RuntimeError, match="simulated crash"):
    evaluate_benchmark(CrashAfter(3), data, power_rule_boundaries, torch.device("cpu"), state_path=state,
                       save_every=-1, **kw)                                  # 3 batches of one roll each, then a crash
  resumed_model = Persistence()
  got, _ = evaluate_benchmark(resumed_model, data, power_rule_boundaries, torch.device("cpu"), state_path=state, **kw)
  artifacts.json("resumed_scores.json", {"ref": ref, "got": got, "rolls_after_resume": len(resumed_model.used)})
  total_batches = -(-len(data.window_starts("test", 128, 8, 4)) * 2 // 16)
  assert len(resumed_model.used) == total_batches - 3                         # only the batches not scored before
  for H in (8, 64):
    assert got[H]["n"] == ref[H]["n"] and np.isclose(got[H]["mse"], ref[H]["mse"]) and np.isclose(got[H]["mae"], ref[H]["mae"])


def test_train_loader_skip_continues_the_same_order():
  from timesqueeze.benchmarks.evaluate import benchmark_train_loader
  vals = np.random.default_rng(1).standard_normal((900, 2)).astype(np.float32)
  data = BenchmarkData("toy", vals, ["a", "b"], {"train": (0, 900), "val": (0, 0), "test": (0, 0)})
  full = [b["y"] for b in benchmark_train_loader(data, power_rule_boundaries, 128, 8, stride=16, seed=3)]
  rest = [b["y"] for b in benchmark_train_loader(data, power_rule_boundaries, 128, 8, stride=16, seed=3, skip_batches=2)]
  assert len(rest) == len(full) - 2 and all(torch.equal(a, b) for a, b in zip(full[2:], rest))


def test_interrupted_full_shot_finetuning_continues_from_its_checkpoint(tmp_path, monkeypatch, artifacts):
  """Crash after 2 optimizer steps, call again: the same weights as one uninterrupted epoch (CPU, so deterministic);
  a third call only loads the finished weights."""
  from stub_mamba import StubMamba

  import timesqueeze.training.benchmark as bm
  from timesqueeze.model.forecaster import TimeSqueeze
  from timesqueeze.model.moe import MoEConfig

  def model():
    torch.manual_seed(0)
    return TimeSqueeze(d_model=32, d_state=8, moe_config=MoEConfig(hidden_size=32, intermediate_size=64, num_layers=1,
                                                                   num_heads=4, num_experts=2), mamba_cls=StubMamba)
  vals = np.cumsum(np.random.default_rng(2).standard_normal((700, 2)), 0).astype(np.float32)
  data = BenchmarkData("toy", vals, ["a", "b"], {"train": (0, 700), "val": (0, 0), "test": (0, 0)})
  cpu, kw = torch.device("cpu"), dict(batch_size=4, stride=40)
  ref = model()
  bm.full_shot_finetune(ref, data, power_rule_boundaries, cpu, 128, **kw)

  real = bm.train_one_epoch
  def crash_after_two(*a, after_step=None, **k):
    def step(n, loss):
      after_step(n, loss)
      if n == 2:
        raise RuntimeError("simulated crash")
    return real(*a, after_step=step, **k)
  ckpt = str(tmp_path / "ft.pt")
  monkeypatch.setattr(bm, "train_one_epoch", crash_after_two)
  with pytest.raises(RuntimeError, match="simulated crash"):
    bm.full_shot_finetune(model(), data, power_rule_boundaries, cpu, 128, ckpt_path=ckpt, save_every=-1, **kw)
  monkeypatch.setattr(bm, "train_one_epoch", real)
  assert torch.load(ckpt, weights_only=False)["step"] == 2
  resumed = model()
  bm.full_shot_finetune(resumed, data, power_rule_boundaries, cpu, 128, ckpt_path=ckpt, **kw)
  diff = max(float((a - b).abs().max()) for a, b in zip(ref.state_dict().values(), resumed.state_dict().values()))
  artifacts.json("finetune_resume.json", {"max_weight_diff": diff})
  assert diff < 1e-5 and torch.load(ckpt, weights_only=False)["done"]
  loaded = model()
  monkeypatch.setattr(bm, "train_one_epoch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not train")))
  bm.full_shot_finetune(loaded, data, power_rule_boundaries, cpu, 128, ckpt_path=ckpt, **kw)
  assert all(torch.equal(a, b) for a, b in zip(resumed.state_dict().values(), loaded.state_dict().values()))


def test_an_interrupted_benchmark_skips_what_it_already_scored(tmp_path, monkeypatch, artifacts):
  """metrics.json is written after every (dataset, mode); a rerun with the same settings does not score them again
  (here no backbone may even be loaded), and other settings start over."""
  import json

  import timesqueeze.training.benchmark as bm
  try:
    load_benchmark("ETTh1")
  except Exception as e:
    pytest.skip(f"ETTh1 not available: {type(e).__name__}")
  monkeypatch.chdir(tmp_path)
  settings = {"patcher": "power", "context": 512, "stride": 4, "train_stride": 1, "max_windows": None,
              "horizons": [96, 192, 336, 720]}
  score = {str(H): {"mse": 0.5, "mae": 0.4, "n": 10} for H in (96, 192, 336, 720)}
  (tmp_path / "res").mkdir()
  json.dump({"settings": settings, "results": {"ETTh1": {"zero-shot": score}}}, open(tmp_path / "res/metrics.json", "w"))
  monkeypatch.setattr(bm, "get_backbone", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")))
  res = bm.run_benchmarks("power", datasets=("ETTh1",), modes=("zero-shot",), stride=4, results_dir=str(tmp_path / "res"))
  assert res["ETTh1"]["zero-shot"][96]["mse"] == 0.5 and (tmp_path / "res/results.csv").exists()
  with pytest.raises(AssertionError, match="must not load"):              # different settings: scored again
    bm.run_benchmarks("power", datasets=("ETTh1",), modes=("zero-shot",), stride=8, results_dir=str(tmp_path / "res"))
  artifacts.json("resumed.json", {"kept": res["ETTh1"]["zero-shot"][96]})


def test_a_part_with_empty_figures_is_scored_again(tmp_path, monkeypatch):
  """A crash right after the figures were created can leave them empty: such a (dataset, mode) is not kept on resume."""
  import json

  import timesqueeze.training.benchmark as bm
  try:
    load_benchmark("ETTh1")
  except Exception as e:
    pytest.skip(f"ETTh1 not available: {type(e).__name__}")
  monkeypatch.chdir(tmp_path)
  settings = {"patcher": "power", "context": 512, "stride": 4, "train_stride": 1, "max_windows": None,
              "horizons": [96, 192, 336, 720]}
  score = {str(H): {"mse": 0.5, "mae": 0.4, "n": 10} for H in (96, 192, 336, 720)}
  res = tmp_path / "res"
  res.mkdir()
  json.dump({"settings": settings, "results": {"ETTh1": {"zero-shot": score}}}, open(res / "metrics.json", "w"))
  for H in (96, 192, 336, 720):
    (res / f"forecasts_ETTh1_zero-shot_H{H}.png").write_bytes(b"" if H == 336 else b"png")
  monkeypatch.setattr(bm, "get_backbone", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")))
  with pytest.raises(AssertionError, match="must not load"):              # the empty H=336 figure: scored again
    bm.run_benchmarks("power", datasets=("ETTh1",), modes=("zero-shot",), stride=4, results_dir=str(res))


def test_joint_and_trilevel_score_their_phase_a_forecaster_with_its_learned_cutrule(tmp_path, monkeypatch):
  """--patcher joint / trilevel: method 3's / 4's Time-300B forecaster, patched by the CutRule it learned (not the
  stage-1 one), with results under their own name."""
  import timesqueeze.training.benchmark as bm
  import timesqueeze.training.joint as joint
  from timesqueeze.cutrule.model import CutRule
  from timesqueeze.cutrule.stages import CutRulePatcher
  cut = CutRule(np.ones(2), device="cpu")
  cut.b2.data.fill_(1.25)                                                   # marks the learned CutRule
  loaded, seen = [], []

  def fake_pretrained(leader, device, mamba_cls=None, **kw):
    loaded.append(leader)
    return Persistence(), cut.state_dict(), 0.5

  def fake_eval(model, data, patch_fn, *a, **k):
    seen.append((model, patch_fn))
    return {H: {"mse": 0.1, "mae": 0.2, "n": 1} for H in (96, 192, 336, 720)}, []
  monkeypatch.setattr(joint, "get_joint_pretrained", fake_pretrained)
  monkeypatch.setattr(bm, "get_backbone", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not a staged backbone")))
  monkeypatch.setattr(bm, "load_benchmark", lambda name: BenchmarkData(name, np.zeros((10, 2), np.float32), ["a", "b"],
                                                                      {"train": (0, 10), "val": (0, 0), "test": (0, 0)}))
  monkeypatch.setattr(bm, "evaluate_benchmark", fake_eval)
  for patcher, leader in (("joint", "reinforce"), ("trilevel", "trilevel")):
    res = bm.run_benchmarks(patcher, datasets=("ETTh1",), modes=("zero-shot",), results_dir=str(tmp_path / patcher))
    model, patch_fn = seen[-1]
    assert loaded[-1] == leader and res["ETTh1"]["zero-shot"][96]["mse"] == 0.1
    assert isinstance(model, Persistence) and isinstance(patch_fn, CutRulePatcher)
    assert patch_fn.segmenter.b2 == 1.25 and patch_fn.return_scale == 0.5
    assert (tmp_path / patcher / "metrics.json").exists()


def test_reports_are_written_whole_with_no_temporary_files_left(tmp_path):
  from timesqueeze.utils.reporting import plot_forecasts, save_csv, save_json
  save_json(tmp_path / "a.json", {"x": [1, 2]})
  save_csv(tmp_path / "a.csv", ["h", "v"], [[1, 2.5]])
  plot_forecasts(tmp_path / "f.png", [np.arange(20.0)], [np.ones(4)], [np.zeros(4)], 4, "t")
  import json
  assert json.load(open(tmp_path / "a.json")) == {"x": [1, 2]}
  assert (tmp_path / "a.csv").read_text().splitlines() == ["h,v", "1,2.5"]
  assert (tmp_path / "f.png").read_bytes()[:4] == b"\x89PNG"
  assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv", "a.json", "f.png"]
