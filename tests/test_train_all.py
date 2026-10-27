"""train-all: the stage plan and the resumable status file (stages are faked; each real stage has its own tests)."""
import json

import timesqueeze.training.train_all as ta


def test_stage_plan_order_and_arguments():
  stages = dict(ta.stage_list(1234, 8))
  names = list(stages)
  assert names[:4] == ["finetune-cutrule-time", "finetune-power-time", "finetune-cutrule-even", "finetune-power-even"]
  assert names[4:8] == ["power-bilevel-time", "power-bilevel-even", "benchmark-cutrule", "benchmark-power"]
  assert names[8:12] == ["joint-time", "joint-even", "trilevel-time", "trilevel-even"]
  assert names[12:] == ["benchmark-joint", "benchmark-trilevel"]
  assert stages["benchmark-trilevel"] == ["benchmark", "--patcher", "trilevel", "--stride", "8", "--train-stride", "8",
                                          "--pretrain-steps", "1234"]
  assert stages["finetune-power-even"] == ["finetune", "--patcher", "power", "--spacing", "even", "--pretrain-steps", "1234"]
  assert stages["benchmark-cutrule"][:7] == ["benchmark", "--patcher", "cutrule", "--stride", "8", "--train-stride", "8"]
  assert stages["trilevel-even"] == ["trilevel", "--spacing", "even", "--pretrain-steps", "1234"]   # same Time-300B budget


def test_failures_do_not_stop_the_run_and_done_stages_are_skipped(tmp_path, monkeypatch, artifacts):
  monkeypatch.chdir(tmp_path)
  ta.use_quick_outputs(True)
  calls = []

  def fake(name, args, quick, log_dir, restart=False):
    calls.append(name)
    return 1 if name == "power-bilevel-time" else 0

  monkeypatch.setattr(ta, "run_stage", fake)
  status = ta.train_all(quick=True)
  assert len(calls) == 14 and status["power-bilevel-time"]["result"].startswith("failed")
  assert all(s["result"] == "done" for n, s in status.items() if n != "power-bilevel-time")
  calls.clear()
  ta.train_all(quick=True)                                         # second run: only the failed stage again
  assert calls == ["power-bilevel-time"]
  saved = json.load(open(tmp_path / "outputs/quick/train_all_status.json"))
  artifacts.json("status.json", saved)
  assert set(saved) == {n for n, _ in ta.stage_list(5000, 4)}
