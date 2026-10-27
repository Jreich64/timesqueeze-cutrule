"""Shared test fixtures.

* `mamba_cls`: the real mamba_ssm.Mamba when it is installed and CUDA is available (WSL2 / Linux), otherwise the causal
  stand-in in stub_mamba.py (native Windows). Tests report which one they ran with.
* `artifacts`: every test gets a folder outputs/test_results/<test name>/ with run_info.json, result.json (pass/fail,
  duration) and whatever results and figures the test itself writes (forecast CSVs, plots, ...).
"""
import re
import time
from pathlib import Path

import pytest
import torch
from stub_mamba import StubMamba

from timesqueeze.utils import reporting

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "test_results"


def _real_mamba():
  if not torch.cuda.is_available():
    return None
  try:
    from mamba_ssm import Mamba
  except Exception:
    return None
  return Mamba


@pytest.fixture(scope="session")
def mamba_cls():
  return _real_mamba() or StubMamba


@pytest.fixture(scope="session")
def mamba_kind(mamba_cls):
  return "stub" if mamba_cls is StubMamba else "real"


@pytest.fixture(scope="session")
def device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def exact_fp32_matmul():
  """Production code enables TF32 matmuls (utils.set_seed) for speed; tests comparing outputs to ~1e-5 need exact fp32.
  Without this a test's result depended on whether another test in the same worker had called set_seed first (a padded
  vs unpadded forward differs by 2.7e-4 with TF32 on, 1.8e-7 with it off)."""
  prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision("highest")
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev[0], prev[1]
  torch.set_float32_matmul_precision(prev[2])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
  outcome = yield
  rep = outcome.get_result()
  setattr(item, f"rep_{rep.when}", rep)


class Artifacts:
  """Where a test writes its results and figures."""

  def __init__(self, directory):
    self.dir = directory

  def json(self, name, obj):
    return reporting.save_json(self.dir / name, obj)

  def csv(self, name, header, rows):
    return reporting.save_csv(self.dir / name, header, rows)

  def lines(self, name, curves, title, xlabel="", ylabel="", **kw):
    return reporting.plot_lines(self.dir / name, curves, title, xlabel, ylabel, **kw)

  def bars(self, name, categories, groups, title, ylabel=""):
    return reporting.plot_bars(self.dir / name, categories, groups, title, ylabel)

  def hist(self, name, values, title, xlabel, **kw):
    return reporting.plot_hist(self.dir / name, values, title, xlabel, **kw)

  def patches(self, name, x, rules, title, **kw):
    return reporting.plot_patches(self.dir / name, x, rules, title, **kw)

  def forecasts(self, name, contexts, truths, preds, horizon, title, **kw):
    return reporting.plot_forecasts(self.dir / name, contexts, truths, preds, horizon, title, **kw)


@pytest.fixture(autouse=True)
def artifacts(request, mamba_kind, device):
  name = re.sub(r"[^\w\-\[\]]+", "_", request.node.name)
  d = RESULTS / name
  d.mkdir(parents=True, exist_ok=True)
  a = Artifacts(d)
  a.json("run_info.json", {"test": request.node.nodeid, "mamba": mamba_kind, "device": str(device),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
  t0 = time.time()
  yield a
  rep = getattr(request.node, "rep_call", None)
  a.json("result.json", {"test": request.node.nodeid, "outcome": rep.outcome if rep else "unknown",
                         "seconds": round(time.time() - t0, 2), "mamba": mamba_kind})
