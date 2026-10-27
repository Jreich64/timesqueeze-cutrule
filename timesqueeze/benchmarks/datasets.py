"""The long-term forecasting benchmarks of the paper (section 4): ETTh1, ETTh2, ETTm1, ETTm2 and Weather.

Source: the CSV files of the Time-Series-Library (thuml), mirrored on Hugging Face as the dataset thuml/Time-Series-Library.
They are downloaded once to data/benchmarks/ (public files; no token needed).

The protocol follows that library's data_loader.py (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom), which is what
Time-MoE and the paper report against:
  * splits - ETTh: 12 / 4 / 4 months of hourly rows; ETTm: the same months at 15-minute resolution; Weather: 70 / 10 / 20 %.
  * every channel (all columns except the date) is a separate univariate series ("channel independence").
  * a StandardScaler is fitted per channel on the train split only and applied to the whole series; metrics are MSE and
    MAE in these scaled units.
  * a window = `context` input points + H targets, at stride 1. The context of the first test window reaches back into
    the validation rows (the test split's start is moved back by `context`), so every test target is a test row.
    Each horizon H has its own window set: every start whose H targets all lie inside the split.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.paths import DATA_DIR

BENCH_DIR = DATA_DIR / "benchmarks"
HF_REPO = "thuml/Time-Series-Library"
HOURS_PER_MONTH = 30 * 24

# name -> (file inside the Hugging Face repo, rows per hour (ETT) or None for the 70/10/20 split)
DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", 1),
    "ETTh2": ("ETT-small/ETTh2.csv", 1),
    "ETTm1": ("ETT-small/ETTm1.csv", 4),
    "ETTm2": ("ETT-small/ETTm2.csv", 4),
    "Weather": ("weather/weather.csv", None),
}
PAPER_HORIZONS = (96, 192, 336, 720)


def download(name):
  """Path of the benchmark CSV, downloaded to data/benchmarks/ the first time."""
  repo_file, _ = DATASETS[name]
  path = BENCH_DIR / f"{name}.csv"
  if not path.exists():
    from huggingface_hub import hf_hub_download
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} from {HF_REPO}")
    tmp = hf_hub_download(HF_REPO, repo_file, repo_type="dataset", local_dir=BENCH_DIR / "_hf")
    path.write_bytes(open(tmp, "rb").read())
  return path


def split_rows(n_rows, rows_per_hour):
  """[(start, end)] row ranges of the train / validation / test targets (before the context is prepended)."""
  if rows_per_hour is not None:
    m = HOURS_PER_MONTH * rows_per_hour
    return [(0, 12 * m), (12 * m, 16 * m), (16 * m, 20 * m)]
  n_train, n_test = int(n_rows * 0.7), int(n_rows * 0.2)
  return [(0, n_train), (n_train, n_rows - n_test), (n_rows - n_test, n_rows)]


@dataclass
class BenchmarkData:
  """One benchmark: scaled values [rows, channels] and the row range of each split."""
  name: str
  values: np.ndarray          # float32, scaled with the train split's per-channel mean / std
  columns: list
  splits: dict                # "train" / "val" / "test" -> (start, end) rows of that split's targets

  def window_starts(self, split, context, horizon, stride=1):
    """Starts of the windows [s, s + context + horizon) whose targets lie in `split` (the paper's window set for H)."""
    lo, hi = self.splits[split]
    first = max(lo - context, 0) if split != "train" else 0      # test/val contexts reach back into the previous split
    return np.arange(first, hi - context - horizon + 1, stride)


def load_benchmark(name, data_dir=None):
  """Load, split and scale one benchmark (see the module docstring for the protocol)."""
  path = download(name) if data_dir is None else data_dir / f"{name}.csv"
  df = pd.read_csv(path)
  cols = [c for c in df.columns if c.lower() != "date"]
  raw = df[cols].to_numpy(dtype=np.float64)
  bounds = split_rows(len(raw), DATASETS[name][1])
  train = raw[bounds[0][0]:bounds[0][1]]
  mean, std = train.mean(0), train.std(0)                         # sklearn StandardScaler: population std
  std = np.where(std > 0, std, 1.0)
  values = ((raw - mean) / std).astype(np.float32)
  return BenchmarkData(name, values, cols, dict(zip(("train", "val", "test"), bounds)))
