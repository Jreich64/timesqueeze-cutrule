"""Download extra Time-300B pretraining windows from Hugging Face with HTTP range requests.

Mirrors the layout of the existing data/Time-300B-subset: each window is 131072 float32 values cut from a
.bin file at `float_offset`, saved as series_XXXXXX.npy, with an index.json describing the source. Adjacent
windows are fetched together in one range request (a "cluster"). Sampling follows the TimeSqueeze paper:
datasets are weighted by number of points, with the three dominant Nature sources (WeatherBench hourly, CMIP6,
ERA5) downsampled to 30%.

Needs the HF_TOKEN environment variable. The token is only sent as an Authorization header, never printed.

    uv run python -m timesqueeze download --gb 10

Each run writes a new folder (data/Time-300B-subset2, -subset3, ...) and never re-downloads a range already on disk;
pretraining uses every Time-300B-subset* folder it finds. Running the same command again resumes an interrupted download.
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_url

from ..utils.paths import DATA_DIR
from .time300b_segments import build_segment_index
from .time300b_windows import SUBSET_PREFIX, local_subsets

REPO = "Maple728/Time-300B"
WINDOW = 131072
CLUSTER = 32                         # adjacent windows per range request (~16.8 MB)
TOP3_NATURE = ("nature/weatherbench_hourly", "nature/cmip6", "nature/era5")
NATURE_KEEP = 0.3


def _headers():
  token = os.environ.get("HF_TOKEN")
  if not token:
    raise SystemExit("Set the HF_TOKEN environment variable to a Hugging Face access token.")
  return {"Authorization": f"Bearer {token}"}


def file_sizes(cache=DATA_DIR / "time300b_file_sizes.json"):
  if cache.exists():
    return json.load(open(cache))
  info = HfApi(token=os.environ["HF_TOKEN"]).dataset_info(REPO, files_metadata=True)
  sizes = {s.rfilename: s.size for s in info.siblings}
  json.dump(sizes, open(cache, "w"))
  return sizes


def _existing_ranges(dirs):
  """{bin path: [(start_float, end_float)]} for windows already on disk."""
  taken = {}
  for d in dirs:
    idx_path = Path(d) / "index.json"
    if not idx_path.exists():
      continue
    for e in json.load(open(idx_path)):
      taken.setdefault(e["source"], []).append((e["float_offset"], e["float_offset"] + e["length"]))
  return taken


def make_plan(n_windows, existing_dirs, seed=0):
  """Choose clusters of adjacent windows: [(bin path, float_offset, n_windows)], no overlaps with existing data."""
  rng = random.Random(seed)
  sizes = file_sizes()
  bins = [(n, s // 4) for n, s in sizes.items() if n.endswith(".bin") and s // 4 >= WINDOW * CLUSTER]
  weights = [f * (NATURE_KEEP if n.startswith(TOP3_NATURE) else 1.0) for n, f in bins]
  taken = _existing_ranges(existing_dirs)
  plan, remaining = [], n_windows
  attempts = 0
  while remaining > 0 and attempts < 100 * n_windows:
    attempts += 1
    (name, floats), = rng.choices(bins, weights=weights, k=1)
    k = min(CLUSTER, remaining)
    span = k * WINDOW
    start = rng.randrange(0, floats - span + 1)
    ranges = taken.setdefault(name, [])
    if any(start < e and s < start + span for s, e in ranges):
      continue
    ranges.append((start, start + span))
    plan.append((name, start, k))
    remaining -= k
  return plan


def _fetch(name, float_offset, n_windows, retries=8):
  url = hf_hub_url(REPO, name, repo_type="dataset")
  lo = float_offset * 4
  hi = (float_offset + n_windows * WINDOW) * 4 - 1
  for attempt in range(retries):
    try:
      r = requests.get(url, headers={**_headers(), "Range": f"bytes={lo}-{hi}"}, timeout=180)
      if r.status_code in (429, 500, 502, 503, 504):
        time.sleep(min(float(r.headers.get("Retry-After", 2 ** attempt)), 60))
        continue
      r.raise_for_status()
      if len(r.content) != hi - lo + 1:
        raise IOError(f"short read: {len(r.content)} of {hi - lo + 1} bytes")
      return np.frombuffer(r.content, dtype="<f4").reshape(n_windows, WINDOW)
    except (requests.RequestException, IOError):
      if attempt == retries - 1:
        raise
      time.sleep(2 ** attempt)


def rebuild_index(out_dir, plan_windows):
  """index.json lists only windows whose .npy file exists."""
  out = []
  for i, (name, off) in enumerate(plan_windows):
    f = f"series_{i:06d}.npy"
    if (out_dir / f).exists():
      out.append({"file": f, "source": name, "float_offset": off, "length": WINDOW})
  json.dump(out, open(out_dir / "index.json", "w"))
  return len(out)


def next_subset_dir(existing):
  """data/Time-300B-subset2, -subset3, ...: the first numbered folder not taken. A folder whose index.json lists fewer
  windows than its plan.json (an interrupted download) is reused instead, so running the command again resumes it."""
  for f in sorted(DATA_DIR.glob(SUBSET_PREFIX + "*")):
    if (f / "plan.json").exists():
      planned = sum(k for _, _, k in json.load(open(f / "plan.json")))
      listed = len(json.load(open(f / "index.json"))) if (f / "index.json").exists() else 0
      if listed < planned:
        return f
  n = 2
  while (DATA_DIR / f"{SUBSET_PREFIX}{n}").exists():
    n += 1
  return DATA_DIR / f"{SUBSET_PREFIX}{n}"


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--gb", type=float, default=10.0, help="approximate download size in GB (decimal)")
  p.add_argument("--out", default=None, help="default: the next free data/Time-300B-subsetN folder")
  p.add_argument("--exclude", nargs="*", default=None,
                 help="existing window folders whose ranges must not be re-downloaded (default: every local subset)")
  p.add_argument("--workers", type=int, default=6)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--max-clusters", type=int, default=None, help="stop after N clusters (for testing)")
  args = p.parse_args()
  _headers()

  existing = [DATA_DIR / name for name in local_subsets()]
  out_dir = Path(args.out) if args.out else next_subset_dir(existing)
  exclude = [e for e in existing if e != out_dir] if args.exclude is None else [Path(e) for e in args.exclude]
  print(f"downloading into {out_dir}; not re-downloading anything in {', '.join(e.name for e in exclude) or 'nothing'}")
  out_dir.mkdir(parents=True, exist_ok=True)
  plan_path = out_dir / "plan.json"
  n_windows = int(args.gb * 1e9 // (WINDOW * 4))
  if plan_path.exists():
    plan = [tuple(c) for c in json.load(open(plan_path))]   # resume the same plan
  else:
    plan = make_plan(n_windows, [*exclude, out_dir], seed=args.seed)
    json.dump(plan, open(plan_path, "w"))
  plan_windows = [(name, off + j * WINDOW) for name, off, k in plan for j in range(k)]
  first = {}
  w = 0
  for ci, (name, off, k) in enumerate(plan):
    first[ci] = w
    w += k
  print(f"plan: {len(plan_windows)} windows (~{len(plan_windows) * WINDOW * 4 / 1e9:.1f} GB) in {len(plan)} clusters")

  done = 0

  def work(ci):
    name, off, k = plan[ci]
    start = first[ci]
    if all((out_dir / f"series_{start + j:06d}.npy").exists() for j in range(k)):
      return ci
    block = _fetch(name, off, k)
    for j in range(k):
      tmp = out_dir / f"series_{start + j:06d}.tmp.npy"
      np.save(tmp, block[j])
      os.replace(tmp, out_dir / f"series_{start + j:06d}.npy")
    return ci

  todo = list(range(len(plan)))[: args.max_clusters]
  try:
    with ThreadPoolExecutor(args.workers) as pool:
      for fut in as_completed([pool.submit(work, ci) for ci in todo]):
        fut.result()
        done += 1
        if done % 25 == 0 or done == len(todo):
          print(f"{done}/{len(todo)} clusters", flush=True)
  finally:
    print(f"index.json lists {rebuild_index(out_dir, plan_windows)} windows")
  # the series boundaries inside the new windows (see time300b_segments), so pretraining can use them right away
  win, _, length = build_segment_index(out_dir, force=True)
  print(f"{out_dir.name}: {len(win)} series pieces, {int(length.sum()) / 1e9:.2f}B points")


if __name__ == "__main__":
  main()
