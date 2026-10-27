"""Recover the true series boundaries inside Time-300B windows.

The windows in data/Time-300B-subset* are 131072-float cuts at arbitrary offsets of concatenated .bin files, so a
window can contain the end of one series and the start of the next. Training windows must come from a single
series, so for every window we look up the dataset's meta.json (per-series offset/length, global across the
dataset's .bin files) and store the series segments that fall inside the window.

meta.json files are large (up to ~430 MB), so they are streamed to data/meta_cache and parsed with a chunked
regex instead of json.load. Needs HF_TOKEN.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_url

from ..utils.paths import DATA_DIR

REPO = "Maple728/Time-300B"
META_CACHE = DATA_DIR / "meta_cache"
_PAIR = re.compile(rb'"offset": (\d+), "length": (\d+)')
_FILES = re.compile(rb'"files": \{([^}]*)\}')
_FILE_ENTRY = re.compile(rb'"(data-(\d+)-of-\d+\.bin)": (\d+)')


def _download_meta(dataset, path):
  token = os.environ.get("HF_TOKEN")
  if not token:
    raise SystemExit("Set HF_TOKEN to download meta.json files.")
  url = hf_hub_url(REPO, f"{dataset}/meta.json", repo_type="dataset")
  tmp = path.with_suffix(".part")
  with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=180) as r:
    r.raise_for_status()
    with open(tmp, "wb") as f:
      for chunk in r.iter_content(1 << 22):
        f.write(chunk)
  os.replace(tmp, path)


def dataset_meta(dataset):
  """(file_start {bin name: global float start}, seq_offsets, seq_lengths) for e.g. 'nature/cmip6_1880'."""
  META_CACHE.mkdir(parents=True, exist_ok=True)
  cache = META_CACHE / (dataset.replace("/", "__") + ".npz")
  if cache.exists():
    z = np.load(cache, allow_pickle=False)
    return json.loads(str(z["files"])), z["offsets"], z["lengths"]
  raw_path = META_CACHE / (dataset.replace("/", "__") + ".json")
  if not raw_path.exists():
    _download_meta(dataset, raw_path)
  offsets, lengths, files, buf = [], [], None, b""
  with open(raw_path, "rb") as f:
    while True:
      chunk = f.read(1 << 25)
      data = buf + chunk
      if files is None:
        m = _FILES.search(data)
        if m:
          entries = sorted(((int(n), name.decode(), int(c)) for name, n, c in _FILE_ENTRY.findall(m.group(1))))
          files, start = {}, 0
          for _, name, count in entries:
            files[name] = start
            start += count
      last = 0
      pairs = []
      for m in _PAIR.finditer(data):
        pairs.append((int(m.group(1)), int(m.group(2))))
        last = m.end()
      if pairs:
        a = np.array(pairs, dtype=np.int64)
        offsets.append(a[:, 0])
        lengths.append(a[:, 1])
      buf = data[last:] if chunk else b""
      if not chunk:
        break
  offsets, lengths = np.concatenate(offsets), np.concatenate(lengths)
  assert files, f"no file table found in {raw_path}"
  np.savez(cache, files=json.dumps(files), offsets=offsets, lengths=lengths)
  raw_path.unlink()                                  # keep only the compact arrays
  return files, offsets, lengths


def window_segments(dataset, bin_name, float_offset, window=131072):
  """[(start_in_window, length)] of every series piece inside the window starting at float_offset of bin_name."""
  files, offsets, lengths = dataset_meta(dataset)
  g0 = files[bin_name] + float_offset
  g1 = g0 + window
  i = max(int(np.searchsorted(offsets, g0, side="right")) - 1, 0)
  segs = []
  while i < len(offsets) and offsets[i] < g1:
    lo, hi = max(offsets[i], g0), min(offsets[i] + lengths[i], g1)
    if hi > lo:
      segs.append((int(lo - g0), int(hi - lo)))
    i += 1
  return segs


def build_segment_index(subset_dir, force=False):
  """Cache of series segments for every window in a subset folder -> subset_dir/segments.npz.

  Arrays: win (index into index.json order), start (offset within window), length.
  """
  subset_dir = Path(subset_dir)
  out = subset_dir / "segments.npz"
  if out.exists() and not force:
    z = np.load(out)
    return z["win"], z["start"], z["length"]
  idx = json.load(open(subset_dir / "index.json"))
  win, start, length = [], [], []
  for n, e in enumerate(idx):
    dataset, bin_name = os.path.dirname(e["source"]), os.path.basename(e["source"])
    for s, l in window_segments(dataset, bin_name, e["float_offset"], e["length"]):
      win.append(n); start.append(s); length.append(l)
    if (n + 1) % 2000 == 0:
      print(f"segments: {n + 1}/{len(idx)} windows", flush=True)
  win, start, length = (np.array(a, dtype=np.int64) for a in (win, start, length))
  np.savez(out, win=win, start=start, length=length)
  return win, start, length


def main():
  """Rebuild the segment index of the given subset folders (default: both data/Time-300B-subset*)."""
  import argparse
  p = argparse.ArgumentParser(description="Rebuild the series-boundary index (segments.npz) of Time-300B subset folders.")
  p.add_argument("folders", nargs="*", type=Path, default=[DATA_DIR / "Time-300B-subset", DATA_DIR / "Time-300B-subset2"])
  for d in p.parse_args().folders:
    w, s, l = build_segment_index(d, force=True)
    print(d, f"{len(w)} segments in {len(set(w.tolist()))} windows; median length {int(np.median(l))}; "
             f"{(l >= 2112).mean():.1%} of segments >= 2112 points")


if __name__ == "__main__":
  main()
