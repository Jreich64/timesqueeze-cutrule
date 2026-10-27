import numpy as np
import torch
from reference_patch import EWMAVol, RelativeDeviationPatch

from timesqueeze.cutrule.bilevel import train
from timesqueeze.data.orcl import load_orcl, load_trades, trade_returns, trade_returns_by_day
from timesqueeze.patching.batching import collate_patches
from timesqueeze.patching.power_rule import elapsed_time, power_rule_boundaries, time_feature
from timesqueeze.patching.segmenter import CutRuleSegmenter
from timesqueeze.patching.volatility import ewma_vol, ewma_vol_timeaware


def reference_bounds(x):
  patcher = RelativeDeviationPatch(L_choice=32, tau=0.3, max_patch_length=8)
  return patcher.calc_boundaries(torch.tensor(x, dtype=torch.float64)[None])[0].numpy()


def zscore(w):
  return (w - w.mean()) / (w.std() + 1e-8)


def test_power_rule_matches_reference_random(artifacts):
  rng = np.random.default_rng(0)
  counts, example = [], None
  for scale in (1e-3, 1.0, 50.0):
    for _ in range(5):
      x = (rng.standard_normal(600).cumsum() * 0.2 + rng.standard_normal(600)) * scale
      fast, ref = power_rule_boundaries(x), reference_bounds(x)
      assert np.array_equal(fast, ref)
      counts.append(len(fast))
      example = example if example is not None else (zscore(x)[:200], fast[fast < 200], ref[ref < 200])
  artifacts.json("comparison.json", {"windows_compared": len(counts), "all_identical": True, "patches_per_window": counts})
  artifacts.patches("boundaries_fast_vs_reference.png", example[0], {"fast": example[1], "notebook reference": example[2]},
                    "Power-rule patch starts: vectorised code vs the notebook's loop (identical)")


def test_power_rule_matches_reference_real_windows(artifacts):
  r = trade_returns(load_trades())
  rows = []
  for s in (0, 3000, 8000):
    w = zscore(r[s:s + 512])
    fast, ref = power_rule_boundaries(w), reference_bounds(w)
    assert np.array_equal(fast, ref)
    rows.append({"start": s, "patches": len(fast), "compression": 512 / len(fast)})
    if s == 3000:
      artifacts.patches("orcl_window_patches.png", w[:256], {"power rule": fast[fast < 256]},
                        "Power-rule patches on a real ORCL window (first 256 ticks)")
  artifacts.json("windows.json", rows)


def test_power_rule_edge_cases(artifacts):
  out = {"single_point": power_rule_boundaries(np.zeros(1)).tolist(), "flat_series_20": power_rule_boundaries(np.zeros(20)).tolist()}
  artifacts.json("edge_cases.json", out)
  assert out["single_point"] == [0]
  assert out["flat_series_20"] == [0, 8, 16]                               # flat series: only the max-length cap


def test_ewma_matches_original(artifacts):
  r = trade_returns(load_trades())[:2000]
  curves, diffs = {}, {}
  for span in (10, 50):
    fast, orig = ewma_vol(r, span), EWMAVol(r, span)
    diffs[span] = float(np.abs(fast - orig).max())
    assert np.allclose(fast, orig)
    curves[f"span {span}"] = (np.arange(400), fast[:400])
  artifacts.json("max_difference.json", diffs)
  artifacts.lines("ewma_volatility.png", curves, "EWMA volatility (vectorised = original loop)", "tick", "volatility")


def test_cutrule_segmenter_matches_model(artifacts):
  r = trade_returns(load_trades())[:11400]
  torch.manual_seed(0)
  model, trace = train(True, r, epochs=15, verbose=False)
  starts, ends = model.segment_data()
  expected = [0] + [int(e) + 1 for e in ends if e + 1 < len(r)]
  got = CutRuleSegmenter(model).boundaries(r)
  artifacts.json("segmentation.json", {"bars": len(expected), "identical": got.tolist() == expected, "final_epoch": trace[-1]})
  artifacts.patches("cutrule_patches.png", zscore(r[:600]), {"CutRule (model)": np.array(expected)[np.array(expected) < 600],
                                                              "CutRule (segmenter)": got[got < 600]},
                    "CutRule patch starts, first 600 ticks: training code vs fast segmenter (identical)")
  assert got.tolist() == expected


def test_collate_and_patch_ids(artifacts):
  b = [np.array([0, 3, 5]), np.array([0, 4])]
  pos, mask, ids = collate_patches(b, 8)
  artifacts.json("collate.json", {"positions": pos, "mask": mask, "patch_id": ids})
  assert pos.tolist() == [[0, 3, 5], [0, 4, 0]]
  assert mask.tolist() == [[True] * 3, [True, True, False]]
  assert ids[0].tolist() == [0, 0, 0, 1, 1, 2, 2, 2] and ids[1].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_returns_are_per_day_and_split_by_days(artifacts):
  d = load_orcl()
  edges = list(d.day_starts) + [len(d.returns)]
  sizes = np.diff(edges)
  split = ["train" if e < d.train_end else "validation" if e < d.val_end else "test" for e in edges[:-1]]
  artifacts.json("split.json", {"days": len(d.day_starts), "train_end": d.train_end, "val_end": d.val_end,
                                "returns": len(d.returns), "max_abs_return": float(np.abs(d.returns).max()),
                                "days_per_split": {s: split.count(s) for s in ("train", "validation", "test")}})
  artifacts.bars("trades_per_day.png", list(range(1, len(sizes) + 1)),
                 {s: [int(z) if sp == s else 0 for z, sp in zip(sizes, split)] for s in ("train", "validation", "test")},
                 "Returns per trading day, by split", "returns")
  assert len(d.day_starts) == 43 and 0 < d.train_end < d.val_end < len(d.returns)
  assert d.train_end in d.day_starts and d.val_end in d.day_starts      # splits fall on day boundaries
  assert np.isfinite(d.returns).all() and np.abs(d.returns).max() < 0.1  # no overnight-gap returns


def session_trades():
  """Reference: the regular-session trades (09:30-16:00 New York time), their timestamps, prices and trading dates."""
  import pandas as pd
  trades = load_trades()
  ts = trades["ts_event"].astype("int64").values
  et = pd.to_datetime(trades["ts_event"], utc=True).dt.tz_convert("America/New_York")
  hhmm = (et.dt.hour * 100 + et.dt.minute).values
  keep = (hhmm >= 930) & (hhmm < 1600)
  return ts[keep], trades["price"].values.astype(np.float64)[keep], et.dt.strftime("%Y%m%d").values[keep], keep


def test_only_regular_session_trades_are_used(artifacts):
  ts, price, day, keep = session_trades()
  d = load_orcl()
  n_days = len(np.unique(day))
  artifacts.json("session.json", {"trades_kept": int(keep.sum()), "trades_dropped": int((~keep).sum()),
                                  "days": n_days, "returns": len(d.returns), "largest_gap_seconds": float(d.dt.max())})
  assert len(d.returns) == len(ts) - n_days                             # one return per kept trade except each day's first
  assert d.dt.max() < 6.5 * 3600                                        # no gap can reach outside one session


def test_tick_gaps_are_the_raw_ts_event_deltas(artifacts):
  d = load_orcl()
  ts, _, day, _ = session_trades()
  raw = (np.diff(ts) / 1e9)[day[1:] == day[:-1]]
  stats = {"ticks": len(raw), "exact_zero_share": float((raw == 0).mean()), "under_1ms_share": float((raw < 1e-3).mean()),
           "median_seconds": float(np.median(raw)), "mean_seconds": float(raw.mean()), "max_seconds": float(raw.max())}
  artifacts.json("gap_statistics.json", stats)
  artifacts.hist("gap_histogram.png", raw[raw > 0], "ORCL tick gaps from ts_event (zero gaps excluded)", "seconds between trades",
                 logx=True, annotate=f"{stats['exact_zero_share']:.1%} of gaps are exactly 0")
  assert np.array_equal(d.dt, raw)                                      # untouched: no cap, no clipping, no resampling
  assert d.dt.max() > 60 and (d.dt == 0).mean() > 0.2                   # long quiet gaps and exact zero gaps survive
  assert np.allclose(d.dtu * d.time_unit, d.dt)                         # dtu is only a constant unit change


def test_returns_are_log_price_changes(artifacts):
  d = load_orcl()
  _, price, day, _ = session_trades()
  ref = np.log(price[1:] / price[:-1])[day[1:] == day[:-1]]
  first_day = slice(int(d.day_starts[0]), int(d.day_starts[1]))
  p = price[day == day[0]]                                              # within a day they add up to the log price path
  artifacts.json("log_returns.json", {"max_abs_difference": float(np.abs(d.returns - ref).max())})
  assert np.allclose(d.returns, ref, rtol=1e-12, atol=1e-15)
  assert np.isclose(d.returns[first_day].sum(), np.log(p[-1] / p[0]))


def test_no_window_crosses_a_day_boundary(artifacts):
  from timesqueeze.data.orcl_windows import OrclWindows
  d = load_orcl()
  day_of = np.searchsorted(d.day_starts, np.arange(len(d.returns)), side="right")
  counts = {}
  for name, (lo, hi) in {"train": (0, d.train_end), "validation": (d.train_end, d.val_end),
                         "test": (d.val_end, len(d.returns))}.items():
    ds = OrclWindows(d, lo, hi, 512, 64, "power")
    assert len(ds) > 0
    counts[name] = len(ds)
    for s in ds.starts[::25]:
      end = s + 512 + 64 - 1                                            # last forecast target
      assert lo <= s and end < hi and day_of[s] == day_of[end]
  artifacts.json("windows_per_split.json", counts)
  artifacts.bars("windows_per_split.png", list(counts), {"windows": list(counts.values())},
                 "Windows per split (all inside one trading day)", "windows")


def test_time_aware_power_rule_equals_paper_rule_for_regular_ticks(artifacts):
  rng = np.random.default_rng(3)
  x = rng.standard_normal(500).cumsum() * 0.3 + rng.standard_normal(500)
  t = elapsed_time(np.ones(500))
  a, b = power_rule_boundaries(x, t=t), power_rule_boundaries(x)
  artifacts.json("equality.json", {"identical": bool(np.array_equal(a, b)), "patches": len(a)})
  artifacts.patches("boundaries.png", zscore(x)[:200], {"paper rule": b[b < 200], "time-aware, regular ticks": a[a < 200]},
                    "Time-aware power rule with regular ticks = the paper's rule")
  assert np.array_equal(a, b)


def test_time_aware_power_rule_responds_to_gaps(artifacts):
  x = np.zeros(40)
  t = np.arange(40, dtype=float)
  t[20:] += 100.0                                   # a long pause before tick 20
  b = power_rule_boundaries(x, t=t)
  tick_based = power_rule_boundaries(x)
  t2 = np.concatenate([np.full(30, 0.0), 1.0 + np.arange(10)])   # 30 ticks with an identical timestamp
  b2 = power_rule_boundaries(np.zeros(40), t=t2)
  artifacts.json("gaps.json", {"with_pause": b, "tick_based": tick_based, "burst_of_identical_timestamps": b2})
  artifacts.patches("pause.png", np.where(np.arange(40) < 20, 0.0, 1.0), {"tick-based": tick_based, "time-aware (pause at 20)": b},
                    "A long pause closes a patch under the time-aware rule")
  assert 20 in b.tolist()                           # a patch closes at the pause even though only 4 ticks passed
  assert tick_based.tolist() == [0, 8, 16, 24, 32]
  assert np.diff(b2[:4]).tolist() == [8, 8, 8]       # tick cap still bounds patches


def test_time_feature_is_unclipped_and_zero_for_equal_timestamps(artifacts):
  gaps = np.array([0.0, 1e-6, 1.0, 1e3])
  f = time_feature(gaps)
  artifacts.json("time_feature.json", {"gap": gaps, "feature": f})
  assert f[0] == 0.0 and np.all(np.diff(f) > 0) and f[3] == np.float32(np.log1p(1e3))


def test_timeaware_ewma_counts_simultaneous_ticks_and_forgets_after_gaps(artifacts):
  r = np.array([0.01, 0.01, 0.0, 0.0, 0.0])
  burst = ewma_vol_timeaware(r, 10, np.array([1, 0, 0, 0, 0.0]))      # all five ticks share one instant
  gap = ewma_vol_timeaware(r, 10, np.array([1, 1, 1e4, 1, 1.0]))      # long pause before tick 2
  tick = ewma_vol(r, 10)
  artifacts.json("ewma.json", {"burst": burst, "gap": gap, "tick_based": tick})
  artifacts.lines("ewma_variants.png", {"tick-based": (np.arange(5), tick), "burst (one instant)": (np.arange(5), burst),
                                         "long pause before tick 2": (np.arange(5), gap)},
                  "Time-aware volatility estimate", "tick", "volatility", markers=True)
  assert np.allclose(burst[-1], np.sqrt((0.01**2 * 2) / 5))           # all five count in full
  assert gap[2] < 1e-3 * gap[1]                                       # old volatility is forgotten


def test_cutrule_bars_never_span_days(artifacts):
  returns, day_starts, dt = trade_returns_by_day(load_trades())
  n = day_starts[3]                                                  # first three days
  r, ds, dtu = returns[:n], day_starts[:3], dt[:n] / dt.mean()
  torch.manual_seed(0)
  model, _ = train(True, r, epochs=10, verbose=False, day_starts=ds, dt=dtu)
  starts, ends = model.segment_data()
  day_of = np.searchsorted(ds, np.arange(n), side="right")
  seg = CutRuleSegmenter(model)
  slow, fast = seg.ewma_features(r, ds, dtu)
  expected = []
  for a, b in zip(list(ds), list(ds[1:]) + [n]):
    expected += (a + seg.boundaries(r[a:b], slow[a:b], fast[a:b], dtu[a:b])).tolist()
  bars_per_day = np.bincount(day_of[starts] - 1, minlength=3)
  artifacts.json("bars.json", {"bars": len(starts), "bars_per_day": bars_per_day})
  artifacts.bars("bars_per_day.png", [1, 2, 3], {"bars": bars_per_day.tolist()}, "CutRule bars per trading day", "bars")
  assert (day_of[starts] == day_of[ends]).all()                      # every bar lies inside one day
  assert [0] + [int(e) + 1 for e in ends if e + 1 < n] == expected
