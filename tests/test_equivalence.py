"""The optimised implementations (compiled scans, vectorised collation, sorted MoE routing, GPU windows) must give the
same results as straightforward reference implementations."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import timesqueeze.cutrule.bilevel as bilevel
from timesqueeze.cutrule.model import CutRule
from timesqueeze.data.orcl import load_orcl
from timesqueeze.model.moe import MoEConfig, SparseMoE
from timesqueeze.patching.batching import collate_patches, patch_ids
from timesqueeze.patching.segmenter import CutRuleSegmenter
from timesqueeze.patching.volatility import ewma_vol_timeaware

# ---------------- plain numpy reference of the cut scan (the original Python implementation) ----------------

def reference_scan(r, slow, fast, W1, b1, W2, b2, c_sigma, c_n, max_len, forced=None, u=None, temp=0.5):
  r = np.asarray(r, dtype=np.float64)
  T = len(r)
  r2_cum = np.concatenate([[0.0], np.cumsum(r * r)])
  starts, ends, cuts, s = [], [], np.zeros(T, dtype=np.float32), 0
  for t in range(T):
    x = np.array([np.sqrt(max(r2_cum[t + 1] - r2_cum[s], 0.0)) / c_sigma, (t - s + 1) / c_n,
                  slow[t], fast[t]], dtype=np.float32)
    logit = np.tanh(x @ W1 + b1) @ W2 + b2
    capped = t - s + 1 >= max_len
    if u is None:
      cut = logit >= 0
    else:
      cut = u[t] < 1 / (1 + np.exp(-np.clip(float(logit) / temp, -50, 50)))
      cuts[t] = -1.0 if capped else float(cut)
    if cut or capped or (forced is not None and forced[t]):
      starts.append(s); ends.append(t); s = t + 1
  return np.array(starts), np.array(ends), cuts


def trained_segmenter(time_aware):
  d = load_orcl()
  r, dtu = d.returns[:11400], d.dtu[:11400]
  torch.manual_seed(0)
  model, _ = bilevel.train(True, r, epochs=4, verbose=False, dt=dtu if time_aware else None)
  return CutRuleSegmenter(model), r, dtu


@pytest.mark.parametrize("time_aware", [True, False])
def test_compiled_cut_scan_matches_the_python_reference(time_aware, artifacts):
  seg, r, dtu = trained_segmenter(time_aware)
  dtu = dtu if time_aware else None
  w = slice(2000, 2600)
  slow, fast = seg.ewma_features(r[w], None, None if dtu is None else dtu[w])
  _, ends_ref, _ = reference_scan(r[w], slow, fast, seg.W1, seg.b1, seg.W2, seg.b2, seg.c_sigma, seg.c_n, seg.max_len)
  expected = np.concatenate([[0], ends_ref[ends_ref + 1 < 600] + 1])
  got = seg.boundaries(r[w], slow, fast, None if dtu is None else dtu[w])
  artifacts.json("deterministic.json", {"time_aware": time_aware, "patches": len(got), "identical": got.tolist() == expected.tolist()})
  assert got.tolist() == expected.tolist()
  assert np.diff(np.concatenate([got, [600]])).max() <= seg.max_len


@pytest.mark.parametrize("time_aware", [True, False])
def test_compiled_sampled_scan_matches_the_python_reference(time_aware, artifacts):
  seg, r, dtu = trained_segmenter(time_aware)
  dtu = dtu if time_aware else None
  w = slice(3000, 3500)
  slow, fast = seg.ewma_features(r[w], None, None if dtu is None else dtu[w])
  starts, feats, cuts = seg.sample_boundaries(r[w], slow, fast, None if dtu is None else dtu[w], np.random.default_rng(5))
  u = np.random.default_rng(5).random(500)                           # the same uniforms the scan consumed
  _, ends_ref, cuts_ref = reference_scan(r[w], slow, fast, seg.W1, seg.b1, seg.W2, seg.b2, seg.c_sigma, seg.c_n,
                                         seg.max_len, u=u, temp=seg.temp)
  artifacts.json("sampled.json", {"cuts": int((cuts > 0).sum()), "capped": int((cuts < 0).sum()),
                                  "reference_cuts": int((cuts_ref > 0).sum()),
                                  "identical": bool(np.array_equal(cuts, cuts_ref))})
  assert np.array_equal(cuts, cuts_ref)
  assert starts[0] == 0 and np.isfinite(feats).all()
  assert np.diff(np.concatenate([starts, [500]])).max() <= seg.max_len


def test_timeaware_ewma_kernel_matches_a_python_loop(artifacts):
  rng = np.random.default_rng(0)
  r = rng.standard_normal(400) * 1e-4
  dtu = rng.exponential(1.0, 400)
  dtu[::7] = 0.0
  keep = 1 - 2 / 51
  S = W = 0.0
  ref = []
  for v, d in zip(r ** 2, keep ** dtu):
    S = d * S + v
    W = d * W + 1
    ref.append(np.sqrt(S / W))
  got = ewma_vol_timeaware(r, 50, dtu)
  artifacts.json("ewma.json", {"max_abs_difference": float(np.abs(got - np.array(ref)).max())})
  assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_vectorised_collate_matches_a_per_sample_loop(artifacts):
  rng = np.random.default_rng(1)
  T = 50
  bounds = [np.unique(np.concatenate([[0], rng.integers(1, T, rng.integers(0, 15))])) for _ in range(9)]
  pos, mask, ids = collate_patches(bounds, T)
  P = max(len(b) for b in bounds)
  ref_pos = np.zeros((len(bounds), P), dtype=np.int64)
  ref_mask = np.zeros((len(bounds), P), dtype=bool)
  for i, b in enumerate(bounds):
    ref_pos[i, :len(b)], ref_mask[i, :len(b)] = b, True
  assert np.array_equal(pos, ref_pos) and np.array_equal(mask, ref_mask)
  assert np.array_equal(ids, np.stack([patch_ids(b, T) for b in bounds]))
  artifacts.json("collate.json", {"batch": len(bounds), "max_patches": P, "identical": True})


def test_sorted_moe_routing_matches_the_per_expert_loop(artifacts, device):
  """The MoE layer now sorts tokens by expert (one host sync); compare with the original loop over experts."""
  torch.manual_seed(0)
  cfg = MoEConfig(hidden_size=32, intermediate_size=64, num_experts=6, top_k=2)
  layer = SparseMoE(cfg).to(device)
  x = torch.randn(3, 41, 32, device=device)
  out, logits = layer(x)
  flat = x.reshape(-1, 32)
  weights, selected = torch.topk(F.softmax(logits, dim=1, dtype=torch.float), cfg.top_k, dim=-1)
  ref = torch.zeros_like(flat)
  for e, expert in enumerate(layer.experts):
    token_idx, slot = torch.where(selected == e)
    if token_idx.numel():
      ref.index_add_(0, token_idx, expert(flat[token_idx]) * weights[token_idx, slot, None])
  ref = ref + torch.sigmoid(layer.shared_expert_gate(flat)) * layer.shared_expert(flat)
  diff = (out.reshape(-1, 32) - ref).abs().max().item()
  artifacts.json("moe_routing.json", {"max_abs_difference": diff, "tokens_per_expert": torch.bincount(selected.reshape(-1), minlength=6)})
  assert diff < 1e-5


def test_gpu_windows_match_numpy_windows(artifacts):
  """segment_data_helper gathers windows on the device; check against a straightforward numpy construction."""
  d = load_orcl()
  n = int(d.day_starts[2])
  torch.manual_seed(0)
  cut = CutRule(d.returns[:n], day_starts=d.day_starts[:2], dt=d.dtu[:n])
  starts, ends, x, r2, mask, in_hard, dt = cut.segment_data_helper()
  T = n
  H = x.shape[1]
  assert H == cut.max_len                                          # a soft bar ends at the length cap, like a hard one
  idx = starts[:, None] + np.arange(H)
  forced_at = np.nonzero(cut.forced_cut)[0]
  limit = np.array([forced_at[forced_at >= s][0] if (forced_at >= s).any() else T - 1 for s in starts])
  valid = idx <= limit[:, None]                                    # a window stops at the end of its day
  idc = np.minimum(idx, T - 1)
  ref_r2 = d.returns[:n][idc] ** 2 * valid
  ref_dt = valid.astype(np.float64)                                # durations are tick counts
  ref_x = np.stack([np.sqrt(np.cumsum(ref_r2, 1)) / cut.c_sigma, np.cumsum(ref_dt, 1) / cut.c_n,
                    cut.EWMA_slow[idc], cut.EWMA_fast[idc]], -1)
  err = float(np.abs(x.cpu().numpy() - ref_x).max())
  artifacts.json("windows.json", {"bars": len(starts), "window_length": H, "max_abs_difference": err})
  assert np.allclose(x.cpu().numpy(), ref_x, rtol=1e-4, atol=1e-6)
  assert np.array_equal(mask.cpu().numpy(), valid) and np.allclose(dt.cpu().numpy(), ref_dt, rtol=1e-5)
  assert np.array_equal(in_hard.cpu().numpy().astype(bool), valid & (np.arange(H) < (ends - starts + 1)[:, None]))


def test_bilevel_training_recovers_from_a_collapsed_update(monkeypatch, artifacts):
  """If an update leaves the CutRule never cutting, the trainer rolls back, halves the learning rate and carries on."""
  d = load_orcl()
  r = d.returns[:11400]
  real_epoch = bilevel.epoch
  calls = {"n": 0}

  def flaky_epoch(model, opt, **kw):
    calls["n"] += 1
    return None if calls["n"] == 3 else real_epoch(model, opt, **kw)       # the third call reports a collapse

  monkeypatch.setattr(bilevel, "epoch", flaky_epoch)
  torch.manual_seed(0)
  model, trace = bilevel.train(True, r, epochs=5, verbose=False)
  artifacts.json("recovery.json", {"epochs_completed": len(trace), "epoch_calls": calls["n"], "final": trace[-1]})
  assert len(trace) == 5 and calls["n"] == 7                                  # 5 epochs + one failed call + one redone epoch
  assert len(model.segment_data()[0]) > 0


def test_dataset_index_matches_the_slow_per_piece_loop(artifacts):
  """Time300BSegments is vectorised now; it must select exactly the same series pieces as the original loop that
  checked every piece's file with Path.exists()."""
  import json
  import zlib

  from timesqueeze.data.time300b_segments import build_segment_index
  from timesqueeze.data.time300b_windows import MIN_SEGMENT, VAL_BIN_MOD, Time300BSegments
  from timesqueeze.utils.paths import DATA_DIR
  name = "Time-300B-subset"
  folder = DATA_DIR / name
  idx = json.load(open(folder / "index.json"))
  win, start, length = build_segment_index(folder)
  seg = Time300BSegments(subsets=(name,))
  n_check = 40_000
  files, starts, lengths, val = [], [], [], []
  for w, s, l in list(zip(win, start, length))[:n_check]:                # the original per-piece implementation
    if l >= MIN_SEGMENT and (folder / idx[w]["file"]).exists():
      files.append(str(folder / idx[w]["file"])); starts.append(s); lengths.append(l)
      val.append(zlib.crc32(idx[w]["source"].encode()) % VAL_BIN_MOD == 0)
  n = len(files)
  assert list(seg.files[:n]) == files and list(seg.starts[:n]) == starts and list(seg.lengths[:n]) == lengths
  assert list(seg.is_val[:n]) == val
  artifacts.json("index.json", {"pieces_checked": n_check, "pieces_kept": n, "total_kept": len(seg.files), "identical": True})
