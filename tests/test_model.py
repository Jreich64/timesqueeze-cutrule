import numpy as np
import pytest
import torch

from timesqueeze.model.forecaster import HORIZONS, TimeSqueeze, forecasting_loss, per_sample_loss
from timesqueeze.model.moe import MoEConfig, MoETransformer
from timesqueeze.patching.batching import collate_patches
from timesqueeze.patching.power_rule import elapsed_time, power_rule_boundaries, time_feature


def small_model(mamba_cls, device):
  model = TimeSqueeze(d_model=48, d_state=16, moe_config=MoEConfig(hidden_size=48, intermediate_size=96, num_layers=2,
                                                                    num_heads=4, num_experts=4), mamba_cls=mamba_cls)
  for head in model.heads.values():         # the heads start at zero ("no change"); random here so outputs can differ
    torch.nn.init.normal_(head.weight, 0, 0.1)
  return model.to(device)


def test_untrained_model_forecasts_no_change(mamba_cls, device):
  model = TimeSqueeze(d_model=48, d_state=16, moe_config=MoEConfig(hidden_size=48, intermediate_size=96, num_layers=2,
                                                                    num_heads=4, num_experts=4), mamba_cls=mamba_cls).to(device)
  with torch.no_grad():
    preds, _ = model(*make_batch(np.random.default_rng(0).standard_normal((2, 50)), device))
  assert all(float(p.abs().max()) == 0.0 for p in preds.values())


def test_step_targets_and_level_reconstruction():
  from timesqueeze.model.forecaster import levels_from_steps, step_targets
  rng = np.random.default_rng(3)
  T, H = 30, max(HORIZONS)
  y = torch.tensor(np.cumsum(rng.standard_normal((2, T + H)), 1), dtype=torch.float32)
  d, s = step_targets(y, torch.full((2,), T + H), T)
  ctx_steps = np.diff(y[:, :T].numpy(), axis=1)
  assert np.allclose(s.numpy(), np.sqrt((ctx_steps ** 2).mean(1)), rtol=1e-5)       # RMS of the context's changes only
  assert torch.allclose(levels_from_steps(d[:, T:], y[:, T - 1], s), y[:, T:], atol=1e-4)   # exact reconstruction
  y2 = y.clone()
  y2[:, T:] += 100.0                                                    # the future never enters the scale
  assert torch.equal(step_targets(y2, torch.full((2,), T + H), T)[1], s)


def make_batch(x, device):
  bounds = [power_rule_boundaries(row) for row in x]
  pos, mask, ids = collate_patches(bounds, x.shape[1])
  return tuple(t.to(device) for t in (torch.tensor(x, dtype=torch.float32), torch.tensor(pos), torch.tensor(mask),
                                      torch.tensor(ids)))


def test_paper_parameter_counts(artifacts):
  cfg = MoEConfig()
  model = MoETransformer(cfg)
  total = sum(p.numel() for p in model.parameters())
  per_expert = 3 * cfg.hidden_size * (cfg.intermediate_size // cfg.top_k)
  active = total - cfg.num_layers * (cfg.num_experts - cfg.top_k) * per_expert
  artifacts.json("parameter_counts.json", {"total": total, "active": active, "paper_total": 113e6, "paper_active": 50e6})
  artifacts.bars("parameters.png", ["total", "active"], {"this model": [total / 1e6, active / 1e6],
                                                          "paper": [113, 50]}, "MoE backbone parameters (millions)", "M")
  assert 112e6 < total < 114e6, total          # paper: 113M total
  assert 49e6 < active < 51e6                  # paper: 50M active


def test_model_is_causal(artifacts, mamba_cls, device):
  torch.manual_seed(0)
  model = small_model(mamba_cls, device).eval()
  rng = np.random.default_rng(0)
  x = rng.standard_normal((2, 200))
  x2 = x.copy()
  t0 = 120
  x2[:, t0:] += rng.standard_normal((2, 80)) * 5
  with torch.no_grad():
    p1, _ = model(*make_batch(x, device))
    p2, _ = model(*make_batch(x2, device))
  diff = (p1[8] - p2[8]).abs().amax((0, 2)).cpu().numpy()
  artifacts.lines("output_change_over_time.png", {"max |change| in 8-step forecast": (np.arange(200), diff + 1e-12)},
                  "Editing the input from t=120 never changes earlier outputs", "timestep", "|output change|",
                  logy=True, vlines=[t0])
  artifacts.json("causality.json", {"edit_at": t0, "max_change_before": float(diff[:t0].max()),
                                    "max_change_after": float(diff[t0:].max())})
  for h in HORIZONS:
    assert torch.allclose(p1[h][:, :t0], p2[h][:, :t0], atol=1e-5), h
    assert not torch.allclose(p1[h][:, t0:], p2[h][:, t0:], atol=1e-3)


def test_padding_does_not_change_valid_outputs(artifacts, mamba_cls, device):
  torch.manual_seed(0)
  model = small_model(mamba_cls, device).eval()
  x = np.random.default_rng(1).standard_normal((1, 100))
  xp = np.concatenate([x, np.zeros((1, 60))], axis=1)
  with torch.no_grad():
    p1, _ = model(*make_batch(x, device))
    p2, _ = model(*make_batch(xp, device))
  d = (p1[8] - p2[8][:, :100]).abs().max().item()
  artifacts.json("padding.json", {"max_difference_on_valid_positions": d})
  assert torch.allclose(p1[8], p2[8][:, :100], atol=1e-5)


def test_loss_masks_targets_past_the_end(artifacts):
  T, H = 6, max(HORIZONS)
  preds = {h: torch.zeros(1, T, h) for h in HORIZONS}
  y = torch.zeros(1, T + H)
  y[0, 4:] = 100.0                      # garbage beyond the real data (n_valid = 4)
  loss, per_h = forecasting_loss(preds, y, torch.tensor([4]))
  loss2, _ = forecasting_loss(preds, y, torch.tensor([T + H]))
  ps = per_sample_loss(preds, y, torch.tensor([4]))
  artifacts.json("loss_masking.json", {"loss_with_garbage_masked": loss.item(), "loss_unmasked": loss2.item(),
                                       "per_sample_loss_masked": ps.tolist()})
  assert loss.item() == 0.0 and ps.item() == 0.0 and loss2.item() > 0


def test_per_sample_loss_matches_forecasting_loss_for_equal_windows(artifacts):
  torch.manual_seed(0)
  B, T = 3, 40
  preds = {h: torch.randn(B, T, h) for h in HORIZONS}
  y = torch.randn(B, T + max(HORIZONS))
  n = torch.full((B,), T + max(HORIZONS))
  total, _ = forecasting_loss(preds, y, n)
  per = per_sample_loss(preds, y, n)
  artifacts.json("loss_equivalence.json", {"forecasting_loss": total.item(), "mean_per_sample": per.mean().item()})
  assert torch.allclose(total, per.mean(), rtol=1e-5)                 # same masks, same weights


def test_train_step_runs_and_learns_on_available_device(artifacts, mamba_cls, device):
  torch.manual_seed(0)
  model = small_model(mamba_cls, device)
  opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
  rng = np.random.default_rng(0)
  T = 128
  t = np.arange(T + 64)
  y = np.stack([np.sin(t / 5 + p) for p in rng.uniform(0, 6, 8)])         # easy periodic series
  batch = make_batch(y[:, :T], device)
  yt, nv = torch.tensor(y, dtype=torch.float32, device=device), torch.full((8,), T + 64, device=device)
  losses = []
  for _ in range(60):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
      preds, aux = model(*batch)
    loss, _ = forecasting_loss(preds, yt, nv, aux)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
  with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
    preds, _ = model(*batch)
  pred64 = preds[64][:, -1].float().cpu().numpy()
  truth64 = y[:, T:T + 64]
  artifacts.lines("training_loss.png", {"loss": (np.arange(60), losses)}, "Training loss on periodic series", "step", "loss")
  artifacts.forecasts("forecast_vs_actual_h64.png", y[:, :T], truth64, pred64, 64, "64-step forecasts after 60 steps")
  artifacts.csv("forecasts_h64.csv", ["window", "step", "actual", "forecast"],
                [[i, k + 1, float(truth64[i, k]), float(pred64[i, k])] for i in range(8) for k in range(64)])
  artifacts.json("training.json", {"first_loss": losses[0], "last_loss": losses[-1]})
  assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])


def timing_inputs(x, device):
  dtu = np.random.default_rng(7).exponential(1.0, x.shape)
  dtu[:, ::5] = 0.0                                   # simultaneous ticks
  t = np.stack([elapsed_time(r) for r in dtu])
  return (torch.tensor(t, dtype=torch.float32, device=device),
          torch.tensor(np.stack([time_feature(r) for r in dtu]), device=device))


def test_regular_time_equals_no_time_and_zero_init_channel_is_inert(artifacts, mamba_cls, device):
  torch.manual_seed(0)
  model = small_model(mamba_cls, device).eval()
  x = np.random.default_rng(2).standard_normal((2, 120))
  batch = make_batch(x, device)
  regular = torch.arange(120, dtype=torch.float32, device=device).expand(2, -1)
  t, dtf = timing_inputs(x, device)
  with torch.no_grad():
    base, _ = model(*batch)
    same, _ = model(*batch, regular, None)
    inert, _ = model(*batch, regular, dtf)             # time_proj starts at zero: the extra channel changes nothing
    shifted, _ = model(*batch, t, None)                # irregular elapsed time changes the rotary positions
  res = {"regular_time_vs_none": (base[8] - same[8]).abs().max().item(),
         "zero_init_channel_effect": (base[8] - inert[8]).abs().max().item(),
         "irregular_time_effect": (base[8] - shifted[8]).abs().max().item()}
  artifacts.json("time_inputs.json", res)
  artifacts.bars("effects.png", list(res), {"max |difference|": list(res.values())}, "Effect of timing inputs at initialisation")
  assert res["regular_time_vs_none"] < 1e-5 and res["zero_init_channel_effect"] < 1e-5
  assert res["irregular_time_effect"] > 1e-6          # tiny at init (attention weights are small) but nonzero


def test_time_channel_is_used_once_trained_and_stays_causal(artifacts, mamba_cls, device):
  torch.manual_seed(0)
  model = small_model(mamba_cls, device).eval()
  torch.nn.init.normal_(model.encoder.time_proj.weight, 0, 0.5)     # as if fine-tuning had learned to use time
  rng = np.random.default_rng(4)
  x = rng.standard_normal((2, 160))
  t, dtf = timing_inputs(x, device)
  with torch.no_grad():
    a, _ = model(*make_batch(x, device), t, dtf)
    b, _ = model(*make_batch(x, device), t, torch.zeros_like(dtf))
    x2, t0 = x.copy(), 100
    x2[:, t0:] += 5.0
    t2, dtf2 = t.clone(), dtf.clone()
    t2[:, t0:] += 50.0; dtf2[:, t0:] += 1.0
    c, _ = model(*make_batch(x2, device), t2, dtf2)
  diff = (a[8] - c[8]).abs().amax((0, 2)).cpu().numpy()
  artifacts.lines("output_change_over_time.png", {"max |change|": (np.arange(160), diff + 1e-12)},
                  "Editing values and timing from t=100 never changes earlier outputs", "timestep", "|output change|",
                  logy=True, vlines=[t0])
  artifacts.json("time_channel.json", {"channel_effect": (a[8] - b[8]).abs().max().item(),
                                       "max_change_before_edit": float(diff[:t0].max())})
  assert not torch.allclose(a[8], b[8], atol=1e-4)
  for h in HORIZONS:
    assert torch.allclose(a[h][:, :t0], c[h][:, :t0], atol=1e-5), h   # future values and timing never leak backwards


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Mamba needs CUDA")
def test_real_mamba_layer_is_causal_and_trainable(artifacts, mamba_kind):
  """Only meaningful with the real kernels (WSL2/Linux): forward, backward, bf16 and causality of one Mamba layer."""
  if mamba_kind != "real":
    pytest.skip("mamba_ssm is not installed here")
  from mamba_ssm import Mamba
  torch.manual_seed(0)
  layer = Mamba(d_model=384, d_state=128, d_conv=4, expand=4).cuda()
  x = torch.randn(2, 512, 384, device="cuda", requires_grad=True)
  y = layer(x)
  y.square().mean().backward()
  with torch.no_grad():
    x2 = x.detach().clone()
    x2[:, 300:] += 1.0
    change_before = (layer(x.detach())[:, :300] - layer(x2)[:, :300]).abs().max().item()
    with torch.autocast("cuda", dtype=torch.bfloat16):
      yb = layer(x.detach())
  grads_ok = all(torch.isfinite(p.grad).all().item() for p in layer.parameters() if p.grad is not None)
  artifacts.json("mamba_layer.json", {"forward_finite": bool(torch.isfinite(y).all()), "grads_finite": grads_ok,
                                      "bf16_finite": bool(torch.isfinite(yb).all()), "max_change_before_edit": change_before})
  assert torch.isfinite(y).all() and grads_ok and torch.isfinite(yb).all() and change_before == 0.0
