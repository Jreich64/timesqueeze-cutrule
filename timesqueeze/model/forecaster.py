"""The TimeSqueeze forecaster (paper section 3) and its training loss.

Data flow for one batch of windows x [B, T] (one value per tick):
  1. encoder  - a linear embedding of each value (plus log(1 + tick gap) when timing is used) and 2 Mamba layers:
                a state per tick, [B, T, 384], that has seen every earlier tick.
  2. patch    - keep only the encoder state at the first tick of every patch (boundaries come from the patch rule):
                [B, P, 384] with P << T. This is where the sequence is "squeezed".
  3. backbone - the 12-layer Mixture-of-Experts Transformer (moe.py) over the patch tokens, causal, with rotary
                positions = patch start time (tick index, or elapsed time when timing is used).
  4. unpatch  - copy every patch token back over the ticks of its patch: [B, T, 384] again.
  5. decoder  - 2 Mamba layers over (encoder state + unpatched token), so each tick combines its own fine detail with
                the patch-level context.
  6. heads    - one linear head per horizon (1, 8, 32, 64 steps ahead), applied at *every* tick.

The heads forecast the next one-step *changes* of the input series, scaled by the context's RMS change (step_targets);
the level forecast is the last observed value plus their running sum (levels_from_steps). An untrained model therefore
starts at "no change". On ORCL the input is the log price path, so the changes are log returns.

Loss (forecasting_loss): Huber (delta 2) between each head's forecast changes and the true future changes at every
position whose targets lie inside the window, averaged over horizons, plus 0.02 x the router load-balancing loss.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .moe import MoEConfig, MoETransformer

# mamba_ssm needs Linux + CUDA; import it lazily so the rest of the package works on Windows.
try:
  from mamba_ssm import Mamba
except ImportError:
  Mamba = None

HORIZONS = (1, 8, 32, 64)
HUBER_DELTA = 2.0      # delta used by the Time-MoE reference training code


def _mamba_cls(mamba_cls):
  mamba_cls = mamba_cls or Mamba
  if mamba_cls is None:
    raise ImportError("mamba-ssm is not installed (Linux/WSL2 + CUDA only). See README.")
  return mamba_cls


class MambaStack(nn.Module):
  """Pre-norm residual stack of Mamba layers."""

  def __init__(self, d_model, d_state, d_conv, expand, num_layers, mamba_cls=None):
    super().__init__()
    cls = _mamba_cls(mamba_cls)
    self.layers = nn.ModuleList([cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                                 for _ in range(num_layers)])
    self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

  def forward(self, h):
    for layer, norm in zip(self.layers, self.norms):
      h = h + layer(norm(h))
    return h


class TimeSqueezeEncoder(nn.Module):
  def __init__(self, d_model=384, d_state=128, d_conv=4, expand=4, num_layers=2, mamba_cls=None):
    super().__init__()
    self.input_proj = nn.Linear(1, d_model)
    # Channel for the time since the previous tick (log, in tick units; 0 for evenly spaced data). Zero-initialised
    # and fed zeros during pretraining on regularly sampled Time-300B, so the pretrained function is untouched
    # until fine-tuning on timestamped ticks teaches it to use time.
    self.time_proj = nn.Linear(1, d_model, bias=False)
    nn.init.zeros_(self.time_proj.weight)
    self.stack = MambaStack(d_model, d_state, d_conv, expand, num_layers, mamba_cls)

  def forward(self, x, dtf=None):          # x: [B, T] -> [B, T, D]; dtf: [B, T] log tick interval or None
    h = self.input_proj(x.unsqueeze(-1))
    if dtf is not None:
      h = h + self.time_proj(dtf.unsqueeze(-1).to(h.dtype))
    return self.stack(h)


class TimeSqueezeDecoder(nn.Module):
  def __init__(self, d_model=384, d_state=128, d_conv=4, expand=4, num_layers=2, mamba_cls=None):
    super().__init__()
    self.fusion = nn.Linear(2 * d_model, d_model)
    self.stack = MambaStack(d_model, d_state, d_conv, expand, num_layers, mamba_cls)

  def forward(self, encoder_features, restored_features):
    return self.stack(self.fusion(torch.cat([encoder_features, restored_features], dim=-1)))


class TimeSqueeze(nn.Module):
  """Mamba encoder -> patch (boundaries given) -> MoE Transformer -> unpatch -> Mamba decoder -> horizon heads.

  Patch boundaries come from a rule (patching.py) and are supplied with the batch, so the same model works with
  the power rule or the CutRule. Every position forecasts every horizon (causal, as in Time-MoE pretraining).
  """

  def __init__(self, horizons=HORIZONS, d_model=384, d_state=128, d_conv=4, expand=4, num_mamba_layers=2,
               moe_config=None, mamba_cls=None):
    super().__init__()
    moe_config = moe_config or MoEConfig(hidden_size=d_model)
    self.horizons = tuple(horizons)
    self.encoder = TimeSqueezeEncoder(d_model, d_state, d_conv, expand, num_mamba_layers, mamba_cls)
    self.transformer = MoETransformer(moe_config)
    self.decoder = TimeSqueezeDecoder(d_model, d_state, d_conv, expand, num_mamba_layers, mamba_cls)
    self.heads = nn.ModuleDict({str(h): nn.Linear(d_model, h, bias=False) for h in self.horizons})
    for head in self.heads.values():                    # start at "no change": every forecast step is 0
      nn.init.zeros_(head.weight)

  def forward(self, x, patch_pos, patch_mask, patch_id, time=None, dtf=None):
    """x [B,T] normalized series; patch_pos [B,P] start index of each patch; patch_mask [B,P] real patches;
    patch_id [B,T] patch of each timestep. Optional timing (omit for evenly spaced series): time [B,T] elapsed time
    of each tick in tick units (used as the rotary position of each patch) and dtf [B,T] log time since the
    previous tick (encoder input channel). Returns ({h: [B,T,h]}, aux_loss)."""
    enc = self.encoder(x, dtf)                                                 # [B, T, D]
    D = enc.shape[-1]
    tokens = enc.gather(1, patch_pos.unsqueeze(-1).expand(-1, -1, D))          # fine-grained state at each boundary
    positions = patch_pos if time is None else time.gather(1, patch_pos)       # absolute (time-aware) positions
    tokens, aux = self.transformer(tokens, positions, patch_mask)              # [B, P, D]
    restored = tokens.gather(1, patch_id.unsqueeze(-1).expand(-1, -1, D))      # repeat each token over its patch
    dec = self.decoder(enc, restored)
    return {h: self.heads[str(h)](dec) for h in self.horizons}, aux


def run_model(model, batch):
  """Forward pass on a collated batch dict (keys x, patch_pos, patch_mask, patch_id and optionally time, dtf)."""
  return model(batch["x"], batch["patch_pos"], batch["patch_mask"], batch["patch_id"],
               batch.get("time"), batch.get("dtf"))


def step_targets(y, n_valid, T):
  """One-step changes of the normalised series y [B, L], scaled per window: d[:, j] = (y[:, j] - y[:, j-1]) / s with
  d[:, 0] = 0, where s [B] is the root-mean-square of the changes inside the window's context (the first min(n_valid,
  T) points; context only, so no future information in the scale). Returns (d [B, L], s [B]).

  The heads forecast these changes; the level forecast is the last observed value plus their running sum (see
  levels_from_steps). RMS rather than the standard deviation, so a context that moves by a constant step (a straight
  line) still has a non-zero scale; for zero-mean changes such as returns the two agree. On ORCL, where y is the
  z-scored log price path, d is the log return divided by the context's RMS log return."""
  y = y.float()
  d = torch.zeros_like(y)
  d[:, 1:] = y[:, 1:] - y[:, :-1]
  j = torch.arange(y.shape[1], device=y.device)
  ctx = ((j[None] >= 1) & (j[None] < torch.clamp(n_valid, max=T)[:, None].to(y.device))).float()
  s = ((d ** 2 * ctx).sum(1) / ctx.sum(1).clamp(min=1.0)).sqrt().clamp(min=1e-6)
  return d / s[:, None], s


def levels_from_steps(steps, last, scale):
  """Level forecast [B, h] from forecast steps [B, h] (scaled units), the last observed level [B] and the scale [B]."""
  return last[:, None] + scale[:, None] * torch.cumsum(steps, dim=1)


def per_sample_loss(predictions, y, n_valid, delta=HUBER_DELTA):
  """Multi-horizon Huber loss of every window separately, [B] (same masking as forecasting_loss). Used to compare
  alternative patchings of the same window."""
  B, T = next(iter(predictions.values())).shape[:2]
  t = torch.arange(T, device=y.device)
  d, _ = step_targets(y, n_valid, T)
  per_h = []
  for h, pred in predictions.items():
    tgt = d[:, 1:T + h].unfold(1, h, 1)
    idx = t[:, None] + 1 + torch.arange(h, device=y.device)[None, :]
    mask = (idx[None] < n_valid[:, None, None]).to(pred.dtype)
    err = F.huber_loss(pred.float(), tgt.float(), delta=delta, reduction="none") * mask
    per_h.append(err.sum((1, 2)) / mask.sum((1, 2)).clamp(min=1.0))
  return torch.stack(per_h).mean(0)


def forecasting_loss(predictions, y, n_valid, aux_loss=None, aux_factor=0.02, delta=HUBER_DELTA):
  """Multi-horizon Huber loss over all positions, averaged over horizons, plus alpha * router aux loss.

  y [B, T + max_h] is the normalized series (input is y[:, :T]); n_valid [B] counts real (unpadded) points.
  Position t forecasts the next h one-step changes d[t+1 : t+1+h] (step_targets: scaled by the context's RMS change);
  targets past the end of the real data are masked out.
  """
  B, T = next(iter(predictions.values())).shape[:2]
  t = torch.arange(T, device=y.device)
  d, _ = step_targets(y, n_valid, T)
  losses = {}
  for h, pred in predictions.items():
    tgt = d[:, 1:T + h].unfold(1, h, 1)                                        # [B, T, h]
    idx = t[:, None] + 1 + torch.arange(h, device=y.device)[None, :]           # absolute index of each target
    mask = (idx[None] < n_valid[:, None, None]).to(pred.dtype)
    err = F.huber_loss(pred.float(), tgt.float(), delta=delta, reduction="none") * mask
    losses[h] = err.sum() / mask.sum().clamp(min=1.0)
  loss = torch.stack(list(losses.values())).mean()
  if aux_loss is not None:
    loss = loss + aux_factor * aux_loss
  return loss, losses
