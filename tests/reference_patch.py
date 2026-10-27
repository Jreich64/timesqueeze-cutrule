import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# Reference (slow, per-timestep Python loop) patcher from the original notebook; used only to test patching.py.
class RelativeDeviationPatch(nn.Module):
  def __init__(self, L_choice=32, tau=0.3, max_patch_length=8, eps=1e-12):
    super().__init__()
    self.L_choice = L_choice
    self.tau = tau
    self.max_patch_length = max_patch_length
    self.eps = eps

  def calc_boundaries(self, x):
    B, T = x.shape
    all_boundaries = []
    for b in range(B):
      boundaries = [0]
      last_start = 0
      for t in range(1, T):
        start = max(t - self.L_choice, 0)
        history = x[b, start:t]
        power = history.square().mean()
        deviation = torch.abs(x[b, t] - x[b, t-1])
        threshold = self.tau*torch.sqrt(power + self.eps)
        cut = bool((deviation > threshold).item())
        max_length_reached = ((t - last_start) >= self.max_patch_length)
        if cut or max_length_reached:
          boundaries.append(t)
          last_start = t
      all_boundaries.append(torch.tensor(boundaries, dtype=torch.long, device=x.device))
    return all_boundaries

  def forward(self, embeddings, x):
    B, T, D = embeddings.shape
    boundaries = self.calc_boundaries(x.squeeze(-1))
    lengths = torch.tensor([len(b) for b in boundaries], device=x.device)
    max_patch_size = lengths.max().item()
    padded = torch.stack([F.pad(b, (0, max_patch_size-len(b)), value=b[-1].item()) for b in boundaries])
    batch_index = torch.arange(B, device=x.device)[:, None]
    tokens = embeddings[batch_index, padded]
    padding_mask = (torch.arange(max_patch_size, device=x.device)[None, :]  >= lengths[:, None])
    return_dict = {
        'Tokens': tokens,
        "Pos_IDS": padded,
        "Boundaries": boundaries,
        "Padding_Mask": padding_mask
    }
    return return_dict

  def unpatch(self, patch_embeddings, boundaries, output_len):
    restored_patches = []
    for i, bound in enumerate(boundaries):
      t = torch.arange(output_len, device=patch_embeddings.device)
      index = torch.searchsorted(bound, t, right=True) - 1
      restored_patches.append(patch_embeddings[i, index, :])
    return torch.stack(restored_patches)


def EWMAVol(returns, span):
  """Original notebook EWMA (a Python loop); patching.volatility.ewma_vol is the vectorised equivalent used by the code."""
  lambda_ = 2.0 / (span + 1)
  returns_squared = np.asarray(returns)**2
  EWMA = np.empty(len(returns))
  EWMA[0] = returns_squared[0]
  for i in range(1, len(returns)):
    EWMA[i] = lambda_*returns_squared[i] + (1-lambda_)*EWMA[i-1]
  return np.sqrt(EWMA)
