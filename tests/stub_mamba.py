import torch
from torch import nn


class StubMamba(nn.Module):
  """Causal stand-in for mamba_ssm.Mamba, for testing on machines without it (e.g. native Windows). Test-only."""

  def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
    super().__init__()
    self.conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
    self.proj = nn.Linear(d_model, d_model)
    self.gate = nn.Linear(d_model, d_model)

  def forward(self, x):                                   # [B, L, D], causal in L
    c = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
    return self.proj(torch.nn.functional.silu(c)) * torch.sigmoid(self.gate(x))
