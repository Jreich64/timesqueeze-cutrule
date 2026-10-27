"""Decoder-only Mixture-of-Experts Transformer backbone (Time-MoE style), as used by TimeSqueeze.

Ported from the Time-MoE reference implementation (github.com/Time-MoE/Time-MoE, modeling_time_moe.py):
RMSNorm, rotary embeddings, causal attention, and a sparse layer with `num_experts` routed experts (top-k) plus a
sigmoid-gated shared expert. Differences: the input embedding is dropped (TimeSqueeze feeds in patch tokens from
its Mamba encoder), rotary positions are passed in explicitly (original timestep of each patch start), and
attention uses torch's fused scaled_dot_product_attention.

Defaults are the paper's TimeSqueeze-base backbone: 12 layers, 12 heads, d_model 384, d_ff 1536, 8 experts, top-2
(~113M parameters, ~50M active).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MoEConfig:
  hidden_size: int = 384
  intermediate_size: int = 1536      # shared expert width; routed experts use intermediate_size // top_k
  num_layers: int = 12
  num_heads: int = 12
  num_experts: int = 8
  top_k: int = 2
  rope_theta: float = 10000.0
  rms_norm_eps: float = 1e-6
  aux_loss_factor: float = 0.02
  init_std: float = 0.02


class RMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return self.weight * x.to(dtype)


def rotate_half(x):
  h = x.shape[-1] // 2
  return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Rotary(nn.Module):
  """Rotary embedding computed on the fly so positions can be fractional (elapsed time in tick units)."""

  def __init__(self, head_dim, theta):
    super().__init__()
    self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)),
                         persistent=False)

  def forward(self, q, k, position_ids):
    # q, k: [B, H, P, hd]; position_ids: [B, P] position of every token (timestep index or elapsed time)
    freqs = position_ids.float().unsqueeze(-1) * self.inv_freq              # [B, P, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(1).to(q.dtype), emb.sin().unsqueeze(1).to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.num_heads = cfg.num_heads
    self.head_dim = cfg.hidden_size // cfg.num_heads
    self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
    self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
    self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
    self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
    self.rotary = Rotary(self.head_dim, cfg.rope_theta)

  def forward(self, x, position_ids):
    B, P, D = x.shape
    split = lambda t: t.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
    q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
    q, k = self.rotary(q, k, position_ids)
    # Padding sits at the end of each sequence, so the causal mask alone keeps real tokens from seeing it.
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.o_proj(out.transpose(1, 2).reshape(B, P, D))


class SwiGLU(nn.Module):
  def __init__(self, hidden, inter):
    super().__init__()
    self.gate_proj = nn.Linear(hidden, inter, bias=False)
    self.up_proj = nn.Linear(hidden, inter, bias=False)
    self.down_proj = nn.Linear(inter, hidden, bias=False)

  def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.top_k, self.num_experts = cfg.top_k, cfg.num_experts
    self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
    self.experts = nn.ModuleList(
        [SwiGLU(cfg.hidden_size, cfg.intermediate_size // cfg.top_k) for _ in range(cfg.num_experts)])
    self.shared_expert = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
    self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

  def forward(self, x):
    B, P, D = x.shape
    flat = x.reshape(-1, D)
    router_logits = self.gate(flat)
    weights, selected = torch.topk(F.softmax(router_logits, dim=1, dtype=torch.float), self.top_k, dim=-1)
    weights = weights.to(flat.dtype)
    # Sort the (token, expert) pairs by expert so each expert processes one contiguous slice. The expert counts are the
    # only host sync of the layer (the previous torch.where per expert synced once per expert).
    flat_experts = selected.reshape(-1)                                       # [N * k]
    order = torch.argsort(flat_experts, stable=True)
    counts = torch.bincount(flat_experts, minlength=self.num_experts).tolist()
    token_idx = order // self.top_k
    x_sorted = flat[token_idx]
    w_sorted = weights.reshape(-1)[order].unsqueeze(-1)
    pieces, start = [], 0
    for e, c in enumerate(counts):
      if c:
        pieces.append(self.experts[e](x_sorted[start:start + c]) * w_sorted[start:start + c])
        start += c
    out = torch.zeros_like(flat).index_add_(0, token_idx, torch.cat(pieces).to(flat.dtype))
    out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
    return out.view(B, P, D), router_logits


class DecoderLayer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.self_attn = Attention(cfg)
    self.ffn = SparseMoE(cfg)
    self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

  def forward(self, x, position_ids):
    x = x + self.self_attn(self.input_layernorm(x), position_ids)
    ffn_out, router_logits = self.ffn(self.post_attention_layernorm(x))
    return x + ffn_out, router_logits


def load_balancing_loss(router_logits, top_k, num_experts, token_mask=None):
  """Switch-Transformer auxiliary loss over all layers (L_aux = N * sum_i f_i * r_i); padded tokens are ignored.

  router_logits: list of [B*P, E] per layer; token_mask: [B, P] bool, True for real tokens.
  """
  logits = torch.cat(router_logits, dim=0)
  probs = F.softmax(logits, dim=-1, dtype=torch.float)
  selected = torch.topk(probs, top_k, dim=-1).indices
  expert_mask = F.one_hot(selected, num_experts).float()            # [L*B*P, k, E]
  if token_mask is None:
    tokens_per_expert = expert_mask.mean(0)                          # [k, E]
    router_prob = probs.mean(0)                                      # [E]
  else:
    m = token_mask.reshape(-1).float().repeat(len(router_logits))    # [L*B*P]
    denom = m.sum().clamp(min=1.0)
    tokens_per_expert = (expert_mask * m[:, None, None]).sum(0) / denom
    router_prob = (probs * m[:, None]).sum(0) / denom
  return (tokens_per_expert * router_prob.unsqueeze(0)).sum() * num_experts


class MoETransformer(nn.Module):
  """Causal MoE Transformer over patch tokens [B, P, D] with explicit rotary positions [B, P]."""

  def __init__(self, cfg=None):
    super().__init__()
    self.cfg = cfg or MoEConfig()
    self.layers = nn.ModuleList([DecoderLayer(self.cfg) for _ in range(self.cfg.num_layers)])
    self.norm = RMSNorm(self.cfg.hidden_size, self.cfg.rms_norm_eps)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, 0.0, self.cfg.init_std)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def forward(self, tokens, position_ids, token_mask=None):
    router_logits = []
    h = tokens
    for layer in self.layers:
      h, logits = layer(h, position_ids)
      router_logits.append(logits)
    aux = load_balancing_loss(router_logits, self.cfg.top_k, self.cfg.num_experts, token_mask)
    return self.norm(h), aux
