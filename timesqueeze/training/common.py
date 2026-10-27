"""Pieces shared by every training loop: optimizer, learning-rate schedule, batch-size finder, device transfer.

  * make_adamw / param_groups - AdamW as in the paper (lr 1e-3, weight decay 0.1 on matrices only, betas 0.9 / 0.95),
                                with the fused CUDA kernel on the GPU.
  * lr_at                     - linear warmup, then cosine decay to 5e-5.
  * find_batch_size           - doubles the micro-batch on real samples until it would use more than half the GPU
                                memory (plus the optimizer state it will need) or throughput stops improving.
  * default_workers           - dataloader worker processes: about half the CPU cores, at most 8.
  * train_one_epoch           - one pass over a loader with the forecasting loss (fine-tuning on ORCL, full-shot
                                benchmark fine-tuning), each step split into micro-batches to bound GPU memory.
  * split_batch               - a collated batch cut into micro-batches along the batch dimension.
  * BestOnValidation          - keeps a copy of the weights with the lowest validation loss (model selection).
"""
import math
import os
import time

import torch

from ..model.forecaster import forecasting_loss, run_model

LR, MIN_LR, WEIGHT_DECAY, BETAS = 1e-3, 5e-5, 0.1, (0.9, 0.95)
ORCL_BATCH = 32          # windows per optimizer step on ORCL (finetune and phase B of methods 3 / 4: equal step counts)
# ORCL fine-tuning learning rate. The paper fine-tunes at 1e-4; on ORCL that overfit within the first epoch (validation
# loss best after epoch 1, then rising while training loss fell), so every ORCL phase uses 1e-5.
FINETUNE_LR = 1e-5
# Validation (and best-checkpoint selection) every this many training windows - 200 steps at 32 windows, ~5 per epoch -
# because the best point turned out to lie inside the first epoch.
VAL_EVERY_WINDOWS = 6400


def lr_at(step, total_steps, warmup_steps, lr=LR, min_lr=MIN_LR):
  if step < warmup_steps:
    return lr * (step + 1) / warmup_steps
  progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
  return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def param_groups(model, weight_decay):
  decay = [p for p in model.parameters() if p.ndim >= 2]
  no_decay = [p for p in model.parameters() if p.ndim < 2]       # biases and norm weights
  return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def make_adamw(model, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS):
  """AdamW with the fused CUDA kernel when the model is on the GPU."""
  return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas,
                           fused=next(model.parameters()).is_cuda)


def default_workers():
  """Dataloader worker processes: about half the logical cores (the patch rules are CPU work), at most 8."""
  return max(2, min(8, (os.cpu_count() or 4) // 2))


def find_batch_size(model, make_batch, device, cap, headroom=0.5, min_gain=1.1, verbose=True, reserve_bytes=0):
  """Largest power-of-two batch size (<= cap) that trains comfortably on this GPU.

  Doubles the batch, running a real forward + backward on `make_batch(b)` (a device batch dict), until it runs out of
  memory, would use more than `headroom` of GPU memory, or stops being at least `min_gain` times faster per sample
  than the previous size. reserve_bytes is memory that will be needed later but is not in use during the probe (the
  optimizer state, 8 bytes per parameter for AdamW). Returns the last size that was good. On the CPU: min(cap, 4).

  headroom is deliberately 0.5: the probe sees two fixed batches, but real training draws windows with more patches,
  fragments the allocator and keeps the optimizer state, and peaked ~3 GB above the probe (power-rule pretraining:
  4.1 GB probed, 7.4 GB in training on an 8 GB GPU). Past ~90% the WSL driver pages GPU memory to system RAM."""
  if device.type != "cuda":
    return min(cap, 4)
  total = torch.cuda.get_device_properties(device).total_memory
  best, best_rate, b = 1, 0.0, 1
  was_training = model.training
  model.train()
  while b <= cap:
    try:
      batch = make_batch(b)
      torch.cuda.synchronize(device)
      torch.cuda.reset_peak_memory_stats(device)
      for i in range(2):                                           # the first pass warms up kernels / allocator
        if i == 1:
          torch.cuda.synchronize(device)
          t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
          preds, aux = run_model(model, batch)
        loss, _ = forecasting_loss(preds, batch["y"], batch["n_valid"], aux)
        loss.backward()
      torch.cuda.synchronize(device)
      sec = time.perf_counter() - t0
      peak = torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError:
      if verbose:
        print(f"  batch {b}: out of memory")
      break
    finally:
      model.zero_grad(set_to_none=True)
      batch = preds = aux = loss = None
      torch.cuda.empty_cache()
    rate = b / sec
    if verbose:
      print(f"  batch {b}: {sec * 1e3:.0f} ms, {rate:.1f} samples/s, peak {peak / 2**30:.1f} of {total / 2**30:.1f} GB")
    if peak + reserve_bytes > headroom * total or (best_rate and rate < min_gain * best_rate):
      break
    best, best_rate, b = b, rate, b * 2
  model.train(was_training)
  return best


def to_device(batch, device):
  return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class BestOnValidation:
  """Model selection for the ORCL phases: remembers the weights (and anything else, e.g. the CutRule or tau) with the
  lowest validation loss seen so far. The first full run showed fine-tuning on ORCL overfitting from the first epoch
  (validation loss rising every epoch while training loss fell), so every method is evaluated with its best validation
  epoch - and epoch 0, the model before fine-tuning, is a candidate too, so fine-tuning can never make it worse on the
  validation days. Copies are kept on the CPU."""

  def __init__(self):
    self.loss, self.epoch, self.state, self.extra = float("inf"), None, None, None

  def update(self, val_loss, epoch, model, extra=None):
    """Record this epoch if it is the best so far. Returns True if it was."""
    if val_loss < self.loss:
      self.loss, self.epoch = val_loss, epoch
      self.state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
      self.extra = extra
      return True
    return False

  def to_dict(self):
    """Plain dict for a checkpoint (torch.load's safe mode only reads tensors and builtin types)."""
    return {"loss": self.loss, "epoch": self.epoch, "state": self.state, "extra": self.extra}

  @classmethod
  def from_dict(cls, d):
    b = cls()
    b.loss, b.epoch, b.state, b.extra = d["loss"], d["epoch"], d["state"], d["extra"]
    return b

  def restore(self, model):
    """Load the best weights into `model`; returns the extra object stored with them."""
    model.load_state_dict(self.state)
    print(f"selected epoch {self.epoch} (lowest validation loss {self.loss:.4f})", flush=True)
    return self.extra


def split_batch(batch, size):
  """Cut a collated batch dict into micro-batches of at most `size` rows (every value is batch-first)."""
  n = len(batch["y"])
  return [{k: v[i:i + size] for k, v in batch.items()} for i in range(0, n, size)]


def train_one_epoch(model, opt, loader, device, log_every=0, label="train", micro_batch=8, after_step=None):
  """One epoch of the multi-horizon forecasting loss (bf16 autocast, gradient clipping 1.0). Returns the mean loss.

  Each loader batch is one optimizer step, computed in micro-batches of `micro_batch` windows whose gradients are
  accumulated (weighted by their share of the batch), so the step is the same as one big batch but GPU memory is
  bounded. The running loss stays a GPU tensor (no host sync per batch); with log_every > 0 a progress line is printed
  every log_every batches. after_step(k, mean_loss_so_far) is called after optimizer step k (1-based) of the epoch;
  mean_loss_so_far is a function, so the host sync only happens when the callback asks for it (e.g. to validate)."""
  model.train()
  run, k, t0 = 0.0, 0, time.time()
  n = len(loader) if hasattr(loader, "__len__") else None
  for batch in loader:
    opt.zero_grad(set_to_none=True)
    total = len(batch["y"])
    step_loss = 0.0
    for mb in split_batch(batch, micro_batch):
      b = to_device(mb, device)
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        preds, aux = run_model(model, b)
      loss, _ = forecasting_loss(preds, b["y"], b["n_valid"], aux)
      (loss * (len(mb["y"]) / total)).backward()
      step_loss = step_loss + loss.detach() * (len(mb["y"]) / total)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    run = run + step_loss; k += 1
    if after_step is not None:
      after_step(k, lambda: float(run) / k)
    if log_every and k % log_every == 0:
      eta = (n - k) * (time.time() - t0) / k / 60 if n else float("nan")
      print(f"  {label} batch {k}/{n or '?'} loss {float(run) / k:.4f} ETA {eta:.1f} min", flush=True)
  return float(run) / max(k, 1)
