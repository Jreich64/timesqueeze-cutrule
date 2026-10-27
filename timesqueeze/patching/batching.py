"""Turning per-window patch boundaries into batch tensors for the forecaster (vectorised over the batch).

For a batch of windows of length T with different numbers of patches P_i:
  patch_pos  [B, P]  the start index of every patch (padded with 0),
  patch_mask [B, P]  True for real patches, False for padding,
  patch_id   [B, T]  which patch every tick belongs to (used to "unpatch": copy each patch token back over its ticks).
"""
import numpy as np


def patch_ids(bounds, T):
  """patch index of every timestep: id[t] = number of boundaries <= t, minus 1."""
  marks = np.zeros(T, dtype=np.int64)
  marks[bounds] = 1
  return np.cumsum(marks) - 1


def collate_patches(bounds_list, T):
  """Pad a list of boundary arrays into batch tensors' numpy arrays (vectorised over the batch).

  Returns patch_pos [B, P] (original start index of each patch, padded with 0), patch_mask [B, P] (True = real
  patch) and patch_id [B, T] (which patch each timestep belongs to).
  """
  B = len(bounds_list)
  lens = np.fromiter((len(b) for b in bounds_list), dtype=np.int64, count=B)
  P = int(lens.max())
  flat = np.concatenate(bounds_list).astype(np.int64)
  mask = np.arange(P)[None, :] < lens[:, None]
  pos = np.zeros((B, P), dtype=np.int64)
  pos[mask] = flat                                              # row-major order of the mask = concatenation order
  marks = np.zeros((B, T), dtype=np.int64)
  marks[np.repeat(np.arange(B), lens), flat] = 1
  return pos, mask, np.cumsum(marks, axis=1) - 1
