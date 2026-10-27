"""Reproducibility helpers: seed every random number generator, and silence known harmless deprecation warnings.
"""
import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
  """Seed every RNG. deterministic=True trades speed for bit-reproducible kernels; the default enables the fast paths
  (TF32 matmuls, cudnn autotuning)."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
  return np.random.default_rng(seed)


def ignore_deprecation_warnings():
  warnings.filterwarnings("ignore", category=DeprecationWarning)
