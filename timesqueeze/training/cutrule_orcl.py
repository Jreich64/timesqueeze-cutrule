"""Stage 3 on its own: continue training the CutRule on the ORCL training days.

It starts from the CutRule pretrained on Time-300B (stage 1, trained first if missing; --from-scratch skips it) and runs
the bilevel training of cutrule/bilevel.py on the 30 training days with the real tick timing (or evenly spaced ticks).
The result, outputs/checkpoint[_ticks].pt, is the patch rule used for fine-tuning. `finetune` runs this stage itself
when the checkpoint is missing; this command is for training or inspecting the rule alone.
"""
import argparse
import time

import torch

from ..cutrule.stages import train_orcl_cutrule
from ..data.orcl import load_orcl
from ..utils.paths import DATA_DIR, use_quick_outputs
from ..utils.quick import CUTRULE_ORCL, CUTRULE_PRETRAIN, add_quick_flags, add_spacing_flag, banner, spacings
from ..utils.seed import ignore_deprecation_warnings, set_seed


def model_device():
  return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'


def main():
  parser = argparse.ArgumentParser(description="Train the bilevel CutRule on ORCL trades (pretrained on Time-300B first).")
  add_quick_flags(parser)
  add_spacing_flag(parser)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--from-scratch", action="store_true", help="skip the Time-300B pretraining stage")
  parser.add_argument("--data-dir", default=str(DATA_DIR))
  parser.add_argument("--ckpt", default=None, help="single rolling checkpoint, default outputs/checkpoint[_ticks].pt (each save replaces the last)")
  parser.add_argument("--resume", action="store_true", help="continue from --ckpt if it exists")
  parser.add_argument("--no-time-aware", action="store_true", help="same as --spacing even")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  use_quick_outputs(args.quick)
  ignore_deprecation_warnings()
  set_seed(args.seed)
  data = load_orcl(args.data_dir)
  print(f"training on {model_device()}")
  start = time.time()
  epochs, pre = args.epochs, None
  if args.quick:
    banner(True)
    epochs, pre = CUTRULE_ORCL["epochs"], dict(CUTRULE_PRETRAIN)
  variants = [False] if args.no_time_aware else spacings(args.spacing)
  for time_aware in variants:
    print(f"--- CutRule on ORCL, {'real tick timing' if time_aware else 'ticks evenly spaced'}")
    model, trace = train_orcl_cutrule(data, time_aware=time_aware, ckpt_path=args.ckpt if len(variants) == 1 else None,
                                      epochs=epochs, from_scratch=args.from_scratch, resume=args.resume,
                                      pretrain_kwargs=pre)
  print(f"done in {time.time() - start:.1f}s")
  return model, trace


if __name__ == "__main__":
  main()
