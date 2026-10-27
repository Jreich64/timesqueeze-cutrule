"""Command line: `python -m timesqueeze <command> [options]` (run from the project folder, inside WSL).

Every command runs in QUICK-TEST mode unless you pass --full: one epoch / step per stage on tiny data, written to
outputs/quick/, to check the code path in minutes. Every training command also takes --spacing both|time|even (real tick
timing, evenly spaced ticks, or both; default both). `python -m timesqueeze <command> --help` lists a command's options.
"""
import importlib
import sys

# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is deliberately NOT set. It routes allocations through CUDA's
# virtual-memory mapping calls, which on WSL go through the Windows graphics kernel; a run with it ended in a Windows
# crash (bugcheck 0x113, VIDEO_DXGKRNL_FATAL_ERROR). GPU memory is kept in bounds by the batch-size probe instead.

# command -> (module, function, keyword arguments, one-line description), in the order a full run uses them
COMMANDS = {
    "train-all":        ("timesqueeze.training.train_all", "main", {},
                         "everything in sequence: both patch rules x both spacings, method 5, benchmarks, methods 3-4"),
    "check-mamba":      ("timesqueeze.utils.check_mamba", "main", {},
                         "GPU smoke test of the Mamba CUDA kernels (forward, backward, bf16, causality)"),
    "pretrain-cutrule": ("timesqueeze.cutrule.stages", "main", {},
                         "stage 1: pretrain the bilevel CutRule on Time-300B"),
    "pretrain":         ("timesqueeze.training.pretrain", "main", {},
                         "stage 2: pretrain the forecaster on Time-300B (--patcher power|cutrule; --benchmark to time it)"),
    "train-cutrule":    ("timesqueeze.training.cutrule_orcl", "main", {},
                         "stage 3: continue the CutRule on the ORCL training days"),
    "finetune":         ("timesqueeze.training.finetune", "main", {},
                         "stage 4: fine-tune on ORCL and evaluate (runs every missing earlier stage first)"),
    "joint":            ("timesqueeze.training.joint", "main", {"default_leader": "reinforce"},
                         "method 3: CutRule and forecaster trained together (REINFORCE feedback)"),
    "trilevel":         ("timesqueeze.training.joint", "main", {"default_leader": "trilevel"},
                         "method 4: the CutRule as a three-level problem, trained with the forecaster"),
    "power-bilevel":    ("timesqueeze.training.power_bilevel", "main", {},
                         "method 5: power rule with tau as the follower, forecasting as the leader (on ORCL)"),
    "benchmark":        ("timesqueeze.training.benchmark", "main", {},
                         "paper benchmarks ETTh1/2, ETTm1/2, Weather: zero-shot and full-shot, horizons 96-720"),
    "download":         ("timesqueeze.data.time300b_download", "main", {},
                         "download more Time-300B windows from Hugging Face (needs HF_TOKEN)"),
    "index":            ("timesqueeze.data.time300b_segments", "main", {},
                         "rebuild the series-boundary index of Time-300B subset folders"),
}


def usage():
  lines = [__doc__.strip(), "", "commands:"]
  lines += [f"  {name:<17} {desc}" for name, (_, _, _, desc) in COMMANDS.items()]
  return "\n".join(lines)


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  if not argv or argv[0] in ("-h", "--help", "help"):
    print(usage())
    return 0
  name, rest = argv[0], argv[1:]
  if name not in COMMANDS:
    print(f"unknown command {name!r}\n\n{usage()}", file=sys.stderr)
    return 2
  module, function, kwargs, _ = COMMANDS[name]
  sys.argv = [f"python -m timesqueeze {name}", *rest]          # the command's own argparse reads sys.argv
  getattr(importlib.import_module(module), function)(**kwargs)
  return 0


if __name__ == "__main__":
  sys.exit(main())
