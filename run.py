"""One launcher for every training command: edit the settings below, then press Run in VS Code (connected to WSL) or run
`uv run python run.py` in the WSL terminal. It is the same as typing `python -m timesqueeze <TASK> ...` yourself.

Tasks ("train-all" runs every one of them in sequence; "finetune" alone runs the main pipeline for one PATCHER/SPACING):
  "train-all"         everything, one stage after another (resumable; see timesqueeze/training/train_all.py)
  "pretrain-cutrule"  stage 1: the CutRule on Time-300B
  "pretrain"          stage 2: the forecaster on Time-300B
  "train-cutrule"     stage 3: the CutRule on the ORCL training days
  "finetune"          stage 4: the forecaster on ORCL, then the test-day evaluation and saved report
  "joint" / "trilevel" methods 3 and 4: CutRule and forecaster trained together
  "power-bilevel"     method 5: power rule with tau as the follower, forecasting as the leader
  "benchmark"         the paper's ETT / Weather benchmarks (zero-shot and full-shot) for the PATCHER's backbone
  "check-mamba"       GPU smoke test of the Mamba kernels
"""
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------ settings
TASK = "train-all"
QUICK_TEST = True        # True: one epoch/step per stage on tiny data, to check the code runs (writes to outputs/quick).
                         # False: the real run (writes to outputs/).
PATCHER = "cutrule"      # "cutrule" (learned bilevel cut rule) or "power" (the paper's rule); ignored by stages 1 and 3
SPACING = "time"         # "time" (real tick timing), "even" (ticks evenly spaced) or "both" (one after the other)
EXTRA_ARGS = [
    # "--pretrain-steps", "1400",      # finetune: backbone steps if no backbone exists yet (about 2 hours)
    # "--epochs", "5",
    # "--restart",                     # start interrupted long phases over (default: they continue)
    # "--benchmark",                   # pretrain: time 3 steps, print batch size, speed and memory, then exit
    # "--stride", "4",                 # benchmark: score every 4th test window (1 = the paper's protocol)
]
# ---------------------------------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TAKES_PATCHER = {"pretrain", "finetune", "joint", "trilevel", "benchmark"}
NO_SPACING = {"check-mamba", "train-all", "pretrain", "pretrain-cutrule", "benchmark"}   # shared by both spacings


def command():
  args = [TASK]
  if TASK != "check-mamba":
    args.append("--quick" if QUICK_TEST else "--full")
  if TASK in TAKES_PATCHER:
    args += ["--patcher", PATCHER]
  if TASK not in NO_SPACING:
    args += ["--spacing", SPACING]
  return args + EXTRA_ARGS


if __name__ == "__main__":
  python = ROOT / ".venv" / "bin" / "python"      # the project environment made by `uv sync` (WSL)
  if not python.exists():
    sys.exit(f"{python} not found: run `uv sync` in the project folder inside WSL first.")
  argv = [str(python), "-m", "timesqueeze", *command()]
  print(" ".join(argv[1:]), flush=True)
  sys.exit(subprocess.call(argv, cwd=ROOT))          # cwd: outputs/ lands in the project folder
