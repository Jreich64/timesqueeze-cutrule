"""Train and evaluate everything, one stage after another: `python -m timesqueeze train-all --full`.

Stages, in order (the most important comparison first, so it is ready earliest):
   1. finetune-cutrule-time   CutRule pretrained on Time-300B -> cutrule backbone -> ORCL CutRule -> fine-tune
   2. finetune-power-time     power-rule backbone -> fine-tune                      (the paper's method; the baseline)
   3. finetune-cutrule-even   evenly spaced ticks: reuses the CutRule and backbone of 1, own ORCL CutRule and fine-tune
   4. finetune-power-even     reuses the backbone of 2
   5. power-bilevel-time      method 5 on the power backbone (tau = follower, forecasting = leader)
   6. power-bilevel-even
   7. benchmark-cutrule       ETT / Weather, zero-shot and full-shot, on the cutrule backbone
   8. benchmark-power         the same on the power backbone (the paper's Tables 1 and 2 setting)
   9. joint-time / 10. joint-even          method 3: CutRule learns by REINFORCE during both training phases
  11. trilevel-time / 12. trilevel-even    method 4: the same with the three-level CutRule
  13. benchmark-joint / 14. benchmark-trilevel   ETT / Weather on the Time-300B forecasters of methods 3 / 4
The Time-300B phases (inside stages 1, 2, 9 and 11) take the same 5,000 steps x 64 windows each; methods 3 and 4 forecast
every window twice (two sampled patchings), so their Time-300B phase takes about twice as long as a backbone pretraining
(~12 h). The ORCL phases all run 5 epochs over the same windows (methods 3/4 and finetune at 32 windows per step).

Each stage runs as its own process (`python -m timesqueeze <command> ...`), so one failing stage does not stop the rest;
its output goes to the console and to outputs/logs/<stage>.log. Progress is recorded in outputs/train_all_status.json.

Pausing and resuming: stop at any time (Ctrl+C, closing the tmux session, a reboot). Running the same command again
skips the stages already done, and the interrupted stage continues from its rolling checkpoint (pretraining every 100
steps, methods 3/4 phase A every 50, ORCL phases and method 5 every epoch, benchmarks after every dataset and mode), so
at most a few minutes to an epoch of work is lost.
--redo runs finished stages again from scratch (their long phases with --restart).
Every stage writes its own results under outputs/results/ (see the README).
"""
import argparse
import json
import os
import subprocess
import sys
import time

from ..utils.paths import output_dir, use_quick_outputs
from ..utils.quick import add_quick_flags, banner
from ..utils.reporting import save_json


def stage_list(pretrain_steps, bench_stride):
  """[(name, command line arguments)] in run order."""
  steps = ["--pretrain-steps", str(pretrain_steps)]
  bench = ["--stride", str(bench_stride), "--train-stride", str(bench_stride)]
  stages = [(f"finetune-{p}-{sp}", ["finetune", "--patcher", p, "--spacing", sp, *steps])
            for sp, p in (("time", "cutrule"), ("time", "power"), ("even", "cutrule"), ("even", "power"))]
  stages += [(f"power-bilevel-{sp}", ["power-bilevel", "--spacing", sp, *steps]) for sp in ("time", "even")]
  stages += [(f"benchmark-{p}", ["benchmark", "--patcher", p, *bench, *steps]) for p in ("cutrule", "power")]
  # methods 3 and 4: phase A on Time-300B with the same steps as a backbone pretraining (shared by both spacings, run
  # inside the -time stage), then phase B on ORCL with the same epochs and batch as finetune
  stages += [(f"{m}-{sp}", [m, "--spacing", sp, *steps]) for m in ("joint", "trilevel") for sp in ("time", "even")]
  # the benchmarks again, on the phase-A forecasters of methods 3 and 4 with the CutRule each learned
  stages += [(f"benchmark-{m}", ["benchmark", "--patcher", m, *bench, *steps]) for m in ("joint", "trilevel")]
  return stages


RESTARTABLE = {"finetune", "joint", "trilevel", "power-bilevel", "benchmark"}   # commands that take --restart


def run_stage(name, args, quick, log_dir, restart=False):
  """Run one stage as a child process, streaming its output to the console and to log_dir/<name>.log.
  Returns the exit code."""
  cmd = [sys.executable, "-u", "-m", "timesqueeze", args[0], "--quick" if quick else "--full", *args[1:]]
  if restart and args[0] in RESTARTABLE:
    cmd.append("--restart")
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf8") as log:
    log.write(" ".join(cmd[2:]) + "\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
      sys.stdout.write(line)
      sys.stdout.flush()                                  # live progress when the console is a pipe (e.g. `| tee`)
      log.write(line)
      log.flush()
    return proc.wait()


def train_all(quick=True, pretrain_steps=5000, bench_stride=4, only=None, skip=(), redo=False, stop_on_failure=False):
  """Run the stages in order; returns {stage: status}."""
  status_path = os.path.join(output_dir(), "train_all_status.json")
  status = json.load(open(status_path)) if os.path.exists(status_path) and not redo else {}
  stages = [(n, a) for n, a in stage_list(pretrain_steps, bench_stride)
            if (not only or n in only) and n not in skip]
  print(f"train-all: {len(stages)} stages; status in {status_path}", flush=True)
  for i, (name, args) in enumerate(stages, 1):
    if status.get(name, {}).get("result") == "done":
      print(f"[{i}/{len(stages)}] {name}: already done, skipping", flush=True)
      continue
    print(f"\n{'=' * 100}\n[{i}/{len(stages)}] {name}: python -m timesqueeze {' '.join(args)}"
          f"   ({time.strftime('%Y-%m-%d %H:%M')})\n{'=' * 100}", flush=True)
    t0 = time.time()
    code = run_stage(name, args, quick, os.path.join(output_dir(), "logs"), restart=redo)
    status[name] = {"result": "done" if code == 0 else f"failed (exit code {code})",
                    "minutes": round((time.time() - t0) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M")}
    save_json(status_path, status)                                # written durably: a crash keeps the old or new file
    print(f"[{i}/{len(stages)}] {name}: {status[name]['result']} after {status[name]['minutes']} min", flush=True)
    if code != 0 and stop_on_failure:
      break
  print("\ntrain-all summary:")
  for name, _ in stages:
    s = status.get(name, {"result": "not run"})
    print(f"  {name:<24} {s['result']}" + (f"  ({s['minutes']} min)" if "minutes" in s else ""))
  return status


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  add_quick_flags(p)
  p.add_argument("--pretrain-steps", type=int, default=5000, help="Time-300B pretraining steps per backbone (~4.5 s each)")
  p.add_argument("--bench-stride", type=int, default=4,
                 help="benchmarks: score / fine-tune on every n-th window (1 = the paper's protocol, ~4x longer)")
  names = [n for n, _ in stage_list(5000, 4)]
  p.add_argument("--only", nargs="+", choices=names, help="run just these stages")
  p.add_argument("--skip", nargs="+", choices=names, default=[], help="leave these stages out")
  p.add_argument("--redo", action="store_true", help="run stages again even if the status file says they are done")
  p.add_argument("--stop-on-failure", action="store_true", help="stop at the first failing stage (default: go on)")
  a = p.parse_args()
  use_quick_outputs(a.quick)
  banner(a.quick)
  status = train_all(a.quick, a.pretrain_steps, a.bench_stride, a.only, a.skip, a.redo, a.stop_on_failure)
  sys.exit(0 if all(s.get("result") == "done" for s in status.values()) else 1)


if __name__ == "__main__":
  main()
