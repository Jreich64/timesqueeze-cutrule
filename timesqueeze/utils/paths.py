"""Where things live on disk: the data folder (fixed) and the outputs folder (relative to where you run from).

  * DATA_DIR  - <project>/data, computed from this file's location so it is right no matter where a command runs.
  * outputs   - "outputs/" relative to the current directory (run commands from the project folder). It is relative on
                purpose: the tests switch to a temporary directory so their checkpoints never land in the real outputs/.
  * Quick-test runs (the default) redirect every output to outputs/quick, so their meaningless 1-epoch weights can never
    be picked up by a later full run as if they were pretrained.
  * The evenly-spaced variant writes its ORCL-stage files (ORCL CutRule, fine-tuned models, methods 3-5) under the same
    names with a "_ticks" suffix (variant_suffix), so the two variants never read each other's ORCL weights. The Time-300B
    stages (pretrained CutRule, backbones) are shared: Time-300B has no timestamps.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # <project>/timesqueeze/utils/paths.py -> <project>
DATA_DIR = ROOT / "data"

QUICK_OUTPUT_DIR = os.path.join("outputs", "quick")
_state = {"dir": "outputs"}


def set_output_dir(path):
  _state["dir"] = path


def output_dir():
  return _state["dir"]


def out(*parts):
  """Path of an output file, e.g. out("backbone_power.pt") -> outputs/backbone_power.pt (or outputs/quick/...)."""
  return os.path.join(_state["dir"], *parts)


def variant_suffix(time_aware):
  """ORCL-stage files of the evenly-spaced variant (every tick one time step) carry this suffix so they never mix
  with the time-aware ones: checkpoint_ticks.pt, finetuned_power_ticks.pt, ..."""
  return "" if time_aware else "_ticks"


def use_quick_outputs(quick):
  """Call at the start of a command line: quick runs write to outputs/quick, full runs to outputs."""
  set_output_dir(QUICK_OUTPUT_DIR if quick else "outputs")
