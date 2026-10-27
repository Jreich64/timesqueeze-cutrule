"""Quick-test mode: run every stage for a single epoch/step on tiny data so the whole code path can be verified in
minutes. It is ON by default everywhere (so a casual run is a smoke test, not a multi-hour job); pass --full to any
command line (or set QUICK_TEST = False in a run_*.py file) for the real settings.

The numbers below are deliberately minimal: one epoch/round/step per stage, small corpora, few windows. They exercise
every code path (data, patching, CutRule stages, pretraining, fine-tuning, saving results) but produce meaningless
models.
"""

QUICK_DEFAULT = True

# Per stage: keyword overrides for the corresponding function.
CUTRULE_PRETRAIN = {"n_segments": 40, "segment_len": 512, "epochs": 1}
BACKBONE_PRETRAIN = {"steps": 1, "micro_batch": 2, "global_batch": 2, "context": 256, "num_workers": 0,
                     "val_every": 0, "ckpt_every": 1, "log_every": 1}
FINETUNE = {"epochs": 1, "cutrule_epochs": 1, "stride": 512, "context": 256, "batch_size": 4}
JOINT = {"pretrain": {"steps": 1, "rounds": 1, "global_batch": 2, "micro_windows": 1, "context": 256, "bilevel_epochs": 1,
                      "corpus_series": 20, "corpus_len": 512, "bars": 300, "inner_iters": 1,
                      "cutrule_kwargs": CUTRULE_PRETRAIN},
         "epochs": 1, "batch_windows": 2, "micro_windows": 1, "context": 256, "stride": 512, "cutrule_epochs": 1,
         "bilevel_epochs": 1, "eval_windows": 8}
TRILEVEL = {**JOINT, "inner_iters": 1, "bars": 300}
CUTRULE_ORCL = {"epochs": 1}
BENCHMARK = {"max_windows": 2, "train_stride": 8000, "train_batch_size": 4, "batch_size": 64}
POWER_BILEVEL = {"steps": 2, "batch_size": 2, "context": 256, "stride": 512, "follower_evals": 3, "eval_every": 0}


def add_quick_flags(parser):
  """--quick (default on), --no-quick and --full (alias of --no-quick)."""
  g = parser.add_mutually_exclusive_group()
  g.add_argument("--quick", dest="quick", action="store_true", default=QUICK_DEFAULT,
                 help="1 epoch/step per stage on tiny data, to verify the code runs (default: %(default)s)")
  g.add_argument("--full", "--no-quick", dest="quick", action="store_false",
                 help="real settings instead of the quick smoke test")
  return parser


def add_spacing_flag(parser):
  """--spacing both|time|even: run the pipeline with the real tick timing, with ticks treated as evenly spaced, or both
  (default). Each variant saves its own ORCL models (the evenly spaced ones with a _ticks suffix); the Time-300B
  pretraining is shared."""
  parser.add_argument("--spacing", choices=["both", "time", "even"], default="both",
                      help="time = real ts_event gaps, even = ticks as evenly spaced, both = run both (default)")
  return parser


def spacings(spacing):
  """The time_aware values to run for a --spacing choice."""
  return {"both": [True, False], "time": [True], "even": [False]}[spacing]


def banner(quick):
  if quick:
    print("QUICK TEST MODE: 1 epoch/step per stage on tiny data. Results are not meaningful; use --full for real runs.")
