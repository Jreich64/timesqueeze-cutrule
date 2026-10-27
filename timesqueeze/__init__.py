"""TimeSqueeze: dynamic-patching time-series forecasting (Papers/Time_Squeeze.pdf), pretrained on Time-300B and fine-tuned on
ORCL trade data, with a learned patch rule (the CutRule) as an alternative to the paper's.

Code map (each subpackage has its own docstring with more detail)

  data/        where the numbers come from
               orcl.py              ORCL trades -> per-day returns, tick timing, the 30/7/6-day split
               orcl_windows.py      ORCL training windows (inside one day) with their patch boundaries
               time300b_download.py fetch extra Time-300B windows from Hugging Face
               time300b_segments.py recover the true series boundaries inside downloaded windows
               time300b_windows.py  Time-300B training windows (inside one series) for pretraining

  patching/    where a series is cut into patches (CPU, compiled with numba)
               power_rule.py        the paper's rule (+ tick-timing features)
               volatility.py        EWMA volatility (tick-based and time-aware)
               segmenter.py         apply a trained CutRule to a window (deterministic or sampled)
               batching.py          per-window boundaries -> batch tensors
               kernels.py           the compiled sequential loops

  cutrule/     the learned patch rule
               model.py             the CutRule MLP and its soft (differentiable) relaxation
               bilevel.py           bilevel training: leader = MLP weights, follower = threshold
               stages.py            pretraining on Time-300B, continued training on ORCL, checkpoints
               trilevel.py          method 4: duration / volatility / forecasting as three nested levels

  model/       the forecaster
               moe.py               the 12-layer Mixture-of-Experts Transformer (Time-MoE style)
               forecaster.py        Mamba encoder -> patch -> Transformer -> unpatch -> Mamba decoder -> heads; losses
               backbone.py          build, save, load (or pretrain-if-missing) the forecaster

  training/    the training programs (each has a command line, see `python -m timesqueeze --help`)
               common.py            optimizer, LR schedule, batch-size finder
               pretrain.py          stage 2: forecaster on Time-300B
               cutrule_orcl.py      stage 3: CutRule on the ORCL training days
               finetune.py          stage 4: forecaster on ORCL (runs every missing earlier stage first)
               evaluation.py        test-set scores and the saved forecast report
               joint.py             method 3: CutRule + forecaster together (REINFORCE)
               (method 4 is joint.py with --leader trilevel: `python -m timesqueeze trilevel`)
               power_bilevel.py     method 5: power rule, tau = follower, forecasting = leader
               benchmark.py         the paper's ETT / Weather benchmarks, zero-shot and full-shot
               train_all.py         every stage in sequence (resumable): `python -m timesqueeze train-all`

  benchmarks/  datasets.py (download, standard splits, train-fitted scaling), rolling.py (forecasts past the 64-step
               head), evaluate.py (scoring and full-shot training windows)

  utils/       paths.py (data/outputs folders), quick.py (quick-test mode), seed.py, reporting.py (CSV/plots),
               check_mamba.py (GPU smoke test of the Mamba kernels)

How a full run flows (`python -m timesqueeze finetune --full --patcher cutrule`)
  1. CutRule pretrained on Time-300B        -> outputs/cutrule_pretrained.pt
  2. forecaster pretrained on Time-300B     -> outputs/backbone_cutrule.pt       (patches from 1)
  3. CutRule continued on ORCL train days   -> outputs/checkpoint.pt
  4. forecaster fine-tuned on ORCL          -> outputs/finetuned_cutrule.pt, outputs/results/cutrule/
  Each step is skipped if its output already exists. Steps 1-2 are shared by both spacing variants (Time-300B has no
  timestamps); the evenly-spaced variant writes steps 3-4 under the same names with "_ticks".
  `python -m timesqueeze train-all --full` runs this for both rules and both spacings, then methods 3-5 and the benchmarks.
"""
