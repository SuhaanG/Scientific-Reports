# IDS Seed Variance Study

Tests whether machine learning-based intrusion detection models that show
stable aggregate accuracy across random seeds actually show unstable
per-attack-category detection underneath, and whether this differs across
architectures (deep learning vs. ensemble methods).

## Current scope

- **Dataset implemented:** NSL-KDD only. CICIDS2017 / CSE-CIC-IDS2018 are
  registered as future extensions (see "Adding a new dataset" below) but
  have no loader yet, using them will raise `NotImplementedError`.
- **Architectures implemented:** DNN (PyTorch), Random Forest
  (scikit-learn), XGBoost. All three are enabled by default in
  `config.ARCHITECTURES`.
- **Seed counts:** 10 for the pilot, 40 for the full run
  (`config.PILOT_SEEDS` / `config.FULL_SEEDS`).

## What was specifically caught and fixed during development

Documenting this here deliberately, since it's evidence the pipeline was
tested against failure modes, not just the happy path.

1. **XGBoost's `random_state` initially had no effect on training.**
   With default `subsample=1.0` and `colsample_bytree=1.0`, every boosting
   round deterministically used all rows and all features, so the seed
   had nothing to actually randomize. This was caught by an explicit
   same-seed-vs-different-seed determinism test, not discovered by
   accident. Fixed by setting `XGB_SUBSAMPLE=0.8` and
   `XGB_COLSAMPLE_BYTREE=0.8` in `config.py`. Without this fix, the
   architecture-comparison analysis would have shown artificially
   "perfect" XGBoost stability that wasn't real.

2. **Duplicate seed rows from an interrupted-and-restarted run.** The
   crash-safe incremental CSV writing means a partially completed run,
   restarted without clearing old output, appends duplicate
   `(dataset, architecture, seed)` rows, silently over-weighting those
   seeds. This was caught during testing, then fixed at two levels:
   `run_pilot.py` and `run_matrix.py` are now idempotent (they check
   which seeds are already complete and only run what's missing), and
   `stats_analysis.py` additionally refuses to analyze a results file
   that somehow still contains duplicates.

3. **The bootstrap variance decomposition was validated against
   synthetic data with a known ground-truth variance**, not just
   checked for plausible-looking output on real data. Recovered
   variance was within 1.6% of the true value when a genuine effect
   was simulated, and correctly near-zero in a simulated null case
   (no genuine effect).

## Setup

```bash
pip install -r requirements.txt
# if the environment is externally managed:
pip install -r requirements.txt --break-system-packages
```

Download `KDDTrain+.txt` and `KDDTest+.txt` from the official NSL-KDD
source and place them in `data/`. `data.py` automatically validates the
row count on load (125973 train / 22544 test) and raises an error rather
than silently proceeding if they don't match.

## Running the pilot (Stage 1: 10 seeds, DNN only, NSL-KDD only)

```bash
python run_pilot.py
python stats_analysis.py
```

`run_pilot.py` is resumable: if interrupted, rerunning it skips any
seeds already completed rather than duplicating them. To force a clean
rerun (e.g. after changing a hyperparameter), delete or rename the
existing results file first.

## Running the full matrix (Stage 2: after the pilot is reviewed)

```bash
python run_matrix.py --seeds pilot   # sanity check with 10 seeds first
python run_matrix.py --seeds full    # 40 seeds, only after reviewing the above
```

Also resumable, across every `(dataset, architecture)` combination
independently.

Analyze a specific combination:

```python
from stats_analysis import run_full_analysis, compare_architectures, load_results, load_per_instance

run_full_analysis(
    "results/nsl_kdd_matrix_summary.csv",
    "results/nsl_kdd_matrix_per_instance.csv",
    dataset="nsl_kdd", architecture="random_forest",
)

# Compare genuine between-seed variance for one category ACROSS architectures:
summary_df = load_results("results/nsl_kdd_matrix_summary.csv")
per_instance_df = load_per_instance("results/nsl_kdd_matrix_per_instance.csv")
compare_architectures(summary_df, per_instance_df, "r2l", "nsl_kdd",
                       ["dnn", "random_forest", "xgboost"])
```

## Adding a new dataset (e.g. CICIDS2017)

1. Add an entry to `config.DATASETS` following the `nsl_kdd` pattern
   (paths, attack map, categories, expected row counts).
2. Add a loader function in `data.py` following `_load_nsl_kdd_raw` /
   the body of `load_and_preprocess`, including the same row-count
   validation, category-completeness check, NaN check, and train/test
   overlap audit, don't skip these for a new dataset just because they
   passed for NSL-KDD.
3. Update `load_and_preprocess`'s dispatch logic to call the new loader.

## Files

- `config.py` — all pre-registered parameters, dataset/architecture
  registries, results schema version. Changing parameters after seeing
  results should be documented, not done silently.
- `data.py` — dataset loading with row-count validation,
  category-completeness checking, NaN/Inf detection, and a train/test
  feature-overlap audit (found 2.95% overlap on NSL-KDD, worth stating
  explicitly in the paper's limitations section).
- `model.py` — all three architectures behind a consistent
  `fit`/`predict` interface, plus full-determinism seeding. Includes the
  documented XGBoost subsample fix.
- `train.py` — single `(architecture, seed)` training and evaluation,
  crash-safe incremental CSV writing, schema-version stamping, loud
  failure behavior with full context on error.
- `run_pilot.py` — Stage 1 runner, resumable.
- `run_matrix.py` — Stage 2 general runner across every enabled
  dataset/architecture/seed combination, resumable.
- `stats_analysis.py` — real bootstrap variance decomposition (validated
  against synthetic ground truth), Benjamini-Hochberg-corrected Levene's
  tests, effect sizes, Clopper-Pearson intervals for small classes,
  seed-adequacy analysis, cross-architecture comparison, duplicate-run
  and schema-consistency checks on load.
- `environment_capture.py` — records exact software versions, hardware,
  git commit hash (and working-tree-dirty flag), active hyperparameters,
  and data file checksums for every run, written to
  `environment_logs/env_<timestamp>.json`.

## Before running the full-scale experiments for the paper

1. Commit all code changes (`git status` clean) so the git commit hash in
   each environment log actually reflects the code that produced the
   results, not just that the pilot passed.
2. If running the DNN on a GPU node, check `gpu_info.cuda_available` in
   the environment log, don't assume the GPU is being used, an
   unexplained slowdown once suggested a real configuration problem on
   a shared cluster node.
3. Re-run the pilot's analysis and confirm the same pattern holds before
   committing to the full 40-seed matrix across all three architectures.