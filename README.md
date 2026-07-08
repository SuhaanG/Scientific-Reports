# IDS Seed Variance Study — Pilot Code

Stage 1 pilot: 10 seeds, one architecture (feedforward DNN), NSL-KDD only.
Do not scale up to the full matrix (30-40 seeds x 3 architectures x
multiple datasets) until this pilot's results have been checked with
`stats_analysis.py` and discussed with the team.

## Setup

1. Install dependencies: `pip install -r requirements.txt --break-system-packages`
   (adjust for your environment; on Delta, use whatever module/env system
   is standard there).
2. Download `KDDTrain+.txt` and `KDDTest+.txt` from the official NSL-KDD
   source (Canadian Institute for Cybersecurity) and place them in `data/`.
   Do not substitute a random GitHub mirror without first confirming it
   matches the official file (row counts, 41 features).

## Running the pilot

```bash
python run_pilot.py
```

This will:
- Load and preprocess NSL-KDD (fit encoder/scaler on train only, avoiding
  leakage into test).
- Print a data audit (class distribution, flags small classes).
- Train the DNN for each of the 10 pilot seeds, reseeding every RNG
  (Python, NumPy, PyTorch CPU/GPU) before each run.
- Append aggregate + per-attack-category metrics to
  `results/pilot_nsl_kdd_dnn_summary.csv` after every seed (crash-safe,
  not held in memory).
- Append per-instance predictions to
  `results/pilot_nsl_kdd_dnn_per_instance.csv` for later bootstrap
  resampling.

## Analyzing the pilot

```bash
python stats_analysis.py
```

This prints:
- Between-seed variance decomposition for aggregate accuracy and each
  attack category's recall.
- Seed-adequacy analysis: how the confidence interval width changes as
  you use the first 3, 5, or 10 seeds (extend to 20/30/40 only after the
  full-scale run).
- Clopper-Pearson exact intervals for any class with small median support
  (below 500 test instances), since naive bootstrap intervals are
  unreliable there.
- A go/no-go summary to guide the decision on whether to scale up.

## What this pilot does NOT yet include

- Random Forest / XGBoost architecture comparisons (add after the DNN
  pilot result is reviewed).
- CICIDS2017 / CSE-CIC-IDS2018 (NSL-KDD only for the pilot).
- The full 30-40 seed count (10 seeds only, by design, to keep the pilot
  fast and cheap before committing more compute).

## Files

- `config.py` — all pre-registered parameters (seed counts, architecture
  hyperparameters, statistical settings). Changing these after seeing
  results should be documented, not done silently.
- `data.py` — NSL-KDD loading, attack-category mapping, preprocessing.
- `model.py` — the DNN architecture and the full-determinism seeding
  function.
- `train.py` — single-seed training and evaluation, with per-category
  metrics and crash-safe incremental CSV writing.
- `run_pilot.py` — orchestrates the 10-seed pilot run.
- `stats_analysis.py` — variance decomposition, Clopper-Pearson intervals,
  bootstrap resampling, and the seed-adequacy analysis.
