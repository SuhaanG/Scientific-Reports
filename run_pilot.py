"""
Runs the Stage 1 pilot: 10 seeds, DNN architecture only, NSL-KDD only.

This is a hard gate. Do not proceed to the full-scale matrix (more seeds,
additional architectures, additional datasets) until this pilot's output
has been checked with stats_analysis.py and reviewed by the team.

IDEMPOTENT / RESUMABLE BY DESIGN: if results_csv already contains
completed runs for some seeds (e.g. from a previous run that was
interrupted), those seeds are skipped rather than rerun. This eliminates
the duplicate-seed risk at the source, rather than relying only on
stats_analysis.py's duplicate detection to catch it after the fact.

Usage:
    python run_pilot.py
"""

import os
import pandas as pd
import config
from data import load_and_preprocess, audit_dataset
from train import train_one_run
from environment_capture import capture_environment

DATASET_NAME = "nsl_kdd"
ARCHITECTURE = "dnn"

RESULTS_CSV = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_summary.csv")
PER_INSTANCE_CSV = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_per_instance.csv")


def _get_already_completed_seeds(results_csv_path, dataset_name, architecture):
    if not os.path.exists(results_csv_path):
        return set()
    df = pd.read_csv(results_csv_path)
    if not all(c in df.columns for c in ("dataset", "architecture", "seed")):
        return set()
    subset = df[(df["dataset"] == dataset_name) & (df["architecture"] == architecture)]
    return set(subset["seed"].unique())


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Capturing environment and data provenance before training...")
    capture_environment(dataset_name=DATASET_NAME)

    print(f"\nLoading and preprocessing {DATASET_NAME}...")
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess(DATASET_NAME)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    categories = config.DATASETS[DATASET_NAME]["categories"]
    audit_dataset(y_train, y_test, categories)

    already_done = _get_already_completed_seeds(RESULTS_CSV, DATASET_NAME, ARCHITECTURE)
    remaining_seeds = [s for s in config.PILOT_SEEDS if s not in already_done]

    if already_done:
        print(
            f"\nFound {len(already_done)} already-completed seed(s) in "
            f"{RESULTS_CSV}: {sorted(already_done)}. Skipping these, only "
            f"running the {len(remaining_seeds)} remaining seed(s). If this "
            f"is not what you want (e.g. you changed a hyperparameter and "
            f"want a clean rerun), delete or rename the existing results "
            f"file first."
        )

    if not remaining_seeds:
        print("\nAll pilot seeds already completed. Nothing to do.")
        print("Run stats_analysis.py to analyze the existing results.")
        return

    print(f"\nStarting pilot run: {len(remaining_seeds)} seed(s), architecture={ARCHITECTURE}")
    print(f"Results -> {RESULTS_CSV}")
    print(f"Per-instance predictions -> {PER_INSTANCE_CSV}\n")

    for seed in remaining_seeds:
        train_one_run(
            architecture=ARCHITECTURE,
            seed=seed,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            categories=categories,
            results_csv_path=RESULTS_CSV,
            per_instance_csv_path=PER_INSTANCE_CSV,
            dataset_name=DATASET_NAME,
        )

    print("\nPilot run complete.")
    print("Next: run stats_analysis.py on these results before deciding")
    print("whether to scale up to the full seed/architecture/dataset matrix.")


if __name__ == "__main__":
    main()