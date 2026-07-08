"""
Runs the Stage 1 pilot: 10 seeds, one architecture (feedforward DNN),
NSL-KDD only.

This is a hard gate, not just a first draft of the full experiment.
Per the project plan: do not proceed to the full 30-40 seed x 3 architecture
x multi-dataset matrix until this pilot's output has been checked with
stats_analysis.py and the per-attack-category instability (or lack of it)
has been confirmed to survive the bootstrap noise decomposition.

Usage:
    python run_pilot.py
"""

import os
import config
from data import load_and_preprocess, audit_dataset
from train import train_one_seed

RESULTS_CSV = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_summary.csv")
PER_INSTANCE_CSV = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_per_instance.csv")


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Loading and preprocessing NSL-KDD...")
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess()
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    audit_dataset(y_train, y_test)

    print(f"\nStarting pilot run: {len(config.PILOT_SEEDS)} seeds")
    print(f"Results will be written incrementally to: {RESULTS_CSV}")
    print(f"Per-instance predictions written to: {PER_INSTANCE_CSV}\n")

    for seed in config.PILOT_SEEDS:
        train_one_seed(
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            categories=config.ATTACK_CATEGORIES,
            results_csv_path=RESULTS_CSV,
            per_instance_csv_path=PER_INSTANCE_CSV,
        )

    print("\nPilot run complete.")
    print("Next step: run stats_analysis.py on the pilot results before")
    print("deciding whether to scale up to the full seed/architecture/dataset matrix.")


if __name__ == "__main__":
    main()
