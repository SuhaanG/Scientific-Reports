"""
General matrix runner: loops over every (dataset, architecture, seed)
combination currently enabled in config.py.

IDEMPOTENT / RESUMABLE BY DESIGN, same as run_pilot.py: already-completed
(dataset, architecture, seed) combinations are skipped, not rerun. This
matters more here than in the pilot script, since a full matrix run
(potentially 40 seeds x 3 architectures x multiple datasets) is the run
most likely to actually get interrupted partway through.

Usage:
    python run_matrix.py --seeds pilot     # 10 seeds (default, safe)
    python run_matrix.py --seeds full      # 40 seeds (only after pilot review)
"""

import os
import argparse
import pandas as pd
import config
from data import load_and_preprocess, audit_dataset
from train import train_one_run
from environment_capture import capture_environment


def _get_already_completed_seeds(results_csv_path, dataset_name, architecture):
    if not os.path.exists(results_csv_path):
        return set()
    df = pd.read_csv(results_csv_path)
    if not all(c in df.columns for c in ("dataset", "architecture", "seed")):
        return set()
    subset = df[(df["dataset"] == dataset_name) & (df["architecture"] == architecture)]
    return set(subset["seed"].unique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", choices=["pilot", "full"], default="pilot",
        help="Use the 10-seed pilot list or the 40-seed full list. "
             "Do not use 'full' until the pilot has been reviewed.",
    )
    args = parser.parse_args()
    seeds = config.PILOT_SEEDS if args.seeds == "pilot" else config.FULL_SEEDS

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    total_combinations = len(config.DATASETS) * len(config.ARCHITECTURES) * len(seeds)
    print(f"Matrix scope: {len(config.DATASETS)} dataset(s) x "
          f"{len(config.ARCHITECTURES)} architecture(s) x {len(seeds)} seed(s) "
          f"= {total_combinations} total runs.\n")

    for dataset_name in config.DATASETS:
        print(f"\n{'#'*78}\nDATASET: {dataset_name}\n{'#'*78}")

        print("Capturing environment and data provenance...")
        capture_environment(dataset_name=dataset_name)

        X_train, y_train, X_test, y_test, feature_names = load_and_preprocess(dataset_name)
        categories = config.DATASETS[dataset_name]["categories"]
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        audit_dataset(y_train, y_test, categories)

        results_csv = os.path.join(config.RESULTS_DIR, f"{dataset_name}_matrix_summary.csv")
        per_instance_csv = os.path.join(config.RESULTS_DIR, f"{dataset_name}_matrix_per_instance.csv")

        for architecture in config.ARCHITECTURES:
            already_done = _get_already_completed_seeds(results_csv, dataset_name, architecture)
            remaining_seeds = [s for s in seeds if s not in already_done]

            print(f"\n--- Architecture: {architecture} ---")
            if already_done:
                print(
                    f"{len(already_done)} seed(s) already completed: "
                    f"{sorted(already_done)}. Skipping these."
                )
            if not remaining_seeds:
                print(f"All {len(seeds)} seed(s) already completed for "
                      f"{dataset_name}/{architecture}. Nothing to do here.")
                continue

            print(f"Running {len(remaining_seeds)} remaining seed(s)...")
            for seed in remaining_seeds:
                train_one_run(
                    architecture=architecture,
                    seed=seed,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    categories=categories,
                    results_csv_path=results_csv,
                    per_instance_csv_path=per_instance_csv,
                    dataset_name=dataset_name,
                )

    print("\nMatrix run complete. Run stats_analysis.py on each dataset's results.")


if __name__ == "__main__":
    main()