"""
Trains and evaluates a single (architecture, seed) combination on a given
dataset, computing aggregate and per-attack-category metrics.

Design principles:
- Results are appended to CSV immediately after each run, not held in
  memory, so a crash partway through a long matrix doesn't lose completed
  runs.
- Per-instance predictions are also saved, needed for bootstrap resampling.
- Every RNG is reseeded before each run, for all three architectures.
- A failed run raises loudly with full context (dataset, architecture,
  seed) rather than being silently caught and skipped, a silently missing
  seed in a 40-row results file is easy to miss and would corrupt the
  variance analysis without any visible sign.
- schema_version is written into every results row, not just the
  environment log, so a schema mismatch is visible directly in the data.
"""

import os
import csv
import time
import random
import numpy as np

import config
from model import build_model, set_full_determinism


def _category_to_index(categories):
    return {cat: i for i, cat in enumerate(categories)}


def compute_per_category_metrics(y_true, y_pred, categories):
    cat_to_idx = _category_to_index(categories)
    y_true_idx = np.array([cat_to_idx[c] for c in y_true])
    y_pred_idx = np.array([cat_to_idx[c] for c in y_pred])

    results = {}
    for cat, idx in cat_to_idx.items():
        true_positive = int(np.sum((y_true_idx == idx) & (y_pred_idx == idx)))
        false_positive = int(np.sum((y_true_idx != idx) & (y_pred_idx == idx)))
        support = int(np.sum(y_true_idx == idx))

        recall = true_positive / support if support > 0 else float("nan")
        denom = true_positive + false_positive
        precision = true_positive / denom if denom > 0 else float("nan")

        results[cat] = {"recall": recall, "precision": precision, "support": support}
    return results


def train_one_run(architecture, seed, X_train, y_train, X_test, y_test,
                   categories, results_csv_path, per_instance_csv_path,
                   dataset_name):
    """
    Trains and evaluates ONE (architecture, seed) combination.
    Appends a summary row and per-instance predictions to their
    respective CSVs. Raises with full context on failure rather than
    silently skipping.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"[{dataset_name}|{architecture}|seed {seed}] X_train has "
            f"{len(X_train)} rows but y_train has {len(y_train)}. "
            f"Data pipeline bug, stopping before training on misaligned data."
        )
    if len(X_test) != len(y_test):
        raise ValueError(
            f"[{dataset_name}|{architecture}|seed {seed}] X_test has "
            f"{len(X_test)} rows but y_test has {len(y_test)}. "
            f"Data pipeline bug, stopping before evaluating on misaligned data."
        )

    try:
        set_full_determinism(seed)
        random.seed(seed)
        np.random.seed(seed)

        cat_to_idx = _category_to_index(categories)
        y_train_idx = np.array([cat_to_idx[c] for c in y_train])
        y_test_idx = np.array([cat_to_idx[c] for c in y_test])

        model = build_model(architecture, input_dim=X_train.shape[1], num_classes=len(categories))

        start_time = time.time()
        model.fit(X_train, y_train_idx, seed)
        train_time = time.time() - start_time

        preds_idx = model.predict(X_test)

        idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
        y_pred = np.array([idx_to_cat[i] for i in preds_idx])

        aggregate_accuracy = float(np.mean(preds_idx == y_test_idx))
        per_category = compute_per_category_metrics(y_test, y_pred, categories)
        epochs_run = getattr(model, "epochs_run", None)

    except Exception as e:
        raise RuntimeError(
            f"Run FAILED at [{dataset_name} | {architecture} | seed {seed}]: "
            f"{type(e).__name__}: {e}. Stopping here rather than silently "
            f"skipping this seed, a missing seed in the results file would "
            f"corrupt the variance analysis without any visible sign."
        ) from e

    # --- Append summary row ---
    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                "schema_version", "dataset", "architecture", "seed",
                "aggregate_accuracy", "train_time_sec", "epochs_run",
            ]
            for cat in categories:
                header += [f"{cat}_recall", f"{cat}_precision", f"{cat}_support"]
            writer.writerow(header)
        row = [
            config.RESULTS_SCHEMA_VERSION, dataset_name, architecture, seed,
            aggregate_accuracy, round(train_time, 2), epochs_run,
        ]
        for cat in categories:
            row += [
                per_category[cat]["recall"],
                per_category[cat]["precision"],
                per_category[cat]["support"],
            ]
        writer.writerow(row)

    # --- Append per-instance predictions ---
    instance_file_exists = os.path.exists(per_instance_csv_path)
    with open(per_instance_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not instance_file_exists:
            writer.writerow([
                "schema_version", "dataset", "architecture", "seed",
                "instance_id", "true_category", "pred_category", "correct",
            ])
        for i in range(len(y_test)):
            writer.writerow([
                config.RESULTS_SCHEMA_VERSION, dataset_name, architecture, seed, i,
                y_test[i], y_pred[i], int(y_test[i] == y_pred[i]),
            ])

    print(
        f"[{dataset_name} | {architecture} | seed {seed}] done in "
        f"{train_time:.1f}s, epochs_run={epochs_run}, "
        f"aggregate_accuracy={aggregate_accuracy:.4f}"
    )

    return aggregate_accuracy, per_category