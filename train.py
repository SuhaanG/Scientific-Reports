"""
Trains and evaluates the IDS classifier for a single random seed.

Design notes:
- Results are appended to a CSV file immediately after each seed finishes,
  not held in memory and written at the end. If a job crashes at seed 7 of
  10, you keep seeds 0-6's results rather than losing everything.
- Per-attack-category metrics (not just aggregate accuracy) are computed
  and saved for every seed, since the per-category breakdown is the core
  measurement this whole project is built around.
"""

import os
import csv
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

import config
from model import IDSClassifier, set_full_determinism

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _category_to_index(categories):
    return {cat: i for i, cat in enumerate(categories)}


def compute_per_category_metrics(y_true, y_pred, categories):
    """
    Returns a dict: {category: {'recall': ..., 'precision': ..., 'support': ...,
    'false_negatives': ..., 'true_positives': ...}}

    Recall here is computed one-vs-rest per category, which is what you
    need for the "does the model miss this specific attack type" question,
    not a multi-class macro-average that hides category-level detail.
    """
    cat_to_idx = _category_to_index(categories)
    y_true_idx = np.array([cat_to_idx[c] for c in y_true])
    y_pred_idx = np.array([cat_to_idx[c] for c in y_pred])

    results = {}
    for cat, idx in cat_to_idx.items():
        true_positive = int(np.sum((y_true_idx == idx) & (y_pred_idx == idx)))
        false_negative = int(np.sum((y_true_idx == idx) & (y_pred_idx != idx)))
        false_positive = int(np.sum((y_true_idx != idx) & (y_pred_idx == idx)))
        support = int(np.sum(y_true_idx == idx))

        recall = true_positive / support if support > 0 else float("nan")
        denom = true_positive + false_positive
        precision = true_positive / denom if denom > 0 else float("nan")

        results[cat] = {
            "recall": recall,
            "precision": precision,
            "support": support,
            "true_positives": true_positive,
            "false_negatives": false_negative,
        }
    return results


def train_one_seed(seed, X_train, y_train, X_test, y_test, categories,
                    results_csv_path, per_instance_csv_path):
    """
    Trains the DNN for one seed, evaluates on the test set, appends
    aggregate + per-category metrics to results_csv_path, and appends
    per-instance predictions to per_instance_csv_path (needed later for
    bootstrap resampling of the test set).
    """
    set_full_determinism(seed)

    cat_to_idx = _category_to_index(categories)
    y_train_idx = np.array([cat_to_idx[c] for c in y_train])
    y_test_idx = np.array([cat_to_idx[c] for c in y_test])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_idx, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    # NOTE: generator is seeded too, so shuffling order is part of what
    # varies (and is controlled) across seeds, matching Dodge et al.'s
    # finding that data order is a distinct contributor to seed variance.
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.DNN_BATCH_SIZE, shuffle=True,
        generator=generator,
    )

    model = IDSClassifier(
        input_dim=X_train.shape[1], num_classes=len(categories)
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.DNN_LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    start_time = time.time()
    for epoch in range(config.DNN_EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_dataset)

        # Simple early stopping on training loss plateau.
        # (A held-out validation split can be added later if needed; kept
        # simple here since the pilot's goal is the variance measurement,
        # not squeezing out maximum accuracy.)
        if epoch_loss < best_val_loss - 1e-5:
            best_val_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.DNN_EARLY_STOP_PATIENCE:
            break

    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds_idx = torch.argmax(logits, dim=1).cpu().numpy()

    idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
    y_pred = np.array([idx_to_cat[i] for i in preds_idx])

    aggregate_accuracy = float(np.mean(preds_idx == y_test_idx))

    per_category = compute_per_category_metrics(y_test, y_pred, categories)

    # --- Append aggregate + per-category summary row ---
    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["seed", "aggregate_accuracy", "train_time_sec", "epochs_run"]
            for cat in categories:
                header += [f"{cat}_recall", f"{cat}_precision", f"{cat}_support"]
            writer.writerow(header)
        row = [seed, aggregate_accuracy, round(train_time, 2), epoch + 1]
        for cat in categories:
            row += [
                per_category[cat]["recall"],
                per_category[cat]["precision"],
                per_category[cat]["support"],
            ]
        writer.writerow(row)

    # --- Append per-instance predictions (needed for bootstrap resampling) ---
    instance_file_exists = os.path.exists(per_instance_csv_path)
    with open(per_instance_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not instance_file_exists:
            writer.writerow(["seed", "instance_id", "true_category", "pred_category", "correct"])
        for i in range(len(y_test)):
            writer.writerow([seed, i, y_test[i], y_pred[i], int(y_test[i] == y_pred[i])])

    print(
        f"[seed {seed}] done in {train_time:.1f}s, "
        f"{epoch + 1} epochs, aggregate_accuracy={aggregate_accuracy:.4f}"
    )

    return aggregate_accuracy, per_category
