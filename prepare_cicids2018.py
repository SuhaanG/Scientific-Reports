"""
One-time preparation script for CSE-CIC-IDS2018.

Unlike NSL-KDD, this dataset does not ship as a single clean file with an
official train/test split, it's distributed as multiple daily CSVs, with
well-documented real-world messiness: inconsistent column name spacing
(e.g. " Label" with a leading space), and Infinity/NaN values in
Flow Bytes/s and Flow Packets/s from division-by-zero during feature
extraction. This script handles all of that once and writes two clean,
flat output files that data.py's loader can then read simply.

DELIBERATE METHODOLOGICAL CHOICE, stated explicitly for the paper's
Methods section: no SMOTE or other class-balancing is applied here,
unlike the ACAFS paper's preprocessing. This study measures seed-driven
instability under NATURAL class imbalance, consistent with how NSL-KDD
is handled elsewhere in this codebase. Applying synthetic oversampling
would introduce a different confound and make the two datasets
inconsistent within the same paper.

Usage:
    python prepare_cicids2018.py
"""

import os
import glob
import numpy as np
import pandas as pd

import config

RAW_DATA_DIR = os.path.join(config.DATA_DIR, "cicids2018_raw")
OUTPUT_TRAIN_PATH = config.DATASETS["cse_cic_ids2018"]["train_path"]
OUTPUT_TEST_PATH = config.DATASETS["cse_cic_ids2018"]["test_path"]

# Columns known/suspected to be identifiers or leakage risks rather than
# genuine traffic features. Matched case-insensitively, substring match
# on the "ip" ones since exact column naming isn't confirmed yet.
COLUMNS_TO_DROP_EXACT = {"timestamp", "flow id", "flow_id"}
COLUMNS_TO_DROP_SUBSTRING = ["ip"]  # e.g. "Src IP", "Dst IP", "Source IP"


def _clean_column_names(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def _identify_columns_to_drop(columns):
    to_drop = []
    for col in columns:
        col_lower = col.strip().lower()
        if col_lower in COLUMNS_TO_DROP_EXACT:
            to_drop.append(col)
        elif any(sub in col_lower for sub in COLUMNS_TO_DROP_SUBSTRING) and col_lower != "label":
            to_drop.append(col)
    return to_drop


def load_all_raw_files():
    csv_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DATA_DIR}. Download the dataset "
            f"there first (see the aws s3 cp command)."
        )

    print(f"Found {len(csv_paths)} raw file(s):")
    for p in csv_paths:
        print(f"  {p}")

    frames = []
    for path in csv_paths:
        print(f"Loading {path}...")
        df = pd.read_csv(path, low_memory=False)
        df = _clean_column_names(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined raw shape: {combined.shape}")
    return combined


def check_label_mapping(df, attack_map):
    """
    Collects ALL unmapped label values before failing, rather than
    erroring on the first one, since fixing one label at a time on a
    multi-gigabyte, multi-file dataset with a slow reload cycle each time
    would be a bad use of your time.
    """
    if "Label" not in df.columns:
        raise ValueError(
            f"No 'Label' column found after cleaning. Columns present: "
            f"{list(df.columns)}. This dataset's label column name may "
            f"differ from expected, check the raw file's actual header."
        )

    raw_labels = df["Label"].astype(str).str.strip().str.lower().unique()
    unmapped = [l for l in raw_labels if l not in attack_map]

    if unmapped:
        raise ValueError(
            f"Found {len(unmapped)} label value(s) in the data with no "
            f"entry in config.CIC_IDS2018_ATTACK_MAP: {unmapped}\n"
            f"Add these to the map in config.py before proceeding. Do not "
            f"guess or silently drop them, an incomplete map would corrupt "
            f"category-level results in a way that's hard to notice later."
        )

    print(f"All {len(raw_labels)} unique raw label values map cleanly "
          f"to the 7-category taxonomy.")


def clean_and_map(df, attack_map):
    columns_to_drop = _identify_columns_to_drop(df.columns)
    if columns_to_drop:
        print(f"Dropping identifier/leakage-risk columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    check_label_mapping(df, attack_map)

    df["category"] = df["Label"].astype(str).str.strip().str.lower().map(attack_map)
    df = df.drop(columns=["Label"])

    feature_cols = [c for c in df.columns if c != "category"]

    # Known issue: Flow Bytes/s and Flow Packets/s can be Infinity from
    # division by zero during feature extraction. Coerce to numeric,
    # report and drop any resulting NaN/Inf rows rather than silently
    # letting them corrupt training.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    n_after = len(df)
    n_dropped = n_before - n_after
    print(
        f"Dropped {n_dropped} of {n_before} rows ({n_dropped/n_before*100:.2f}%) "
        f"due to NaN/Inf values in feature columns (a documented, known "
        f"issue in this dataset, not a bug in this script)."
    )

    return df


def stratified_split_and_subsample(df, categories, target_train_rows,
                                    target_test_rows, rng_seed):
    """
    Stratified train/test split (this dataset has no official split,
    unlike NSL-KDD) followed by stratified subsampling to the target
    sizes, preserving NATURAL class proportions throughout, no balancing.
    """
    rng = np.random.default_rng(rng_seed)

    train_frames = []
    test_frames = []

    total_rows = len(df)
    for cat in categories:
        cat_df = df[df["category"] == cat]
        n_cat = len(cat_df)
        if n_cat == 0:
            print(f"WARNING: category '{cat}' has zero rows in the raw "
                  f"data. Check whether the expected attack file was "
                  f"actually downloaded.")
            continue

        cat_proportion = n_cat / total_rows
        cat_target_train = max(1, int(round(target_train_rows * cat_proportion)))
        cat_target_test = max(1, int(round(target_test_rows * cat_proportion)))

        shuffled_idx = rng.permutation(n_cat)
        cat_df_shuffled = cat_df.iloc[shuffled_idx]

        n_take = min(n_cat, cat_target_train + cat_target_test)
        if n_take < cat_target_train + cat_target_test:
            print(
                f"WARNING: category '{cat}' has only {n_cat} rows available, "
                f"fewer than the {cat_target_train + cat_target_test} "
                f"requested (train+test). Using all {n_cat} available, "
                f"split proportionally. This will make this category's "
                f"final count smaller than planned, report the ACTUAL "
                f"count, don't assume the target was hit."
            )
            split_point = int(n_cat * (cat_target_train / (cat_target_train + cat_target_test)))
        else:
            split_point = cat_target_train

        selected = cat_df_shuffled.iloc[:n_take]
        train_frames.append(selected.iloc[:split_point])
        test_frames.append(selected.iloc[split_point:])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)
    test_df = test_df.iloc[rng.permutation(len(test_df))].reset_index(drop=True)

    return train_df, test_df


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    combined = load_all_raw_files()
    cleaned = clean_and_map(combined, config.CIC_IDS2018_ATTACK_MAP)

    print("\nCategory distribution in cleaned data:")
    print(cleaned["category"].value_counts())

    train_df, test_df = stratified_split_and_subsample(
        cleaned,
        config.CIC_IDS2018_CATEGORIES,
        config.CIC_IDS2018_TARGET_TRAIN_ROWS,
        config.CIC_IDS2018_TARGET_TEST_ROWS,
        config.CIC_IDS2018_SUBSAMPLE_RNG_SEED,
    )

    print(f"\nFinal train shape: {train_df.shape}")
    print("Train category distribution:")
    print(train_df["category"].value_counts())

    print(f"\nFinal test shape: {test_df.shape}")
    print("Test category distribution:")
    print(test_df["category"].value_counts())

    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test_df.to_csv(OUTPUT_TEST_PATH, index=False)

    print(f"\nWrote {OUTPUT_TRAIN_PATH}")
    print(f"Wrote {OUTPUT_TEST_PATH}")
    print(
        "\nIMPORTANT: update config.py's 'expected_train_rows' and "
        "'expected_test_rows' for cse_cic_ids2018 with the exact numbers "
        "printed above, so data.py's row-count validation actually checks "
        "against a real, confirmed number instead of None."
    )


if __name__ == "__main__":
    main()