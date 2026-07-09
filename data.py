"""
Data loading and preprocessing, generalized across the dataset registry
in config.py. NSL-KDD is implemented now; additional datasets can be
added as further loader functions following this same pattern.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config

NSL_KDD_COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
NSL_KDD_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def _validate_row_count(df, expected, path, tolerance=0):
    actual = len(df)
    if expected is not None and abs(actual - expected) > tolerance:
        raise ValueError(
            f"Row count mismatch for {path}: expected {expected}, got "
            f"{actual}. This means the file does not match the official "
            f"dataset. Do not proceed, results from a mismatched file "
            f"are not trustworthy."
        )


def _load_nsl_kdd_raw(path, expected_rows):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the official NSL-KDD file "
            f"and place it there before running anything."
        )
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMN_NAMES)
    _validate_row_count(df, expected_rows, path)
    return df


def _map_attack_category(label, attack_map):
    label = label.strip().lower()
    if label not in attack_map:
        raise ValueError(
            f"Unrecognized label '{label}' not in the attack map for this "
            f"dataset. Do not silently drop this row, either the file is "
            f"non-standard or the map needs an added entry."
        )
    return attack_map[label]


def _check_category_completeness(y_train, y_test, expected_categories, split_name_pair):
    train_name, test_name = split_name_pair
    train_present = set(np.unique(y_train))
    test_present = set(np.unique(y_test))
    expected = set(expected_categories)

    missing_train = expected - train_present
    missing_test = expected - test_present

    if missing_train:
        raise ValueError(
            f"Expected categories {missing_train} do not appear at all in "
            f"{train_name}. This will silently break per-category metrics "
            f"for those classes (zero support), stop and investigate "
            f"before proceeding."
        )
    if missing_test:
        raise ValueError(
            f"Expected categories {missing_test} do not appear at all in "
            f"{test_name}. Per-category recall for these classes cannot "
            f"be computed at all if this is not fixed."
        )


def _check_no_nan_or_inf(X, array_name):
    if not np.all(np.isfinite(X)):
        n_bad = int(np.sum(~np.isfinite(X)))
        raise ValueError(
            f"{array_name} contains {n_bad} NaN or Inf values after "
            f"preprocessing. Likely cause: a numeric column had zero "
            f"variance in the training data, making StandardScaler "
            f"divide by zero. Do not proceed with corrupted features, "
            f"identify and handle the zero-variance column explicitly."
        )


def _audit_train_test_overlap(train_features_df, test_features_df):
    """
    Reports how many exact-duplicate feature rows exist between train and
    test. NSL-KDD was specifically created to fix KDD-99's duplicate-row
    problem, this checks that rather than assuming it, and gives you an
    honest, citable number for the Methods/Limitations section instead
    of an unverified assumption.
    """
    train_hashes = pd.util.hash_pandas_object(train_features_df, index=False)
    test_hashes = pd.util.hash_pandas_object(test_features_df, index=False)
    overlap_count = int(test_hashes.isin(set(train_hashes)).sum())
    overlap_fraction = overlap_count / len(test_features_df) if len(test_features_df) > 0 else 0.0
    print(
        f"Train/test overlap audit: {overlap_count} of {len(test_features_df)} "
        f"test rows ({overlap_fraction*100:.2f}%) have feature values "
        f"identical to some training row (hash-based check, not a "
        f"guarantee against hash collisions, but a strong signal)."
    )
    return overlap_count, overlap_fraction


def load_and_preprocess(dataset_name="nsl_kdd"):
    """
    Loads and preprocesses the named dataset from the registry.
    Fitting (encoder, scaler) is done on TRAIN ONLY and applied to test,
    avoiding leakage.

    Returns:
        X_train, y_train_category, X_test, y_test_category, feature_names
    """
    if dataset_name != "nsl_kdd":
        raise NotImplementedError(
            f"'{dataset_name}' is registered in config.DATASETS but its "
            f"loader is not yet implemented. Only nsl_kdd is implemented. "
            f"Add a loader following the NSL-KDD pattern before using "
            f"this dataset."
        )

    ds_config = config.DATASETS[dataset_name]
    train_df = _load_nsl_kdd_raw(ds_config["train_path"], ds_config["expected_train_rows"])
    test_df = _load_nsl_kdd_raw(ds_config["test_path"], ds_config["expected_test_rows"])

    attack_map = ds_config["attack_map"]
    expected_categories = ds_config["categories"]

    train_df = train_df.drop(columns=["difficulty"])
    test_df = test_df.drop(columns=["difficulty"])

    train_df["category"] = train_df["label"].apply(lambda l: _map_attack_category(l, attack_map))
    test_df["category"] = test_df["label"].apply(lambda l: _map_attack_category(l, attack_map))

    y_train = train_df["category"].values
    y_test = test_df["category"].values

    _check_category_completeness(
        y_train, y_test, expected_categories,
        (ds_config["train_path"], ds_config["test_path"]),
    )

    train_features = train_df.drop(columns=["label", "category"])
    test_features = test_df.drop(columns=["label", "category"])

    _audit_train_test_overlap(train_features, test_features)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_features[NSL_KDD_CATEGORICAL_COLS])

    train_cat_encoded = encoder.transform(train_features[NSL_KDD_CATEGORICAL_COLS])
    test_cat_encoded = encoder.transform(test_features[NSL_KDD_CATEGORICAL_COLS])
    cat_feature_names = list(encoder.get_feature_names_out(NSL_KDD_CATEGORICAL_COLS))

    numeric_cols = [c for c in train_features.columns if c not in NSL_KDD_CATEGORICAL_COLS]
    train_numeric = train_features[numeric_cols].values.astype(np.float64)
    test_numeric = test_features[numeric_cols].values.astype(np.float64)

    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)

    X_train = np.concatenate([train_numeric_scaled, train_cat_encoded], axis=1)
    X_test = np.concatenate([test_numeric_scaled, test_cat_encoded], axis=1)

    _check_no_nan_or_inf(X_train, "X_train")
    _check_no_nan_or_inf(X_test, "X_test")

    feature_names = numeric_cols + cat_feature_names

    return X_train, y_train, X_test, y_test, feature_names


def audit_dataset(y_train, y_test, categories):
    print("=== Train category distribution ===")
    for cat in categories:
        count = int((y_train == cat).sum())
        print(f"  {cat}: {count}")

    print("=== Test category distribution ===")
    for cat in categories:
        count = int((y_test == cat).sum())
        print(f"  {cat}: {count}")
        if count < config.SMALL_CLASS_SUPPORT_THRESHOLD:
            print(
                f"    WARNING: '{cat}' has only {count} test instances "
                f"(below the {config.SMALL_CLASS_SUPPORT_THRESHOLD}-instance "
                f"threshold). Use Clopper-Pearson exact intervals for this "
                f"class's recall, not naive bootstrap percentile intervals."
            )


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess("nsl_kdd")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features after encoding: {len(feature_names)}")
    audit_dataset(y_train, y_test, config.ATTACK_CATEGORIES)