"""
Data loading and preprocessing for NSL-KDD.

NSL-KDD does not ship with a header row. The 41 features plus the label
and a "difficulty" score are documented on the official NSL-KDD page
(Canadian Institute for Cybersecurity, University of New Brunswick).

Usage:
    Download KDDTrain+.txt and KDDTest+.txt from the official source and
    place them in the data/ directory before running anything. This script
    does NOT auto-download, since mirror URLs can go stale or introduce
    unknown preprocessing; verify you have the official files yourself.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config

# Column names, in order, per the official NSL-KDD documentation.
COLUMN_NAMES = [
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

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def _load_raw(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download KDDTrain+.txt / KDDTest+.txt "
            f"from the official NSL-KDD source and place them in the data/ "
            f"directory. Do not substitute a random mirror without checking "
            f"it matches the official file (row counts, column count)."
        )
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def _map_attack_category(label):
    label = label.strip().lower()
    if label not in config.NSL_KDD_ATTACK_MAP:
        raise ValueError(
            f"Unrecognized label '{label}' not in NSL_KDD_ATTACK_MAP. "
            f"This means either the data file is not standard NSL-KDD, "
            f"or the attack map in config.py is missing an entry. "
            f"Do not silently drop this row, fix the map."
        )
    return config.NSL_KDD_ATTACK_MAP[label]


def load_and_preprocess(train_path=None, test_path=None):
    """
    Loads NSL-KDD train/test, applies coarse attack-category mapping,
    one-hot encodes categorical features, and standardizes numeric features.

    Fitting (encoder, scaler) is done on TRAIN ONLY and applied to test,
    to avoid leakage, matching the standard practice flagged in the
    literature review as an important preprocessing safeguard.

    Returns:
        X_train, y_train_category, X_test, y_test_category, feature_names
        y_*_category are the coarse category strings: normal/dos/probe/r2l/u2r
    """
    train_df = _load_raw(train_path or config.NSL_KDD_TRAIN_PATH)
    test_df = _load_raw(test_path or config.NSL_KDD_TEST_PATH)

    # Drop the difficulty column, not a feature.
    train_df = train_df.drop(columns=["difficulty"])
    test_df = test_df.drop(columns=["difficulty"])

    # Map fine-grained labels to coarse attack categories.
    train_df["category"] = train_df["label"].apply(_map_attack_category)
    test_df["category"] = test_df["label"].apply(_map_attack_category)

    y_train = train_df["category"].values
    y_test = test_df["category"].values

    train_features = train_df.drop(columns=["label", "category"])
    test_features = test_df.drop(columns=["label", "category"])

    # One-hot encode categorical columns, fit on train only.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_features[CATEGORICAL_COLS])

    train_cat_encoded = encoder.transform(train_features[CATEGORICAL_COLS])
    test_cat_encoded = encoder.transform(test_features[CATEGORICAL_COLS])
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS)

    numeric_cols = [c for c in train_features.columns if c not in CATEGORICAL_COLS]
    train_numeric = train_features[numeric_cols].values.astype(np.float64)
    test_numeric = test_features[numeric_cols].values.astype(np.float64)

    # Standardize numeric features, fit on train only.
    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)

    X_train = np.concatenate([train_numeric_scaled, train_cat_encoded], axis=1)
    X_test = np.concatenate([test_numeric_scaled, test_cat_encoded], axis=1)

    feature_names = numeric_cols + list(cat_feature_names)

    return X_train, y_train, X_test, y_test, feature_names


def audit_dataset(y_train, y_test):
    """
    Prints class distribution for train/test. Run this once and manually
    confirm it matches published NSL-KDD statistics before trusting any
    downstream results, per the data-audit step in the project plan.
    """
    print("=== Train category distribution ===")
    for cat in config.ATTACK_CATEGORIES:
        count = int((y_train == cat).sum())
        print(f"  {cat}: {count}")

    print("=== Test category distribution ===")
    for cat in config.ATTACK_CATEGORIES:
        count = int((y_test == cat).sum())
        print(f"  {cat}: {count}")
        if count < 300:
            print(
                f"    WARNING: '{cat}' has only {count} test instances. "
                f"This is a small-n class, use Clopper-Pearson exact "
                f"intervals for this class's recall, not naive bootstrap "
                f"percentile intervals, per the statistical plan."
            )


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features after encoding: {len(feature_names)}")
    audit_dataset(y_train, y_test)
