"""
Configuration for the IDS seed-variance study.

PRE-REGISTRATION NOTE:
These parameters are locked before running any experiment stage and
should not change after seeing results. If a parameter genuinely must
change after the pilot, document why and when in a dated commit message,
undocumented post-hoc changes undermine the pre-registration argument in
the paper's Methods section.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
ENV_LOG_DIR = os.path.join(PROJECT_ROOT, "environment_logs")

# ---------------------------------------------------------------------------
# Results schema version
# ---------------------------------------------------------------------------
# Bump this any time the CSV column structure in train.py changes. Stamped
# into every environment log so pilot-stage and full-stage result files
# can never be silently mixed during analysis if the schema diverged
# between them.
RESULTS_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Seed configuration
# ---------------------------------------------------------------------------
PILOT_SEEDS = list(range(10))
FULL_SEEDS = list(range(40))

BOOTSTRAP_RNG_SEED = 999_999  # deliberately separate from training seeds

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
NSL_KDD_ATTACK_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r",
    "sqlattack": "u2r", "xterm": "u2r",
}

ATTACK_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]

DATASETS = {
    "nsl_kdd": {
        "train_path": os.path.join(DATA_DIR, "KDDTrain+.txt"),
        "test_path": os.path.join(DATA_DIR, "KDDTest+.txt"),
        "attack_map": NSL_KDD_ATTACK_MAP,
        "categories": ATTACK_CATEGORIES,
        "expected_train_rows": 125973,
        "expected_test_rows": 22544,
    },
}

# ---------------------------------------------------------------------------
# Architecture hyperparameters
# ---------------------------------------------------------------------------
ARCHITECTURES = ["dnn", "random_forest", "xgboost"]

DNN_HIDDEN_SIZES = [128, 64, 32]
DNN_DROPOUT = 0.2
DNN_LEARNING_RATE = 1e-3
DNN_BATCH_SIZE = 256
DNN_MAX_EPOCHS = 50
DNN_EARLY_STOP_PATIENCE = 5
DNN_VALIDATION_FRACTION = 0.15

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
# subsample/colsample MUST be < 1.0, otherwise random_state has nothing to
# actually randomize (every boosting round would deterministically use all
# rows and all features regardless of seed), which was caught by an
# explicit same-seed-vs-different-seed determinism test during development.
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# ---------------------------------------------------------------------------
# Statistical analysis parameters
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
SEED_SUBSET_CHECKPOINTS = [3, 5, 10, 20, 30, 40]
MULTIPLE_COMPARISON_METHOD = "fdr_bh"
SMALL_CLASS_SUPPORT_THRESHOLD = 500

RANDOM_STATE_FOR_SPLIT = 42  # fixed validation-split reference seed only,
                              # never used as a model training seed