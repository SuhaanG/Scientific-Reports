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
RESULTS_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Seed configuration
# ---------------------------------------------------------------------------
PILOT_SEEDS = list(range(10))
FULL_SEEDS = list(range(40))

BOOTSTRAP_RNG_SEED = 999_999

# ---------------------------------------------------------------------------
# Dataset registry: NSL-KDD
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

NSL_KDD_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]

# ---------------------------------------------------------------------------
# Dataset registry: CSE-CIC-IDS2018
# ---------------------------------------------------------------------------
# 7-category taxonomy deliberately matches the ACAFS paper (Mchina &
# Sinde, Frontiers in Big Data 2026) so results are directly comparable
# to a specific named prior study, not just a generic "second dataset."
# Label spellings below match the dataset's documented raw values,
# including its known inconsistent capitalization/spacing, verify these
# exactly against your real file's unique label values before trusting
# this mapping, don't assume it's complete until confirmed.
CIC_IDS2018_ATTACK_MAP = {
    "benign": "normal",
    "dos attacks-goldeneye": "dos",
    "dos attacks-slowloris": "dos",
    "dos attacks-slowhttptest": "dos",
    "dos attacks-hulk": "dos",
    "ddos attacks-loic-http": "ddos",
    "ddos attack-loic-udp": "ddos",
    "ddos attack-hoic": "ddos",
    "brute force -web": "web_attacks",
    "brute force -xss": "web_attacks",
    "sql injection": "web_attacks",
    "infilteration": "infiltration",
    "bot": "botnet",
    "ftp-bruteforce": "bruteforce",
    "ssh-bruteforce": "bruteforce",
}

CIC_IDS2018_CATEGORIES = [
    "normal", "bruteforce", "dos", "web_attacks", "infiltration", "botnet", "ddos",
]

# Target subsample size (stratified, preserving original class proportions).
# The full dataset is ~16M rows, roughly 128x NSL-KDD's size; using it in
# full would make 40-seed x 3-architecture training impractically slow.
# This subsample keeps runtime comparable to the NSL-KDD experiments.
CIC_IDS2018_TARGET_TRAIN_ROWS = 200_000
CIC_IDS2018_TARGET_TEST_ROWS = 40_000
CIC_IDS2018_SUBSAMPLE_RNG_SEED = 777_777  # fixed, deliberately separate
                                            # from model training seeds

DATASETS = {
    "nsl_kdd": {
        "train_path": os.path.join(DATA_DIR, "KDDTrain+.txt"),
        "test_path": os.path.join(DATA_DIR, "KDDTest+.txt"),
        "attack_map": NSL_KDD_ATTACK_MAP,
        "categories": NSL_KDD_CATEGORIES,
        "expected_train_rows": 125973,
        "expected_test_rows": 22544,
    },
    "cse_cic_ids2018": {
        "train_path": os.path.join(DATA_DIR, "CSECICIDS2018_train.csv"),
        "test_path": os.path.join(DATA_DIR, "CSECICIDS2018_test.csv"),
        "attack_map": CIC_IDS2018_ATTACK_MAP,
        "categories": CIC_IDS2018_CATEGORIES,
        # Left as None until prepare_cicids2018.py actually produces the
        # files and reports real row counts, data.py treats None as
        # "skip the exact row-count check" rather than silently assuming
        # a number that hasn't been confirmed.
        "expected_train_rows": 200000,
        "expected_test_rows": 39999,
    },
}

# Kept for backward compatibility with any code still referencing the old
# flat name; new code should use DATASETS[<name>]["categories"] instead.
ATTACK_CATEGORIES = NSL_KDD_CATEGORIES

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

RANDOM_STATE_FOR_SPLIT = 42