"""
Configuration for the IDS seed-variance pilot study.

PRE-REGISTRATION NOTE:
These parameters (seed count, architecture, hyperparameters) should be
locked before running the pilot and not changed after seeing results.
If you change something here after looking at pilot output, document why
in your lab notebook / commit message, since silent changes undermine the
pre-registration argument in the paper.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

NSL_KDD_TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
NSL_KDD_TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

# ---------------------------------------------------------------------------
# Pilot experiment parameters (Stage 1: NSL-KDD only, one architecture)
# ---------------------------------------------------------------------------
PILOT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 seeds for the pilot

# Full-scale seeds (only use after the pilot passes the go/no-go check)
FULL_SEEDS = list(range(40))  # 40 seeds, adjust after confirming compute budget

# Architecture hyperparameters for the feedforward DNN (Stage 1 architecture)
DNN_HIDDEN_SIZES = [128, 64, 32]
DNN_DROPOUT = 0.2
DNN_LEARNING_RATE = 1e-3
DNN_BATCH_SIZE = 256
DNN_EPOCHS = 50
DNN_EARLY_STOP_PATIENCE = 5

# Attack category mapping for NSL-KDD (41 features + label + difficulty)
# Maps the ~39 fine-grained attack labels to the 4 standard coarse categories
# used throughout the IDS literature (DoS, Probe, R2L, U2R) plus 'normal'.
NSL_KDD_ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L (Remote to Local)
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l",
    # U2R (User to Root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r",
    "sqlattack": "u2r", "xterm": "u2r",
}

ATTACK_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]

# ---------------------------------------------------------------------------
# Statistical analysis parameters
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
SEED_SUBSET_CHECKPOINTS = [3, 5, 10, 20, 30, 40]  # for the seed-adequacy analysis

RANDOM_STATE_FOR_SPLIT = 42  # only used if we need a fixed reference split;
                             # NOT used as a model training seed
