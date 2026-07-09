"""
Captures the exact software/hardware environment, hyperparameters, and
data provenance for a given run, and writes it to a timestamped JSON file.

Call capture_environment() once at the start of any training run and
save its output alongside your results. This is the record that backs
the paper's reproducibility statement, it should be enough on its own
for someone else to understand exactly what produced a given results file.
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
import datetime

import config


def _get_git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN (not a git repo, or git unavailable at capture time)"


def _get_git_dirty_flag():
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        pass
    return None


def _file_sha256(path, chunk_size=8192):
    if not os.path.exists(path):
        return None
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_installed_versions():
    packages = [
        "numpy", "pandas", "scikit-learn", "torch", "scipy",
        "statsmodels", "xgboost", "lightgbm",
    ]
    versions = {}
    for pkg in packages:
        try:
            module_name = "sklearn" if pkg == "scikit-learn" else pkg.replace("-", "_")
            module = __import__(module_name)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "NOT INSTALLED"
    return versions


def _get_gpu_info():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        info = {"cuda_available": cuda_available}
        if cuda_available:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
        return info
    except Exception as e:
        return {"error": str(e)}


def _get_hardware_info():
    """
    Records hostname, CPU count, and total RAM. Directly motivated by
    finding an unexplained 15x speed difference between two machines
    during development, having this recorded automatically makes a
    similar discrepancy diagnosable from the log itself rather than
    reconstructed after the fact.
    """
    info = {
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        info["total_ram_gb"] = "psutil not installed, not captured"
    return info


def _get_active_hyperparameters():
    """
    Captures the ACTUAL config.py values in effect at run time, not just
    the git commit hash. The commit hash only protects you if changes are
    committed; this is a direct, unambiguous record of what was actually
    used, regardless of git state.
    """
    return {
        "results_schema_version": config.RESULTS_SCHEMA_VERSION,
        "pilot_seeds_count": len(config.PILOT_SEEDS),
        "full_seeds_count": len(config.FULL_SEEDS),
        "architectures_enabled": list(config.ARCHITECTURES),
        "dnn": {
            "hidden_sizes": config.DNN_HIDDEN_SIZES,
            "dropout": config.DNN_DROPOUT,
            "learning_rate": config.DNN_LEARNING_RATE,
            "batch_size": config.DNN_BATCH_SIZE,
            "max_epochs": config.DNN_MAX_EPOCHS,
            "early_stop_patience": config.DNN_EARLY_STOP_PATIENCE,
            "validation_fraction": config.DNN_VALIDATION_FRACTION,
        },
        "random_forest": {
            "n_estimators": config.RF_N_ESTIMATORS,
            "max_depth": config.RF_MAX_DEPTH,
        },
        "xgboost": {
            "n_estimators": config.XGB_N_ESTIMATORS,
            "max_depth": config.XGB_MAX_DEPTH,
            "learning_rate": config.XGB_LEARNING_RATE,
            "subsample": config.XGB_SUBSAMPLE,
            "colsample_bytree": config.XGB_COLSAMPLE_BYTREE,
        },
        "bootstrap_iterations": config.BOOTSTRAP_ITERATIONS,
        "confidence_level": config.CONFIDENCE_LEVEL,
        "multiple_comparison_method": config.MULTIPLE_COMPARISON_METHOD,
        "small_class_support_threshold": config.SMALL_CLASS_SUPPORT_THRESHOLD,
    }


def capture_environment(dataset_name=None):
    os.makedirs(config.ENV_LOG_DIR, exist_ok=True)

    env_record = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hardware": _get_hardware_info(),
        "git_commit_hash": _get_git_commit_hash(),
        "git_working_tree_dirty": _get_git_dirty_flag(),
        "package_versions": _get_installed_versions(),
        "gpu_info": _get_gpu_info(),
        "active_hyperparameters": _get_active_hyperparameters(),
        "data_file_checksums": {},
    }

    if dataset_name is not None and dataset_name in config.DATASETS:
        ds = config.DATASETS[dataset_name]
        for key in ("train_path", "test_path"):
            path = ds.get(key)
            if path:
                env_record["data_file_checksums"][key] = {
                    "path": path,
                    "sha256": _file_sha256(path),
                }

    if env_record["git_working_tree_dirty"]:
        print(
            "WARNING: git working tree has uncommitted changes. The commit "
            "hash recorded here does NOT fully capture the exact code that "
            "produced this run's results. The active_hyperparameters field "
            "below still records the actual values used regardless, but "
            "commit your changes before running the final experiments."
        )

    timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(config.ENV_LOG_DIR, f"env_{timestamp_str}.json")
    with open(out_path, "w") as f:
        json.dump(env_record, f, indent=2)

    print(f"Environment snapshot written to: {out_path}")
    return env_record


if __name__ == "__main__":
    record = capture_environment(dataset_name="nsl_kdd")
    print(json.dumps(record, indent=2))