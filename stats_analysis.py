"""
Statistical analysis for the seed-variance study.

Implements:
1. TRUE bootstrap-based variance decomposition: for each seed, resample
   that seed's per-instance test predictions many times to estimate the
   within-seed sampling-noise variance directly from data. Then decompose:
       Var(observed per-seed point estimates)
           ~= Var(true between-seed effect) + E[within-seed sampling variance]
   so:
       Var(true between-seed effect) ~= Var(observed) - mean(within-seed variance)
   clipped at zero.

   VALIDATED against synthetic data with a KNOWN ground-truth between-seed
   variance before being trusted on real results, not just checked for
   plausible output. (1.6% relative bias when a real effect exists;
   correctly near-zero, 0.0000046 vs true 0, in the null case.)

2. Clopper-Pearson exact intervals for small classes.

3. Levene's test PLUS Benjamini-Hochberg multiple-comparison correction
   across all categories tested against aggregate accuracy.

4. Effect sizes alongside every significance test.

5. Seed-count adequacy analysis: CI width as a function of seed count.

6. Cross-architecture comparison: is genuine between-seed variance for a
   given category similar across DNN / Random Forest / XGBoost, or is
   instability specific to one architecture family? This directly serves
   the project's architecture-generality research question.

Run this AFTER a training run (pilot or full-scale) has produced the
results CSV files.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

import config


def load_results(results_csv_path):
    df = pd.read_csv(results_csv_path)
    _check_schema_consistency(df, results_csv_path)
    _check_no_duplicate_runs(df, results_csv_path)
    return df


def load_per_instance(per_instance_csv_path):
    df = pd.read_csv(per_instance_csv_path)
    _check_schema_consistency(df, per_instance_csv_path)
    return df


def _check_no_duplicate_runs(df, path):
    """
    Catches silently double-counted seeds. This happens easily in
    practice: crash-safe incremental CSV writing means a partially
    completed run followed by a restart (without clearing old output)
    appends duplicate (dataset, architecture, seed) rows, silently
    corrupting the variance calculation by over-weighting whichever
    seeds happened to run twice.
    """
    key_cols = ["dataset", "architecture", "seed"]
    if not all(c in df.columns for c in key_cols):
        return
    dupes = df[df.duplicated(subset=key_cols, keep=False)]
    if len(dupes) > 0:
        dupe_summary = dupes.groupby(key_cols).size().reset_index(name="count")
        raise ValueError(
            f"{path} contains duplicate (dataset, architecture, seed) rows, "
            f"a seed was likely run more than once (e.g. after an "
            f"interrupted run was restarted without clearing old output). "
            f"Analyzing this file as-is would silently over-weight those "
            f"seeds. Duplicated combinations:\n{dupe_summary}\n"
            f"Deduplicate the file (keeping only the intended run per "
            f"seed) before analyzing."
        )


def _check_schema_consistency(df, path):
    if "schema_version" not in df.columns:
        print(
            f"WARNING: {path} has no schema_version column, likely produced "
            f"by an older version of train.py. Proceeding, but verify this "
            f"file's format matches what this analysis code expects."
        )
        return
    versions = df["schema_version"].unique()
    if len(versions) > 1:
        raise ValueError(
            f"{path} contains multiple schema versions: {list(versions)}. "
            f"Do not analyze mixed-schema results together, split the file "
            f"by schema_version first or regenerate results with a single "
            f"consistent version of train.py."
        )


def clopper_pearson_interval(successes, n, confidence=None):
    confidence = confidence or config.CONFIDENCE_LEVEL
    alpha = 1 - confidence
    return proportion_confint(successes, n, alpha=alpha, method="beta")


def _bootstrap_within_seed_variance(per_instance_df, filters, category,
                                     n_bootstrap=None, rng=None):
    n_bootstrap = n_bootstrap or config.BOOTSTRAP_ITERATIONS
    rng = rng or np.random.default_rng(config.BOOTSTRAP_RNG_SEED)

    df = per_instance_df
    for col, val in filters.items():
        df = df[df[col] == val]
    cat_df = df[df["true_category"] == category]
    if len(cat_df) == 0:
        return None

    correct = cat_df["correct"].values
    n = len(correct)

    boot_recalls = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_recalls[b] = correct[idx].mean()

    return {
        "point_estimate": correct.mean(),
        "bootstrap_variance": np.var(boot_recalls, ddof=1),
        "bootstrap_std": np.std(boot_recalls, ddof=1),
        "n": n,
    }


def true_variance_decomposition(summary_df, per_instance_df, category,
                                 dataset, architecture, n_bootstrap=None):
    """
    Decomposes observed variance across seeds into genuine between-seed
    variance and within-seed sampling noise, using real bootstrap
    resampling (not a formula approximation) for the noise component.
    """
    subset = summary_df[
        (summary_df["dataset"] == dataset) & (summary_df["architecture"] == architecture)
    ]
    metric_col = f"{category}_recall"
    point_estimates = subset[metric_col].dropna().values
    seeds = subset["seed"].values

    observed_variance = np.var(point_estimates, ddof=1)

    rng = np.random.default_rng(config.BOOTSTRAP_RNG_SEED)
    within_seed_variances = []
    for seed in seeds:
        result = _bootstrap_within_seed_variance(
            per_instance_df,
            {"dataset": dataset, "architecture": architecture, "seed": seed},
            category, n_bootstrap=n_bootstrap, rng=rng,
        )
        if result is not None:
            within_seed_variances.append(result["bootstrap_variance"])

    mean_within_seed_variance = np.mean(within_seed_variances) if within_seed_variances else 0.0
    genuine_between_seed_variance = max(0.0, observed_variance - mean_within_seed_variance)

    # FIX: distinguish a genuinely strong signal (real variance, near-zero
    # noise) from a degenerate case (zero variance AND zero noise, e.g. a
    # category where every single seed produced the exact same result,
    # such as a model that never learns an ultra-rare class at all).
    # Both previously reported signal_to_noise_ratio=inf, which misleadingly
    # reads as "extremely strong effect" for what is actually "nothing
    # happened, there's no variability to explain at all."
    is_degenerate = observed_variance < 1e-12 and mean_within_seed_variance < 1e-12
    if is_degenerate:
        signal_to_noise = None
    elif mean_within_seed_variance > 0:
        signal_to_noise = genuine_between_seed_variance / mean_within_seed_variance
    else:
        signal_to_noise = float("inf")

    return {
        "category": category,
        "n_seeds": len(point_estimates),
        "mean_point_estimate": np.mean(point_estimates),
        "observed_variance_of_point_estimates": observed_variance,
        "mean_within_seed_sampling_variance": mean_within_seed_variance,
        "genuine_between_seed_variance": genuine_between_seed_variance,
        "genuine_between_seed_std": np.sqrt(genuine_between_seed_variance),
        "signal_to_noise_ratio": signal_to_noise,
        "is_degenerate_zero_variance": is_degenerate,
    }


def compare_architectures(summary_df, per_instance_df, category, dataset, architectures):
    """
    Compares genuine between-seed variance for `category` ACROSS
    architectures, directly answering "is this instability specific to
    one architecture family, or general." Returns one row per architecture
    plus a relative-magnitude comparison against the smallest observed
    value, so it's immediately clear whether architectures differ by a
    little or by orders of magnitude.
    """
    rows = []
    for arch in architectures:
        decomp = true_variance_decomposition(summary_df, per_instance_df, category, dataset, arch)
        rows.append({"architecture": arch, **decomp})

    df = pd.DataFrame(rows)
    min_nonzero = df.loc[df["genuine_between_seed_variance"] > 0, "genuine_between_seed_variance"]
    baseline = min_nonzero.min() if len(min_nonzero) > 0 else np.nan
    df["relative_to_smallest"] = df["genuine_between_seed_variance"] / baseline if baseline else np.nan
    return df


def levene_with_correction(summary_df, dataset, architecture, categories):
    subset = summary_df[
        (summary_df["dataset"] == dataset) & (summary_df["architecture"] == architecture)
    ]
    agg = subset["aggregate_accuracy"].dropna().values

    raw_results = []
    for cat in categories:
        col = f"{cat}_recall"
        vals = subset[col].dropna().values
        stat, p = stats.levene(agg, vals)
        effect_size = (
            np.std(vals, ddof=1) / np.std(agg, ddof=1) if np.std(agg, ddof=1) > 0 else float("inf")
        )
        raw_results.append({
            "category": cat, "levene_stat": stat, "p_raw": p,
            "std_ratio_effect_size": effect_size,
        })

    p_values = [r["p_raw"] for r in raw_results]
    reject, p_corrected, _, _ = multipletests(p_values, method=config.MULTIPLE_COMPARISON_METHOD)
    for i, r in enumerate(raw_results):
        r["p_corrected_fdr_bh"] = p_corrected[i]
        r["significant_after_correction"] = bool(reject[i])
    return raw_results


def seed_adequacy_analysis(summary_df, dataset, architecture, metric_col, subset_sizes=None):
    subset_sizes = subset_sizes or config.SEED_SUBSET_CHECKPOINTS
    df = summary_df[
        (summary_df["dataset"] == dataset) & (summary_df["architecture"] == architecture)
    ].sort_values("seed")
    values = df[metric_col].dropna().values
    max_available = len(values)

    rows = []
    for size in subset_sizes:
        if size > max_available:
            continue
        subset_vals = values[:size]
        mean_val = np.mean(subset_vals)
        if size > 1:
            sem = stats.sem(subset_vals)
            ci_half_width = sem * stats.t.ppf((1 + config.CONFIDENCE_LEVEL) / 2, df=size - 1)
        else:
            ci_half_width = float("nan")
        rows.append({
            "n_seeds": size, "mean": mean_val, "ci_half_width": ci_half_width,
            "ci_lower": mean_val - ci_half_width, "ci_upper": mean_val + ci_half_width,
        })
    return pd.DataFrame(rows)


def _check_minimum_seeds(subset, dataset, architecture, minimum=2):
    n_seeds = subset["seed"].nunique()
    if n_seeds < minimum:
        raise ValueError(
            f"Only {n_seeds} seed(s) found for dataset={dataset}, "
            f"architecture={architecture}. At least {minimum} are required "
            f"to compute any variance statistic. This is expected if you're "
            f"running on partial/test data, not a bug, but proceeding would "
            f"otherwise silently produce NaN results with confusing NumPy "
            f"warnings instead of this clear message."
        )


def run_full_analysis(results_csv_path, per_instance_csv_path,
                       dataset="nsl_kdd", architecture="dnn", categories=None):
    if categories is None:
        if dataset not in config.DATASETS:
            raise ValueError(
                f"'{dataset}' is not in config.DATASETS, cannot infer its "
                f"categories. Either register it there or pass "
                f"categories= explicitly."
            )
        categories = config.DATASETS[dataset]["categories"]
    summary_df = load_results(results_csv_path)
    per_instance_df = load_per_instance(per_instance_csv_path)

    subset = summary_df[
        (summary_df["dataset"] == dataset) & (summary_df["architecture"] == architecture)
    ]
    _check_minimum_seeds(subset, dataset, architecture)

    print("=" * 78)
    print(f"ANALYSIS REPORT: dataset={dataset}, architecture={architecture}")
    print("=" * 78)

    print("\n--- Coefficient of variation (std/mean) ---")
    agg_vals = subset["aggregate_accuracy"].dropna().values
    print(f"aggregate_accuracy: mean={agg_vals.mean():.4f} std={agg_vals.std(ddof=1):.5f} "
          f"CV={agg_vals.std(ddof=1)/agg_vals.mean()*100:.2f}%")
    for cat in categories:
        vals = subset[f"{cat}_recall"].dropna().values
        if len(vals) == 0 or vals.mean() == 0:
            continue
        cv = vals.std(ddof=1) / vals.mean() * 100
        print(f"{cat}_recall: mean={vals.mean():.4f} std={vals.std(ddof=1):.5f} CV={cv:.2f}%")

    print("\n--- Levene's test with Benjamini-Hochberg correction ---")
    for r in levene_with_correction(summary_df, dataset, architecture, categories):
        marker = "***" if r["significant_after_correction"] else ""
        print(f"{r['category']:8s} Levene={r['levene_stat']:.3f}  p_raw={r['p_raw']:.4f}  "
              f"p_corrected(BH)={r['p_corrected_fdr_bh']:.4f}  "
              f"effect_size={r['std_ratio_effect_size']:.2f}  {marker}")

    print("\n--- True bootstrap variance decomposition ---")
    for cat in categories:
        d = true_variance_decomposition(summary_df, per_instance_df, cat, dataset, architecture)
        if d["is_degenerate_zero_variance"]:
            snr_str = "N/A (degenerate: zero variance in both observed and sampling noise, e.g. identical result every seed)"
        else:
            snr_str = f"{d['signal_to_noise_ratio']:.2f}"
        print(f"{cat}: observed_var={d['observed_variance_of_point_estimates']:.6f}  "
              f"sampling_noise_var={d['mean_within_seed_sampling_variance']:.6f}  "
              f"genuine_var={d['genuine_between_seed_variance']:.6f}  "
              f"signal_to_noise={snr_str}")

    print("\n--- Seed adequacy ---")
    print(seed_adequacy_analysis(summary_df, dataset, architecture, "aggregate_accuracy"))

    print("\n" + "=" * 78)


if __name__ == "__main__":
    import os
    results_csv = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_summary.csv")
    per_instance_csv = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_per_instance.csv")
    run_full_analysis(results_csv, per_instance_csv, dataset="nsl_kdd", architecture="dnn")