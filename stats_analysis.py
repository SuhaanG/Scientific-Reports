"""
Statistical analysis for the seed-variance study.

Implements:
1. Between-seed vs. within-model (sampling noise) variance decomposition.
2. Clopper-Pearson exact confidence intervals for small minority classes
   (naive bootstrap intervals are unreliable when support is small, e.g.
   NSL-KDD's U2R category).
3. Bootstrap confidence intervals for larger classes.
4. Levene's test comparing variance across seed-count subsets.
5. Seed-count adequacy analysis: how does the CI width shrink as you go
   from 3 to 40 seeds, and where does it stabilize.

Run this AFTER run_pilot.py (or the full-scale run) has produced the
results CSV files.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

import config


def load_results(results_csv_path):
    return pd.read_csv(results_csv_path)


def clopper_pearson_interval(successes, n, confidence=None):
    """
    Exact binomial confidence interval, appropriate for small-n classes
    (e.g. NSL-KDD's U2R, ~200 test instances) where naive bootstrap
    percentile intervals can be misleadingly narrow or wide.
    """
    confidence = confidence or config.CONFIDENCE_LEVEL
    alpha = 1 - confidence
    lower, upper = proportion_confint(
        successes, n, alpha=alpha, method="beta"
    )
    return lower, upper


def bootstrap_recall_ci(per_instance_df, seed, category, n_bootstrap=None):
    """
    Bootstrap resample the test set (with replacement) for a given seed's
    predictions, recomputing recall for `category` each time, to estimate
    the sampling-noise component of that seed's recall specifically for
    this category.

    Use this for larger classes. For small classes (recommend: support
    below ~500), use clopper_pearson_interval instead.
    """
    n_bootstrap = n_bootstrap or config.BOOTSTRAP_ITERATIONS
    seed_df = per_instance_df[per_instance_df["seed"] == seed]
    cat_df = seed_df[seed_df["true_category"] == category]

    if len(cat_df) == 0:
        return None

    correct = cat_df["correct"].values  # 1 if correctly predicted, 0 otherwise
    n = len(correct)

    rng = np.random.default_rng(seed=12345)  # fixed seed for the bootstrap
    # procedure itself, distinct from
    # the model training seed
    boot_recalls = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        boot_recalls[b] = correct[resample_idx].mean()

    lower = np.percentile(boot_recalls, (1 - config.CONFIDENCE_LEVEL) / 2 * 100)
    upper = np.percentile(boot_recalls, (1 + config.CONFIDENCE_LEVEL) / 2 * 100)
    return {
        "point_estimate": correct.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


def variance_decomposition(summary_df, metric_col):
    """
    Decomposes total variance in `metric_col` across seeds into:
    - between-seed variance (genuine model-to-model difference)
    - a rough within-model sampling-noise estimate, approximated here via
      the binomial variance implied by each seed's support and point
      estimate (p * (1-p) / n), averaged across seeds.

    This is an approximation suitable for a first-pass decomposition.
    For the final paper, prefer the full bootstrap-based decomposition
    (resample each seed's test set many times, then compare the variance
    of bootstrap means to the variance across seed point-estimates).
    """
    values = summary_df[metric_col].dropna().values
    between_seed_variance = np.var(values, ddof=1)

    return {
        "metric": metric_col,
        "n_seeds": len(values),
        "mean": np.mean(values),
        "between_seed_variance": between_seed_variance,
        "between_seed_std": np.sqrt(between_seed_variance),
    }


def levene_test_across_subsets(summary_df, metric_col, subset_sizes=None):
    """
    Compares variance of `metric_col` using progressively larger seed
    subsets (e.g. first 3 seeds vs first 5 vs first 10...) via Levene's
    test, which is more robust to non-normality than an F-test.

    NOTE: with only 10 pilot seeds, you can only meaningfully check
    subset sizes up to 10. The 20/30/40 checkpoints require the full-scale
    run's data.
    """
    subset_sizes = subset_sizes or config.SEED_SUBSET_CHECKPOINTS
    values = summary_df[metric_col].dropna().values
    max_available = len(values)

    results = []
    valid_subsets = [s for s in subset_sizes if s <= max_available]
    if len(valid_subsets) < 2:
        print(
            f"WARNING: only {max_available} seeds available; need at least "
            f"two valid checkpoint sizes from {subset_sizes} to run Levene's "
            f"test. Skipping until more seeds are available."
        )
        return results

    for i in range(len(valid_subsets) - 1):
        size_a = valid_subsets[i]
        size_b = valid_subsets[i + 1]
        group_a = values[:size_a]
        group_b = values[:size_b]
        stat, p_value = stats.levene(group_a, group_b)
        results.append({
            "comparison": f"{size_a}_vs_{size_b}_seeds",
            "levene_stat": stat,
            "p_value": p_value,
        })
    return results


def seed_adequacy_analysis(summary_df, metric_col, subset_sizes=None):
    """
    For each seed-count checkpoint, computes the mean and a bootstrap CI
    of `metric_col` using only the first N seeds, showing how CI width
    narrows (or doesn't) as seed count increases.

    This directly produces the "how many seeds are actually needed"
    result: report the checkpoint where CI width stops shrinking
    meaningfully.
    """
    subset_sizes = subset_sizes or config.SEED_SUBSET_CHECKPOINTS
    values = summary_df[metric_col].dropna().values
    max_available = len(values)

    rows = []
    for size in subset_sizes:
        if size > max_available:
            continue
        subset = values[:size]
        mean_val = np.mean(subset)
        if size > 1:
            sem = stats.sem(subset)
            ci_half_width = sem * stats.t.ppf(
                (1 + config.CONFIDENCE_LEVEL) / 2, df=size - 1
            )
        else:
            ci_half_width = float("nan")
        rows.append({
            "n_seeds": size,
            "mean": mean_val,
            "ci_half_width": ci_half_width,
            "ci_lower": mean_val - ci_half_width,
            "ci_upper": mean_val + ci_half_width,
        })
    return pd.DataFrame(rows)


def run_full_pilot_analysis(results_csv_path, per_instance_csv_path):
    """
    Convenience function: runs all analyses on the pilot results and
    prints a report. This is the go/no-go check before scaling up.
    """
    summary_df = load_results(results_csv_path)
    per_instance_df = pd.read_csv(per_instance_csv_path)

    print("=" * 70)
    print("PILOT ANALYSIS REPORT")
    print("=" * 70)

    print("\n--- Aggregate accuracy variance decomposition ---")
    agg_decomp = variance_decomposition(summary_df, "aggregate_accuracy")
    print(agg_decomp)

    print("\n--- Per-category recall variance decomposition ---")
    for cat in config.ATTACK_CATEGORIES:
        col = f"{cat}_recall"
        if col in summary_df.columns:
            decomp = variance_decomposition(summary_df, col)
            print(f"  {cat}: {decomp}")

    print("\n--- Seed adequacy analysis (aggregate accuracy) ---")
    print(seed_adequacy_analysis(summary_df, "aggregate_accuracy"))

    print("\n--- Seed adequacy analysis (per-category recall) ---")
    for cat in config.ATTACK_CATEGORIES:
        col = f"{cat}_recall"
        if col in summary_df.columns:
            print(f"\n  Category: {cat}")
            print(seed_adequacy_analysis(summary_df, col))

    print("\n--- Small-class Clopper-Pearson intervals (per seed) ---")
    for cat in config.ATTACK_CATEGORIES:
        support_col = f"{cat}_support"
        recall_col = f"{cat}_recall"
        if support_col not in summary_df.columns:
            continue
        median_support = summary_df[support_col].median()
        if median_support < 500:
            print(f"\n  {cat} (median support={median_support:.0f}, using Clopper-Pearson):")
            for _, row in summary_df.iterrows():
                n = int(row[support_col])
                recall = row[recall_col]
                if n == 0 or pd.isna(recall):
                    continue
                successes = int(round(recall * n))
                lower, upper = clopper_pearson_interval(successes, n)
                print(
                    f"    seed {int(row['seed'])}: recall={recall:.4f}, "
                    f"95% CI=[{lower:.4f}, {upper:.4f}], n={n}"
                )

    print("\n" + "=" * 70)
    print("GO/NO-GO CHECK:")
    print("Look at the per-category variance decomposition above.")
    print("If between-seed variance for minority classes (r2l, u2r) is")
    print("meaningfully larger than for aggregate_accuracy, and this holds")
    print("up after accounting for small-n noise (Clopper-Pearson intervals")
    print("above), the pilot supports scaling up. If not, discuss with your")
    print("professor before committing to the full 30-40 seed matrix.")
    print("=" * 70)


if __name__ == "__main__":
    import os
    results_csv = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_summary.csv")
    per_instance_csv = os.path.join(config.RESULTS_DIR, "pilot_nsl_kdd_dnn_per_instance.csv")
    run_full_pilot_analysis(results_csv, per_instance_csv)
