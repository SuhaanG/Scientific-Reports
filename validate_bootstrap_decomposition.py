"""
Validates the bootstrap variance decomposition in stats_analysis.py
against synthetic data with a KNOWN ground-truth genuine between-seed
variance, rather than trusting the method on real data alone.

This script is the reproducible artifact backing the claim (stated in
the paper's Methods section) that the decomposition recovers a known
true variance to within a small margin, and correctly reports a
near-zero result in a simulated null case. Run this and report the
printed output directly, do not restate the claim in the paper without
having actually run this and gotten a consistent result.

Usage:
    python validate_bootstrap_decomposition.py
"""

import numpy as np
import pandas as pd

from stats_analysis import true_variance_decomposition


def _run_one_simulation(true_between_seed_std, n_instances_per_seed,
                         n_seeds, mean_p, rng):
    true_between_seed_variance = true_between_seed_std ** 2
    seed_true_ps = rng.normal(mean_p, true_between_seed_std, size=n_seeds)
    seed_true_ps = np.clip(seed_true_ps, 0.01, 0.99)

    summary_rows, instance_rows = [], []
    for seed_idx, p in enumerate(seed_true_ps):
        outcomes = rng.binomial(1, p, size=n_instances_per_seed)
        summary_rows.append({
            "dataset": "validation_sim", "architecture": "sim_arch",
            "seed": seed_idx, "test_recall": outcomes.mean(),
        })
        for i, correct in enumerate(outcomes):
            instance_rows.append({
                "dataset": "validation_sim", "architecture": "sim_arch",
                "seed": seed_idx, "true_category": "test", "correct": correct,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(instance_rows), true_between_seed_variance


def validate_recovers_known_effect(n_simulations=100, seed=12345):
    rng = np.random.default_rng(seed)
    true_std = 0.02
    n_instances = 2754
    n_seeds = 10

    recovered = []
    for _ in range(n_simulations):
        summary_df, per_instance_df, true_var = _run_one_simulation(
            true_std, n_instances, n_seeds, mean_p=0.09, rng=rng
        )
        d = true_variance_decomposition(
            summary_df, per_instance_df, "test",
            "validation_sim", "sim_arch", n_bootstrap=500,
        )
        recovered.append(d["genuine_between_seed_variance"])

    recovered = np.array(recovered)
    true_variance = true_std ** 2
    mean_recovered = recovered.mean()
    relative_bias = (mean_recovered - true_variance) / true_variance * 100

    print("=== TEST 1: Recovery of a KNOWN nonzero effect ===")
    print(f"True between-seed variance:      {true_variance:.6f}")
    print(f"Mean recovered variance (n={n_simulations} sims): {mean_recovered:.6f}")
    print(f"Std of recovered variance across sims: {recovered.std():.6f}")
    print(f"Relative bias: {relative_bias:.1f}%")
    print()
    return relative_bias


def validate_null_case(n_simulations=100, seed=54321):
    rng = np.random.default_rng(seed)
    n_instances = 2754
    n_seeds = 10

    recovered = []
    for _ in range(n_simulations):
        summary_df, per_instance_df, true_var = _run_one_simulation(
            0.0, n_instances, n_seeds, mean_p=0.09, rng=rng
        )
        d = true_variance_decomposition(
            summary_df, per_instance_df, "test",
            "validation_sim", "sim_arch", n_bootstrap=500,
        )
        recovered.append(d["genuine_between_seed_variance"])

    recovered = np.array(recovered)
    print("=== TEST 2: Null case (true genuine variance = 0) ===")
    print(f"Mean recovered (after zero-clipping): {recovered.mean():.7f}")
    print(f"Fraction of sims with recovered > 0: {(recovered > 0).mean()*100:.1f}%")
    print(f"Median recovered: {np.median(recovered):.7f}")
    print()
    return recovered.mean()


if __name__ == "__main__":
    bias = validate_recovers_known_effect()
    null_mean = validate_null_case()

    print("=== SUMMARY (copy these exact numbers into the Methods section) ===")
    print(f"Relative bias recovering a known nonzero effect: {bias:.1f}%")
    print(f"Mean recovered value in the null (no true effect) case: {null_mean:.7f}")