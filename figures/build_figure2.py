import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CHECKPOINTS = [3, 5, 10, 20, 30, 40]


def seed_adequacy(df, dataset, architecture, metric_col="aggregate_accuracy"):
    subset = df[
        (df["dataset"] == dataset) & (df["architecture"] == architecture)
    ].sort_values("seed")
    values = subset[metric_col].dropna().values

    rows = []
    for n in CHECKPOINTS:
        if n > len(values):
            continue
        subset_vals = values[:n]
        mean_val = np.mean(subset_vals)
        sem = stats.sem(subset_vals)
        half_width = sem * stats.t.ppf(0.975, df=n - 1)
        rows.append({"n_seeds": n, "mean": mean_val, "ci_half_width": half_width})
    return pd.DataFrame(rows)


# Adjust these paths if your results/ folder is located elsewhere relative
# to where you run this script from.
nsl_kdd_df = pd.read_csv("../results/nsl_kdd_matrix_summary.csv")
cic_df = pd.read_csv("../results/cse_cic_ids2018_matrix_summary.csv")

architectures = ["dnn", "random_forest", "xgboost"]
arch_labels = ["DNN", "Random Forest", "XGBoost"]
arch_colors = ["#2166ac", "#b2182b", "#4d9221"]
markers = ["o", "s", "^"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (df, dataset, title) in zip(
    axes,
    [
        (nsl_kdd_df, "nsl_kdd", "NSL-KDD"),
        (cic_df, "cse_cic_ids2018", "CSE-CIC-IDS2018"),
    ],
):
    for arch, label, color, marker in zip(architectures, arch_labels, arch_colors, markers):
        result = seed_adequacy(df, dataset, arch)
        ax.plot(
            result["n_seeds"], result["ci_half_width"],
            marker=marker, color=color, label=label, linewidth=1.3, markersize=5,
        )
    ax.set_xlabel("Number of seeds", fontsize=9)
    ax.set_ylabel("95% CI half-width\n(aggregate accuracy)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(CHECKPOINTS)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)

axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figure2_seed_adequacy.pdf", format="pdf")
print("Figure 2 saved.")