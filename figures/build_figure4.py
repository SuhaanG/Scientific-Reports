import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("../results/cse_cic_ids2018_matrix_summary.csv")

architectures = ["dnn", "random_forest", "xgboost"]
arch_labels = ["DNN", "Random Forest", "XGBoost"]
arch_colors = ["#2166ac", "#b2182b", "#4d9221"]

panels = [
    ("aggregate_accuracy", "Aggregate accuracy"),
    ("bruteforce_recall", "Brute-force recall"),
    ("web_attacks_recall", "Web attacks recall"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
rng = np.random.default_rng(0)

for ax, (col, title) in zip(axes, panels):
    for i, (arch, label, color) in enumerate(zip(architectures, arch_labels, arch_colors)):
        values = df[df["architecture"] == arch][col].values
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter, values,
            color=color, alpha=0.6, s=18, edgecolor="none",
        )
        ax.hlines(
            np.mean(values), i - 0.2, i + 0.2,
            color=color, linewidth=2.2, zorder=5,
        )

    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels(arch_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Value", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")

plt.tight_layout()
plt.savefig("figure4_cic_ids2018_raw_distributions.pdf", format="pdf")
print("Figure 4 saved.")