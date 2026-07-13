import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# All values below are the VERIFIED real signal-to-noise ratios, cross-checked
# directly against the raw stats_analysis.py output, not estimated or
# reconstructed. None = degenerate case (zero variance and zero sampling
# noise simultaneously), plotted as a blank "N/A" cell, not zero.

nsl_kdd = {
    "normal":  {"dnn": 70.09, "random_forest": 0.00, "xgboost": 0.00},
    "dos":     {"dnn": 16.79, "random_forest": 11.82, "xgboost": 0.55},
    "probe":   {"dnn": 12.17, "random_forest": 0.00, "xgboost": 0.70},
    "r2l":     {"dnn": 5.15,  "random_forest": 9.93, "xgboost": 7.54},
    "u2r":     {"dnn": 1.68,  "random_forest": 0.00, "xgboost": 0.00},
}

cic_ids2018 = {
    "normal":       {"dnn": None, "random_forest": None, "xgboost": None},
    "bruteforce":   {"dnn": 2.56, "random_forest": 0.00, "xgboost": 0.00},
    "dos":          {"dnn": 0.00, "random_forest": 0.00, "xgboost": 0.00},
    "web_attacks":  {"dnn": None, "random_forest": None, "xgboost": 0.00},
    "infiltration": {"dnn": 1.42, "random_forest": 0.00, "xgboost": 0.00},
    "botnet":       {"dnn": 0.00, "random_forest": 0.00, "xgboost": 0.00},
    "ddos":         {"dnn": 1.09, "random_forest": 0.00, "xgboost": None},
}

architectures = ["dnn", "random_forest", "xgboost"]
arch_labels = ["DNN", "Random Forest", "XGBoost"]
arch_colors = ["#2166ac", "#b2182b", "#4d9221"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

FLOOR = 0.05  # visual floor for true-zero SNR values on a log scale

for ax, (data, title) in zip(
    axes, [(nsl_kdd, "NSL-KDD"), (cic_ids2018, "CSE-CIC-IDS2018")]
):
    categories = list(data.keys())
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.25

    for i, arch in enumerate(architectures):
        values = [data[cat][arch] for cat in categories]
        plot_values = [np.nan if v is None else max(v, FLOOR) for v in values]
        ax.bar(
            x + (i - 1) * width, plot_values, width,
            label=arch_labels[i], color=arch_colors[i], edgecolor="black",
            linewidth=0.4,
        )
        for j, v in enumerate(values):
            if v is None:
                ax.text(
                    x[j] + (i - 1) * width, FLOOR, "N/A",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                    color=arch_colors[i],
                )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 150)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Signal-to-noise ratio\n(genuine / sampling-noise variance, log scale)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

axes[0].legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig("figure1_snr_comparison.pdf", format="pdf")
print("Figure 1 saved.")