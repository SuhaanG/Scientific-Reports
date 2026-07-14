import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --- Standardize Font Sizes (Nature Journal Typography) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.75
# Set unified baseline font sizes
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 11  # 6. Y-axis label slightly larger
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 12  # 3. Panel titles larger

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
        # Taking the first n seeds for the calculation
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

# 3. Colors match Figure 1
arch_colors = ["#4C72B0", "#DD8452", "#55A868"]
markers = ["o", "s", "^"]

# Adjust margins by using constrained_layout for optimal whitespace usage
fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.6), constrained_layout=True)

for ax_idx, (ax, (df, dataset, title)) in enumerate(zip(
    axes,
    [
        (nsl_kdd_df, "nsl_kdd", "NSL-KDD"),
        (cic_df, "cse_cic_ids2018", "CSE-CIC-IDS2018"),
    ],
)):
    for arch, label, color, marker in zip(architectures, arch_labels, arch_colors, markers):
        result = seed_adequacy(df, dataset, arch)
        # 1. Plotting with default ax.plot maps n_seeds to a true numeric X-axis
        ax.plot(
            result["n_seeds"], result["ci_half_width"],
            marker=marker, color=color, label=label, 
            linewidth=2.0, markersize=7.5, zorder=3  # 4 & 5. Increased line width and marker size
        )
    
    # --- Chart Junk Removal and Formatting ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    
    # 2. Add very light horizontal gridlines
    ax.grid(axis='y', color='gray', alpha=0.15, linestyle='-', linewidth=0.5, zorder=1)
    
    # Tick marks facing outward
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.75, colors='black')
    
    ax.set_xlabel("Training seeds") # 7. Concise x-axis label
    ax.set_xticks(CHECKPOINTS)
    ax.set_xlim(0, 43) # Give a little breathing room on the edges for the numeric scale
    
    ax.set_title(title, pad=8, fontweight='medium')
    
    # Only label Y-axis on the left panel to reduce clutter
    if ax_idx == 0:
        ax.set_ylabel("95% CI half-width (aggregate accuracy)")
        
    # Nature style panel labels (a, b) pushed outward slightly
    panel_label = 'a' if ax_idx == 0 else 'b'
    ax.text(-0.16, 1.05, panel_label, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='bottom', ha='right')

# Legend: borderless, placed cleanly
axes[0].legend(frameon=False, loc="upper right")

plt.savefig("figure2_seed_adequacy.pdf", format="pdf", bbox_inches="tight")
print("Figure 2 saved.")