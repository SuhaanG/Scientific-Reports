import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Standardize Font Sizes (Nature Journal Typography) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.75
# Set unified baseline font sizes
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 12

# Adjust this path if your results/ folder is located elsewhere relative
# to where you run this script from.
df = pd.read_csv("../results/nsl_kdd_matrix_summary.csv")

architectures = ["dnn", "random_forest", "xgboost"]
arch_labels = ["DNN", "Random Forest", "XGBoost"]

# Muted, sophisticated color palette (colorblind-friendly, matching Figs 1 & 2)
arch_colors = ["#4C72B0", "#DD8452", "#55A868"]

panels = [
    ("aggregate_accuracy", "Aggregate accuracy"),
    ("r2l_recall", "R2L recall"),
    ("probe_recall", "Probe recall"),
]

# 7.08 inches is the Nature double-column width. 
# 3.2 inches height provides a sleek, panoramic aspect ratio for 3 panels.
fig, axes = plt.subplots(1, 3, figsize=(7.08, 3.2), constrained_layout=True)
rng = np.random.default_rng(0)  # purely visual horizontal jitter

panel_letters = ['a', 'b', 'c']

for ax_idx, (ax, (col, title)) in enumerate(zip(axes, panels)):
    for i, (arch, label, color) in enumerate(zip(architectures, arch_labels, arch_colors)):
        values = df[df["architecture"] == arch][col].dropna().values
        
        # Slightly wider jitter for better point visibility
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        
        # Scatter points (semi-transparent)
        ax.scatter(
            np.full(len(values), i) + jitter, values,
            color=color, alpha=0.65, s=20, edgecolor="none", zorder=2
        )
        
        # Mean line: Dark gray, fully opaque, and slightly wider than the jitter 
        # so the average stands out sharply against the colored points
        ax.hlines(
            np.mean(values), i - 0.25, i + 0.25,
            color="#333333", linewidth=2.5, zorder=5
        )

    # --- Chart Junk Removal and Formatting ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    
    # Tick marks facing outward
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.75, colors='black')
    
    # Subtle horizontal gridlines to help the eye track values across the spread
    ax.grid(axis='y', color='gray', alpha=0.15, linestyle='-', linewidth=0.5, zorder=1)

    ax.set_xticks(range(len(architectures)))
    
    # 30-degree rotation prevents the labels from crowding
    ax.set_xticklabels(arch_labels, rotation=30, ha="right")
    ax.set_title(title, pad=8, fontweight='medium')
    
    # Only label the Y-axis on the leftmost plot to reduce clutter, 
    # the panel titles provide the specific metric context.
    if ax_idx == 0:
        ax.set_ylabel("Metric value")
        
    # Nature style panel labels (a, b, c) pushed outward slightly
    # The first panel needs a slightly larger negative offset to clear the y-axis label
    x_offset = -0.28 if ax_idx == 0 else -0.15
    ax.text(x_offset, 1.05, panel_letters[ax_idx], 
            transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')

plt.savefig("figure3_nsl_kdd_raw_distributions.pdf", format="pdf", bbox_inches="tight")
print("Figure 3 saved.")