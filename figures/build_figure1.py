import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- 2. Standardize Font Sizes (Nature Journal Typography) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.75
# Set unified baseline font sizes
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 11     # Increased axis title font (Point 3)
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 11

# Verified signal-to-noise ratios
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

# Muted, sophisticated color palette (colorblind-friendly)
arch_colors = ["#4C72B0", "#DD8452", "#55A868"]

# Adjust margins by using constrained_layout for optimal whitespace usage
fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.6), constrained_layout=True)

FLOOR = 0.05  # visual floor for true-zero SNR values on a log scale

for ax_idx, (ax, (data, title)) in enumerate(zip(axes, [(nsl_kdd, "NSL-KDD"), (cic_ids2018, "CSE-CIC-IDS2018")])):
    categories = list(data.keys())
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.28

    for i, arch in enumerate(architectures):
        values = [data[cat][arch] for cat in categories]
        plot_values = [np.nan if v is None else max(v, FLOOR) for v in values]
        
        # Determine exact bar position
        offset = (i - 1) * width
        
        ax.bar(
            x + offset, plot_values, width,
            label=arch_labels[i], color=arch_colors[i], edgecolor="black",
            linewidth=0.5, zorder=3
        )
        
        # Replace awkward "N/A" text with clean gray "x" markers
        for j, v in enumerate(values):
            if v is None:
                ax.plot(
                    x[j] + offset, FLOOR * 1.15, marker='x',
                    color='dimgray', markersize=5, markeredgewidth=1.0, zorder=4
                )

    # --- Chart Junk Removal and Formatting ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    
    # Tick marks (Point 9): Major ticks are standard, minor ticks are thinner
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.75, colors='black')
    ax.tick_params(axis='y', which='minor', direction='out', length=2.5, width=0.4, colors='black')

    # Make significance threshold line prominent
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
    
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 150)
    ax.set_xticks(x)
    
    # Consistent x-axis labels (Point 1)
    if title == "CSE-CIC-IDS2018":
        overrides = {
            'normal': 'Normal', 'bruteforce': 'Brute', 'dos': 'DoS', 
            'web_attacks': 'Web', 'infiltration': 'Infil', 'botnet': 'Botnet', 'ddos': 'DDoS'
        }
    else:
        overrides = {'normal': 'Normal', 'dos': 'DoS', 'probe': 'Probe', 'r2l': 'R2L', 'u2r': 'U2R'}
        
    clean_categories = [overrides.get(c, c) for c in categories]
    
    ax.set_xticklabels(clean_categories, rotation=30, ha="right")
    
    # Bring titles down slightly (Point 10)
    ax.set_title(title, pad=6, fontweight='medium')
    
    if ax_idx == 0:
        ax.set_ylabel("Signal-to-noise ratio (log$_{10}$ scale)")
        
    # Nature style panel labels pushed outward slightly (Point 4)
    panel_label = 'a' if ax_idx == 0 else 'b'
    ax.text(-0.16, 1.05, panel_label, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='bottom', ha='right')

# Legend: borderless, placed cleanly (Point 5)
axes[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 1.02))

plt.savefig("figure1_snr_comparison.pdf", format="pdf", bbox_inches="tight")
print("Figure 1 saved.")