# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUTPUT_DIR = r"C:\Users\Admin\Documents\paper\PINN_ECT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication styling - Clean, minimal, high-impact IEEE Transactions style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.0,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.2,
    "figure.titlesize": 12.0,
    "figure.titleweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Color palette for 2 Evaluation Protocols:
# Protocol 1: Known Specimen (Repeat Scan) -> Deep Sapphire Blue
# Protocol 2: Unseen Blind Specimen (10-Fold LODO) -> Vivid Amber Orange
C_METHOD1 = "#1E88E5"   # Blue (Known Specimen - Repeat Scan)
C_METHOD2 = "#E65100"   # Amber Orange (Unseen Specimen - 10-Fold LODO)

# All 4 Models across ECT Evaluation
all_models = [
    "CNN PINN\n(PI-LGL)",
    "CNN Baseline\n(NoPINN)",
    "Multitask MLP\n(1D PINN)",
    "Xiong et al.\nMLP (1D)"
]

# NMAE (%) Data for [Protocol 1 (Repeat Scan), Protocol 2 (10-Fold LODO)]
# Width W (Range = 1.0 mm)
nmae_w_vals = {
    "CNN PINN\n(PI-LGL)":       [8.10, 10.50],
    "CNN Baseline\n(NoPINN)":   [7.30, 9.80],
    "Multitask MLP\n(1D PINN)": [36.30, 20.10],
    "Xiong et al.\nMLP (1D)":   [6981.0, 132.0]
}

# Length L (Range = 10.0 mm)
nmae_l_vals = {
    "CNN PINN\n(PI-LGL)":       [4.48, 5.24],
    "CNN Baseline\n(NoPINN)":   [11.92, 7.71],
    "Multitask MLP\n(1D PINN)": [61.09, 90.20],
    "Xiong et al.\nMLP (1D)":   [15469.5, 15389.8]
}

# Depth D (Range = 2.0 mm)
nmae_d_vals = {
    "CNN PINN\n(PI-LGL)":       [33.60, 30.60],
    "CNN Baseline\n(NoPINN)":   [23.20, 28.70],
    "Multitask MLP\n(1D PINN)": [118.85, 145.50],
    "Xiong et al.\nMLP (1D)":   [12425.0, 12505.0]
}

# Overall Spatial NMAE (%) = Mean of (W, L, D)
nmae_overall_vals = {
    "CNN PINN\n(PI-LGL)":       [15.39, 15.45],
    "CNN Baseline\n(NoPINN)":   [14.14, 15.40],
    "Multitask MLP\n(1D PINN)": [72.08, 85.27],
    "Xiong et al.\nMLP (1D)":   [11625.17, 9342.27]
}

def plot_nmae_all_dimensions():
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.5), dpi=300)
    plt.subplots_adjust(wspace=0.22, hspace=0.34, top=0.82, bottom=0.07)
    x = np.arange(len(all_models))
    w_bar = 0.34

    # =========================================================================
    # PANEL (a): Width Sizing NMAE_W (%)
    # =========================================================================
    ax = axs[0, 0]
    m1_w = [nmae_w_vals[m][0] for m in all_models]
    m2_w = [nmae_w_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_w, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_w, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(1.0, 30000)
    ax.axhline(20.0, color="#2e7d32", ls="--", lw=1.2)  # 0.20 mm / 1.0 mm

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{NMAE}_W$ (%) [Log Scale]", fontweight="bold")
    ax.set_title(r"(a) Crack Width Error ($\mathrm{NMAE}_W$)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # PANEL (b): Length Sizing NMAE_L (%) - CRITICAL NDT CRITERION
    # =========================================================================
    ax = axs[0, 1]
    m1_l = [nmae_l_vals[m][0] for m in all_models]
    m2_l = [nmae_l_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_l, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_l, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(1.0, 30000)
    ax.axhline(5.0, color="#2e7d32", ls="--", lw=1.2)    # 0.50 mm / 10.0 mm
    ax.axhline(10.0, color="#d32f2f", ls="-.", lw=1.2)  # 1.00 mm / 10.0 mm

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{NMAE}_L$ (%) [Log Scale]", fontweight="bold")
    ax.set_title(r"(b) Crack Length Error ($\mathrm{NMAE}_L$)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # PANEL (c): Depth Sizing NMAE_D (%)
    # =========================================================================
    ax = axs[1, 0]
    m1_d = [nmae_d_vals[m][0] for m in all_models]
    m2_d = [nmae_d_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_d, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_d, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(1.0, 30000)
    ax.axhline(25.0, color="#2e7d32", ls="--", lw=1.2)  # 0.50 mm / 2.0 mm
    ax.axhline(50.0, color="#d32f2f", ls="-.", lw=1.2)  # 1.00 mm / 2.0 mm

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{NMAE}_D$ (%) [Log Scale]", fontweight="bold")
    ax.set_title(r"(c) Crack Depth Error ($\mathrm{NMAE}_D$)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # PANEL (d): Overall 3D Spatial Sizing NMAE (%)
    # =========================================================================
    ax = axs[1, 1]
    m1_o = [nmae_overall_vals[m][0] for m in all_models]
    m2_o = [nmae_overall_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_o, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_o, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(1.0, 30000)
    ax.axhline(16.7, color="#2e7d32", ls="--", lw=1.2)  # (20+5+25)/3 = 16.7%

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{NMAE}_{\mathrm{Overall}}$ (%) [Log Scale]", fontweight="bold")
    ax.set_title(r"(d) Overall 3D Sizing Error ($\mathrm{NMAE}_{\mathrm{Overall}}$)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # SINGLE UNIFIED EXTERNAL LEGEND STRIP (TOP OUTSIDE ALL SUBPLOTS)
    # =========================================================================
    unified_handles = [
        Patch(facecolor=C_METHOD1, edgecolor="black", lw=0.8, 
              label="Protocol 1: Known Specimen (Repeat Scan)"),
        Patch(facecolor=C_METHOD2, edgecolor="black", lw=0.8, 
              label="Protocol 2: Unseen Specimen (10-Fold LODO)"),
        Line2D([0], [0], color="#2e7d32", ls="--", lw=1.4, 
               label=r"Strict NDT Threshold ($W < 20\%,\ L < 5\%,\ D < 25\%$)"),
        Line2D([0], [0], color="#d32f2f", ls="-.", lw=1.4, 
               label=r"NDT Tolerance Limit ($L < 10\%,\ D < 50\%$)")
    ]

    fig.legend(handles=unified_handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=2, frameon=True, fontsize=8.4, edgecolor="#cbd5e1", framealpha=0.96,
               handletextpad=0.6, columnspacing=1.8)

    # Figure Suptitle at the very top
    fig.suptitle("NORMALIZED MEAN ABSOLUTE ERROR (NMAE) ACROSS ALL 3D SPATIAL DIMENSIONS (W, L, D)\nBenchmarking Sizing Accuracy Under Repeat Scan vs. 10-Fold LODO Blind Evaluation Protocols", 
                 fontsize=12.0, fontweight="bold", y=0.985)

    # Footnote at the bottom
    fig.text(0.5, 0.02, r"*Note: Normalized MAE defined as $\mathrm{NMAE} = (\mathrm{MAE} / \mathrm{Range}) \times 100\%$, where $\mathrm{Range}_W=1.0\,\mathrm{mm},\ \mathrm{Range}_L=10.0\,\mathrm{mm},\ \mathrm{Range}_D=2.0\,\mathrm{mm}$.", 
             ha="center", fontsize=8.5, fontstyle="italic", color="#334155")

    out_file1 = os.path.join(OUTPUT_DIR, "fig_nmae_wld_models_comparison.png")
    out_file2 = os.path.join(OUTPUT_DIR, "fig_nmae_length_models_comparison.png")
    
    plt.savefig(out_file1)
    plt.savefig(out_file2)
    plt.close()
    print(f"[OK] Successfully saved all-dimensions NMAE plots to:\n  - {out_file1}\n  - {out_file2}")

if __name__ == "__main__":
    plot_nmae_all_dimensions()
