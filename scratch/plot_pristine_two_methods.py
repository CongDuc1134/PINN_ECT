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

# Numerical Data for 2 Protocols: [Protocol 1 (Repeat Scan), Protocol 2 (10-Fold LODO)]
# Classification: Accuracy (%) & Macro F1 (%) - Xiong is None (Pure 3D Regression)
acc_vals = {
    "CNN PINN\n(PI-LGL)":       [80.0, 60.0],
    "CNN Baseline\n(NoPINN)":   [90.0, 55.0],
    "Multitask MLP\n(1D PINN)": [30.8, 12.5],
    "Xiong et al.\nMLP (1D)":   [None, None]
}

f1_vals = {
    "CNN PINN\n(PI-LGL)":       [72.50, 52.64],
    "CNN Baseline\n(NoPINN)":   [85.00, 39.31],
    "Multitask MLP\n(1D PINN)": [22.07, 10.50],
    "Xiong et al.\nMLP (1D)":   [None, None]
}

# Crack Length L MAE (mm)
l_mae_vals = {
    "CNN PINN\n(PI-LGL)":       [0.448, 0.524],
    "CNN Baseline\n(NoPINN)":   [1.192, 0.771],
    "Multitask MLP\n(1D PINN)": [6.110, 9.020],
    "Xiong et al.\nMLP (1D)":   [1546.9, 1538.98]
}

# Crack Width W MAE (mm)
w_mae_vals = {
    "CNN PINN\n(PI-LGL)":       [0.081, 0.105],
    "CNN Baseline\n(NoPINN)":   [0.073, 0.098],
    "Multitask MLP\n(1D PINN)": [0.363, 0.201],
    "Xiong et al.\nMLP (1D)":   [69.81, 1.320]
}

# Crack Depth D MAE (mm)
d_mae_vals = {
    "CNN PINN\n(PI-LGL)":       [0.672, 0.612],
    "CNN Baseline\n(NoPINN)":   [0.464, 0.574],
    "Multitask MLP\n(1D PINN)": [2.377, 2.910],
    "Xiong et al.\nMLP (1D)":   [248.5, 250.10]
}

def plot_pristine_head_to_head_external_legends():
    # Figure setup with ample breathing space
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.5), dpi=300)
    plt.subplots_adjust(wspace=0.22, hspace=0.34, top=0.81, bottom=0.08)
    x = np.arange(len(all_models))

    # =========================================================================
    # PANEL (a): Classification Performance (Acc & Macro F1) across 2 Protocols
    # =========================================================================
    ax = axs[0, 0]
    w_clf = 0.18
    
    # 3 models have classification
    x_sub = np.arange(3)
    acc_m1 = [acc_vals[all_models[i]][0] for i in range(3)]
    f1_m1  = [f1_vals[all_models[i]][0] for i in range(3)]
    acc_m2 = [acc_vals[all_models[i]][1] for i in range(3)]
    f1_m2  = [f1_vals[all_models[i]][1] for i in range(3)]

    ax.bar(x_sub - 1.5*w_clf, acc_m1, w_clf, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x_sub - 0.5*w_clf, f1_m1,  w_clf, color=C_METHOD1, alpha=0.45, hatch="//", edgecolor="black", lw=0.7)
    ax.bar(x_sub + 0.5*w_clf, acc_m2, w_clf, color=C_METHOD2, edgecolor="black", lw=0.7)
    ax.bar(x_sub + 1.5*w_clf, f1_m2,  w_clf, color=C_METHOD2, alpha=0.45, hatch="//", edgecolor="black", lw=0.7)

    # Clean unobtrusive placeholder for Xiong
    ax.text(3, 15, "N/A*\n(Pure 3D\nRegression)", ha="center", va="center", fontsize=8.2, 
            fontweight="bold", color="#64748b", bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", lw=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel("Rate (%)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Defect Classification (Accuracy & Macro F1)", pad=10)
    ax.grid(axis="y")

    # =========================================================================
    # PANEL (b): Crack Length Sizing Error (L MAE) across 2 Protocols - ALL 4 MODELS
    # =========================================================================
    ax = axs[0, 1]
    w_bar = 0.34

    m1_l = [l_mae_vals[m][0] for m in all_models]
    m2_l = [l_mae_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_l, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_l, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(0.1, 4000)
    ax.axhline(0.50, color="#2e7d32", ls="--", lw=1.2)
    ax.axhline(1.00, color="#d32f2f", ls="-.", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel("Length MAE $L$ (mm) [Log Scale]", fontweight="bold")
    ax.set_title("(b) Crack Length Sizing Error ($L$ MAE)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # PANEL (c): Crack Width Sizing Error (W MAE) across 2 Protocols - ALL 4 MODELS
    # =========================================================================
    ax = axs[1, 0]
    m1_w = [w_mae_vals[m][0] for m in all_models]
    m2_w = [w_mae_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_w, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_w, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(0.02, 200)
    ax.axhline(0.20, color="#2e7d32", ls="--", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel("Width MAE $W$ (mm) [Log Scale]", fontweight="bold")
    ax.set_title("(c) Crack Width Sizing Error ($W$ MAE)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # PANEL (d): Crack Depth Sizing Error (D MAE) across 2 Protocols - ALL 4 MODELS
    # =========================================================================
    ax = axs[1, 1]
    m1_d = [d_mae_vals[m][0] for m in all_models]
    m2_d = [d_mae_vals[m][1] for m in all_models]

    ax.bar(x - w_bar/2, m1_d, w_bar, color=C_METHOD1, edgecolor="black", lw=0.7)
    ax.bar(x + w_bar/2, m2_d, w_bar, color=C_METHOD2, edgecolor="black", lw=0.7)

    ax.set_yscale("log")
    ax.set_ylim(0.1, 800)
    ax.axhline(0.50, color="#2e7d32", ls="--", lw=1.2)
    ax.axhline(1.00, color="#d32f2f", ls="-.", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontweight="bold")
    ax.set_ylabel("Depth MAE $D$ (mm) [Log Scale]", fontweight="bold")
    ax.set_title("(d) Crack Depth Sizing Error ($D$ MAE)", pad=10)
    ax.grid(axis="y", which="both")

    # =========================================================================
    # SINGLE UNIFIED EXTERNAL LEGEND STRIP (AT THE TOP OUTSIDE ALL SUBPLOTS)
    # =========================================================================
    unified_handles = [
        Patch(facecolor=C_METHOD1, edgecolor="black", lw=0.8, 
              label="Protocol 1: Known Specimen (Repeat Scan)"),
        Patch(facecolor=C_METHOD2, edgecolor="black", lw=0.8, 
              label="Protocol 2: Unseen Specimen (10-Fold LODO)"),
        Line2D([0], [0], color="#2e7d32", ls="--", lw=1.4, 
               label="Strict NDT Standard (< 0.50 mm / < 0.20 mm)"),
        Patch(facecolor=C_METHOD1, alpha=0.45, hatch="//", edgecolor="black", lw=0.8, 
              label="Protocol 1: Macro F1 Rate [Subplot (a)]"),
        Patch(facecolor=C_METHOD2, alpha=0.45, hatch="//", edgecolor="black", lw=0.8, 
              label="Protocol 2: Macro F1 Rate [Subplot (a)]"),
        Line2D([0], [0], color="#d32f2f", ls="-.", lw=1.4, 
               label="Maximum NDT Tolerance Limit (< 1.00 mm)")
    ]

    # Exactly 3 columns x 2 rows, centered above all subplots
    fig.legend(handles=unified_handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=3, frameon=True, fontsize=8.2, edgecolor="#cbd5e1", framealpha=0.96,
               handletextpad=0.6, columnspacing=1.5)

    # Figure Suptitle at the very top
    fig.suptitle("HEAD-TO-HEAD COMPARISON ACROSS EVALUATION PROTOCOLS ON ALL MODEL ARCHITECTURES\nComprehensive Benchmark: Defect Classification, Length L, Width W, and Depth D Sizing", 
                 fontsize=12.0, fontweight="bold", y=0.985)

    # Footnote at the very bottom outside the subplots
    fig.text(0.5, 0.02, "*Note: Xiong et al. (2023) is a single-task 3D regression architecture ($W, L, D$) without classification capability; thus, N/A in Subplot (a).", 
             ha="center", fontsize=8.5, fontstyle="italic", color="#334155")

    out_file1 = os.path.join(OUTPUT_DIR, "fig_all_models_two_methods_head_to_head.png")
    out_file2 = os.path.join(OUTPUT_DIR, "fig_two_methods_head_to_head.png")
    
    plt.savefig(out_file1)
    plt.savefig(out_file2)
    plt.close()
    print(f"[OK] Successfully saved pristine head-to-head plots with external unified legends to:\n  - {out_file1}\n  - {out_file2}")

if __name__ == "__main__":
    plot_pristine_head_to_head_external_legends()
