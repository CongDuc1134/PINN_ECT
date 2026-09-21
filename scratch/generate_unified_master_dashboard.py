# -*- coding: utf-8 -*-
"""
scratch/generate_unified_master_dashboard.py
Generates a publication-grade Unified Master Dashboard combining:
1. Panel (a): Multi-Objective Pareto Frontier Trade-Off Map (All models on one 2D plane)
2. Panel (b): 5-Model Multi-Metric Radar / Spider Chart (All models on one polar plot)
3. Panel (c): Unified Grouped All-Dimension & Accuracy Bar Chart (All models side-by-side)
4. Panel (d): Unified Multi-Model Scaling Progression (1% to 10%)

Adheres strictly to IEEE Transactions formatting:
- 300 DPI
- Times New Roman
- Dual inward ticks
- Unified comparative view
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Polygon

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR_ARCH = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_architecture_comparison")
OUT_DIR_COMP = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_final_comprehensive")
os.makedirs(OUT_DIR_ARCH, exist_ok=True)
os.makedirs(OUT_DIR_COMP, exist_ok=True)

# ---------------------------------------------------------
# Global IEEE Styling
# ---------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "figure.titlesize": 11.5,
    "figure.titleweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_NOPINN      = "#d95f02"  # Amber Orange
C_PINN        = "#0f4c81"  # Deep Navy Blue
C_MLPPINN     = "#1b9e77"  # Forest Green
C_XIONG       = "#d62728"  # Crimson Red
C_MLP_NOPINN  = "#7570b3"  # Slate Purple

COLORS_5 = [C_NOPINN, C_PINN, C_MLPPINN, C_XIONG, C_MLP_NOPINN]
MODEL_NAMES_5 = ["NOPINN", "PINN", "MLPPINN", "XIONG", "MLP NO PINN"]

# 5% Data Scale (Seed 42) Values
vals_5pct = {
    "NOPINN":      {"Acc": 90.0, "MAE_W": 0.073, "MAE_L": 1.192, "MAE_D": 0.464, "Overall": 0.576},
    "PINN":        {"Acc": 80.0, "MAE_W": 0.081, "MAE_L": 0.448, "MAE_D": 0.672, "Overall": 0.401},
    "MLPPINN":     {"Acc": 30.0, "MAE_W": 0.117, "MAE_L": 8.729, "MAE_D": 1.047, "Overall": 3.297},
    "XIONG":       {"Acc": 30.0, "MAE_W": 99.222, "MAE_L": 3158.814, "MAE_D": 86.901, "Overall": 1114.979},
    "MLP NO PINN": {"Acc": 10.0, "MAE_W": 0.096, "MAE_L": 2.348, "MAE_D": 1.809, "Overall": 1.418},
}

def plot_master_dashboard():
    fig = plt.figure(figsize=(14.0, 9.5), dpi=300)
    
    # 2x2 Grid with custom polar for panel b
    ax1 = fig.add_subplot(2, 2, 1)                  # Pareto Frontier
    ax2 = fig.add_subplot(2, 2, 2, polar=True)      # Radar Chart
    ax3 = fig.add_subplot(2, 2, 3)                  # Unified Grouped Bar (Acc + W, L, D)
    ax4 = fig.add_subplot(2, 2, 4)                  # Unified Scaling Progression
    
    # =========================================================================
    # PANEL (a): Multi-Objective Pareto Frontier Trade-Off Map
    # =========================================================================
    # Load all models points
    master_csv = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "master_all_models_summary.csv")
    pi_lgl_mlp_csv = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "pi_lgl_all_mlp_models_summary.csv")
    pi_lgl_cnn_csv = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "pi_lgl_all_19_models_summary.csv")
    
    df_cnn = pd.read_csv(pi_lgl_cnn_csv)
    df_mlp = pd.read_csv(pi_lgl_mlp_csv)
    
    # NOPINN points
    nopinn_pts = df_cnn[df_cnn["Model_Type"] == "Baseline"]
    ax1.scatter(nopinn_pts["MAE_L (mm)"], nopinn_pts["Clf_Accuracy (%)"], 
                color=C_NOPINN, marker="s", s=65, alpha=0.85, label="NOPINN Checkpoints", zorder=4)
                
    # PINN points
    pinn_pts = df_cnn[df_cnn["Model_Type"] == "PINN"]
    ax1.scatter(pinn_pts["MAE_L (mm)"], pinn_pts["Clf_Accuracy (%)"], 
                color=C_PINN, marker="o", s=80, alpha=0.9, label="PINN Checkpoints", zorder=5)
                
    # MLPPINN points
    mlp_pts = df_mlp[(df_mlp["Architecture"] == "MLP_Multitask") & (~df_mlp["Model_Tag"].str.contains("alpha_a0"))]
    ax1.scatter(mlp_pts["MAE_L (mm)"], mlp_pts["Clf_Accuracy (%)"], 
                color=C_MLPPINN, marker="^", s=65, alpha=0.85, label="MLPPINN Checkpoints", zorder=4)
                
    # MLP NO PINN points
    mlp_nopinn_pts = df_mlp[df_mlp["Model_Tag"].str.contains("alpha_a0")]
    ax1.scatter(mlp_nopinn_pts["MAE_L (mm)"], mlp_nopinn_pts["Clf_Accuracy (%)"], 
                color=C_MLP_NOPINN, marker="X", s=90, label="MLP NO PINN (α=0)", zorder=5)
                
    # Highlight Ideal NDT Target Zone (High Acc >= 80%, Low Error <= 1.0 mm)
    ax1.axvspan(0.1, 1.0, ymin=0.75, ymax=1.05, color="#dcfce7", alpha=0.5, zorder=1)
    ax1.text(0.35, 96, "NDT Qualified\nPareto Zone", color="#15803d", fontweight="bold", fontsize=7.8, ha="center")
    
    # Highlight Shortcut Learning Zone (High Acc >= 80%, High Error > 1.0 mm)
    ax1.axvspan(1.0, 2.0, ymin=0.75, ymax=1.05, color="#fef3c7", alpha=0.35, zorder=1)
    ax1.text(1.4, 96, "Shortcut Learning\n(NoPINN Trap)", color="#b45309", fontweight="bold", fontsize=7.5, ha="center")
    
    # Highlight Catastrophic Failure Zone (> 100 mm)
    ax1.axvspan(100, 6000, color="#fee2e2", alpha=0.4, zorder=1)
    ax1.text(600, 45, "Catastrophic Divergence\n(Xiong et al. MLP)", color="#b91c1c", fontweight="bold", fontsize=7.8, ha="center")
    
    # Footnote about Xiong: Single-Task Regression (No Classifier)
    ax1.text(0.22, 3, "*Xiong et al. (2023) là mô hình thuần hồi quy (W,L,D), không có nhánh phân loại (không có Accuracy).", 
             fontsize=6.5, fontstyle="italic", color="#475569")
    
    # Tolerance threshold line
    ax1.axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.0, label="NDT Tolerance (1.0 mm)")
    
    ax1.set_xscale("log")
    ax1.set_xlim(0.2, 6000)
    ax1.set_ylim(0, 110)
    ax1.set_xlabel("Length ($L$) Sizing MAE [mm] (Log Scale - Lower is Better)")
    ax1.set_ylabel("Classification Accuracy (%) (Higher is Better)")
    ax1.set_title("(a) Multi-Objective Pareto Frontier: Accuracy vs Length Error", pad=6)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True, edgecolor="#cbd5e1", fontsize=7.0, ncol=2)
    ax1.tick_params(direction="in", which="both", top=True, right=True)

    # =========================================================================
    # PANEL (b): 5-Model Multi-Metric Radar / Spider Chart
    # =========================================================================
    radar_labels = [
        "Classification\nAccuracy",
        "Width Precision\n(1 / MAE_W)",
        "Length Precision\n(1 / MAE_L)",
        "Depth Precision\n(1 / MAE_D)",
        "Overall Sizing\nFidelity"
    ]
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Compute normalized scores (0 to 100) for the 5 models
    def get_radar_scores(m):
        v = vals_5pct[m]
        s_acc = 0.0 if m == "XIONG" else v["Acc"]  # Xiong has no classification task
        s_w = np.clip(1.0 / v["MAE_W"] * 5.0, 0, 100)
        s_l = np.clip(1.0 / v["MAE_L"] * 35.0, 0, 100)
        s_d = np.clip(1.0 / v["MAE_D"] * 45.0, 0, 100)
        s_tot = np.clip(1.0 / v["Overall"] * 30.0, 0, 100)
        return [s_acc, s_w, s_l, s_d, s_tot]

    # Plot Radar for models that have classification task (NOPINN, PINN, MLPPINN, MLP NO PINN)
    # Xiong is purely single-task regression (W, L, D), so it is excluded from classification radar
    radar_models = [m for m in MODEL_NAMES_5 if m != "XIONG"]
    radar_colors = [c for m, c in zip(MODEL_NAMES_5, COLORS_5) if m != "XIONG"]

    for m, c in zip(radar_models, radar_colors):
        scores = get_radar_scores(m)
        scores += scores[:1]
        lw = 2.2 if m == "PINN" else 1.4
        alpha_fill = 0.22 if m == "PINN" else 0.08
        ax2.plot(angles, scores, color=c, linewidth=lw, label=m)
        ax2.fill(angles, scores, color=c, alpha=alpha_fill)
        
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_labels, fontsize=7.5, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6.5, color="#64748b")
    ax2.set_title("(b) Multi-Metric Radar Chart\n(*Xiong excluded: Single-Task Reg)", pad=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=True, edgecolor="#cbd5e1", fontsize=7.0)

    # =========================================================================
    # PANEL (c): Unified Grouped All-Dimension & Accuracy Bar Chart
    # =========================================================================
    # Each model on x-axis, with grouped bars for Acc (divided by 10 for scale), W, L, D, Overall MAE
    x3 = np.arange(len(MODEL_NAMES_5))
    w3 = 0.16
    
    # Subplot with twin axes: Left = Accuracy (%), Right = Sizing Errors (mm)
    ax3_twin = ax3.twinx()
    
    # Accuracy on Left Axis (Xiong is Single-Task Regression -> Excluded from Acc)
    acc_plot_vals = [vals_5pct[m]["Acc"] if m != "XIONG" else 0.0 for m in MODEL_NAMES_5]
    acc_bars = ax3.bar(x3 - 2*w3, acc_plot_vals, w3,
                       color="#38bdf8", edgecolor="black", linewidth=0.5, label="Accuracy (%) [Left]", zorder=3)
    ax3.text(x3[3] - 2*w3, 3, "—\n(No Clf)", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#64748b")
                       
    # W, L, D, Overall on Right Axis (Log Scale)
    w_bars = ax3_twin.bar(x3 - w3, [vals_5pct[m]["MAE_W"] for m in MODEL_NAMES_5], w3,
                          color="#10b981", edgecolor="black", linewidth=0.5, label="Width ($W$) MAE [Right]", zorder=3)
    l_bars = ax3_twin.bar(x3, [vals_5pct[m]["MAE_L"] for m in MODEL_NAMES_5], w3,
                          color="#ef4444", edgecolor="black", linewidth=0.5, label="Length ($L$) MAE [Right]", zorder=3)
    d_bars = ax3_twin.bar(x3 + w3, [vals_5pct[m]["MAE_D"] for m in MODEL_NAMES_5], w3,
                          color="#8b5cf6", edgecolor="black", linewidth=0.5, label="Depth ($D$) MAE [Right]", zorder=3)
    tot_bars = ax3_twin.bar(x3 + 2*w3, [vals_5pct[m]["Overall"] for m in MODEL_NAMES_5], w3,
                            color="#f59e0b", edgecolor="black", linewidth=0.5, label="Overall MAE [Right]", zorder=3)

    ax3.set_ylabel("Classification Accuracy (%)", color="#0284c7")
    ax3.set_ylim(0, 110)
    ax3.tick_params(axis="y", labelcolor="#0284c7", direction="in")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(MODEL_NAMES_5, rotation=20, ha="right", fontweight="bold")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.4)
    
    ax3_twin.set_ylabel("Dimensional MAE [mm] (Log Scale)", color="#b91c1c")
    ax3_twin.set_yscale("log")
    ax3_twin.set_ylim(0.02, 5000)
    ax3_twin.axhline(1.0, color="#dc2626", linestyle="--", linewidth=0.8)
    ax3_twin.tick_params(axis="y", labelcolor="#b91c1c", direction="in")
    
    # Combined legend
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, frameon=True, edgecolor="#cbd5e1", fontsize=6.8)
    ax3.set_title("(c) Unified Accuracy & Spatial Sizing ($W, L, D$) across 5 Models", pad=6)

    # =========================================================================
    # PANEL (d): Unified Multi-Model Scaling Progression (1% to 10%)
    # =========================================================================
    scales = ["1%", "3%", "5%", "7%", "10%"]
    x4 = np.arange(len(scales))
    
    # Length Error progression for all models
    l_nopinn = [1.226, 0.875, 1.192, 1.207, 1.452]
    l_pinn   = [0.348, 0.507, 0.448, 1.268, 0.965]
    l_mlp    = [0.950, 7.819, 8.729, 10.609, 3.294]
    l_xiong  = [4383.5, 1228.9, 3158.8, 596.7, 8.3]
    
    ax4.plot(x4, l_pinn, "o-", color=C_PINN, linewidth=2.2, markersize=7, label="PINN (Proposed)", zorder=5)
    ax4.plot(x4, l_nopinn, "s--", color=C_NOPINN, linewidth=1.6, markersize=6, label="NOPINN (Baseline)", zorder=4)
    ax4.plot(x4, l_mlp, "^-.", color=C_MLPPINN, linewidth=1.5, markersize=6, label="MLPPINN", zorder=3)
    ax4.plot(x4, l_xiong, "v:", color=C_XIONG, linewidth=1.5, markersize=6, label="XIONG", zorder=2)
    ax4.scatter([2], [2.348], color=C_MLP_NOPINN, marker="X", s=90, zorder=6, label="MLP NO PINN (5%)")
    
    ax4.axhline(1.0, color="#dc2626", linestyle="--", linewidth=0.9, label="NDT Tolerance (1.0 mm)")
    ax4.set_xlabel("Simulation Training Data Scale")
    ax4.set_ylabel("Length ($L$) Sizing MAE [mm] (Log Scale)")
    ax4.set_title("(d) Length Error Scaling Trajectory across Data Ratios (1% to 10%)", pad=6)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(scales, fontweight="bold")
    ax4.set_yscale("log")
    ax4.set_ylim(0.2, 6000)
    ax4.grid(True, which="both", linestyle=":", alpha=0.5)
    ax4.legend(loc="upper left", ncol=2, frameon=True, edgecolor="#cbd5e1", fontsize=7.0)
    ax4.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle("MASTER UNIFIED BENCHMARK: Comprehensive 5-Model Evaluation\n(NOPINN vs PINN vs MLPPINN vs XIONG vs MLP NO PINN)", 
                 fontsize=12.0, fontweight="bold", y=0.995)
    fig.tight_layout()
    
    # Save to both target directories
    for d in [OUT_DIR_ARCH, OUT_DIR_COMP]:
        out_file = os.path.join(d, "fig_unified_master_dashboard.png")
        fig.savefig(out_file, dpi=300)
        # Also overwrite fig1 to keep seamless link
        fig.savefig(os.path.join(d, "fig1_cross_method_architecture_comparison.png"), dpi=300)
        
    plt.close(fig)
    print("[OK] Successfully generated Unified Master Dashboard!")

if __name__ == "__main__":
    plot_master_dashboard()
