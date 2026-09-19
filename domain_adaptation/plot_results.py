#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/plot_results.py
Publication-Grade Automated Plotting Engine (IEEE Transactions Standard)

Generates 7 IEEE-formatted figures for individual models at 300 DPI:
1. Fig 0: Consolidated Master Evaluation Dashboard
2. Fig 1: Overall Classification Accuracy & Regression MAE (W, L, D)
3. Fig 2: Per-Class Regression MAE Breakdown across Shapes
4. Fig 3: Per-Class Classification Accuracy Breakdown
5. Fig 4: Parity Scatter Plots (Ground Truth vs Prediction for W, L, D)
6. Fig 5: Confusion Matrix Heatmaps across Methods
7. Fig 6: Normalized MAE (NMAE %) Dimensionless Comparison

Plus Global Multi-Model Comparison Figures (300 DPI):
- fig_global_architecture_comparison.png
- fig_global_pinn_vs_baseline.png
- fig_global_method_ranking.png

IEEE Style Rules Applied:
- Typography: Serif (Times New Roman / STIX Math font)
- Dimensions: Single Column (3.5 in) and Double Column (7.16 in)
- Ticks: Inward-facing ticks, dual-axis ticks enabled
- Spines: 0.8 pt dark border
- Resolution: High-resolution 300 DPI
================================================================================
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. IEEE TRANSACTIONS TYPOGRAPHY & STYLE CONFIGURATION
# ==============================================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sklearn.metrics import confusion_matrix
from domain_adaptation.utils import UNIQUE_SHAPES

# Standard Method Display Names, Order & IEEE Color Palette
METHOD_ORDER = [
    "Real_Only_Scratch",
    "Source_Only_ZeroShot",
    "FewShot_PEFT",
    "Domain_Transfer_MMD",
    "Physics_TTA",
    "Physics_Informed_MMD",
]

METHOD_LABELS = {
    "Real_Only_Scratch": "Real-Only (Scratch)",
    "Source_Only_ZeroShot": "Source-Only (Zero-Shot)",
    "FewShot_PEFT": "Few-Shot PEFT",
    "Domain_Transfer_MMD": "Domain Transfer MMD",
    "Physics_TTA": "Physics-TTA",
    "Physics_Informed_MMD": "PI-MMD (Proposed)",
}

PALETTE = {
    "Real_Only_Scratch": "#8c564b",         # Brown (Ablation)
    "Source_Only_ZeroShot": "#7f7f7f",      # Neutral Gray (Sim-to-Real Gap)
    "FewShot_PEFT": "#1f77b4",              # Deep Blue (Standard PEFT)
    "Domain_Transfer_MMD": "#2ca02c",       # Forest Green (Statistical MMD)
    "Physics_TTA": "#d62728",               # Crimson Red (TTA)
    "Physics_Informed_MMD": "#e6550d",      # Deep Amber / Vibrant Orange (Proposed Novel Method)
}


def find_all_evaluated_model_dirs(results_root=None):
    """Recursively finds all model directories containing lodo_oof_master_predictions.csv."""
    if results_root is None:
        results_root = os.path.join(SCRIPT_DIR, "results")
    if not os.path.exists(results_root):
        raise FileNotFoundError(f"Directory not found: {results_root}")

    pred_files = glob.glob(os.path.join(results_root, "**", "lodo_oof_master_predictions.csv"), recursive=True)
    model_dirs = sorted(list(set(os.path.dirname(p) for p in pred_files)))
    return model_dirs


def plot_domain_adaptation_results(target_dir):
    """Generates all 7 IEEE figures for a single model directory."""
    model_name = os.path.basename(target_dir)
    pred_path = os.path.join(target_dir, "lodo_oof_master_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df_preds = pd.read_csv(pred_path)
    plots_dir = os.path.join(target_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    available_methods = [m for m in METHOD_ORDER if m in df_preds["method"].unique()]

    # =========================================================================
    # FIGURE 1: OVERALL ACCURACY & REGRESSION MAE (DOUBLE COLUMN, 7.16 in)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))

    # (a) Overall Classification Accuracy
    acc_data = []
    for m in available_methods:
        sub = df_preds[df_preds["method"] == m]
        acc = 100.0 * sub["shape_correct"].mean()
        acc_data.append({"Method": METHOD_LABELS.get(m, m), "Accuracy": acc, "raw": m})
    df_acc = pd.DataFrame(acc_data)

    bars = axes[0].bar(
        range(len(df_acc)),
        df_acc["Accuracy"],
        color=[PALETTE[m] for m in df_acc["raw"]],
        edgecolor="#222222",
        linewidth=0.8,
        width=0.55,
    )
    axes[0].set_title("(a) Defect Classification Accuracy", fontweight="bold", pad=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(range(len(df_acc)))
    axes[0].set_xticklabels([m.replace(" ", "\n") for m in df_acc["Method"]], rotation=25, ha="right", fontsize=7.5)
    axes[0].set_ylim(0, max(65, df_acc["Accuracy"].max() + 15))
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # (b) Regression MAE for W, L, D
    reg_rows = []
    for m in available_methods:
        sub = df_preds[df_preds["method"] == m]
        reg_rows.append({"Method": METHOD_LABELS.get(m, m), "Dimension": "Width (W)", "MAE (mm)": sub["err_w"].mean(), "raw": m})
        reg_rows.append({"Method": METHOD_LABELS.get(m, m), "Dimension": "Length (L)", "MAE (mm)": sub["err_l"].mean(), "raw": m})
        reg_rows.append({"Method": METHOD_LABELS.get(m, m), "Dimension": "Depth (D)", "MAE (mm)": sub["err_d"].mean(), "raw": m})
    df_reg = pd.DataFrame(reg_rows)

    # Safe clipping for display if out-of-scale values exist (e.g. Xiong MLP Depth)
    max_raw_mae = df_reg["MAE (mm)"].max()
    display_cap = 10.0 if max_raw_mae > 15.0 else max(3.5, max_raw_mae * 1.15)

    sns.barplot(
        data=df_reg,
        x="Dimension",
        y="MAE (mm)",
        hue="Method",
        palette=[PALETTE[m] for m in available_methods],
        edgecolor="#222222",
        linewidth=0.8,
        ax=axes[1],
    )
    axes[1].set_title("(b) Dimension Regression Error (MAE)", fontweight="bold", pad=8)
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, display_cap)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=6.5, loc="upper right")

    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            txt = f"{h:.2f}" if h <= display_cap else f">{display_cap:.0f}"
            axes[1].annotate(txt, (p.get_x() + p.get_width() / 2.0, min(h, display_cap - 0.5)),
                             ha="center", va="bottom", fontsize=6.5, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    fig1_path = os.path.join(plots_dir, "fig1_overall_acc_and_wld_mae.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 2: PER-CLASS REGRESSION MAE BREAKDOWN (7.16 in x 2.8 in)
    # =========================================================================
    crack_classes = ["Rectangular", "Ellipse", "Triangular", "Step_R", "Step_T"]
    classes_present = [c for c in crack_classes if c in df_preds["true_shape"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), sharey=False)
    dims = [("err_w", "Width (W)", axes[0]), ("err_l", "Length (L)", axes[1]), ("err_d", "Depth (D)", axes[2])]

    class_metric_list = []
    for cls in classes_present:
        for m in available_methods:
            sub = df_preds[(df_preds["true_shape"] == cls) & (df_preds["method"] == m)]
            if len(sub) > 0:
                class_metric_list.append({
                    "Class": cls,
                    "Method": METHOD_LABELS.get(m, m),
                    "MAE_W": sub["err_w"].mean(),
                    "MAE_L": sub["err_l"].mean(),
                    "MAE_D": sub["err_d"].mean(),
                    "Accuracy": 100.0 * sub["shape_correct"].mean(),
                    "raw": m,
                })
    df_class = pd.DataFrame(class_metric_list)

    for err_col, title_dim, ax in dims:
        val_col = err_col.replace("err", "MAE").upper()
        max_v = df_class[val_col].max()
        cap_v = 10.0 if max_v > 12.0 else max(2.5, max_v * 1.15)

        sns.barplot(
            data=df_class,
            x="Class",
            y=val_col,
            hue="Method",
            palette=[PALETTE[m] for m in available_methods],
            edgecolor="#222222",
            linewidth=0.8,
            ax=ax,
        )
        ax.set_title(f"MAE for {title_dim}", fontweight="bold", pad=6)
        ax.set_ylabel(f"{title_dim} MAE (mm)", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylim(0, cap_v)
        ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(available_methods),
               bbox_to_anchor=(0.5, 1.06), frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.0)

    plt.tight_layout()
    fig2_path = os.path.join(plots_dir, "fig2_per_class_wld_mae.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 3: PER-CLASS ACCURACY (SINGLE COLUMN, 3.5 in)
    # =========================================================================
    plt.figure(figsize=(3.5, 2.7))
    ax = sns.barplot(
        data=df_class,
        x="Class",
        y="Accuracy",
        hue="Method",
        palette=[PALETTE[m] for m in available_methods],
        edgecolor="#222222",
        linewidth=0.8,
    )
    plt.title("Per-Class Accuracy (%)", fontweight="bold", pad=8)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("")
    plt.ylim(0, 115)
    plt.xticks(rotation=25, ha="right", fontsize=7.5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=6.5, loc="upper right")
    plt.tight_layout()
    fig3_path = os.path.join(plots_dir, "fig3_per_class_accuracy.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 4: PARITY PLOTS (7.16 in x 2.6 in)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    dim_configs = [
        ("true_w", "pred_w", "Width W (mm)", axes[0]),
        ("true_l", "pred_l", "Length L (mm)", axes[1]),
        ("true_d", "pred_d", "Depth D (mm)", axes[2]),
    ]

    for true_col, pred_col, label_name, ax in dim_configs:
        t_vals = df_preds[true_col].values
        p_vals = df_preds[pred_col].values
        # Filter finite for safe range
        valid_mask = np.isfinite(p_vals) & (np.abs(p_vals) < 30.0)
        safe_p = p_vals[valid_mask] if valid_mask.sum() > 0 else t_vals
        min_v = min(np.min(t_vals), np.min(safe_p)) * 0.9
        max_v = max(np.max(t_vals), np.max(safe_p)) * 1.1

        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.6, linewidth=0.9, label="Ideal (y=x)")

        for m in available_methods:
            sub = df_preds[df_preds["method"] == m]
            ax.scatter(
                sub[true_col],
                sub[pred_col],
                label=METHOD_LABELS.get(m, m),
                color=PALETTE[m],
                alpha=0.8,
                edgecolors="none",
                s=25,
            )

        ax.set_title(f"Parity: {label_name}", fontweight="bold", pad=6)
        ax.set_xlabel(f"True {label_name}", fontsize=8.5)
        ax.set_ylabel(f"Predicted {label_name}", fontsize=8.5)
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=6.0, loc="upper left")
    plt.tight_layout()
    fig4_path = os.path.join(plots_dir, "fig4_parity_scatter_wld.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 5: CONFUSION MATRIX HEATMAPS (7.16 in x 2.4 in)
    # =========================================================================
    n_m = len(available_methods)
    fig, axes = plt.subplots(1, n_m, figsize=(1.6 * n_m + 1.2, 2.6), squeeze=False)
    for idx_m, m in enumerate(available_methods):
        ax = axes[0, idx_m]
        sub = df_preds[df_preds["method"] == m]
        cm = confusion_matrix(sub["true_shape"], sub["pred_shape"], labels=UNIQUE_SHAPES)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[s[:3] for s in UNIQUE_SHAPES],
            yticklabels=[s[:3] for s in UNIQUE_SHAPES] if idx_m == 0 else False,
            ax=ax,
            linewidths=0.4,
            linecolor="#EAEAEA",
            annot_kws={"fontsize": 7.5},
        )
        acc_val = 100.0 * sub["shape_correct"].mean()
        short_title = METHOD_LABELS.get(m, m).replace(" (Proposed)", "*")
        ax.set_title(f"{short_title}\n({acc_val:.0f}%)", fontsize=7.5, fontweight="bold", pad=6)
        ax.set_xlabel("Pred", fontsize=8.0)
        if idx_m == 0:
            ax.set_ylabel("True", fontsize=8.0)

    plt.tight_layout()
    fig5_path = os.path.join(plots_dir, "fig5_confusion_matrices.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 6: NORMALIZED MAE (NMAE %) COMPARISON (3.5 in x 2.7 in)
    # =========================================================================
    w_range, d_range, l_nominal = 0.4, 2.0, 10.0
    nmae_rows = []
    for m in available_methods:
        sub = df_preds[df_preds["method"] == m]
        nmae_rows.append({"Method": METHOD_LABELS.get(m, m), "Dimension": "W", "NMAE (%)": (sub["err_w"].mean() / w_range) * 100.0, "raw": m})
        nmae_rows.append({"Method": METHOD_LABELS.get(m, m), "Dimension": "L", "NMAE (%)": (sub["err_l"].mean() / l_nominal) * 100.0, "raw": m})
        nmae_rows.append({"Method": METHOD_LABELS.get(m, m), "Dimension": "D", "NMAE (%)": (sub["err_d"].mean() / d_range) * 100.0, "raw": m})
    df_nmae = pd.DataFrame(nmae_rows)

    plt.figure(figsize=(3.5, 2.7))
    ax = sns.barplot(
        data=df_nmae,
        x="Dimension",
        y="NMAE (%)",
        hue="Method",
        palette=[PALETTE[m] for m in available_methods],
        edgecolor="#222222",
        linewidth=0.8,
    )
    plt.title("Normalized MAE (NMAE %)", fontweight="bold", pad=8)
    plt.ylabel("NMAE (%)")
    plt.xlabel("Parameter")
    plt.ylim(0, min(120, max(50, df_nmae["NMAE (%)"].max() + 15)))
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=6.0, loc="upper right")
    plt.tight_layout()
    fig6_path = os.path.join(plots_dir, "fig6_nmae_comparison.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 0: CONSOLIDATED MASTER DASHBOARD (7.16 in x 5.2 in)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))

    # Top-Left: Accuracy
    bars0 = axes[0, 0].bar(
        range(len(df_acc)),
        df_acc["Accuracy"],
        color=[PALETTE[m] for m in df_acc["raw"]],
        edgecolor="#222222",
        linewidth=0.8,
        width=0.55,
    )
    axes[0, 0].set_title("(A) Classification Accuracy (%)", fontweight="bold", pad=6)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xticks(range(len(df_acc)))
    axes[0, 0].set_xticklabels([m.replace(" ", "\n") for m in df_acc["Method"]], rotation=25, ha="right", fontsize=7.0)
    axes[0, 0].set_ylim(0, max(65, df_acc["Accuracy"].max() + 15))
    axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in bars0:
        h = b.get_height()
        axes[0, 0].text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold")

    # Top-Right: MAE
    sns.barplot(
        data=df_reg,
        x="Dimension",
        y="MAE (mm)",
        hue="Method",
        palette=[PALETTE[m] for m in available_methods],
        edgecolor="#222222",
        linewidth=0.8,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("(B) Overall MAE (W, L, D)", fontweight="bold", pad=6)
    axes[0, 1].set_ylabel("MAE (mm)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylim(0, display_cap)
    axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0, 1].get_legend().remove()

    # Bottom-Left: Per-Class Accuracy
    sns.barplot(
        data=df_class,
        x="Class",
        y="Accuracy",
        hue="Method",
        palette=[PALETTE[m] for m in available_methods],
        edgecolor="#222222",
        linewidth=0.8,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("(C) Per-Class Accuracy (%)", fontweight="bold", pad=6)
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].tick_params(axis="x", rotation=25, labelsize=7.5)
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1, 0].get_legend().remove()

    # Bottom-Right: NMAE
    sns.barplot(
        data=df_nmae,
        x="Dimension",
        y="NMAE (%)",
        hue="Method",
        palette=[PALETTE[m] for m in available_methods],
        edgecolor="#222222",
        linewidth=0.8,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("(D) Normalized MAE (NMAE %)", fontweight="bold", pad=6)
    axes[1, 1].set_ylabel("NMAE (%)")
    axes[1, 1].set_xlabel("Parameter")
    axes[1, 1].set_ylim(0, min(120, max(50, df_nmae["NMAE (%)"].max() + 15)))
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1, 1].get_legend().remove()

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(available_methods),
               bbox_to_anchor=(0.5, 1.04), frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.5)

    plt.suptitle(f"Consolidated Benchmark Dashboard (10-Fold LODO OOF)\nModel: {model_name[:70]}", fontweight="bold", y=1.07, fontsize=10.0)
    plt.tight_layout()
    fig0_path = os.path.join(plots_dir, "fig0_master_overview.png")
    plt.savefig(fig0_path, dpi=300)
    plt.close()

    print(f"[SAVED] All 7 IEEE Figures for: {model_name}")


def generate_global_ieee_plots(master_csv=None, output_dir=None):
    """Generates overarching cross-model IEEE figures comparing architectures and methods."""
    if master_csv is None:
        master_csv = os.path.join(SCRIPT_DIR, "results", "master_all_models_summary.csv")
    if not os.path.exists(master_csv):
        return

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results", "plots_global")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(master_csv)

    # 1. GLOBAL FIGURE 1: Architecture Comparison (CNN vs Multitask MLP vs Xiong MLP)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))

    arch_map = {"CNN": "CNN", "MLP_Multitask": "MLP (Multitask)", "MLP_Xiong": "MLP (Xiong et al.)"}
    df_copy = df.copy()
    df_copy["Arch_Clean"] = df_copy["Architecture"].map(arch_map).fillna(df_copy["Architecture"])
    df_copy["Method_Clean"] = df_copy["Method"].map(METHOD_LABELS).fillna(df_copy["Method"])

    arch_acc = df_copy.groupby(["Arch_Clean", "Method_Clean"])["Clf_Accuracy (%)"].mean().reset_index()
    sns.barplot(
        data=arch_acc,
        x="Arch_Clean",
        y="Clf_Accuracy (%)",
        hue="Method_Clean",
        palette=[PALETTE[m] for m in METHOD_ORDER if m in df["Method"].unique()],
        edgecolor="#222222",
        linewidth=0.8,
        ax=axes[0],
    )
    axes[0].set_title("(a) Accuracy Across Architectures", fontweight="bold", pad=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 65)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].get_legend().remove()

    arch_nmae = df_copy.groupby(["Arch_Clean", "Method_Clean"])["Overall_NMAE (%)"].mean().reset_index()
    # Cap Xiong extreme NMAE for visual clarity
    arch_nmae["NMAE_Capped"] = arch_nmae["Overall_NMAE (%)"].clip(upper=150.0)

    sns.barplot(
        data=arch_nmae,
        x="Arch_Clean",
        y="NMAE_Capped",
        hue="Method_Clean",
        palette=[PALETTE[m] for m in METHOD_ORDER if m in df["Method"].unique()],
        edgecolor="#222222",
        linewidth=0.8,
        ax=axes[1],
    )
    axes[1].set_title("(b) Overall NMAE (%) by Architecture", fontweight="bold", pad=8)
    axes[1].set_ylabel("Overall NMAE (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 160)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=6.5, loc="upper right")

    plt.tight_layout()
    p1 = os.path.join(output_dir, "fig_global_architecture_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[SAVED] Global Fig 1: {p1}")

    # 2. GLOBAL FIGURE 2: CNN Baseline vs PINN across Simulation Budgets (1% - 10%)
    cnn_df = df[df["Architecture"] == "CNN"].copy()
    if len(cnn_df) > 0:
        def get_pct(name):
            for p in ["01pct", "03pct", "05pct", "07pct", "10pct"]:
                if p in name:
                    return p.replace("0", "").replace("pct", "%")
            return "5%"

        cnn_df["Budget"] = cnn_df["Evaluated_Model"].apply(get_pct)
        cnn_df["Model_Type"] = cnn_df["Evaluated_Model"].apply(lambda x: "Baseline (NoPINN)" if "NoPINN" in x else "PINN")

        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
        cnn_mmd = cnn_df[cnn_df["Method"] == "Domain_Transfer_MMD"]

        sns.barplot(
            data=cnn_mmd,
            x="Budget",
            y="Clf_Accuracy (%)",
            hue="Model_Type",
            palette=["#7f7f7f", "#1f77b4"],
            edgecolor="#222222",
            linewidth=0.8,
            order=["1%", "3%", "5%", "7%", "10%"],
            ax=axes[0],
        )
        axes[0].set_title("(a) Accuracy across Simulation Budgets", fontweight="bold", pad=8)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_xlabel("Source Pre-training Budget")
        axes[0].set_ylim(0, 65)
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
        axes[0].legend(frameon=True, fontsize=7.0)

        sns.barplot(
            data=cnn_mmd,
            x="Budget",
            y="Overall_MAE (mm)",
            hue="Model_Type",
            palette=["#7f7f7f", "#1f77b4"],
            edgecolor="#222222",
            linewidth=0.8,
            order=["1%", "3%", "5%", "7%", "10%"],
            ax=axes[1],
        )
        axes[1].set_title("(b) Overall MAE (mm) across Budgets", fontweight="bold", pad=8)
        axes[1].set_ylabel("Overall MAE (mm)")
        axes[1].set_xlabel("Source Pre-training Budget")
        axes[1].set_ylim(0, 1.2)
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
        axes[1].get_legend().remove()

        plt.tight_layout()
        p2 = os.path.join(output_dir, "fig_global_pinn_vs_baseline.png")
        plt.savefig(p2, dpi=300)
        plt.close()
        print(f"[SAVED] Global Fig 2: {p2}")


def plot_all_models():
    """Batch plots all available evaluated models and overarching IEEE comparison figures."""
    model_dirs = find_all_evaluated_model_dirs()
    print("=" * 85)
    print(f"BATCH IEEE PLOTTING: Generating 7 figures at 300 DPI for ALL {len(model_dirs)} models")
    print("=" * 85)

    for idx, d in enumerate(model_dirs, 1):
        tag = os.path.basename(d)
        print(f"[{idx:2d}/{len(model_dirs)}] Plotting: {tag}")
        try:
            plot_domain_adaptation_results(d)
        except Exception as e:
            print(f"  [WARN] Failed to plot {tag}: {e}")

    # Generate overarching global figures
    print("\n[INFO] Generating overarching Global Comparison Figures (300 DPI)...")
    generate_global_ieee_plots()
    print("=" * 85)
    print("[FINISHED] All figures generated successfully in IEEE format!")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ phân tích chuẩn IEEE cho Domain Adaptation")
    parser.add_argument("--all", action="store_true", default=False, help="Tự động vẽ lại cho toàn bộ 43 models trong kết quả")
    parser.add_argument("--target-dir", type=str, default=None, help="Thư mục kết quả cụ thể cần vẽ")
    args = parser.parse_args()

    if args.all or (args.target_dir is None):
        plot_all_models()
    else:
        plot_domain_adaptation_results(target_dir=args.target_dir)
