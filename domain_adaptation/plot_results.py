#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_results.py
Tự động vẽ biểu đồ đối sánh cho các phương pháp Domain Adaptation:
1. Overall Classification Accuracy (%) & Regression MAE (W, L, D) mm
2. Per-class Regression MAE (W, L, D) cho từng hình dạng khuyết tật (Rectangular, Ellipse, Triangular, Step_R, Step_T)
3. Per-class Classification Accuracy (%)
4. Parity Plots (True vs Predicted) cho W, L, D

Lưu kết quả tự động vào thư mục: domain_adaptation/results/<model_tag>/plots/
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập phong cách học thuật chuẩn Q1 / IEEE
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#E0E0E0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sklearn.metrics import confusion_matrix
from domain_adaptation.utils import list_all_available_checkpoints, get_model_tag, UNIQUE_SHAPES


def find_target_results_dir(keyword=None):
    results_root = os.path.join(SCRIPT_DIR, "results")
    if not os.path.exists(results_root):
        raise FileNotFoundError(f"Chưa tìm thấy thư mục kết quả: {results_root}")

    # Lấy các thư mục con trong results
    subdirs = [
        d for d in glob.glob(os.path.join(results_root, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "lodo_oof_master_predictions.csv"))
    ]

    if not subdirs:
        raise FileNotFoundError(f"Không tìm thấy thư mục model nào có file lodo_oof_master_predictions.csv trong {results_root}")

    if keyword is not None:
        matched = [d for d in subdirs if keyword.lower() in os.path.basename(d).lower()]
        if matched:
            return matched[0]

    # Mặc định lấy thư mục mới nhất
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return subdirs[0]


def plot_domain_adaptation_results(target_dir=None, keyword=None):
    if target_dir is None:
        target_dir = find_target_results_dir(keyword)

    model_name = os.path.basename(target_dir)
    print(f"\n[PLOT] Target Results Dir: {target_dir}")
    print(f"[PLOT] Model Identifier   : {model_name}")

    pred_path = os.path.join(target_dir, "lodo_oof_master_predictions.csv")
    summary_path = os.path.join(target_dir, "lodo_oof_master_summary.csv")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing file: {pred_path}")

    df_preds = pd.read_csv(pred_path)
    df_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    # Thư mục lưu biểu đồ
    plots_dir = os.path.join(target_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Đặt thứ tự hiển thị phương pháp và màu sắc chuyên nghiệp
    method_order = [
        "Real_Only_Scratch",
        "Source_Only_ZeroShot",
        "FewShot_PEFT",
        "Domain_Transfer_MMD",
        "Physics_TTA",
        "Physics_Informed_MMD",
    ]
    # Lọc những phương pháp có trong data
    available_methods = [m for m in method_order if m in df_preds["method"].unique()]
    palette = {
        "Real_Only_Scratch": "#8c564b",         # Nâu (Scratch baseline)
        "Source_Only_ZeroShot": "#7f7f7f",      # Xám baseline
        "FewShot_PEFT": "#1f77b4",              # Xanh dương
        "Domain_Transfer_MMD": "#2ca02c",       # Xanh lá
        "Physics_TTA": "#d62728",               # Đỏ nổi bật
        "Physics_Informed_MMD": "#ff7f0e",      # Cam nổi bật (Proposed Method)
    }

    # =========================================================================
    # 1. FIGURE 1: OVERALL ACCURACY & REGRESSION MAE (W, L, D)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Overall Classification Accuracy
    acc_data = []
    for m in available_methods:
        sub = df_preds[df_preds["method"] == m]
        acc = 100.0 * sub["shape_correct"].mean()
        acc_data.append({"Method": m.replace("_", "\n"), "Accuracy": acc, "raw_method": m})
    df_acc = pd.DataFrame(acc_data)

    bars = axes[0].bar(
        df_acc["Method"],
        df_acc["Accuracy"],
        color=[palette[m] for m in df_acc["raw_method"]],
        edgecolor="#222222",
        linewidth=1.2,
        width=0.55,
    )
    axes[0].set_title("(a) Defect Classification Accuracy", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_ylim(0, max(60, df_acc["Accuracy"].max() + 15))
    for bar in bars:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # (b) Regression MAE for W, L, D
    reg_rows = []
    for m in available_methods:
        sub = df_preds[df_preds["method"] == m]
        reg_rows.append({"Method": m, "Dimension": "Width (W)", "MAE (mm)": sub["err_w"].mean()})
        reg_rows.append({"Method": m, "Dimension": "Length (L)", "MAE (mm)": sub["err_l"].mean()})
        reg_rows.append({"Method": m, "Dimension": "Depth (D)", "MAE (mm)": sub["err_d"].mean()})
    df_reg = pd.DataFrame(reg_rows)

    sns.barplot(
        data=df_reg,
        x="Dimension",
        y="MAE (mm)",
        hue="Method",
        palette=palette,
        edgecolor="#222222",
        linewidth=1.0,
        ax=axes[1],
    )
    axes[1].set_title("(b) Regression MAE for W, L, D Dimensions", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Mean Absolute Error (mm)", fontsize=12)
    axes[1].set_xlabel("Geometric Parameter", fontsize=12)
    axes[1].legend(title="Method", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)

    # Thêm giá trị số lên từng thanh
    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.suptitle(f"10-Fold LODO Pooled OOF Evaluation Benchmark\nModel: {model_name[:65]}...", fontsize=13, y=1.03)
    plt.tight_layout()
    fig1_path = os.path.join(plots_dir, "fig1_overall_acc_and_wld_mae.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 1: {fig1_path}")

    # =========================================================================
    # 2. FIGURE 2: PER-CLASS REGRESSION MAE (W, L, D TỪNG CLASS)
    # =========================================================================
    crack_classes = ["Rectangular", "Ellipse", "Triangular", "Step_R", "Step_T"]
    classes_present = [c for c in crack_classes if c in df_preds["true_shape"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    dims = [("err_w", "Width (W)", axes[0]), ("err_l", "Length (L)", axes[1]), ("err_d", "Depth (D)", axes[2])]

    # Chuẩn bị bảng phân tích per-class
    class_metric_list = []
    for cls in classes_present:
        for m in available_methods:
            sub = df_preds[(df_preds["true_shape"] == cls) & (df_preds["method"] == m)]
            if len(sub) > 0:
                class_metric_list.append({
                    "Class": cls,
                    "Method": m,
                    "MAE_W": sub["err_w"].mean(),
                    "MAE_L": sub["err_l"].mean(),
                    "MAE_D": sub["err_d"].mean(),
                    "Accuracy": 100.0 * sub["shape_correct"].mean(),
                    "Count": len(sub),
                })
    df_class = pd.DataFrame(class_metric_list)

    for err_col, title_dim, ax in dims:
        sns.barplot(
            data=df_class,
            x="Class",
            y=err_col.replace("err", "MAE").upper(),
            hue="Method",
            palette=palette,
            edgecolor="#222222",
            linewidth=1.0,
            ax=ax,
        )
        ax.set_title(f"MAE for {title_dim} by Crack Shape", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(f"{title_dim} MAE (mm)", fontsize=11)
        ax.set_xlabel("Crack Shape", fontsize=11)
        ax.tick_params(axis="x", rotation=15)
        ax.get_legend().remove()

    # Tạo legend chung
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(available_methods),
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=11,
    )

    plt.suptitle("Per-Class Regression Error Breakdown (Width, Length, Depth)", fontsize=14, fontweight="bold", y=1.12)
    plt.tight_layout()
    fig2_path = os.path.join(plots_dir, "fig2_per_class_wld_mae.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 2: {fig2_path}")

    # =========================================================================
    # 3. FIGURE 3: PER-CLASS CLASSIFICATION ACCURACY
    # =========================================================================
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_class,
        x="Class",
        y="Accuracy",
        hue="Method",
        palette=palette,
        edgecolor="#222222",
        linewidth=1.0,
    )
    plt.title("Per-Class Defect Shape Classification Accuracy (%)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("True Defect Shape Class", fontsize=12)
    plt.ylim(0, 110)
    plt.legend(title="Method", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.0f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    fig3_path = os.path.join(plots_dir, "fig3_per_class_accuracy.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 3: {fig3_path}")

    # =========================================================================
    # 4. FIGURE 4: PARITY SCATTER PLOTS (PREDICTED VS TRUE FOR W, L, D)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    dim_configs = [
        ("true_w", "pred_w", "Width W (mm)", axes[0]),
        ("true_l", "pred_l", "Length L (mm)", axes[1]),
        ("true_d", "pred_d", "Depth D (mm)", axes[2]),
    ]

    for true_col, pred_col, label_name, ax in dim_configs:
        # Đường chuẩn lý tưởng y = x
        all_vals = np.concatenate([df_preds[true_col].values, df_preds[pred_col].values])
        min_v, max_v = np.min(all_vals) * 0.9, np.max(all_vals) * 1.1
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.6, label="Ideal (y=x)")

        for m in available_methods:
            sub = df_preds[df_preds["method"] == m]
            ax.scatter(
                sub[true_col],
                sub[pred_col],
                label=m,
                color=palette[m],
                alpha=0.75,
                edgecolors="none",
                s=40,
            )

        ax.set_title(f"Parity Plot: {label_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"True {label_name}", fontsize=11)
        ax.set_ylabel(f"Predicted {label_name}", fontsize=11)
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)

    fig4_path = os.path.join(plots_dir, "fig4_parity_scatter_wld.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 4: {fig4_path}")

    # =========================================================================
    # 5. FIGURE 0: CONSOLIDATED MASTER DASHBOARD (ALL-IN-ONE OVERVIEW)
    # =========================================================================
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.20)

    # (1) Top-Left: Overall Accuracy
    ax_acc = fig.add_subplot(gs[0, 0])
    bars = ax_acc.bar(
        df_acc["Method"],
        df_acc["Accuracy"],
        color=[palette[m] for m in df_acc["raw_method"]],
        edgecolor="#222222",
        linewidth=1.2,
        width=0.55,
    )
    ax_acc.set_title("(A) Overall Classification Accuracy (%)", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_ylim(0, max(65, df_acc["Accuracy"].max() + 18))
    for bar in bars:
        h = bar.get_height()
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # (2) Top-Right: Overall Regression MAE (W, L, D)
    ax_reg = fig.add_subplot(gs[0, 1])
    sns.barplot(
        data=df_reg,
        x="Dimension",
        y="MAE (mm)",
        hue="Method",
        palette=palette,
        edgecolor="#222222",
        linewidth=1.0,
        ax=ax_reg,
    )
    ax_reg.set_title("(B) Overall Regression MAE for W, L, D (mm)", fontsize=13, fontweight="bold", pad=10)
    ax_reg.set_ylabel("MAE (mm)", fontsize=11)
    ax_reg.set_xlabel("Geometric Parameter", fontsize=11)
    ax_reg.legend(title="Method", frameon=True, fontsize=10)
    for p in ax_reg.patches:
        h = p.get_height()
        if h > 0:
            ax_reg.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # (3) Bottom-Left: Per-Class Accuracy
    ax_cls_acc = fig.add_subplot(gs[1, 0])
    sns.barplot(
        data=df_class,
        x="Class",
        y="Accuracy",
        hue="Method",
        palette=palette,
        edgecolor="#222222",
        linewidth=1.0,
        ax=ax_cls_acc,
    )
    ax_cls_acc.set_title("(C) Per-Class Defect Classification Accuracy (%)", fontsize=13, fontweight="bold", pad=10)
    ax_cls_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_cls_acc.set_xlabel("Crack Shape", fontsize=11)
    ax_cls_acc.set_ylim(0, 115)
    ax_cls_acc.legend(title="Method", frameon=True, fontsize=9)
    for p in ax_cls_acc.patches:
        h = p.get_height()
        if h > 0:
            ax_cls_acc.annotate(
                f"{h:.0f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # (4) Bottom-Right: Per-Class Width & Length MAE Comparison
    ax_cls_w = fig.add_subplot(gs[1, 1])
    # Melts df_class to show Width and Length errors across classes
    df_cls_w_melt = pd.melt(
        df_class,
        id_vars=["Class", "Method"],
        value_vars=["MAE_W", "MAE_D"],
        var_name="Dimension",
        value_name="MAE_mm"
    )
    df_cls_w_melt["Dimension"] = df_cls_w_melt["Dimension"].replace({"MAE_W": "Width (W)", "MAE_D": "Depth (D)"})
    sns.barplot(
        data=df_cls_w_melt,
        x="Class",
        y="MAE_mm",
        hue="Method",
        palette=palette,
        edgecolor="#222222",
        linewidth=1.0,
        ax=ax_cls_w,
    )
    ax_cls_w.set_title("(D) Per-Class MAE for Width (W) and Depth (D) (mm)", fontsize=13, fontweight="bold", pad=10)
    ax_cls_w.set_ylabel("MAE (mm)", fontsize=11)
    ax_cls_w.set_xlabel("Crack Shape", fontsize=11)
    ax_cls_w.legend(title="Method", frameon=True, fontsize=9)

    plt.suptitle(f"Consolidated Domain Adaptation Evaluation Dashboard (10-Fold LODO OOF)\nModel: {model_name}", fontsize=14, fontweight="bold", y=0.99)
    fig0_path = os.path.join(plots_dir, "fig0_master_overview.png")
    plt.savefig(fig0_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 0: {fig0_path}")

    # =========================================================================
    # 5. FIGURE 5: CONFUSION MATRIX HEATMAPS FOR ALL METHODS
    # =========================================================================
    n_m = len(available_methods)
    fig, axes = plt.subplots(1, n_m, figsize=(4.8 * n_m, 4.4), squeeze=False)
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
            xticklabels=[s[:4] for s in UNIQUE_SHAPES],
            yticklabels=[s[:4] for s in UNIQUE_SHAPES] if idx_m == 0 else False,
            ax=ax,
            linewidths=0.5,
            linecolor="#EEEEEE"
        )
        acc_val = 100.0 * sub["shape_correct"].mean()
        ax.set_title(f"{m}\n(Acc: {acc_val:.1f}%)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Shape", fontsize=10)
        if idx_m == 0:
            ax.set_ylabel("True Shape", fontsize=10)

    plt.suptitle("Pooled OOF Confusion Matrices Across Methods (Labels: Ellipse, Rectangular, Step_R, Step_T, Triangular)", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig5_path = os.path.join(plots_dir, "fig5_confusion_matrices.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 5: {fig5_path}")

    # =========================================================================
    # 6. FIGURE 6: NORMALIZED MAE (NMAE %) COMPARISON ACROSS W, L, D
    # =========================================================================
    w_range, d_range, l_nominal = 0.4, 2.0, 10.0
    nmae_rows = []
    for m in available_methods:
        sub = df_preds[df_preds["method"] == m]
        nmae_rows.append({"Method": m, "Dimension": "Width (W)", "NMAE (%)": (sub["err_w"].mean() / w_range) * 100.0})
        nmae_rows.append({"Method": m, "Dimension": "Length (L)", "NMAE (%)": (sub["err_l"].mean() / l_nominal) * 100.0})
        nmae_rows.append({"Method": m, "Dimension": "Depth (D)", "NMAE (%)": (sub["err_d"].mean() / d_range) * 100.0})
    df_nmae = pd.DataFrame(nmae_rows)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_nmae,
        x="Dimension",
        y="NMAE (%)",
        hue="Method",
        palette=palette,
        edgecolor="#222222",
        linewidth=1.0,
    )
    plt.title("Normalized MAE (NMAE %) Comparison (Dimensionless & Scale-Invariant)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("NMAE (% of dynamic range)", fontsize=12)
    plt.xlabel("Geometric Dimension", fontsize=12)
    plt.legend(title="Method", frameon=True, fontsize=10)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")
    plt.tight_layout()
    fig6_path = os.path.join(plots_dir, "fig6_nmae_comparison.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Fig 6: {fig6_path}")

    # =========================================================================
    # 5. PRINT PER-CLASS BREAKDOWN TABLE (W, L, D) TO CONSOLE
    # =========================================================================
    print("\n" + "=" * 90)
    print("PER-CLASS SUMMARY: ACCURACY & REGRESSION MAE (W, L, D)")
    print("=" * 90)
    df_display = df_class.copy()
    df_display["Accuracy (%)"] = df_display["Accuracy"].map("{:.1f}%".format)
    df_display["MAE_W (mm)"] = df_display["MAE_W"].map("{:.4f}".format)
    df_display["MAE_L (mm)"] = df_display["MAE_L"].map("{:.4f}".format)
    df_display["MAE_D (mm)"] = df_display["MAE_D"].map("{:.4f}".format)
    print(df_display[["Class", "Method", "Count", "Accuracy (%)", "MAE_W (mm)", "MAE_L (mm)", "MAE_D (mm)"]].to_string(index=False))
    print("=" * 90)

    # Save per-class metrics csv
    class_csv = os.path.join(plots_dir, "per_class_wld_metrics.csv")
    df_class.to_csv(class_csv, index=False)
    print(f"[SAVED] Per-class metrics table: {class_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ phân tích ACC & W, L, D cho Domain Adaptation")
    parser.add_argument("--target-dir", type=str, default=None, help="Thư mục chứa kết quả của model cần vẽ")
    parser.add_argument("--keyword", type=str, default=None, help="Từ khóa tìm kiếm model (vd: 05pct, 10pct)")
    args = parser.parse_args()

    plot_domain_adaptation_results(target_dir=args.target_dir, keyword=args.keyword)
