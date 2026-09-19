"""
domain_adaptation/plot_ieee_comprehensive.py
Generates publication-grade IEEE Transactions plots:
1. fig1_master_acc_mae_19models.png: Accuracy & Overall MAE across all 19 models (1% to 10% data)
2. fig2_dimension_wld_mae_19models.png: Dimension breakdown (W, L, D MAE) across all 19 models
3. fig3_per_class_accuracy.png: Per-class classification accuracy (5 defect geometries)
4. fig4_per_class_wld_mae.png: Per-class dimension MAE (W, L, D across 5 defect geometries)
5. fig5_parity_plots_pred_vs_true.png: Parity scatter plots (Pred vs True for W, L, D)

Strictly adheres to IEEE Transactions standards:
- 300 DPI resolution
- Times New Roman serif typography
- Dual inward ticks (major and minor)
- Professional scientific color scheme
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.utils import UNIQUE_SHAPES

# IEEE Styling Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "figure.autolayout": False,
})

PINN_COLOR = "#0f4c81"       # Deep Navy Blue
BASE_COLOR = "#d95f02"       # Muted Vermillion
ACCENT_GREEN = "#1b9e77"     # Forest Green
ACCENT_PURPLE = "#7570b3"    # Slate Purple

OUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_ieee_comprehensive")
os.makedirs(OUT_DIR, exist_ok=True)

def load_master_and_predictions():
    summary_path = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "pi_lgl_all_19_models_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    df_summary = pd.read_csv(summary_path)

    # Load all individual prediction CSVs
    pred_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "pi_lgl_all_19_models")
    pred_files = glob.glob(os.path.join(pred_dir, "**", "pi_lgl_predictions.csv"), recursive=True)

    all_preds = []
    for pf in pred_files:
        df_p = pd.read_csv(pf)
        folder_name = os.path.basename(os.path.dirname(pf))
        mtype = "PINN" if "PINN" in folder_name else "Baseline"
        df_p["Model_Type"] = mtype
        df_p["Model_Tag"] = folder_name
        all_preds.append(df_p)

    df_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return df_summary, df_preds

def apply_ieee_ticks(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# ==============================================================================
# PLOT 1: MASTER ACCURACY & OVERALL MAE ACROSS 19 MODELS
# ==============================================================================
def plot_fig1_master_acc_mae(df_summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    pinn_rows = df_summary[df_summary["Model_Type"] == "PINN"].copy()
    base_rows = df_summary[df_summary["Model_Type"] == "Baseline"].copy()

    # Sort by overall data scale
    scale_order = {
        "1%": 1, "3%": 2, "5% (seed 42)": 3, "5% (seed 123)": 4,
        "5% (seed 456)": 5, "5% (seed 789)": 6, "5% (alpha 10)": 7,
        "5% (alpha 100)": 8, "5% (alpha 1000)": 9, "7%": 10, "10%": 11
    }
    pinn_rows["Order"] = pinn_rows["Data_Scale"].map(scale_order).fillna(99)
    base_rows["Order"] = base_rows["Data_Scale"].map(scale_order).fillna(99)
    pinn_rows = pinn_rows.sort_values("Order")
    base_rows = base_rows.sort_values("Order")

    # Subplot A: Accuracy
    ax1.plot(range(len(base_rows)), base_rows["Clf_Accuracy (%)"], "s--", color=BASE_COLOR, label="NoPINN Baseline (Shortcut)", linewidth=1.5, markersize=6)
    ax1.plot(range(len(pinn_rows)), pinn_rows["Clf_Accuracy (%)"], "o-", color=PINN_COLOR, label="Proposed PI-LGL (PINN)", linewidth=2.0, markersize=6)
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_xlabel("Checkpoint Index across Data Scales (1% to 10%)")
    ax1.set_ylim(40, 105)
    ax1.set_title("(a) Inspection Classification Accuracy", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True, edgecolor="#cbd5e1")
    apply_ieee_ticks(ax1)

    # Subplot B: Overall MAE
    ax2.plot(range(len(base_rows)), base_rows["Overall_MAE (mm)"], "s--", color=BASE_COLOR, label="NoPINN Baseline", linewidth=1.5, markersize=6)
    ax2.plot(range(len(pinn_rows)), pinn_rows["Overall_MAE (mm)"], "o-", color=PINN_COLOR, label="Proposed PI-LGL (PINN)", linewidth=2.0, markersize=6)
    ax2.set_ylabel("Overall Dimension MAE (mm)")
    ax2.set_xlabel("Checkpoint Index across Data Scales (1% to 10%)")
    ax2.set_ylim(0.2, 0.8)
    ax2.set_title("(b) Overall Defect Sizing MAE (W, L, D)", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, edgecolor="#cbd5e1")
    apply_ieee_ticks(ax2)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, "fig1_master_acc_mae_19models.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Fig 1 to: {out_file}")

# ==============================================================================
# PLOT 2: DIMENSION W, L, D MAE BREAKDOWN ACROSS 19 MODELS
# ==============================================================================
def plot_fig2_dimension_breakdown(df_summary):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.8), dpi=300)

    pinn_rows = df_summary[df_summary["Model_Type"] == "PINN"]
    base_rows = df_summary[df_summary["Model_Type"] == "Baseline"]

    # Panel 1: Length L (Highlighting PINN's massive superiority)
    b_l = base_rows["MAE_L (mm)"].values
    p_l = pinn_rows["MAE_L (mm)"].values
    data_l = [p_l, b_l]
    bp1 = ax1.boxplot(data_l, tick_labels=["PI-LGL (PINN)", "NoPINN Base"], patch_artist=True, widths=0.45)
    bp1['boxes'][0].set_facecolor(PINN_COLOR)
    bp1['boxes'][1].set_facecolor(BASE_COLOR)
    for box in bp1['boxes']:
        box.set_alpha(0.7)
    ax1.set_ylabel("Length ($L$) MAE (mm)")
    ax1.set_title("(a) Crack Length Error (MAE $L$)", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax1)

    # Panel 2: Width W
    b_w = base_rows["MAE_W (mm)"].values
    p_w = pinn_rows["MAE_W (mm)"].values
    data_w = [p_w, b_w]
    bp2 = ax2.boxplot(data_w, tick_labels=["PI-LGL (PINN)", "NoPINN Base"], patch_artist=True, widths=0.45)
    bp2['boxes'][0].set_facecolor(PINN_COLOR)
    bp2['boxes'][1].set_facecolor(BASE_COLOR)
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    ax2.set_ylabel("Width ($W$) MAE (mm)")
    ax2.set_title("(b) Crack Width Error (MAE $W$)", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax2)

    # Panel 3: Depth D
    b_d = base_rows["MAE_D (mm)"].values
    p_d = pinn_rows["MAE_D (mm)"].values
    data_d = [p_d, b_d]
    bp3 = ax3.boxplot(data_d, tick_labels=["PI-LGL (PINN)", "NoPINN Base"], patch_artist=True, widths=0.45)
    bp3['boxes'][0].set_facecolor(PINN_COLOR)
    bp3['boxes'][1].set_facecolor(BASE_COLOR)
    for box in bp3['boxes']:
        box.set_alpha(0.7)
    ax3.set_ylabel("Depth ($D$) MAE (mm)")
    ax3.set_title("(c) Crack Depth Error (MAE $D$)", pad=8)
    ax3.grid(True, linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax3)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, "fig2_dimension_wld_mae_19models.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Fig 2 to: {out_file}")

# ==============================================================================
# PLOT 3: PER-CLASS ACCURACY COMPARISON
# ==============================================================================
def plot_fig3_per_class_accuracy(df_preds):
    if df_preds.empty:
        print("[WARN] No predictions to plot per-class accuracy.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)

    classes = UNIQUE_SHAPES
    pinn_accs = []
    base_accs = []

    for c in classes:
        p_sub = df_preds[(df_preds["Model_Type"] == "PINN") & (df_preds["true_shape"] == c)]
        b_sub = df_preds[(df_preds["Model_Type"] == "Baseline") & (df_preds["true_shape"] == c)]
        p_acc = (p_sub["shape_correct"].mean() * 100.0) if len(p_sub) > 0 else 0.0
        b_acc = (b_sub["shape_correct"].mean() * 100.0) if len(b_sub) > 0 else 0.0
        pinn_accs.append(p_acc)
        base_accs.append(b_acc)

    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width/2, pinn_accs, width, label="Proposed PI-LGL (PINN)", color=PINN_COLOR, alpha=0.85, edgecolor="#0a2540")
    rects2 = ax.bar(x + width/2, base_accs, width, label="NoPINN Baseline (Shortcut)", color=BASE_COLOR, alpha=0.85, edgecolor="#8c3300")

    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Per-Class Defect Geometry Classification Accuracy", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, edgecolor="#cbd5e1")
    apply_ieee_ticks(ax)

    # Value labels
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, "fig3_per_class_accuracy.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Fig 3 to: {out_file}")

# ==============================================================================
# PLOT 4: PER-CLASS W, L, D MAE COMPARISON
# ==============================================================================
def plot_fig4_per_class_wld_mae(df_preds):
    if df_preds.empty:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    classes = UNIQUE_SHAPES
    x = np.arange(len(classes))
    width = 0.35

    # 1. Width MAE
    p_err_w = [df_preds[(df_preds["Model_Type"] == "PINN") & (df_preds["true_shape"] == c)]["err_w"].mean() for c in classes]
    b_err_w = [df_preds[(df_preds["Model_Type"] == "Baseline") & (df_preds["true_shape"] == c)]["err_w"].mean() for c in classes]
    ax1.bar(x - width/2, p_err_w, width, label="PI-LGL (PINN)", color=PINN_COLOR, alpha=0.85)
    ax1.bar(x + width/2, b_err_w, width, label="NoPINN Base", color=BASE_COLOR, alpha=0.85)
    ax1.set_ylabel("Width Error MAE $W$ (mm)")
    ax1.set_title("(a) Per-Class Width Error", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=25, ha="right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax1)

    # 2. Length MAE
    p_err_l = [df_preds[(df_preds["Model_Type"] == "PINN") & (df_preds["true_shape"] == c)]["err_l"].mean() for c in classes]
    b_err_l = [df_preds[(df_preds["Model_Type"] == "Baseline") & (df_preds["true_shape"] == c)]["err_l"].mean() for c in classes]
    ax2.bar(x - width/2, p_err_l, width, label="PI-LGL (PINN)", color=PINN_COLOR, alpha=0.85)
    ax2.bar(x + width/2, b_err_l, width, label="NoPINN Base", color=BASE_COLOR, alpha=0.85)
    ax2.set_ylabel("Length Error MAE $L$ (mm)")
    ax2.set_title("(b) Per-Class Length Error", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=25, ha="right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, edgecolor="#cbd5e1")
    apply_ieee_ticks(ax2)

    # 3. Depth MAE
    p_err_d = [df_preds[(df_preds["Model_Type"] == "PINN") & (df_preds["true_shape"] == c)]["err_d"].mean() for c in classes]
    b_err_d = [df_preds[(df_preds["Model_Type"] == "Baseline") & (df_preds["true_shape"] == c)]["err_d"].mean() for c in classes]
    ax3.bar(x - width/2, p_err_d, width, label="PI-LGL (PINN)", color=PINN_COLOR, alpha=0.85)
    ax3.bar(x + width/2, b_err_d, width, label="NoPINN Base", color=BASE_COLOR, alpha=0.85)
    ax3.set_ylabel("Depth Error MAE $D$ (mm)")
    ax3.set_title("(c) Per-Class Depth Error", pad=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=25, ha="right")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax3)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, "fig4_per_class_wld_mae.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Fig 4 to: {out_file}")

# ==============================================================================
# PLOT 5: PARITY SCATTER PLOTS (PRED VS TRUE FOR W, L, D)
# ==============================================================================
def plot_fig5_parity_plots(df_preds):
    if df_preds.empty:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    pinn_sub = df_preds[df_preds["Model_Type"] == "PINN"]
    base_sub = df_preds[df_preds["Model_Type"] == "Baseline"]

    # 1. Width W Parity
    ax1.plot([0.4, 1.0], [0.4, 1.0], "k--", linewidth=1.2, label="Ideal ($y = x$)")
    ax1.scatter(base_sub["true_w"], base_sub["pred_w"], color=BASE_COLOR, alpha=0.4, s=30, label="NoPINN Base")
    ax1.scatter(pinn_sub["true_w"], pinn_sub["pred_w"], color=PINN_COLOR, alpha=0.5, s=35, label="PI-LGL (PINN)")
    ax1.set_xlabel("True Width $W$ (mm)")
    ax1.set_ylabel("Predicted Width $W$ (mm)")
    ax1.set_title("(a) Width $W$ Parity Plot", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax1)

    # 2. Length L Parity
    ax2.plot([6.0, 14.0], [6.0, 14.0], "k--", linewidth=1.2, label="Ideal ($y = x$)")
    ax2.scatter(base_sub["true_l"], base_sub["pred_l"], color=BASE_COLOR, alpha=0.4, s=30, label="NoPINN Base")
    ax2.scatter(pinn_sub["true_l"], pinn_sub["pred_l"], color=PINN_COLOR, alpha=0.5, s=35, label="PI-LGL (PINN)")
    ax2.set_xlabel("True Length $L$ (mm)")
    ax2.set_ylabel("Predicted Length $L$ (mm)")
    ax2.set_title("(b) Length $L$ Parity Plot", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, edgecolor="#cbd5e1")
    apply_ieee_ticks(ax2)

    # 3. Depth D Parity
    ax3.plot([0.5, 3.5], [0.5, 3.5], "k--", linewidth=1.2, label="Ideal ($y = x$)")
    ax3.scatter(base_sub["true_d"], base_sub["pred_d"], color=BASE_COLOR, alpha=0.4, s=30, label="NoPINN Base")
    ax3.scatter(pinn_sub["true_d"], pinn_sub["pred_d"], color=PINN_COLOR, alpha=0.5, s=35, label="PI-LGL (PINN)")
    ax3.set_xlabel("True Depth $D$ (mm)")
    ax3.set_ylabel("Predicted Depth $D$ (mm)")
    ax3.set_title("(c) Depth $D$ Parity Plot", pad=8)
    ax3.grid(True, linestyle=":", alpha=0.6)
    apply_ieee_ticks(ax3)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, "fig5_parity_plots_pred_vs_true.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Fig 5 to: {out_file}")

def main():
    print("=" * 80)
    print("GENERATING COMPREHENSIVE IEEE TRANSACTIONS PLOTS FOR PINN_ECT")
    print("=" * 80)
    df_summary, df_preds = load_master_and_predictions()
    print(f"[OK] Loaded {len(df_summary)} summary models and {len(df_preds)} individual predictions.")

    plot_fig1_master_acc_mae(df_summary)
    plot_fig2_dimension_breakdown(df_summary)
    plot_fig3_per_class_accuracy(df_preds)
    plot_fig4_per_class_wld_mae(df_preds)
    plot_fig5_parity_plots(df_preds)
    print("\n[ALL DONE] All 5 IEEE publication plots successfully generated in:", OUT_DIR)

if __name__ == "__main__":
    main()
