# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/plot_new_methods.py
Generates IEEE Transactions Publication-Standard Figures (300 DPI, Serif fonts,
dual inward ticks, professional palette) for the new Domain Adaptation methods
(PI-LoRA, Signal Calibration, DANN, and Ablation Study).
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.utils import UNIQUE_SHAPES

# IEEE Transactions Standard Plotting Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def format_ieee_axis(ax):
    """Applies strict IEEE Transactions axis styling: ticks on both sides, inward."""
    ax.tick_params(direction='in', which='both', top=True, right=True, left=True, bottom=True)
    ax.grid(True, linestyle='--', alpha=0.4, color='#b0b0b0', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#222222')
        spine.set_linewidth(0.8)


def generate_all_ieee_figures():
    output_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_new_methods")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[START] Generating IEEE Figures into: {output_dir}")

    # =========================================================================
    # FIGURE 1: Head-to-Head Comparison: PINN vs NoPINN (Old PEFT vs PI-LoRA)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2), dpi=300)

    categories = ['Old Standard PEFT', 'PI-LoRA Calibrated\n(Proposed)']
    pinn_acc = [30.0, 60.0]
    nopinn_acc = [50.0, 55.0]

    pinn_mae = [0.770, 0.581]
    nopinn_mae = [0.613, 0.647]

    x = np.arange(len(categories))
    width = 0.32

    # Accuracy Plot
    rects1 = ax1.bar(x - width/2, nopinn_acc, width, label='Baseline (NoPINN)', color='#b0bec5', edgecolor='#37474f', hatch='//', zorder=3)
    rects2 = ax1.bar(x + width/2, pinn_acc, width, label='PINN-ECT', color='#1565c0', edgecolor='#0d47a1', zorder=3)
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('(a) Shape Classification Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 80)
    format_ieee_axis(ax1)
    ax1.legend(loc='upper left', framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0d47a1')

    # MAE Plot
    rects3 = ax2.bar(x - width/2, nopinn_mae, width, label='Baseline (NoPINN)', color='#b0bec5', edgecolor='#37474f', hatch='//', zorder=3)
    rects4 = ax2.bar(x + width/2, pinn_mae, width, label='PINN-ECT', color='#2e7d32', edgecolor='#1b5e20', zorder=3)
    ax2.set_ylabel('Overall MAE (mm)')
    ax2.set_title('(b) Crack Dimension Error (MAE)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1.0)
    format_ieee_axis(ax2)
    ax2.legend(loc='upper right', framealpha=0.9)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1b5e20')

    fig.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_pinn_vs_nopinn_headtohead.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"[OK] Saved: {fig1_path}")

    # =========================================================================
    # FIGURE 2: Dimensional Breakdown (MAE W, L, D) for PINN vs NoPINN
    # =========================================================================
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    dims = ['Width ($W$)', 'Length ($L$)', 'Depth ($D$)']
    pinn_dims = [0.118, 0.524, 0.965]
    nopinn_dims = [0.107, 0.771, 1.062]

    xd = np.arange(len(dims))
    w_bar = 0.35

    ax.bar(xd - w_bar/2, nopinn_dims, w_bar, label='Baseline NoPINN', color='#b0bec5', edgecolor='#37474f', hatch='//', zorder=3)
    ax.bar(xd + w_bar/2, pinn_dims, w_bar, label='PI-LoRA PINN', color='#1565c0', edgecolor='#0d47a1', zorder=3)
    ax.set_ylabel('Mean Absolute Error (mm)')
    ax.set_title('Dimension-Specific Error Breakdown')
    ax.set_xticks(xd)
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 1.3)
    format_ieee_axis(ax)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_dimension_breakdown_mae.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"[OK] Saved: {fig2_path}")

    # =========================================================================
    # FIGURE 3: Method Progression / Ablation Study
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.16, 3.2), dpi=300)
    steps = [
        'Source-Only\n(Zero-Shot)',
        'Calibrated\nPEFT (Freeze)',
        'Standard\nPEFT (Old)',
        'Standard\nMMD (Old)',
        'PI-MMD\n(Direction 5)',
        'PI-LoRA\n(Rank 4)',
        'PI-LoRA\nCalibrated'
    ]
    acc_vals = [22.7, 25.0, 30.0, 40.0, 40.0, 45.0, 60.0]
    mae_vals = [1.430, 0.739, 0.770, 0.727, 0.469, 0.537, 0.530]

    xs = np.arange(len(steps))
    color_acc = '#1565c0'
    color_mae = '#d32f2f'

    ax.plot(xs, acc_vals, marker='o', linewidth=2.0, color=color_acc, label='Classification Accuracy (%)', zorder=4)
    for i, txt in enumerate(acc_vals):
        ax.annotate(f'{txt:.1f}%', (xs[i], acc_vals[i]), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color=color_acc, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', color=color_acc)
    ax.tick_params(axis='y', labelcolor=color_acc)
    ax.set_ylim(10, 75)
    ax.set_xticks(xs)
    ax.set_xticklabels(steps)
    format_ieee_axis(ax)

    # Twin axis for MAE
    ax2 = ax.twinx()
    ax2.plot(xs, mae_vals, marker='s', linewidth=2.0, linestyle='--', color=color_mae, label='Overall MAE (mm)', zorder=4)
    for i, txt in enumerate(mae_vals):
        ax2.annotate(f'{txt:.3f}', (xs[i], mae_vals[i]), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=8, color=color_mae, fontweight='bold')

    ax2.set_ylabel('Overall MAE (mm)', color=color_mae)
    ax2.tick_params(axis='y', labelcolor=color_mae)
    ax2.set_ylim(0.3, 1.6)
    ax2.tick_params(direction='in', which='both')

    ax.set_title('Progression of Domain Adaptation Strategies for PINN-ECT (Sim-to-Real 5kHz)')
    fig.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_ablation_progression.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"[OK] Saved: {fig3_path}")

    # =========================================================================
    # FIGURE 4: Confusion Matrix of Best PI-LoRA Calibrated Model
    # =========================================================================
    best_pred_csv = os.path.join(
        PROJECT_ROOT,
        "domain_adaptation",
        "results",
        "exp_hybrid_pilor",
        "cnn_train_05pct_PINN_base_a1_W100_E300_seed_42_run_20260819_103124",
        "pi_lora_calibrated_predictions.csv"
    )
    if os.path.exists(best_pred_csv):
        df_best = pd.read_csv(best_pred_csv)
        cm = confusion_matrix(df_best['true_shape'], df_best['pred_shape'], labels=UNIQUE_SHAPES)

        fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(direction='in', labelsize=8)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=UNIQUE_SHAPES,
            yticklabels=UNIQUE_SHAPES,
            ylabel='True Defect Class',
            xlabel='Predicted Defect Class',
            title='Confusion Matrix (PI-LoRA Calibrated)'
        )
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9, fontweight='bold' if cm[i, j] > 0 else 'normal')

        format_ieee_axis(ax)
        fig.tight_layout()
        fig4_path = os.path.join(output_dir, "fig4_confusion_matrix_best_lora.png")
        fig.savefig(fig4_path)
        plt.close(fig)
        print(f"[OK] Saved: {fig4_path}")

    print("\n[SUCCESS] All IEEE Figures successfully generated!")


if __name__ == "__main__":
    generate_all_ieee_figures()
