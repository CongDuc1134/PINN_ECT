# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: Comprehensive Metrics & Error Distribution Visualizations.

Key Rules:
- No internal figure/subplot titles.
- Regression & Classification metric panels.
- Violin / Boxplots of parameter reconstruction residuals (W, L, D).
- Dual export: Vector PDF & 300 DPI PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .style import setup_ieee_style, save_ieee_figure, SHAPE_COLORS, COLOR_W, COLOR_L, COLOR_D


def generate_comprehensive_metrics_visualizations(
    test_metrics,
    clf_acc,
    prec,
    rec,
    f1,
    y_test_denorm,
    y_pred_wld_denorm,
    y_shape_test,
    unique_shapes,
    output_dir,
):
    """
    Generate comprehensive IEEE metric bar charts and error distribution violin plots without internal titles.
    """
    setup_ieee_style()
    metrics_names = ['Width, $W$', 'Length, $L$', 'Depth, $D$']
    colors = [COLOR_W, COLOR_L, COLOR_D]

    # --- 1. REGRESSION METRICS GRID (MAE, RMSE, R2, NRMSE) ---
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))

    # Panel (a): MAE
    ax = axes[0, 0]
    mae_vals = test_metrics['mae']
    bars = ax.bar(metrics_names, mae_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('MAE (mm)')
    max_m = max(mae_vals) if mae_vals else 1.0
    ax.set_ylim([0, max_m * 1.25])
    for bar, v in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + (max_m * 0.03), f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Panel (b): RMSE
    ax = axes[0, 1]
    rmse_vals = test_metrics['rmse']
    bars = ax.bar(metrics_names, rmse_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('RMSE (mm)')
    max_r = max(rmse_vals) if rmse_vals else 1.0
    ax.set_ylim([0, max_r * 1.25])
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + (max_r * 0.03), f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Panel (c): R2 Score
    ax = axes[1, 0]
    r2_vals = test_metrics['r2']
    bars = ax.bar(metrics_names, r2_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel(r'Coefficient of Det. ($R^2$)')
    ax.set_ylim([0, 1.15])
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.03, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Panel (d): NRMSE
    ax = axes[1, 1]
    nrmse_vals = test_metrics['nrmse']
    bars = ax.bar(metrics_names, nrmse_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('NRMSE (%)')
    max_nr = max(nrmse_vals) if nrmse_vals else 10.0
    ax.set_ylim([0, max_nr * 1.25])
    for bar, v in zip(bars, nrmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + (max_nr * 0.03), f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    save_ieee_figure(fig, os.path.join(output_dir, 'regression_metrics_comprehensive'))

    # --- 2. CLASSIFICATION METRICS BAR CHART ---
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    clf_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    clf_vals = [clf_acc * 100.0, prec * 100.0, rec * 100.0, f1 * 100.0]
    clf_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

    bars = ax.bar(clf_names, clf_vals, color=clf_colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('Score (%)')
    ax.set_ylim([0, 115])
    for bar, v in zip(bars, clf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 2.0, f'{v:.1f}%', ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    plt.tight_layout()
    save_ieee_figure(fig, os.path.join(output_dir, 'classification_metrics_comprehensive'))

    # --- 3. RESIDUAL ERROR DISTRIBUTIONS (VIOLIN PLOT) ---
    errors_wld = y_pred_wld_denorm - y_test_denorm
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    dim_titles = [r'$\Delta W = W_{\mathrm{pred}} - W_{\mathrm{true}}$ (mm)',
                  r'$\Delta L = L_{\mathrm{pred}} - L_{\mathrm{true}}$ (mm)',
                  r'$\Delta D = D_{\mathrm{pred}} - D_{\mathrm{true}}$ (mm)']

    for i in range(3):
        ax = axes[i]
        err = errors_wld[:, i]
        parts = ax.violinplot(err, positions=[0], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xticks([])
        ax.set_ylabel(dim_titles[i], fontsize=8.0)

    plt.tight_layout()
    save_ieee_figure(fig, os.path.join(output_dir, 'error_distributions_violin'))
