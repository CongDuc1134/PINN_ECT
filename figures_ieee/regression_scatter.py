# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: 3D Dimension Regression Scatter Visualizations.

Key Rules:
- No internal figure/subplot titles.
- Metrics (R2, MAE, RMSE) displayed in clean bounding boxes or outside legend.
- Diagonal ideal line y=x with 1:1 square parity box.
- Dual export: Vector PDF & 300 DPI PNG.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .style import setup_ieee_style, save_ieee_figure, COLOR_W, COLOR_L, COLOR_D


def plot_fig3_regression_scatter(y_true_wld, y_pred_wld, output_dir, filename_prefix="fig3_regression_scatter"):
    """
    Plot IEEE standard Fig 3: 3-Panel Regression Scatter Plots (Width, Length, Depth).
    Features square 1:1 parity scaling (y=x 45-degree diagonal) and crisp IEEE typography.
    """
    from matplotlib.lines import Line2D

    setup_ieee_style()
    targets = [
        ("Width, $W$ (mm)", COLOR_W),
        ("Length, $L$ (mm)", COLOR_L),
        ("Depth, $D$ (mm)", COLOR_D),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))

    for i, (dim_label, color) in enumerate(targets):
        ax = axes[i]
        t = y_true_wld[:, i]
        p = y_pred_wld[:, i]

        r2 = r2_score(t, p)
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))

        # Determine unified limits for true 1:1 parity box
        min_v = min(np.min(t), np.min(p))
        max_v = max(np.max(t), np.max(p))
        span = max_v - min_v if max_v > min_v else 1.0
        pad = 0.05 * span
        lims = [min_v - pad, max_v + pad]

        # Scatter points (clean size and opacity to avoid heavy overplotting)
        ax.scatter(t, p, alpha=0.40, color=color, edgecolors='none', s=12)

        # Ideal y=x diagonal line (true 45-degree parity)
        ax.plot(lims, lims, 'k--', linewidth=1.2, label='Ideal ($y=x$)')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_box_aspect(1)

        ax.set_xlabel(f"True {dim_label}")
        ax.set_ylabel(f"Pred. {dim_label}")

        # Metrics box in upper-left
        metrics_text = f"$R^2 = {r2:.3f}$\nMAE = {mae:.3f} mm\nRMSE = {rmse:.3f} mm"
        ax.text(0.06, 0.94, metrics_text, transform=ax.transAxes,
                fontsize=7.5, verticalalignment='top',
                bbox=dict(boxstyle='square,pad=0.35', facecolor='white', edgecolor='#cccccc', alpha=0.92))

    # Shared legend placed cleanly outside above the subplots
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='k', linewidth=1.2, label=r'Ideal ($y=x$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=6, alpha=0.7, label='Predictions'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)


def save_regression_plots(y_true_wld, y_pred_wld, output_dir, metrics_names=None, prefix="test_", **kwargs):
    """
    Convenience wrapper for backward compatibility across existing evaluation scripts.
    """
    filename = f"{prefix}regression_scatter_ieee"
    plot_fig3_regression_scatter(y_true_wld, y_pred_wld, output_dir, filename_prefix=filename)
