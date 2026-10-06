# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: 3D Dimension Regression Scatter Visualizations.

Key Rules:
- No internal figure/subplot titles.
- Metrics (R2, MAE, RMSE) displayed in clean bounding boxes or outside legend.
- Diagonal ideal line y=x and +/-10% error envelope.
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
    """
    setup_ieee_style()
    targets = [
        ("Width, $W$ (mm)", COLOR_W),
        ("Length, $L$ (mm)", COLOR_L),
        ("Depth, $D$ (mm)", COLOR_D),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    for i, (dim_label, color) in enumerate(targets):
        ax = axes[i]
        t = y_true_wld[:, i]
        p = y_pred_wld[:, i]

        r2 = r2_score(t, p)
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))

        # Scatter points
        ax.scatter(t, p, alpha=0.45, color=color, edgecolors='none', s=16, label='Predicted')

        # Ideal y=x line
        min_v = min(np.min(t), np.min(p))
        max_v = max(np.max(t), np.max(p))
        pad = 0.05 * (max_v - min_v) if max_v > min_v else 0.1
        line_range = np.linspace(min_v - pad, max_v + pad, 100)

        ax.plot(line_range, line_range, 'k--', linewidth=1.0, label='Ideal ($y=x$)')
        # +/-10% error margin
        ax.fill_between(line_range, line_range * 0.9, line_range * 1.1,
                        color='gray', alpha=0.15, label=r'$\pm 10\%$ Error')

        ax.set_xlim([min_v - pad, max_v + pad])
        ax.set_ylim([min_v - pad, max_v + pad])

        ax.set_xlabel(f"True {dim_label}")
        ax.set_ylabel(f"Pred. {dim_label}")

        # Metrics box in lower-right or upper-left
        metrics_text = f"$R^2 = {r2:.3f}$\nMAE = {mae:.3f} mm\nRMSE = {rmse:.3f} mm"
        ax.text(0.05, 0.92, metrics_text, transform=ax.transAxes,
                fontsize=7.5, verticalalignment='top',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    # Shared legend placed outside above the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)


def save_regression_plots(y_true_wld, y_pred_wld, output_dir, metrics_names=None, prefix="test_", **kwargs):
    """
    Convenience wrapper for backward compatibility across existing evaluation scripts.
    """
    filename = f"{prefix}regression_scatter_ieee"
    plot_fig3_regression_scatter(y_true_wld, y_pred_wld, output_dir, filename_prefix=filename)
