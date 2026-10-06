# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: Per-Shape Performance Comparisons.

Key Rules:
- No internal figure/subplot titles.
- 4-panel multi-metric analysis (Accuracy, MAE, RMSE, NRMSE).
- Direct value callouts above bars for clean presentation.
- Dual export: Vector PDF & 300 DPI PNG.
"""

import os
import matplotlib.pyplot as plt
from .style import setup_ieee_style, save_ieee_figure, SHAPE_COLORS


def plot_per_shape_comparison(per_shape_results, output_dir, filename_prefix="test_metrics_per_shape_comparison"):
    """
    Plot 4-panel per-shape performance analysis without internal titles.
    Panel (a): Classification Accuracy (%)
    Panel (b): Average MAE (mm)
    Panel (c): Average RMSE (mm)
    Panel (d): Average NRMSE (%)
    """
    setup_ieee_style()
    if not per_shape_results:
        return

    shapes = [r['shape'] for r in per_shape_results]
    n_shapes = len(shapes)
    colors = [SHAPE_COLORS[i % len(SHAPE_COLORS)] for i in range(n_shapes)]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))

    # --- Panel 1: Accuracy (%) ---
    ax = axes[0, 0]
    accs = [r['clf_acc'] * 100.0 for r in per_shape_results]
    bars = ax.bar(shapes, accs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([0, 110])
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.5, f"{val:.1f}%",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    # --- Panel 2: MAE Avg (mm) ---
    ax = axes[0, 1]
    maes = [(r['mae_w'] + r['mae_l'] + r['mae_d']) / 3.0 for r in per_shape_results]
    bars = ax.bar(shapes, maes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('MAE (mm)')
    max_mae = max(maes) if maes else 1.0
    ax.set_ylim([0, max_mae * 1.25])
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + (max_mae * 0.03), f"{val:.3f}",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    # --- Panel 3: RMSE Avg (mm) ---
    ax = axes[1, 0]
    rmses = [(r['rmse_w'] + r['rmse_l'] + r['rmse_d']) / 3.0 for r in per_shape_results]
    bars = ax.bar(shapes, rmses, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('RMSE (mm)')
    max_rmse = max(rmses) if rmses else 1.0
    ax.set_ylim([0, max_rmse * 1.25])
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + (max_rmse * 0.03), f"{val:.3f}",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    # --- Panel 4: NRMSE Avg (%) ---
    ax = axes[1, 1]
    nrmses = [r['nrmse_avg'] for r in per_shape_results]
    bars = ax.bar(shapes, nrmses, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    ax.set_ylabel('NRMSE (%)')
    max_nrmse = max(nrmses) if nrmses else 10.0
    ax.set_ylim([0, max_nrmse * 1.25])
    for bar, val in zip(bars, nrmses):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + (max_nrmse * 0.03), f"{val:.2f}%",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)
