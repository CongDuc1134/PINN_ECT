# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: Confusion Matrix & Per-Class Classification Visualizations.

Key Rules:
- No internal figure/subplot titles.
- Clean colorbar with explicit range and ticks.
- Values formatted with percentage and count.
- Dual export: Vector PDF & 300 DPI PNG.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .style import setup_ieee_style, save_ieee_figure, SHAPE_COLORS


def plot_fig2_confusion_matrix(y_true, y_pred, unique_shapes, output_dir, filename_prefix="fig2_confusion_matrix"):
    """
    Plot IEEE standard Fig 2: Normalized Confusion Matrix with clean count/accuracy annotations.
    Features square aspect ratio, clean muted zero cells, and overall accuracy indication.
    """
    setup_ieee_style()
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    n_classes = len(unique_shapes)
    if np.issubdtype(y_t.dtype, np.integer):
        labels_arr = np.arange(n_classes)
    else:
        labels_arr = list(unique_shapes)
    cm = confusion_matrix(y_t, y_p, labels=labels_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    n_classes = len(unique_shapes)
    display_names = [str(s).replace('_', '-') for s in unique_shapes]
    overall_acc = float(np.mean(y_t == y_p) * 100.0)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    im = ax.imshow(cm_norm * 100.0, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Classification Accuracy (%)', rotation=270, labelpad=14)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(display_names, rotation=35, ha='right')
    ax.set_yticklabels(display_names)
    ax.set_aspect('equal')

    thresh = 50.0
    for i in range(n_classes):
        for j in range(n_classes):
            pct = cm_norm[i, j] * 100.0
            cnt = cm[i, j]
            if cnt == 0:
                ax.text(j, i, "0", ha="center", va="center", color="#888888", fontsize=7.5)
            else:
                text_color = "white" if pct > thresh else "black"
                ax.text(j, i, f"{pct:.1f}%\n({cnt})",
                        ha="center", va="center", color=text_color, fontsize=8.0, fontweight='medium')

    ax.set_ylabel('Ground Truth Class')
    ax.set_xlabel(f'Predicted Class (Overall Acc: {overall_acc:.2f}%)')
    ax.grid(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)


def plot_per_class_accuracy(y_true, y_pred, unique_shapes, output_dir, filename_prefix="per_class_accuracy"):
    """
    Plot per-class recall accuracy bar chart without internal title.
    """
    setup_ieee_style()
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100.0
        per_class_acc = np.nan_to_num(per_class_acc)

    display_names = [str(s).replace('_', '-') for s in unique_shapes]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x_pos = np.arange(len(unique_shapes))
    colors = [SHAPE_COLORS[i % len(SHAPE_COLORS)] for i in range(len(unique_shapes))]

    bars = ax.bar(x_pos, per_class_acc, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7, width=0.55)

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.5, f"{val:.1f}%",
                ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=25, ha='right')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_xlabel('Defect Geometry Class')
    ax.set_ylim([0, 110])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)
