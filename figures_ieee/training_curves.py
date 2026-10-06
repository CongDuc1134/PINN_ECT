# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: Training & Loss Curves Visualization.

Key Rules:
- No internal figure/subplot titles (all captions belong in LaTeX \caption).
- Legends positioned cleanly outside or along borders without obscuring curves.
- Axis labels explicitly include metric names, symbols, and units.
- Saves both vector PDF and 300 DPI PNG.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .style import setup_ieee_style, save_ieee_figure, COLOR_TRAIN, COLOR_VAL, COLOR_PINN


def plot_fig1_training_curves(history, output_dir, pinn_activated_epoch=None, filename_prefix="fig1_training_curves"):
    """
    Plot IEEE standard Fig 1: Dual-panel Training & Validation Trajectories.
    Panel (a): Total Loss (Train vs Val)
    Panel (b): Classification Accuracy (Train vs Val)
    """
    setup_ieee_style()
    epochs = history.get('epoch', [])
    if not epochs and history.get('train_loss'):
        epochs = list(range(1, len(history['train_loss']) + 1))
    if not epochs:
        return

    n_epochs = len(epochs)
    markevery = max(1, n_epochs // 15)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # --- Panel (a): Total Loss ---
    ax = axes[0]
    ax.plot(epochs, history.get('train_loss', []),
            label='Train', color=COLOR_TRAIN, linewidth=1.5,
            marker='o', markersize=3.5, markevery=markevery)
    ax.plot(epochs, history.get('val_loss', []),
            label='Validation', color=COLOR_VAL, linewidth=1.5, linestyle='--',
            marker='s', markersize=3.5, markevery=markevery)

    if pinn_activated_epoch is not None and pinn_activated_epoch > 1:
        ax.axvline(x=pinn_activated_epoch, color='#666666', linestyle=':', linewidth=1.2,
                   label=f'PINN (Ep. {pinn_activated_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss (a.u.)')
    ax.set_yscale('log')
    # Legend placed outside on top of subplot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True)

    # --- Panel (b): Classification Accuracy or Physics Loss (if regression-only) ---
    ax = axes[1]
    train_acc = [acc * 100.0 if acc <= 1.0 else acc for acc in history.get('train_clf_acc', [])]
    val_acc = [acc * 100.0 if acc <= 1.0 else acc for acc in history.get('val_clf_acc', [])]

    if train_acc or val_acc:
        if train_acc:
            ax.plot(epochs[:len(train_acc)], train_acc,
                    label='Train Acc.', color=COLOR_TRAIN, linewidth=1.5,
                    marker='^', markersize=3.5, markevery=markevery)
        if val_acc:
            ax.plot(epochs[:len(val_acc)], val_acc,
                    label='Val. Acc.', color=COLOR_VAL, linewidth=1.5, linestyle='--',
                    marker='v', markersize=3.5, markevery=markevery)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0, 105])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)
    elif history.get('physics_loss') and any(p > 0 for p in history.get('physics_loss', [])):
        phys = history.get('physics_loss', [])
        ax.plot(epochs[:len(phys)], phys,
                label='PINN Loss', color=COLOR_PINN, linewidth=1.5,
                marker='D', markersize=3.5, markevery=markevery)
        ax.set_ylabel('Physics Loss')
        if any(p > 0 for p in phys):
            ax.set_yscale('log')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=True)
    else:
        ax.set_ylabel('Accuracy / Physics')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, color='#888888')

    if pinn_activated_epoch is not None and pinn_activated_epoch > 1:
        ax.axvline(x=pinn_activated_epoch, color='#666666', linestyle=':', linewidth=1.2)

    ax.set_xlabel('Epoch')

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)


def plot_loss_components_breakdown(history, output_dir, filename_prefix="loss_components_breakdown"):
    """
    Plot detailed training & validation loss components without internal titles.
    Subplot 1: Training components (Clf, W, L, D, Physics)
    Subplot 2: Validation components
    """
    setup_ieee_style()
    train_w = history.get('train_w_loss', [])
    if not train_w:
        return

    epochs = list(range(1, len(train_w) + 1))
    markevery = max(1, len(epochs) // 15)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # Panel 1: Training components
    ax1 = axes[0]
    ax1.plot(epochs, history.get('train_clf_loss', []), label=r'$\mathcal{L}_{\mathrm{clf}}$',
             color='#1f77b4', linewidth=1.2, marker='o', markersize=3, markevery=markevery)
    ax1.plot(epochs, history.get('train_w_loss', []), label=r'$\mathcal{L}_{W}$',
             color='#2ca02c', linewidth=1.2, marker='s', markersize=3, markevery=markevery)
    ax1.plot(epochs, history.get('train_l_loss', []), label=r'$\mathcal{L}_{L}$',
             color='#ff7f0e', linewidth=1.2, marker='^', markersize=3, markevery=markevery)
    ax1.plot(epochs, history.get('train_d_loss', []), label=r'$\mathcal{L}_{D}$',
             color='#9467bd', linewidth=1.2, marker='d', markersize=3, markevery=markevery)

    phys_losses = np.array(history.get('physics_loss', []))
    if len(phys_losses) > 0 and np.nanmax(phys_losses) > 0:
        act_idx = np.where(phys_losses > 0)[0]
        if len(act_idx) > 0:
            ax1.plot(act_idx + 1, phys_losses[act_idx], label=r'$\mathcal{L}_{\mathrm{PINN}}$',
                     color='#d62728', linewidth=1.4, linestyle=':', marker='x', markersize=3.5, markevery=max(1, len(act_idx)//15))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss Components')
    ax1.set_yscale('log')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=8)

    # Panel 2: Validation components
    ax2 = axes[1]
    val_w = history.get('val_w_loss', [])
    if val_w:
        ax2.plot(epochs[:len(val_w)], history.get('val_clf_loss', [])[:len(val_w)], label=r'$\mathcal{L}_{\mathrm{clf}}$ (Val)',
                 color='#1f77b4', linewidth=1.2, linestyle='--', marker='o', markersize=3, markevery=markevery)
        ax2.plot(epochs[:len(val_w)], val_w, label=r'$\mathcal{L}_{W}$ (Val)',
                 color='#2ca02c', linewidth=1.2, linestyle='--', marker='s', markersize=3, markevery=markevery)
        ax2.plot(epochs[:len(val_w)], history.get('val_l_loss', [])[:len(val_w)], label=r'$\mathcal{L}_{L}$ (Val)',
                 color='#ff7f0e', linewidth=1.2, linestyle='--', marker='^', markersize=3, markevery=markevery)
        ax2.plot(epochs[:len(val_w)], history.get('val_d_loss', [])[:len(val_w)], label=r'$\mathcal{L}_{D}$ (Val)',
                 color='#9467bd', linewidth=1.2, linestyle='--', marker='d', markersize=3, markevery=markevery)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss Components')
    ax2.set_yscale('log')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)
