# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: PINN Physics & Field Reconstruction Diagnostics.

Key Rules:
- No internal figure/subplot titles.
- 4-panel field diagnostic:
    (a) Ground-truth MFL field map H_true
    (b) Analytical PINN reconstructed field map H_pred
    (c) Absolute differential residual |H_true - H_pred|
    (d) Centerline 1D cross-section profile
- Colorbars explicitly labeled with units.
- Dual export: Vector PDF & 300 DPI PNG.
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from .style import setup_ieee_style, save_ieee_figure, COLOR_TRAIN, COLOR_VAL


def plot_physics_comparison(
    H_true,
    H_pred,
    output_dir,
    sample_idx=0,
    filename_prefix="physics_field_comparison",
):
    """
    Plot 4-panel physics field diagnostic without internal titles.
    """
    setup_ieee_style()
    # Convert tensors to numpy
    if isinstance(H_true, torch.Tensor):
        H_t = H_true.detach().cpu().numpy()
    else:
        H_t = np.asarray(H_true)

    if isinstance(H_pred, torch.Tensor):
        H_p = H_pred.detach().cpu().numpy()
    else:
        H_p = np.asarray(H_pred)

    if H_t.ndim == 3:
        H_t = H_t[:, :, 0] if H_t.shape[2] <= 3 else H_t[0]
    if H_p.ndim == 3:
        H_p = H_p[:, :, 0] if H_p.shape[2] <= 3 else H_p[0]

    residual = np.abs(H_t - H_p)

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5))

    # Common color scale for True and Pred
    vmin = min(np.min(H_t), np.min(H_p))
    vmax = max(np.max(H_t), np.max(H_p))

    # --- Panel 1: True Field ---
    im1 = axes[0].imshow(H_t, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label(r'$H_{\mathrm{true}}$ (a.u.)', rotation=270, labelpad=10, fontsize=8)
    axes[0].set_xlabel('Scan Position $x$')
    axes[0].set_ylabel('Scan Position $y$')

    # --- Panel 2: Predicted Field ---
    im2 = axes[1].imshow(H_p, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label(r'$H_{\mathrm{pred}}$ (a.u.)', rotation=270, labelpad=10, fontsize=8)
    axes[1].set_xlabel('Scan Position $x$')
    axes[1].set_yticks([])

    # --- Panel 3: Differential Residual ---
    im3 = axes[2].imshow(residual, cmap='hot')
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label(r'$|\Delta H|$ (a.u.)', rotation=270, labelpad=10, fontsize=8)
    axes[2].set_xlabel('Scan Position $x$')
    axes[2].set_yticks([])

    # --- Panel 4: Centerline Section Profile ---
    mid_row = H_t.shape[0] // 2
    profile_true = H_t[mid_row, :]
    profile_pred = H_p[mid_row, :]
    x_coords = np.arange(len(profile_true))

    axes[3].plot(x_coords, profile_true, label='True', color=COLOR_TRAIN, linewidth=1.5)
    axes[3].plot(x_coords, profile_pred, label='Pred.', color=COLOR_VAL, linestyle='--', linewidth=1.5)
    axes[3].set_xlabel('Centerline $x$')
    axes[3].set_ylabel('$H$ (a.u.)')
    axes[3].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=7.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{filename_prefix}_sample{sample_idx:02d}")
    save_ieee_figure(fig, save_path)


def visualize_physics_comparison(
    X_field,
    y_wld,
    y_shape,
    unique_shapes,
    shape_map,
    y_scaler,
    output_dir,
    num_samples=5,
    prefix='',
    verbose=True,
):
    """
    Compute analytical PINN forward fields and save IEEE publication-ready diagnostics.
    """
    freq, sicma, mu = 5000, 35461000, 0.0000012566
    csi, K, I, G = 0.085, 1.5, 0.01, 3981
    delta = 1.0 / math.sqrt(math.pi * freq * mu * sicma) * 1000.0
    z_lift = 1.0
    N, Res = 16, 0.78

    samples_per_shape = max(1, num_samples // len(unique_shapes))
    sample_indices = []

    for shape_idx in range(len(unique_shapes)):
        shape_mask = np.where(y_shape == shape_idx)[0]
        if len(shape_mask) > 0:
            selected = np.random.choice(shape_mask, min(samples_per_shape, len(shape_mask)), replace=False)
            sample_indices.extend(selected)

    sample_indices = sample_indices[:num_samples]

    for idx, sample_id in enumerate(sample_indices):
        try:
            H_true = X_field[sample_id]
            if H_true.ndim == 3:
                H_true = H_true[:, :, 0]

            w_norm, l_norm, d_norm = y_wld[sample_id]
            if y_scaler is not None and hasattr(y_scaler, 'data_max_'):
                w_max, l_max, d_max = y_scaler.data_max_
                w_val = float(w_norm * w_max)
                l_val = float(l_norm * l_max)
                d_val = float(d_norm * d_max)
            else:
                w_val, l_val, d_val = float(w_norm), float(l_norm), float(d_norm)

            w_val = np.clip(w_val, 0.3, 1.5)
            l_val = np.clip(l_val, 2.0, 20.0)
            d_val = np.clip(d_val, 0.1, 3.0)

            shape_idx = int(y_shape[sample_id])
            shape_name = unique_shapes[shape_idx]

            if shape_name not in shape_map:
                continue

            physics_module = shape_map[shape_name]

            if shape_name == 'Ellipse':
                H_np = physics_module.compute_map(w_val, l_val, d_val, delta, z_lift)
                H_raw = (csi / (4 * math.pi)) * H_np * K * I * G
                H_pred = np.rot90(H_raw, 2)
                H_pred = np.diff(H_pred, axis=1)
                H_pred = np.concatenate([H_pred, H_pred[:, -1:]], axis=1)

            elif shape_name == 'Rectangular':
                H_np, _, _ = physics_module.compute_magnetic_field(
                    w_val, l_val, d_val, delta, z_lift, N, Res, angle=0, K=K, I=I, G=G, csi=csi
                )
                H_pred = np.rot90(H_np, 2)
                H_pred = np.diff(H_pred, axis=1)
                H_pred = np.concatenate([H_pred, H_pred[:, -1:]], axis=1)

            elif shape_name == 'Triangular':
                H_np = physics_module.compute_triangular_map(w_val, l_val, d_val, delta, z_lift)
                H_raw = (csi / (4 * math.pi)) * H_np * K * I * G
                H_pred = np.rot90(H_raw, 2)
                H_pred = np.diff(H_pred, axis=1)
                H_pred = np.concatenate([H_pred, H_pred[:, -1:]], axis=1)

            elif shape_name == 'Step_R':
                xs = np.linspace(-N * Res, N * Res, 2 * N)
                ys = np.linspace(-N * Res, N * Res, 2 * N)
                X_grid, Y_grid = np.meshgrid(xs, ys)
                H_val = physics_module.calculate_field_Step_R(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
                H_raw = H_val * (csi / (4 * math.pi)) * K * I * G
                H_pred = np.rot90(H_raw, 2)
                H_pred = np.diff(H_pred, axis=1)
                H_pred = np.flip(H_pred, axis=1)
                H_pred = np.concatenate([H_pred, H_pred[:, -1:]], axis=1)

            elif shape_name == 'Step_T':
                xs = np.linspace(-N * Res, N * Res, 2 * N)
                ys = np.linspace(-N * Res, N * Res, 2 * N)
                X_grid, Y_grid = np.meshgrid(xs, ys)
                H_val = physics_module.calculate_field_Step_T(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
                H_raw = H_val * (csi / (4 * math.pi)) * K * I * G
                H_pred = np.rot90(H_raw, 2)
                H_pred = np.diff(H_pred, axis=1)
                H_pred = np.flip(H_pred, axis=1)
                H_pred = np.concatenate([H_pred, H_pred[:, -1:]], axis=1)
            else:
                continue

            plot_physics_comparison(
                H_true=H_true,
                H_pred=H_pred,
                output_dir=output_dir,
                sample_idx=idx,
                filename_prefix=f"{prefix}physics_field_comparison_{shape_name}",
            )
        except Exception as e:
            if verbose:
                print(f"[WARN] Error during physics visualization sample {idx}: {e}")
            continue

    if verbose:
        print(f"[OK] IEEE physics comparison figures saved to: {output_dir}")
