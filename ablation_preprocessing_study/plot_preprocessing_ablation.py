# -*- coding: utf-8 -*-
"""
scratch/plot_preprocessing_ablation.py
Generates publication-quality IEEE figures for the Preprocessing Ablation Study:
1. fig_ablation_preprocessing_head_to_head.png:
   - Impact of (1) Baseline Nulling and (2) Gaussian Smoothing on Acc, F1, and Length MAE.
2. fig_sensor_drift_and_jitter_profiles.png:
   - Signal waveform and Laplacian field explosion comparison (Raw vs Calibrated).
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ablation_preprocessing_study")

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


def plot_ablation_head_to_head():
    csv_path = os.path.join(OUTPUT_DIR, "ablation_summary_table.csv")
    df = pd.read_csv(csv_path)

    configs = [
        "Full Preprocessing (Proposed)",
        "Ablation: No Baseline Nulling",
        "Ablation: No Gaussian Smoothing",
        "Completely Raw Sensor (No Nulling & No Smoothing)",
    ]
    labels = [
        "Full Pipeline\n(Nulling + Smooth)",
        "No Nulling\n(DC Drift Kept)",
        "No Smoothing\n(Jitter Kept)",
        "Completely Raw\n(No Null & No Smooth)"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    x = np.arange(len(configs))
    w = 0.18

    # Colors
    C_PINN_REP   = "#1565C0"  # Dark Blue
    C_NOPINN_REP = "#78909C"  # Greyish Blue
    C_PINN_LODO  = "#E65100"  # Amber Orange
    C_NOPINN_LODO= "#B0BEC5"  # Light Grey

    # Data arrays
    # 1. Macro F1
    f1_pinn_rep, f1_nopinn_rep = [], []
    f1_pinn_lodo, f1_nopinn_lodo = [], []
    # 2. Accuracy
    acc_pinn_rep, acc_nopinn_rep = [], []
    acc_pinn_lodo, acc_nopinn_lodo = [], []
    # 3. MAE Length
    mae_l_pinn_rep, mae_l_nopinn_rep = [], []
    mae_l_pinn_lodo, mae_l_nopinn_lodo = [], []

    for cfg in configs:
        # PINN Repeat
        r1 = df[(df["Configuration"] == cfg) & (df["Model"] == "CNN PINN (PI-LGL)") & (df["Protocol"] == "Repeat Scan (In-Dist)")].iloc[0]
        f1_pinn_rep.append(r1["Macro F1 (%)"])
        acc_pinn_rep.append(r1["Accuracy (%)"])
        mae_l_pinn_rep.append(r1["MAE L (mm)"])

        # NoPINN Repeat
        r2 = df[(df["Configuration"] == cfg) & (df["Model"] == "CNN Baseline (NoPINN)") & (df["Protocol"] == "Repeat Scan (In-Dist)")].iloc[0]
        f1_nopinn_rep.append(r2["Macro F1 (%)"])
        acc_nopinn_rep.append(r2["Accuracy (%)"])
        mae_l_nopinn_rep.append(r2["MAE L (mm)"])

        # PINN LODO
        r3 = df[(df["Configuration"] == cfg) & (df["Model"] == "CNN PINN (PI-LGL)") & (df["Protocol"] == "10-Fold LODO (Out-of-Dist)")].iloc[0]
        f1_pinn_lodo.append(r3["Macro F1 (%)"])
        acc_pinn_lodo.append(r3["Accuracy (%)"])
        mae_l_pinn_lodo.append(r3["MAE L (mm)"])

        # NoPINN LODO
        r4 = df[(df["Configuration"] == cfg) & (df["Model"] == "CNN Baseline (NoPINN)") & (df["Protocol"] == "10-Fold LODO (Out-of-Dist)")].iloc[0]
        f1_nopinn_lodo.append(r4["Macro F1 (%)"])
        acc_nopinn_lodo.append(r4["Accuracy (%)"])
        mae_l_nopinn_lodo.append(r4["MAE L (mm)"])

    # Panel 1: Accuracy
    ax1 = axes[0]
    ax1.bar(x - 1.5*w, acc_pinn_rep, w, color=C_PINN_REP, edgecolor="black", lw=0.8, label="PINN (Repeat Scan)")
    ax1.bar(x - 0.5*w, acc_nopinn_rep, w, color=C_NOPINN_REP, edgecolor="black", lw=0.8, label="NoPINN (Repeat Scan)")
    ax1.bar(x + 0.5*w, acc_pinn_lodo, w, color=C_PINN_LODO, edgecolor="black", lw=0.8, label="PINN (10-Fold LODO)")
    ax1.bar(x + 1.5*w, acc_nopinn_lodo, w, color=C_NOPINN_LODO, edgecolor="black", lw=0.8, label="NoPINN (10-Fold LODO)")
    ax1.set_title("(a) Classification Accuracy (%)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Macro F1
    ax2 = axes[1]
    ax2.bar(x - 1.5*w, f1_pinn_rep, w, color=C_PINN_REP, edgecolor="black", lw=0.8)
    ax2.bar(x - 0.5*w, f1_nopinn_rep, w, color=C_NOPINN_REP, edgecolor="black", lw=0.8)
    ax2.bar(x + 0.5*w, f1_pinn_lodo, w, color=C_PINN_LODO, edgecolor="black", lw=0.8)
    ax2.bar(x + 1.5*w, f1_nopinn_lodo, w, color=C_NOPINN_LODO, edgecolor="black", lw=0.8)
    ax2.set_title("(b) Macro F1-Score (%)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Macro F1 (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: Length MAE
    ax3 = axes[2]
    ax3.bar(x - 1.5*w, mae_l_pinn_rep, w, color=C_PINN_REP, edgecolor="black", lw=0.8)
    ax3.bar(x - 0.5*w, mae_l_nopinn_rep, w, color=C_NOPINN_REP, edgecolor="black", lw=0.8)
    ax3.bar(x + 0.5*w, mae_l_pinn_lodo, w, color=C_PINN_LODO, edgecolor="black", lw=0.8)
    ax3.bar(x + 1.5*w, mae_l_nopinn_lodo, w, color=C_NOPINN_LODO, edgecolor="black", lw=0.8)
    ax3.axhline(0.5, color="red", linestyle=":", lw=1.2, label="NDT Precision Limit (0.5 mm)")
    ax3.set_title("(c) Crack Length Estimation Error MAE (mm)", fontsize=11, fontweight="bold", pad=10)
    ax3.set_ylabel("MAE Length $L$ (mm)", fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 1.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # Consolidated Legend outside at the top
    handles, leg_labels = ax1.get_legend_handles_labels()
    h_ndt, l_ndt = ax3.get_legend_handles_labels()
    all_h = handles + h_ndt
    all_l = leg_labels + l_ndt

    fig.legend(all_h, all_l, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=5, frameon=True, fontsize=9.0, edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(OUTPUT_DIR, "fig_ablation_preprocessing_head_to_head.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure: {save_path}")


def plot_signal_drift_and_jitter_profiles():
    """Visualizes 2D image and 1D slice of raw sensor signal vs calibrated signal."""
    from load_real_experiment_data import find_experiment_1_dir
    exp1_dir = find_experiment_1_dir()
    train_dir = os.path.join(exp1_dir, "Trainning")
    raw_csv = os.path.join(train_dir, "5khz_No1_1.csv")

    arr_raw = pd.read_csv(raw_csv, header=None).values.astype(np.float32)
    if arr_raw.shape != (32, 32):
        arr_raw = scipy.ndimage.zoom(arr_raw, (32.0/arr_raw.shape[0], 32.0/arr_raw.shape[1]), order=1)

    # 1. Raw with DC offset & jitter
    field_raw = arr_raw.copy()

    # 2. Calibrated: Nulling + Gaussian smooth
    borders = np.concatenate([
        field_raw[:3, :].flatten(), field_raw[-3:, :].flatten(),
        field_raw[:, :3].flatten(), field_raw[:, -3:].flatten()
    ])
    baseline_offset = np.median(borders)
    field_calib = scipy.ndimage.gaussian_filter(field_raw - baseline_offset, sigma=0.5)

    # Compute Laplacians
    def get_lap(f):
        d2x = np.gradient(np.gradient(f, axis=0), axis=0)
        d2y = np.gradient(np.gradient(f, axis=1), axis=1)
        return np.abs(d2x + d2y)

    lap_raw = get_lap(field_raw)
    lap_calib = get_lap(field_calib)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=300)

    # Row 1: 2D Signal and 1D Profile
    im1 = axes[0, 0].imshow(field_raw, cmap="viridis", origin="lower")
    axes[0, 0].set_title("(a) Raw Sensor Signal $H_{raw}$\n(DC Offset + Mechanical Jitter)", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(field_calib, cmap="viridis", origin="lower")
    axes[0, 1].set_title("(b) Calibrated Signal $H_{calib}$\n(Zero-Nulled + $\\sigma=0.5$ Filtered)", fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 1D Cross section profile (row 16)
    row_idx = 16
    axes[0, 2].plot(field_raw[row_idx, :], color="#d32f2f", lw=1.5, marker="o", markersize=3, label="Raw Sensor Profile")
    axes[0, 2].plot(field_calib[row_idx, :], color="#1976d2", lw=2.0, label="Calibrated Profile (Zeroed)")
    axes[0, 2].axhline(0, color="black", linestyle="--", alpha=0.5, label="True Physical Zero")
    axes[0, 2].set_title(f"(c) Cross-Section Scan Profile (Row {row_idx})", fontsize=10, fontweight="bold")
    axes[0, 2].set_xlabel("Scanning Pixel $X$", fontsize=9)
    axes[0, 2].set_ylabel("Magnetic Potential $H$ (a.u.)", fontsize=9)
    axes[0, 2].legend(fontsize=8, loc="upper right")
    axes[0, 2].grid(True, linestyle=":", alpha=0.5)

    # Row 2: Laplacian 2D Fields & Laplacian 1D Profile
    im3 = axes[1, 0].imshow(lap_raw, cmap="magma", origin="lower")
    axes[1, 0].set_title(f"(d) Raw Spatial Laplacian ||$\\nabla^2 H_{{raw}}||$\n(Max={lap_raw.max():.3f} - High Noise Spike)", fontsize=10, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(lap_calib, cmap="magma", origin="lower")
    axes[1, 1].set_title(f"(e) Calibrated Laplacian ||$\\nabla^2 H_{{calib}}||$\n(Max={lap_calib.max():.3f} - True Defect Curvature)", fontsize=10, fontweight="bold")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 1D Laplacian Profile
    axes[1, 2].plot(lap_raw[row_idx, :], color="#d32f2f", lw=1.5, marker="x", markersize=4, label="Raw Laplacian (Noise Amplified)")
    axes[1, 2].plot(lap_calib[row_idx, :], color="#388e3c", lw=2.0, label="Calibrated Laplacian")
    axes[1, 2].axhline(0.10, color="orange", linestyle="--", lw=1.2, label="Curvature Guard Threshold (0.10)")
    axes[1, 2].set_title(f"(f) Laplacian Second Derivative ||$\\nabla^2 H||$ (Row {row_idx})", fontsize=10, fontweight="bold")
    axes[1, 2].set_xlabel("Scanning Pixel $X$", fontsize=9)
    axes[1, 2].set_ylabel("Laplacian Magnitude ||$\\nabla^2 H||$", fontsize=9)
    axes[1, 2].legend(fontsize=8, loc="upper right")
    axes[1, 2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig_sensor_drift_and_jitter_profiles.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure: {save_path}")


if __name__ == "__main__":
    plot_ablation_head_to_head()
    plot_signal_drift_and_jitter_profiles()
