# -*- coding: utf-8 -*-
"""
===============================================================================
ECT 5kHz SIMULATION & REAL EXPERIMENTAL SECTION VIEW VISUALIZATION
===============================================================================
Mô phỏng trường ECT 5kHz bằng mô hình giải tích trong thư mục cnn/:
- Rectangular (regtangular.py)
- Ellipse (ellipse.py)
- Triangular (triangular.py)
- Step_R (step_r.py)
- Step_T (step_t.py)

Đối chiếu trực diện với dữ liệu đo thực nghiệm trong Experiment_1/5khz (10 mẫu).
Vẽ biểu đồ mặt cắt (Section View - Centerline cut tại Width ~ 0 dọc theo Length)
chuẩn IEEE Transactions (300 DPI, PNG + PDF).
===============================================================================
"""

import os
import sys
import math
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CNN_DIR = os.path.join(PROJECT_ROOT, "cnn")
if CNN_DIR not in sys.path:
    sys.path.insert(0, CNN_DIR)

import regtangular
import ellipse
import triangular
import step_r
import step_t
from domain_adaptation.load_real_experiment_data import TABLE_2_GROUND_TRUTH, find_experiment_1_dir

# Thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hằng số vật lý chuẩn 5kHz
FREQ = 5000.0
SICMA = 35461000.0
MU = 1.2566e-6
CSI = 0.085
K = 1.5
I = 0.01
G = 3981.0
Z_LIFT = 1.0
N_GRID = 16
RES = 0.78
DELTA = 1.0 / math.sqrt(math.pi * FREQ * MU * SICMA) * 1000.0  # mm

# Ánh xạ tên file thực nghiệm cho 10 vết nứt
REAL_FILES = {
    1: "5khz_No1.csv",
    2: "5khz_No2.csv",
    3: "5khz_No3_.csv",
    4: "5khz_No4.csv",
    5: "5khz_No5.csv",
    6: "5khz_No6.csv",
    7: "5khz_No7.csv",
    8: "5khz_No8.csv",
    9: "5khz_No9.csv",
    10: "5khz_No10.csv",
}


def simulate_defect(shape: str, wc: float, lc: float, dc: float) -> np.ndarray:
    """
    Chạy mô phỏng tín hiệu vi sai ECT 32x32 cho từng loại vết nứt.
    """
    shape_lower = shape.lower()

    if shape_lower == "rectangular":
        H_tensor, _, _ = regtangular.compute_magnetic_field(
            wc=wc, lc=lc, dc=dc, delta=DELTA, z=Z_LIFT, angle=0, K=K, I=I, G=G, csi=CSI
        )
        H_raw = H_tensor.detach().cpu().numpy()
        proc_sig, _ = regtangular.process_simulation_output(H_raw)
        return proc_sig

    elif shape_lower == "ellipse":
        H_tensor = ellipse.compute_map_gpu(wc=wc, lc=lc, dc=dc, delta=DELTA, z=Z_LIFT, N=N_GRID, Res=RES)
        H_np = (CSI / 4.0 / np.pi) * K * I * G * H_tensor.detach().cpu().numpy()
        final_data = ellipse.process_simulation_data(H_np)
        return final_data[:, :, 0]

    elif shape_lower == "triangular":
        H_tensor = triangular.compute_triangular_map_gpu(wc=wc, lc=lc, dc=dc, delta=DELTA, z=Z_LIFT, N=N_GRID, Res=RES)
        H_raw = (CSI / 4.0 / math.pi) * K * I * G * H_tensor.detach().cpu().numpy().squeeze()
        proc_sig, _ = triangular.process_simulation_output(H_raw, flip_rows=False)
        return proc_sig

    elif shape_lower in ["step_r", "stepr"]:
        xs = torch.linspace(-N_GRID * RES, N_GRID * RES, 2 * N_GRID, device=device)
        ys = torch.linspace(-N_GRID * RES, N_GRID * RES, 2 * N_GRID, device=device)
        Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
        H_tensor = step_r.calculate_field_Step_R(Xg, Yg, Z_LIFT, wc, lc, dc, DELTA)
        H_raw = (H_tensor * (CSI / 4.0 / math.pi) * K * I * G).detach().cpu().numpy()
        proc_sig = step_r.process_like_load_data(H_raw)
        return proc_sig

    elif shape_lower in ["step_t", "stept"]:
        xs = torch.linspace(-N_GRID * RES, N_GRID * RES, 2 * N_GRID, device=device)
        ys = torch.linspace(-N_GRID * RES, N_GRID * RES, 2 * N_GRID, device=device)
        Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
        H_tensor = step_t.calculate_field_Step_T(Xg, Yg, Z_LIFT, wc, lc, dc, DELTA)
        H_raw = (H_tensor * (CSI / 4.0 / math.pi) * K * I * G).detach().cpu().numpy()
        proc_sig = step_t.process_like_load_data(H_raw)
        return proc_sig

    else:
        raise ValueError(f"Không hỗ trợ hình dạng mô phỏng: {shape}")


def load_real_matrix(filepath: str) -> np.ndarray:
    """Nạp file CSV thực nghiệm 32x32."""
    df = pd.read_csv(filepath, header=None)
    arr = df.values.astype(np.float32)
    return arr


def main():
    print("=" * 80)
    print("MÔ PHỎNG ECT 5kHz & TRÍCH XUẤT SECTION VIEW CHO 10 VẾT NỨT EXPERIMENT_1")
    print(f"Thiết bị: {device} | Tần số: {FREQ} Hz | Độ sâu xuyên sâu delta: {DELTA:.3f} mm")
    print("=" * 80)

    exp_dir = find_experiment_1_dir()
    exp_5khz_dir = os.path.join(exp_dir, "5khz")
    out_dir = os.path.join(PROJECT_ROOT, "cnn", "section_views")
    os.makedirs(out_dir, exist_ok=True)

    dist_length = np.linspace(-N_GRID * RES, N_GRID * RES, 32)
    mid_idx = 16  # Centerline (Width ~ 0)
    extent = [dist_length[0], dist_length[-1], dist_length[0], dist_length[-1]]

    results = []

    # 1. Chạy mô phỏng và nạp dữ liệu thật cho từng vết nứt
    for crack_no in range(1, 11):
        gt = TABLE_2_GROUND_TRUTH[crack_no]
        shape = gt["shape"]
        w = gt["W"]
        l = gt["L"]
        d = gt["D"]
        real_fname = REAL_FILES[crack_no]
        real_fpath = os.path.join(exp_5khz_dir, real_fname)

        print(f"\n[*] Đang xử lý Vết nứt No.{crack_no:02d}: {shape:11s} | W={w}mm, L={l}mm, D={d}mm ...")
        t0 = time.perf_counter()
        sim_data = simulate_defect(shape, w, l, d)
        t1 = time.perf_counter()
        print(f"    -> Mô phỏng hoàn tất trong {t1 - t0:.3f}s. Min/Max: {sim_data.min():.4f} / {sim_data.max():.4f}")

        real_data = load_real_matrix(real_fpath)
        print(f"    -> Đã nạp file thật '{real_fname}'. Min/Max: {real_data.min():.4f} / {real_data.max():.4f}")

        sim_section = sim_data[mid_idx, :]
        real_section = real_data[mid_idx, :]

        results.append({
            "crack_no": crack_no,
            "shape": shape,
            "w": w,
            "l": l,
            "d": d,
            "sim_data": sim_data,
            "real_data": real_data,
            "sim_section": sim_section,
            "real_section": real_section,
        })

        # Vẽ đồ thị 3-panel chi tiết cho từng vết nứt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

        # Panel 1: Simulation Heatmap
        im1 = axes[0].imshow(sim_data, cmap="jet", origin="lower", extent=extent, aspect="auto")
        axes[0].axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.5, label="Section Line (y=0)")
        axes[0].set_title(f"(a) 5kHz Simulation Heatmap\n{shape} ({w}×{l}×{d} mm)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Length (mm)", fontsize=10)
        axes[0].set_ylabel("Width (mm)", fontsize=10)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: Real Experiment Heatmap
        im2 = axes[1].imshow(real_data, cmap="jet", origin="lower", extent=extent, aspect="auto")
        axes[1].axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.5, label="Section Line (y=0)")
        axes[1].set_title(f"(b) Real Experiment Scan\nFile: {real_fname}", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Length (mm)", fontsize=10)
        axes[1].set_ylabel("Width (mm)", fontsize=10)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Centerline Section View
        axes[2].plot(dist_length, sim_section, color="#1f77b4", linewidth=2.2, label="Simulation (5kHz)")
        axes[2].plot(dist_length, real_section, color="#d62728", linewidth=2.0, linestyle="--", marker="o", markersize=3, label="Real Experiment")
        axes[2].axvline(x=0, color="gray", linestyle=":", alpha=0.6)
        axes[2].axhline(y=0, color="gray", linestyle=":", alpha=0.6)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].set_title(f"(c) Centerline Section View (Width ≈ 0)", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Length (mm)", fontsize=10)
        axes[2].set_ylabel("Signal Intensity ($H_z$)", fontsize=10)
        axes[2].legend(loc="upper right", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        save_base = os.path.join(out_dir, f"section_view_crack_{crack_no:02d}_{shape}")
        fig.savefig(f"{save_base}.png", bbox_inches="tight", dpi=300)
        fig.savefig(f"{save_base}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    -> Đã lưu đồ thị vào: {os.path.basename(save_base)}.png & .pdf")

    # 2. VẼ BẢNG TỔNG HỢP 10 MẶT CẮT (SECTION VIEWS) CHO TOÀN BỘ 10 VẾT NỨT
    print("\n>>> Đang tạo đồ thị tổng hợp 10 Section Views (2x5 Grid)...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=300, sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for idx, res in enumerate(results):
        ax = axes_flat[idx]
        ax.plot(dist_length, res["sim_section"], color="#1f77b4", linewidth=2.0, label="Simulation")
        ax.plot(dist_length, res["real_section"], color="#d62728", linewidth=1.8, linestyle="--", label="Real Exp.")
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(
            f"No.{res['crack_no']} - {res['shape']}\n({res['w']}×{res['l']}×{res['d']} mm)",
            fontsize=10,
            fontweight="bold",
        )
        if idx % 5 == 0:
            ax.set_ylabel("Intensity ($H_z$)", fontsize=9)
        if idx >= 5:
            ax.set_xlabel("Length (mm)", fontsize=9)

    # Thêm chú thích thống nhất ở đỉnh
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    all_save_base = os.path.join(out_dir, "fig_all_10_defects_section_views")
    fig.savefig(f"{all_save_base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{all_save_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Đã lưu đồ thị tổng hợp 10 vết nứt vào: {all_save_base}.png & .pdf")
    print(f"\nToàn bộ đồ thị đã được xuất thành công vào: {out_dir}")


if __name__ == "__main__":
    main()
