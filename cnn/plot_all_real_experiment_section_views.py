# -*- coding: utf-8 -*-
"""
===============================================================================
TOÀN BỘ SECTION VIEW & HEATMAP CHO DỮ LIỆU THỰC NGHIỆM EXPERIMENT_1/5KHZ
===============================================================================
Nạp đầy đủ 20 mẫu thực nghiệm (10 vết nứt × 2 lần quét: Scan 1 và Scan 2).
Mô phỏng tín hiệu giải tích 5kHz tương ứng cho từng vết nứt.
Trích xuất mặt cắt trung tâm (Section View tại Width ≈ 0) và vẽ đồ thị IEEE chuẩn:
1. Đồ thị tổng hợp 10 vết nứt (So sánh Scan 1 vs Scan 2 vs Simulation).
2. 10 đồ thị chi tiết cho từng vết nứt (Heatmap Scan 1, Heatmap Scan 2, Heatmap Sim, Section View).
===============================================================================
"""

import os
import sys
import math
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thông số vật lý 5kHz
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
DELTA = 1.0 / math.sqrt(math.pi * FREQ * MU * SICMA) * 1000.0

REAL_FILES_SCAN1 = {
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

REAL_FILES_SCAN2 = {
    1: "5khz_No1_1.csv",
    2: "5khz_No2_1.csv",
    3: "5khz_No3_1.csv",
    4: "5khz_No4_1.csv",
    5: "5khz_No5_1.csv",
    6: "5khz_No6_1.csv",
    7: "5khz_No7_1.csv",
    8: "5khz_No8_1.csv",
    9: "5khz_No9_1.csv",
    10: "5khz_No10_1.csv",
}


def simulate_defect(shape: str, wc: float, lc: float, dc: float) -> np.ndarray:
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
        raise ValueError(f"Không hỗ trợ hình dạng: {shape}")


def load_matrix(path: str) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(np.float32)


def main():
    exp_base = find_experiment_1_dir()
    exp_5khz = os.path.join(exp_base, "5khz")
    out_dir = os.path.join(PROJECT_ROOT, "Experiment_1", "5khz_section_views")
    os.makedirs(out_dir, exist_ok=True)

    dist_length = np.linspace(-N_GRID * RES, N_GRID * RES, 32)
    mid_idx = 16  # Centerline tại Width ≈ 0
    extent = [dist_length[0], dist_length[-1], dist_length[0], dist_length[-1]]

    defect_data = []

    print("=" * 80)
    print("NẠP TOÀN BỘ 20 MẪU EXPERIMENT_1/5KHZ & TRÍCH XUẤT SECTION VIEW")
    print(f"Thư mục dữ liệu: {exp_5khz}")
    print(f"Thư mục lưu đồ thị: {out_dir}")
    print("=" * 80)

    for crack_no in range(1, 11):
        gt = TABLE_2_GROUND_TRUTH[crack_no]
        shape = gt["shape"]
        w = gt["W"]
        l = gt["L"]
        d = gt["D"]

        f_scan1 = os.path.join(exp_5khz, REAL_FILES_SCAN1[crack_no])
        f_scan2 = os.path.join(exp_5khz, REAL_FILES_SCAN2[crack_no])

        arr_scan1 = load_matrix(f_scan1)
        arr_scan2 = load_matrix(f_scan2)
        arr_sim = simulate_defect(shape, w, l, d)

        sec_scan1 = arr_scan1[mid_idx, :]
        sec_scan2 = arr_scan2[mid_idx, :]
        sec_sim = arr_sim[mid_idx, :]

        defect_data.append({
            "crack_no": crack_no,
            "shape": shape,
            "w": w,
            "l": l,
            "d": d,
            "scan1_file": REAL_FILES_SCAN1[crack_no],
            "scan2_file": REAL_FILES_SCAN2[crack_no],
            "arr_scan1": arr_scan1,
            "arr_scan2": arr_scan2,
            "arr_sim": arr_sim,
            "sec_scan1": sec_scan1,
            "sec_scan2": sec_scan2,
            "sec_sim": sec_sim,
        })

        # --- ĐỒ THỊ 4-PANEL CHI TIẾT TỪNG VẾT NỨT ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=300)

        # 1. Heatmap Scan 1
        im1 = axes[0].imshow(arr_scan1, cmap="jet", origin="lower", extent=extent, aspect="auto")
        axes[0].axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.2)
        axes[0].set_title(f"(a) Scan 1 (Thực nghiệm)\n{REAL_FILES_SCAN1[crack_no]}", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Length (mm)", fontsize=9); axes[0].set_ylabel("Width (mm)", fontsize=9)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # 2. Heatmap Scan 2
        im2 = axes[1].imshow(arr_scan2, cmap="jet", origin="lower", extent=extent, aspect="auto")
        axes[1].axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.2)
        axes[1].set_title(f"(b) Scan 2 (Lần quét lại)\n{REAL_FILES_SCAN2[crack_no]}", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Length (mm)", fontsize=9); axes[1].set_ylabel("Width (mm)", fontsize=9)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Heatmap Mô phỏng 5kHz
        im3 = axes[2].imshow(arr_sim, cmap="jet", origin="lower", extent=extent, aspect="auto")
        axes[2].axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.2)
        axes[2].set_title(f"(c) Simulation 5kHz (Mô phỏng)\n{shape} ({w}×{l}×{d} mm)", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Length (mm)", fontsize=9); axes[2].set_ylabel("Width (mm)", fontsize=9)
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Section View đối chiếu trực diện 3 đường
        axes[3].plot(dist_length, sec_scan1, color="#1f77b4", linewidth=2.2, label="Scan 1 (Real)")
        axes[3].plot(dist_length, sec_scan2, color="#d62728", linewidth=2.0, linestyle="--", marker="o", markersize=3, label="Scan 2 (Real)")
        axes[3].plot(dist_length, sec_sim, color="#2ca02c", linewidth=2.0, linestyle=":", label="Simulation")
        axes[3].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        axes[3].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        axes[3].grid(True, linestyle="--", alpha=0.5)
        axes[3].set_title(f"(d) Section View (Width ≈ 0)", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("Length (mm)", fontsize=9); axes[3].set_ylabel("Signal Intensity ($H_z$)", fontsize=9)
        axes[3].legend(loc="upper right", fontsize=8, framealpha=0.9)

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"section_view_crack_{crack_no:02d}_{shape}")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[*] Vết nứt No.{crack_no:02d} ({shape:11s}): Đã lưu đồ thị vào {os.path.basename(save_path)}.png & .pdf")

    # --- ĐỒ THỊ TỔNG HỢP 1: 10 SECTION VIEWS SO SÁNH SCAN 1 VS SCAN 2 ---
    print("\n>>> Đang tạo Đồ thị Tổng hợp 1: So sánh Scan 1 vs Scan 2 cho 10 vết nứt...")
    fig1, axes1 = plt.subplots(2, 5, figsize=(22, 8.5), dpi=300, sharex=True, sharey=False)
    for idx, d in enumerate(defect_data):
        ax = axes1.flatten()[idx]
        ax.plot(dist_length, d["sec_scan1"], color="#1f77b4", linewidth=2.0, label="Scan 1 (Real)")
        ax.plot(dist_length, d["sec_scan2"], color="#d62728", linewidth=1.8, linestyle="--", marker="o", markersize=2.5, label="Scan 2 (Real)")
        ax.plot(dist_length, d["sec_sim"], color="#2ca02c", linewidth=1.8, linestyle=":", label="Simulation (5kHz)")
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(f"No.{d['crack_no']} - {d['shape']}\n({d['w']}×{d['l']}×{d['d']} mm)", fontsize=10, fontweight="bold")
        if idx % 5 == 0:
            ax.set_ylabel("Intensity ($H_z$)", fontsize=9)
        if idx >= 5:
            ax.set_xlabel("Length (mm)", fontsize=9)

    handles1, labels1 = axes1.flatten()[0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_master1 = os.path.join(out_dir, "fig_all_10_defects_scan1_vs_scan2_section_views")
    fig1.savefig(f"{save_master1}.png", bbox_inches="tight", dpi=300)
    fig1.savefig(f"{save_master1}.pdf", bbox_inches="tight")
    plt.close(fig1)
    print(f"[OK] Đã lưu Đồ thị Tổng hợp 1 vào: {save_master1}.png & .pdf")

    # --- ĐỒ THỊ TỔNG HỢP 2: LƯỚI 20 HEATMAPS TOÀN BỘ DỮ LIỆU THỰC NGHIỆM ---
    print(">>> Đang tạo Đồ thị Tổng hợp 2: Lưới 20 Heatmaps toàn bộ dữ liệu thực nghiệm...")
    fig2, axes2 = plt.subplots(4, 5, figsize=(20, 14), dpi=300)
    for crack_no in range(1, 11):
        d = defect_data[crack_no - 1]
        col = (crack_no - 1) % 5

        # Hàng 0, 1: Scan 1
        row_s1 = 0 if crack_no <= 5 else 2
        ax_s1 = axes2[row_s1, col]
        im_s1 = ax_s1.imshow(d["arr_scan1"], cmap="jet", origin="lower", extent=extent, aspect="auto")
        ax_s1.axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.0)
        ax_s1.set_title(f"No.{crack_no} Scan 1: {d['shape']}\n({d['w']}×{d['l']}×{d['d']} mm)", fontsize=9, fontweight="bold")
        ax_s1.set_xticks([]); ax_s1.set_yticks([])

        # Hàng 1, 3: Scan 2
        row_s2 = 1 if crack_no <= 5 else 3
        ax_s2 = axes2[row_s2, col]
        im_s2 = ax_s2.imshow(d["arr_scan2"], cmap="jet", origin="lower", extent=extent, aspect="auto")
        ax_s2.axhline(y=dist_length[mid_idx], color="white", linestyle="--", linewidth=1.0)
        ax_s2.set_title(f"No.{crack_no} Scan 2 (Repeat)\n({d['scan2_file']})", fontsize=9)
        ax_s2.set_xticks([]); ax_s2.set_yticks([])

    plt.tight_layout()
    save_master2 = os.path.join(out_dir, "fig_all_20_real_experimental_heatmaps_grid")
    fig2.savefig(f"{save_master2}.png", bbox_inches="tight", dpi=300)
    fig2.savefig(f"{save_master2}.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Đã lưu Đồ thị Tổng hợp 2 vào: {save_master2}.png & .pdf")
    print(f"\nHOÀN TẤT TOÀN BỘ! Các file đồ thị chuẩn IEEE đã sẵn sàng tại:\n{out_dir}")


if __name__ == "__main__":
    main()
