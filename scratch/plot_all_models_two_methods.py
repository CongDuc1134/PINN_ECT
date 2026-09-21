import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\Admin\Documents\paper\PINN_ECT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 12.0,
    "figure.titleweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Color palette
C_CNN_PINN  = "#00897B"   # Deep Teal Green (Proposed PINN)
C_CNN_BASE  = "#D81B60"   # Vivid Berry Rose (NoPINN Baseline)
C_MLP_MULTI = "#1E88E5"   # Sapphire Blue (Multitask MLP PINN)
C_MLP_XIONG = "#F59E0B"   # Amber Gold (Xiong et al. MLP PINN)

# ==============================================================================
# DATA: ALL MODELS UNDER BOTH EVALUATION METHODS
# Method 1: Repeat Scan (Standard Holdout on Known Specimens)
# Method 2: 10-Fold LODO (Leave-One-Defect-Out on Unseen Blind Specimens)
# ==============================================================================

# Classification models (3 models: CNN PINN, CNN Baseline, Multitask MLP)
# Format: [Repeat_Scan_Value, LODO_Value]
acc_data = {
    "CNN PINN (PI-LGL)":         [80.0, 60.0],
    "CNN Baseline (NoPINN)":     [90.0, 55.0],
    "Multitask MLP (1D PINN)":   [30.8, 12.5],
}

f1_data = {
    "CNN PINN (PI-LGL)":         [72.50, 52.64],
    "CNN Baseline (NoPINN)":     [85.00, 39.31],
    "Multitask MLP (1D PINN)":   [22.07, 10.50],
}

# Crack Length L MAE (mm) - 4 models including Xiong
l_mae_data = {
    "CNN PINN (PI-LGL)":         [0.448, 0.524],    # Mean across 11: 0.739, Best: 0.348
    "CNN Baseline (NoPINN)":     [1.192, 0.771],    # Mean across 8:  1.091
    "Multitask MLP (1D PINN)":   [6.110, 9.020],
    "Xiong et al. MLP (1D)":     [1546.9, 1538.98],
}

# Crack Length L NMAE (%) - 4 models
l_nmae_data = {
    "CNN PINN (PI-LGL)":         [4.48, 5.24],      # Mean across 11: 7.40%
    "CNN Baseline (NoPINN)":     [11.92, 7.71],     # Mean across 8: 10.91%
    "Multitask MLP (1D PINN)":   [61.09, 90.20],
    "Xiong et al. MLP (1D)":     [15469.5, 15389.8],
}

# Width W MAE (mm)
w_mae_data = {
    "CNN PINN (PI-LGL)":         [0.081, 0.105],
    "CNN Baseline (NoPINN)":     [0.073, 0.098],
    "Multitask MLP (1D PINN)":   [0.363, 0.201],
    "Xiong et al. MLP (1D)":     [69.81, 1.320],
}

# Depth D MAE (mm)
d_mae_data = {
    "CNN PINN (PI-LGL)":         [0.672, 0.612],
    "CNN Baseline (NoPINN)":     [0.464, 0.574],
    "Multitask MLP (1D PINN)":   [2.377, 2.910],
    "Xiong et al. MLP (1D)":     [248.5, 250.10],
}

def plot_all_models_two_methods():
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=300)
    plt.subplots_adjust(wspace=0.26, hspace=0.36)

    # =========================================================================
    # PANEL (a): Classification Accuracy & Macro F1 Across Both Methods
    # =========================================================================
    # STRICT RULE: Xiong has NO classification head -> Excluded with clear note!
    ax = axs[0, 0]
    clf_names = ["CNN PINN\n(PI-LGL)", "CNN Baseline\n(NoPINN)", "Multitask MLP\n(1D PINN)"]
    x = np.arange(len(clf_names))
    w = 0.18

    # Repeat Scan Acc & F1
    b1 = ax.bar(x - 1.5*w, [acc_data[m][0] for m in acc_data], w, color=C_CNN_PINN, edgecolor="black", lw=0.8, label="Acc: Phôi Quen (Repeat)")
    b2 = ax.bar(x - 0.5*w, [f1_data[m][0] for m in f1_data], w, color=C_CNN_PINN, alpha=0.50, hatch="//", edgecolor="black", lw=0.8, label="F1: Phôi Quen (Repeat)")
    
    # LODO Acc & F1
    b3 = ax.bar(x + 0.5*w, [acc_data[m][1] for m in acc_data], w, color="#FB8C00", edgecolor="black", lw=0.8, label="Acc: Phôi Lạ (LODO)")
    b4 = ax.bar(x + 1.5*w, [f1_data[m][1] for m in f1_data], w, color="#FB8C00", alpha=0.50, hatch="//", edgecolor="black", lw=0.8, label="F1: Phôi Lạ (LODO)")

    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, fontweight="bold")
    ax.set_ylabel("Tỷ Lệ (%)", fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_title("(a) Khả Năng Phân Loại: Phôi Quen vs Phôi Lạ (LODO)", pad=8)
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.2)

    # Values on bars
    for bars_grp in [b1, b2, b3, b4]:
        for bar in bars_grp:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=6.8, fontweight="bold")

    # Footnote explaining Xiong
    ax.text(0.02, 0.88, "*Xiong et al.: Thuần hồi quy (W, L, D)\nKhông có tác vụ phân loại (N/A)", 
            transform=ax.transAxes, ha="left", fontsize=7.0, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8))

    # =========================================================================
    # PANEL (b): Crack Length Sizing Error (L MAE & NMAE) Across Both Methods
    # =========================================================================
    # All 4 models included!
    ax = axs[0, 1]
    all_names = ["CNN PINN\n(PI-LGL)", "CNN Baseline\n(NoPINN)", "Multitask MLP\n(1D PINN)", "Xiong et al.\nMLP (1D)"]
    x4 = np.arange(len(all_names))
    w4 = 0.32

    # Repeat Scan vs LODO L-MAE
    b_l1 = ax.bar(x4 - w4/2, [l_mae_data[m][0] for m in l_mae_data], w4, color=C_CNN_PINN, edgecolor="black", lw=0.8, label="MAE $L$: Phôi Quen (Repeat)")
    b_l2 = ax.bar(x4 + w4/2, [l_mae_data[m][1] for m in l_mae_data], w4, color="#FB8C00", edgecolor="black", lw=0.8, label="MAE $L$: Phôi Lạ (LODO)")

    ax.set_yscale("log")
    ax.set_ylim(0.1, 5000)
    ax.axhline(0.50, color="#2e7d32", ls="--", lw=1.2, label="Chuẩn NDT Nghiêm Ngặt (< 0.5 mm / 5%)")
    ax.axhline(1.00, color="#d32f2f", ls="-.", lw=1.2, label="Ngưỡng Trần NDT (< 1.0 mm / 10%)")

    ax.set_xticks(x4)
    ax.set_xticklabels(all_names, fontweight="bold")
    ax.set_ylabel("MAE Chiều Dài $L$ (mm) [Thang Log]", fontweight="bold")
    ax.set_title("(b) Sai Số Chiều Dài $L$: So Sánh Cả 4 Dòng Kiến Trúc", pad=8)
    ax.grid(axis="y", which="both")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=7.2)

    # Values on bars
    for b in b_l1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.25, f"{b.get_height():.2f}", ha="center", fontsize=6.8, fontweight="bold", color=C_CNN_PINN)
    for b in b_l2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.25, f"{b.get_height():.2f}", ha="center", fontsize=6.8, fontweight="bold", color="#c2410c")

    # =========================================================================
    # PANEL (c): 3D Dimensional Error (W, L, D) on Phôi Lạ (LODO Blind Test)
    # =========================================================================
    ax = axs[1, 0]
    dims = ["Chiều Rộng ($W$)", "Chiều Dài ($L$)", "Độ Sâu ($D$)"]
    xd = np.arange(len(dims))
    wd = 0.18

    # In LODO:
    # CNN PINN:   W=0.105, L=0.524, D=0.612
    # CNN Base:   W=0.098, L=0.771, D=0.574
    # MLP Multi:  W=0.201, L=9.020, D=2.910
    # MLP Xiong:  W=1.320, L=1538.98, D=250.10
    vals_pinn  = [w_mae_data["CNN PINN (PI-LGL)"][1], l_mae_data["CNN PINN (PI-LGL)"][1], d_mae_data["CNN PINN (PI-LGL)"][1]]
    vals_base  = [w_mae_data["CNN Baseline (NoPINN)"][1], l_mae_data["CNN Baseline (NoPINN)"][1], d_mae_data["CNN Baseline (NoPINN)"][1]]
    vals_multi = [w_mae_data["Multitask MLP (1D PINN)"][1], l_mae_data["Multitask MLP (1D PINN)"][1], d_mae_data["Multitask MLP (1D PINN)"][1]]
    vals_xiong = [w_mae_data["Xiong et al. MLP (1D)"][1], l_mae_data["Xiong et al. MLP (1D)"][1], d_mae_data["Xiong et al. MLP (1D)"][1]]

    ax.bar(xd - 1.5*wd, vals_pinn,  wd, color=C_CNN_PINN,  edgecolor="black", lw=0.8, label="CNN PINN (PI-LGL)")
    ax.bar(xd - 0.5*wd, vals_base,  wd, color=C_CNN_BASE,  edgecolor="black", lw=0.8, label="CNN Baseline (NoPINN)")
    ax.bar(xd + 0.5*wd, vals_multi, wd, color=C_MLP_MULTI, edgecolor="black", lw=0.8, label="Multitask MLP (1D)")
    ax.bar(xd + 1.5*wd, vals_xiong, wd, color=C_MLP_XIONG, edgecolor="black", lw=0.8, label="Xiong et al. MLP (1D)")

    ax.set_yscale("log")
    ax.set_ylim(0.04, 3000)
    ax.axhline(0.50, color="#2e7d32", ls="--", lw=1.0)
    ax.axhline(1.00, color="#d32f2f", ls="-.", lw=1.0)

    ax.set_xticks(xd)
    ax.set_xticklabels(dims, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Sai Số MAE (mm) [Thang Log]", fontweight="bold")
    ax.set_title("(c) Phân Rã Kích Thước 3 Chiều Trên Phôi Lạ (10-Fold LODO)", pad=8)
    ax.grid(axis="y", which="both")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=7.2)

    # =========================================================================
    # PANEL (d): Robustness Scorecard: Độ Bền Vững & Sụt Giảm Hiệu Năng
    # =========================================================================
    ax = axs[1, 1]
    metrics_names = ["Acc Sụt Giảm\n(Phôi Quen → Lạ)", "F1 Sụt Giảm\n(Phôi Quen → Lạ)", "Sai Số $L$ Tăng\n(Phôi Quen)"]
    xm = np.arange(2)
    wm = 0.22

    # Accuracy drop:
    # PINN: 80 - 60 = 20%
    # NoPINN: 90 - 55 = 35%
    # MLP Multi: 30.8 - 12.5 = 18.3%
    # F1 drop:
    # PINN: 72.5 - 52.64 = 19.86%
    # NoPINN: 85.0 - 39.31 = 45.69%
    # MLP Multi: 22.07 - 10.50 = 11.57%

    b_d1 = ax.bar(xm - wm, [20.0, 19.9], wm, color=C_CNN_PINN, edgecolor="black", lw=0.8, label="CNN PINN (Ổn định nhất)")
    b_d2 = ax.bar(xm,      [35.0, 45.7], wm, color=C_CNN_BASE, edgecolor="black", lw=0.8, label="CNN NoPINN (Sụp đổ học vẹt)")
    b_d3 = ax.bar(xm + wm, [18.3, 11.6], wm, color=C_MLP_MULTI, edgecolor="black", lw=0.8, label="Multitask MLP (Kẹt trần thấp)")

    ax.set_xticks(xm)
    ax.set_xticklabels(metrics_names[:2], fontweight="bold")
    ax.set_ylabel("Mức Độ Sụt Giảm (%) [Càng Nhỏ Càng Tốt]", fontweight="bold")
    ax.set_ylim(0, 55)
    ax.set_title("(d) Mức Độ Sụt Giảm Khi Chuyển Sang Phôi Lạ (Robustness)", pad=8)
    ax.grid(axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=7.2)

    ax.text(xm[0] - wm, 21.5, "-20.0%", ha="center", fontsize=7.2, fontweight="bold", color=C_CNN_PINN)
    ax.text(xm[0],      36.5, "-35.0%", ha="center", fontsize=7.2, fontweight="bold", color=C_CNN_BASE)
    ax.text(xm[0] + wm, 19.8, "-18.3%", ha="center", fontsize=7.2, fontweight="bold", color=C_MLP_MULTI)

    ax.text(xm[1] - wm, 21.5, "-19.9%", ha="center", fontsize=7.2, fontweight="bold", color=C_CNN_PINN)
    ax.text(xm[1],      47.0, "-45.7%", ha="center", fontsize=7.2, fontweight="bold", color="#b91c1c")
    ax.text(xm[1] + wm, 13.0, "-11.6%", ha="center", fontsize=7.2, fontweight="bold", color=C_MLP_MULTI)

    # Highlight NoPINN catastrophic drop
    ax.annotate("F1 NoPINN rớt 45.7%!\n(Mất vân tay phôi cũ)", xy=(xm[1], 45.7), xytext=(xm[1] + 0.35, 35),
                arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.0, headwidth=4),
                fontsize=7.2, fontweight="bold", color="#b91c1c")

    fig.suptitle("ĐỐI ĐẦU 2 PHƯƠNG PHÁP ĐÁNH GIÁ (PHÔI QUEN VS PHÔI LẠ LODO) TRÊN TOÀN BỘ CÁC DÒNG KIẾN TRÚC\nSo Sánh Toàn Diện: CNN PINN (PI-LGL) vs CNN Baseline (NoPINN) vs Multitask MLP vs Xiong et al. MLP", 
                 fontsize=11.5, fontweight="bold", y=0.995)

    # Save to both target filenames in OUTPUT_DIR
    out_file1 = os.path.join(OUTPUT_DIR, "fig_all_models_two_methods_head_to_head.png")
    out_file2 = os.path.join(OUTPUT_DIR, "fig_two_methods_head_to_head.png")
    
    plt.savefig(out_file1)
    plt.savefig(out_file2)
    plt.close()
    print(f"[OK] Saved: {out_file1}")
    print(f"[OK] Overwritten & Updated: {out_file2}")

if __name__ == "__main__":
    plot_all_models_two_methods()
