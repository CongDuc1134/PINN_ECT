import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Target output directory
OUTPUT_DIR = r"C:\Users\Admin\Documents\paper\PINN_ECT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
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

# Palette
C_CNN_PINN  = "#00897B"   # Deep Teal Green (Proposed SOTA)
C_CNN_BASE  = "#D81B60"   # Berry Rose (NoPINN Baseline)
C_MLP_MULTI = "#1E88E5"   # Sapphire Blue (Multitask MLP PINN)
C_MLP_XIONG = "#F59E0B"   # Amber Gold (Xiong et al. MLP PINN)

# ==============================================================================
# EXPERIMENTAL DATA: All Models Under The Proposed PI-LGL Adaptation Method
# ==============================================================================
# Data source: pi_lgl_all_19_models_summary.csv and pi_lgl_all_mlp_models_summary.csv
# CNN PINN:       Acc: 77.3% (Peak 90%), MAE_W: 0.102, MAE_L: 0.739 (Peak 0.348, Rep 0.448), MAE_D: 0.659, Overall: 0.500
# Multitask MLP:  Acc: 30.8%, MAE_W: 0.363, MAE_L: 6.110, MAE_D: 2.377, Overall: 2.950
# Xiong MLP:      Acc: N/A (Thuần hồi quy), MAE_W: 69.81, MAE_L: 1546.9, MAE_D: 248.5, Overall: 621.8

# ------------------------------------------------------------------------------
# 1. FIGURE 1: Comprehensive Comparison Dashboard (CNN vs MLP Under Proposed Method)
# ------------------------------------------------------------------------------
def plot_fig1_mlp_vs_cnn():
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.2), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.35)

    # (a) Classification Accuracy (%)
    # STRICT RULE: Only CNN PINN, CNN NoPINN, and Multitask MLP have classification!
    # Xiong has NO classification head -> Excluded with note!
    ax = axs[0, 0]
    clf_models = ["CNN PINN\n(Đề Xuất)", "CNN Baseline\n(NoPINN)", "Multitask MLP\nPINN (1D)"]
    clf_accs   = [77.3, 95.0, 30.8]
    clf_errs   = [11.9, 5.3, 10.0]
    clf_colors = [C_CNN_PINN, C_CNN_BASE, C_MLP_MULTI]

    bars = ax.bar(range(3), clf_accs, yerr=clf_errs, capsize=4, color=clf_colors, edgecolor="black", lw=0.8, width=0.48)
    ax.axhline(20.0, color="#94a3b8", ls=":", lw=1.2, label="Đoán Mò Ngẫu Nhiên (1/5 = 20%)")
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(clf_models, fontweight="bold")
    ax.set_ylabel("Accuracy Phân Loại (%)", fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_title("(a) Khả Năng Phân Loại Hình Dạng: CNN vs MLP", pad=8)
    ax.grid(axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=7.5)

    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 13, f"{b.get_height():.1f}%", ha="center", fontsize=8.0, fontweight="bold")

    # Footnote about Xiong
    ax.text(0.98, 0.88, "*Xiong et al.: Thuần hồi quy W, L, D\n(Không có nhánh phân loại)", 
            transform=ax.transAxes, ha="right", fontsize=7.0, fontstyle="italic", 
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8))

    # (b) Crack Length NMAE (%) Error - LOG SCALE (Mean ± Std)
    ax = axs[0, 1]
    all_models = ["CNN PINN\n(Đề Xuất)", "CNN Baseline\n(NoPINN)", "Multitask MLP\n(1D PINN)", "Xiong et al.\nMLP (1D)"]
    nmae_l_means = [7.40, 10.91, 61.09, 15469.5]
    nmae_l_stds  = [2.67, 2.31, 35.52, 14745.9]
    all_colors   = [C_CNN_PINN, C_CNN_BASE, C_MLP_MULTI, C_MLP_XIONG]

    b_l = ax.bar(range(4), nmae_l_means, yerr=[[0, 0, 0, 0], [2.67, 2.31, 35.52, 14745.9]], 
                 capsize=4, color=all_colors, edgecolor="black", lw=0.8, width=0.52)
    ax.set_yscale("log")
    ax.set_ylim(1.0, 60000)
    ax.axhline(5.0, color="#2e7d32", ls="--", lw=1.2, label="Chuẩn NDT Nghiêm Ngặt (< 5.0% / 0.5mm)")
    ax.axhline(10.0, color="#d32f2f", ls="-.", lw=1.2, label="Ngưỡng Cho Phép NDT (< 10.0% / 1.0mm)")

    ax.set_xticks(range(4))
    ax.set_xticklabels(all_models, fontweight="bold", fontsize=8.0)
    ax.set_ylabel("NMAE Chiều Dài $L$ (%) [Thang Log]", fontweight="bold")
    ax.set_title("(b) Sai Số Chiều Dài Chuẩn Hóa NMAE ($L$): Trung Bình $\\pm$ Std", pad=8)
    ax.grid(axis="y", which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=7.2)

    # Values above bars (NMAE% and equivalent MAE mm)
    ax.text(0, 7.40 * 1.5, "7.40% ± 2.7%\n(0.739 mm)", ha="center", fontsize=7.0, fontweight="bold", color=C_CNN_PINN)
    ax.text(1, 10.91 * 1.5, "10.91% ± 2.3%\n(1.091 mm - Vượt)", ha="center", fontsize=7.0, fontweight="bold", color=C_CNN_BASE)
    ax.text(2, 61.09 * 1.4, "61.09% ± 35.5%\n(6.110 mm)", ha="center", fontsize=7.0, fontweight="bold", color=C_MLP_MULTI)
    ax.text(3, 15469.5 * 1.25, "> 15,000%\n(1546.9 mm)", ha="center", fontsize=7.0, fontweight="bold", color="#b91c1c")

    # (c) All 3 Dimensions: Width (W), Length (L), Depth (D) Comparison
    ax = axs[1, 0]
    dims = ["Chiều Rộng ($W$)", "Chiều Dài ($L$)", "Độ Sâu ($D$)", "Tổng Hợp (Overall)"]
    x = np.arange(len(dims))
    w = 0.20

    cnn_wld   = [0.102, 0.739, 0.659, 0.500]
    multi_wld = [0.363, 6.110, 2.377, 2.950]
    xiong_wld = [69.81, 1546.9, 248.5, 621.8]

    ax.bar(x - w, cnn_wld, w, color=C_CNN_PINN, edgecolor="black", lw=0.8, label="CNN PINN (2D Conv)")
    ax.bar(x, multi_wld, w, color=C_MLP_MULTI, edgecolor="black", lw=0.8, label="Multitask MLP (1D)")
    ax.bar(x + w, xiong_wld, w, color=C_MLP_XIONG, edgecolor="black", lw=0.8, label="Xiong et al. MLP (1D)")

    ax.set_yscale("log")
    ax.set_ylim(0.04, 3500)
    ax.axhline(0.50, color="#2e7d32", ls="--", lw=1.0)
    ax.axhline(1.00, color="#d32f2f", ls="-.", lw=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=8.0)
    ax.set_ylabel("Sai Số MAE (mm) [Thang Log]", fontweight="bold")
    ax.set_title("(c) Phân Rã Toàn Diện Sai Số Ba Chiều ($W, L, D$)", pad=8)
    ax.grid(axis="y", which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=7.5)

    # (d) Accuracy Scaling Trajectory: CNN NoPINN vs CNN PINN vs Multitask MLP
    ax = axs[1, 1]
    scales = ["1%", "3%", "5%", "7%", "10%"]
    xs = np.arange(len(scales))

    # Real experimental trajectories for 1% -> 10% under PI-LGL
    acc_cnn_nopinn = [100.0, 100.0, 97.5, 90.0, 90.0]  # CNN Baseline NoPINN (Học vẹt 90-100%)
    acc_cnn_pinn   = [80.0,  90.0,  84.3, 80.0, 60.0]  # CNN PINN (Vững chắc 80-90%)
    acc_mlp_multi  = [30.0,  30.0,  30.0, 30.0, 40.0]  # Multitask MLP (Kẹt cứng trần 30-40%)

    ax.plot(xs, acc_cnn_nopinn, "s-", color=C_CNN_BASE, lw=2.2, ms=6, label="CNN Baseline (NoPINN - Học vẹt)", zorder=5)
    ax.plot(xs, acc_cnn_pinn, "o-", color=C_CNN_PINN, lw=2.4, ms=7, label="CNN PINN (PI-LGL Đề Xuất)", zorder=6)
    ax.plot(xs, acc_mlp_multi, "^--", color=C_MLP_MULTI, lw=1.8, ms=7, label="Multitask MLP (Kẹt cứng trần 30%)", zorder=4)
    ax.axhline(20.0, color="#94a3b8", ls=":", lw=1.2, label="Ngưỡng Đoán Mò (20%)")

    ax.set_xticks(xs)
    ax.set_xticklabels(scales, fontweight="bold")
    ax.set_ylabel("Accuracy Phân Loại (%)", fontweight="bold")
    ax.set_ylim(10, 115)
    ax.set_title("(d) Quỹ Đạo Phân Loại Dự Đoán (1% → 10% Dữ Liệu)", pad=8)
    ax.grid(axis="y")
    ax.legend(loc="lower left", frameon=True, fontsize=7.2)

    # Annotate NoPINN shortcut vs MLP ceiling
    ax.text(xs[1], 103, "NoPINN: 100% (Học vẹt biên độ)", color=C_CNN_BASE, fontsize=7.2, fontweight="bold", ha="center")
    ax.annotate("MLP kẹt cứng tại trần ~30%\n(Do phẳng hóa 1D làm mất tô-pô 2D)", xy=(xs[2], 30.0), xytext=(xs[1], 48.0),
                arrowprops=dict(facecolor=C_MLP_MULTI, shrink=0.08, width=1.0, headwidth=4),
                fontsize=7.0, fontweight="bold", color=C_MLP_MULTI)

    fig.suptitle("KHẢO SÁT CÁC MÔ HÌNH DƯỚI PHƯƠNG PHÁP THÍCH ỨNG: CNN (PINN & NoPINN) VS MLP\nĐịnh Lượng Sai Số Chuẩn Hóa NMAE Chiều Dài và Quỹ Đạo Phân Loại Khi Dự Đoán", 
                 fontsize=11.2, fontweight="bold", y=0.995)

    out_file = os.path.join(OUTPUT_DIR, "fig_mlp_vs_cnn_adaptation_comparison.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved: {out_file}")

# ------------------------------------------------------------------------------
# 2. FIGURE 2: Direct 3D Error Gap (W, L, D) Comparing CNN vs 2 MLP Architectures
# ------------------------------------------------------------------------------
def plot_fig2_mlp_error_gap():
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)

    models = ["CNN PINN\n(Đề Xuất)", "Multitask MLP\nPINN (1D)", "Xiong et al.\nMLP PINN (1D)"]
    x = np.arange(len(models))
    w = 0.22

    w_vals = [0.102, 0.363, 69.81]
    l_vals = [0.739, 6.110, 1546.9]
    d_vals = [0.659, 2.377, 248.5]

    b1 = ax.bar(x - w, w_vals, w, color="#3b82f6", edgecolor="black", lw=0.8, label="Chiều Rộng $W$")
    b2 = ax.bar(x, l_vals, w, color="#ef4444", edgecolor="black", lw=0.8, label="Chiều Dài $L$ (Then chốt)")
    b3 = ax.bar(x + w, d_vals, w, color="#10b981", edgecolor="black", lw=0.8, label="Độ Sâu $D$")

    ax.set_yscale("log")
    ax.set_ylim(0.03, 3500)
    ax.axhline(0.50, color="#2e7d32", ls="--", lw=1.2, label="Chuẩn NDT (< 0.50 mm)")
    ax.axhline(1.00, color="#dc2626", ls="-.", lw=1.2, label="Giới Hạn NDT (< 1.00 mm)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel("Sai Số MAE Kích Thước (mm) [Thang Log]", fontweight="bold")
    ax.set_title("So Sánh Mức Độ Sai Số Kích Thước 3 Chiều ($W, L, D$)\nCNN PINN Kiểm Soát Hoàn Hảo, MLP Bị Phân Kỳ Số Học", pad=12)
    ax.grid(axis="y", which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Annotate Length values
    ax.text(x[0], l_vals[0]*1.3, f"{l_vals[0]:.3f} mm", ha="center", fontsize=7.5, fontweight="bold", color="#ef4444")
    ax.text(x[1], l_vals[1]*1.3, f"{l_vals[1]:.2f} mm", ha="center", fontsize=7.5, fontweight="bold", color="#ef4444")
    ax.text(x[2], l_vals[2]*1.25, f"{l_vals[2]:.1f} mm", ha="center", fontsize=7.5, fontweight="bold", color="#ef4444")

    out_file = os.path.join(OUTPUT_DIR, "fig_mlp_wld_error_gap.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved: {out_file}")

# ------------------------------------------------------------------------------
# 3. FIGURE 3: Why MLP Fails (Spatial Topology vs Flattening 1D)
# ------------------------------------------------------------------------------
def plot_fig3_topology_explanation():
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # Panel (a): CNN 2D Conv Preserves Spatial Neighbor & Maxwell Gradient
    ax = axs[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(a) CNN (ImprovedMultimodelNet):\nBảo Toàn Tô-Pô Không Gian 2D", fontsize=10.0, fontweight="bold")

    # Draw a 2D grid patch
    for r in range(4):
        for c in range(4):
            rect = plt.Rectangle((1.5 + c*1.6, 2.5 + r*1.6), 1.3, 1.3, facecolor="#e0f2fe", edgecolor="#0284c7", lw=1.2)
            ax.add_patch(rect)
    
    # Kernel overlay
    kernel = plt.Rectangle((3.1, 4.1), 2.9, 2.9, facecolor="#00897B", alpha=0.35, edgecolor="#004D40", lw=2.0, ls="--")
    ax.add_patch(kernel)
    ax.text(4.55, 5.55, "Bộ Lọc Conv2D 3x3\n($\\nabla^2 H = \\partial^2 V / \\partial x^2 + \\partial^2 V / \\partial y^2$)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#004D40")
    ax.text(4.55, 1.2, "(+) Bảo toàn liên kết lân cận 2D\n(+) Nhận biết cặp lưỡng cực vi sai\n(+) LoRA thích ứng biên cực kỳ hiệu quả", ha="center", fontsize=8.0, color="#047857", fontweight="bold")

    # Panel (b): MLP 1D Flattening Destroys Neighbors
    ax = axs[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(b) MLP (Multitask & Xiong et al.):\nPhẳng Hóa 1D Phá Hủy Hoàn Toàn Tô-Pô Lân Cận", fontsize=10.0, fontweight="bold")

    # Draw flattened vector 1D
    for i in range(12):
        col = "#fef3c7" if i % 2 == 0 else "#fee2e2"
        ec  = "#b45309" if i % 2 == 0 else "#b91c1c"
        rect = plt.Rectangle((0.6 + i*0.72, 5.0), 0.6, 2.2, facecolor=col, edgecolor=ec, lw=1.0)
        ax.add_patch(rect)
        ax.text(0.9 + i*0.72, 6.1, f"x{i+1}", ha="center", va="center", fontsize=6.5)

    # Disconnected arrows
    ax.annotate("Mất hoàn toàn khoảng cách 2D!", xy=(5.0, 4.5), xytext=(5.0, 3.2),
                arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.2, headwidth=5),
                fontsize=8.0, fontweight="bold", color="#b91c1c", ha="center")
    
    ax.text(5.0, 1.2, "(-) 1089 điểm đo bị xem như biến độc lập\n(-) Triệt tiêu đạo hàm không gian Maxwell\n(-) Cổng Laplacian & LoRA không thể hoạt động", ha="center", fontsize=8.0, color="#b91c1c", fontweight="bold")

    out_file = os.path.join(OUTPUT_DIR, "fig_mlp_vs_cnn_topology_mechanism.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved: {out_file}")

# ------------------------------------------------------------------------------
# 4. FIGURE 4: Dedicated NMAE Length Error (Mean ± Std)
# ------------------------------------------------------------------------------
def plot_fig4_nmae_l_dedicated():
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

    models = ["CNN PINN\n(PI-LGL Đề Xuất)", "CNN Baseline\n(NoPINN)", "Multitask MLP\nPINN (1D)", "Xiong et al.\nMLP PINN (1D)"]
    nmae_means = [7.40, 10.91, 61.09, 15469.5]
    nmae_stds  = [2.67, 2.31, 35.52, 14745.9]
    colors     = [C_CNN_PINN, C_CNN_BASE, C_MLP_MULTI, C_MLP_XIONG]

    # Bar chart with error bars
    bars = ax.bar(range(4), nmae_means, yerr=[[0, 0, 0, 0], nmae_stds], capsize=5, 
                  color=colors, edgecolor="black", lw=0.8, width=0.50)

    ax.set_yscale("log")
    ax.set_ylim(1.5, 60000)

    # NDT Thresholds
    ax.axhline(5.0, color="#2e7d32", ls="--", lw=1.3, label="Chuẩn NDT Nghiêm Ngặt (< 5.0% / 0.5 mm)")
    ax.axhline(10.0, color="#d32f2f", ls="-.", lw=1.3, label="Ngưỡng Trần Cho Phép NDT (< 10.0% / 1.0 mm)")

    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontweight="bold", fontsize=8.5)
    ax.set_ylabel("Sai Số Chuẩn Hóa Chiều Dài NMAE ($L$) [%] (Thang Log)", fontweight="bold")
    ax.set_title("ĐỐI SÁNH SAI SỐ CHUẨN HÓA CHIỀU DÀI NMAE ($L$): TRUNG BÌNH ± ĐỘ LỆCH CHUẨN\nCNN PINN Đạt Chuẩn NDT, NoPINN Vượt Ngưỡng, MLP Phân Kỳ", pad=12)
    ax.grid(axis="y", which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Annotate values
    ax.text(0, 7.40 * 1.55, "7.40% ± 2.67%\n(0.739 mm - Đạt NDT)", ha="center", fontsize=7.5, fontweight="bold", color=C_CNN_PINN)
    ax.text(1, 10.91 * 1.55, "10.91% ± 2.31%\n(1.091 mm - VƯỢT TRẦN)", ha="center", fontsize=7.5, fontweight="bold", color=C_CNN_BASE)
    ax.text(2, 61.09 * 1.45, "61.09% ± 35.52%\n(6.110 mm - Thất bại)", ha="center", fontsize=7.5, fontweight="bold", color=C_MLP_MULTI)
    ax.text(3, 15469.5 * 1.25, "> 15,000%\n(1546.9 mm - Phân kỳ)", ha="center", fontsize=7.5, fontweight="bold", color="#b91c1c")

    out_file = os.path.join(OUTPUT_DIR, "fig_nmae_length_models_comparison.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved: {out_file}")

# ------------------------------------------------------------------------------
# 5. FIGURE 5: Dedicated Full Trajectory (CNN NoPINN vs CNN PINN vs Multitask MLP)
# ------------------------------------------------------------------------------
def plot_fig5_trajectory_dedicated():
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)

    scales = ["1%", "3%", "5%", "7%", "10%"]
    xs = np.arange(len(scales))

    # Accuracy trajectories
    acc_nopinn = [100.0, 100.0, 97.5, 90.0, 90.0]
    acc_pinn   = [80.0,  90.0,  84.3, 80.0, 60.0]
    acc_mlp    = [30.0,  30.0,  30.0, 30.0, 40.0]

    # Shaded zones
    ax.axhspan(88, 103, color="#fef3c7", alpha=0.45, label="Vùng Học Vẹt Biên Độ (NoPINN)")
    ax.axhspan(60, 92,  color="#dcfce7", alpha=0.35, label="Vùng Ràng Buộc Vật Lý Ổn Định (PINN)")
    ax.axhspan(15, 45,  color="#fee2e2", alpha=0.35, label="Vùng Sụp Đổ Tô-Pô 1D (MLP)")

    ax.plot(xs, acc_nopinn, "s-", color=C_CNN_BASE, lw=2.4, ms=7, label="CNN Baseline (NoPINN) - Học vẹt 90%–100%", zorder=5)
    ax.plot(xs, acc_pinn,   "o-", color=C_CNN_PINN, lw=2.6, ms=8, label="CNN PINN (PI-LGL) - Ràng buộc Maxwell", zorder=6)
    ax.plot(xs, acc_mlp,    "^--", color=C_MLP_MULTI, lw=2.0, ms=7, label="Multitask MLP (1D) - Kẹt trần 30%–40%", zorder=4)

    ax.axhline(20.0, color="#64748b", ls=":", lw=1.2, label="Đoán Mò Ngẫu Nhiên (1/5 = 20%)")

    ax.set_xticks(xs)
    ax.set_xticklabels(scales, fontweight="bold", fontsize=9.0)
    ax.set_xlabel("Tỷ Lệ Dữ Liệu Mô Phỏng Dùng Huấn Luyện", fontweight="bold")
    ax.set_ylabel("Accuracy Phân Loại (%)", fontweight="bold")
    ax.set_ylim(10, 112)
    ax.set_title("QUỸ ĐẠO PHÂN LOẠI KHI DỰ ĐOÁN QUA CÁC MỨC DỮ LIỆU (1% → 10%)\nĐối So Sánh Đầy Đủ: CNN NoPINN vs CNN PINN vs Multitask MLP", pad=12)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, fontsize=7.8)

    # Annotate points
    for i, (p, np_val, m) in enumerate(zip(acc_pinn, acc_nopinn, acc_mlp)):
        ax.text(i, np_val + 2.0, f"{np_val:.1f}%", ha="center", fontsize=7.2, fontweight="bold", color=C_CNN_BASE)
        ax.text(i, p - 3.8, f"{p:.1f}%", ha="center", fontsize=7.2, fontweight="bold", color=C_CNN_PINN)
        ax.text(i, m + 2.2, f"{m:.0f}%", ha="center", fontsize=7.0, fontweight="bold", color=C_MLP_MULTI)

    out_file = os.path.join(OUTPUT_DIR, "fig_classification_accuracy_trajectory_3models.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved: {out_file}")

if __name__ == "__main__":
    plot_fig1_mlp_vs_cnn()
    plot_fig2_mlp_error_gap()
    plot_fig3_topology_explanation()
    plot_fig4_nmae_l_dedicated()
    plot_fig5_trajectory_dedicated()
    print(f"[ALL DONE] Generated 5 figures in {OUTPUT_DIR}")
