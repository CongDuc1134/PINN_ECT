# -*- coding: utf-8 -*-
"""
===============================================================================
IEEE TRANSACTIONS PLOTTING UTILITIES FOR SIM-TO-REAL ECT
===============================================================================
Quy chuẩn đồ thị khoa học xuất bản chuẩn IEEE Transactions (Q1/Q2):
1. Font chữ: Times New Roman (hoặc hệ serif tương đương), kích thước chuẩn mực.
2. Xuất bản 2 định dạng: 300 DPI PNG và vector PDF (dùng trực tiếp cho LaTeX).
3. TẤT CẢ CHÚ THÍCH (LEGENDS) ĐỀU ĐƯỢC BỐ TRÍ Ở BÊN NGOÀI KHUNG BIỂU ĐỒ
   (sử dụng bbox_to_anchor và bbox_inches='tight' để tránh đè lấp dữ liệu).
4. Vẽ riêng biệt các đường Loss khi Finetuning cho từng phương pháp.
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# BẢNG MÀU ĐỒNG BỘ NHẤT QUÁN CHUẨN IEEE TRANSACTIONS (Q1/Q2)
# =============================================================================
# 1. Hai giao thức thử nghiệm (Protocol 1 vs Protocol 2):
COLOR_PROTOCOL_1 = "#1f77b4"  # Royal Blue (Cách 1: Scan Split)
COLOR_PROTOCOL_2 = "#e6550d"  # Amber Orange (Cách 2: 10-Fold LODO)

# 2. Các chỉ số phân loại và kích thước (Đồng bộ xuyên suốt):
COLOR_ACCURACY   = "#1f77b4"  # Royal Blue: Classification Accuracy (%)
COLOR_NMAE       = "#7b4173"  # Deep Plum / Purple: Overall NMAE (%)
COLOR_WIDTH_W    = "#2ca02c"  # Emerald Green: Width W (mm)
COLOR_LENGTH_L   = "#ff7f0e"  # Vivid Orange: Length L (mm)
COLOR_DEPTH_D    = "#d62728"  # Crimson Red: Depth D (mm)

# 3. Bảng màu đại diện cố định cho từng mô hình khi so sánh giữa các mô hình:
COLOR_MODEL_MAP = {
    "CNN_Proposed": "#d62728",  # Crimson Red (Mô hình Đề xuất - PINN)
    "CNN_NoPINN":   "#1f77b4",  # Steel Blue (Baseline CNN không PINN)
    "MLP":          "#2ca02c",  # Forest Green (Baseline MLP)
    "XIONG":        "#9467bd",  # Violet Purple (Baseline XIONG)
}
DEFAULT_MODEL_COLOR = "#7f7f7f"


def setup_ieee_style():
    """
    Cấu hình toàn bộ tham số Matplotlib theo quy chuẩn chuẩn của IEEE Transactions.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.fontsize": 9.0,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "black",
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save_ieee_figure(fig: plt.Figure, output_base_path: str):
    """
    Lưu đồ thị thành cả 2 định dạng: PNG 300 DPI và vector PDF.
    """
    base_no_ext = os.path.splitext(output_base_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base_no_ext)), exist_ok=True)
    png_path = f"{base_no_ext}.png"
    pdf_path = f"{base_no_ext}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    [IEEE Plot] Đã lưu: {os.path.basename(png_path)} và {os.path.basename(pdf_path)}")


def plot_separate_loss_curve_single_model(
    history: dict,
    model_name: str,
    output_base_path: str,
    title_suffix: str = "",
):
    """
    VẼ RIÊNG BIỆT ĐƯỜNG LOSS KHI FINETUNING CHO MỘT MÔ HÌNH:
    - Bố cục 2 panel theo chiều ngang (Tổng Loss và Phân rã Loss / Learning Rate).
    - Toàn bộ chú thích (Legend) được đặt Ở BÊN NGOÀI KHUNG BIỂU ĐỒ.
    """
    setup_ieee_style()
    epochs = history.get("epoch", [])
    if not epochs:
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_clf = history.get("train_loss_clf", [])
    train_reg = history.get("train_loss_reg", [])
    val_clf = history.get("val_loss_clf", [])
    val_reg = history.get("val_loss_reg", [])
    best_ep = history.get("best_epoch", None)
    stopped_at = history.get("early_stopped_at", None)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

    # Panel 1: Tổng Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, color="#1f77b4", lw=1.6, label="Train Total Loss")
    if val_loss and any(not np.isnan(v) for v in val_loss):
        ax1.plot(epochs, val_loss, color="#d62728", lw=1.6, ls="--", label="Val Total Loss")
    if best_ep and best_ep <= len(epochs):
        ax1.axvline(best_ep, color="green", ls=":", lw=1.4, label=f"Best Ep ({best_ep})")
    if stopped_at and stopped_at <= len(epochs):
        ax1.axvline(stopped_at, color="purple", ls="-.", lw=1.4, label=f"Early Stop ({stopped_at})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title(f"(a) Total Loss Convergence - {model_name}", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.0)

    # Panel 2: Phân rã Loss (Clf vs Reg)
    ax2 = axes[1]
    has_decomp = False
    if train_clf and any(c > 0 for c in train_clf):
        ax2.plot(epochs, train_clf, color="#ff7f0e", lw=1.4, label="Train Clf (CE)")
        has_decomp = True
    if train_reg and any(r > 0 for r in train_reg):
        ax2.plot(epochs, train_reg, color="#2ca02c", lw=1.4, label="Train Reg (MSE)")
        has_decomp = True
    if val_clf and any(c > 0 for c in val_clf):
        ax2.plot(epochs, val_clf, color="#e377c2", lw=1.4, ls="--", label="Val Clf (CE)")
        has_decomp = True
    if val_reg and any(r > 0 for r in val_reg):
        ax2.plot(epochs, val_reg, color="#17becf", lw=1.4, ls="--", label="Val Reg (MSE)")
        has_decomp = True

    if not has_decomp:
        # Nếu mô hình không có phân rã, vẽ Learning Rate curve
        lr_hist = history.get("lr", [])
        if lr_hist:
            ax2.plot(epochs, lr_hist, color="#8c564b", lw=1.5, label="Cosine LR")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title(f"(b) Learning Rate Schedule - {model_name}", pad=8)
    else:
        ax2.set_ylabel("Decomposed Loss")
        ax2.set_title(f"(b) Task Decomposition - {model_name}", pad=8)

    ax2.set_xlabel("Epoch")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.0)

    plt.tight_layout()
    save_ieee_figure(fig, output_base_path)


def plot_all_methods_loss_grid(
    histories_dict: dict,
    output_base_path: str,
    title: str = "Training & Validation Loss Convergence across Adaptation Methods",
):
    """
    Vẽ bảng ghép so sánh đường Loss của tất cả các phương pháp.
    Mỗi phương pháp 1 subplot riêng biệt, chú thích nằm toàn bộ BÊN NGOÀI.
    """
    setup_ieee_style()
    method_keys = list(histories_dict.keys())
    n_methods = len(method_keys)
    if n_methods == 0:
        return

    cols = 2
    rows = (n_methods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.6 * rows), dpi=300, squeeze=False)

    for idx, key in enumerate(method_keys):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        hist = histories_dict[key]
        epochs = hist.get("epoch", [])
        train_l = hist.get("train_loss", [])
        val_l = hist.get("val_loss", [])
        best_ep = hist.get("best_epoch", None)

        if epochs:
            ax.plot(epochs, train_l, color="#1f77b4", lw=1.5, label="Train Loss")
            if val_l and any(not np.isnan(v) for v in val_l):
                ax.plot(epochs, val_l, color="#d62728", lw=1.5, ls="--", label="Val Loss")
            if best_ep and best_ep <= len(epochs):
                ax.axvline(best_ep, color="green", ls=":", lw=1.2, label=f"Best ({best_ep})")

        ax.set_title(f"{key}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=":", alpha=0.6)
        # CHÚ THÍCH RA NGOÀI BIỂU ĐỒ
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, borderaxespad=0.0)

    # Ẩn các ô trống nếu có
    for idx in range(n_methods, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    save_ieee_figure(fig, output_base_path)


def plot_confusion_matrix_ieee(
    confusion_matrix: np.ndarray,
    class_names: list,
    model_name: str,
    output_base_path: str,
    acc_val: float = None,
):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) chuẩn IEEE với Colorbar và chú thích bên ngoài.
    """
    setup_ieee_style()
    cm = np.asarray(confusion_matrix)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(4.0, 3.4), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # Colorbar đặt bên phải ngoài rìa
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    # Hiển thị số lượng mẫu trong ô
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    acc_str = f" (ACC = {acc_val:.1f}%)" if acc_val is not None else ""
    ax.set_title(f"Confusion Matrix: {model_name}{acc_str}", pad=10)
    ax.set_ylabel("True Defect Class")
    ax.set_xlabel("Predicted Defect Class")

    plt.tight_layout()
    save_ieee_figure(fig, output_base_path)


def plot_benchmark_comparison_ieee(
    df_results: pd.DataFrame,
    output_base_path: str,
    title_suffix: str = "",
):
    """
    Biểu đồ đánh giá kết quả FINETUNED thuần túy của các mô hình (KHÔNG VẼ ZEROSHOT):
    Bố cục 3 Panel chuẩn IEEE:
    - Panel (a): Accuracy (%) phân loại của các mô hình sau khi finetuned.
    - Panel (b): Sai số kích thước chuẩn hóa tổng thể NMAE (%) sau khi finetuned.
    - Panel (c): Sai số kích thước MAE (mm) từng chiều (W, L, D) sau khi finetuned.
    Toàn bộ Chú thích (Legend) đặt hoàn toàn RA NGOÀI KHUNG BIỂU ĐỒ.
    Màu sắc cố định, đồng bộ nhất quán chuẩn xuất bản Q1/Q2.
    """
    setup_ieee_style()
    if df_results.empty:
        return

    # Lọc chỉ lấy kết quả Finetuned nếu có cột Stage (loại bỏ hoàn toàn Zero-shot / Pretrained)
    if "Stage" in df_results.columns:
        df_plot = df_results[df_results["Stage"] == "Finetuned"].copy()
    else:
        df_plot = df_results.copy()

    if df_plot.empty:
        df_plot = df_results.copy()

    models = df_plot["Model"].unique()
    x = np.arange(len(models))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300)

    # 1. Panel Phân loại (Accuracy của mô hình sau Finetune)
    has_acc = "ACC(%)" in df_plot.columns
    if has_acc:
        accs = [
            df_plot[df_plot["Model"] == m]["ACC(%)"].values[0]
            if not df_plot[df_plot["Model"] == m].empty and not np.isnan(df_plot[df_plot["Model"] == m]["ACC(%)"].values[0])
            else 0.0
            for m in models
        ]
        bars1 = ax1.bar(x, accs, width=0.45, color=COLOR_ACCURACY, edgecolor="black", lw=0.8)
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(f"(a) Classification Accuracy {title_suffix}".strip(), fontsize=9.5, pad=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
        ax1.set_ylim(0, 115)
        ax1.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 2. Panel Hồi quy Tổng thể NMAE (%)
    has_nmae = "NMAE_Overall(%)" in df_plot.columns
    if has_nmae:
        nmaes = [
            df_plot[df_plot["Model"] == m]["NMAE_Overall(%)"].values[0]
            if not df_plot[df_plot["Model"] == m].empty and not np.isnan(df_plot[df_plot["Model"] == m]["NMAE_Overall(%)"].values[0])
            else 0.0
            for m in models
        ]
        bars2 = ax2.bar(x, nmaes, width=0.45, color=COLOR_NMAE, edgecolor="black", lw=0.8)
        max_nmae = max(nmaes) if nmaes else 10.0
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., h + max_nmae*0.02, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

        ax2.set_ylabel("Overall NMAE (%)")
        ax2.set_title(f"(b) Overall Dimension NMAE {title_suffix}".strip(), fontsize=9.5, pad=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
        ax2.set_ylim(0, max_nmae * 1.25 if max_nmae > 0 else 10)
        ax2.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 3. Panel Hồi quy Từng chiều (MAE W, L, D tính bằng mm)
    w_vals = [df_plot[df_plot["Model"] == m]["MAE_W(mm)"].values[-1] for m in models]
    l_vals = [df_plot[df_plot["Model"] == m]["MAE_L(mm)"].values[-1] for m in models]
    d_vals = [df_plot[df_plot["Model"] == m]["MAE_D(mm)"].values[-1] for m in models]

    w_bar = 0.25
    ax3.bar(x - w_bar, w_vals, w_bar, label="Width (W)", color=COLOR_WIDTH_W, edgecolor="black", lw=0.8)
    ax3.bar(x, l_vals, w_bar, label="Length (L)", color=COLOR_LENGTH_L, edgecolor="black", lw=0.8)
    ax3.bar(x + w_bar, d_vals, w_bar, label="Depth (D)", color=COLOR_DEPTH_D, edgecolor="black", lw=0.8)

    ax3.set_ylabel("MAE (mm)")
    ax3.set_title(f"(c) Dimension MAE (W, L, D) {title_suffix}".strip(), fontsize=9.5, pad=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax3.grid(True, linestyle=":", axis="y", alpha=0.6)
    # Legend đặt trong khoảng trắng phía trên bên trái (tránh hoàn toàn va chạm Title)
    ax3.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    plt.tight_layout()
    save_ieee_figure(fig, output_base_path)


def plot_dimensional_mae_breakdown_ieee(
    df_results: pd.DataFrame,
    output_base_path: str,
    title_suffix: str = "",
):
    """
    VẼ RIÊNG BIỆT MAE (mm) CHO TỪNG CHIỀU W, L, D TRÊN CÁC TRỤC TỌA ĐỘ ĐỘC LẬP:
    - Giải quyết triệt để vấn đề chênh lệch bậc độ lớn giữa Chiều dài (L > 10 mm)
      với Chiều rộng (W < 1 mm) và Chiều sâu (D 1-5 mm).
    - Mỗi chiều có một Panel với trục tung tối ưu riêng để nhìn rõ sai số vật lý.
    - Không cần Legend đơn lẻ gây đè lấp Title.
    """
    setup_ieee_style()
    if df_results.empty:
        return

    if "Stage" in df_results.columns:
        df_plot = df_results[df_results["Stage"] == "Finetuned"].copy()
    else:
        df_plot = df_results.copy()

    if df_plot.empty:
        df_plot = df_results.copy()

    models = df_plot["Model"].unique()
    x = np.arange(len(models))

    fig, (ax_w, ax_l, ax_d) = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300)

    # 1. Width (W) MAE (mm)
    w_vals = [df_plot[df_plot["Model"] == m]["MAE_W(mm)"].values[-1] for m in models]
    bars_w = ax_w.bar(x, w_vals, width=0.45, color=COLOR_WIDTH_W, edgecolor="black", lw=0.8)
    max_w = max(w_vals) if w_vals else 0.5
    for bar in bars_w:
        h = bar.get_height()
        if h > 0:
            ax_w.text(bar.get_x() + bar.get_width()/2., h + max_w*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_w.set_ylabel("Width MAE (mm)")
    ax_w.set_title(f"(a) Width (W) Error {title_suffix}".strip(), fontsize=9.5, pad=8)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_w.set_ylim(0, max_w * 1.25 if max_w > 0 else 0.5)
    ax_w.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 2. Length (L) MAE (mm)
    l_vals = [df_plot[df_plot["Model"] == m]["MAE_L(mm)"].values[-1] for m in models]
    bars_l = ax_l.bar(x, l_vals, width=0.45, color=COLOR_LENGTH_L, edgecolor="black", lw=0.8)
    max_l = max(l_vals) if l_vals else 5.0
    for bar in bars_l:
        h = bar.get_height()
        if h > 0:
            ax_l.text(bar.get_x() + bar.get_width()/2., h + max_l*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_l.set_ylabel("Length MAE (mm)")
    ax_l.set_title(f"(b) Length (L) Error {title_suffix}".strip(), fontsize=9.5, pad=8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_l.set_ylim(0, max_l * 1.25 if max_l > 0 else 5.0)
    ax_l.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 3. Depth (D) MAE (mm)
    d_vals = [df_plot[df_plot["Model"] == m]["MAE_D(mm)"].values[-1] for m in models]
    bars_d = ax_d.bar(x, d_vals, width=0.45, color=COLOR_DEPTH_D, edgecolor="black", lw=0.8)
    max_d = max(d_vals) if d_vals else 1.0
    for bar in bars_d:
        h = bar.get_height()
        if h > 0:
            ax_d.text(bar.get_x() + bar.get_width()/2., h + max_d*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_d.set_ylabel("Depth MAE (mm)")
    ax_d.set_title(f"(c) Depth (D) Error {title_suffix}".strip(), fontsize=9.5, pad=8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_d.set_ylim(0, max_d * 1.25 if max_d > 0 else 1.0)
    ax_d.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout()
    save_ieee_figure(fig, output_base_path)


def plot_finetuned_two_protocols_comparison(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_base_path: str,
):
    """
    SO SÁNH TRỰC DIỆN KẾT QUẢ FINETUNED GIỮA 2 CÁCH CHIA DỮ LIỆU (KHÔNG DÍNH DÁNG ĐẾN ZEROSHOT):
    - Cách 1 (Scan Split): Train Scan 1 -> Test Scan 2 (10 mẫu).
    - Cách 2 (10-Fold LODO): Leave-One-Defect-Out Cross-Validation (20 mẫu out-of-fold).
    
    Quy chuẩn hiển thị chuẩn IEEE:
    - SỬ DỤNG DUY NHẤT 1 FIGURE LEGEND TẬP TRUNG Ở ĐỈNH BIỂU ĐỒ (Hoàn toàn không đè lấp Title).
    - Tổng thể dùng NMAE (%) (chuẩn hóa tương đối, tránh Length lấn át).
    - Các chiều vật lý W, L, D vẽ riêng biệt MAE (mm).
    """
    import matplotlib.patches as mpatches
    setup_ieee_style()
    if df_p1.empty or df_p2.empty:
        return

    # Lấy riêng hàng Finetuned
    p1 = df_p1[df_p1["Stage"] == "Finetuned"].copy() if "Stage" in df_p1.columns else df_p1.copy()
    p2 = df_p2.copy()

    models = [m for m in p1["Model"].unique() if m in p2["Model"].unique()]
    if not models:
        return

    x = np.arange(len(models))
    width = 0.35

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_PROTOCOL_1, edgecolor="black", label="Protocol 1: Scan Split (Repeatability)"),
        mpatches.Patch(facecolor=COLOR_PROTOCOL_2, edgecolor="black", label="Protocol 2: 10-Fold LODO (Generalization)"),
    ]

    # ---------------------------------------------------------
    # HÌNH 1: BẢNG TOÀN DIỆN 2 HÀNG (ACC, NMAE VÀ MAE W, L, D)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 5.5), dpi=300)
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=0.6, top=0.86, bottom=0.12)

    # Unified Legend trên đỉnh Figure (không đè vào bất kỳ Subplot nào)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
    )

    # Hàng 1: Accuracy (%) và NMAE Tổng thể (%)
    ax_acc = fig.add_subplot(gs[0, :3])
    ax_nmae = fig.add_subplot(gs[0, 3:])

    # Hàng 2: MAE riêng cho từng chiều W (mm), L (mm), D (mm)
    ax_w = fig.add_subplot(gs[1, :2])
    ax_l = fig.add_subplot(gs[1, 2:4])
    ax_d = fig.add_subplot(gs[1, 4:])

    # 1. Panel (a): Accuracy (%)
    acc_p1 = [p1[p1["Model"] == m]["ACC(%)"].values[0] if not p1[p1["Model"] == m].empty and not np.isnan(p1[p1["Model"] == m]["ACC(%)"].values[0]) else 0.0 for m in models]
    acc_p2 = [p2[p2["Model"] == m]["ACC(%)"].values[0] if not p2[p2["Model"] == m].empty and not np.isnan(p2[p2["Model"] == m]["ACC(%)"].values[0]) else 0.0 for m in models]
    bars_acc1 = ax_acc.bar(x - width/2, acc_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_acc2 = ax_acc.bar(x + width/2, acc_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(bars_acc1) + list(bars_acc2):
        h = b.get_height()
        if h > 0:
            ax_acc.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("(a) Classification Accuracy", fontsize=9.5, pad=8)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_acc.set_ylim(0, 115)
    ax_acc.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 2. Panel (b): NMAE Tổng thể (%)
    nmae_p1 = [p1[p1["Model"] == m]["NMAE_Overall(%)"].values[0] if not p1[p1["Model"] == m].empty and not np.isnan(p1[p1["Model"] == m]["NMAE_Overall(%)"].values[0]) else 0.0 for m in models]
    nmae_p2 = [p2[p2["Model"] == m]["NMAE_Overall(%)"].values[0] if not p2[p2["Model"] == m].empty and not np.isnan(p2[p2["Model"] == m]["NMAE_Overall(%)"].values[0]) else 0.0 for m in models]
    bars_n1 = ax_nmae.bar(x - width/2, nmae_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_n2 = ax_nmae.bar(x + width/2, nmae_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    max_nmae_all = max(nmae_p1 + nmae_p2) if (nmae_p1 + nmae_p2) else 10.0
    for b in list(bars_n1) + list(bars_n2):
        h = b.get_height()
        if h > 0:
            ax_nmae.text(b.get_x() + b.get_width()/2., h + max_nmae_all*0.02, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_nmae.set_ylabel("Overall NMAE (%)")
    ax_nmae.set_title("(b) Overall Dimension NMAE (%)", fontsize=9.5, pad=8)
    ax_nmae.set_xticks(x)
    ax_nmae.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_nmae.set_ylim(0, max_nmae_all * 1.25 if max_nmae_all > 0 else 10)
    ax_nmae.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 3. Panel (c): Width (W) MAE (mm)
    w_p1 = [p1[p1["Model"] == m]["MAE_W(mm)"].values[0] for m in models]
    w_p2 = [p2[p2["Model"] == m]["MAE_W(mm)"].values[0] for m in models]
    bars_w1 = ax_w.bar(x - width/2, w_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_w2 = ax_w.bar(x + width/2, w_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    max_w_all = max(w_p1 + w_p2) if (w_p1 + w_p2) else 0.5
    for b in list(bars_w1) + list(bars_w2):
        h = b.get_height()
        if h > 0:
            ax_w.text(b.get_x() + b.get_width()/2., h + max_w_all*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax_w.set_ylabel("Width MAE (mm)")
    ax_w.set_title("(c) Width (W) Error", fontsize=9.5, pad=8)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(models, rotation=20, ha="right", fontsize=8.0)
    ax_w.set_ylim(0, max_w_all * 1.25 if max_w_all > 0 else 0.5)
    ax_w.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 4. Panel (d): Length (L) MAE (mm)
    l_p1 = [p1[p1["Model"] == m]["MAE_L(mm)"].values[0] for m in models]
    l_p2 = [p2[p2["Model"] == m]["MAE_L(mm)"].values[0] for m in models]
    bars_l1 = ax_l.bar(x - width/2, l_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_l2 = ax_l.bar(x + width/2, l_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    max_l_all = max(l_p1 + l_p2) if (l_p1 + l_p2) else 5.0
    for b in list(bars_l1) + list(bars_l2):
        h = b.get_height()
        if h > 0:
            ax_l.text(b.get_x() + b.get_width()/2., h + max_l_all*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax_l.set_ylabel("Length MAE (mm)")
    ax_l.set_title("(d) Length (L) Error", fontsize=9.5, pad=8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(models, rotation=20, ha="right", fontsize=8.0)
    ax_l.set_ylim(0, max_l_all * 1.25 if max_l_all > 0 else 5.0)
    ax_l.grid(True, linestyle=":", axis="y", alpha=0.6)

    # 5. Panel (e): Depth (D) MAE (mm)
    d_p1 = [p1[p1["Model"] == m]["MAE_D(mm)"].values[0] for m in models]
    d_p2 = [p2[p2["Model"] == m]["MAE_D(mm)"].values[0] for m in models]
    bars_d1 = ax_d.bar(x - width/2, d_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_d2 = ax_d.bar(x + width/2, d_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    max_d_all = max(d_p1 + d_p2) if (d_p1 + d_p2) else 1.0
    for b in list(bars_d1) + list(bars_d2):
        h = b.get_height()
        if h > 0:
            ax_d.text(b.get_x() + b.get_width()/2., h + max_d_all*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax_d.set_ylabel("Depth MAE (mm)")
    ax_d.set_title("(e) Depth (D) Error", fontsize=9.5, pad=8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(models, rotation=20, ha="right", fontsize=8.0)
    ax_d.set_ylim(0, max_d_all * 1.25 if max_d_all > 0 else 1.0)
    ax_d.grid(True, linestyle=":", axis="y", alpha=0.6)

    save_ieee_figure(fig, output_base_path)

    # ---------------------------------------------------------
    # HÌNH 2 PHỤ 1: CHỈ SO SÁNH ACC VÀ NMAE TỔNG THỂ (1 HÀNG GỌN)
    # ---------------------------------------------------------
    fig_sub1, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)
    fig_sub1.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
    )

    bars_s1a = ax_s1.bar(x - width/2, acc_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_s1b = ax_s1.bar(x + width/2, acc_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(bars_s1a) + list(bars_s1b):
        h = b.get_height()
        if h > 0:
            ax_s1.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_s1.set_ylabel("Accuracy (%)")
    ax_s1.set_title("(a) Classification Accuracy", fontsize=9.5, pad=8)
    ax_s1.set_xticks(x)
    ax_s1.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_s1.set_ylim(0, 115)
    ax_s1.grid(True, linestyle=":", axis="y", alpha=0.6)

    bars_s2a = ax_s2.bar(x - width/2, nmae_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_s2b = ax_s2.bar(x + width/2, nmae_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(bars_s2a) + list(bars_s2b):
        h = b.get_height()
        if h > 0:
            ax_s2.text(b.get_x() + b.get_width()/2., h + max_nmae_all*0.02, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_s2.set_ylabel("Overall NMAE (%)")
    ax_s2.set_title("(b) Overall Dimension NMAE (%)", fontsize=9.5, pad=8)
    ax_s2.set_xticks(x)
    ax_s2.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_s2.set_ylim(0, max_nmae_all * 1.25 if max_nmae_all > 0 else 10)
    ax_s2.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_ieee_figure(fig_sub1, f"{output_base_path}_overall_acc_nmae")

    # ---------------------------------------------------------
    # HÌNH 2 PHỤ 2: SO SÁNH MAE RIÊNG TỪNG CHIỀU W, L, D (1 HÀNG 3 PANEL)
    # ---------------------------------------------------------
    fig_sub2, (ax_sw, ax_sl, ax_sd) = plt.subplots(1, 3, figsize=(7.2, 2.8), dpi=300)
    fig_sub2.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
    )
    
    # W
    bars_sw1 = ax_sw.bar(x - width/2, w_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_sw2 = ax_sw.bar(x + width/2, w_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(bars_sw1) + list(bars_sw2):
        h = b.get_height()
        if h > 0:
            ax_sw.text(b.get_x() + b.get_width()/2., h + max_w_all*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_sw.set_ylabel("Width MAE (mm)")
    ax_sw.set_title("(a) Width (W) MAE", fontsize=9.5, pad=8)
    ax_sw.set_xticks(x)
    ax_sw.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_sw.set_ylim(0, max_w_all * 1.25 if max_w_all > 0 else 0.5)
    ax_sw.grid(True, linestyle=":", axis="y", alpha=0.6)

    # L
    bars_sl1 = ax_sl.bar(x - width/2, l_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_sl2 = ax_sl.bar(x + width/2, l_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(bars_sl1) + list(bars_sl2):
        h = b.get_height()
        if h > 0:
            ax_sl.text(b.get_x() + b.get_width()/2., h + max_l_all*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_sl.set_ylabel("Length MAE (mm)")
    ax_sl.set_title("(b) Length (L) MAE", fontsize=9.5, pad=8)
    ax_sl.set_xticks(x)
    ax_sl.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_sl.set_ylim(0, max_l_all * 1.25 if max_l_all > 0 else 5.0)
    ax_sl.grid(True, linestyle=":", axis="y", alpha=0.6)

    # D
    bars_sd1 = ax_sd.bar(x - width/2, d_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    bars_sd2 = ax_sd.bar(x + width/2, d_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(bars_sd1) + list(bars_sd2):
        h = b.get_height()
        if h > 0:
            ax_sd.text(b.get_x() + b.get_width()/2., h + max_d_all*0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_sd.set_ylabel("Depth MAE (mm)")
    ax_sd.set_title("(c) Depth (D) MAE", fontsize=9.5, pad=8)
    ax_sd.set_xticks(x)
    ax_sd.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax_sd.set_ylim(0, max_d_all * 1.25 if max_d_all > 0 else 1.0)
    ax_sd.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_ieee_figure(fig_sub2, f"{output_base_path}_wld_mae_breakdown")


def plot_linear_fit_wld_ieee(
    true_wld: np.ndarray,
    pred_wld: np.ndarray,
    model_name: str,
    output_base_path: str,
):
    """
    Vẽ đồ thị tương quan hồi quy tuyến tính 3 chiều W, L, D với đường 1:1 và đường hồi quy fit.
    Legend đặt trong vùng trống trên bên trái của Subplot (tránh hoàn toàn va chạm Title).
    """
    setup_ieee_style()
    dim_names = ["Width (W)", "Length (L)", "Depth (D)"]
    dim_colors = [COLOR_WIDTH_W, COLOR_LENGTH_L, COLOR_DEPTH_D]
    units = "mm"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300)

    for i in range(3):
        ax = axes[i]
        yt = true_wld[:, i]
        yp = pred_wld[:, i]

        # Đường 1:1 lý tưởng
        min_v = min(yt.min(), yp.min()) * 0.9
        max_v = max(yt.max(), yp.max()) * 1.1
        ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.2, label="Ideal 1:1")

        # Điểm dữ liệu với màu sắc đồng bộ theo chiều
        ax.scatter(yt, yp, color=dim_colors[i], edgecolor="black", s=36, zorder=5, label=f"Data ({dim_names[i]})")

        # Hồi quy tuyến tính
        if len(yt) > 1 and np.std(yt) > 1e-6:
            slope, intercept = np.polyfit(yt, yp, 1)
            x_line = np.linspace(min_v, max_v, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="#333333", lw=1.4, ls="-", label=f"Fit: y={slope:.2f}x+{intercept:.2f}")

        ax.set_xlabel(f"True {dim_names[i]} ({units})")
        if i == 0:
            ax.set_ylabel(f"Predicted ({units})")
        ax.set_title(dim_names[i], fontsize=9.5, pad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        # Chú thích đặt góc trên bên trái nơi hoàn toàn không có điểm dữ liệu (đường chéo nằm ở giữa)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    plt.tight_layout()
    save_ieee_figure(fig, output_base_path)
