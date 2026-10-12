# -*- coding: utf-8 -*-
"""
===============================================================================
IEEE TRANSACTIONS PLOTTING SUITE FOR SIM-TO-REAL ECT DOMAIN ADAPTATION
===============================================================================
Mô-đun trực quan hóa dữ liệu khoa học thống nhất, chuẩn mực xuất bản IEEE Transactions (Q1/Q2):
1. Font chữ: Times New Roman / Serif, đồng bộ từ kích cỡ nhãn trục (9.5-10 pt),
   số liệu tick (8.5 pt), chú thích legend (8.0-8.5 pt).
2. Xuất bản đồng thời 2 định dạng: 300 DPI PNG và vector PDF (sử dụng trực tiếp trong LaTeX).
3. Không đặt tiêu đề to (figure title) đè lấp dữ liệu - dành không gian cho \\caption trong LaTeX.
4. Đặt Legend bên ngoài khung hoặc trong khoảng trống an toàn để không đè lên điểm dữ liệu.
5. Tỉ lệ khung hình Parity 3D regression 1:1 chuẩn mực (set_box_aspect(1)), kèm đường y = x.
6. Tổ chức thư mục phân cấp rõ ràng, ngăn nắp theo cấu trúc:
   figures/
   ├── 01_protocol1_scan_split/
   │   ├── confusion_matrices/
   │   ├── regression_scatter/
   │   ├── loss_convergence/
   │   └── performance_metrics/
   ├── 02_protocol2_10fold_lodo/
   │   ├── confusion_matrices/
   │   ├── regression_scatter/
   │   └── performance_metrics/
   └── 03_cross_protocol_comparison/
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Any, Optional, Tuple, Union

# =============================================================================
# BẢNG MÀU ĐỒNG BỘ CHUẨN IEEE TRANSACTIONS (Q1/Q2)
# =============================================================================
COLOR_PROPOSED   = "#d62728"  # Crimson Red (Proposed PINN)
COLOR_NOPINN     = "#1f77b4"  # Royal Steel Blue (NoPINN Baseline)
COLOR_PROTOCOL_1 = "#1f77b4"  # Royal Blue (Protocol 1: Scan Split)
COLOR_PROTOCOL_2 = "#e6550d"  # Amber Orange (Protocol 2: 10-Fold LODO)
COLOR_P1         = "#1f77b4"
COLOR_P2         = "#e6550d"

# Bảng màu cho từng chiều vật lý
COLOR_WIDTH_W    = "#1f77b4"  # Width (W): Royal Blue
COLOR_LENGTH_L   = "#ff7f0e"  # Vivid Orange: Length (L)
COLOR_DEPTH_D    = "#2ca02c"  # Emerald Green: Depth (D)
COLOR_W          = COLOR_WIDTH_W
COLOR_L          = COLOR_LENGTH_L
COLOR_D          = COLOR_DEPTH_D

COLOR_ACCURACY   = "#1f77b4"
COLOR_NMAE       = "#7b4173"

COLOR_MODEL_MAP = {
    "CNN_Proposed": "#d62728",
    "CNN_NoPINN":   "#1f77b4",
    "MLP":          "#2ca02c",
    "XIONG":        "#9467bd",
}
DEFAULT_MODEL_COLOR = "#555555"

DEFAULT_UNIQUE_SHAPES = ["Ellipse", "Rectangular", "Step_R", "Step_T", "Triangular"]
DISPLAY_SHAPE_NAMES   = ["Ellipse", "Rectangular", "Step-R", "Step-T", "Triangular"]


# =============================================================================
# CẤU HÌNH PHONG CÁCH IEEE TRANSACTIONS
# =============================================================================
def setup_ieee_style():
    """
    Cấu hình toàn bộ tham số Matplotlib rcParams theo chuẩn nghiêm ngặt của IEEE Transactions:
    - Font Times New Roman / Serif.
    - Trục với tick hướng vào trong (inward ticks).
    - Grid mờ nét đứt lịch sự (alpha=0.3).
    - Xuất file 300 DPI, bounding box tight.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 0,
        "axes.labelsize": 9.5,
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.fontsize": 8.0,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#888888",
        "legend.fancybox": False,
        "figure.titlesize": 0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def save_dual_figure(fig: plt.Figure, output_base_path: str):
    """
    Lưu đồ thị đồng thời thành cả 2 định dạng:
    1. Ảnh raster 300 DPI PNG (cho xem nhanh, web, tài liệu Word/PowerPoint).
    2. Vector PDF (độ phân giải vô hạn, dùng trực tiếp khi biên dịch LaTeX).
    Tự động khởi tạo thư mục cha nếu chưa tồn tại.
    """
    base_no_ext = os.path.splitext(output_base_path)[0]
    out_dir = os.path.dirname(os.path.abspath(base_no_ext))
    os.makedirs(out_dir, exist_ok=True)
    png_path = f"{base_no_ext}.png"
    pdf_path = f"{base_no_ext}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    except Exception as e:
        print(f"    [Cảnh báo lưu PDF]: {e}")
    plt.close(fig)
    print(f"    [IEEE Plot] Đã lưu: {os.path.basename(png_path)} & .pdf tại {os.path.basename(out_dir)}/")


def save_ieee_figure(fig: plt.Figure, output_base_path: str):
    """Bí danh tương thích ngược cho save_dual_figure."""
    save_dual_figure(fig, output_base_path)


# =============================================================================
# QUẢN LÝ CẤU TRÚC THƯ MỤC BIỂU ĐỒ CHUẨN MỰC
# =============================================================================
def get_protocol_subdirs(base_figures_dir: str, protocol: int) -> Dict[str, str]:
    """
    Tự động khởi tạo và trả về các đường dẫn thư mục con cho từng giao thức:
    - Protocol 1: figures/01_protocol1_scan_split/ (confusion_matrices, regression_scatter, loss_convergence, performance_metrics)
    - Protocol 2: figures/02_protocol2_10fold_lodo/ (confusion_matrices, regression_scatter, performance_metrics)
    """
    if protocol == 1:
        p_dir = os.path.join(base_figures_dir, "01_protocol1_scan_split")
        subdirs = {
            "root": p_dir,
            "cm": os.path.join(p_dir, "confusion_matrices"),
            "reg": os.path.join(p_dir, "regression_scatter"),
            "loss": os.path.join(p_dir, "loss_convergence"),
            "metrics": os.path.join(p_dir, "performance_metrics"),
        }
    else:
        p_dir = os.path.join(base_figures_dir, "02_protocol2_10fold_lodo")
        subdirs = {
            "root": p_dir,
            "cm": os.path.join(p_dir, "confusion_matrices"),
            "reg": os.path.join(p_dir, "regression_scatter"),
            "metrics": os.path.join(p_dir, "performance_metrics"),
        }

    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    return subdirs


def get_cross_protocol_dir(base_figures_dir: str) -> str:
    """Khởi tạo và trả về thư mục so sánh liên giao thức: figures/03_cross_protocol_comparison/"""
    cross_dir = os.path.join(base_figures_dir, "03_cross_protocol_comparison")
    os.makedirs(cross_dir, exist_ok=True)
    return cross_dir


# =============================================================================
# 1. BIỂU ĐỒ MA TRẬN NHẦM LẪN (CONFUSION MATRIX) CHUẨN IEEE
# =============================================================================
def plot_confusion_matrix_ieee(
    confusion_matrix: Union[np.ndarray, List[List[int]]],
    class_names: List[str] = DEFAULT_UNIQUE_SHAPES,
    model_name: str = "CNN_Proposed",
    output_base_path: str = "cm_figure",
    acc_val: Optional[float] = None,
    total_correct: Optional[int] = None,
    total_samples: Optional[int] = None,
):
    """
    Vẽ ma trận nhầm lẫn chuẩn xuất bản IEEE Transactions:
    - Ô vuông tỉ lệ 1:1 (equal aspect ratio).
    - Hiển thị cả tỷ lệ phần trăm chuẩn hóa (%) và số lượng mẫu thực tế (N).
    - Thang đo Colorbar đặt ngoài rìa bên phải với nhãn rõ ràng.
    - Nhãn trục tinh gọn: Ground Truth Defect Shape vs Predicted Shape.
    """
    setup_ieee_style()
    cm = np.asarray(confusion_matrix, dtype=float)
    n_classes = len(class_names)

    # Chuẩn hóa theo từng hàng (recall per class)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    if total_samples is None:
        total_samples = int(np.sum(cm))
    if total_correct is None:
        total_correct = int(np.trace(cm))
    if acc_val is None:
        acc_val = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    # Chuyển tên lớp sang định dạng hiển thị đẹp (vd: Step_R -> Step-R)
    display_names = [c.replace("_", "-") for c in class_names]

    fig, ax = plt.subplots(figsize=(3.9, 3.4), dpi=300)
    im = ax.imshow(cm_norm * 100.0, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8.0)
    cbar.set_label("Classification Accuracy (%)", rotation=270, labelpad=14, fontsize=8.5)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(display_names, fontsize=8.5)
    ax.set_aspect("equal")

    # Hiển thị số liệu trong từng ô
    thresh = 50.0
    for i in range(n_classes):
        for j in range(n_classes):
            pct = cm_norm[i, j] * 100.0
            cnt = int(cm[i, j])
            if cnt == 0:
                ax.text(j, i, "0", ha="center", va="center", color="#999999", fontsize=7.5)
            else:
                text_color = "white" if pct > thresh else "black"
                ax.text(j, i, f"{pct:.0f}%\n({cnt})", ha="center", va="center",
                        color=text_color, fontsize=8.0, fontweight="medium")

    ax.set_ylabel("Ground Truth Defect Shape", fontsize=9.0)
    clean_model_label = model_name.replace(" (Finetuned)", "").replace(" (Pretrained)", "")
    ax.set_xlabel(f"Predicted Shape ({clean_model_label}: {total_correct}/{total_samples} = {acc_val:.1f}%)", fontsize=9.0)
    ax.grid(False)

    plt.tight_layout()
    save_dual_figure(fig, output_base_path)


# =============================================================================
# 2. BIỂU ĐỒ HỒI QUY KÍCH THƯỚC 3D PARITY SCATTER PLOT (1:1 PARITY BOX)
# =============================================================================
def plot_regression_scatter_ieee(
    true_wld: np.ndarray,
    pred_wld: np.ndarray,
    model_name: str,
    output_base_path: str,
    title_suffix: str = "",
):
    """
    Vẽ biểu đồ tương quan hồi quy kích thước 3D (Width, Length, Depth) chuẩn IEEE:
    - 3 panel ngang độc lập với trục tối ưu cho từng chiều vật lý.
    - Tỉ lệ khung hình Parity 1:1 vuông vức (ax.set_box_aspect(1)).
    - Đường chéo lý tưởng 45 độ y = x (Ideal Identity Line).
    - Hộp huy hiệu thông số thống kê nghiêm ngặt: R^2, MAE (mm), RMSE (mm).
    - Chú thích legend chung nằm bên ngoài đỉnh biểu đồ, không đè lấp điểm dữ liệu.
    """
    setup_ieee_style()
    targets = [
        ("Width, $W$ (mm)", COLOR_WIDTH_W, 0),
        ("Length, $L$ (mm)", COLOR_LENGTH_L, 1),
        ("Depth, $D$ (mm)", COLOR_DEPTH_D, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75), dpi=300)

    for i, (dim_label, color, col_idx) in enumerate(targets):
        ax = axes[i]
        t = true_wld[:, col_idx].astype(float)
        p = pred_wld[:, col_idx].astype(float)

        r2 = r2_score(t, p) if len(np.unique(t)) > 1 else 0.0
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))

        min_v = min(np.min(t), np.min(p))
        max_v = max(np.max(t), np.max(p))
        span = max_v - min_v if max_v > min_v else 1.0
        pad = 0.08 * span
        lims = [max(0.0, min_v - pad), max_v + pad]

        ax.scatter(t, p, alpha=0.85, color=color, edgecolors="black", linewidths=0.6, s=28, zorder=3)
        ax.plot(lims, lims, "k--", linewidth=1.1, label=r"Ideal ($y=x$)", zorder=2)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_box_aspect(1)

        ax.set_xlabel(f"True {dim_label}", fontsize=8.5)
        ax.set_ylabel(f"Pred. {dim_label}", fontsize=8.5)

        # Hộp thông số góc trên bên trái
        metrics_text = f"$R^2 = {r2:.3f}$\nMAE = {mae:.3f} mm\nRMSE = {rmse:.3f} mm"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=7.2, verticalalignment="top",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.92))

    legend_elements = [
        Line2D([0], [0], linestyle="--", color="k", linewidth=1.1, label=r"Ideal Parity Line ($y=x$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markeredgecolor="black", markersize=6, label=f"Model: {model_name}"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True, fontsize=8.0)

    plt.tight_layout()
    save_dual_figure(fig, output_base_path)


def plot_linear_fit_wld_ieee(
    true_wld: np.ndarray,
    pred_wld: np.ndarray,
    model_name: str,
    output_base_path: str,
):
    """Bí danh tương thích ngược chuyển tiếp trực tiếp sang plot_regression_scatter_ieee."""
    plot_regression_scatter_ieee(true_wld, pred_wld, model_name, output_base_path)


# =============================================================================
# 3. BIỂU ĐỒ HỘI TỤ LOSS KHI FINETUNING (LOSS CONVERGENCE)
# =============================================================================
def plot_separate_loss_curve_single_model(
    history: Union[Dict[str, Any], pd.DataFrame],
    model_name: str,
    output_base_path: str,
    title_suffix: str = "",
):
    """
    Vẽ riêng biệt đường Loss khi Finetuning cho 1 mô hình:
    - Panel (a): Tổng Loss (Train vs Val) kèm chỉ dấu Best Epoch.
    - Panel (b): Phân rã Loss (Cross-Entropy phân loại vs MSE hồi quy kích thước).
    """
    setup_ieee_style()
    if isinstance(history, pd.DataFrame):
        hist_dict = history.to_dict(orient="list")
    else:
        hist_dict = history

    epochs = hist_dict.get("epoch", [])
    if len(epochs) == 0:
        return

    epochs = np.asarray(epochs)
    train_loss = np.asarray(hist_dict.get("train_loss", []))
    val_loss = np.asarray(hist_dict.get("val_loss", []))
    train_clf = np.asarray(hist_dict.get("train_loss_clf", []))
    train_reg = np.asarray(hist_dict.get("train_loss_reg", []))
    val_clf = np.asarray(hist_dict.get("val_loss_clf", []))
    val_reg = np.asarray(hist_dict.get("val_loss_reg", []))

    best_ep = hist_dict.get("best_epoch", None)
    if isinstance(best_ep, list) and len(best_ep) > 0:
        best_ep = best_ep[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300)

    # Panel 1: Tổng Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, color=COLOR_NOPINN, lw=1.6, label="Train Total Loss")
    if len(val_loss) > 0 and not np.all(np.isnan(val_loss)):
        ax1.plot(epochs, val_loss, color=COLOR_PROPOSED, lw=1.6, ls="--", label="Val Total Loss")
        if best_ep is not None and best_ep <= len(epochs):
            ax1.axvline(best_ep, color="green", ls=":", lw=1.2, label=f"Best Val (Ep {best_ep})")
        else:
            best_idx = int(np.argmin(val_loss))
            ax1.axvline(epochs[best_idx], color="green", ls=":", lw=1.2, label=f"Best Val (Ep {epochs[best_idx]})")

    ax1.set_xlabel("Epoch", fontsize=8.5)
    ax1.set_ylabel("Total Loss", fontsize=8.5)
    ax1.text(0.04, 0.93, f"(a) Total Loss ({model_name})", transform=ax1.transAxes,
             fontsize=8.0, fontweight="bold", va="top")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.5)

    # Panel 2: Phân rã Loss (Clf vs Reg)
    ax2 = axes[1]
    has_decomp = False
    if len(train_clf) > 0 and not np.all(np.isnan(train_clf)) and any(c > 0 for c in train_clf):
        ax2.plot(epochs, train_clf, color="#ff7f0e", lw=1.4, label="Train Clf (CE)")
        has_decomp = True
    if len(val_clf) > 0 and not np.all(np.isnan(val_clf)) and any(c > 0 for c in val_clf):
        ax2.plot(epochs, val_clf, color="#e377c2", lw=1.4, ls="--", label="Val Clf (CE)")
        has_decomp = True
    if len(train_reg) > 0 and not np.all(np.isnan(train_reg)) and any(r > 0 for r in train_reg):
        ax2.plot(epochs, train_reg, color="#2ca02c", lw=1.4, label="Train Reg (MSE)")
        has_decomp = True
    if len(val_reg) > 0 and not np.all(np.isnan(val_reg)) and any(r > 0 for r in val_reg):
        ax2.plot(epochs, val_reg, color="#17becf", lw=1.4, ls="--", label="Val Reg (MSE)")
        has_decomp = True

    ax2.set_xlabel("Epoch", fontsize=8.5)
    ax2.set_ylabel("Decomposed Task Loss", fontsize=8.5)
    ax2.text(0.04, 0.93, f"(b) Task Decomposition ({model_name})", transform=ax2.transAxes,
             fontsize=8.0, fontweight="bold", va="top")
    ax2.grid(True, linestyle=":", alpha=0.6)
    if has_decomp:
        ax2.legend(loc="upper right", frameon=True, fontsize=7.2)

    plt.tight_layout()
    save_dual_figure(fig, output_base_path)


def plot_all_methods_loss_grid(
    histories_dict: Dict[str, Any],
    output_base_path: str,
    title: str = "",
):
    """
    Vẽ so sánh trực diện đường hội tụ Loss giữa tất cả các mô hình:
    - Panel (a): So sánh Training Loss
    - Panel (b): So sánh Validation Loss
    """
    setup_ieee_style()
    if not histories_dict:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), dpi=300)
    ax_train, ax_val = axes[0], axes[1]

    for model_name, hist in histories_dict.items():
        if isinstance(hist, pd.DataFrame):
            h = hist.to_dict(orient="list")
        else:
            h = hist

        epochs = h.get("epoch", [])
        if not epochs:
            continue
        train_l = h.get("train_loss", [])
        val_l = h.get("val_loss", [])
        color = COLOR_MODEL_MAP.get(model_name, DEFAULT_MODEL_COLOR)
        ls = "-" if "proposed" in model_name.lower() or "pinn" in model_name.lower() else "--"

        ax_train.plot(epochs, train_l, color=color, lw=1.5, ls=ls, label=model_name)
        if val_l and not np.all(np.isnan(val_l)):
            ax_val.plot(epochs, val_l, color=color, lw=1.5, ls=ls, label=model_name)

    ax_train.set_xlabel("Epoch", fontsize=8.5)
    ax_train.set_ylabel("Training Total Loss", fontsize=8.5)
    ax_train.text(0.04, 0.93, "(a) Training Loss Comparison", transform=ax_train.transAxes,
                  fontsize=8.0, fontweight="bold", va="top")
    ax_train.grid(True, linestyle=":", alpha=0.6)
    ax_train.legend(loc="upper right", frameon=True, fontsize=7.5)

    ax_val.set_xlabel("Epoch", fontsize=8.5)
    ax_val.set_ylabel("Validation Total Loss", fontsize=8.5)
    ax_val.text(0.04, 0.93, "(b) Validation Loss Comparison", transform=ax_val.transAxes,
                fontsize=8.0, fontweight="bold", va="top")
    ax_val.grid(True, linestyle=":", alpha=0.6)
    ax_val.legend(loc="upper right", frameon=True, fontsize=7.5)

    plt.tight_layout()
    save_dual_figure(fig, output_base_path)


# =============================================================================
# 4. BIỂU ĐỒ CỘT SO SÁNH CHỈ SỐ THEO GIAO THỨC (PERFORMANCE METRICS)
# =============================================================================
def plot_protocol_metrics_ieee(
    df_results: pd.DataFrame,
    out_dir_metrics: str,
    protocol_num: int = 1,
):
    """
    Vẽ 2 bộ biểu đồ cột chuẩn mực cho 1 Giao thức:
    1. acc_and_nmae_comparison: Accuracy (%) & Overall NMAE (%) với nhãn số trực tiếp trên đỉnh cột.
    2. wld_mae_breakdown: MAE từng chiều Width, Length, Depth trên 3 trục tung độc lập (mm),
       ngăn chặn hiện tượng Length (10+ mm) lấn át Width (< 0.5 mm) và Depth (1-3 mm).
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

    # Ưu tiên thứ tự trực quan: CNN_NoPINN trước, CNN_Proposed sau
    all_models = list(df_plot["Model"].unique())
    preferred_order = ["CNN_NoPINN", "CNN_Proposed", "MLP", "XIONG"]
    models = [m for m in preferred_order if m in all_models]
    for m in all_models:
        if m not in models:
            models.append(m)

    x = np.arange(len(models))
    colors = [COLOR_MODEL_MAP.get(m, DEFAULT_MODEL_COLOR) for m in models]
    labels = [m.replace("_", "-") for m in models]

    # ---------------------------------------------------------
    # HÌNH 1: ACCURACY & OVERALL NMAE
    # ---------------------------------------------------------
    fig1, (ax_acc, ax_nmae) = plt.subplots(1, 2, figsize=(6.2, 2.7), dpi=300)

    # (a) Accuracy
    has_acc = "ACC(%)" in df_plot.columns
    if has_acc:
        accs = [
            float(df_plot[df_plot["Model"] == m]["ACC(%)"].values[0])
            if not df_plot[df_plot["Model"] == m].empty and not np.isnan(df_plot[df_plot["Model"] == m]["ACC(%)"].values[0])
            else 0.0
            for m in models
        ]
        bars1 = ax_acc.bar(x, accs, width=0.40, color=colors, edgecolor="black", lw=0.8)
        for bar in bars1:
            h = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width() / 2.0, h + 1.8, f"{h:.1f}%",
                        ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax_acc.set_ylabel("Classification Accuracy (%)", fontsize=8.5)
        ax_acc.text(0.04, 0.93, "(a) Accuracy", transform=ax_acc.transAxes, fontsize=8.0, fontweight="bold", va="top")
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels, fontsize=8.0)
        ax_acc.set_ylim(0, 115)
        ax_acc.grid(True, linestyle=":", axis="y", alpha=0.6)

    # (b) Overall NMAE
    has_nmae = "NMAE_Overall(%)" in df_plot.columns
    if has_nmae:
        nmaes = [
            float(df_plot[df_plot["Model"] == m]["NMAE_Overall(%)"].values[0])
            if not df_plot[df_plot["Model"] == m].empty and not np.isnan(df_plot[df_plot["Model"] == m]["NMAE_Overall(%)"].values[0])
            else 0.0
            for m in models
        ]
        bars2 = ax_nmae.bar(x, nmaes, width=0.40, color=colors, edgecolor="black", lw=0.8)
        max_nm = max(nmaes) if nmaes else 15.0
        for bar in bars2:
            h = bar.get_height()
            ax_nmae.text(bar.get_x() + bar.get_width() / 2.0, h + max_nm * 0.02, f"{h:.2f}%",
                         ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax_nmae.set_ylabel("Overall Dimension NMAE (%)", fontsize=8.5)
        ax_nmae.text(0.04, 0.93, "(b) Overall NMAE", transform=ax_nmae.transAxes, fontsize=8.0, fontweight="bold", va="top")
        ax_nmae.set_xticks(x)
        ax_nmae.set_xticklabels(labels, fontsize=8.0)
        ax_nmae.set_ylim(0, max_nm * 1.25)
        ax_nmae.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout()
    save_dual_figure(fig1, os.path.join(out_dir_metrics, "acc_and_nmae_comparison"))

    # ---------------------------------------------------------
    # HÌNH 2: MAE TỪNG CHIỀU W, L, D TRÊN TRỤC ĐỘC LẬP (mm)
    # ---------------------------------------------------------
    fig2, (ax_w, ax_l, ax_d) = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=300)
    dims_cfg = [
        (ax_w, "MAE_W(mm)", "Width MAE, $W$ (mm)", "(a) Width Error ($W$)"),
        (ax_l, "MAE_L(mm)", "Length MAE, $L$ (mm)", "(b) Length Error ($L$)"),
        (ax_d, "MAE_D(mm)", "Depth MAE, $D$ (mm)", "(c) Depth Error ($D$)"),
    ]
    for ax, col, y_label, p_label in dims_cfg:
        vals = [
            float(df_plot[df_plot["Model"] == m][col].values[0])
            if not df_plot[df_plot["Model"] == m].empty and not np.isnan(df_plot[df_plot["Model"] == m][col].values[0])
            else 0.0
            for m in models
        ]
        bars = ax.bar(x, vals, width=0.40, color=colors, edgecolor="black", lw=0.8)
        max_v = max(vals) if vals else 1.0
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + max_v * 0.02, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=8.0)
        ax.text(0.04, 0.93, p_label, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, max_v * 1.30)
        ax.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout()
    save_dual_figure(fig2, os.path.join(out_dir_metrics, "wld_mae_breakdown"))


def plot_benchmark_comparison_ieee(df_results: pd.DataFrame, output_base_path: str, title_suffix: str = ""):
    """Bí danh tương thích ngược chuyển tiếp trực tiếp sang plot_protocol_metrics_ieee."""
    out_dir = os.path.dirname(os.path.abspath(output_base_path))
    plot_protocol_metrics_ieee(df_results, out_dir, protocol_num=1)


def plot_dimensional_mae_breakdown_ieee(df_results: pd.DataFrame, output_base_path: str, title_suffix: str = ""):
    """Bí danh tương thích ngược chuyển tiếp trực tiếp sang plot_protocol_metrics_ieee."""
    out_dir = os.path.dirname(os.path.abspath(output_base_path))
    plot_protocol_metrics_ieee(df_results, out_dir, protocol_num=1)


# =============================================================================
# 5. BIỂU ĐỒ SO SÁNH TRỰC DIỆN LIÊN GIAO THỨC (CROSS-PROTOCOL COMPARISON)
# =============================================================================
def plot_finetuned_two_protocols_comparison(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_dir_cross: str,
):
    """
    SO SÁNH TRỰC DIỆN KẾT QUẢ GIỮA 2 GIAO THỨC THỬ NGHIỆM:
    - Protocol 1 (Scan Split): Đánh giá độ ổn định và khử trôi cảm biến giữa các lần quét.
    - Protocol 2 (10-Fold LODO): Đánh giá tổng quát hóa trên hình học vết nứt hoàn toàn chưa từng thấy.
    
    Xuất đồng thời:
    1. two_protocols_master_comparison: Bảng toàn diện 5 panel (ACC, NMAE, W, L, D MAE).
    2. cross_protocol_acc_nmae: Bảng tinh gọn 2 panel (cho bài báo IEEE 2 cột).
    3. cross_protocol_wld_mae: Bảng 3 panel chi tiết từng chiều vật lý (mm).
    """
    setup_ieee_style()
    if df_p1.empty or df_p2.empty:
        return

    p1 = df_p1[df_p1["Stage"] == "Finetuned"].copy() if "Stage" in df_p1.columns else df_p1.copy()
    p2 = df_p2.copy()

    preferred_order = ["CNN_NoPINN", "CNN_Proposed", "MLP", "XIONG"]
    available = [m for m in p1["Model"].unique() if m in p2["Model"].unique()]
    models = [m for m in preferred_order if m in available]
    for m in available:
        if m not in models:
            models.append(m)

    if not models:
        return

    x = np.arange(len(models))
    width = 0.32
    labels = [m.replace("_", "-") for m in models]

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_PROTOCOL_1, edgecolor="black", label="Protocol 1: Scan Split (Drift Robustness)"),
        mpatches.Patch(facecolor=COLOR_PROTOCOL_2, edgecolor="black", label="Protocol 2: 10-Fold LODO (Defect Generalization)"),
    ]

    # ---------------------------------------------------------
    # 1. HÌNH MASTER: 5 PANEL (Hàng 1: ACC, NMAE; Hàng 2: W, L, D)
    # ---------------------------------------------------------
    fig_master = plt.figure(figsize=(7.2, 4.4), dpi=300)
    outer_gs = fig_master.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.34, top=0.91, bottom=0.09, left=0.08, right=0.98)
    gs_top = outer_gs[0].subgridspec(1, 2, wspace=0.25)
    gs_bot = outer_gs[1].subgridspec(1, 3, wspace=0.48)

    fig_master.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                      ncol=2, frameon=True, framealpha=0.95, fontsize=8.0)

    ax_acc = fig_master.add_subplot(gs_top[0, 0])
    ax_nmae = fig_master.add_subplot(gs_top[0, 1])
    ax_w = fig_master.add_subplot(gs_bot[0, 0])
    ax_l = fig_master.add_subplot(gs_bot[0, 1])
    ax_d = fig_master.add_subplot(gs_bot[0, 2])

    # (a) Accuracy
    acc_p1 = [float(p1[p1["Model"] == m]["ACC(%)"].values[0]) for m in models]
    acc_p2 = [float(p2[p2["Model"] == m]["ACC(%)"].values[0]) for m in models]
    b1 = ax_acc.bar(x - width/2, acc_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    b2 = ax_acc.bar(x + width/2, acc_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax_acc.text(b.get_x() + b.get_width()/2., h + 1.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=8.5)
    ax_acc.text(0.04, 0.93, "(a) Accuracy (%)", transform=ax_acc.transAxes, fontsize=8.0, fontweight="bold", va="top")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, fontsize=8.0)
    ax_acc.set_ylim(0, 115)
    ax_acc.grid(True, linestyle=":", axis="y", alpha=0.6)

    # (b) Overall NMAE
    nm_p1 = [float(p1[p1["Model"] == m]["NMAE_Overall(%)"].values[0]) for m in models]
    nm_p2 = [float(p2[p2["Model"] == m]["NMAE_Overall(%)"].values[0]) for m in models]
    b1_nm = ax_nmae.bar(x - width/2, nm_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    b2_nm = ax_nmae.bar(x + width/2, nm_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    max_nm = max(nm_p1 + nm_p2)
    for b in list(b1_nm) + list(b2_nm):
        h = b.get_height()
        ax_nmae.text(b.get_x() + b.get_width()/2., h + max_nm * 0.02, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax_nmae.set_ylabel("Overall NMAE (%)", fontsize=8.5)
    ax_nmae.text(0.04, 0.93, "(b) Overall NMAE (%)", transform=ax_nmae.transAxes, fontsize=8.0, fontweight="bold", va="top")
    ax_nmae.set_xticks(x)
    ax_nmae.set_xticklabels(labels, fontsize=8.0)
    ax_nmae.set_ylim(0, max_nm * 1.25)
    ax_nmae.grid(True, linestyle=":", axis="y", alpha=0.6)

    # (c, d, e) W, L, D MAE (mm)
    dim_panels = [
        (ax_w, "MAE_W(mm)", "Width MAE (mm)", "(c) Width ($W$)"),
        (ax_l, "MAE_L(mm)", "Length MAE (mm)", "(d) Length ($L$)"),
        (ax_d, "MAE_D(mm)", "Depth MAE (mm)", "(e) Depth ($D$)"),
    ]
    for ax, col, y_label, p_label in dim_panels:
        v_p1 = [float(p1[p1["Model"] == m][col].values[0]) for m in models]
        v_p2 = [float(p2[p2["Model"] == m][col].values[0]) for m in models]
        b1_v = ax.bar(x - width/2, v_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
        b2_v = ax.bar(x + width/2, v_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
        max_v = max(v_p1 + v_p2)
        for b in list(b1_v) + list(b2_v):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + max_v * 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=8.0)
        ax.text(0.04, 0.93, p_label, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.2)
        ax.set_ylim(0, max_v * 1.30)
        if col in ["MAE_W(mm)", "MAE_L(mm)"]:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(True, linestyle=":", axis="y", alpha=0.6)

    save_dual_figure(fig_master, os.path.join(output_dir_cross, "two_protocols_master_comparison"))

    # ---------------------------------------------------------
    # 2. HÌNH TINH GỌN 2 PANEL (ACC & OVERALL NMAE)
    # ---------------------------------------------------------
    fig_compact, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=300)
    fig_compact.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                       ncol=2, frameon=True, framealpha=0.95, fontsize=8.0)

    b1_c = ax_c1.bar(x - width/2, acc_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    b2_c = ax_c1.bar(x + width/2, acc_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(b1_c) + list(b2_c):
        h = b.get_height()
        ax_c1.text(b.get_x() + b.get_width()/2., h + 1.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_c1.set_ylabel("Accuracy (%)", fontsize=8.5)
    ax_c1.text(0.04, 0.93, "(a) Accuracy (%)", transform=ax_c1.transAxes, fontsize=8.0, fontweight="bold", va="top")
    ax_c1.set_xticks(x)
    ax_c1.set_xticklabels(labels, fontsize=8.0)
    ax_c1.set_ylim(0, 115)
    ax_c1.grid(True, linestyle=":", axis="y", alpha=0.6)

    b1_nc = ax_c2.bar(x - width/2, nm_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
    b2_nc = ax_c2.bar(x + width/2, nm_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
    for b in list(b1_nc) + list(b2_nc):
        h = b.get_height()
        ax_c2.text(b.get_x() + b.get_width()/2., h + max_nm * 0.02, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_c2.set_ylabel("Overall NMAE (%)", fontsize=8.5)
    ax_c2.text(0.04, 0.93, "(b) Overall NMAE (%)", transform=ax_c2.transAxes, fontsize=8.0, fontweight="bold", va="top")
    ax_c2.set_xticks(x)
    ax_c2.set_xticklabels(labels, fontsize=8.0)
    ax_c2.set_ylim(0, max_nm * 1.25)
    ax_c2.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_dual_figure(fig_compact, os.path.join(output_dir_cross, "cross_protocol_acc_nmae"))

    # ---------------------------------------------------------
    # 3. HÌNH 3 PANEL MAE CHI TIẾT (W, L, D)
    # ---------------------------------------------------------
    fig_wld, (ax_cw, ax_cl, ax_cd) = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300)
    fig_wld.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncol=2, frameon=True, framealpha=0.95, fontsize=8.0)

    dim_panels_wld = [
        (ax_cw, "MAE_W(mm)", "Width MAE (mm)", "(a) Width ($W$)"),
        (ax_cl, "MAE_L(mm)", "Length MAE (mm)", "(b) Length ($L$)"),
        (ax_cd, "MAE_D(mm)", "Depth MAE (mm)", "(c) Depth ($D$)"),
    ]
    for ax, col, y_label, p_label in dim_panels_wld:
        v_p1 = [float(p1[p1["Model"] == m][col].values[0]) for m in models]
        v_p2 = [float(p2[p2["Model"] == m][col].values[0]) for m in models]
        b1_v = ax.bar(x - width/2, v_p1, width, color=COLOR_PROTOCOL_1, edgecolor="black", lw=0.8)
        b2_v = ax.bar(x + width/2, v_p2, width, color=COLOR_PROTOCOL_2, edgecolor="black", lw=0.8)
        max_v = max(v_p1 + v_p2)
        for b in list(b1_v) + list(b2_v):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + max_v * 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=8.0)
        ax.text(0.04, 0.93, p_label, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, max_v * 1.30)
        if col in ["MAE_W(mm)", "MAE_L(mm)"]:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_dual_figure(fig_wld, os.path.join(output_dir_cross, "cross_protocol_wld_mae"))


# =============================================================================
# 6. MASTER DISPATCHER: TÁI TẠO TẤT CẢ BIỂU ĐỒ TỪ THƯ MỤC RUN
# =============================================================================
def generate_all_organized_figures_from_run(run_dir: str):
    """
    Quét tự động thư mục run_dir/tables và tái sinh toàn bộ các biểu đồ khoa học
    vào đúng hệ thống thư mục phân cấp chuẩn mực:
    figures/
    ├── 01_protocol1_scan_split/
    ├── 02_protocol2_10fold_lodo/
    └── 03_cross_protocol_comparison/
    """
    tables_dir = os.path.join(run_dir, "tables")
    figures_base = os.path.join(run_dir, "figures")
    if not os.path.exists(tables_dir):
        print(f"    [Cảnh báo] Không tìm thấy thư mục tables: {tables_dir}")
        return

    subdirs_p1 = get_protocol_subdirs(figures_base, protocol=1)
    subdirs_p2 = get_protocol_subdirs(figures_base, protocol=2)
    dir_cross = get_cross_protocol_dir(figures_base)

    print("\n" + "=" * 80)
    print(f"ĐỒ THỊ CHUẨN IEEE TRANSACTIONS ĐƯỢC TÁI TẠO VÀO CÁC FOLDER PHÂN CẤP RIÊNG BIỆT")
    print(f"Nguồn dữ liệu: {tables_dir}")
    print(f"Đích hình vẽ : {figures_base}")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. GIAO THỨC 1 (SCAN SPLIT)
    # ---------------------------------------------------------
    f_p1_res = os.path.join(tables_dir, "protocol1_scan_split_results.csv")
    df_p1_res = None
    if os.path.exists(f_p1_res):
        df_p1_res = pd.read_csv(f_p1_res)

    # Tìm các file predictions protocol 1
    p1_pred_files = [f for f in os.listdir(tables_dir) if f.startswith("predictions_protocol1_") and f.endswith(".csv")]
    histories_dict_p1 = {}

    for pf in p1_pred_files:
        model_part = pf.replace("predictions_protocol1_", "").replace(".csv", "")
        # Tên mô hình chuẩn
        if "proposed" in model_part.lower():
            m_key = "CNN_Proposed"
        elif "nopinn" in model_part.lower():
            m_key = "CNN_NoPINN"
        elif "mlp" in model_part.lower():
            m_key = "MLP"
        elif "xiong" in model_part.lower():
            m_key = "XIONG"
        else:
            m_key = model_part.upper()

        df_pred = pd.read_csv(os.path.join(tables_dir, pf))

        # (a) Confusion Matrix
        if "True_Shape" in df_pred.columns and "Pred_Shape" in df_pred.columns:
            y_t = df_pred["True_Shape"].values
            y_p = df_pred["Pred_Shape"].values
            cm = sk_confusion_matrix(y_t, y_p, labels=DEFAULT_UNIQUE_SHAPES)
            plot_confusion_matrix_ieee(
                confusion_matrix=cm,
                class_names=DEFAULT_UNIQUE_SHAPES,
                model_name=m_key,
                output_base_path=os.path.join(subdirs_p1["cm"], f"cm_{m_key}"),
            )

        # (b) 3D Parity Regression Scatter
        if all(c in df_pred.columns for c in ["True_W", "Pred_W", "True_L", "Pred_L", "True_D", "Pred_D"]):
            true_wld = df_pred[["True_W", "True_L", "True_D"]].values
            pred_wld = df_pred[["Pred_W", "Pred_L", "Pred_D"]].values
            plot_regression_scatter_ieee(
                true_wld=true_wld,
                pred_wld=pred_wld,
                model_name=m_key,
                output_base_path=os.path.join(subdirs_p1["reg"], f"regression_scatter_{m_key}"),
            )

        # (c) Loss history
        lf_name = f"loss_history_protocol1_{model_part}.csv"
        lf_path = os.path.join(tables_dir, lf_name)
        if os.path.exists(lf_path):
            df_loss = pd.read_csv(lf_path)
            histories_dict_p1[m_key] = df_loss
            plot_separate_loss_curve_single_model(
                history=df_loss,
                model_name=m_key,
                output_base_path=os.path.join(subdirs_p1["loss"], f"loss_curve_{m_key}"),
            )

    if histories_dict_p1:
        plot_all_methods_loss_grid(
            histories_dict=histories_dict_p1,
            output_base_path=os.path.join(subdirs_p1["loss"], "loss_comparison_grid"),
        )

    if df_p1_res is not None:
        plot_protocol_metrics_ieee(df_p1_res, subdirs_p1["metrics"], protocol_num=1)

    # ---------------------------------------------------------
    # 2. GIAO THỨC 2 (10-FOLD LODO)
    # ---------------------------------------------------------
    f_p2_res = os.path.join(tables_dir, "protocol2_10fold_results.csv")
    df_p2_res = None
    if os.path.exists(f_p2_res):
        df_p2_res = pd.read_csv(f_p2_res)

    p2_pred_files = [f for f in os.listdir(tables_dir) if f.startswith("predictions_protocol2_") and f.endswith(".csv")]
    for pf in p2_pred_files:
        model_part = pf.replace("predictions_protocol2_", "").replace("_10fold", "").replace(".csv", "")
        if "proposed" in model_part.lower():
            m_key = "CNN_Proposed"
        elif "nopinn" in model_part.lower():
            m_key = "CNN_NoPINN"
        elif "mlp" in model_part.lower():
            m_key = "MLP"
        elif "xiong" in model_part.lower():
            m_key = "XIONG"
        else:
            m_key = model_part.upper()

        df_pred2 = pd.read_csv(os.path.join(tables_dir, pf))
        if "True_Shape" in df_pred2.columns and "Pred_Shape" in df_pred2.columns:
            y_t = df_pred2["True_Shape"].values
            y_p = df_pred2["Pred_Shape"].values
            cm = sk_confusion_matrix(y_t, y_p, labels=DEFAULT_UNIQUE_SHAPES)
            plot_confusion_matrix_ieee(
                confusion_matrix=cm,
                class_names=DEFAULT_UNIQUE_SHAPES,
                model_name=m_key,
                output_base_path=os.path.join(subdirs_p2["cm"], f"cm_{m_key}"),
            )

        if all(c in df_pred2.columns for c in ["True_W", "Pred_W", "True_L", "Pred_L", "True_D", "Pred_D"]):
            true_wld = df_pred2[["True_W", "True_L", "True_D"]].values
            pred_wld = df_pred2[["Pred_W", "Pred_L", "Pred_D"]].values
            plot_regression_scatter_ieee(
                true_wld=true_wld,
                pred_wld=pred_wld,
                model_name=m_key,
                output_base_path=os.path.join(subdirs_p2["reg"], f"regression_scatter_{m_key}"),
            )

    if df_p2_res is not None:
        plot_protocol_metrics_ieee(df_p2_res, subdirs_p2["metrics"], protocol_num=2)

    # ---------------------------------------------------------
    # 3. SO SÁNH LIÊN GIAO THỨC (CROSS-PROTOCOL)
    # ---------------------------------------------------------
    if df_p1_res is not None and df_p2_res is not None:
        plot_finetuned_two_protocols_comparison(df_p1_res, df_p2_res, dir_cross)

    # Dọn sạch các file hình thô nằm lẻ tẻ ở thư mục figures/ gốc (nếu có từ trước)
    for f in os.listdir(figures_base):
        fp = os.path.join(figures_base, f)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    print("\n[HOÀN TẤT TRỰC QUAN HÓA] Toàn bộ hình vẽ đã được tạo lập thành công theo chuẩn IEEE Transactions:")
    print(f"  1. Protocol 1 (Scan Split)  : {subdirs_p1['root']}")
    print(f"  2. Protocol 2 (10-Fold LODO): {subdirs_p2['root']}")
    print(f"  3. Cross-Protocol Comparison: {dir_cross}")
    print("=" * 80 + "\n")
