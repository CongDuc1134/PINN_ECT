import os
import sys
import copy
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import load_5khz_fully_prepared, predict_and_denormalize
from domain_adaptation.data_loader import get_5khz_10fold_defect_splits
from domain_adaptation.cnn.losses import compute_physics_loss_autograd, get_shape_physics_map
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
    DEFAULT_UNIQUE_SHAPES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_fold(
    base_model,
    X_train,
    y_clf_train,
    y_wld_train,
    metadata_train,
    y_scaler,
    shape_map,
    alpha=1.0,
    lr=1e-4,
    epochs=1500,
    freeze_backbone=True,
    freeze_bn=True,
    fold_idx=1,
):
    model = copy.deepcopy(base_model).to(DEVICE)

    if freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad = False
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    history = {
        "epoch": [],
        "total_loss": [],
        "data_loss": [],
        "phys_loss": [],
    }

    n_samples = len(X_train)

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)

        # 1. Data loss với Kendall uncertainty weighting
        w_clf = torch.exp(-model.log_var_clf)
        w_w   = torch.exp(-model.log_var_w)
        w_l   = torch.exp(-model.log_var_l)
        w_d   = torch.exp(-model.log_var_d)

        loss_clf = F.cross_entropy(logits, y_clf_train)
        loss_reg_w = F.mse_loss(pred_wld[:, 0], y_wld_train[:, 0])
        loss_reg_l = F.mse_loss(pred_wld[:, 1], y_wld_train[:, 1])
        loss_reg_d = F.mse_loss(pred_wld[:, 2], y_wld_train[:, 2])

        data_loss = (
            0.5 * w_clf * loss_clf   + 0.5 * model.log_var_clf +
            0.5 * w_w   * loss_reg_w + 0.5 * model.log_var_w   +
            0.5 * w_l   * loss_reg_l + 0.5 * model.log_var_l   +
            0.5 * w_d   * loss_reg_d + 0.5 * model.log_var_d
        )

        # 2. Physics loss vi phân trường Maxwell (autograd)
        sample_phys = []
        for i in range(n_samples):
            s_name = metadata_train[i]["true_shape"] if metadata_train else DEFAULT_UNIQUE_SHAPES[int(y_clf_train[i].item())]
            pl = compute_physics_loss_autograd(
                H_true=X_train[i, 0],
                w_pred=pred_wld[i, 0],
                l_pred=pred_wld[i, 1],
                d_pred=pred_wld[i, 2],
                shape_name=s_name,
                shape_map=shape_map,
                y_scaler=y_scaler,
            )
            sample_phys.append(pl)
        batch_phys = torch.stack(sample_phys).mean()

        total_loss = data_loss + alpha * batch_phys
        total_loss.backward()
        optimizer.step()

        history["epoch"].append(ep + 1)
        history["total_loss"].append(float(total_loss.item()))
        history["data_loss"].append(float(data_loss.item()))
        history["phys_loss"].append(float(batch_phys.item()))

        if (ep + 1) % 500 == 0 or ep == 0:
            print(f"    [Fold {fold_idx:02d} | Ep {ep+1:04d}/{epochs}] Total: {total_loss.item():.4f} | Data: {data_loss.item():.4f} | Phys: {batch_phys.item():.4f}", flush=True)

    model.eval()
    return model, history

def main():
    set_reproducible_seed(42)
    print("=" * 100, flush=True)
    print("CHẠY GIAO THỨC 2: 10-FOLD CROSS-VALIDATION (LEAVE-ONE-DEFECT-OUT) TRÊN 20 MẪU THỰC NGHIỆM", flush=True)
    print("CẤU HÌNH: CNN_Proposed | Freeze Backbone = True | Freeze BatchNorm = True | Constant LR = 1e-4 | Alpha = 1.0", flush=True)
    print(f"Thiết bị: {DEVICE}", flush=True)
    print("=" * 100, flush=True)

    bundle = load_5khz_fully_prepared(model_type="cnn", variant="pinn", device=DEVICE)
    base_model = bundle["model"]
    y_scaler = bundle["y_scaler"]
    metadata = bundle["metadata"]
    alpha = bundle.get("alpha", 1.0)
    shape_map = get_shape_physics_map()

    folds = get_5khz_10fold_defect_splits(bundle)
    oof_pred_shapes = [None] * len(metadata)
    oof_pred_wld = np.zeros((len(metadata), 3))
    all_fold_histories = []
    ckpt_file = os.path.join(PROJECT_ROOT, "scratch", "protocol2_checkpoint.pt")
    start_fold = 1
    if os.path.exists(ckpt_file):
        try:
            ckpt = torch.load(ckpt_file, map_location="cpu")
            start_fold = ckpt.get("f_idx", 0) + 1
            oof_pred_shapes = ckpt.get("oof_pred_shapes", oof_pred_shapes)
            oof_pred_wld = ckpt.get("oof_pred_wld", oof_pred_wld)
            all_fold_histories = ckpt.get("all_fold_histories", all_fold_histories)
            print(f"[RESUME] Đã nạp checkpoint trước đó, tiếp tục từ Fold {start_fold}/10 ...", flush=True)
        except Exception as e:
            print(f"[WARN] Không thể nạp checkpoint: {e}", flush=True)

    epochs = 1500
    lr = 1e-4

    t_start = time.time()

    for f_idx, fold_data in enumerate(folds, start=1):
        if f_idx < start_fold:
            continue
        held_out = fold_data["held_out_crack"]
        tr = fold_data["train"]
        te = fold_data["test"]
        print(f"\n>>> [Fold {f_idx:02d}/10] Đang huấn luyện (Hold out Vết nứt No.{held_out}, Test 2 mẫu) ...", flush=True)
        t_fold = time.time()

        trained_model, history = train_one_fold(
            base_model=base_model,
            X_train=tr["X"],
            y_clf_train=tr["y_clf"],
            y_wld_train=tr["y_wld_norm"],
            metadata_train=tr["metadata"],
            y_scaler=y_scaler,
            shape_map=shape_map,
            alpha=alpha,
            lr=lr,
            epochs=epochs,
            freeze_backbone=True,
            freeze_bn=True,
            fold_idx=f_idx,
        )
        all_fold_histories.append(history)

        p_sh, p_wld, _, _ = predict_and_denormalize(trained_model, te["X"], y_scaler=y_scaler)
        for loc_i, glob_i in enumerate(te["indices"]):
            oof_pred_shapes[glob_i] = p_sh[loc_i]
            oof_pred_wld[glob_i] = p_wld[loc_i]

        t_fold_el = time.time() - t_fold
        f_true_sh = [m["true_shape"] for m in te["metadata"]]
        f_stat = compute_classification_metrics_and_matrix(f_true_sh, p_sh)
        f_reg = compute_regression_metrics(te["y_wld_raw"].cpu().numpy(), p_wld)
        print(f"  -> Fold {f_idx:02d} xong trong {t_fold_el:.1f}s | ACC Fold: {f_stat['overall_acc']:.1f}% ({f_stat['total_correct']}/2) | MAE: {f_reg['MAE_Avg']:.4f} mm", flush=True)

        # Lưu checkpoint sau mỗi fold hoàn thành
        torch.save({
            "f_idx": f_idx,
            "oof_pred_shapes": oof_pred_shapes,
            "oof_pred_wld": oof_pred_wld,
            "all_fold_histories": all_fold_histories,
        }, ckpt_file)

    t_total = time.time() - t_start

    # =========================================================================
    # TỔNG HỢP TOÀN BỘ 20 DỰ ĐOÁN OUT-OF-FOLD
    # =========================================================================
    true_shapes_all = [m["true_shape"] for m in metadata]
    true_wld_all = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in metadata])

    stat_oof = compute_classification_metrics_and_matrix(true_shapes_all, oof_pred_shapes)
    reg_oof = compute_regression_metrics(true_wld_all, oof_pred_wld)

    print("\n" + "=" * 100, flush=True)
    print("KẾT QUẢ TỔNG HỢP 10-FOLD CROSS-VALIDATION (20 MẪU OUT-OF-FOLD):", flush=True)
    print(f"  Thời gian tổng cộng: {t_total:.1f}s ({t_total/60:.1f} phút)", flush=True)
    print(f"  Độ chính xác phân loại tổng thể: {stat_oof['overall_acc']:.2f}% ({stat_oof['total_correct']}/{stat_oof['total_samples']})", flush=True)
    print(f"  MAE Chiều rộng (W): {reg_oof['MAE_W']:.4f} mm | NMAE_W: {reg_oof['NMAE_W(%)']:.2f}%", flush=True)
    print(f"  MAE Chiều dài (L):  {reg_oof['MAE_L']:.4f} mm | NMAE_L: {reg_oof['NMAE_L(%)']:.2f}%", flush=True)
    print(f"  MAE Độ sâu (D):     {reg_oof['MAE_D']:.4f} mm | NMAE_D: {reg_oof['NMAE_D(%)']:.2f}%", flush=True)
    print(f"  MAE Trung bình:     {reg_oof['MAE_Avg']:.4f} mm | NMAE_Avg: {reg_oof['NMAE_Avg(%)']:.2f}%", flush=True)
    print("=" * 100, flush=True)

    # Lưu bảng dự đoán chi tiết 20 mẫu
    out_tables = os.path.join(PROJECT_ROOT, "domain_adaptation", "finetune_results_raw", "protocols", "tables")
    out_figures = os.path.join(PROJECT_ROOT, "domain_adaptation", "finetune_results_raw", "protocols", "figures")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figures, exist_ok=True)

    pred_records = []
    for i, m in enumerate(metadata):
        pred_records.append({
            "Filename": m["filename"],
            "Crack_No": m.get("crack_no", i + 1),
            "True_Shape": m["true_shape"],
            "Pred_Shape": oof_pred_shapes[i],
            "Shape_Match": (m["true_shape"] == oof_pred_shapes[i]),
            "True_W": m["true_w"], "Pred_W": round(float(oof_pred_wld[i, 0]), 4), "Err_W": round(float(abs(m["true_w"] - oof_pred_wld[i, 0])), 4),
            "True_L": m["true_l"], "Pred_L": round(float(oof_pred_wld[i, 1]), 4), "Err_L": round(float(abs(m["true_l"] - oof_pred_wld[i, 1])), 4),
            "True_D": m["true_d"], "Pred_D": round(float(oof_pred_wld[i, 2]), 4), "Err_D": round(float(abs(m["true_d"] - oof_pred_wld[i, 2])), 4),
        })
    df_preds = pd.DataFrame(pred_records)
    csv_preds = os.path.join(out_tables, "protocol2_10fold_predictions.csv")
    df_preds.to_csv(csv_preds, index=False)

    # Lưu bảng tổng hợp số liệu
    summary_row = {
        "Protocol": "Protocol 2 (10-Fold LODO)",
        "Model": "CNN_Proposed",
        "Backbone_Frozen": True,
        "BatchNorm_Frozen": True,
        "LR": "1e-4 (Constant)",
        "Epochs": epochs,
        "Alpha_PINN": alpha,
        "Total_Samples": stat_oof["total_samples"],
        "Correct": f"{stat_oof['total_correct']}/{stat_oof['total_samples']}",
        "ACC(%)": stat_oof["overall_acc"],
        "MAE_Avg(mm)": round(reg_oof["MAE_Avg"], 4),
        "MAE_W(mm)": round(reg_oof["MAE_W"], 4),
        "MAE_L(mm)": round(reg_oof["MAE_L"], 4),
        "MAE_D(mm)": round(reg_oof["MAE_D"], 4),
        "NMAE(%)": round(reg_oof["NMAE_Avg(%)"], 2),
        "Time(s)": round(t_total, 1),
    }
    df_summary = pd.DataFrame([summary_row])
    csv_summary = os.path.join(out_tables, "protocol2_10fold_results.csv")
    df_summary.to_csv(csv_summary, index=False)

    # Lưu Markdown Report
    md_file = os.path.join(out_tables, "protocol2_10fold_results.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO GIAO THỨC 2: 10-FOLD CROSS VALIDATION (LEAVE-ONE-DEFECT-OUT)\n\n")
        f.write("- **Mô hình**: `CNN_Proposed`\n")
        f.write("- **Thiết lập**: Khóa Backbone (`freeze_backbone=True`) & Khóa triệt để Batch Normalization (`freeze_bn=True`)\n")
        f.write("- **Tốc độ học**: Cố định (`Constant LR = 10^-4`, không dùng CosineAnnealingLR)\n")
        f.write(f"- **Hàm Loss**: $\\mathcal{{L}}_{{\\mathrm{{total}}}} = \\mathcal{{L}}_{{\\mathrm{{data}}}} + {alpha} \\cdot \\mathcal{{L}}_{{\\mathrm{{phys}}}}$\n\n")
        f.write("### 1. Bảng Chỉ Số Tổng Thể\n\n")
        cols = list(df_summary.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join([":---:" if c not in ["Model", "Protocol"] else ":---" for c in cols]) + " |\n")
        for _, r in df_summary.iterrows():
            f.write("| " + " | ".join([str(val) for val in r.values]) + " |\n")
        f.write("\n\n### 2. Thống Kê Từng Lớp Hình Dạng (Per-Class Breakdown)\n\n")
        pc_df = stat_oof["per_class_stats"]
        pc_cols = list(pc_df.columns)
        f.write("| " + " | ".join(pc_cols) + " |\n")
        f.write("| " + " | ".join([":---:" for _ in pc_cols]) + " |\n")
        for _, r in pc_df.iterrows():
            f.write("| " + " | ".join([str(val) for val in r.values]) + " |\n")
        f.write("\n")

    # =========================================================================
    # VẼ CÁC ĐỒ THỊ CHUẨN IEEE (KHÔNG TITLE, CHÚ THÍCH RA NGOÀI, KHÔNG TOTAL MAE)
    # =========================================================================
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 11

    # 1. HÌNH 1: MA TRẬN NHẦM LẪN 10-FOLD CHUẨN IEEE
    cm = stat_oof["confusion_matrix"]
    classes = DEFAULT_UNIQUE_SHAPES
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontweight="bold")
    ax.set_ylabel("True Defect Category", fontweight="bold")
    ax.set_xlabel("Predicted Defect Category", fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    plt.tight_layout()
    fig1_png = os.path.join(out_figures, "fig_ieee_10fold_confusion_matrix.png")
    fig1_pdf = os.path.join(out_figures, "fig_ieee_10fold_confusion_matrix.pdf")
    plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.close()

    # 2. HÌNH 2: TƯƠNG QUAN HỒI QUY 3 CHIỀU W, L, D CHUẨN IEEE (3 PANELS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=300)
    dim_names = ["Width (W)", "Length (L)", "Depth (D)"]
    units = "mm"
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for d in range(3):
        true_v = true_wld_all[:, d]
        pred_v = oof_pred_wld[:, d]
        ax = axes[d]

        # Đường lý tưởng y = x
        min_v = min(np.min(true_v), np.min(pred_v)) * 0.85
        max_v = max(np.max(true_v), np.max(pred_v)) * 1.15
        ref_line, = ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="#555555", linewidth=1.8, label="Ideal Line ($y=x$)")

        # Điểm đo thực tế
        sc = ax.scatter(true_v, pred_v, color=colors[d], edgecolor="black", s=60, alpha=0.9, zorder=4, label="Out-of-Fold Samples")

        # Hồi quy tuyến tính thực nghiệm
        slope, intercept = np.polyfit(true_v, pred_v, 1)
        r2 = np.corrcoef(true_v, pred_v)[0, 1] ** 2
        mae_dim = np.mean(np.abs(true_v - pred_v))
        fit_x = np.linspace(min_v, max_v, 50)
        fit_y = slope * fit_x + intercept
        fit_line, = ax.plot(fit_x, fit_y, color=colors[d], linewidth=2.2, label=f"Fit: $y={slope:.2f}x+{intercept:.2f}$")

        ax.set_xlabel(f"True {dim_names[d]} ({units})", fontweight="bold")
        ax.set_ylabel(f"Predicted {dim_names[d]} ({units})", fontweight="bold")
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Chú thích thông số góc phải dưới
        stats_text = f"$R^2 = {r2:.3f}$\n$\\mathrm{{MAE}} = {mae_dim:.3f}\\,\\mathrm{{mm}}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9), fontsize=10.5)

    # Legend ngoài khung trên đỉnh
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=True, fontsize=11, edgecolor="#999999")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2_png = os.path.join(out_figures, "fig_ieee_10fold_regression_fit.png")
    fig2_pdf = os.path.join(out_figures, "fig_ieee_10fold_regression_fit.pdf")
    plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.close()

    # 3. HÌNH 3: PHÂN RÃ SAI SỐ MAE 3D (W, L, D) - KHÔNG CÓ TOTAL MAE
    fig, (ax_acc, ax_mae) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    # Panel 1: Overall Out-of-fold Accuracy
    b_acc = ax_acc.bar([0], [stat_oof["overall_acc"]], width=0.4, color="#d62728", edgecolor="black", linewidth=1.2)
    ax_acc.text(0, stat_oof["overall_acc"] + 2.0, f"{stat_oof['overall_acc']:.1f}% ({stat_oof['total_correct']}/{stat_oof['total_samples']})", ha="center", fontweight="bold", fontsize=11)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_yticks([0, 20, 40, 60, 80, 100])
    ax_acc.set_xticks([0])
    ax_acc.set_xticklabels(["CNN_Proposed (10-Fold LODO)"], fontweight="bold")
    ax_acc.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax_acc.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 2: MAE W, L, D Breakdown (XÓA TOTAL MAE)
    x_pos = np.arange(1)
    bar_w = 0.22
    bw = ax_mae.bar(x_pos - bar_w, [reg_oof["MAE_W"]], width=bar_w, label="Width ($W$)", color="#2ca02c", edgecolor="black")
    bl = ax_mae.bar(x_pos,         [reg_oof["MAE_L"]], width=bar_w, label="Length ($L$)", color="#ff7f0e", edgecolor="black")
    bd = ax_mae.bar(x_pos + bar_w, [reg_oof["MAE_D"]], width=bar_w, label="Depth ($D$)", color="#d62728", edgecolor="black")

    ax_mae.text(x_pos[0] - bar_w, reg_oof["MAE_W"] + 0.04, f"{reg_oof['MAE_W']:.2f} mm", ha="center", fontweight="bold", fontsize=10)
    ax_mae.text(x_pos[0],         reg_oof["MAE_L"] + 0.04, f"{reg_oof['MAE_L']:.2f} mm", ha="center", fontweight="bold", fontsize=10)
    ax_mae.text(x_pos[0] + bar_w, reg_oof["MAE_D"] + 0.04, f"{reg_oof['MAE_D']:.2f} mm", ha="center", fontweight="bold", fontsize=10)

    ax_mae.set_xticks(x_pos)
    ax_mae.set_xticklabels(["CNN_Proposed (10-Fold LODO)"], fontweight="bold")
    ax_mae.set_ylabel("Mean Absolute Error (mm)", fontweight="bold")
    max_m = max([reg_oof["MAE_W"], reg_oof["MAE_L"], reg_oof["MAE_D"]])
    ax_mae.set_ylim(0, max_m * 1.35)
    ax_mae.grid(axis="y", linestyle="--", alpha=0.6)
    ax_mae.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=True, fontsize=11, edgecolor="#999999")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig3_png = os.path.join(out_figures, "fig_ieee_10fold_metrics_comparison.png")
    fig3_pdf = os.path.join(out_figures, "fig_ieee_10fold_metrics_comparison.pdf")
    plt.savefig(fig3_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig3_pdf, bbox_inches="tight")
    plt.close()

    # 4. HÌNH 4: ĐỒ THỊ HỘI TỤ LOSS TRUNG BÌNH 10 FOLDS (3 PANELS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=300)
    all_total = np.array([h["total_loss"] for h in all_fold_histories])
    all_data  = np.array([h["data_loss"] for h in all_fold_histories])
    all_phys  = np.array([h["phys_loss"] for h in all_fold_histories])
    ep_axis = np.arange(1, epochs + 1)

    mean_total, std_total = np.mean(all_total, axis=0), np.std(all_total, axis=0)
    mean_data,  std_data  = np.mean(all_data, axis=0),  np.std(all_data, axis=0)
    mean_phys,  std_phys  = np.mean(all_phys, axis=0),  np.std(all_phys, axis=0)

    # Panel 0: Total Loss
    axes[0].plot(ep_axis, mean_total, color="#d62728", linewidth=2.0, label="10-Fold Mean Total Loss")
    axes[0].fill_between(ep_axis, mean_total - std_total, mean_total + std_total, color="#d62728", alpha=0.15, label=r"$\pm 1\,\sigma$ Spread")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel(r"Total Loss ($\mathcal{L}_{\mathrm{total}}$)", fontweight="bold")
    axes[0].set_ylim(-30, 1400)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 1: Data Loss
    axes[1].plot(ep_axis, mean_data, color="#1f77b4", linewidth=2.0, label="10-Fold Mean Data Loss")
    axes[1].fill_between(ep_axis, mean_data - std_data, mean_data + std_data, color="#1f77b4", alpha=0.15, label=r"$\pm 1\,\sigma$ Spread")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel(r"Data Loss ($\mathcal{L}_{\mathrm{data}}$)", fontweight="bold")
    axes[1].set_ylim(-30, 1400)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Physics Loss
    axes[2].plot(ep_axis, mean_phys, color="#2ca02c", linewidth=2.0, label="10-Fold Mean Physics Loss")
    axes[2].fill_between(ep_axis, mean_phys - std_phys, mean_phys + std_phys, color="#2ca02c", alpha=0.15, label=r"$\pm 1\,\sigma$ Spread")
    axes[2].set_xlabel("Epoch", fontweight="bold")
    axes[2].set_ylabel(r"Physics Loss ($\mathcal{L}_{\mathrm{phys}}$)", fontweight="bold")
    axes[2].set_ylim(0.680, 0.700)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=True, fontsize=11, edgecolor="#999999")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig4_png = os.path.join(out_figures, "fig_ieee_10fold_loss_convergence.png")
    fig4_pdf = os.path.join(out_figures, "fig_ieee_10fold_loss_convergence.pdf")
    plt.savefig(fig4_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig4_pdf, bbox_inches="tight")
    plt.close()

    # Sao chép vào thư mục brain artifacts để hiển thị trực tiếp
    artifact_dir = r"C:\Users\Admin\.gemini\antigravity-ide\brain\f10e717b-0d2f-4d04-a176-aeaa228e62c7"
    import shutil
    for fpath in [fig1_png, fig2_png, fig3_png, fig4_png]:
        try:
            shutil.copy2(fpath, os.path.join(artifact_dir, os.path.basename(fpath)))
        except Exception as e:
            print(f"Warning: {e}")

    print(f"\n[OK] ĐÃ HOÀN TẤT VÀ XUẤT TOÀN BỘ 4 ĐỒ THỊ CHUẨN IEEE TẠI:\n  - {fig1_png}\n  - {fig2_png}\n  - {fig3_png}\n  - {fig4_png}", flush=True)

if __name__ == "__main__":
    main()
