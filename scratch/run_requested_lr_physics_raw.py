import os
import sys
import copy
import time
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

from domain_adaptation.model_loader import (
    load_5khz_fully_prepared,
    predict_and_denormalize,
    extract_pinn_alpha_from_checkpoint,
)
from domain_adaptation.data_loader import split_5khz_scan1_scan2
from domain_adaptation.cnn.losses import compute_physics_loss_autograd, get_shape_physics_map
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
    DEFAULT_UNIQUE_SHAPES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_single_pinn(
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
    freeze_bn=False,
    reset_kendall=False,
):
    model = copy.deepcopy(base_model).to(DEVICE)

    if reset_kendall:
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).data.fill_(0.0)

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
    # CỐ ĐỊNH LR BÌNH THƯỜNG (Constant LR, không dùng CosineAnnealingLR)

    history = {
        "epoch": [],
        "total_loss": [],
        "data_loss": [],
        "phys_loss": [],
        "clf_loss": [],
        "reg_loss": [],
    }

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)

        # 1. Data loss với Kendall uncertainty weighting từ pretrained checkpoint
        w_clf = torch.exp(-model.log_var_clf)
        w_w   = torch.exp(-model.log_var_w)
        w_l   = torch.exp(-model.log_var_l)
        w_d   = torch.exp(-model.log_var_d)

        loss_clf = F.cross_entropy(logits, y_clf_train)
        loss_reg_w = F.mse_loss(pred_wld[:, 0], y_wld_train[:, 0])
        loss_reg_l = F.mse_loss(pred_wld[:, 1], y_wld_train[:, 1])
        loss_reg_d = F.mse_loss(pred_wld[:, 2], y_wld_train[:, 2])
        avg_reg = (loss_reg_w + loss_reg_l + loss_reg_d) / 3.0

        data_loss = (
            0.5 * w_clf * loss_clf   + 0.5 * model.log_var_clf +
            0.5 * w_w   * loss_reg_w + 0.5 * model.log_var_w   +
            0.5 * w_l   * loss_reg_l + 0.5 * model.log_var_l   +
            0.5 * w_d   * loss_reg_d + 0.5 * model.log_var_d
        )

        # 2. Physics loss vi phân trường Maxwell (autograd)
        sample_phys = []
        for i in range(len(X_train)):
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
        history["clf_loss"].append(float(loss_clf.item()))
        history["reg_loss"].append(float(avg_reg.item()))

        if (ep + 1) % 300 == 0 or ep == 0:
            print(f"  [Epoch {ep+1:04d}/{epochs}] Total: {total_loss.item():.4f} | Data: {data_loss.item():.4f} | Phys (alpha={alpha}): {batch_phys.item():.4f}", flush=True)

    model.eval()
    return model, history

def main():
    set_reproducible_seed(42)
    print("=" * 90, flush=True)
    print("KHẢO SÁT TỐC ĐỘ HỌC (LR = 10^-4, 10^-6, 10^-7) CHO MÔ HÌNH CNN_PROPOSED", flush=True)
    print("HÀM LOSS: DATA LOSS + ALPHA_PINN * PHYS_LOSS (ALPHA LOAD TỪ PRETRAINED)", flush=True)
    print(f"Thiết bị: {DEVICE} | Số Epochs: 1500 (KHÔNG CHẠY PHẦN MỞ RỘNG)", flush=True)
    print("=" * 90, flush=True)

    # 1. Nạp mô hình tiền huấn luyện CNN_Proposed
    bundle = load_5khz_fully_prepared("cnn", variant="pinn", device=DEVICE)
    base_model = bundle["model"]
    y_scaler = bundle["y_scaler"]
    ckpt_dir = bundle["checkpoint_dir"]
    alpha = extract_pinn_alpha_from_checkpoint(ckpt_dir, variant="pinn")
    print(f"-> Thư mục Checkpoint: {os.path.basename(ckpt_dir)}", flush=True)
    print(f"-> Alpha PINN nạp từ pretrain: {alpha}", flush=True)

    # In giá trị tham số bất định Kendall ban đầu
    print(f"-> Kendall ban đầu: s_clf = {base_model.log_var_clf.item():.4f} (hệ số nhân = {np.exp(-base_model.log_var_clf.item()):.2f}x)", flush=True)

    shape_map = get_shape_physics_map()

    # 2. Dữ liệu Protocol 1 (Scan 1 -> Scan 2)
    train_set, test_set = split_5khz_scan1_scan2(bundle)
    X_tr = train_set["X"].to(DEVICE)
    y_clf_tr = train_set["y_clf"].to(DEVICE)
    y_wld_tr = train_set["y_wld_norm"].to(DEVICE)
    meta_tr = train_set["metadata"]

    X_te = test_set["X"].to(DEVICE)
    meta_te = test_set["metadata"]
    true_sh_te = [m["true_shape"] for m in meta_te]
    true_wld_te = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_te])

    # 3 CHẾ ĐỘ LR YÊU CẦU DUY NHẤT:
    lr_list = [1e-4, 1e-6, 1e-7]
    configs = [
        ("CNN_Proposed (LR = 10^-4)", 1e-4),
        ("CNN_Proposed (LR = 10^-6)", 1e-6),
        ("CNN_Proposed (LR = 10^-7)", 1e-7),
    ]

    results = []
    histories = {}

    for name, lr in configs:
        print(f"\n>>> [Bắt đầu chạy] {name} | LR = {lr:.1e} | Epochs = 1500 ...", flush=True)
        t0 = time.time()
        trained_model, history = train_single_pinn(
            base_model=base_model,
            X_train=X_tr,
            y_clf_train=y_clf_tr,
            y_wld_train=y_wld_tr,
            metadata_train=meta_tr,
            y_scaler=y_scaler,
            shape_map=shape_map,
            alpha=alpha,
            lr=lr,
            epochs=1500,
            freeze_backbone=True,
            freeze_bn=True,
            reset_kendall=False,
        )
        t_el = time.time() - t0

        p_sh, p_wld, _, _ = predict_and_denormalize(trained_model, X_te, y_scaler=y_scaler)
        stat = compute_classification_metrics_and_matrix(true_sh_te, p_sh)
        reg = compute_regression_metrics(true_wld_te, p_wld)

        match_str = "".join(["V" if t == p else "X" for t, p in zip(true_sh_te, p_sh)])
        step_t_status = "Failed"
        for i, m in enumerate(meta_te):
            if m["true_shape"] == "Step_T" and p_sh[i] == "Step_T":
                step_t_status = "Passed"

        row = {
            "Model": "CNN_Proposed",
            "LR": f"{lr:.0e}",
            "Epochs": 1500,
            "Alpha_PINN": alpha,
            "ACC(%)": stat["overall_acc"],
            "Correct": f"{stat['total_correct']}/{stat['total_samples']}",
            "MAE_Total(mm)": round(reg["MAE_Avg"], 4),
            "MAE_W(mm)": round(reg["MAE_W"], 4),
            "MAE_L(mm)": round(reg["MAE_L"], 4),
            "MAE_D(mm)": round(reg["MAE_D"], 4),
            "NMAE(%)": round(reg["NMAE_Avg(%)"], 2),
            "Step_T": step_t_status,
            "Predictions": match_str,
            "Time(s)": round(t_el, 2),
        }
        results.append(row)
        histories[name] = history
        print(f"  -> Hoàn thành trong {t_el:.1f}s | ACC = {stat['overall_acc']:.1f}% ({stat['total_correct']}/{stat['total_samples']}) | MAE = {reg['MAE_Avg']:.4f} mm | NMAE = {reg['NMAE_Avg(%)']:.2f}% | Step_T: {step_t_status}", flush=True)

    df = pd.DataFrame(results)

    # 3. Lưu kết quả ra thư mục chuẩn
    out_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "finetune_results_raw", "protocols", "tables")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "requested_lr_physics_comparison.csv")
    md_file = os.path.join(out_dir, "requested_lr_physics_comparison.md")
    df.to_csv(csv_file, index=False)

    # Lưu toàn bộ lịch sử Loss 1500 epochs ra file CSV độc lập
    loss_df_dict = {"epoch": list(range(1, epochs + 1))}
    for name, lr in configs:
        prefix = f"lr_{lr:.0e}"
        h = histories[name]
        loss_df_dict[f"{prefix}_total_loss"] = h["total_loss"]
        loss_df_dict[f"{prefix}_data_loss"] = h["data_loss"]
        loss_df_dict[f"{prefix}_phys_loss"] = h["phys_loss"]
    loss_history_df = pd.DataFrame(loss_df_dict)
    loss_history_csv = os.path.join(out_dir, "loss_history_freeze_bn_constant_lr.csv")
    loss_history_df.to_csv(loss_history_csv, index=False)
    print(f"[OK] Đã lưu toàn bộ lịch sử 1500 epochs Loss vào: {loss_history_csv}", flush=True)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐỐI CHỨNG TỐC ĐỘ HỌC (LR = 10^-4, 10^-6, 10^-7)\n")
        f.write(f"Mô hình: CNN_Proposed | Hàm Loss: L_data + alpha * L_phys (alpha = {alpha} load từ pretrain)\n")
        f.write("Thiết lập: Freeze Backbone = True, Freeze BatchNorm = True, Constant LR (No CosineAnnealingLR)\n\n")
        # Xuất markdown table thủ công không cần thư viện tabulate
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join([":---:" if c not in ["Model", "Predictions"] else ":---" for c in cols]) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join([str(val) for val in row.values]) + " |\n")
        f.write("\n")

    print(f"\n[OK] Đã lưu bảng số liệu tại:\n  - {csv_file}\n  - {md_file}", flush=True)

    # 4. Vẽ đồ thị IEEE
    fig_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "finetune_results_raw", "protocols", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 11

    # Đồ thị 1: Biểu đồ hội tụ Loss IEEE (3 panels, không title, legend ngoài trên đỉnh)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=300)
    colors = {"CNN_Proposed (LR = 10^-4)": "#d62728", "CNN_Proposed (LR = 10^-6)": "#1f77b4", "CNN_Proposed (LR = 10^-7)": "#2ca02c"}
    labels = {"CNN_Proposed (LR = 10^-4)": r"LR = $10^{-4}$ (Baseline)", "CNN_Proposed (LR = 10^-6)": r"LR = $10^{-6}$", "CNN_Proposed (LR = 10^-7)": r"LR = $10^{-7}$"}

    lines = []
    for name in configs:
        k = name[0]
        h = histories[k]
        ep = h["epoch"]
        l, = axes[0].plot(ep, h["total_loss"], label=labels[k], color=colors[k], linewidth=2.0)
        axes[1].plot(ep, h["data_loss"], label=labels[k], color=colors[k], linewidth=2.0)
        axes[2].plot(ep, h["phys_loss"], label=labels[k], color=colors[k], linewidth=2.0)
        lines.append(l)

    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel(r"Total Loss ($\mathcal{L}_{\mathrm{total}}$)", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel(r"Data Loss ($\mathcal{L}_{\mathrm{data}}$)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].set_xlabel("Epoch", fontweight="bold")
    axes[2].set_ylabel(r"Physics Loss ($\mathcal{L}_{\mathrm{phys}}$)", fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.legend(
        lines,
        [labels[c[0]] for c in configs],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=True,
        fontsize=12,
        edgecolor="#999999",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_png = os.path.join(fig_dir, "fig_ieee_lr_physics_loss_convergence.png")
    fig1_pdf = os.path.join(fig_dir, "fig_ieee_lr_physics_loss_convergence.pdf")
    plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.close()

    # Đồ thị 2: Biểu đồ so sánh ACC và Sai số MAE W, L, D (Xóa Total MAE, không title, legend ngoài trên đỉnh)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=300)
    x = np.arange(len(lr_list))
    lr_labels = [r"LR = $10^{-4}$", r"LR = $10^{-6}$", r"LR = $10^{-7}$"]
    acc_vals = df["ACC(%)"].values
    mae_w_vals = df["MAE_W(mm)"].values
    mae_l_vals = df["MAE_L(mm)"].values
    mae_d_vals = df["MAE_D(mm)"].values

    b_acc = ax1.bar(x, acc_vals, width=0.45, color=["#d62728", "#1f77b4", "#2ca02c"], edgecolor="black", linewidth=1.2)
    for b, a in zip(b_acc, acc_vals):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, f"{a:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax1.set_ylim(0, max(60, float(np.max(acc_vals)) + 15))
    ax1.set_xticks(x)
    ax1.set_xticklabels(lr_labels, fontweight="bold")
    ax1.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    w = 0.22
    b_w = ax2.bar(x - w, mae_w_vals, width=w, label="Width ($W$)", color="#2ca02c", edgecolor="black")
    b_l = ax2.bar(x,     mae_l_vals, width=w, label="Length ($L$)", color="#ff7f0e", edgecolor="black")
    b_d = ax2.bar(x + w, mae_d_vals, width=w, label="Depth ($D$)", color="#d62728", edgecolor="black")

    for i in range(len(lr_labels)):
        ax2.text(x[i] - w, mae_w_vals[i] + 0.04, f"{mae_w_vals[i]:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        ax2.text(x[i],     mae_l_vals[i] + 0.04, f"{mae_l_vals[i]:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        ax2.text(x[i] + w, mae_d_vals[i] + 0.04, f"{mae_d_vals[i]:.2f}", ha="center", fontsize=9.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(lr_labels, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Error (mm)", fontweight="bold")
    max_mae = max(float(np.max([mae_w_vals, mae_l_vals, mae_d_vals])), 2.0)
    ax2.set_ylim(0, max_mae * 1.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=True,
        fontsize=11,
        edgecolor="#999999",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_png = os.path.join(fig_dir, "fig_ieee_lr_physics_metrics_comparison.png")
    fig2_pdf = os.path.join(fig_dir, "fig_ieee_lr_physics_metrics_comparison.pdf")
    plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.close()

    # Sao chép vào thư mục brain artifacts để hiển thị
    artifact_dir = r"C:\Users\Admin\.gemini\antigravity-ide\brain\f10e717b-0d2f-4d04-a176-aeaa228e62c7"
    import shutil
    try:
        shutil.copy2(fig1_png, os.path.join(artifact_dir, "fig_ieee_lr_physics_loss_convergence.png"))
        shutil.copy2(fig2_png, os.path.join(artifact_dir, "fig_ieee_lr_physics_metrics_comparison.png"))
    except Exception as e:
        print(f"Artifact copy warning: {e}")

    print(f"[OK] Đã vẽ và xuất đồ thị chuẩn IEEE (PNG 300 DPI + PDF) tại:\n  - {fig1_png}\n  - {fig2_png}", flush=True)

if __name__ == "__main__":
    main()
