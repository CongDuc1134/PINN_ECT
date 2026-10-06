import os
import sys
import copy
import time
import re
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

from domain_adaptation.model_loader import load_5khz_fully_prepared, predict_and_denormalize, extract_pinn_alpha_from_checkpoint
from domain_adaptation.data_loader import split_5khz_scan1_scan2
from domain_adaptation.cnn.losses import compute_physics_loss_autograd, get_shape_physics_map
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
    DEFAULT_UNIQUE_SHAPES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_active_pinn_model(
    base_model,
    X_train,
    y_clf_train,
    y_wld_train,
    metadata_train,
    y_scaler,
    shape_map,
    alpha=1.0,
    epochs=1500,
    lr=1e-6,
    freeze_backbone=False,
    freeze_bn=True,
    reset_kendall=True,
):
    model = copy.deepcopy(base_model).to(DEVICE)

    if reset_kendall:
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).data.fill_(0.0)

    if freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    eta_min = min(lr * 0.05, 1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    history = {"epoch": [], "total_loss": [], "data_loss": [], "phys_loss": []}

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)

        # 1. Data loss (Kendall uncertainty weighting)
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

        # 2. Physics loss (Forward Maxwell analytical surrogate via autograd)
        sample_phys = []
        for i in range(len(X_train)):
            s_name = metadata_train[i]["true_shape"] if metadata_train else DEFAULT_UNIQUE_SHAPES[int(y_clf_train[i].item())]
            p_l = compute_physics_loss_autograd(
                H_true=X_train[i, 0],
                w_pred=pred_wld[i, 0],
                l_pred=pred_wld[i, 1],
                d_pred=pred_wld[i, 2],
                shape_name=s_name,
                shape_map=shape_map,
                y_scaler=y_scaler,
            )
            sample_phys.append(p_l)
        batch_phys_loss = torch.stack(sample_phys).mean()

        total_loss = data_loss + alpha * batch_phys_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        history["epoch"].append(ep + 1)
        history["total_loss"].append(total_loss.item())
        history["data_loss"].append(data_loss.item())
        history["phys_loss"].append(batch_phys_loss.item())

        if (ep + 1) % 300 == 0 or ep == 0:
            print(f"    [Ep {ep+1:04d}/{epochs}] Total: {total_loss.item():.4f} | Data: {data_loss.item():.4f} | Phys (alpha={alpha}): {batch_phys_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    return model, history

def main():
    set_reproducible_seed(42)
    print("=" * 100)
    print("THỰC NGHIỆM KHẢO SÁT TỐC ĐỘ HỌC (LR = 10^-4, 10^-6, 10^-7) CHO MÔ HÌNH CNN_PROPOSED")
    print("VỚI HÀM LOSS = DATA LOSS + ALPHA_PINN * PHYS_LOSS (ALPHA LOAD TỪ PRETRAINED)")
    print(f"Thiết bị: {DEVICE} | Số Epochs: 1500")
    print("=" * 100)

    # Nạp mô hình tiền huấn luyện
    bundle = load_5khz_fully_prepared("cnn", variant="pinn", device=DEVICE)
    base_model = bundle["model"]
    y_scaler = bundle["y_scaler"]
    ckpt_dir = bundle["checkpoint_dir"]
    alpha_pretrained = extract_pinn_alpha_from_checkpoint(ckpt_dir, variant="pinn")
    print(f"\n-> Loaded Pretrained Alpha = {alpha_pretrained} from: {os.path.basename(ckpt_dir)}")

    shape_map = get_shape_physics_map()

    # Dữ liệu Protocol 1 (Scan 1 -> Scan 2)
    train_set, test_set = split_5khz_scan1_scan2(bundle)
    X_tr = train_set["X"].to(DEVICE)
    y_clf_tr = train_set["y_clf"].to(DEVICE)
    y_wld_tr = train_set["y_wld_norm"].to(DEVICE)
    meta_tr = train_set["metadata"]

    X_te = test_set["X"].to(DEVICE)
    meta_te = test_set["metadata"]
    true_sh_te = [m["true_shape"] for m in meta_te]
    true_wld_te = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_te])

    # Danh sách cấu hình thử nghiệm
    experiments = [
        # (name, lr, freeze_bn, reset_kendall, freeze_bb)
        # 1. Cấu hình Freeze BN = True, Reset Kendall = True (Cấu hình chuẩn tối ưu của phương pháp đề xuất)
        ("Proposed PINN (Active Physics, lr=1e-4, Freeze BN=True)", 1e-4, True, True, False),
        ("Proposed PINN (Active Physics, lr=1e-6, Freeze BN=True)", 1e-6, True, True, False),
        ("Proposed PINN (Active Physics, lr=1e-7, Freeze BN=True)", 1e-7, True, True, False),
        
        # 2. Cấu hình Freeze BN = False, Reset Kendall = False (Giữ nguyên cấu hình thô như bản Raw Baseline)
        ("Proposed PINN (Active Physics, lr=1e-4, Raw Style: BN=False, Kendall=Raw)", 1e-4, False, False, True),
        ("Proposed PINN (Active Physics, lr=1e-6, Raw Style: BN=False, Kendall=Raw)", 1e-6, False, False, True),
        ("Proposed PINN (Active Physics, lr=1e-7, Raw Style: BN=False, Kendall=Raw)", 1e-7, False, False, True),

        # 3. Cấu hình Freeze BN = False, Reset Kendall = True
        ("Proposed PINN (Active Physics, lr=1e-6, BN=False, Reset Kendall=True)", 1e-6, False, True, False),
        ("Proposed PINN (Active Physics, lr=1e-7, BN=False, Reset Kendall=True)", 1e-7, False, True, False),
    ]

    results = []
    loss_histories = {}

    for idx, (exp_name, lr, freeze_bn, reset_kendall, freeze_bb) in enumerate(experiments):
        print(f"\n[{idx+1}/{len(experiments)}] Running: {exp_name}")
        t0 = time.time()
        trained_model, history = train_active_pinn_model(
            base_model=base_model,
            X_train=X_tr,
            y_clf_train=y_clf_tr,
            y_wld_train=y_wld_tr,
            metadata_train=meta_tr,
            y_scaler=y_scaler,
            shape_map=shape_map,
            alpha=alpha_pretrained,
            epochs=1500,
            lr=lr,
            freeze_backbone=freeze_bb,
            freeze_bn=freeze_bn,
            reset_kendall=reset_kendall,
        )
        t_el = time.time() - t0

        # Dự đoán trên tập kiểm thử (Scan 2)
        p_sh, p_wld, _, _ = predict_and_denormalize(trained_model, X_te, y_scaler=y_scaler)
        stat = compute_classification_metrics_and_matrix(true_sh_te, p_sh)
        reg = compute_regression_metrics(true_wld_te, p_wld)

        match_str = "".join(["V" if t == p else "X" for t, p in zip(true_sh_te, p_sh)])
        
        # Check Step_T accuracy
        step_t_status = "0% (Failed)"
        for i, m in enumerate(meta_te):
            if m["true_shape"] == "Step_T":
                if p_sh[i] == "Step_T":
                    step_t_status = "100% (Passed)"

        row = {
            "Experiment": exp_name,
            "LR": lr,
            "Epochs": 1500,
            "Alpha_PINN": alpha_pretrained,
            "Freeze_BN": freeze_bn,
            "Reset_Kendall": reset_kendall,
            "Freeze_Backbone": freeze_bb,
            "ACC(%)": stat["acc_pct"],
            "Correct": f"{stat['correct']}/{stat['total']}",
            "MAE_Total(mm)": round(reg["overall_mae"], 4),
            "MAE_W(mm)": round(reg["mae_w"], 4),
            "MAE_L(mm)": round(reg["mae_l"], 4),
            "MAE_D(mm)": round(reg["mae_d"], 4),
            "NMAE(%)": round(reg["overall_nmae_pct"], 2),
            "Matches": match_str,
            "Step_T": step_t_status,
            "Time(s)": round(t_el, 2)
        }
        results.append(row)
        loss_histories[exp_name] = history
        print(f"    -> Result: ACC={stat['acc_pct']:.1f}% ({stat['correct']}/{stat['total']}) | MAE={reg['overall_mae']:.4f}mm (W={reg['mae_w']:.4f}, L={reg['mae_l']:.4f}, D={reg['mae_d']:.4f}) | NMAE={reg['overall_nmae_pct']:.2f}% | Step_T: {step_t_status} | Time: {t_el:.1f}s")

    df_res = pd.DataFrame(results)

    # Lưu kết quả
    out_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "pinn_active_physics_finetune", "tables")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "requested_active_pinn_lr_study.csv")
    md_path = os.path.join(out_dir, "requested_active_pinn_lr_study.md")
    df_res.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO THỰC NGHIỆM: KHẢO SÁT TỐC ĐỘ HỌC (LR = 10^-4, 10^-6, 10^-7) CHO MÔ HÌNH CNN_PROPOSED\n")
        f.write("### HÀM LOSS: $\\mathcal{L} = \\mathcal{L}_{\\mathrm{data}} + \\alpha \\cdot \\mathcal{L}_{\\mathrm{phys}}$ ($\\alpha$ load từ pretrain = 1.0)\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n")

    print(f"\n[OK] Đã lưu kết quả tại:\n  - CSV: {csv_path}\n  - MD: {md_path}")

    # Vẽ biểu đồ IEEE
    fig_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "pinn_active_physics_finetune", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 11

    # 1. Biểu đồ đường cong Loss theo LR
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Target 3 curves for Freeze BN = True
    key_configs = [
        ("Proposed PINN (Active Physics, lr=1e-4, Freeze BN=True)", "LR = 1e-4 (Standard)", "#d62728", "-"),
        ("Proposed PINN (Active Physics, lr=1e-6, Freeze BN=True)", "LR = 1e-6 (Ultra-low)", "#1f77b4", "-"),
        ("Proposed PINN (Active Physics, lr=1e-7, Freeze BN=True)", "LR = 1e-7 (Extremely-low)", "#2ca02c", "-"),
    ]

    for cfg_name, label, col, ls in key_configs:
        if cfg_name in loss_histories:
            h = loss_histories[cfg_name]
            ep = h["epoch"]
            axes[0].plot(ep, h["total_loss"], label=label, color=col, linestyle=ls, linewidth=1.8)
            axes[1].plot(ep, h["data_loss"], label=label, color=col, linestyle=ls, linewidth=1.8)
            axes[2].plot(ep, h["phys_loss"], label=label, color=col, linestyle=ls, linewidth=1.8)

    axes[0].set_title("(a) Total Loss ($\mathcal{L}_{\\mathrm{total}}$)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss Value (Log-scale)")
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(frameon=True)

    axes[1].set_title("(b) Data Loss ($\mathcal{L}_{\\mathrm{data}}$)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Data Loss (Log-scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(frameon=True)

    axes[2].set_title(f"(c) Maxwell Physics Loss ($\\alpha={alpha_pretrained}$)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Physics Loss")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(frameon=True)

    plt.tight_layout()
    loss_png = os.path.join(fig_dir, "fig6_active_pinn_lr_1e4_1e6_1e7_loss_curves.png")
    loss_pdf = os.path.join(fig_dir, "fig6_active_pinn_lr_1e4_1e6_1e7_loss_curves.pdf")
    plt.savefig(loss_png, dpi=300, bbox_inches="tight")
    plt.savefig(loss_pdf, bbox_inches="tight")
    plt.close()

    # 2. Biểu đồ cột so sánh ACC và MAE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    df_plot = df_res[df_res["Freeze_BN"] == True].copy()
    labels = ["LR $10^{-4}$", "LR $10^{-6}$", "LR $10^{-7}$"]
    accs = df_plot["ACC(%)"].values
    maes = df_plot["MAE_Total(mm)"].values
    mae_w = df_plot["MAE_W(mm)"].values
    mae_l = df_plot["MAE_L(mm)"].values
    mae_d = df_plot["MAE_D(mm)"].values

    x = np.arange(len(labels))
    bars = ax1.bar(x, accs, width=0.45, color="#1f77b4", edgecolor="black", linewidth=1.2)
    for b, a in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{a:.1f}%", ha="center", fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax1.set_title("(a) Defect Shape Classification Accuracy", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    w = 0.2
    ax2.bar(x - 1.5*w, maes, width=w, label="Total MAE", color="#333333", edgecolor="black")
    ax2.bar(x - 0.5*w, mae_w, width=w, label="Width ($W$)", color="#2ca02c", edgecolor="black")
    ax2.bar(x + 0.5*w, mae_l, width=w, label="Length ($L$)", color="#ff7f0e", edgecolor="black")
    ax2.bar(x + 1.5*w, mae_d, width=w, label="Depth ($D$)", color="#d62728", edgecolor="black")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Error (mm)", fontweight="bold")
    ax2.set_title("(b) 3D Sizing Regression MAE Breakdown", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    bar_png = os.path.join(fig_dir, "fig7_active_pinn_lr_1e4_1e6_1e7_metrics_bar.png")
    bar_pdf = os.path.join(fig_dir, "fig7_active_pinn_lr_1e4_1e6_1e7_metrics_bar.pdf")
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.savefig(bar_pdf, bbox_inches="tight")
    plt.close()

    print(f"[OK] Đã vẽ và lưu 2 biểu đồ IEEE tại:\n  - {loss_png}\n  - {bar_png}")

if __name__ == "__main__":
    main()
