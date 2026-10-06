import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import load_5khz_fully_prepared, predict_and_denormalize
from domain_adaptation.data_loader import split_5khz_scan1_scan2
from domain_adaptation.cnn import compute_physics_loss_autograd, get_shape_physics_map
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
    DEFAULT_UNIQUE_SHAPES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "pinn_active_physics_finetune")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD ALPHA FROM CHECKPOINT
# -----------------------------------------------------------------------------
def get_checkpoint_alpha(ckpt_dir):
    config_file = os.path.join(ckpt_dir, "reeval_model_config.txt")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            for line in f:
                if "pinn_alpha:" in line:
                    return float(line.split("pinn_alpha:")[1].strip())
    # fallback to csv
    csv_file = os.path.join(ckpt_dir, "reeval_model_config.csv")
    if os.path.exists(csv_file):
        df_cfg = pd.read_csv(csv_file)
        if "pinn_alpha" in df_cfg.columns:
            return float(df_cfg["pinn_alpha"].iloc[0])
    return 1.0

# -----------------------------------------------------------------------------
# 2. TRAINING FUNCTION WITH ACTIVE PINN LOSS
# -----------------------------------------------------------------------------
def train_with_active_pinn(
    base_model,
    X_train,
    y_clf_train,
    y_wld_train,
    metadata_train,
    y_scaler,
    shape_map,
    alpha=1.0,
    include_physics_loss=True,
    lr=2e-4,
    epochs=1500,
    freeze_bn=True,
):
    model = copy.deepcopy(base_model).to(DEVICE)

    # Reset Kendall parameters to 0.0
    for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
            getattr(model, attr).data.fill_(0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss_history = {"total": [], "data": [], "physics": []}

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)

        # 1. Data loss with Kendall uncertainty
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

        # 2. Physics loss (Forward Maxwell surrogate)
        if include_physics_loss and shape_map is not None:
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
            phys_val = batch_phys_loss.item()
        else:
            total_loss = data_loss
            phys_val = 0.0

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history["total"].append(total_loss.item())
        loss_history["data"].append(data_loss.item())
        loss_history["physics"].append(phys_val)

        if (ep + 1) % 300 == 0 or ep == 0:
            print(f"Epoch {ep+1:4d}/{epochs} | Total: {total_loss.item():.4f} | Data: {data_loss.item():.4f} | Phys (alpha={alpha}): {phys_val:.4f}")

    model.eval()
    return model, loss_history

# -----------------------------------------------------------------------------
# 3. RUN PROTOCOL 1 (SCAN 1 -> SCAN 2)
# -----------------------------------------------------------------------------
def run_protocol_1_experiment():
    set_reproducible_seed(42)
    bundle_pinn = load_5khz_fully_prepared("cnn", variant="pinn", device=DEVICE)
    ckpt_dir = bundle_pinn["checkpoint_dir"]
    alpha = get_checkpoint_alpha(ckpt_dir)
    print(f"\n[INFO] Loaded alpha = {alpha} from checkpoint: {os.path.basename(ckpt_dir)}")

    train_set, test_set = split_5khz_scan1_scan2(bundle_pinn)
    X_tr = train_set["X"].to(DEVICE)
    y_clf_tr = train_set["y_clf"].to(DEVICE)
    y_wld_tr = train_set["y_wld_norm"].to(DEVICE)
    meta_tr = train_set["metadata"]

    X_te = test_set["X"].to(DEVICE)
    meta_te = test_set["metadata"]
    true_sh_te = [m["true_shape"] for m in meta_te]
    true_wld_te = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_te])

    y_scaler = bundle_pinn["y_scaler"]
    shape_map = get_shape_physics_map()

    # Train Case A: PINN with Active Physics Loss: Loss = Loss_data + alpha * Loss_physics
    print("\n" + "="*85)
    print(f"TRAINING PINN WITH ACTIVE PHYSICS LOSS (Loss = Loss_data + {alpha} * Loss_physics)")
    print("="*85)
    m_pinn_active, hist_active = train_with_active_pinn(
        bundle_pinn["model"], X_tr, y_clf_tr, y_wld_tr, meta_tr,
        y_scaler=y_scaler, shape_map=shape_map, alpha=alpha, include_physics_loss=True,
        lr=2e-4, epochs=1500, freeze_bn=True
    )
    p_sh_act, p_wld_act, _, _ = predict_and_denormalize(m_pinn_active, X_te, y_scaler=y_scaler)
    stat_act = compute_classification_metrics_and_matrix(true_sh_te, p_sh_act)
    reg_act = compute_regression_metrics(true_wld_te, p_wld_act)

    # Train Case B: PINN Data Loss Only (Reference from previous run)
    print("\n" + "="*85)
    print("TRAINING PINN WITHOUT ACTIVE PHYSICS LOSS (Loss = Loss_data only)")
    print("="*85)
    m_pinn_data, hist_data = train_with_active_pinn(
        bundle_pinn["model"], X_tr, y_clf_tr, y_wld_tr, meta_tr,
        y_scaler=y_scaler, shape_map=shape_map, alpha=alpha, include_physics_loss=False,
        lr=2e-4, epochs=1500, freeze_bn=True
    )
    p_sh_dat, p_wld_dat, _, _ = predict_and_denormalize(m_pinn_data, X_te, y_scaler=y_scaler)
    stat_dat = compute_classification_metrics_and_matrix(true_sh_te, p_sh_dat)
    reg_dat = compute_regression_metrics(true_wld_te, p_wld_dat)

    # Also load NoPINN for reference
    bundle_nopinn = load_5khz_fully_prepared("cnn", variant="nopinn", device=DEVICE)
    m_nopinn, _ = train_with_active_pinn(
        bundle_nopinn["model"], X_tr, y_clf_tr, y_wld_tr, meta_tr,
        y_scaler=y_scaler, shape_map=None, alpha=0.0, include_physics_loss=False,
        lr=2e-4, epochs=1500, freeze_bn=True
    )
    p_sh_no, p_wld_no, _, _ = predict_and_denormalize(m_nopinn, X_te, y_scaler=y_scaler)
    stat_no = compute_classification_metrics_and_matrix(true_sh_te, p_sh_no)
    reg_no = compute_regression_metrics(true_wld_te, p_wld_no)

    print("\n" + "="*95)
    print("PROTOCOL 1 BENCHMARK RESULTS (Scan 1 -> Scan 2)")
    print("="*95)
    print(f"1. NoPINN                      | ACC: {stat_no['overall_acc']:5.1f}% ({stat_no['total_correct']}/10) | MAE: {reg_no['MAE_Avg']:.4f}mm (W:{reg_no['MAE_W']:.3f} L:{reg_no['MAE_L']:.3f} D:{reg_no['MAE_D']:.3f}) | NMAE: {reg_no['NMAE_Avg(%)']:5.2f}%")
    print(f"2. PINN (Data Loss Only)       | ACC: {stat_dat['overall_acc']:5.1f}% ({stat_dat['total_correct']}/10) | MAE: {reg_dat['MAE_Avg']:.4f}mm (W:{reg_dat['MAE_W']:.3f} L:{reg_dat['MAE_L']:.3f} D:{reg_dat['MAE_D']:.3f}) | NMAE: {reg_dat['NMAE_Avg(%)']:5.2f}%")
    print(f"3. PINN (Loss = Data + a*Phys) | ACC: {stat_act['overall_acc']:5.1f}% ({stat_act['total_correct']}/10) | MAE: {reg_act['MAE_Avg']:.4f}mm (W:{reg_act['MAE_W']:.3f} L:{reg_act['MAE_L']:.3f} D:{reg_act['MAE_D']:.3f}) | NMAE: {reg_act['NMAE_Avg(%)']:5.2f}%")

    # Save to CSV
    df_p1 = pd.DataFrame([
        {
            "Model": "NoPINN",
            "Loss_Function": "Data Loss",
            "Alpha_Pinn": 0.0,
            "Accuracy(%)": stat_no['overall_acc'],
            "Correct": f"{stat_no['total_correct']}/10",
            "Total_MAE(mm)": reg_no['MAE_Avg'],
            "MAE_W(mm)": reg_no['MAE_W'],
            "MAE_L(mm)": reg_no['MAE_L'],
            "MAE_D(mm)": reg_no['MAE_D'],
            "NMAE(%)": reg_no['NMAE_Avg(%)'],
            "Step_T_Prediction": p_sh_no[0],
            "Step_T_Status": "CORRECT" if p_sh_no[0] == "Step_T" else "WRONG",
        },
        {
            "Model": "Proposed PINN (Data Loss)",
            "Loss_Function": "Data Loss (Pretrained Weights)",
            "Alpha_Pinn": 0.0,
            "Accuracy(%)": stat_dat['overall_acc'],
            "Correct": f"{stat_dat['total_correct']}/10",
            "Total_MAE(mm)": reg_dat['MAE_Avg'],
            "MAE_W(mm)": reg_dat['MAE_W'],
            "MAE_L(mm)": reg_dat['MAE_L'],
            "MAE_D(mm)": reg_dat['MAE_D'],
            "NMAE(%)": reg_dat['NMAE_Avg(%)'],
            "Step_T_Prediction": p_sh_dat[0],
            "Step_T_Status": "CORRECT" if p_sh_dat[0] == "Step_T" else "WRONG",
        },
        {
            "Model": "Proposed PINN (Active Physics)",
            "Loss_Function": f"Data Loss + {alpha} * Physics Loss",
            "Alpha_Pinn": alpha,
            "Accuracy(%)": stat_act['overall_acc'],
            "Correct": f"{stat_act['total_correct']}/10",
            "Total_MAE(mm)": reg_act['MAE_Avg'],
            "MAE_W(mm)": reg_act['MAE_W'],
            "MAE_L(mm)": reg_act['MAE_L'],
            "MAE_D(mm)": reg_act['MAE_D'],
            "NMAE(%)": reg_act['NMAE_Avg(%)'],
            "Step_T_Prediction": p_sh_act[0],
            "Step_T_Status": "CORRECT" if p_sh_act[0] == "Step_T" else "WRONG",
        },
    ])
    df_p1.to_csv(os.path.join(TABLES_DIR, "table_protocol_1_active_physics.csv"), index=False)

    # -------------------------------------------------------------------------
    # 4. PLOT TRAINING CURVE & COMPARISON FIGURES (STRICT IEEE STYLE)
    # -------------------------------------------------------------------------
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    # Plot (1): Training Loss Curves for Active PINN
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(hist_active["total"], label=r"Total Loss ($\mathcal{L}_{\mathrm{total}}$)", color="#E74C3C", linewidth=1.5)
    ax.plot(hist_active["data"], label=r"Data Loss ($\mathcal{L}_{\mathrm{data}}$)", color="#3498DB", linewidth=1.2, linestyle="--")
    ax.plot(hist_active["physics"], label=rf"Physics Loss ($\mathcal{{L}}_{{\mathrm{{phys}}}}$, $\alpha={alpha}$)", color="#2ECC71", linewidth=1.2, linestyle=":")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss Value")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, edgecolor="none")
    plt.tight_layout()
    fig1_png = os.path.join(FIGURES_DIR, "fig1_active_pinn_loss_convergence.png")
    fig1_pdf = os.path.join(FIGURES_DIR, "fig1_active_pinn_loss_convergence.pdf")
    plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.close()

    # Plot (2): Sizing Comparison MAE (W, L, D)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    dims = ["Width (W)", "Length (L)", "Depth (D)"]
    x = np.arange(len(dims))
    w = 0.25

    ax.bar(x - w, [reg_no["MAE_W"], reg_no["MAE_L"], reg_no["MAE_D"]], w, label="NoPINN", color="#4A90E2", edgecolor="black", linewidth=0.8)
    ax.bar(x, [reg_dat["MAE_W"], reg_dat["MAE_L"], reg_dat["MAE_D"]], w, label="Proposed PINN (Data Loss)", color="#95A5A6", edgecolor="black", linewidth=0.8)
    ax.bar(x + w, [reg_act["MAE_W"], reg_act["MAE_L"], reg_act["MAE_D"]], w, label=f"Proposed PINN (Active Phys, $\\alpha={alpha}$)", color="#2ECC71", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Mean Absolute Error (mm)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontweight="bold")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, edgecolor="none")

    # Labels
    for i in range(len(dims)):
        v_no = [reg_no["MAE_W"], reg_no["MAE_L"], reg_no["MAE_D"]][i]
        v_dat = [reg_dat["MAE_W"], reg_dat["MAE_L"], reg_dat["MAE_D"]][i]
        v_act = [reg_act["MAE_W"], reg_act["MAE_L"], reg_act["MAE_D"]][i]
        ax.text(i - w, v_no + 0.012, f"{v_no:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(i, v_dat + 0.012, f"{v_dat:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(i + w, v_act + 0.012, f"{v_act:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig2_png = os.path.join(FIGURES_DIR, "fig2_active_pinn_wld_mae.png")
    fig2_pdf = os.path.join(FIGURES_DIR, "fig2_active_pinn_wld_mae.pdf")
    plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Completed Protocol 1 run and generated figures at: {FIGURES_DIR}")
    return df_p1

if __name__ == "__main__":
    run_protocol_1_experiment()
