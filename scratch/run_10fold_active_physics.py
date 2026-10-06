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
from domain_adaptation.data_loader import get_5khz_10fold_defect_splits, split_5khz_scan1_scan2
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

def train_one_fold(
    base_model,
    X_train,
    y_clf_train,
    y_wld_train,
    metadata_train,
    y_scaler,
    shape_map,
    alpha=1.0,
    include_physics=True,
    lr=2e-4,
    epochs=500,
    freeze_bn=True,
):
    model = copy.deepcopy(base_model).to(DEVICE)
    for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d", "log_var_reg"]:
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
            getattr(model, attr).data.fill_(0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)

        # Kendall uncertainty data loss
        if hasattr(model, "log_var_clf") and hasattr(model, "log_var_w"):
            w_clf = torch.exp(-model.log_var_clf)
            w_w   = torch.exp(-model.log_var_w)
            w_l   = torch.exp(-model.log_var_l)
            w_d   = torch.exp(-model.log_var_d)
            loss_clf = F.cross_entropy(logits, y_clf_train)
            data_loss = (
                0.5 * w_clf * loss_clf + 0.5 * model.log_var_clf +
                0.5 * w_w * F.mse_loss(pred_wld[:, 0], y_wld_train[:, 0]) + 0.5 * model.log_var_w +
                0.5 * w_l * F.mse_loss(pred_wld[:, 1], y_wld_train[:, 1]) + 0.5 * model.log_var_l +
                0.5 * w_d * F.mse_loss(pred_wld[:, 2], y_wld_train[:, 2]) + 0.5 * model.log_var_d
            )
        else:
            loss_clf = F.cross_entropy(logits, y_clf_train)
            loss_reg = F.mse_loss(pred_wld, y_wld_train)
            data_loss = loss_clf + loss_reg

        # Physics loss
        if include_physics and shape_map is not None:
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
            total_loss = data_loss + alpha * torch.stack(sample_phys).mean()
        else:
            total_loss = data_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model

def run_10fold_evaluation():
    set_reproducible_seed(42)
    bundle_pinn = load_5khz_fully_prepared("cnn", variant="pinn", device=DEVICE)
    base_pinn = bundle_pinn["model"]
    y_scaler = bundle_pinn["y_scaler"]
    metadata = bundle_pinn["metadata"]
    shape_map = get_shape_physics_map()

    bundle_nopinn = load_5khz_fully_prepared("cnn", variant="nopinn", device=DEVICE)
    base_nopinn = bundle_nopinn["model"]

    folds = get_5khz_10fold_defect_splits(bundle_pinn)
    true_sh_all = [m["true_shape"] for m in metadata]
    true_wld_all = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in metadata])

    print("=" * 105)
    print("RUNNING 10-FOLD CROSS-VALIDATION BENCHMARK (N=20 SAMPLES)")
    print("=" * 105)

    scenarios = [
        ("NoPINN", base_nopinn, False, 0.0),
        ("Proposed PINN (Data Loss)", base_pinn, False, 0.0),
        ("Proposed PINN (Active Physics, a=1.0)", base_pinn, True, 1.0),
    ]

    results_10fold = []

    for name, base_m, inc_phys, alpha in scenarios:
        print(f"\n--- Running 10-Fold CV for: {name} ---")
        oof_preds_sh = [None] * len(metadata)
        oof_preds_wld = np.zeros((len(metadata), 3))

        for f_idx, fold_data in enumerate(folds):
            tr = fold_data["train"]
            te = fold_data["test"]
            test_indices = te["indices"]

            X_tr = tr["X"].to(DEVICE)
            y_clf_tr = tr["y_clf"].to(DEVICE)
            y_wld_tr = tr["y_wld_norm"].to(DEVICE)
            meta_tr = [metadata[i] for i in tr["indices"]]

            X_te = te["X"].to(DEVICE)

            m = train_one_fold(
                base_m, X_tr, y_clf_tr, y_wld_tr, meta_tr,
                y_scaler=y_scaler, shape_map=shape_map,
                alpha=alpha, include_physics=inc_phys,
                lr=2e-4, epochs=500, freeze_bn=True
            )

            p_sh, p_wld, _, _ = predict_and_denormalize(m, X_te, y_scaler=y_scaler)
            for idx_local, real_idx in enumerate(test_indices):
                oof_preds_sh[real_idx] = p_sh[idx_local]
                oof_preds_wld[real_idx] = p_wld[idx_local]

            print(f"  Fold {f_idx+1}/10 done: Out-of-fold samples {test_indices} evaluated.")

        stat = compute_classification_metrics_and_matrix(true_sh_all, oof_preds_sh)
        reg = compute_regression_metrics(true_wld_all, oof_preds_wld)

        print(f">> {name:<35s} | 10-Fold ACC: {stat['overall_acc']:5.1f}% ({stat['total_correct']}/20) | "
              f"MAE: {reg['MAE_Avg']:.4f}mm (W:{reg['MAE_W']:.3f} L:{reg['MAE_L']:.3f} D:{reg['MAE_D']:.3f}) | NMAE: {reg['NMAE_Avg(%)']:5.2f}%")

        results_10fold.append({
            "Model": name,
            "Evaluation": "10-Fold CV (N=20)",
            "Active_Physics": inc_phys,
            "Alpha": alpha,
            "ACC(%)": stat["overall_acc"],
            "Correct": f"{stat['total_correct']}/20",
            "MAE_Total(mm)": reg["MAE_Avg"],
            "MAE_W(mm)": reg["MAE_W"],
            "MAE_L(mm)": reg["MAE_L"],
            "MAE_D(mm)": reg["MAE_D"],
            "NMAE(%)": reg["NMAE_Avg(%)"],
        })

    df_10fold = pd.DataFrame(results_10fold)
    df_10fold.to_csv(os.path.join(TABLES_DIR, "table_protocol_2_active_physics_10fold.csv"), index=False)
    print(f"\n[OK] Saved 10-Fold CV table to: {TABLES_DIR}")

    # =========================================================================
    # PLOT ACC COMPARISON (TWO PROTOCOLS: SCAN SPLIT vs 10-FOLD CV)
    # Strict IEEE style: outside legend, no titles on top, clean labels
    # =========================================================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
    })

    # Protocol 1 accuracies:
    # NoPINN: 90.0%, PINN Data: 90.0%, PINN Active Phys: 90.0%
    # Protocol 2 accuracies: from 10-Fold CV
    p1_accs = [90.0, 90.0, 90.0]
    p2_accs = [df_10fold.iloc[0]["ACC(%)"], df_10fold.iloc[1]["ACC(%)"], df_10fold.iloc[2]["ACC(%)"]]

    models_labels = ["NoPINN", "Proposed PINN\n(Data Loss)", "Proposed PINN\n(Active Physics)"]
    colors = ['#4A90E2', '#95A5A6', '#2ECC71']

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)
    protocols_names = ['Protocol 1 (Scan Split)', 'Protocol 2 (10-Fold CV)']
    x = np.arange(len(protocols_names))
    width = 0.24

    b1 = ax.bar(x - width, [p1_accs[0], p2_accs[0]], width, label="NoPINN", color='#4A90E2', edgecolor='black', linewidth=0.8)
    b2 = ax.bar(x, [p1_accs[1], p2_accs[1]], width, label="Proposed PINN (Data Loss)", color='#95A5A6', edgecolor='black', linewidth=0.8)
    b3 = ax.bar(x + width, [p1_accs[2], p2_accs[2]], width, label=r"Proposed PINN (Active Phys, $\alpha=1.0$)", color='#2ECC71', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(protocols_names, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, edgecolor='none')

    # Values on top of bars
    for b, v in zip(b1, [p1_accs[0], p2_accs[0]]):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for b, v in zip(b2, [p1_accs[1], p2_accs[1]]):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for b, v in zip(b3, [p1_accs[2], p2_accs[2]]):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    fig3_png = os.path.join(FIGURES_DIR, "fig3_pinn_active_acc_comparison_two_protocols.png")
    fig3_pdf = os.path.join(FIGURES_DIR, "fig3_pinn_active_acc_comparison_two_protocols.pdf")
    plt.savefig(fig3_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig3_pdf, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved ACC comparison figure to: {fig3_png}")

if __name__ == "__main__":
    run_10fold_evaluation()
