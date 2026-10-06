import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import load_5khz_fully_prepared, predict_and_denormalize
from domain_adaptation.data_loader import split_5khz_scan1_scan2, get_5khz_10fold_defect_splits
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
    DEFAULT_UNIQUE_SHAPES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(
    base_model,
    X_train,
    y_clf_train,
    y_wld_train,
    epochs=1500,
    lr=5e-4,
    freeze_backbone=True,
    freeze_bn=True,
    reset_kendall=True,
    reinit_clf=True,
):
    model = copy.deepcopy(base_model).to(DEVICE)

    if reset_kendall:
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).data.fill_(0.0)

    if freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False

    if reinit_clf and hasattr(model, "classifier"):
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)
        loss = F.cross_entropy(logits, y_clf_train) + F.mse_loss(pred_wld, y_wld_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model

def run_both_protocols():
    print("=" * 110)
    print("CHẠY ĐỐI CHỨNG CẢ 2 PHƯƠNG PHÁP: GIAO THỨC 1 (SCAN SPLIT) & GIAO THỨC 2 (10-FOLD LODO)")
    print(f"Device: {DEVICE}")
    print("=" * 110)

    results = []

    for variant, var_name, lr, reinit in [
        ("nopinn", "CNN_NoPINN", 1e-4, False),
        ("pinn", "CNN_Proposed (Optimal)", 5e-4, True),
    ]:
        set_reproducible_seed(42)
        bundle = load_5khz_fully_prepared(model_type="cnn", variant=variant, device=DEVICE)
        base_model = bundle["model"]
        y_scaler = bundle["y_scaler"]
        metadata = bundle["metadata"]

        # -------------------------------------------------------------
        # PROTOCOL 1: SCAN 1 -> SCAN 2
        # -------------------------------------------------------------
        train_set, test_set = split_5khz_scan1_scan2(bundle)
        X_tr = train_set["X"].to(DEVICE)
        y_clf_tr = train_set["y_clf"].to(DEVICE)
        y_wld_tr = train_set["y_wld_norm"].to(DEVICE)

        X_te = test_set["X"].to(DEVICE)
        meta_te = test_set["metadata"]
        true_sh_te = [m["true_shape"] for m in meta_te]
        true_wld_te = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_te])

        # Pretrained Zero-shot
        p_sh_0, p_wld_0, _, _ = predict_and_denormalize(base_model, X_te, y_scaler=y_scaler)
        stat_0 = compute_classification_metrics_and_matrix(true_sh_te, p_sh_0)
        reg_0 = compute_regression_metrics(true_wld_te, p_wld_0)

        # Finetuned
        ep = 300 if variant == "nopinn" else 1500
        f_bb = False if variant == "nopinn" else True
        f_bn = False if variant == "nopinn" else True
        rk = False if variant == "nopinn" else True

        f_model = train_model(
            base_model, X_tr, y_clf_tr, y_wld_tr,
            epochs=ep, lr=lr, freeze_backbone=f_bb, freeze_bn=f_bn, reset_kendall=rk, reinit_clf=reinit
        )
        p_sh_1, p_wld_1, _, _ = predict_and_denormalize(f_model, X_te, y_scaler=y_scaler)
        stat_1 = compute_classification_metrics_and_matrix(true_sh_te, p_sh_1)
        reg_1 = compute_regression_metrics(true_wld_te, p_wld_1)

        results.append({
            "Protocol": "Protocol 1 (Scan 1 -> Scan 2)",
            "Model": var_name,
            "Stage": "Zero-shot",
            "ACC(%)": stat_0["overall_acc"],
            "Correct": f"{stat_0['total_correct']}/10",
            "MAE_Total(mm)": reg_0["MAE_Avg"],
            "MAE_W": reg_0["MAE_W"],
            "MAE_L": reg_0["MAE_L"],
            "MAE_D": reg_0["MAE_D"],
            "NMAE(%)": reg_0["NMAE_Avg(%)"],
        })
        results.append({
            "Protocol": "Protocol 1 (Scan 1 -> Scan 2)",
            "Model": var_name,
            "Stage": "Finetuned",
            "ACC(%)": stat_1["overall_acc"],
            "Correct": f"{stat_1['total_correct']}/10",
            "MAE_Total(mm)": reg_1["MAE_Avg"],
            "MAE_W": reg_1["MAE_W"],
            "MAE_L": reg_1["MAE_L"],
            "MAE_D": reg_1["MAE_D"],
            "NMAE(%)": reg_1["NMAE_Avg(%)"],
        })

        # -------------------------------------------------------------
        # PROTOCOL 2: 10-FOLD LEAVE-ONE-DEFECT-OUT (LODO)
        # -------------------------------------------------------------
        folds = get_5khz_10fold_defect_splits(bundle)
        oof_preds_sh = [None] * len(metadata)
        oof_preds_wld = np.zeros((len(metadata), 3))
        true_sh_all = [m["true_shape"] for m in metadata]
        true_wld_all = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in metadata])

        print(f"\n--- Đang chạy Protocol 2 (10-Fold LODO) cho {var_name} ---")
        for f_idx, fold_data in enumerate(folds):
            tr = fold_data["train"]
            te = fold_data["test"]
            test_indices = te["indices"]

            X_f_tr = tr["X"].to(DEVICE)
            y_clf_f_tr = tr["y_clf"].to(DEVICE)
            y_wld_f_tr = tr["y_wld_norm"].to(DEVICE)
            X_f_te = te["X"].to(DEVICE)

            m_fold = train_model(
                base_model, X_f_tr, y_clf_f_tr, y_wld_f_tr,
                epochs=ep, lr=lr, freeze_backbone=f_bb, freeze_bn=f_bn, reset_kendall=rk, reinit_clf=reinit
            )
            p_sh_f, p_wld_f, _, _ = predict_and_denormalize(m_fold, X_f_te, y_scaler=y_scaler)

            for i, o_idx in enumerate(test_indices):
                oof_preds_sh[o_idx] = p_sh_f[i]
                oof_preds_wld[o_idx] = p_wld_f[i]

            print(f"  Fold {f_idx+1}/10 (Hold out: {fold_data['held_out_crack']}) done.")

        stat_2 = compute_classification_metrics_and_matrix(true_sh_all, oof_preds_sh)
        reg_2 = compute_regression_metrics(true_wld_all, oof_preds_wld)

        results.append({
            "Protocol": "Protocol 2 (10-Fold LODO)",
            "Model": var_name,
            "Stage": "Finetuned (OOF)",
            "ACC(%)": stat_2["overall_acc"],
            "Correct": f"{stat_2['total_correct']}/20",
            "MAE_Total(mm)": reg_2["MAE_Avg"],
            "MAE_W": reg_2["MAE_W"],
            "MAE_L": reg_2["MAE_L"],
            "MAE_D": reg_2["MAE_D"],
            "NMAE(%)": reg_2["NMAE_Avg(%)"],
        })

    df_all = pd.DataFrame(results)
    print("\n" + "=" * 110)
    print("TỔNG HỢP KẾT QUẢ ĐỐI CHỨNG CẢ 2 GIAO THỨC:")
    print("=" * 110)
    print(df_all.to_string(index=False))

    out_csv = "domain_adaptation/finetune_results/comparison_two_protocols_summary.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n[OK] Đã lưu file tổng hợp: {out_csv}")

if __name__ == "__main__":
    run_both_protocols()
