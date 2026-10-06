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
from domain_adaptation.data_loader import split_5khz_scan1_scan2
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
    DEFAULT_UNIQUE_SHAPES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_ablation_variant(
    base_model,
    X_train,
    y_clf_train,
    y_wld_train,
    epochs=1500,
    lr=2e-4,
    use_cosine=True,
    freeze_bn=False,
    reset_kendall=False,
    temp_T=1.0,
    early_stopping=False,
    patience=100,
):
    model = copy.deepcopy(base_model).to(DEVICE)

    # 1. Reset Kendall
    if reset_kendall:
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).data.fill_(0.0)

    # All weights are trained - STRICTLY NO REINIT
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1.0) # constant lr

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0
    stop_ep = epochs

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_train)

        # Temp scaling on logits
        loss_clf = F.cross_entropy(logits / temp_T, y_clf_train)
        loss_reg = F.mse_loss(pred_wld, y_wld_train)

        # If kendall was NOT reset and model has log_var parameters, use kendall weights
        if not reset_kendall and hasattr(model, "log_var_clf") and hasattr(model, "log_var_w"):
            w_clf = torch.exp(-model.log_var_clf)
            w_w = torch.exp(-model.log_var_w)
            w_l = torch.exp(-model.log_var_l)
            w_d = torch.exp(-model.log_var_d)
            loss_clf_w = 0.5 * w_clf * loss_clf + 0.5 * model.log_var_clf
            loss_reg_w = 0.5 * (
                w_w * F.mse_loss(pred_wld[:, 0], y_wld_train[:, 0]) + model.log_var_w +
                w_l * F.mse_loss(pred_wld[:, 1], y_wld_train[:, 1]) + model.log_var_l +
                w_d * F.mse_loss(pred_wld[:, 2], y_wld_train[:, 2]) + model.log_var_d
            )
            total_loss = loss_clf_w + loss_reg_w
        else:
            total_loss = loss_clf + loss_reg

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Check early stopping if enabled
        if early_stopping:
            loss_val = total_loss.item()
            if loss_val < best_val_loss - 1e-4:
                best_val_loss = loss_val
                best_weights = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience and ep >= 30:
                    stop_ep = ep + 1
                    model.load_state_dict(best_weights)
                    break

    model.eval()
    return model, stop_ep

def run_individual_ablation():
    set_reproducible_seed(42)
    bundle = load_5khz_fully_prepared(model_type="cnn", variant="pinn", device=DEVICE)
    base_model = bundle["model"]
    y_scaler = bundle["y_scaler"]
    train_set, test_set = split_5khz_scan1_scan2(bundle)

    X_tr = train_set["X"].to(DEVICE)
    y_clf_tr = train_set["y_clf"].to(DEVICE)
    y_wld_tr = train_set["y_wld_norm"].to(DEVICE)

    X_te = test_set["X"].to(DEVICE)
    meta_te = test_set["metadata"]
    true_sh_te = [m["true_shape"] for m in meta_te]
    true_wld_te = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_te])

    print("=" * 115)
    print("THỬ NGHIỆM ĐỘC LẬP TỪNG KỸ THUẬT (ABLATION STUDY: INDIVIDUAL TECHNIQUES)")
    print("STRICTLY NO REINIT - GIỮ NGUYÊN 100% TRỌNG SỐ TIỀN HUẤN LUYỆN TỪ FEM")
    print("=" * 115)

    cases = [
        # (name, reset_kendall, freeze_bn, temp_T, lr, use_cosine, epochs, early_stop, desc)
        ("0. Proposed Raw (Baseline cũ)", False, False, 1.0, 1e-4, False, 300, True, "Không làm gì, giữ nguyên bẫy Kendall 78x + BN thường + ngắt sớm"),
        ("1. CHỈ Reset Kendall (A)", True, False, 1.0, 1e-4, False, 1500, False, "Chỉ đưa Kendall 78x về 1x, còn lại giữ nguyên (BN thường, T=1.0)"),
        ("2. CHỈ Đóng băng BatchNorm (B)", False, True, 1.0, 1e-4, False, 1500, False, "Chỉ Freeze BN, vẫn giữ nguyên bẫy Kendall 78x, T=1.0"),
        ("3. CHỈ Hiệu chỉnh Logit T=2.0 (C)", False, False, 2.0, 1e-4, False, 1500, False, "Chỉ dùng T=2.0, vẫn giữ bẫy Kendall 78x, BN thường"),
        ("4. CHỈ Hạ LR & Cosine (D)", False, False, 1.0, 5e-5, True, 1500, False, "Chỉ hạ LR xuống 5e-5 + Cosine, vẫn giữ Kendall 78x, BN thường"),
        ("5. KẾT HỢP: Reset Kendall + Freeze BN", True, True, 1.0, 1e-4, False, 1500, False, "Bỏ bẫy Kendall + khóa BN, dùng CE thường (T=1.0)"),
        ("6. KẾT HỢP: Reset Kendall + Freeze BN + Cosine (lr 2e-4)", True, True, 1.0, 2e-4, True, 1500, False, "Bỏ bẫy Kendall + khóa BN + Cosine Annealing"),
        ("7. KẾT HỢP CẢ 3: Reset Kendall + Freeze BN + T=1.3", True, True, 1.3, 2e-4, True, 1500, False, "Cả 3 kỹ thuật với T=1.3"),
        ("8. KẾT HỢP CẢ 3: Reset Kendall + Freeze BN + T=1.5", True, True, 1.5, 2e-4, True, 1500, False, "Cả 3 kỹ thuật với T=1.5"),
        ("9. KẾT HỢP CẢ 3: Reset Kendall + Freeze BN + T=2.0 (TỐI ƯU NHẤT)", True, True, 2.0, 2e-4, True, 1500, False, "Cả 3 kỹ thuật với T=2.0"),
        ("10. KẾT HỢP CẢ 3: Reset Kendall + Freeze BN + T=2.5", True, True, 2.5, 2e-4, True, 1500, False, "Cả 3 kỹ thuật với T=2.5"),
        ("11. KẾT HỢP CẢ 3: Reset Kendall + Freeze BN + T=3.0", True, True, 3.0, 2e-4, True, 1500, False, "Cả 3 kỹ thuật với T=3.0"),
    ]

    records = []
    for name, rk, fbn, t_T, lr, cos, ep, es, desc in cases:
        m, stop_ep = train_ablation_variant(
            base_model, X_tr, y_clf_tr, y_wld_tr,
            epochs=ep, lr=lr, use_cosine=cos, freeze_bn=fbn,
            reset_kendall=rk, temp_T=t_T, early_stopping=es,
        )
        p_sh, p_wld, _, _ = predict_and_denormalize(m, X_te, y_scaler=y_scaler)
        stat = compute_classification_metrics_and_matrix(true_sh_te, p_sh)
        reg = compute_regression_metrics(true_wld_te, p_wld)
        matches = ["V" if t == p else "X" for t, p in zip(true_sh_te, p_sh)]
        match_str = "".join(matches)

        print(
            f"{name:<58s} | Ep:{stop_ep:<4d} | ACC: {stat['overall_acc']:5.1f}% ({stat['total_correct']}/10) [{match_str}] | "
            f"MAE: {reg['MAE_Avg']:.4f}mm (W:{reg['MAE_W']:.3f} L:{reg['MAE_L']:.3f} D:{reg['MAE_D']:.3f}) | NMAE: {reg['NMAE_Avg(%)']:5.2f}%",
            flush=True
        )
        records.append({
            "Experiment": name,
            "Reset_Kendall": rk,
            "Freeze_BN": fbn,
            "Temp_T": t_T,
            "LR": lr,
            "Cosine_Annealing": cos,
            "Stop_Epoch": stop_ep,
            "ACC(%)": stat["overall_acc"],
            "Correct": f"{stat['total_correct']}/10",
            "MAE_Total(mm)": reg["MAE_Avg"],
            "MAE_W(mm)": reg["MAE_W"],
            "MAE_L(mm)": reg["MAE_L"],
            "MAE_D(mm)": reg["MAE_D"],
            "NMAE(%)": reg["NMAE_Avg(%)"],
            "Matches": match_str,
            "Description": desc,
        })

    df = pd.DataFrame(records)
    out_csv = "domain_adaptation/pure_physics_study/tables/ablation_individual_techniques.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Đã lưu bảng kết quả chi tiết vào: {out_csv}")

if __name__ == "__main__":
    run_individual_ablation()
