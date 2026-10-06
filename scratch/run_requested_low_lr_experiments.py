import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import load_5khz_fully_prepared, predict_and_denormalize
from domain_adaptation.data_loader import split_5khz_scan1_scan2
from domain_adaptation.cnn.losses import CNNTotalLoss
from domain_adaptation.benchmark_evaluation_protocols import (
    compute_classification_metrics_and_matrix,
    compute_regression_metrics,
    set_reproducible_seed,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_experiment(
    exp_name: str,
    model_type: str = "cnn",
    variant: str = "pinn",
    reset_kendall: bool = True,
    freeze_bn: bool = False,
    freeze_backbone: bool = True,
    epochs: int = 1000,
    lr: float = 1e-4,
    save_final: bool = True,
    early_stopping: bool = False,
    patience: int = 60,
):
    set_reproducible_seed(42)
    bundle = load_5khz_fully_prepared(model_type=model_type, variant=variant, device=DEVICE)
    base_model = bundle["model"].to(DEVICE)
    y_scaler = bundle["y_scaler"]
    train_set, test_set = split_5khz_scan1_scan2(bundle)

    X_tr = train_set["X"].to(DEVICE)
    y_clf_tr = train_set["y_clf"].to(DEVICE)
    y_wld_tr = train_set["y_wld_norm"].to(DEVICE)

    X_te = test_set["X"].to(DEVICE)
    y_clf_te = test_set["y_clf"].to(DEVICE)
    y_wld_te = test_set["y_wld_norm"].to(DEVICE)

    meta_te = test_set["metadata"]
    true_sh_te = [m["true_shape"] for m in meta_te]
    true_wld_te = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_te])

    model = copy.deepcopy(base_model)

    if reset_kendall:
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).data.fill_(0.0)

    if freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)
    eta_min = min(lr * 0.05, 1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    loss_fn = CNNTotalLoss(alpha=0.0, use_pinn=False).to(DEVICE)

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0
    stopped_at = epochs

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        optimizer.zero_grad()
        logits, pred_wld = model(X_tr)
        loss, _ = loss_fn(
            shape_logits=logits,
            shape_targets=y_clf_tr,
            pred_wld=pred_wld,
            true_wld=y_wld_tr,
            log_var_clf=getattr(model, "log_var_clf", None),
            log_var_w=getattr(model, "log_var_w", None),
            log_var_l=getattr(model, "log_var_l", None),
            log_var_d=getattr(model, "log_var_d", None),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Val check
        model.eval()
        with torch.no_grad():
            v_logits, v_pred_wld = model(X_te)
            v_loss, _ = loss_fn(
                shape_logits=v_logits,
                shape_targets=y_clf_te,
                pred_wld=v_pred_wld,
                true_wld=y_wld_te,
                log_var_clf=getattr(model, "log_var_clf", None),
                log_var_w=getattr(model, "log_var_w", None),
                log_var_l=getattr(model, "log_var_l", None),
                log_var_d=getattr(model, "log_var_d", None),
            )
            v_val = v_loss.item()
            if v_val < best_loss - 1e-4:
                best_loss = v_val
                best_weights = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if early_stopping and patience_cnt >= patience and ep >= 30:
                    stopped_at = ep + 1
                    break

    if not save_final:
        model.load_state_dict(best_weights)

    model.eval()
    p_sh, p_wld, _, _ = predict_and_denormalize(model, X_te, y_scaler=y_scaler)
    stat = compute_classification_metrics_and_matrix(true_sh_te, p_sh)
    reg = compute_regression_metrics(true_wld_te, p_wld)

    res = {
        "Config_Name": exp_name,
        "Reset_Kendall": reset_kendall,
        "Freeze_BN": freeze_bn,
        "Freeze_Backbone": freeze_backbone,
        "Epochs": epochs,
        "LR": lr,
        "ACC(%)": stat["overall_acc"],
        "Correct": f"{stat['total_correct']}/10",
        "MAE_Total(mm)": round(reg["MAE_Avg"], 4),
        "MAE_W(mm)": round(reg["MAE_W"], 4),
        "MAE_L(mm)": round(reg["MAE_L"], 4),
        "MAE_D(mm)": round(reg["MAE_D"], 4),
        "NMAE(%)": round(reg["NMAE_Avg(%)"], 2),
        "Predictions": "".join(["V" if t == p else "X" for t, p in zip(true_sh_te, p_sh)]),
    }

    print(
        f"{exp_name:65s} | ACC: {stat['overall_acc']:5.1f}% ({stat['total_correct']}/10) | "
        f"MAE: {reg['MAE_Avg']:.4f}mm (W:{reg['MAE_W']:.3f} L:{reg['MAE_L']:.3f} D:{reg['MAE_D']:.3f}) | "
        f"NMAE: {reg['NMAE_Avg(%)']:5.2f}%",
        flush=True
    )
    return res

if __name__ == "__main__":
    print(f"Bắt đầu thực nghiệm trên Device: {DEVICE}")
    print("=" * 115)

    experiments_to_run = [
        # 1. Baseline gốc để đối chiếu (Row 5 cũ)
        {
            "exp_name": "Proposed Baseline (Reset Kendall, 1000 ep, lr 1e-4)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 1000, "lr": 1e-4
        },
        # 2. Nhóm LR = 1e-5 (Freeze BN = False theo đúng yêu cầu)
        {
            "exp_name": "Proposed (Reset Kendall, Freeze BN: No, 2000 ep, lr 1e-5)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 2000, "lr": 1e-5
        },
        {
            "exp_name": "Proposed (Reset Kendall, Freeze BN: No, 3000 ep, lr 1e-5)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 3000, "lr": 1e-5
        },
        {
            "exp_name": "Proposed (Reset Kendall, Freeze BN: No, 5000 ep, lr 1e-5)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 5000, "lr": 1e-5
        },
        # 3. Nhóm LR = 1e-6 (Freeze BN = False theo đúng yêu cầu)
        {
            "exp_name": "Proposed (Reset Kendall, Freeze BN: No, 2000 ep, lr 1e-6)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 2000, "lr": 1e-6
        },
        {
            "exp_name": "Proposed (Reset Kendall, Freeze BN: No, 3000 ep, lr 1e-6)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 3000, "lr": 1e-6
        },
        {
            "exp_name": "Proposed (Reset Kendall, Freeze BN: No, 5000 ep, lr 1e-6)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": True,
            "epochs": 5000, "lr": 1e-6
        },
        # 4. Nhóm mở rộng đối chứng: Khi unfreeze backbone (trainable all) với LR siêu nhỏ 1e-5 & 1e-6
        {
            "exp_name": "Proposed (Train All Layers, Freeze BN: No, 3000 ep, lr 1e-5)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": False,
            "epochs": 3000, "lr": 1e-5
        },
        {
            "exp_name": "Proposed (Train All Layers, Freeze BN: No, 3000 ep, lr 1e-6)",
            "reset_kendall": True, "freeze_bn": False, "freeze_backbone": False,
            "epochs": 3000, "lr": 1e-6
        },
        # 5. Nhóm mở rộng đối chứng: Khi kết hợp Freeze BN = True với LR 1e-5 & 1e-6
        {
            "exp_name": "Proposed (Freeze BN: Yes, 3000 ep, lr 1e-5)",
            "reset_kendall": True, "freeze_bn": True, "freeze_backbone": True,
            "epochs": 3000, "lr": 1e-5
        },
        {
            "exp_name": "Proposed (Freeze BN: Yes, 3000 ep, lr 1e-6)",
            "reset_kendall": True, "freeze_bn": True, "freeze_backbone": True,
            "epochs": 3000, "lr": 1e-6
        },
    ]

    results = []
    for cfg in experiments_to_run:
        res = run_experiment(**cfg)
        results.append(res)

    df = pd.DataFrame(results)

    # Lưu kết quả
    out_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "finetune_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "requested_low_lr_ablation.csv")
    md_path = os.path.join(out_dir, "requested_low_lr_ablation.md")

    df.to_csv(csv_path, index=False)

    md_content = "# BÁO CÁO THỰC NGHIỆM: GIẢM SÂU TỐC ĐỘ HỌC (LR = 10^-5 VÀ 10^-6) VÀ TĂNG SỐ EPOCHS\n\n"
    md_content += "Thực nghiệm kiểm chứng trên cấu hình: `Reset Kendall = True`, `Freeze BN = False` (và các biến thể so sánh đối chứng) trên 10 mẫu kiểm thử thực nghiệm mù (Scan 2).\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n### Nhận xét & Kết luận chính:\n"
    md_content += "1. Khi giảm LR xuống $10^{-5}$ và $10^{-6}$ mà vẫn giữ `Freeze BN = False`:\n"
    md_content += "   - Xem bảng kết quả trên để thấy tác động trực tiếp của learning rate siêu nhỏ đối với khả năng hội tụ và thích nghi đặc trưng.\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 115)
    print(f"Đã hoàn thành toàn bộ thực nghiệm và lưu file thành công tại:")
    print(f"  - CSV: {csv_path}")
    print(f"  - MD : {md_path}")
