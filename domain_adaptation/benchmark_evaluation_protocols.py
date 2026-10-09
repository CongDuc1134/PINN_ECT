# -*- coding: utf-8 -*-
"""
===============================================================================
BENCHMARK EVALUATION PROTOCOLS FOR SIM-TO-REAL ECT DOMAIN ADAPTATION
===============================================================================
Triển khai toàn diện 2 hướng nghiên cứu thích ứng miền trên dữ liệu thực nghiệm 5kHz
sử dụng module nạp dữ liệu load_real_experiment_data.py:

HƯỚNG 1 (GIAO THỨC 1 - SCAN-BASED REPEATABILITY & DRIFT ROBUSTNESS):
- Tập Train (10 mẫu): Lần quét 1 (Tập 'No', không chứa '_1.csv').
- Tập Test  (10 mẫu): Lần quét 2 (Tập '_1', chứa '_1.csv').
- Mục tiêu: Kiểm tra khả năng chống trôi dạt tín hiệu và rung chấn cơ khí cảm biến.
- Vẽ riêng các đường Loss khi Finetuning cho từng phương pháp.

HƯỚNG 2 (GIAO THỨC 2 - 10-FOLD LEAVE-ONE-DEFECT-OUT CROSS VALIDATION):
- Chia 10 folds theo từng khuyết tật độc lập (crack_no = 1..10).
- Mỗi fold: Bỏ 1 vết nứt (cả Scan 1 & Scan 2 = 2 file) ra KHÔNG HỌC làm tập Test mù.
  Finetune mô hình trên 18 file của 9 vết nứt còn lại.
- Gom toàn bộ 20 dự đoán out-of-fold độc lập.
- Thống kê toàn diện True Positives (TP), False Positives (FP), False Negatives (FN),
  True Negatives (TN) cho từng class, tính toán Overall Accuracy, Precision, Recall,
  F1-Score và sai số kích thước MAE (W, L, D).

QUY CHUẨN ĐỒ THỊ:
- Tất cả đồ thị vẽ theo phong cách IEEE Transactions (300 DPI, PNG + PDF).
- TẤT CẢ CHÚ THÍCH (LEGENDS) ĐỀU ĐƯỢC ĐẶT Ở NGOÀI KHUNG BIỂU ĐỒ.
===============================================================================
"""

import os
import sys
import copy
import argparse
import random
import json
import datetime
import shutil
from typing import Optional, Tuple, Dict, Any, List

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import module nạp dữ liệu chuyên dụng
from domain_adaptation.load_real_experiment_data import (
    DEFAULT_UNIQUE_SHAPES,
    TABLE_2_GROUND_TRUTH,
    denormalize_regression_predictions,
)
from domain_adaptation.data_loader import (
    load_all_5khz_samples,
    split_5khz_scan1_scan2,
    get_5khz_10fold_defect_splits,
)
from domain_adaptation.model_loader import (
    load_5khz_fully_prepared,
    predict_and_denormalize,
)
from domain_adaptation.cnn.losses import (
    CNNTotalLoss,
    compute_physics_loss_autograd,
    get_shape_physics_map,
)
from domain_adaptation.mlp.losses import MultitaskMLPTotalLoss
from domain_adaptation.xiong.losses import XiongTotalLoss
from domain_adaptation.plot_ieee_utils import (
    setup_ieee_style,
    save_ieee_figure,
    plot_separate_loss_curve_single_model,
    plot_all_methods_loss_grid,
    plot_confusion_matrix_ieee,
    plot_benchmark_comparison_ieee,
    plot_dimensional_mae_breakdown_ieee,
    plot_finetuned_two_protocols_comparison,
    plot_linear_fit_wld_ieee,
)


# =============================================================================
# CỐ ĐỊNH SEED TÁI TẠO KẾT QUẢ KHOA HỌC
# =============================================================================
def set_reproducible_seed(seed: int = 42):
    """Cố định nguồn ngẫu nhiên cho toàn bộ các thư viện."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# HÀM TÍNH TOÁN METRIC CHUẨN XÁC: CONFUSION MATRIX, TP, FP, FN, TN, MAE
# =============================================================================
def compute_classification_metrics_and_matrix(
    true_classes: List[str],
    pred_classes: List[str],
    class_names: List[str] = DEFAULT_UNIQUE_SHAPES,
) -> Dict[str, Any]:
    """
    Tính toán chi tiết ma trận nhầm lẫn và các chỉ số TP, FP, FN, TN, Precision, Recall, F1.
    """
    n_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for tc, pc in zip(true_classes, pred_classes):
        if tc in class_to_idx and pc in class_to_idx:
            cm[class_to_idx[tc], class_to_idx[pc]] += 1

    total_samples = len(true_classes)
    total_correct = int(np.trace(cm))
    overall_acc = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    per_class_stats = []
    for i, cname in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(total_samples - (tp + fp + fn))

        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc_class = ((tp + tn) / total_samples * 100.0) if total_samples > 0 else 0.0

        per_class_stats.append({
            "class": cname,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Support": tp + fn,
            "Precision(%)": round(precision, 2),
            "Recall(%)": round(recall, 2),
            "F1(%)": round(f1, 2),
            "Accuracy(%)": round(acc_class, 2),
        })

    return {
        "confusion_matrix": cm,
        "class_names": class_names,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_acc": round(overall_acc, 2),
        "per_class_stats": pd.DataFrame(per_class_stats),
    }


def compute_regression_metrics(true_wld_np: np.ndarray, pred_wld_np: np.ndarray) -> Dict[str, float]:
    """
    Tính MAE (mm) và NMAE (%) cho từng chiều W, L, D và giá trị trung bình.
    Quy chuẩn NMAE (%):
      NMAE_W = (MAE_W / Mean_W) * 100%
      NMAE_L = (MAE_L / Mean_L) * 100%
      NMAE_D = (MAE_D / Mean_D) * 100%
      NMAE_Avg = (NMAE_W + NMAE_L + NMAE_D) / 3 (Chuẩn hóa vĩ mô bình đẳng từng chiều, tránh Length lấn át)
    """
    err = np.abs(pred_wld_np - true_wld_np)
    mae_w = float(np.mean(err[:, 0]))
    mae_l = float(np.mean(err[:, 1]))
    mae_d = float(np.mean(err[:, 2]))
    mae_avg = float(np.mean(err))

    mean_w = float(np.mean(true_wld_np[:, 0]))
    mean_l = float(np.mean(true_wld_np[:, 1]))
    mean_d = float(np.mean(true_wld_np[:, 2]))

    nmae_w = (mae_w / mean_w * 100.0) if mean_w > 0 else 0.0
    nmae_l = (mae_l / mean_l * 100.0) if mean_l > 0 else 0.0
    nmae_d = (mae_d / mean_d * 100.0) if mean_d > 0 else 0.0
    nmae_avg = (nmae_w + nmae_l + nmae_d) / 3.0

    return {
        "MAE_W": round(mae_w, 4),
        "MAE_L": round(mae_l, 4),
        "MAE_D": round(mae_d, 4),
        "MAE_Avg": round(mae_avg, 4),
        "NMAE_W(%)": round(nmae_w, 2),
        "NMAE_L(%)": round(nmae_l, 2),
        "NMAE_D(%)": round(nmae_d, 2),
        "NMAE_Avg(%)": round(nmae_avg, 2),
    }


# =============================================================================
# HÀM HUẤN LUYỆN FINETUNING LÕI VỚI THEO DÕI LOSS VÀ EARLY STOPPING
# =============================================================================
def finetune_single_model(
    model_type: str,
    base_model: nn.Module,
    X_train: torch.Tensor,
    y_clf_train: torch.Tensor,
    y_wld_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_clf_val: Optional[torch.Tensor] = None,
    y_wld_val: Optional[torch.Tensor] = None,
    epochs: int = 500,
    lr: float = 5e-3,
    freeze_backbone: bool = True,
    patience: int = 100,
    device: str = "cpu",
    variant: str = "pinn",
    y_scaler: Optional[Any] = None,
    x_scaler: Optional[Any] = None,
    reset_kendall: bool = True,
    alpha_pinn: float = 1.0,
    early_stopping: bool = False,
    use_final_epoch: bool = False,
    use_pinn_loss: Optional[bool] = None,
    fixed_lr: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Finetuning mô hình:
    - Nạp mô hình từ Pre-trained Checkpoint.
    - Reset tham số bất định Kendall về 0.0 (hệ số exp(-s) = 1.0) để cân bằng các tác vụ (hoặc giữ nguyên nếu reset_kendall=False).
    - Đóng băng Backbone (bảo tồn tri thức trích xuất đặc trưng vật lý) & khóa BatchNorm eval của Backbone.
    - Cập nhật các tầng Linear của Heads chuẩn (không chứa BatchNorm).
    - Hỗ trợ LR cố định (Fixed LR) hoặc Cosine Annealing Learning Rate Scheduler.
    - Hỗ trợ tắt Early Stopping (huấn luyện trọn vẹn số epochs yêu cầu).
    - Tùy chọn lưu giữ và đánh giá bằng trọng số Epoch cuối cùng (final epoch) hoặc Best Val Loss.
    - use_pinn_loss: Mặc định None -> Mỗi mô hình gọi đúng hàm Loss bản thể (PINN dùng Physics PINN loss, NoPINN dùng thuần Supervised).
    """
    model = copy.deepcopy(base_model).to(device)

    # 1. Reset tham số bất định Kendall về 0.0 (exp(-s) = 1.0) nếu được yêu cầu
    if reset_kendall:
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d", "log_var_reg"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).data.fill_(0.0)

    # Đóng băng Backbone nếu được yêu cầu, cập nhật triệt để các Heads (Headers)
    if freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False
        # Đảm bảo các Heads và tham số bất định luôn ở chế độ Trainable
        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        if hasattr(model, "regressor_backbone"):
            for p in model.regressor_backbone.parameters():
                p.requires_grad = True
        if hasattr(model, "reg_head"):
            for p in model.reg_head.parameters():
                p.requires_grad = True
        for attr in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d", "log_var_reg"]:
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Parameter):
                getattr(model, attr).requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)

    if fixed_lr:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Khởi tạo hàm Loss tương ứng theo từng mô hình (Loss nào thì gọi đúng loss đấy)
    # - Model PINN (variant == "pinn"): Gọi đúng hàm Loss PINN có thành phần Physics Loss (alpha * L_physics)
    # - Model NoPINN (variant == "nopinn"): Gọi đúng hàm Loss NoPINN thuần supervised data loss
    if use_pinn_loss is None:
        use_pinn = (variant.lower() == "pinn")
    else:
        use_pinn = use_pinn_loss and (variant.lower() == "pinn")
    shape_map = get_shape_physics_map() if use_pinn else None

    if model_type == "cnn":
        loss_fn = CNNTotalLoss(alpha=alpha_pinn if use_pinn else 0.0, use_pinn=use_pinn)
    elif model_type == "mlp":
        loss_fn = MultitaskMLPTotalLoss(alpha=alpha_pinn if use_pinn else 0.0, use_pinn=use_pinn)
    elif model_type == "xiong":
        loss_fn = XiongTotalLoss(alpha=alpha_pinn if use_pinn else 0.0, use_pinn=use_pinn)
    else:
        raise ValueError(f"model_type không được hỗ trợ: {model_type}")

    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_loss_clf": [],
        "train_loss_reg": [],
        "val_loss": [],
        "val_loss_clf": [],
        "val_loss_reg": [],
        "best_epoch": 1,
        "early_stopped_at": None,
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    has_val = (X_val is not None and y_wld_val is not None)

    # In chi tiết toàn bộ tham số khi bắt đầu Finetuning
    n_train = len(X_train)
    n_val = len(X_val) if X_val is not None else 0
    batch_size = n_train  # Full-batch optimization cho tập mẫu thực nghiệm
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    loss_desc = f"{model_type.upper()} Total Loss"
    if model_type in ["cnn", "mlp"]:
        loss_desc += " (CrossEntropy + MSE"
    else:
        loss_desc += " (MSE"
    if use_pinn:
        loss_desc += f" + alpha * Physics_Loss, alpha={alpha_pinn})"
    else:
        loss_desc += " [Thuần Supervised Data Loss - Không Physics PINN])"

    print("\n" + "-" * 75)
    print(f"  [FINETUNING CONFIG] Mô hình: {model_type.upper()} ({variant.upper()})")
    print(f"  * Dữ liệu huấn luyện : {n_train} mẫu thực nghiệm (Batch Size = {batch_size})")
    print(f"  * Dữ liệu kiểm thử   : {n_val} mẫu thực nghiệm")
    print(f"  * Số Epochs huấn luyện: {epochs} epochs")
    if early_stopping and patience > 0:
        print(f"  * Ngưỡng dừng sớm    : {patience} epochs (Early Stopping Patience)")
    else:
        print(f"  * Early Stopping     : ĐÃ TẮT (Chạy trọn vẹn {epochs} epochs)")
    lr_desc = f"{lr:.1e} (Cố định - Fixed LR)" if fixed_lr else f"{lr:.1e} (CosineAnnealingLR, eta_min=1e-5)"
    print(f"  * Tốc độ học (LR)    : {lr_desc}")
    print(f"  * Trọng số PINN Alpha: {alpha_pinn}")
    print(f"  * Hàm Loss chi tiết  : {loss_desc}")
    print(f"  * Đóng băng Backbone : {freeze_backbone} (Trainable params: {trainable_params:,} / Total: {total_params:,})")
    print(f"  * Reset Kendall Unc. : {reset_kendall} {'(log_var -> 0.0, exp(-s) = 1.0)' if reset_kendall else '(Giữ nguyên bản từ checkpoint)'}")
    print(f"  * Chế độ lưu kết quả : {'Trọng số Epoch cuối cùng' if use_final_epoch else 'Trọng số Tối ưu Best Val Loss'}")
    print("-" * 75)

    v_val_item = 0.0
    for ep in range(epochs):
        model.train()
        if freeze_backbone and hasattr(model, "backbone"):
            model.backbone.eval()  # Khóa cứng BatchNorm stats của Backbone để chống trôi đặc trưng
        optimizer.zero_grad()

        # Forward train
        if model_type in ["cnn", "mlp"]:
            logits, pred_wld = model(X_train)
            batch_phys = None
            if use_pinn and shape_map is not None:
                phys_list = []
                for i in range(len(X_train)):
                    h_field = X_train[i, 0] if model_type == "cnn" else X_train[i, :1024].reshape(32, 32)
                    cid = int(y_clf_train[i].item())
                    s_name = DEFAULT_UNIQUE_SHAPES[cid] if 0 <= cid < len(DEFAULT_UNIQUE_SHAPES) else "Rectangular"
                    pl = compute_physics_loss_autograd(
                        H_true=h_field,
                        w_pred=pred_wld[i, 0],
                        l_pred=pred_wld[i, 1],
                        d_pred=pred_wld[i, 2],
                        shape_name=s_name,
                        shape_map=shape_map,
                        y_scaler=y_scaler,
                        X_scaler=x_scaler,
                    )
                    phys_list.append(pl)
                if phys_list:
                    batch_phys = torch.stack(phys_list).mean()

            if model_type == "cnn":
                loss, metrics = loss_fn(
                    shape_logits=logits,
                    shape_targets=y_clf_train,
                    pred_wld=pred_wld,
                    true_wld=y_wld_train,
                    log_var_clf=getattr(model, "log_var_clf", None),
                    log_var_w=getattr(model, "log_var_w", None),
                    log_var_l=getattr(model, "log_var_l", None),
                    log_var_d=getattr(model, "log_var_d", None),
                    physics_loss=batch_phys,
                )
            else: # mlp
                loss, metrics = loss_fn(
                    shape_logits=logits,
                    shape_targets=y_clf_train,
                    pred_wld=pred_wld,
                    true_wld=y_wld_train,
                    log_var_clf=getattr(model, "log_var_clf", None),
                    log_var_reg=getattr(model, "log_var_reg", None),
                    physics_loss=batch_phys,
                )
            tr_clf = metrics.get("loss_clf", 0.0)
            tr_reg = metrics.get("avg_reg_loss", metrics.get("loss_reg", 0.0))
        else: # xiong
            pred_wld = model(X_train)
            batch_phys = None
            if use_pinn and shape_map is not None:
                phys_list = []
                for i in range(len(X_train)):
                    h_field = X_train[i, :1024].reshape(32, 32)
                    pl = compute_physics_loss_autograd(
                        H_true=h_field,
                        w_pred=pred_wld[i, 0],
                        l_pred=pred_wld[i, 1],
                        d_pred=pred_wld[i, 2],
                        shape_name="Rectangular",
                        shape_map=shape_map,
                        y_scaler=y_scaler,
                        X_scaler=x_scaler,
                    )
                    phys_list.append(pl)
                if phys_list:
                    batch_phys = torch.stack(phys_list).mean()

            loss, metrics = loss_fn(
                pred_wld=pred_wld,
                true_wld=y_wld_train,
                physics_loss=batch_phys,
            )
            tr_clf = 0.0
            tr_reg = metrics.get("avg_reg_loss", float(loss.item()))

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history["epoch"].append(ep + 1)
        history["lr"].append(lr if fixed_lr else float(scheduler.get_last_lr()[0]))
        history["train_loss"].append(float(loss.item()))
        history["train_loss_clf"].append(float(tr_clf))
        history["train_loss_reg"].append(float(tr_reg))

        # Forward validation
        if has_val:
            model.eval()
            with torch.no_grad():
                v_phys = None
                if model_type in ["cnn", "mlp"]:
                    v_logits, v_pred_wld = model(X_val)
                    if model_type == "cnn":
                        v_loss, v_met = loss_fn(
                            shape_logits=v_logits,
                            shape_targets=y_clf_val,
                            pred_wld=v_pred_wld,
                            true_wld=y_wld_val,
                            log_var_clf=getattr(model, "log_var_clf", None),
                            log_var_w=getattr(model, "log_var_w", None),
                            log_var_l=getattr(model, "log_var_l", None),
                            log_var_d=getattr(model, "log_var_d", None),
                            physics_loss=v_phys,
                        )
                    else:
                        v_loss, v_met = loss_fn(
                            shape_logits=v_logits,
                            shape_targets=y_clf_val,
                            pred_wld=v_pred_wld,
                            true_wld=y_wld_val,
                            log_var_clf=getattr(model, "log_var_clf", None),
                            log_var_reg=getattr(model, "log_var_reg", None),
                            physics_loss=v_phys,
                        )
                    val_clf = v_met.get("loss_clf", 0.0)
                    val_reg = v_met.get("avg_reg_loss", v_met.get("loss_reg", 0.0))
                else:
                    v_pred_wld = model(X_val)
                    v_loss, v_met = loss_fn(pred_wld=v_pred_wld, true_wld=y_wld_val, physics_loss=v_phys)
                    val_clf = 0.0
                    val_reg = v_met.get("avg_reg_loss", float(v_loss.item()))

            v_val_item = float(v_loss.item())
            history["val_loss"].append(v_val_item)
            history["val_loss_clf"].append(float(val_clf))
            history["val_loss_reg"].append(float(val_reg))

            # Early stopping check
            if v_val_item < best_val_loss - 1e-4:
                best_val_loss = v_val_item
                history["best_epoch"] = ep + 1
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping and patience > 0 and patience_counter >= patience and ep >= 30:
                    history["early_stopped_at"] = ep + 1
                    print(f"    [Dừng sớm] Không cải thiện Loss sau {patience} epochs liên tiếp -> Dừng tại Epoch {ep + 1}.")
                    break
        else:
            best_weights = copy.deepcopy(model.state_dict())
            history["best_epoch"] = ep + 1

        # In tiến độ định kỳ (mỗi 500 epoch nếu >= 2000 epoch) hoặc tại epoch cuối
        log_interval = 500 if epochs >= 2000 else (100 if epochs >= 500 else 50)
        if (ep + 1) % log_interval == 0 or (ep + 1) == epochs:
            val_info = f" | Val Loss = {v_val_item:.4f}" if has_val else ""
            current_lr_val = lr if (fixed_lr or scheduler is None) else scheduler.get_last_lr()[0]
            print(f"    [Epoch {ep+1:04d}/{epochs}] Train Loss = {loss.item():.4f} (Clf={tr_clf:.4f}, Reg={tr_reg:.4f}){val_info} | LR = {current_lr_val:.2e}")

    # Xử lý trọng số mô hình đầu ra
    if use_final_epoch:
        # Giữ nguyên trọng số tại epoch cuối cùng (Epoch 2000)
        model.eval()
        stop_msg = f"Dừng sớm tại Epoch {history['early_stopped_at']}" if history['early_stopped_at'] else f"Hoàn tất đủ {epochs} epochs"
        print(f"    -> [Hoàn thành] {stop_msg} | Sử dụng và lưu trữ trọng số mô hình tại Epoch cuối cùng (Epoch {ep + 1})\n")
    else:
        # Nạp lại trọng số tại best epoch
        model.load_state_dict(best_weights)
        model.eval()
        stop_msg = f"Dừng sớm tại Epoch {history['early_stopped_at']}" if history['early_stopped_at'] else f"Hoàn tất đủ {epochs} epochs"
        val_disp = f" (Best Val Loss = {best_val_loss:.4f})" if has_val else ""
        print(f"    -> [Hoàn thành] {stop_msg} | Khôi phục trọng số tối ưu tại Epoch {history['best_epoch']}{val_disp}\n")
    return model, history


# =============================================================================
# HƯỚNG 1: GIAO THỨC 1 (FINETUNE SCAN 1 -> TEST SCAN 2)
# =============================================================================
def run_protocol_1_scan_split(
    models_to_test: List[Dict[str, str]],
    epochs: int = 500,
    lr: float = 5e-3,
    freeze_backbone: bool = True,
    patience: int = 100,
    device: str = "cpu",
    output_dir: str = "domain_adaptation/finetune_results/protocols",
    reset_kendall: bool = True,
    early_stopping: bool = False,
    use_final_epoch: bool = False,
    use_pinn_loss: Optional[bool] = None,
    fixed_lr: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Thực thi toàn diện Hướng 1:
    - Train trên 10 mẫu Scan 1.
    - Test trên 10 mẫu Scan 2.
    - Đánh giá cả Pretrained (Zero-shot) và Finetuned.
    - Vẽ riêng các đường Loss khi Finetuning cho từng phương pháp.
    """
    if use_pinn_loss is None:
        loss_mode_label = "Mỗi mô hình dùng đúng Loss tương ứng (PINN có Physics Loss, NoPINN thuần Data)"
    elif use_pinn_loss:
        loss_mode_label = "Ép buộc bật PINN Physics Loss cho tất cả mô hình"
    else:
        loss_mode_label = "Tắt hoàn toàn PINN Physics Loss (chỉ dùng Data Loss)"

    print("\n" + "=" * 80)
    print("HƯỚNG 1 (GIAO THỨC 1): FINETUNE SCAN 1 -> TEST MÙ SCAN 2")
    print(f"Cấu hình: Freeze Backbone = {freeze_backbone} | Max Epochs = {epochs} | LR = {lr} | Early Stopping = {early_stopping} | Hàm Loss = {loss_mode_label} | Save Final = {use_final_epoch}")
    print("=" * 80)

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    results_summary = []
    histories_dict = {}

    for info in models_to_test:
        key = info["key"]
        m_type = info["type"]
        var = info["variant"]
        print(f"\n>>> [Hướng 1] Xử lý mô hình: {key} ({m_type.upper()}, {var}) ...")

        try:
            bundle = load_5khz_fully_prepared(model_type=m_type, variant=var, device=device)
        except FileNotFoundError as e:
            print(f"\n[CẢNH BÁO] Bỏ qua mô hình {key} ({m_type.upper()}, {var}) do chưa có checkpoint: {e}")
            print(f"            (Vui lòng huấn luyện mô hình trong thư mục {m_type}/ trước khi đánh giá)")
            continue

        base_model = bundle["model"]
        y_scaler = bundle["y_scaler"]

        train_set, test_set = split_5khz_scan1_scan2(bundle)
        X_tr, y_clf_tr, y_wld_tr = train_set["X"], train_set["y_clf"], train_set["y_wld_norm"]
        X_te, y_clf_te, y_wld_te = test_set["X"], test_set["y_clf"], test_set["y_wld_norm"]
        meta_test = test_set["metadata"]

        true_sh_test = [m["true_shape"] for m in meta_test]
        true_wld_test = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in meta_test])

        # 1. Đánh giá Zero-shot Pretrained trên Scan 2
        p_sh_pre, p_wld_pre, _, _ = predict_and_denormalize(base_model, X_te, y_scaler=y_scaler)
        stat_pre = compute_classification_metrics_and_matrix(true_sh_test, p_sh_pre)
        reg_pre = compute_regression_metrics(true_wld_test, p_wld_pre)

        alpha_pinn = bundle.get("alpha", 1.0)
        print(f"    [Alpha PINN] Hệ số alpha = {alpha_pinn} (load từ {os.path.basename(bundle['checkpoint_dir'])})")

        # 2. Finetune trên Scan 1
        finetuned_m, hist = finetune_single_model(
            model_type=m_type,
            base_model=base_model,
            X_train=X_tr,
            y_clf_train=y_clf_tr,
            y_wld_train=y_wld_tr,
            X_val=X_te,
            y_clf_val=y_clf_te if m_type in ["cnn", "mlp"] else None,
            y_wld_val=y_wld_te,
            epochs=epochs,
            lr=lr,
            freeze_backbone=freeze_backbone,
            patience=patience,
            device=device,
            variant=var,
            y_scaler=y_scaler,
            x_scaler=bundle.get("x_scaler", None),
            reset_kendall=reset_kendall,
            alpha_pinn=alpha_pinn,
            early_stopping=early_stopping,
            use_final_epoch=use_final_epoch,
            use_pinn_loss=use_pinn_loss,
            fixed_lr=fixed_lr,
        )
        histories_dict[key] = hist

        # Lưu checkpoint mô hình đã finetune (lưu rõ ràng trạng thái epoch cuối cùng)
        ckpt_save_path = os.path.join(checkpoints_dir, f"finetuned_{key.lower()}.pth")
        torch.save({
            "model_state_dict": finetuned_m.state_dict(),
            "model_type": m_type,
            "variant": var,
            "total_epochs": epochs,
            "final_epoch": hist["epoch"][-1] if hist["epoch"] else epochs,
            "best_epoch": hist["best_epoch"],
            "use_final_epoch": use_final_epoch,
        }, ckpt_save_path)
        print(f"    [Checkpoint] Đã lưu mô hình finetuned vào: {os.path.basename(ckpt_save_path)}")

        # 3. Đánh giá Finetuned trên Scan 2 (Inference chi tiết)
        p_sh_post, p_wld_post, _, _ = predict_and_denormalize(finetuned_m, X_te, y_scaler=y_scaler)
        stat_post = compute_classification_metrics_and_matrix(true_sh_test, p_sh_post)
        reg_post = compute_regression_metrics(true_wld_test, p_wld_post)

        # Xuất bảng chi tiết từng mẫu khi inference (Scan 2)
        pred_records = []
        for i, m in enumerate(meta_test):
            pred_records.append({
                "Filename": m["filename"],
                "Crack_No": m.get("crack_no", i + 1),
                "True_Shape": m["true_shape"],
                "Pred_Shape": p_sh_post[i],
                "Shape_Match": (m["true_shape"] == p_sh_post[i]) if m_type != "xiong" else np.nan,
                "True_W": m["true_w"], "Pred_W": round(float(p_wld_post[i, 0]), 4), "Err_W": round(float(abs(m["true_w"] - p_wld_post[i, 0])), 4),
                "True_L": m["true_l"], "Pred_L": round(float(p_wld_post[i, 1]), 4), "Err_L": round(float(abs(m["true_l"] - p_wld_post[i, 1])), 4),
                "True_D": m["true_d"], "Pred_D": round(float(p_wld_post[i, 2]), 4), "Err_D": round(float(abs(m["true_d"] - p_wld_post[i, 2])), 4),
            })
        df_pred_details = pd.DataFrame(pred_records)
        pred_detail_csv = os.path.join(tables_dir, f"predictions_protocol1_{key.lower()}.csv")
        df_pred_details.to_csv(pred_detail_csv, index=False)

        # 4. Vẽ riêng đường Loss khi Finetune cho mô hình này vào figures_dir
        loss_fig_path = os.path.join(figures_dir, f"fig_protocol1_separate_loss_{key}")
        plot_separate_loss_curve_single_model(
            history=hist,
            model_name=key,
            output_base_path=loss_fig_path,
        )

        # 5. Vẽ ma trận nhầm lẫn chuẩn IEEE vào figures_dir
        cm_fig_path = os.path.join(figures_dir, f"fig_protocol1_confusion_matrix_{key}")
        plot_confusion_matrix_ieee(
            confusion_matrix=stat_post["confusion_matrix"],
            class_names=DEFAULT_UNIQUE_SHAPES,
            model_name=f"{key} (Finetuned)",
            output_base_path=cm_fig_path,
            acc_val=stat_post["overall_acc"] if m_type != "xiong" else None,
        )

        # 5b. Xuất đồ thị chuẩn gói figures_ieee (Fig 2 & Fig 3 Parity Box)
        try:
            from figures_ieee import plot_fig2_confusion_matrix, plot_fig3_regression_scatter
            if m_type != "xiong":
                plot_fig2_confusion_matrix(
                    y_true=true_sh_test,
                    y_pred=p_sh_post,
                    unique_shapes=DEFAULT_UNIQUE_SHAPES,
                    output_dir=figures_dir,
                    filename_prefix=f"fig2_confusion_matrix_ieee_{key}",
                )
            plot_fig3_regression_scatter(
                y_true_wld=true_wld_test,
                y_pred_wld=p_wld_post,
                output_dir=figures_dir,
                filename_prefix=f"fig3_regression_scatter_ieee_{key}",
            )
        except Exception as e_fig:
            print(f"    [figures_ieee] Cảnh báo xuất hình: {e_fig}")

        # 6. Ghi nhận kết quả
        for stage, stat_s, reg_s in [("Pretrained", stat_pre, reg_pre), ("Finetuned", stat_post, reg_post)]:
            results_summary.append({
                "Protocol": "Protocol 1 (Scan Split)",
                "Model": key,
                "Stage": stage,
                "Test_Samples": len(X_te),
                "Total_Correct": stat_s["total_correct"] if m_type != "xiong" else np.nan,
                "ACC(%)": stat_s["overall_acc"] if m_type != "xiong" else np.nan,
                "NMAE_Overall(%)": reg_s["NMAE_Avg(%)"],
                "MAE_Overall(mm)": reg_s["MAE_Avg"],
                "MAE_W(mm)": reg_s["MAE_W"],
                "MAE_L(mm)": reg_s["MAE_L"],
                "MAE_D(mm)": reg_s["MAE_D"],
                "NMAE_W(%)": reg_s["NMAE_W(%)"],
                "NMAE_L(%)": reg_s["NMAE_L(%)"],
                "NMAE_D(%)": reg_s["NMAE_D(%)"],
            })

    df_res = pd.DataFrame(results_summary)
    csv_out = os.path.join(tables_dir, "protocol1_scan_split_results.csv")
    df_res.to_csv(csv_out, index=False)

    # Vẽ bảng ghép Loss của tất cả các phương pháp vào figures_dir
    plot_all_methods_loss_grid(
        histories_dict=histories_dict,
        output_base_path=os.path.join(figures_dir, "fig_protocol1_all_methods_loss_grid"),
    )

    # Vẽ biểu đồ so sánh tổng hợp Accuracy & NMAE vào figures_dir
    plot_benchmark_comparison_ieee(
        df_results=df_res,
        output_base_path=os.path.join(figures_dir, "fig_protocol1_comparison_acc_mae"),
    )

    # Vẽ biểu đồ MAE từng chiều W, L, D độc lập với trục tung tối ưu
    plot_dimensional_mae_breakdown_ieee(
        df_results=df_res,
        output_base_path=os.path.join(figures_dir, "fig_protocol1_wld_mae_breakdown"),
    )

    print(f"\n[HOÀN THÀNH HƯỚNG 1] Bảng tổng hợp đã lưu vào: {csv_out}")
    print(f"                     Các bảng dự đoán chi tiết: {tables_dir}")
    print(f"                     Đồ thị IEEE 300 DPI: {figures_dir}")
    print(f"                     Checkpoints đã lưu: {checkpoints_dir}")
    return df_res, histories_dict


# =============================================================================
# HƯỚNG 2: GIAO THỨC 2 (10-FOLD LEAVE-ONE-DEFECT-OUT CROSS VALIDATION)
# =============================================================================
def run_protocol_2_kfold_defect_split(
    models_to_test: List[Dict[str, str]],
    epochs: int = 500,
    lr: float = 5e-3,
    freeze_backbone: bool = True,
    patience: int = 100,
    device: str = "cpu",
    output_dir: str = "domain_adaptation/finetune_results/protocols",
    reset_kendall: bool = True,
    early_stopping: bool = False,
    use_final_epoch: bool = False,
    use_pinn_loss: Optional[bool] = None,
    fixed_lr: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Thực thi toàn diện Hướng 2:
    - 10-Fold Leave-One-Defect-Out (LODO).
    - Mỗi fold bỏ ra 1 khuyết tật (cả 2 lần quét = 2 mẫu) KHÔNG HỌC làm Test mù.
    - Huấn luyện trên 18 mẫu của 9 khuyết tật còn lại.
    - Gom toàn bộ 20 dự đoán out-of-fold.
    - Tính TP, FP, FN, TN, Accuracy từng class và Overall Accuracy.
    - Vẽ ma trận nhầm lẫn và biểu đồ so sánh chuẩn IEEE với chú thích ở ngoài.
    """
    if use_pinn_loss is None:
        loss_mode_label = "Mỗi mô hình dùng đúng Loss tương ứng (PINN có Physics Loss, NoPINN thuần Data)"
    elif use_pinn_loss:
        loss_mode_label = "Ép buộc bật PINN Physics Loss cho tất cả mô hình"
    else:
        loss_mode_label = "Tắt hoàn toàn PINN Physics Loss (chỉ dùng Data Loss)"

    print("\n" + "=" * 80)
    print("HƯỚNG 2 (GIAO THỨC 2): 10-FOLD LEAVE-ONE-DEFECT-OUT (LODO) CROSS VALIDATION")
    print(f"Cấu hình: Freeze Backbone = {freeze_backbone} | Max Epochs = {epochs} | LR = {lr} | Early Stopping = {early_stopping} | Hàm Loss = {loss_mode_label} | Save Final = {use_final_epoch}")
    print("=" * 80)

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    summary_rows = []
    all_models_tp_fp = []

    for info in models_to_test:
        key = info["key"]
        m_type = info["type"]
        var = info["variant"]
        print(f"\n>>> [Hướng 2] Đang chạy 10-Fold LODO cho mô hình: {key} ...")

        try:
            bundle = load_5khz_fully_prepared(model_type=m_type, variant=var, device=device)
        except FileNotFoundError as e:
            print(f"\n[CẢNH BÁO] Bỏ qua mô hình {key} ({m_type.upper()}, {var}) do chưa có checkpoint: {e}")
            print(f"            (Vui lòng huấn luyện mô hình trong thư mục {m_type}/ trước khi đánh giá)")
            continue

        base_model = bundle["model"]
        y_scaler = bundle["y_scaler"]
        metadata = bundle["metadata"]

        alpha_pinn = bundle.get("alpha", 1.0)
        print(f"    [Alpha PINN] Hệ số alpha = {alpha_pinn} (load từ {os.path.basename(bundle['checkpoint_dir'])})")

        folds = get_5khz_10fold_defect_splits(bundle)
        oof_pred_shapes = [None] * len(metadata)
        oof_pred_wld = np.zeros((len(metadata), 3))

        for f_idx, fold_data in enumerate(folds, start=1):
            held_out = fold_data["held_out_crack"]
            tr = fold_data["train"]
            te = fold_data["test"]

            f_model, _ = finetune_single_model(
                model_type=m_type,
                base_model=base_model,
                X_train=tr["X"],
                y_clf_train=tr["y_clf"],
                y_wld_train=tr["y_wld_norm"],
                X_val=te["X"],
                y_clf_val=te["y_clf"] if m_type in ["cnn", "mlp"] else None,
                y_wld_val=te["y_wld_norm"],
                epochs=epochs,
                lr=lr,
                freeze_backbone=freeze_backbone,
                patience=patience,
                device=device,
                variant=var,
                y_scaler=y_scaler,
                x_scaler=bundle.get("x_scaler", None),
                reset_kendall=reset_kendall,
                alpha_pinn=alpha_pinn,
                early_stopping=early_stopping,
                use_final_epoch=use_final_epoch,
                use_pinn_loss=use_pinn_loss,
                fixed_lr=fixed_lr,
            )

            # Dự đoán out-of-fold trên 2 mẫu của fold này
            p_sh, p_wld, _, _ = predict_and_denormalize(f_model, te["X"], y_scaler=y_scaler)
            for loc_i, glob_i in enumerate(te["indices"]):
                oof_pred_shapes[glob_i] = p_sh[loc_i]
                oof_pred_wld[glob_i] = p_wld[loc_i]

            print(f"    Fold {f_idx:02d}/10: Vết nứt No.{held_out} đã hoàn tất.")

        # Gom toàn bộ 20 dự đoán out-of-fold
        true_shapes_all = [m["true_shape"] for m in metadata]
        true_wld_all = np.array([[m["true_w"], m["true_l"], m["true_d"]] for m in metadata])

        stat_oof = compute_classification_metrics_and_matrix(true_shapes_all, oof_pred_shapes)
        reg_oof = compute_regression_metrics(true_wld_all, oof_pred_wld)

        # Xuất bảng chi tiết từng mẫu khi inference (toàn bộ 20 mẫu Out-of-Fold)
        pred_records_oof = []
        for i, m in enumerate(metadata):
            pred_records_oof.append({
                "Filename": m["filename"],
                "Crack_No": m.get("crack_no", i + 1),
                "True_Shape": m["true_shape"],
                "Pred_Shape": oof_pred_shapes[i],
                "Shape_Match": (m["true_shape"] == oof_pred_shapes[i]) if m_type != "xiong" else np.nan,
                "True_W": m["true_w"], "Pred_W": round(float(oof_pred_wld[i, 0]), 4), "Err_W": round(float(abs(m["true_w"] - oof_pred_wld[i, 0])), 4),
                "True_L": m["true_l"], "Pred_L": round(float(oof_pred_wld[i, 1]), 4), "Err_L": round(float(abs(m["true_l"] - oof_pred_wld[i, 1])), 4),
                "True_D": m["true_d"], "Pred_D": round(float(oof_pred_wld[i, 2]), 4), "Err_D": round(float(abs(m["true_d"] - oof_pred_wld[i, 2])), 4),
            })
        df_oof_details = pd.DataFrame(pred_records_oof)
        oof_detail_csv = os.path.join(tables_dir, f"predictions_protocol2_{key.lower()}.csv")
        df_oof_details.to_csv(oof_detail_csv, index=False)

        # Lưu TP, FP breakdown
        df_tp_fp = stat_oof["per_class_stats"]
        df_tp_fp["Model"] = key
        all_models_tp_fp.append(df_tp_fp)

        # Vẽ Confusion Matrix Out-of-Fold 20 mẫu vào figures_dir
        cm_path = os.path.join(figures_dir, f"fig_protocol2_10fold_confusion_matrix_{key}")
        plot_confusion_matrix_ieee(
            confusion_matrix=stat_oof["confusion_matrix"],
            class_names=DEFAULT_UNIQUE_SHAPES,
            model_name=f"{key} (10-Fold LODO)",
            output_base_path=cm_path,
            acc_val=stat_oof["overall_acc"] if m_type != "xiong" else None,
        )

        # Vẽ tương quan hồi quy tuyến tính kích thước W, L, D vào figures_dir
        fit_path = os.path.join(figures_dir, f"fig_protocol2_regression_fit_{key}")
        plot_linear_fit_wld_ieee(
            true_wld=true_wld_all,
            pred_wld=oof_pred_wld,
            model_name=key,
            output_base_path=fit_path,
        )

        # Xuất đồ thị chuẩn gói figures_ieee cho Protocol 2 (Fig 2 & Fig 3 Parity Box)
        try:
            from figures_ieee import plot_fig2_confusion_matrix, plot_fig3_regression_scatter
            if m_type != "xiong":
                plot_fig2_confusion_matrix(
                    y_true=true_shapes_all,
                    y_pred=oof_pred_shapes,
                    unique_shapes=DEFAULT_UNIQUE_SHAPES,
                    output_dir=figures_dir,
                    filename_prefix=f"fig2_confusion_matrix_ieee_p2_{key}",
                )
            plot_fig3_regression_scatter(
                y_true_wld=true_wld_all,
                y_pred_wld=oof_pred_wld,
                output_dir=figures_dir,
                filename_prefix=f"fig3_regression_scatter_ieee_p2_{key}",
            )
        except Exception as e_fig:
            print(f"    [figures_ieee] Cảnh báo xuất hình: {e_fig}")

        summary_rows.append({
            "Protocol": "Protocol 2 (10-Fold LODO)",
            "Model": key,
            "Total_Samples": len(metadata),
            "Total_Correct": stat_oof["total_correct"] if m_type != "xiong" else np.nan,
            "ACC(%)": stat_oof["overall_acc"] if m_type != "xiong" else np.nan,
            "NMAE_Overall(%)": reg_oof["NMAE_Avg(%)"],
            "MAE_Overall(mm)": reg_oof["MAE_Avg"],
            "MAE_W(mm)": reg_oof["MAE_W"],
            "MAE_L(mm)": reg_oof["MAE_L"],
            "MAE_D(mm)": reg_oof["MAE_D"],
            "NMAE_W(%)": reg_oof["NMAE_W(%)"],
            "NMAE_L(%)": reg_oof["NMAE_L(%)"],
            "NMAE_D(%)": reg_oof["NMAE_D(%)"],
        })

    df_p2 = pd.DataFrame(summary_rows)
    csv_p2 = os.path.join(tables_dir, "protocol2_10fold_results.csv")
    df_p2.to_csv(csv_p2, index=False)

    df_tp_fp_all = pd.concat(all_models_tp_fp, ignore_index=True)
    csv_tp_fp = os.path.join(tables_dir, "protocol2_tp_fp_breakdown.csv")
    df_tp_fp_all.to_csv(csv_tp_fp, index=False)

    # Vẽ biểu đồ so sánh tổng hợp vào figures_dir
    plot_benchmark_comparison_ieee(
        df_results=df_p2,
        output_base_path=os.path.join(figures_dir, "fig_protocol2_comparison_acc_mae"),
    )

    # Vẽ biểu đồ MAE từng chiều W, L, D độc lập với trục tung tối ưu
    plot_dimensional_mae_breakdown_ieee(
        df_results=df_p2,
        output_base_path=os.path.join(figures_dir, "fig_protocol2_wld_mae_breakdown"),
    )

    print(f"\n[HOÀN THÀNH HƯỚNG 2] Bảng tổng hợp đã lưu vào: {csv_p2}")
    print(f"                     Bảng TP/FP Breakdown: {csv_tp_fp}")
    print(f"                     Các bảng dự đoán chi tiết: {tables_dir}")
    print(f"                     Đồ thị IEEE 300 DPI: {figures_dir}")
    return df_p2, df_tp_fp_all


# =============================================================================
# ĐIỂM VÀO DÒNG LỆNH (MAIN ENTRY POINT)
# =============================================================================
MODELS_SUITE = [
    {"key": "CNN_NoPINN",   "type": "cnn",   "variant": "nopinn"},
    {"key": "CNN_Proposed", "type": "cnn",   "variant": "pinn"},
    {"key": "MLP",          "type": "mlp",   "variant": "pinn"},
    {"key": "XIONG",        "type": "xiong", "variant": "pinn"},
]


def setup_run_directory(args: argparse.Namespace, project_root: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Tự động khởi tạo thư mục lưu trữ riêng biệt (isolated run directory) cho mỗi lần chạy / finetuning.
    Quy tắc đặt tên:
    1. Nếu người dùng chỉ định --run_name: Sử dụng trực tiếp tên đó trong finetune_results/runs/<run_name>.
    2. Nếu người dùng chỉ định --output_dir và KHÔNG có --run_name: Dùng chính xác --output_dir.
    3. Mặc định: Sinh tên duy nhất kèm thời gian và siêu tham số:
       run_YYYYMMDD_HHMMSS_<protocol>_<freeze>_<kendall>_lr<lr>_ep<epochs>[_<tag>]
    """
    finetune_base = os.path.join(project_root, "domain_adaptation", "finetune_results")
    runs_base = os.path.join(finetune_base, "runs")
    os.makedirs(runs_base, exist_ok=True)

    if getattr(args, "run_name", None) is not None and str(args.run_name).strip():
        run_name = str(args.run_name).strip()
        run_dir = os.path.join(runs_base, run_name)
    elif getattr(args, "output_dir", None) is not None and str(args.output_dir).strip():
        out = str(args.output_dir).strip()
        if not os.path.isabs(out):
            out = os.path.join(project_root, out)
        run_name = os.path.basename(out.rstrip("\\/"))
        run_dir = os.path.abspath(out)
    else:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p_str = f"p{args.protocol}" if args.protocol in ["1", "2"] else "pAll"
        frz_str = "freeze_bb" if args.freeze_backbone else "train_all"
        kd_str = "reset_kd" if args.reset_kendall else "keep_kd"
        lr_str = f"fixed_lr{args.lr:g}" if getattr(args, "fixed_lr", True) else f"cos_lr{args.lr:g}"
        ep_str = f"ep{args.epochs}"
        if args.use_pinn_loss is None:
            loss_str = "native_loss"
        elif args.use_pinn_loss:
            loss_str = "force_pinn"
        else:
            loss_str = "force_nopinn"

        parts = ["run", now_str, p_str, frz_str, kd_str, lr_str, ep_str, loss_str]
        if getattr(args, "tag", None) and str(args.tag).strip():
            parts.append(str(args.tag).strip())
        run_name = "_".join(parts)
        run_dir = os.path.join(runs_base, run_name)

    os.makedirs(os.path.join(run_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    config_dict = {
        "run_name": run_name,
        "run_dir": run_dir,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": args.protocol,
        "models": args.models,
        "epochs": args.epochs,
        "lr": args.lr,
        "fixed_lr": getattr(args, "fixed_lr", True),
        "patience": args.patience,
        "early_stopping": not args.no_early_stopping,
        "freeze_backbone": args.freeze_backbone,
        "reset_kendall": args.reset_kendall,
        "use_pinn_loss": args.use_pinn_loss,
        "save_final": args.save_final,
        "device": args.device,
    }
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    return run_dir, run_name, config_dict


def finalize_run_environment(
    run_dir: str,
    run_name: str,
    config_dict: Dict[str, Any],
    df_p1: Optional[pd.DataFrame] = None,
    df_p2: Optional[pd.DataFrame] = None,
    project_root: str = PROJECT_ROOT,
):
    """
    1. Cập nhật run_config.json với thời gian kết thúc và tóm tắt kết quả.
    2. Xuất file run_summary.txt dễ đọc trực quan.
    3. Cập nhật nhật ký các lần chạy runs_history.csv.
    4. Đồng bộ (mirror) kết quả vào finetune_results/protocols/ và finetune_results/latest/.
    """
    end_time = datetime.datetime.now()
    try:
        start_time = datetime.datetime.strptime(config_dict["start_time"], "%Y-%m-%d %H:%M:%S")
        duration_sec = (end_time - start_time).total_seconds()
    except Exception:
        duration_sec = 0.0

    config_dict["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    config_dict["duration_seconds"] = round(duration_sec, 2)

    summary_lines = [
        "=" * 80,
        f"RUN SUMMARY REPORT: {run_name}",
        "=" * 80,
        f"Thời gian bắt đầu : {config_dict['start_time']}",
        f"Thời gian kết thúc: {config_dict['end_time']} (Tổng cộng: {duration_sec:.1f}s)",
        f"Thư mục lưu trữ   : {run_dir}",
        f"Cấu hình mô hình  : Freeze Backbone = {config_dict['freeze_backbone']} | Reset Kendall = {config_dict['reset_kendall']}",
        f"Siêu tham số      : LR = {config_dict['lr']} | Epochs = {config_dict['epochs']} | Loss Mode = {'Native (Mô hình nào gọi đúng Loss đó)' if config_dict['use_pinn_loss'] is None else ('Force PINN' if config_dict['use_pinn_loss'] else 'Force NoPINN')}",
        "-" * 80,
    ]

    p1_best_acc = None
    p1_best_mae = None
    if df_p1 is not None and not df_p1.empty:
        summary_lines.append("\n[KẾT QUẢ PROTOCOL 1 - SCAN SPLIT]:")
        summary_lines.append(df_p1.to_string(index=False))
        sub1 = df_p1[(df_p1["Stage"] == "Finetuned") & (df_p1["Model"].str.contains("Proposed", case=False, na=False))]
        if not sub1.empty:
            p1_best_acc = float(sub1.iloc[0].get("ACC(%)", np.nan))
            p1_best_mae = float(sub1.iloc[0].get("MAE_Overall(mm)", np.nan))

    p2_best_acc = None
    p2_best_mae = None
    if df_p2 is not None and not df_p2.empty:
        summary_lines.append("\n[KẾT QUẢ PROTOCOL 2 - 10-FOLD LODO]:")
        summary_lines.append(df_p2.to_string(index=False))
        sub2 = df_p2[df_p2["Model"].str.contains("Proposed", case=False, na=False)]
        if not sub2.empty:
            p2_best_acc = float(sub2.iloc[0].get("ACC(%)", np.nan))
            p2_best_mae = float(sub2.iloc[0].get("MAE_Overall(mm)", np.nan))

    summary_lines.append("\n" + "=" * 80)
    summary_text = "\n".join(summary_lines)

    summary_txt_path = os.path.join(run_dir, "run_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    finetune_base = os.path.join(project_root, "domain_adaptation", "finetune_results")
    history_csv = os.path.join(finetune_base, "runs_history.csv")
    history_record = {
        "Timestamp": config_dict["start_time"],
        "Run_Name": run_name,
        "Protocol": config_dict["protocol"],
        "Models": config_dict["models"],
        "Freeze_Backbone": config_dict["freeze_backbone"],
        "Reset_Kendall": config_dict["reset_kendall"],
        "LR": config_dict["lr"],
        "Epochs": config_dict["epochs"],
        "Use_PINN_Loss": config_dict["use_pinn_loss"],
        "Duration_Sec": round(duration_sec, 1),
        "P1_Proposed_ACC(%)": p1_best_acc,
        "P1_Proposed_MAE(mm)": p1_best_mae,
        "P2_Proposed_ACC(%)": p2_best_acc,
        "P2_Proposed_MAE(mm)": p2_best_mae,
        "Run_Dir": run_dir,
    }
    df_hist_new = pd.DataFrame([history_record])
    if os.path.exists(history_csv):
        try:
            df_hist_old = pd.read_csv(history_csv)
            if df_hist_old.empty or df_hist_old.dropna(how="all").empty:
                df_hist_new.to_csv(history_csv, index=False)
            else:
                df_hist_all = pd.concat([df_hist_old, df_hist_new], ignore_index=True)
                df_hist_all.to_csv(history_csv, index=False)
        except Exception:
            df_hist_new.to_csv(history_csv, index=False)
    else:
        df_hist_new.to_csv(history_csv, index=False)

    # Đồng bộ sang finetune_results/latest và finetune_results/protocols
    latest_dir = os.path.join(finetune_base, "latest")
    protocols_dir = os.path.join(finetune_base, "protocols")
    for target in [latest_dir, protocols_dir]:
        try:
            os.makedirs(target, exist_ok=True)
            shutil.copytree(run_dir, target, dirs_exist_ok=True)
        except Exception:
            pass

    print(f"\n" + "=" * 80)
    print(f"[THÀNH CÔNG] Toàn bộ kết quả đợt chạy đã được lưu riêng biệt vào folder:")
    print(f"  >>> {run_dir}")
    print(f"  - Config JSON : {os.path.join(run_dir, 'run_config.json')}")
    print(f"  - Summary TXT : {os.path.join(run_dir, 'run_summary.txt')}")
    print(f"  - Lịch sử Run : {history_csv}")
    print(f"  - Bản sao mới : {latest_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Chương trình Đánh giá Độc lập 2 Giao thức Thích ứng miền ECT 5kHz")
    parser.add_argument("--protocol", type=str, choices=["1", "2", "all"], default="all",
                        help="Giao thức cần chạy: '1' (Scan Split), '2' (10-Fold LODO), hoặc 'all' (mặc định: 'all')")
    parser.add_argument("--models", "--model", type=str, default="cnn_proposed",
                        help="Mô hình cần chạy: 'cnn_proposed', 'cnn_nopinn', 'mlp', 'xiong', hoặc 'all' (mặc định: 'cnn_proposed')")
    parser.add_argument("--epochs", type=int, default=500, help="Số epochs finetuning (mặc định: 500)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Tốc độ học cố định (Fixed LR) khi finetuning (mặc định: 5e-3)")
    parser.add_argument("--fixed_lr", dest="fixed_lr", action="store_true", default=True,
                        help="Giữ tốc độ học (LR) cố định trong suốt quá trình finetuning (mặc định: True)")
    parser.add_argument("--use_scheduler", dest="fixed_lr", action="store_false",
                        help="Bật CosineAnnealingLR scheduler để giảm dần LR về 1e-5 (mặc định: False - dùng LR cố định)")
    parser.add_argument("--patience", type=int, default=100, help="Patience cho Early Stopping (khi bật)")
    parser.add_argument("--no_early_stopping", action="store_true", default=True,
                        help="Bỏ Early Stopping, chạy đủ số epochs (mặc định: True)")
    parser.add_argument("--early_stopping", dest="no_early_stopping", action="store_false",
                        help="Bật lại Early Stopping (mặc định: tắt Early Stopping)")
    parser.add_argument("--use_pinn_loss", dest="use_pinn_loss", action="store_true", default=None,
                        help="Ép buộc bật PINN physics loss cho tất cả các mô hình")
    parser.add_argument("--no_pinn_loss", dest="use_pinn_loss", action="store_false",
                        help="Tắt hoàn toàn PINN physics loss (chỉ dùng supervised data loss)")
    parser.add_argument("--save_best", dest="save_final", action="store_false", default=False,
                        help="Khôi phục mô hình tại best val epoch thay vì epoch cuối (mặc định: True - lấy kết quả tốt nhất)")
    parser.add_argument("--save_final", dest="save_final", action="store_true",
                        help="Lưu và đánh giá kết quả tại epoch cuối cùng thay vì checkpoint tốt nhất")
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true", default=True,
                        help="Đóng băng Backbone khi finetuning (mặc định: True)")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false",
                        help="Không đóng băng Backbone, finetune toàn bộ mô hình (mặc định: False)")
    parser.add_argument("--reset_kendall", dest="reset_kendall", action="store_true", default=True,
                        help="Reset tham số Kendall uncertainty về 0.0 (exp(-s) = 1.0) khi finetuning (mặc định: True)")
    parser.add_argument("--no_reset_kendall", dest="reset_kendall", action="store_false",
                        help="Bảo tồn nguyên vẹn tham số Kendall uncertainty từ pretrained checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Tên thư mục run riêng biệt (ví dụ: 'exp_freeze_lr1e4'). Nếu không cung cấp, hệ thống tự động sinh tên thư mục kèm timestamp và tham số cấu hình.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Hậu tố tag bổ sung vào tên thư mục run tự động (ví dụ: 'trial1', 'test_freeze')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Thư mục lưu kết quả trực tiếp nếu muốn ghi đè đường dẫn cố định.")
    args = parser.parse_args()

    set_reproducible_seed(42)
    reset_kendall = args.reset_kendall
    early_stopping = not args.no_early_stopping
    use_final_epoch = args.save_final
    use_pinn_loss = args.use_pinn_loss

    # Lọc mô hình cần chạy
    selected_arg = args.models.strip().lower()
    if selected_arg in ["all", ""]:
        models_to_run = MODELS_SUITE
    else:
        alias_map = {
            "cnn_proposed": "CNN_Proposed",
            "proposed": "CNN_Proposed",
            "pinn": "CNN_Proposed",
            "cnn_nopinn": "CNN_NoPINN",
            "nopinn": "CNN_NoPINN",
            "mlp": "MLP",
            "xiong": "XIONG",
        }
        req_keys = [alias_map.get(k.strip(), k.strip()).lower() for k in selected_arg.split(",")]
        models_to_run = [m for m in MODELS_SUITE if m["key"].lower() in req_keys or m["type"].lower() in req_keys]
        if not models_to_run:
            raise ValueError(f"Không tìm thấy mô hình phù hợp với: '{args.models}'. Lựa chọn hợp lệ: {[m['key'] for m in MODELS_SUITE]}")

    if use_pinn_loss is None:
        loss_mode_str = "Tự động theo bản thể mô hình (PINN gọi Physics PINN Loss, NoPINN gọi thuần Data Loss)"
    elif use_pinn_loss:
        loss_mode_str = "Ép buộc bật PINN Physics Loss cho toàn bộ mô hình"
    else:
        loss_mode_str = "Ép buộc tắt PINN Physics Loss (Chỉ dùng Supervised Data Loss)"
    print(f"\n[DANH SÁCH MÔ HÌNH THỰC THI]: {[m['key'] for m in models_to_run]}")
    print(f"[CẤU HÌNH BENCHMARK]: Epochs = {args.epochs} | Early Stopping = {early_stopping} | Reset Kendall = {reset_kendall} | Hàm Loss = {loss_mode_str} | Lưu kết quả = {'Epoch cuối cùng' if use_final_epoch else 'Best Val Loss (Mặc định)'}")

    # Tự động khởi tạo thư mục lưu trữ riêng biệt (isolated run directory)
    output_dir, run_name, config_dict = setup_run_directory(args, PROJECT_ROOT)
    print(f"[OUTPUT FOLDER RIÊNG]: {output_dir}\n")

    df_p1 = None
    df_p2 = None

    if args.protocol in ["1", "all"]:
        df_p1, _ = run_protocol_1_scan_split(
            models_to_test=models_to_run,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone=args.freeze_backbone,
            patience=args.patience,
            device=args.device,
            output_dir=output_dir,
            reset_kendall=reset_kendall,
            early_stopping=early_stopping,
            use_final_epoch=use_final_epoch,
            use_pinn_loss=use_pinn_loss,
            fixed_lr=args.fixed_lr,
        )

    if args.protocol in ["2", "all"]:
        df_p2, _ = run_protocol_2_kfold_defect_split(
            models_to_test=models_to_run,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone=args.freeze_backbone,
            patience=args.patience,
            device=args.device,
            output_dir=output_dir,
            reset_kendall=reset_kendall,
            early_stopping=early_stopping,
            use_final_epoch=use_final_epoch,
            use_pinn_loss=use_pinn_loss,
            fixed_lr=args.fixed_lr,
        )

    # Nếu chạy cả 2 giao thức, tự động vẽ đồ thị so sánh trực diện kết quả FINETUNED giữa 2 cách chia data
    if df_p1 is not None and df_p2 is not None:
        cross_fig_path = os.path.join(output_dir, "figures", "fig_comparison_finetuned_two_protocols")
        plot_finetuned_two_protocols_comparison(df_p1, df_p2, cross_fig_path)
        print(f"\n[OK] Đã vẽ đồ thị so sánh kết quả Finetuned giữa 2 cách chia data vào: {cross_fig_path}.png & .pdf")

    # Tổng kết và hoàn tất đợt chạy: ghi summary, history csv và đồng bộ kết quả
    finalize_run_environment(
        run_dir=output_dir,
        run_name=run_name,
        config_dict=config_dict,
        df_p1=df_p1,
        df_p2=df_p2,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
