# -*- coding: utf-8 -*-
"""
===============================================================================
DEMO: NẠP DỮ LIỆU ĐÃ NORM CẢ LABEL LẪN DATA VÀ NẠP TRỌNG SỐ GỐC CỦA MODEL
===============================================================================
Kịch bản minh họa đúng yêu cầu:
1. Nạp trọng số mô hình gốc (best_model_pytorch.pth) từ folder cnn/ và mlp/.
2. Nạp bộ chuẩn hóa gốc (X_scaler: StandardScaler, y_scaler: SeparateMaxScaler).
3. Chuẩn hóa đồng thời cả 2:
   - Data đầu vào:    X_norm = X_scaler.transform(X_real)
   - Nhãn đầu ra:     y_wld_norm = y_scaler.transform(y_wld_real)
4. Đưa X_norm vào mô hình và tính hàm Loss / sai số trực tiếp trên không gian NORM
   bằng chính các tham số bất định Kendall (log_var_clf, log_var_w, log_var_l, log_var_d)
   được học từ trọng số gốc.
5. Giải chuẩn hóa kết quả về đơn vị vật lý (mm) và kiểm chứng.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Thiết lập đường dẫn thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import (
    load_real_data_fully_normalized,
    predict_and_denormalize,
)
from domain_adaptation.cnn import CNNTotalLoss
from domain_adaptation.mlp import MultitaskMLPTotalLoss
from domain_adaptation.xiong import XiongTotalLoss


def demo_full_normalized_pipeline(model_type="cnn", split="5khz", train_pct="10pct"):
    print("\n" + "=" * 85)
    print(f"PIPELINE: {model_type.upper()} VỚI DATA & LABEL ĐÃ NORM + TRỌNG SỐ HUẤN LUYỆN GỐC")
    print(f"Tập dữ liệu: {split.upper()} | Cấu hình huấn luyện gốc: train_{train_pct}")
    print("=" * 85)

    # 1. Nạp toàn bộ: Model trọng số gốc + Scaler gốc + X_norm + y_wld_norm + y_clf
    bundle = load_real_data_fully_normalized(
        model_type=model_type,
        split=split,
        train_pct=train_pct,
        variant="pinn",
        device="cpu",
    )

    model = bundle["model"]
    X_norm = bundle["X_norm"]
    y_clf = bundle["y_clf"]
    y_wld_norm = bundle["y_wld_norm"]
    y_wld_raw = bundle["y_wld_raw"]
    metadata = bundle["metadata"]
    x_scaler = bundle["x_scaler"]
    y_scaler = bundle["y_scaler"]
    ckpt_dir = bundle["checkpoint_dir"]

    print(f"\n[*] Kiểm tra kích thước và miền giá trị đã chuẩn hóa:")
    print(f"    - X_norm shape        : {X_norm.shape}")
    print(f"      Miền giá trị X_norm : [{X_norm.min().item():.3f}, {X_norm.max().item():.3f}] (qua StandardScaler)")
    print(f"    - y_wld_norm shape    : {y_wld_norm.shape}")
    print(f"      Miền giá trị y_norm : [{y_wld_norm.min().item():.3f}, {y_wld_norm.max().item():.3f}] (chia cho [W_max, L_max, D_max])")
    print(f"    - y_clf shape         : {y_clf.shape} (Nhãn hình dạng 0..4)")
    print(f"    - Trọng số gốc từ     : {os.path.basename(ckpt_dir)}")

    # 2. Forward pass qua mô hình với trọng số gốc
    pred_shapes, pred_wld_mm, pred_logits, pred_wld_norm = predict_and_denormalize(
        model=model,
        X_normalized=X_norm,
        y_scaler=y_scaler,
    )

    # 3. Tính sai số và Loss trực tiếp trên KHÔNG GIAN NORM
    print(f"\n[*] Tính toán trên không gian chuẩn hóa (Normalized Loss Space):")
    if model_type == "cnn":
        loss_fn = CNNTotalLoss(alpha=0.0, use_pinn=False)
        total_loss, metrics = loss_fn(
            shape_logits=pred_logits,
            shape_targets=y_clf,
            pred_wld=pred_wld_norm,
            true_wld=y_wld_norm,
            log_var_clf=model.log_var_clf,
            log_var_w=model.log_var_w,
            log_var_l=model.log_var_l,
            log_var_d=model.log_var_d,
        )
        print(f"    - Total Data Loss (Kendall 4-var) : {total_loss.item():.4f}")
        print(f"    - CE Classification Loss           : {metrics['loss_clf']:.4f}")
        print(f"    - MSE Width (Normalized)          : {metrics['w_loss']:.4f}")
        print(f"    - MSE Length (Normalized)         : {metrics['l_loss']:.4f}")
        print(f"    - MSE Depth (Normalized)          : {metrics['d_loss']:.4f}")

    elif model_type == "mlp":
        loss_fn = MultitaskMLPTotalLoss(alpha=0.0, use_pinn=False)
        total_loss, metrics = loss_fn(
            shape_logits=pred_logits,
            shape_targets=y_clf,
            pred_wld=pred_wld_norm,
            true_wld=y_wld_norm,
            log_var_clf=model.log_var_clf,
            log_var_reg=model.log_var_reg,
        )
        print(f"    - Total Data Loss (Kendall 2-var) : {total_loss.item():.4f}")
        print(f"    - CE Classification Loss           : {metrics['loss_clf']:.4f}")
        print(f"    - Avg MSE Regression (Normalized) : {metrics['avg_reg_loss']:.4f}")

    elif model_type == "xiong":
        loss_fn = XiongTotalLoss(alpha=0.0, use_pinn=False)
        total_loss, metrics = loss_fn(
            pred_wld=pred_wld_norm,
            true_wld=y_wld_norm,
        )
        print(f"    - Total MSE Loss (Normalized)     : {total_loss.item():.4f}")
        print(f"    - MSE Width (Normalized)          : {metrics['w_loss']:.4f}")
        print(f"    - MSE Length (Normalized)         : {metrics['l_loss']:.4f}")
        print(f"    - MSE Depth (Normalized)          : {metrics['d_loss']:.4f}")

    # 4. Bảng so sánh chi tiết: Kích thước Norm vs Kích thước Thật (mm)
    rows = []
    y_raw_np = y_wld_raw.numpy()
    y_norm_np = y_wld_norm.numpy()
    p_norm_np = pred_wld_norm.numpy()

    for i in range(len(metadata)):
        meta = metadata[i]
        rows.append({
            "File": meta["filename"],
            "True_Shape": meta["true_shape"],
            "Pred_Shape": pred_shapes[i],
            "Norm_True_W": round(float(y_norm_np[i, 0]), 3),
            "Norm_Pred_W": round(float(p_norm_np[i, 0]), 3),
            "True_W(mm)": round(float(y_raw_np[i, 0]), 2),
            "Pred_W(mm)": round(float(pred_wld_mm[i, 0]), 2),
            "True_D(mm)": round(float(y_raw_np[i, 2]), 2),
            "Pred_D(mm)": round(float(pred_wld_mm[i, 2]), 2),
        })

    df = pd.DataFrame(rows)
    print("\n--- BẢNG SO SÁNH 4 MẪU ĐẠI DIỆN TRONG KHÔNG GIAN NORM VÀ VẬT LÝ ---")
    print(df.head(4).to_string(index=False))


if __name__ == "__main__":
    demo_full_normalized_pipeline(model_type="cnn", split="5khz", train_pct="10pct")
    demo_full_normalized_pipeline(model_type="mlp", split="10khz", train_pct="10pct")
    demo_full_normalized_pipeline(model_type="xiong", split="20khz", train_pct="10pct")
    print("\n" + "=" * 85)
    print("[HOÀN TẤT] NẠP DỮ LIỆU ĐÃ NORM ĐỒNG THỜI DATA & LABEL VỚI TRỌNG SỐ GỐC THÀNH CÔNG 100%!")
    print("=" * 85)
