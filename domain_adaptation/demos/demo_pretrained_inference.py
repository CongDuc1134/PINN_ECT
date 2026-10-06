# -*- coding: utf-8 -*-
"""
===============================================================================
DEMO: NẠP MODEL ĐÃ HUẤN LUYỆN TỪ CNN/ VÀ MLP/, NẠP SCALER VÀ DỮ LIỆU THỰC TẾ
===============================================================================
Minh họa toàn diện:
1. Nạp trọng số tốt nhất (best_model_pytorch.pth) từ thư mục gốc cnn/ và mlp/.
2. Nạp bộ chuẩn hóa tương ứng (X_scaler.pkl, y_scaler.pkl).
3. Nạp dữ liệu thực nghiệm (Experiment_1) và chuẩn hóa trực tiếp bằng X_scaler.
4. Chạy dự đoán và giải chuẩn hóa (inverse_transform) kích thước W, L, D về đơn vị mm.
5. So sánh trực tiếp với nhãn thật (Ground Truth từ Table 2 Le et al., 2013).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Thiết lập đường dẫn thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import (
    load_pretrained_model,
    load_real_data_normalized,
    predict_and_denormalize,
    find_trained_checkpoint_dir,
)


def evaluate_model_pipeline(model_type="cnn", split="5khz", train_pct="10pct"):
    print("\n" + "=" * 80)
    print(f"KIỂM THỬ: {model_type.upper()} PRETRAINED TRÊN TẬP THỰC NGHIỆM {split.upper()} (TRAIN_PCT={train_pct})")
    print("=" * 80)

    # 1. Nạp mô hình và bộ scaler đã lưu trong quá trình huấn luyện
    model, x_scaler, y_scaler, ckpt_dir = load_pretrained_model(
        model_type=model_type,
        train_pct=train_pct,
        variant="pinn",
        device="cpu",
    )

    # 2. Nạp dữ liệu thực nghiệm và chuẩn hóa bằng X_scaler của mô hình
    X_norm, y_clf, y_wld_raw, metadata, _ = load_real_data_normalized(
        model_type=model_type,
        split=split,
        x_scaler=x_scaler,
        device="cpu",
    )

    # 3. Dự đoán và giải chuẩn hóa W, L, D về mm
    pred_shapes, pred_wld_mm, logits, _ = predict_and_denormalize(
        model=model,
        X_normalized=X_norm,
        y_scaler=y_scaler,
    )

    # 4. Tổng hợp và so sánh với Ground Truth
    rows = []
    for i, meta in enumerate(metadata):
        true_shape = meta["true_shape"]
        pred_shape = pred_shapes[i]
        shape_ok = (true_shape == pred_shape) if pred_shape != "N/A" else None

        true_w, true_l, true_d = meta["true_w"], meta["true_l"], meta["true_d"]
        pw, pl, pd_val = pred_wld_mm[i]

        err_w = abs(true_w - pw) if not np.isnan(true_w) else np.nan
        err_l = abs(true_l - pl) if not np.isnan(true_l) else np.nan
        err_d = abs(true_d - pd_val) if not np.isnan(true_d) else np.nan

        rows.append({
            "File": meta["filename"],
            "Crack": f"No.{meta['crack_no']}",
            "True_Shape": true_shape,
            "Pred_Shape": pred_shape,
            "Match": shape_ok,
            "True_W": round(true_w, 2), "Pred_W": round(float(pw), 2), "Err_W": round(float(err_w), 2),
            "True_L": round(true_l, 2), "Pred_L": round(float(pl), 2), "Err_L": round(float(err_l), 2),
            "True_D": round(true_d, 2), "Pred_D": round(float(pd_val), 2), "Err_D": round(float(err_d), 2),
        })

    df = pd.DataFrame(rows)
    print("\n--- BẢNG SO SÁNH 5 MẪU ĐẦU TIÊN ---")
    cols_display = ["File", "Crack", "True_Shape", "Pred_Shape", "Match", "True_W", "Pred_W", "True_L", "Pred_L", "True_D", "Pred_D"]
    print(df[cols_display].head(5).to_string(index=False))

    # Tính độ chính xác và MAE tổng hợp
    if model_type != "xiong":
        acc = df["Match"].mean() * 100.0
        print(f"\n[+] Độ chính xác phân loại hình học: {acc:.2f}%")
    mae_w = df["Err_W"].mean()
    mae_l = df["Err_L"].mean()
    mae_d = df["Err_D"].mean()
    mae_overall = (mae_w + mae_l + mae_d) / 3.0
    print(f"[+] Sai số kích thước trung bình (MAE):")
    print(f"    - W (Width)  : {mae_w:.4f} mm")
    print(f"    - L (Length) : {mae_l:.4f} mm")
    print(f"    - D (Depth)  : {mae_d:.4f} mm")
    print(f"    -> Overall MAE: {mae_overall:.4f} mm")


if __name__ == "__main__":
    try:
        evaluate_model_pipeline(model_type="cnn", split="5khz", train_pct="10pct")
        evaluate_model_pipeline(model_type="mlp", split="10khz", train_pct="10pct")
        evaluate_model_pipeline(model_type="xiong", split="20khz", train_pct="10pct")
        print("\n" + "=" * 80)
        print("[THÀNH CÔNG] ĐÃ NẠP MÔ HÌNH PRETRAINED, BỘ SCALER VÀ DỮ LIỆU THỰC TẾ CHUẨN HÓA THÀNH CÔNG!")
        print("=" * 80)
    except Exception as e:
        import traceback
        traceback.print_exc()
