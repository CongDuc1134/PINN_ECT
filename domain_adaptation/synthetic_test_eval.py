# -*- coding: utf-8 -*-
"""
===============================================================================
SYNTHETIC TEST SET EVALUATION & BACKWARD TRANSFER (FORGETTING DIAGNOSIS)
===============================================================================
Module đánh giá hiện tượng "Quên tri thức cũ" (Catastrophic Forgetting) bằng cách
kiểm thử ngược lại (Backward Transfer) trên tập Test mô phỏng Synthetic (3,306 mẫu):
1. Nạp tập Test 10% chuẩn được sinh từ Crack_Shape_Images & labels.csv (seed=42).
2. Tự động lưu cache nén (synthetic_test_cache.npz) sau lần nạp đầu tiên để nạp tức thì trong 0.1s.
3. Đánh giá mô hình Pretrained (Zero-shot) và mô hình Finetuned trên tập Test mô phỏng.
4. Đo lường chính xác mức độ suy thoái (Delta Accuracy, Delta MAE) và chẩn đoán:
   - Bảo toàn tri thức (Retention Success)
   - Suy thoái nhẹ (Minor Drift)
   - Quên thảm khốc (Catastrophic Forgetting)
5. Xuất bảng báo cáo tổng hợp vào tables/ để phục vụ viết bài báo Q1.
===============================================================================
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.load_real_experiment_data import DEFAULT_UNIQUE_SHAPES
from domain_adaptation.model_loader import predict_and_denormalize


def get_synthetic_data_paths() -> Tuple[str, str]:
    """Tìm đường dẫn đến Crack_Shape_Images và labels.csv."""
    data_path = os.environ.get("DATA_PATH", os.path.join(PROJECT_ROOT, "Crack_Shape_Images"))
    labels_path = os.environ.get("LABELS_PATH", os.path.join(PROJECT_ROOT, "labels.csv"))
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh Crack_Shape_Images: {data_path}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Không tìm thấy file nhãn labels.csv: {labels_path}")
    return data_path, labels_path


def get_or_create_synthetic_test_cache(
    cache_path: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Nạp tập Test mô phỏng Synthetic (10% = 3,306 mẫu) từ cache hoặc sinh mới nếu chưa có.
    Trả về:
        (X_test_raw, y_test_wld, y_shape_test, unique_shapes)
    """
    if cache_path is None:
        cache_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "synthetic_test_cache.npz")

    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            X_test = cached["X_test"]
            y_test_wld = cached["y_test_wld"]
            y_shape_test = cached["y_shape_test"]
            unique_shapes = list(cached["unique_shapes"])
            return X_test, y_test_wld, y_shape_test, unique_shapes
        except Exception as e:
            print(f"    [Cache Notice] Lỗi đọc file cache ({e}), sẽ sinh mới lại tập test.")

    print("\n" + "=" * 78)
    print("  [SYNTHETIC TEST SET] KHỞI TẠO TẬP KIỂM THỬ MÔ PHỎNG (3,306 MẪU - SEED 42)")
    print("  (Quá trình nạp 33,060 file chỉ thực hiện 1 lần duy nhất, sau đó sẽ lưu cache)")
    print("=" * 78)

    from sklearn.model_selection import train_test_split
    import importlib.util
    lib_path = os.path.join(PROJECT_ROOT, "cnn", "library_functions.py")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Không tìm thấy file library_functions.py tại: {lib_path}")
    spec = importlib.util.spec_from_file_location("root_cnn_library_functions", lib_path)
    lib_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib_mod)
    Load_Data_With_Labels = lib_mod.Load_Data_With_Labels

    data_path, labels_path = get_synthetic_data_paths()
    X, y, matched_filenames = Load_Data_With_Labels(data_path, labels_path)

    # Trích xuất nhãn hình dạng (Shape Labels)
    labels_df = pd.read_csv(labels_path)
    filename_to_shape = {}
    for _, row in labels_df.iterrows():
        try:
            fn = str(row["filename"]).strip()
            sh = str(row["shape"]).strip()
            if "type" in labels_df.columns and sh == "Step":
                type_val = str(row["type"]).strip()
                if type_val and type_val.lower() != "nan":
                    sh = f"Step_{type_val}"
            if sh and sh.lower() != "nan":
                filename_to_shape[fn] = sh
        except Exception:
            continue

    y_shape_str = [filename_to_shape.get(f, f.split("_")[0]) for f in matched_filenames]
    unique_shapes = sorted(list(set(y_shape_str)))
    shape_to_idx = {shape: i for i, shape in enumerate(unique_shapes)}
    y_shape = np.array([shape_to_idx[s] for s in y_shape_str])

    # Tách chính xác 10% test bằng train_test_split (seed=42)
    _, X_test, _, y_test_wld, _, y_shape_test = train_test_split(
        X, y, y_shape, test_size=0.10, random_state=random_state
    )

    # Lưu cache nén
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            X_test=X_test,
            y_test_wld=y_test_wld,
            y_shape_test=y_shape_test,
            unique_shapes=unique_shapes,
        )
        print(f"  [SYNTHETIC TEST SET] Đã lưu cache thành công vào: {cache_path}")
        print(f"                       (Kích thước: {len(X_test):,} mẫu | Các lần chạy sau sẽ nạp trong 0.1s)\n")
    except Exception as e_save:
        print(f"  [CẢNH BÁO] Không thể lưu cache: {e_save}")

    return X_test, y_test_wld, y_shape_test, unique_shapes


def evaluate_model_on_synthetic_data(
    model: nn.Module,
    bundle: Dict[str, Any],
    device: str = "cpu",
    batch_size: int = 256,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thực hiện suy luận và tính toán chỉ số chi tiết trên tập Test mô phỏng Synthetic (3,306 mẫu).
    """
    X_test_raw, y_test_wld, y_shape_test, unique_shapes = get_or_create_synthetic_test_cache(cache_path=cache_path)
    n_samples = len(X_test_raw)
    x_scaler = bundle.get("x_scaler")
    y_scaler = bundle.get("y_scaler")
    is_1d = bundle.get("is_1d", False)

    # 1. Chuẩn hóa X bằng x_scaler
    if x_scaler is not None and hasattr(x_scaler, "transform"):
        X_flat = X_test_raw.reshape(-1, 2)
        X_flat_norm = x_scaler.transform(X_flat)
        X_norm = X_flat_norm.reshape(n_samples, 32, 32, 2)
    else:
        X_norm = X_test_raw

    # Chuyển định dạng Tensor: (N, 2, 32, 32) cho CNN hoặc (N, 2048) cho 1D
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).permute(0, 3, 1, 2)
    if is_1d:
        X_tensor = X_tensor.reshape(n_samples, -1)

    model.eval()
    pred_shapes_all = []
    pred_wld_list = []

    # 2. Suy luận theo mini-batch để tối ưu bộ nhớ GPU/CPU
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = X_tensor[start_idx:end_idx].to(device)
            p_sh, p_wld, _, _ = predict_and_denormalize(model, batch_x, y_scaler=y_scaler)
            pred_shapes_all.extend(p_sh)
            pred_wld_list.append(p_wld)

    pred_wld_all = np.vstack(pred_wld_list)
    true_shapes_all = [unique_shapes[idx] for idx in y_shape_test]

    # 3. Tính toán các chỉ số
    # Accuracy phân loại hình dạng
    correct_cnt = sum(1 for ts, ps in zip(true_shapes_all, pred_shapes_all) if ts == ps)
    acc = (correct_cnt / n_samples * 100.0) if n_samples > 0 else 0.0

    # Sai số hồi quy kích thước MAE (mm) và NMAE (%)
    err = np.abs(pred_wld_all - y_test_wld)
    mae_w = float(np.mean(err[:, 0]))
    mae_l = float(np.mean(err[:, 1]))
    mae_d = float(np.mean(err[:, 2]))
    mae_avg = float(np.mean(err))

    mean_w = float(np.mean(y_test_wld[:, 0]))
    mean_l = float(np.mean(y_test_wld[:, 1]))
    mean_d = float(np.mean(y_test_wld[:, 2]))

    nmae_w = (mae_w / mean_w * 100.0) if mean_w > 0 else 0.0
    nmae_l = (mae_l / mean_l * 100.0) if mean_l > 0 else 0.0
    nmae_d = (mae_d / mean_d * 100.0) if mean_d > 0 else 0.0
    nmae_avg = (nmae_w + nmae_l + nmae_d) / 3.0

    return {
        "n_samples": n_samples,
        "total_correct": correct_cnt,
        "ACC(%)": round(acc, 2),
        "MAE_Overall(mm)": round(mae_avg, 4),
        "MAE_W(mm)": round(mae_w, 4),
        "MAE_L(mm)": round(mae_l, 4),
        "MAE_D(mm)": round(mae_d, 4),
        "NMAE_Overall(%)": round(nmae_avg, 2),
        "NMAE_W(%)": round(nmae_w, 2),
        "NMAE_L(%)": round(nmae_l, 2),
        "NMAE_D(%)": round(nmae_d, 2),
    }


def evaluate_and_log_forgetting(
    pretrained_model: nn.Module,
    finetuned_model: nn.Module,
    model_name: str,
    bundle: Dict[str, Any],
    output_dir: str,
    protocol_name: str = "protocol1",
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Đánh giá độ suy giảm tri thức (Catastrophic Forgetting / Backward Transfer)
    giữa mô hình Pretrained và Finetuned trên tập Test mô phỏng Synthetic (3,306 mẫu).
    """
    print("\n" + "-" * 78)
    print(f"  [KIỂM ĐỊNH QUÊN TRI THỨC] Đánh giá trên Tập Test Mô phỏng Synthetic (3,306 mẫu)")
    print(f"  Mô hình: {model_name} | Giao thức: {protocol_name}")
    print("-" * 78)

    # Đánh giá Pretrained Zero-shot
    res_pre = evaluate_model_on_synthetic_data(pretrained_model, bundle, device=device)
    # Đánh giá Finetuned
    res_post = evaluate_model_on_synthetic_data(finetuned_model, bundle, device=device)

    # Tính độ lệch
    diff_acc = res_post["ACC(%)"] - res_pre["ACC(%)"]
    diff_mae = res_post["MAE_Overall(mm)"] - res_pre["MAE_Overall(mm)"]
    diff_nmae = res_post["NMAE_Overall(%)"] - res_pre["NMAE_Overall(%)"]

    # Chẩn đoán trạng thái
    if diff_acc >= -2.0 and diff_mae <= 0.02:
        diagnosis = "Bảo toàn tri thức (Không bị quên - Retention Success)"
    elif diff_acc >= -5.0 and diff_mae <= 0.05:
        diagnosis = "Suy thoái nhẹ chấp nhận được (Minor Drift)"
    else:
        diagnosis = "Có dấu hiệu quên tri thức cũ (Catastrophic Forgetting)"

    df_report = pd.DataFrame([
        {
            "Protocol": protocol_name,
            "Model": model_name,
            "Stage": "Pretrained (Zero-shot)",
            "Synthetic_Samples": res_pre["n_samples"],
            "ACC(%)": res_pre["ACC(%)"],
            "MAE_Avg(mm)": res_pre["MAE_Overall(mm)"],
            "NMAE_Avg(%)": res_pre["NMAE_Overall(%)"],
            "MAE_W(mm)": res_pre["MAE_W(mm)"],
            "MAE_L(mm)": res_pre["MAE_L(mm)"],
            "MAE_D(mm)": res_pre["MAE_D(mm)"],
            "Status": "Baseline (Gốc)",
        },
        {
            "Protocol": protocol_name,
            "Model": model_name,
            "Stage": "Finetuned (Sau thích ứng)",
            "Synthetic_Samples": res_post["n_samples"],
            "ACC(%)": res_post["ACC(%)"],
            "MAE_Avg(mm)": res_post["MAE_Overall(mm)"],
            "NMAE_Avg(%)": res_post["NMAE_Overall(%)"],
            "MAE_W(mm)": res_post["MAE_W(mm)"],
            "MAE_L(mm)": res_post["MAE_L(mm)"],
            "MAE_D(mm)": res_post["MAE_D(mm)"],
            "Status": diagnosis,
        },
    ])

    print(f"  * Pretrained (Zero-shot) : ACC = {res_pre['ACC(%)']}% | MAE = {res_pre['MAE_Overall(mm)']} mm | NMAE = {res_pre['NMAE_Overall(%)']}%")
    print(f"  * Finetuned (Sau thích ứng): ACC = {res_post['ACC(%)']}% | MAE = {res_post['MAE_Overall(mm)']} mm | NMAE = {res_post['NMAE_Overall(%)']}%")
    print(f"  * Biến thiên (Delta)    : ΔACC = {diff_acc:+.2f}% | ΔMAE = {diff_mae:+.4f} mm | ΔNMAE = {diff_nmae:+.2f}%")
    print(f"  >>> KẾT LUẬN CHẨN ĐOÁN  : {diagnosis}")
    print("-" * 78)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_file = os.path.join(tables_dir, f"synthetic_test_forgetting_{protocol_name}_{model_name.lower()}.csv")
    df_report.to_csv(csv_file, index=False)
    print(f"    [Báo cáo Quên tri thức] Đã lưu vào: {os.path.basename(csv_file)}")

    return df_report
