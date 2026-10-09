# -*- coding: utf-8 -*-
"""
===============================================================================
DEDICATED DATA LOADER FOR 5kHz REAL EXPERIMENT DATA (SIM-TO-REAL ECT)
===============================================================================
Module chuyên dụng nạp dữ liệu thực nghiệm 5kHz từ Experiment_1 phục vụ
2 hướng nghiên cứu thích ứng miền (Domain Adaptation):
1. Hướng 1: Finetune trên Scan 1 (10 mẫu) và kiểm thử mù trên Scan 2 (10 mẫu).
2. Hướng 2: 10-Fold Leave-One-Defect-Out (LODO) Cross Validation:
   Mỗi fold bỏ 1 khuyết tật (cả 2 lần quét = 2 mẫu) không học, finetune trên 18 mẫu,
   dự đoán out-of-fold để gom toàn bộ 20 mẫu tính TP, FP, TN, FN và Accuracy.

Sử dụng trực tiếp pipeline tiền xử lý từ `load_real_experiment_data.py`.
===============================================================================
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Đảm bảo đường dẫn thư mục gốc repo được thêm vào sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import từ load_real_experiment_data
try:
    from domain_adaptation.load_real_experiment_data import (
        find_experiment_1_dir,
        load_real_experiment_for_inference,
        preprocess_single_real_image,
        denormalize_regression_predictions,
        DEFAULT_UNIQUE_SHAPES,
        TABLE_2_GROUND_TRUTH,
        SeparateMaxScaler,
        MaxScaler,
    )
except ImportError:
    from load_real_experiment_data import (
        find_experiment_1_dir,
        load_real_experiment_for_inference,
        preprocess_single_real_image,
        denormalize_regression_predictions,
        DEFAULT_UNIQUE_SHAPES,
        TABLE_2_GROUND_TRUTH,
        SeparateMaxScaler,
        MaxScaler,
    )


def get_experiment_1_5khz_dir() -> str:
    """
    Xác định chính xác đường dẫn thư mục 5kHz trong Experiment_1.
    """
    exp_base = find_experiment_1_dir()
    if exp_base is None:
        raise FileNotFoundError(
            "Không tìm thấy thư mục Experiment_1 trong hệ thống! "
            "Vui lòng thiết lập biến môi trường EXP1_DIR hoặc đặt Experiment_1 tại thư mục gốc."
        )

    cand_5k = os.path.join(exp_base, "5khz")
    if os.path.isdir(cand_5k):
        return cand_5k

    cand_training = os.path.join(exp_base, "Training")
    if os.path.isdir(cand_training):
        return cand_training

    cand_train = os.path.join(exp_base, "Trainning")
    if os.path.isdir(cand_train):
        return cand_train

    raise FileNotFoundError(f"Không tìm thấy thư mục con '5khz', 'Training' hoặc 'Trainning' bên trong: {exp_base}")


class Real5kHzDataset(Dataset):
    """
    PyTorch Dataset chứa dữ liệu thực nghiệm 5kHz chuẩn hóa.
    Tự động căn chỉnh tensor:
    - 2D Tensor (2, 32, 32) cho CNN.
    - 1D Tensor (2048,) cho MLP hoặc Xiong et al.
    """
    def __init__(
        self,
        X_tensor: torch.Tensor,
        y_clf: torch.Tensor,
        y_wld_norm: torch.Tensor,
        y_wld_raw: torch.Tensor,
        metadata: List[Dict[str, Any]],
    ):
        self.X_tensor = X_tensor
        self.y_clf = y_clf
        self.y_wld_norm = y_wld_norm
        self.y_wld_raw = y_wld_raw
        self.metadata = metadata

    def __len__(self) -> int:
        return self.X_tensor.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return (
            self.X_tensor[idx],
            self.y_clf[idx],
            self.y_wld_norm[idx],
            self.metadata[idx],
        )


def load_all_5khz_samples(
    is_1d: bool = False,
    x_scaler: Optional[Any] = None,
    y_scaler: Optional[Any] = None,
    scale_factor: float = 1.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Nạp toàn bộ 20 mẫu thực nghiệm ở tần số 5kHz (10 vết nứt x 2 lượt quét lặp).

    Tham số:
        is_1d: True nếu làm phẳng thành (N, 2048) cho MLP/Xiong, False giữ (N, 2, 32, 32) cho CNN.
        x_scaler: StandardScaler từ checkpoint để chuẩn hóa đầu vào X.
        y_scaler: Scaler để chuẩn hóa nhãn kích thước W, L, D.
        scale_factor: Hệ số co dãn biên độ (mặc định 1.0).
        device: Thiết bị lưu trữ tensor ('cpu' hoặc 'cuda').

    Trả về:
        Dict gồm:
            - 'X_tensor': Tensor đầu vào (20, 2, 32, 32) hoặc (20, 2048)
            - 'y_clf': Tensor nhãn hình dạng (20,) kiểu torch.long
            - 'y_wld_norm': Tensor nhãn kích thước đã chuẩn hóa (20, 3)
            - 'y_wld_raw': Tensor kích thước thực tế mm (20, 3)
            - 'metadata': Danh sách 20 dict thông tin chi tiết từng file
            - 'x_scaler': Bộ chuẩn hóa đầu vào được sử dụng
            - 'y_scaler': Bộ chuẩn hóa nhãn được sử dụng
            - 'data_dir': Đường dẫn thư mục 5kHz
    """
    data_dir_5k = get_experiment_1_5khz_dir()

    # Nạp qua hàm nạp chuẩn của module load_real_experiment_data
    X_tensor, metadata = load_real_experiment_for_inference(
        target_path=data_dir_5k,
        x_scaler=x_scaler,
        scale_factor=scale_factor,
        device=device,
        freq_filter="5khz",
    )

    if is_1d:
        X_tensor = X_tensor.reshape(X_tensor.size(0), -1)

    y_clf_list = []
    y_wld_raw_list = []

    for m in metadata:
        cid = m.get("true_shape_id", -1)
        y_clf_list.append(cid if cid is not None and cid >= 0 else 0)

        w = m.get("true_w", 0.0)
        l = m.get("true_l", 0.0)
        d = m.get("true_d", 0.0)
        w = 0.0 if np.isnan(w) else float(w)
        l = 0.0 if np.isnan(l) else float(l)
        d = 0.0 if np.isnan(d) else float(d)
        y_wld_raw_list.append([w, l, d])

    y_clf = torch.tensor(y_clf_list, dtype=torch.long, device=device)
    y_wld_raw = torch.tensor(y_wld_raw_list, dtype=torch.float32, device=device)

    # Chuẩn hóa nhãn kích thước W, L, D nếu có y_scaler
    if y_scaler is not None:
        raw_np = y_wld_raw.cpu().numpy()
        if hasattr(y_scaler, "transform"):
            norm_np = y_scaler.transform(raw_np)
        elif isinstance(y_scaler, (list, tuple)) and len(y_scaler) == 3:
            cols = []
            for i, sc in enumerate(y_scaler):
                col = raw_np[:, i:i+1]
                cols.append(sc.transform(col) if hasattr(sc, "transform") else col)
            norm_np = np.hstack(cols)
        elif isinstance(y_scaler, dict) and all(k in y_scaler for k in ("w", "l", "d")):
            cols = []
            for i, k in enumerate(("w", "l", "d")):
                col = raw_np[:, i:i+1]
                sc = y_scaler[k]
                cols.append(sc.transform(col) if hasattr(sc, "transform") else col)
            norm_np = np.hstack(cols)
        else:
            norm_np = raw_np
        y_wld_norm = torch.tensor(norm_np, dtype=torch.float32, device=device)
    else:
        y_wld_norm = y_wld_raw.clone()

    return {
        "X_tensor": X_tensor,
        "y_clf": y_clf,
        "y_wld_norm": y_wld_norm,
        "y_wld_raw": y_wld_raw,
        "metadata": metadata,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "data_dir": data_dir_5k,
    }


def split_5khz_scan1_scan2(bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Phân chia 20 mẫu 5kHz thành 2 tập theo Giao thức 1 (Hướng 1):
    - Scan 1 (Tập Huấn luyện Thích ứng / Adaptation Set): 10 mẫu không kết thúc bằng '_1.csv'
    - Scan 2 (Tập Kiểm thử Mù / Blind Test Set): 10 mẫu kết thúc bằng '_1.csv'

    Trả về:
        (train_bundle, test_bundle)
    """
    metadata = bundle["metadata"]
    X = bundle.get("X_tensor", bundle.get("X_norm", bundle.get("X", None)))
    if X is None:
        raise KeyError("Không tìm thấy tensor đầu vào ('X_tensor' hoặc 'X_norm') trong bundle!")
    y_clf = bundle["y_clf"]
    y_wld_norm = bundle["y_wld_norm"]
    y_wld_raw = bundle["y_wld_raw"]

    train_idx = [i for i, m in enumerate(metadata) if not m["filename"].endswith("_1.csv")]
    test_idx = [i for i, m in enumerate(metadata) if m["filename"].endswith("_1.csv")]

    if len(train_idx) != 10 or len(test_idx) != 10:
        print(f"[CẢNH BÁO] Số mẫu phân tách Scan1/Scan2 không cân bằng: Scan1={len(train_idx)}, Scan2={len(test_idx)}")

    train_bundle = {
        "X": X[train_idx],
        "y_clf": y_clf[train_idx],
        "y_wld_norm": y_wld_norm[train_idx],
        "y_wld_raw": y_wld_raw[train_idx],
        "metadata": [metadata[i] for i in train_idx],
        "indices": train_idx,
    }

    test_bundle = {
        "X": X[test_idx],
        "y_clf": y_clf[test_idx],
        "y_wld_norm": y_wld_norm[test_idx],
        "y_wld_raw": y_wld_raw[test_idx],
        "metadata": [metadata[i] for i in test_idx],
        "indices": test_idx,
    }

    return train_bundle, test_bundle


def get_5khz_10fold_defect_splits(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tạo 10 folds cho Giao thức 2 (Hướng 2 - Leave-One-Defect-Out Cross Validation):
    - Mỗi fold bỏ ra 1 khuyết tật (crack_no từ 1 đến 10, gồm cả Scan 1 và Scan 2 = 2 file) làm Test.
    - Train trên 18 file của 9 khuyết tật còn lại.

    Trả về:
        Danh sách 10 dict, mỗi dict đại diện cho 1 fold:
        {
            'fold': 1..10,
            'held_out_crack': int,
            'train': {'X', 'y_clf', 'y_wld_norm', 'y_wld_raw', 'metadata', 'indices'},
            'test':  {'X', 'y_clf', 'y_wld_norm', 'y_wld_raw', 'metadata', 'indices'}
        }
    """
    metadata = bundle["metadata"]
    X = bundle.get("X_tensor", bundle.get("X_norm", bundle.get("X", None)))
    if X is None:
        raise KeyError("Không tìm thấy tensor đầu vào ('X_tensor' hoặc 'X_norm') trong bundle!")
    y_clf = bundle["y_clf"]
    y_wld_norm = bundle["y_wld_norm"]
    y_wld_raw = bundle["y_wld_raw"]

    crack_ids = sorted(list(set(m["crack_no"] for m in metadata if m["crack_no"] is not None)))
    if len(crack_ids) != 10:
        print(f"[CẢNH BÁO] Số vết nứt tìm thấy ({len(crack_ids)}) khác 10!")

    folds = []
    for fold_num, held_out in enumerate(crack_ids, start=1):
        test_idx = [i for i, m in enumerate(metadata) if m["crack_no"] == held_out]
        train_idx = [i for i, m in enumerate(metadata) if m["crack_no"] != held_out]

        folds.append({
            "fold": fold_num,
            "held_out_crack": held_out,
            "train": {
                "X": X[train_idx],
                "y_clf": y_clf[train_idx],
                "y_wld_norm": y_wld_norm[train_idx],
                "y_wld_raw": y_wld_raw[train_idx],
                "metadata": [metadata[i] for i in train_idx],
                "indices": train_idx,
            },
            "test": {
                "X": X[test_idx],
                "y_clf": y_clf[test_idx],
                "y_wld_norm": y_wld_norm[test_idx],
                "y_wld_raw": y_wld_raw[test_idx],
                "metadata": [metadata[i] for i in test_idx],
                "indices": test_idx,
            },
        })

    return folds


# =============================================================================
# BACKWARD COMPATIBILITY & HELPER FUNCTIONS
# =============================================================================
RealExperimentDataset = Real5kHzDataset


def load_real_data_for_model(
    model_type: str = "cnn",
    split: str = "5khz",
    x_scaler: Optional[Any] = None,
    y_scaler: Optional[Any] = None,
    scale_factor: float = 1.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Nạp dữ liệu thực nghiệm theo tần số (5khz, 10khz, 20khz) và định dạng tensor phù hợp với model_type:
    - 2D (B, 2, 32, 32) cho CNN.
    - 1D (B, 2048) cho MLP và Xiong et al.
    """
    is_1d = (model_type.lower() in ["mlp", "xiong"])
    exp_base = find_experiment_1_dir()
    if exp_base is None:
        raise FileNotFoundError("Không tìm thấy thư mục Experiment_1.")

    freq_clean = split.lower().replace("split", "").replace("_", "")
    target_dir = os.path.join(exp_base, freq_clean)
    if not os.path.isdir(target_dir):
        target_dir = os.path.join(exp_base, "5khz") if os.path.isdir(os.path.join(exp_base, "5khz")) else exp_base

    X_tensor, metadata = load_real_experiment_for_inference(
        target_path=target_dir,
        x_scaler=x_scaler,
        scale_factor=scale_factor,
        device=device,
        freq_filter=freq_clean if any(f in freq_clean for f in ["5k", "10k", "20k"]) else None,
    )

    if is_1d:
        X_tensor = X_tensor.reshape(X_tensor.size(0), -1)

    y_clf_list = []
    y_wld_raw_list = []
    for m in metadata:
        cid = m.get("true_shape_id", -1)
        y_clf_list.append(cid if cid is not None and cid >= 0 else 0)
        w = float(m.get("true_w", 0.0) or 0.0)
        l = float(m.get("true_l", 0.0) or 0.0)
        d = float(m.get("true_d", 0.0) or 0.0)
        y_wld_raw_list.append([w, l, d])

    y_clf = torch.tensor(y_clf_list, dtype=torch.long, device=device)
    y_wld = torch.tensor(y_wld_raw_list, dtype=torch.float32, device=device)
    return X_tensor, y_clf, y_wld, metadata


def get_real_dataloader(
    split: str = "5khz",
    model_type: str = "cnn",
    batch_size: int = 4,
    shuffle: bool = True,
    x_scaler: Optional[Any] = None,
    y_scaler: Optional[Any] = None,
    scale_factor: float = 1.0,
    device: str = "cpu",
) -> DataLoader:
    """
    Tạo PyTorch DataLoader cho dữ liệu thực nghiệm phục vụ huấn luyện và kiểm thử.
    """
    X_tensor, y_clf, y_wld_raw, metadata = load_real_data_for_model(
        model_type=model_type,
        split=split,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        scale_factor=scale_factor,
        device=device,
    )
    y_wld_norm = y_wld_raw.clone()
    ds = Real5kHzDataset(X_tensor, y_clf, y_wld_norm, y_wld_raw, metadata)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

