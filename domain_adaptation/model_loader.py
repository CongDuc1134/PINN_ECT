# -*- coding: utf-8 -*-
"""
===============================================================================
MODEL & SCALER LOADER MODULE FOR DOMAIN ADAPTATION (SIM-TO-REAL ECT)
===============================================================================
Module chuyên dụng:
1. Tự động tìm kiếm và nạp các checkpoint mô hình tiền huấn luyện (best_model_pytorch.pth)
   cho cả 4 phương pháp đối chứng:
   - 'cnn_nopinn': CNN Baseline 2D không PINN
   - 'cnn_proposed': CNN Proposed 2D với ràng buộc PINN & Kendall uncertainty
   - 'mlp': Multitask MLP 1D Baseline
   - 'xiong': Xiong et al. 2023 Single-task 3D Regression Baseline
2. Nạp bộ chuẩn hóa tương ứng: X_scaler.pkl và y_scaler.pkl (SeparateMaxScaler/MaxScaler).
3. Đóng gói đầy đủ tensor 5kHz và nhãn phục vụ các quy trình finetuning.
===============================================================================
"""

import os
import sys
import re
import pickle
import warnings
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Đảm bảo các lớp scaler được nhận diện trong __main__ cho pickle
try:
    from domain_adaptation.load_real_experiment_data import (
        denormalize_regression_predictions,
        DEFAULT_UNIQUE_SHAPES,
        SeparateMaxScaler,
        MaxScaler,
    )
except ImportError:
    from load_real_experiment_data import (
        denormalize_regression_predictions,
        DEFAULT_UNIQUE_SHAPES,
        SeparateMaxScaler,
        MaxScaler,
    )

from domain_adaptation.cnn.models import ImprovedMultimodelNet as CNN_Multitask
from domain_adaptation.mlp.models import MultitaskMLP_PINN as MLP_Multitask
from domain_adaptation.xiong.models import RegressionMLP_PINN as Xiong_Regression
from domain_adaptation.data_loader import load_all_5khz_samples


def find_trained_checkpoint_dir(
    model_type: str = "cnn",
    train_pct: str = "10pct",
    variant: str = "pinn",
) -> Optional[str]:
    """
    Tự động tìm kiếm thư mục checkpoint đã huấn luyện tốt nhất trong repo.

    Tham số:
        model_type: 'cnn', 'mlp', hoặc 'xiong'.
        train_pct: '10pct', '07pct', '05pct', '03pct', '01pct'.
        variant: 'pinn' hoặc 'nopinn' / 'baseline'.
    """
    m_type = model_type.lower()
    var = variant.lower()
    search_dirs = []

    if m_type == "cnn":
        if var == "pinn":
            cand = os.path.join(PROJECT_ROOT, "cnn", "Outputs_cnn_pinn", "loss_log1p_norm_sse", f"train_{train_pct}")
        else:
            cand = os.path.join(PROJECT_ROOT, "cnn", "Outputs_cnn_baseline", f"train_{train_pct}")
        search_dirs.append(cand)
    elif m_type == "mlp":
        cand = os.path.join(PROJECT_ROOT, "mlp", "Outputs_mlp_pinn", "loss_mse", f"train_{train_pct}")
        search_dirs.append(cand)
    elif m_type == "xiong":
        cand = os.path.join(PROJECT_ROOT, "mlp", "Outputs_xiong_pinn", "loss_mse", f"train_{train_pct}")
        search_dirs.append(cand)

    for s_dir in search_dirs:
        if os.path.isdir(s_dir):
            if os.path.exists(os.path.join(s_dir, "best_model_pytorch.pth")):
                return s_dir
            # Quét các thư mục con run_*
            subdirs = sorted(
                [os.path.join(s_dir, d) for d in os.listdir(s_dir) if os.path.isdir(os.path.join(s_dir, d))],
                reverse=True,
            )
            for sub in subdirs:
                if os.path.exists(os.path.join(sub, "best_model_pytorch.pth")):
                    return sub

    return None


def load_scalers(checkpoint_dir: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Nạp X_scaler.pkl và y_scaler.pkl từ thư mục checkpoint.
    """
    x_scaler = None
    y_scaler = None

    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None, None

    x_scaler_path = os.path.join(checkpoint_dir, "X_scaler.pkl")
    y_scaler_path = os.path.join(checkpoint_dir, "y_scaler.pkl")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.path.exists(x_scaler_path):
            try:
                with open(x_scaler_path, "rb") as f:
                    x_scaler = pickle.load(f)
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp X_scaler: {e}")

        if os.path.exists(y_scaler_path):
            try:
                with open(y_scaler_path, "rb") as f:
                    y_scaler = pickle.load(f)
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp y_scaler: {e}")

    return x_scaler, y_scaler


def extract_pinn_alpha_from_checkpoint(checkpoint_dir: str, variant: str = "pinn") -> float:
    """
    Trích xuất chính xác hệ số alpha của PINN từ checkpoint đã huấn luyện:
    1. Nếu variant == 'nopinn' hoặc 'baseline' -> alpha = 0.0.
    2. Đọc từ reeval_model_config.txt (dòng 'pinn_alpha: <val>').
    3. Đọc từ summary_results.txt (dòng 'Alpha init: alpha=<val>' hoặc 'alpha=<val>').
    4. Phân tích từ tên thư mục chứa checkpoint qua regex '_a([0-9\.]+)_' (ví dụ '_a1_' -> 1.0, '_a10_' -> 10.0).
    5. Mặc định dự phòng an toàn: 1.0 (nếu là PINN) hoặc 0.0 (nếu là NoPINN).
    """
    if not variant or variant.lower() in ["nopinn", "baseline"]:
        return 0.0

    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return 1.0

    # 1. Đọc từ reeval_model_config.txt
    cfg_txt = os.path.join(checkpoint_dir, "reeval_model_config.txt")
    if os.path.exists(cfg_txt):
        try:
            with open(cfg_txt, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                m = re.search(r"pinn_alpha:\s*([0-9\.]+)", content, re.IGNORECASE)
                if m:
                    return float(m.group(1))
        except Exception:
            pass

    # 2. Đọc từ summary_results.txt
    sum_txt = os.path.join(checkpoint_dir, "summary_results.txt")
    if os.path.exists(sum_txt):
        try:
            with open(sum_txt, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                m = re.search(r"alpha(?:\s*init)?:\s*(?:alpha\s*=\s*)?([0-9\.]+)", content, re.IGNORECASE)
                if not m:
                    m = re.search(r"alpha\s*=\s*([0-9\.]+)", content, re.IGNORECASE)
                if m:
                    return float(m.group(1))
        except Exception:
            pass

    # 3. Phân tích tên thư mục
    dir_name = os.path.basename(checkpoint_dir)
    m_dir = re.search(r"_a([0-9\.]+)_", dir_name)
    if m_dir:
        try:
            return float(m_dir.group(1))
        except Exception:
            pass

    return 1.0


def load_pretrained_model(
    model_type: str = "cnn",
    checkpoint_dir: Optional[str] = None,
    train_pct: str = "10pct",
    variant: str = "pinn",
    device: str = "cpu",
) -> Tuple[nn.Module, Optional[Any], Optional[Any], str]:
    """
    Khởi tạo kiến trúc và nạp trọng số đã tiền huấn luyện kèm scalers.
    """
    m_type = model_type.lower()
    if checkpoint_dir is None:
        checkpoint_dir = find_trained_checkpoint_dir(model_type=m_type, train_pct=train_pct, variant=variant)

    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục checkpoint cho model_type='{model_type}', train_pct='{train_pct}', variant='{variant}'"
        )

    model_weight_path = os.path.join(checkpoint_dir, "best_model_pytorch.pth")
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Không tìm thấy best_model_pytorch.pth trong: {checkpoint_dir}")

    # Khởi tạo mô hình
    if m_type == "cnn":
        model = CNN_Multitask(num_shapes=5).to(device)
    elif m_type == "mlp":
        model = MLP_Multitask(num_shapes=5).to(device)
    elif m_type == "xiong":
        model = Xiong_Regression().to(device)
    else:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    # Nạp weights
    ckpt = torch.load(model_weight_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    elif isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
    model.eval()

    # Nạp scalers
    x_scaler, y_scaler = load_scalers(checkpoint_dir)

    print(f"[OK] Đã nạp thành công mô hình '{m_type.upper()}' ({variant}) từ: {os.path.basename(checkpoint_dir)}")
    return model, x_scaler, y_scaler, checkpoint_dir


def load_5khz_fully_prepared(
    model_type: str = "cnn",
    checkpoint_dir: Optional[str] = None,
    train_pct: str = "10pct",
    variant: str = "pinn",
    scale_factor: float = 1.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Nạp toàn diện:
    1. Mô hình với weights tiền huấn luyện
    2. Cả 2 bộ scaler (X_scaler, y_scaler)
    3. Toàn bộ 20 mẫu thực nghiệm 5kHz đã chuẩn hóa đầu vào X và nhãn kích thước W, L, D.

    Trả về:
        Dict gồm:
        'model', 'X_norm', 'y_clf', 'y_wld_norm', 'y_wld_raw', 'metadata',
        'x_scaler', 'y_scaler', 'checkpoint_dir', 'is_1d', 'model_type', 'variant'
    """
    is_1d = (model_type.lower() in ["mlp", "xiong"])

    # 1. Nạp mô hình và scalers
    model, x_scaler, y_scaler, ckpt_dir = load_pretrained_model(
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        train_pct=train_pct,
        variant=variant,
        device=device,
    )

    # 2. Nạp dữ liệu 5kHz đã chuẩn hóa theo scalers này
    data_bundle = load_all_5khz_samples(
        is_1d=is_1d,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        scale_factor=scale_factor,
        device=device,
    )

    alpha = extract_pinn_alpha_from_checkpoint(ckpt_dir, variant=variant)

    return {
        "model": model,
        "X_tensor": data_bundle["X_tensor"],
        "X_norm": data_bundle["X_tensor"],
        "y_clf": data_bundle["y_clf"],
        "y_wld_norm": data_bundle["y_wld_norm"],
        "y_wld_raw": data_bundle["y_wld_raw"],
        "metadata": data_bundle["metadata"],
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "checkpoint_dir": ckpt_dir,
        "is_1d": is_1d,
        "model_type": model_type.lower(),
        "variant": variant.lower(),
        "alpha": alpha,
    }


def predict_and_denormalize(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_scaler: Optional[Any] = None,
    unique_shapes: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Thực hiện suy luận từ tensor đầu vào X và giải chuẩn hóa:
    - Nhãn hình học dạng chuỗi ký tự (pred_shapes)
    - Kích thước [W, L, D] thực tế (mm) (pred_wld_denorm)
    - shape_logits gốc
    - pred_wld_norm gốc
    """
    if unique_shapes is None:
        unique_shapes = DEFAULT_UNIQUE_SHAPES

    model.eval()
    with torch.no_grad():
        out = model(X_tensor)
        if isinstance(out, (tuple, list)):
            shape_logits, pred_wld_norm = out[0], out[1]
        else:
            if hasattr(out, "shape") and out.shape[1] == len(unique_shapes):
                shape_logits = out
                pred_wld_norm = torch.zeros((X_tensor.size(0), 3), device=X_tensor.device)
            else:
                shape_logits = None
                pred_wld_norm = out

        # Phân loại hình dạng
        if shape_logits is not None:
            pred_ids = torch.argmax(shape_logits, dim=1).cpu().numpy()
            pred_shapes = [
                unique_shapes[sid] if 0 <= sid < len(unique_shapes) else "Unknown"
                for sid in pred_ids
            ]
        else:
            pred_shapes = ["N/A"] * X_tensor.size(0)

        # Giải chuẩn hóa kích thước [W, L, D]
        pred_wld_denorm = denormalize_regression_predictions(pred_wld_norm, y_scaler)

    return pred_shapes, pred_wld_denorm, shape_logits, pred_wld_norm
