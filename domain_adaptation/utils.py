"""
================================================================================
domain_adaptation/utils.py
Utilities to locate checkpoints, scalers, real data, and manage 10-Fold LODO splits.
================================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from load_real_experiment_data import (
    find_experiment_1_dir,
    load_real_experiment_for_inference,
    denormalize_regression_predictions,
)
from domain_adaptation.models import ImprovedMultimodelNet

UNIQUE_SHAPES = ['Ellipse', 'Rectangular', 'Step_R', 'Step_T', 'Triangular']


def list_all_available_checkpoints():
    """Dynamically finds all directories containing best_model_pytorch.pth."""
    import glob
    pinn_pat = os.path.join(PROJECT_ROOT, "cnn", "Outputs_cnn_pinn", "**", "best_model_pytorch.pth")
    base_pat = os.path.join(PROJECT_ROOT, "cnn", "Outputs_cnn_baseline", "**", "best_model_pytorch.pth")
    all_pth = glob.glob(pinn_pat, recursive=True) + glob.glob(base_pat, recursive=True)
    dirs = [os.path.dirname(p) for p in all_pth]
    return sorted(list(set(dirs)))


def find_default_checkpoint_dir(keyword=None):
    """
    Automatically finds the checkpoint directory.
    - If keyword is given (e.g. '10pct', 'a100', 'seed_123'), matches the best candidate.
    - If keyword is None, auto-selects the 5% PINN base model.
    """
    all_dirs = list_all_available_checkpoints()
    if not all_dirs:
        return None

    if keyword:
        # Check direct path first
        if os.path.isdir(keyword) and os.path.exists(os.path.join(keyword, "best_model_pytorch.pth")):
            return keyword
        # Fuzzy match
        for d in all_dirs:
            if keyword.lower() in d.lower():
                return d

    # Default preference: PINN base a1 seed 42 (5% or 10%)
    for pref in ["train_05pct/PINN_base_a1", "PINN_base_a1", "Outputs_cnn_pinn"]:
        for d in all_dirs:
            if pref.replace("/", os.sep) in d or pref in d:
                return d

    return all_dirs[0]


def get_model_tag(model_dir):
    """
    Generates a clean directory tag for organizing results by model.
    E.g.: 'train_05pct_PINN_base_a1_W100_E300_seed_42_run_20260819_103124'
    """
    if not model_dir:
        return "default_model"
    norm = os.path.normpath(model_dir)
    parts = norm.split(os.sep)
    pct = "model"
    for p in parts:
        if "train_" in p:
            pct = p
            break
    run_name = os.path.basename(norm)
    return f"{pct}_{run_name}"


def load_pretrained_checkpoint(model_dir=None, device="cpu"):
    """
    Loads pretrained model weights and normalization scalers.
    Returns: (model, x_scaler, y_scaler, model_dir)
    """
    if not model_dir:
        model_dir = find_default_checkpoint_dir()
    if not model_dir or not os.path.exists(model_dir):
        raise FileNotFoundError(f"Could not locate checkpoint directory: {model_dir}")

    model_path = os.path.join(model_dir, "best_model_pytorch.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load scalers
    x_scaler = None
    y_scaler = None
    x_scaler_path = os.path.join(model_dir, "X_scaler.pkl")
    y_scaler_path = os.path.join(model_dir, "y_scaler.pkl")
    if os.path.exists(x_scaler_path):
        with open(x_scaler_path, "rb") as f:
            x_scaler = pickle.load(f)
    if os.path.exists(y_scaler_path):
        with open(y_scaler_path, "rb") as f:
            y_scaler = pickle.load(f)

    # Initialize model and load weights
    model = ImprovedMultimodelNet(num_shapes=len(UNIQUE_SHAPES)).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    return model, x_scaler, y_scaler, model_dir


def load_5khz_real_data(x_scaler=None, device="cpu"):
    """
    Loads all 20 real experiment measurement samples for 5kHz from Experiment_1/Trainning.
    Returns: (X_tensor, metadata_list)
    """
    exp1_dir = find_experiment_1_dir()
    if not exp1_dir:
        raise FileNotFoundError("Could not find Experiment_1 directory!")
    train_5khz_dir = os.path.join(exp1_dir, "Trainning")

    X_tensor, metadata_list = load_real_experiment_for_inference(
        train_5khz_dir,
        x_scaler=x_scaler,
        device=device,
        freq_filter="5khz"
    )
    return X_tensor, metadata_list


def build_10fold_lodo_splits(metadata_list):
    """
    Builds 10 Leave-One-Defect-Out (LODO) folds from the 20 real samples.
    Each fold leaves out 2 measurement files of 1 crack for test, using 18 for train/adaptation.
    """
    crack_to_indices = {}
    for idx, meta in enumerate(metadata_list):
        c_no = meta['crack_no']
        if c_no not in crack_to_indices:
            crack_to_indices[c_no] = []
        crack_to_indices[c_no].append(idx)

    all_indices = set(range(len(metadata_list)))
    folds = {}
    for fold_id, (c_no, test_idx_list) in enumerate(sorted(crack_to_indices.items()), 1):
        test_set = set(test_idx_list)
        train_set = all_indices - test_set
        folds[fold_id] = {
            'crack_no': c_no,
            'test_indices': sorted(list(test_set)),
            'train_indices': sorted(list(train_set)),
            'test_files': [metadata_list[i]['filename'] for i in test_idx_list],
            'true_shape': metadata_list[test_idx_list[0]]['true_shape'],
        }
    return folds


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    r2_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


def compute_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=95, seed=42):
    """Computes non-parametric bootstrap confidence interval (ci%) on pooled predictions."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        try:
            val = metric_fn(y_true[indices], y_pred[indices])
            scores.append(val)
        except Exception:
            continue
    if len(scores) < 10:
        return np.nan, np.nan
    alpha = (100 - ci) / 2.0
    low = np.percentile(scores, alpha)
    high = np.percentile(scores, 100 - alpha)
    return low, high


def compute_pooled_oof_summary(df_predictions, output_dir=None):
    """
    Computes rigorous global pooled Out-of-Fold (OOF) metrics across all 20 samples:
    - Classification: Accuracy (with 95% Bootstrap CI), Balanced Accuracy, Macro F1 (with 95% CI), MCC, and Confusion Matrix
    - Regression: MAE, RMSE, R^2, and Normalized MAE (NMAE) for W, L, D
    """
    summary_rows = []
    # Dynamic ranges from Table 2 ground truth:
    # W in [0.5, 0.9] -> range = 0.4 mm
    # D in [1.0, 3.0] -> range = 2.0 mm
    # L = 10.0 mm (constant nominal length -> relative % error)
    w_range = 0.4
    d_range = 2.0
    l_nominal = 10.0

    for method, df_m in df_predictions.groupby('method', sort=False):
        y_true = df_m['true_shape'].values
        y_pred = df_m['pred_shape'].values

        acc = accuracy_score(y_true, y_pred) * 100.0
        acc_low, acc_high = compute_bootstrap_ci(
            y_true, y_pred, lambda yt, yp: accuracy_score(yt, yp) * 100.0
        )

        bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
        f1_low, f1_high = compute_bootstrap_ci(
            y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0) * 100.0
        )

        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = 0.0

        # Regression metrics
        mae_w = df_m['err_w'].mean()
        mae_l = df_m['err_l'].mean()
        mae_d = df_m['err_d'].mean()
        overall_mae = (mae_w + mae_l + mae_d) / 3.0

        rmse_w = np.sqrt(mean_squared_error(df_m['true_w'], df_m['pred_w']))
        rmse_l = np.sqrt(mean_squared_error(df_m['true_l'], df_m['pred_l']))
        rmse_d = np.sqrt(mean_squared_error(df_m['true_d'], df_m['pred_d']))
        overall_rmse = (rmse_w + rmse_l + rmse_d) / 3.0

        # Normalized MAE (%)
        nmae_w = (mae_w / w_range) * 100.0
        nmae_l = (mae_l / l_nominal) * 100.0
        nmae_d = (mae_d / d_range) * 100.0
        overall_nmae = (nmae_w + nmae_l + nmae_d) / 3.0

        # R^2 Score
        try:
            r2_w = r2_score(df_m['true_w'], df_m['pred_w'])
        except Exception:
            r2_w = np.nan
        try:
            r2_d = r2_score(df_m['true_d'], df_m['pred_d'])
        except Exception:
            r2_d = np.nan

        # Save Confusion Matrix if output_dir provided
        cm = confusion_matrix(y_true, y_pred, labels=UNIQUE_SHAPES)
        if output_dir:
            cm_df = pd.DataFrame(cm, index=UNIQUE_SHAPES, columns=UNIQUE_SHAPES)
            cm_path = os.path.join(output_dir, f"confusion_matrix_{method}.csv")
            cm_df.to_csv(cm_path)

        summary_rows.append({
            'Method': method,
            'Num_Samples': len(df_m),
            'Clf_Accuracy (%)': round(acc, 2),
            'Acc_95CI': f"[{acc_low:.1f}, {acc_high:.1f}]",
            'Balanced_Acc (%)': round(bal_acc, 2),
            'F1_Macro (%)': round(f1_macro, 2),
            'F1_95CI': f"[{f1_low:.1f}, {f1_high:.1f}]",
            'MCC': round(mcc, 3),
            'MAE_W (mm)': round(mae_w, 4),
            'MAE_L (mm)': round(mae_l, 4),
            'MAE_D (mm)': round(mae_d, 4),
            'NMAE_W (%)': round(nmae_w, 1),
            'NMAE_L (%)': round(nmae_l, 1),
            'NMAE_D (%)': round(nmae_d, 1),
            'Overall_NMAE (%)': round(overall_nmae, 1),
            'Overall_MAE (mm)': round(overall_mae, 4),
            'Overall_RMSE (mm)': round(overall_rmse, 4),
            'R2_W': round(r2_w, 3) if not np.isnan(r2_w) else 0.0,
            'R2_D': round(r2_d, 3) if not np.isnan(r2_d) else 0.0,
        })

    return pd.DataFrame(summary_rows)
