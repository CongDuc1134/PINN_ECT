# -*- coding: utf-8 -*-
"""
Single-Task 2D CNN Defect Sizing Regression WITHOUT PINN (NoPINN Baseline)
Architecture: 2D CNN Backbone + Joint Sizing Regression Head (W, L, D)
Purely data-driven regression without shape classification branch and without physics loss.
"""

import os
import sys
import copy
import traceback
import faulthandler
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tempfile
import errno
import io

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, max_error
)
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

try:
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.abspath(os.path.join(RESULT_DIR, ".."))
CHECK_DIR = os.path.abspath(os.path.join(PAPER_DIR, "check"))
REP_CODE_DIR = os.path.abspath(os.path.join(PAPER_DIR, "rep_code"))
MLP_DIR = os.path.abspath(os.path.join(RESULT_DIR, "mlp"))

for d in [SCRIPT_DIR, RESULT_DIR, PAPER_DIR, CHECK_DIR, REP_CODE_DIR, MLP_DIR]:
    if d not in sys.path and os.path.exists(d):
        sys.path.insert(0, d)

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH or not os.path.exists(DATA_PATH):
    for candidate in [
        os.path.join(CHECK_DIR, "Crack_Shape_Images"),
        os.path.join(PAPER_DIR, "check", "Crack_Shape_Images"),
        os.path.join(SCRIPT_DIR, "Crack_Shape_Images"),
        os.path.join(REP_CODE_DIR, "Crack_Shape_Images"),
        os.path.join(PAPER_DIR, "Crack_Shape_Images"),
    ]:
        if os.path.exists(candidate):
            DATA_PATH = candidate
            break

LABELS_PATH = os.environ.get("LABELS_PATH")
if not LABELS_PATH or not os.path.exists(LABELS_PATH):
    for candidate in [
        os.path.join(CHECK_DIR, "labels.csv"),
        os.path.join(PAPER_DIR, "check", "labels.csv"),
        os.path.join(SCRIPT_DIR, "labels.csv"),
        os.path.join(REP_CODE_DIR, "labels.csv"),
        os.path.join(PAPER_DIR, "labels.csv"),
    ]:
        if os.path.exists(candidate):
            LABELS_PATH = candidate
            break

OUTPUT_ROOT_DIR = os.environ.get(
    "OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "Outputs_cnn_single_task_regression_nopinn")
)
if not os.path.isabs(OUTPUT_ROOT_DIR):
    OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_ROOT_DIR)

try:
    from library_functions import Load_Data_With_Labels
except ImportError:
    print("[ERROR] Could not find 'library_functions.py'")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[OK] Device: {device} | PyTorch version: {torch.__version__}")

import argparse

parser = argparse.ArgumentParser(description="Single-Task 2D CNN Regression NoPINN")
parser.add_argument("--train_percent", type=int, default=int(os.environ.get("TRAIN_PERCENT", "5")), help="Training data percentage (1, 3, 5, 7, 10)")
parser.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_STATE", os.environ.get("SEED", "42"))), help="Random seed (42, 123, 456, 789)")
parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "300")), help="Total training epochs")
parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")), help="Mini-batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--run_tag", type=str, default=os.environ.get("RUN_TAG", ""), help="Optional run tag")
args, unknown = parser.parse_known_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
RANDOM_STATE = args.seed
TRAIN_PERCENT = args.train_percent
LEARNING_RATE = args.lr

RUN_TAG = args.run_tag if args.run_tag else f"train_{TRAIN_PERCENT:02d}pct_seed_{RANDOM_STATE}"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, f"train_{TRAIN_PERCENT:02d}pct", f"seed_{RANDOM_STATE}", RUN_TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# ============================================================================
# MODEL DEFINITION: SINGLE-TASK REGRESSION CNN (NO CLASSIFICATION HEAD)
# ============================================================================

class CNN_SingleTask_Regression(nn.Module):
    """
    2D CNN for defect sizing regression (W, L, D) with Kendall et al. Homoscedastic
    Uncertainty Weighting across W, L, D (without classification head).
    """
    def __init__(self):
        super(CNN_SingleTask_Regression, self).__init__()
        
        # Learnable Kendall Homoscedastic Uncertainty Parameters for W, L, D
        self.log_var_w = nn.Parameter(torch.tensor(0.0))
        self.log_var_l = nn.Parameter(torch.tensor(0.0))
        self.log_var_d = nn.Parameter(torch.tensor(0.0))
        
        # Shared 2D CNN Backbone
        self.backbone = nn.Sequential(
            # Block 1 (32x32 -> 16x16)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 2 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 3 (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Dropout(0.1)
        )
        
        # Regression Feature Extractor
        self.regressor_backbone = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Regression Output Head (W, L, D)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.reshape(feat.size(0), -1)
        reg_feat = self.regressor_backbone(feat)
        y_pred_wld = self.reg_head(reg_feat)
        return y_pred_wld


def compute_regression_losses(model, y_pred_wld, y_wld, criterion_reg):
    w_loss = criterion_reg(y_pred_wld[:, 0], y_wld[:, 0])
    l_loss = criterion_reg(y_pred_wld[:, 1], y_wld[:, 1])
    d_loss = criterion_reg(y_pred_wld[:, 2], y_wld[:, 2])
    
    # Kendall Uncertainty Weighting
    precision_w = torch.exp(-model.log_var_w)
    precision_l = torch.exp(-model.log_var_l)
    precision_d = torch.exp(-model.log_var_d)
    
    weighted_reg_loss = (
        precision_w * w_loss + 0.5 * model.log_var_w +
        precision_l * l_loss + 0.5 * model.log_var_l +
        precision_d * d_loss + 0.5 * model.log_var_d
    )
    raw_avg_loss = (w_loss + l_loss + d_loss) / 3.0
    return w_loss, l_loss, d_loss, weighted_reg_loss, raw_avg_loss


class MaxScaler:
    def __init__(self, max_val):
        self.max_val = float(max_val)
    def transform(self, X):
        return np.asarray(X) / self.max_val
    def inverse_transform(self, X_norm):
        return np.asarray(X_norm) * self.max_val

class SeparateMaxScaler:
    def __init__(self, w_scaler, l_scaler, d_scaler):
        self.w_scaler = w_scaler
        self.l_scaler = l_scaler
        self.d_scaler = d_scaler
    def transform(self, X):
        X = np.asarray(X).astype(np.float64)
        if X.ndim == 1:
            return np.array([self.w_scaler.transform(X[0]), self.l_scaler.transform(X[1]), self.d_scaler.transform(X[2])])
        X_norm = X.copy()
        X_norm[:, 0] = self.w_scaler.transform(X[:, 0])
        X_norm[:, 1] = self.l_scaler.transform(X[:, 1])
        X_norm[:, 2] = self.d_scaler.transform(X[:, 2])
        return X_norm
    def inverse_transform(self, X_norm):
        X_norm = np.asarray(X_norm).astype(np.float64)
        if X_norm.ndim == 1:
            return np.array([self.w_scaler.inverse_transform(X_norm[0]), self.l_scaler.inverse_transform(X_norm[1]), self.d_scaler.inverse_transform(X_norm[2])])
        X = X_norm.copy()
        X[:, 0] = self.w_scaler.inverse_transform(X_norm[:, 0])
        X[:, 1] = self.l_scaler.inverse_transform(X_norm[:, 1])
        X[:, 2] = self.d_scaler.inverse_transform(X_norm[:, 2])
        return X


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"SINGLE-TASK 2D CNN REGRESSION WITHOUT PINN (NOPINN BASELINE)")
    print(f"TRAIN PERCENT: {TRAIN_PERCENT}% | SEED: {RANDOM_STATE} | EPOCHS: {EPOCHS}")
    print(f"OUTPUT DIR: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

    # 1. Load Data
    X, y, matched_filenames = Load_Data_With_Labels(DATA_PATH, LABELS_PATH)
    total_samples = len(X)
    
    # Extract Shape Labels for Stratified Splitting
    labels_df = pd.read_csv(LABELS_PATH)
    filename_to_shape = {}
    for idx, row in labels_df.iterrows():
        try:
            fn = str(row['filename']).strip()
            sh = str(row['shape']).strip()
            if 'type' in labels_df.columns and sh == 'Step':
                tv = str(row['type']).strip()
                if tv and tv.lower() != 'nan':
                    sh = f"Step_{tv}"
            filename_to_shape[fn] = sh
        except Exception:
            continue

    shapes_list = [filename_to_shape.get(fn, fn.split('_')[0]) for fn in matched_filenames]
    unique_shapes = sorted(list(set(shapes_list)))
    shape_to_idx = {s: i for i, s in enumerate(unique_shapes)}
    y_shape = np.array([shape_to_idx[s] for s in shapes_list], dtype=np.int64)

    # 2. Train / Val / Test Split (70 / 20 / 10)
    indices = np.arange(total_samples)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.10, random_state=RANDOM_STATE, stratify=y_shape)
    y_shape_tv = y_shape[train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.222222, random_state=RANDOM_STATE, stratify=y_shape_tv)

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    y_shape_train_full = y_shape[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]
    y_shape_val = y_shape[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]
    y_shape_test = y_shape[test_idx]

    # 3. Stratified Sparse Training Subset Selection
    subset_parts = []
    for class_idx in sorted(np.unique(y_shape_train_full)):
        c_mask = np.where(y_shape_train_full == class_idx)[0]
        c_count = len(c_mask)
        c_take = max(1, min(c_count, int(np.floor(c_count * (TRAIN_PERCENT / 100.0)))))
        class_seed = RANDOM_STATE + int(class_idx) * 10007
        class_rng = np.random.default_rng(class_seed)
        chosen = class_rng.permutation(c_mask)[:c_take]
        subset_parts.append(np.sort(chosen))

    sub_indices = np.sort(np.concatenate(subset_parts))
    X_train = X_train_full[sub_indices]
    y_train = y_train_full[sub_indices]
    y_shape_train = y_shape_train_full[sub_indices]

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    print(f"Sample counts: Train={n_train} ({TRAIN_PERCENT}%) | Val={n_val} | Test={n_test}")

    # 4. Normalization (Fit on Train Subset ONLY)
    X_scaler = StandardScaler()
    n_tr, h, w, ch = X_train.shape
    X_train_flat = X_scaler.fit_transform(X_train.reshape(n_tr * h * w, ch))
    X_val_flat = X_scaler.transform(X_val.reshape(n_val * h * w, ch))
    X_test_flat = X_scaler.transform(X_test.reshape(n_test * h * w, ch))

    X_train_norm = X_train_flat.reshape(n_tr, h, w, ch)
    X_val_norm = X_val_flat.reshape(n_val, h, w, ch)
    X_test_norm = X_test_flat.reshape(n_test, h, w, ch)

    w_max = float(np.max(y_train[:, 0]))
    l_max = float(np.max(y_train[:, 1]))
    d_max = float(np.max(y_train[:, 2]))
    y_scaler = SeparateMaxScaler(MaxScaler(w_max), MaxScaler(l_max), MaxScaler(d_max))

    y_train_norm = y_scaler.transform(y_train)
    y_val_norm = y_scaler.transform(y_val)
    y_test_norm = y_scaler.transform(y_test)

    # 5. DataLoaders
    X_tr_t = torch.FloatTensor(X_train_norm).permute(0, 3, 1, 2)
    y_tr_t = torch.FloatTensor(y_train_norm)
    X_va_t = torch.FloatTensor(X_val_norm).permute(0, 3, 1, 2)
    y_va_t = torch.FloatTensor(y_val_norm)
    X_te_t = torch.FloatTensor(X_test_norm).permute(0, 3, 1, 2)
    y_te_t = torch.FloatTensor(y_test_norm)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=BATCH_SIZE, shuffle=False)

    # 6. Model, Optimizer, Scheduler
    model = CNN_SingleTask_Regression().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    criterion_reg = nn.MSELoss()
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    # 7. Training Loop
    best_val_loss = float('inf')
    best_model_weights = None
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with autocast('cuda'):
                    pred = model(bx)
                    _, _, _, loss, _ = compute_regression_losses(model, pred, by, criterion_reg)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(bx)
                _, _, _, loss, _ = compute_regression_losses(model, pred, by, criterion_reg)
                loss.backward()
                optimizer.step()
            train_loss_sum += loss.item() * len(bx)

        train_loss = train_loss_sum / n_train

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                _, _, _, loss, _ = compute_regression_losses(model, pred, by, criterion_reg)
                val_loss_sum += loss.item() * len(bx)

        val_loss = val_loss_sum / n_val
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, os.path.join(OUTPUT_DIR, "best_model_pytorch.pth"))

        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d}/{EPOCHS:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best Val: {best_val_loss:.6f}")

    # 8. Test Evaluation
    print("\nEvaluating on Independent Test Set (3,306 samples)...")
    model.load_state_dict(best_model_weights)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            pred = model(bx)
            test_preds.append(pred.cpu().numpy())

    y_pred_norm = np.concatenate(test_preds, axis=0)
    y_pred = y_scaler.inverse_transform(y_pred_norm)

    mae_w = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_l = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mae_d = mean_absolute_error(y_test[:, 2], y_pred[:, 2])
    overall_mae = (mae_w + mae_l + mae_d) / 3.0

    rmse_w = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_l = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    rmse_d = np.sqrt(mean_squared_error(y_test[:, 2], y_pred[:, 2]))
    overall_rmse = (rmse_w + rmse_l + rmse_d) / 3.0

    r2_w = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_l = r2_score(y_test[:, 1], y_pred[:, 1])
    r2_d = r2_score(y_test[:, 2], y_pred[:, 2])
    overall_r2 = (r2_w + r2_l + r2_d) / 3.0

    nmae = (mae_w / 1.2 + mae_l / 18.0 + mae_d / 2.9) / 3.0 * 100.0

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (SINGLE-TASK REGRESSION NOPINN):")
    print(f"  MAE Width:  {mae_w:.4f} mm | R2: {r2_w:.4f} | RMSE: {rmse_w:.4f} mm")
    print(f"  MAE Length: {mae_l:.4f} mm | R2: {r2_l:.4f} | RMSE: {rmse_l:.4f} mm")
    print(f"  MAE Depth:  {mae_d:.4f} mm | R2: {r2_d:.4f} | RMSE: {rmse_d:.4f} mm")
    print(f"  Overall MAE: {overall_mae:.4f} mm | Overall R2: {overall_r2:.4f} | NMAE: {nmae:.2f}%")
    print(f"{'='*60}\n")

    res_df = pd.DataFrame([{
        'Model': '2D CNN Single-Task Regression (NoPINN)',
        'Train_Percent': f"{TRAIN_PERCENT}%",
        'Seed': RANDOM_STATE,
        'MAE_W': mae_w, 'MAE_L': mae_l, 'MAE_D': mae_d, 'Overall_MAE': overall_mae,
        'RMSE_W': rmse_w, 'RMSE_L': rmse_l, 'RMSE_D': rmse_d, 'Overall_RMSE': overall_rmse,
        'R2_W': r2_w, 'R2_L': r2_l, 'R2_D': r2_d, 'Overall_R2': overall_r2,
        'NMAE_Percent': nmae
    }])
    res_df.to_csv(os.path.join(OUTPUT_DIR, "test_summary_metrics.csv"), index=False)

    pred_df = pd.DataFrame({
        'True_W': y_test[:, 0], 'Pred_W': y_pred[:, 0],
        'True_L': y_test[:, 1], 'Pred_L': y_pred[:, 1],
        'True_D': y_test[:, 2], 'Pred_D': y_pred[:, 2],
        'Shape_Class': [unique_shapes[s] for s in y_shape_test]
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"[OK] Saved results to: {OUTPUT_DIR}")

    # Real experiment evaluation (Experiment_1)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from load_real_experiment_data import evaluate_real_experiment
        evaluate_real_experiment(model, X_scaler, y_scaler, unique_shapes, OUTPUT_DIR, device=device)
    except Exception as e:
        print(f"[WARN] Real experiment evaluation encountered an error: {e}")

if __name__ == "__main__":
    main()
