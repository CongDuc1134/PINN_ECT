# -*- coding: utf-8 -*-
"""
Single-Task 2D CNN Defect Shape Classification WITH PINN (Physics-Informed)
Architecture: 2D CNN Backbone + 5-Class Defect Shape Classification Head + Auxiliary DMM Physics Loss
Regularizes feature representations with analytical DMM dipole forward solver across candidate defect geometries.
"""

import os
import sys
import copy
import traceback
import faulthandler
import random
import math
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
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
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
    "OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "Outputs_cnn_single_task_classification_pinn")
)
if not os.path.isabs(OUTPUT_ROOT_DIR):
    OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_ROOT_DIR)

try:
    from library_functions import Load_Data_With_Labels
    import regtangular
    import ellipse
    import triangular
    import step_r
    import step_t
except ImportError as e:
    print(f"[ERROR] Could not import required module: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[OK] Device: {device} | PyTorch version: {torch.__version__}")

import argparse

parser = argparse.ArgumentParser(description="Single-Task 2D CNN Classification with PINN")
parser.add_argument("--train_percent", type=int, default=int(os.environ.get("TRAIN_PERCENT", "5")), help="Training data percentage (1, 3, 5, 7, 10)")
parser.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_STATE", os.environ.get("SEED", "42"))), help="Random seed (42, 123, 456, 789)")
parser.add_argument("--alpha", type=float, default=float(os.environ.get("ALPHA_INIT", "1.0")), help="Physics loss weight alpha")
parser.add_argument("--warmup", type=int, default=int(os.environ.get("PINN_ACTIVATION_EPOCH", "100")), help="Warmup epochs before PINN activation (0, 50, 100, 150)")
parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "300")), help="Total training epochs")
parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")), help="Mini-batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--run_tag", type=str, default=os.environ.get("RUN_TAG", ""), help="Optional run tag")
args, unknown = parser.parse_known_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
RANDOM_STATE = args.seed
TRAIN_PERCENT = args.train_percent
ALPHA_INIT = args.alpha
PINN_ACTIVATION_EPOCH = args.warmup
LEARNING_RATE = args.lr

RUN_TAG = args.run_tag if args.run_tag else f"alpha_{ALPHA_INIT}_warmup_{PINN_ACTIVATION_EPOCH}_train_{TRAIN_PERCENT:02d}pct_seed_{RANDOM_STATE}"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, f"train_{TRAIN_PERCENT:02d}pct", f"seed_{RANDOM_STATE}", f"warmup_{PINN_ACTIVATION_EPOCH}", RUN_TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# ============================================================================
# MODEL DEFINITION: SINGLE-TASK CLASSIFICATION CNN WITH AUXILIARY PINN
# ============================================================================

class CNN_SingleTask_Classification_PINN(nn.Module):
    """
    2D CNN for 5-class defect shape classification regularized with analytical DMM physics loss.
    Takes W, L, D from ground-truth labels and applies physics regularization on predicted shape probabilities.
    """
    def __init__(self, num_shapes=5):
        super(CNN_SingleTask_Classification_PINN, self).__init__()
        self.num_shapes = num_shapes
        
        # 2D CNN Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Dropout(0.1)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_shapes)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.reshape(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits


def calculate_pinn_sample_loss_torch(w_pred, l_pred, d_pred, shape_name, H_true_tensor,
                                     z_lift=1.0, N=16, Res=0.78):
    wc = torch.clamp(w_pred, min=0.3, max=1.5)
    lc = torch.clamp(l_pred, min=2.0, max=20.0)
    dc = torch.clamp(d_pred, min=0.1, max=3.0)

    freq, sicma, mu = 5000.0, 35461000.0, 0.0000012566
    csi, K, I, G = 0.085, 1.5, 0.01, 3981.0
    delta = 1.0 / math.sqrt(math.pi * freq * mu * sicma) * 1000.0

    dev = H_true_tensor.device if torch.is_tensor(H_true_tensor) else device

    if shape_name == 'Ellipse':
        H_tensor = ellipse.compute_map_gpu(wc, lc, dc, delta, z_lift)
        H_raw = (csi / (4.0 * math.pi)) * H_tensor * K * I * G
        H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
        H_pred = torch.diff(H_pred, dim=1)
        H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

    elif shape_name == 'Rectangular':
        H_tensor, _, _ = regtangular.compute_magnetic_field(
            wc, lc, dc, delta, z_lift, N, Res, angle=0, K=K, I=I, G=G, csi=csi
        )
        H_pred = torch.rot90(H_tensor, 2, dims=(0, 1))
        H_pred = torch.diff(H_pred, dim=1)
        H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

    elif shape_name == 'Triangular':
        H_tensor = triangular.compute_triangular_map_gpu(wc, lc, dc, delta, z_lift)
        H_raw = (csi / (4.0 * math.pi)) * H_tensor * K * I * G
        H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
        H_pred = torch.diff(H_pred, dim=1)
        H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

    elif shape_name == 'Step_R':
        xs = torch.linspace(-N * Res, N * Res, 2 * N, device=dev)
        ys = torch.linspace(-N * Res, N * Res, 2 * N, device=dev)
        X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
        H_val = step_r.calculate_field_Step_R(X_grid, Y_grid, z_lift, wc, lc, dc, delta)
        H_raw = H_val * (csi / (4.0 * math.pi)) * K * I * G
        H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
        H_pred = torch.diff(H_pred, dim=1)
        H_pred = torch.flip(H_pred, dims=(1,))
        H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

    elif shape_name == 'Step_T':
        xs = torch.linspace(-N * Res, N * Res, 2 * N, device=dev)
        ys = torch.linspace(-N * Res, N * Res, 2 * N, device=dev)
        X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
        H_val = step_t.calculate_field_Step_T(X_grid, Y_grid, z_lift, wc, lc, dc, delta)
        H_raw = H_val * (csi / (4.0 * math.pi)) * K * I * G
        H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
        H_pred = torch.diff(H_pred, dim=1)
        H_pred = torch.flip(H_pred, dims=(1,))
        H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
    else:
        return torch.tensor(0.0, device=dev)

    if not torch.is_tensor(H_true_tensor):
        H_true_t = torch.tensor(H_true_tensor, device=dev, dtype=torch.float32)
    else:
        H_true_t = H_true_tensor.to(device=dev, dtype=torch.float32)

    sse = torch.sum((H_true_t - H_pred) ** 2)
    energy = torch.sum(H_true_t ** 2) + 1e-12
    return torch.log(1.0 + (sse / energy))


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"SINGLE-TASK 2D CNN CLASSIFICATION WITH PINN")
    print(f"TRAIN PERCENT: {TRAIN_PERCENT}% | SEED: {RANDOM_STATE} | ALPHA: {ALPHA_INIT} | WARMUP: {PINN_ACTIVATION_EPOCH}")
    print(f"OUTPUT DIR: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

    # 1. Load Data
    X, y, matched_filenames = Load_Data_With_Labels(DATA_PATH, LABELS_PATH)
    total_samples = len(X)
    
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
    num_shapes = len(unique_shapes)
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
    y_shape_val = y_shape[val_idx]

    X_test = X[test_idx]
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

    X_train_field = torch.FloatTensor(X_train[:, :, :, 0]).to(device)

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

    # 5. DataLoaders
    X_tr_t = torch.FloatTensor(X_train_norm).permute(0, 3, 1, 2)
    y_sh_tr_t = torch.LongTensor(y_shape_train)
    y_wld_tr_t = torch.FloatTensor(y_train)  # Ground truth W, L, D in mm
    idx_tr_t = torch.arange(n_train)

    X_va_t = torch.FloatTensor(X_val_norm).permute(0, 3, 1, 2)
    y_sh_va_t = torch.LongTensor(y_shape_val)
    X_te_t = torch.FloatTensor(X_test_norm).permute(0, 3, 1, 2)
    y_sh_te_t = torch.LongTensor(y_shape_test)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_sh_tr_t, y_wld_tr_t, idx_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va_t, y_sh_va_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_sh_te_t), batch_size=BATCH_SIZE, shuffle=False)

    # 6. Model, Optimizer, Scheduler
    model = CNN_SingleTask_Classification_PINN(num_shapes=num_shapes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    criterion_clf = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_weights = None
    history = {'train_loss': [], 'val_loss': [], 'pinn_loss': []}

    print("\nStarting Training with PINN Warm-up Schedule...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        pinn_loss_sum = 0.0
        pinn_active = (epoch > PINN_ACTIVATION_EPOCH)
        current_alpha = ALPHA_INIT if pinn_active else 0.0

        for bx, by_clf, b_wld, b_idx in train_loader:
            bx, by_clf, b_wld = bx.to(device), by_clf.to(device), b_wld.to(device)
            optimizer.zero_grad()
            
            logits = model(bx)
            clf_loss = criterion_clf(logits, by_clf)

            loss = clf_loss
            if pinn_active:
                probs = torch.softmax(logits, dim=1)
                batch_pinn_loss = 0.0
                for i in range(len(bx)):
                    s_idx = b_idx[i].item()
                    w_true = b_wld[i, 0]
                    l_true = b_wld[i, 1]
                    d_true = b_wld[i, 2]
                    H_true_s = X_train_field[s_idx]
                    
                    sample_pinn = 0.0
                    for c_idx, s_name in enumerate(unique_shapes):
                        p_c_loss = calculate_pinn_sample_loss_torch(w_true, l_true, d_true, s_name, H_true_s)
                        sample_pinn = sample_pinn + probs[i, c_idx] * p_c_loss
                    
                    batch_pinn_loss = batch_pinn_loss + sample_pinn

                batch_pinn_loss = batch_pinn_loss / len(bx)
                loss = loss + current_alpha * batch_pinn_loss
                pinn_loss_sum += batch_pinn_loss.item() * len(bx)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(bx)

        train_loss = train_loss_sum / n_train
        epoch_pinn_loss = pinn_loss_sum / n_train if pinn_active else 0.0

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for bx, by_clf in val_loader:
                bx, by_clf = bx.to(device), by_clf.to(device)
                logits = model(bx)
                loss = criterion_clf(logits, by_clf)
                val_loss_sum += loss.item() * len(bx)

        val_loss = val_loss_sum / n_val
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pinn_loss'].append(epoch_pinn_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, os.path.join(OUTPUT_DIR, "best_model_pytorch.pth"))

        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d}/{EPOCHS:03d} | Train: {train_loss:.6f} | PINN: {epoch_pinn_loss:.6f} | Val: {val_loss:.6f} | Best Val: {best_val_loss:.6f}")

    # 8. Test Evaluation
    print("\nEvaluating on Independent Test Set (3,306 samples)...")
    model.load_state_dict(best_model_weights)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1)
            test_preds.append(preds.cpu().numpy())

    y_pred_shape = np.concatenate(test_preds, axis=0)

    acc = accuracy_score(y_shape_test, y_pred_shape) * 100.0
    bal_acc = balanced_accuracy_score(y_shape_test, y_pred_shape) * 100.0
    prec_macro = precision_score(y_shape_test, y_pred_shape, average='macro', zero_division=0) * 100.0
    rec_macro = recall_score(y_shape_test, y_pred_shape, average='macro', zero_division=0) * 100.0
    f1_macro = f1_score(y_shape_test, y_pred_shape, average='macro', zero_division=0) * 100.0
    mcc = matthews_corrcoef(y_shape_test, y_pred_shape)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (SINGLE-TASK CLASSIFICATION PINN):")
    print(f"  Accuracy:          {acc:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc:.2f}%")
    print(f"  Macro Precision:   {prec_macro:.2f}%")
    print(f"  Macro Recall:      {rec_macro:.2f}%")
    print(f"  Macro F1-Score:    {f1_macro:.2f}%")
    print(f"  Matthews CorrCoef: {mcc:.4f}")
    print(f"{'='*60}\n")

    res_df = pd.DataFrame([{
        'Model': '2D CNN Single-Task Classification (PINN)',
        'Train_Percent': f"{TRAIN_PERCENT}%",
        'Seed': RANDOM_STATE,
        'Alpha': ALPHA_INIT,
        'Warmup': PINN_ACTIVATION_EPOCH,
        'Accuracy': acc,
        'Balanced_Accuracy': bal_acc,
        'Precision_Macro': prec_macro,
        'Recall_Macro': rec_macro,
        'F1_Macro': f1_macro,
        'MCC': mcc
    }])
    res_df.to_csv(os.path.join(OUTPUT_DIR, "test_summary_metrics.csv"), index=False)

    pred_df = pd.DataFrame({
        'True_Class_ID': y_shape_test,
        'Pred_Class_ID': y_pred_shape,
        'True_Shape': [unique_shapes[s] for s in y_shape_test],
        'Pred_Shape': [unique_shapes[s] for s in y_pred_shape]
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"[OK] Saved results to: {OUTPUT_DIR}")

    # Real experiment evaluation (Experiment_1)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from load_real_experiment_data import evaluate_real_experiment
        evaluate_real_experiment(model, X_scaler, None, unique_shapes, OUTPUT_DIR, device=device)
    except Exception as e:
        print(f"[WARN] Real experiment evaluation encountered an error: {e}")

if __name__ == "__main__":
    main()
