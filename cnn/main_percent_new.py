import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

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

# Set default visible CUDA device before importing torch.
# run_eval_all.py may override this per subprocess for parallel GPU scheduling.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import time
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, max_error,
    classification_report, confusion_matrix, precision_recall_fscore_support,
    precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

def format_time(seconds):
    if seconds is None or np.isnan(seconds):
        return "N/A"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

try:
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# Force tempdir to /work/dat.lt19010205/Cong_Duc/check/tmp or local /tmp to prevent NFS '.nfsXXX' file locking
_work_tmp = '/work/dat.lt19010205/Cong_Duc/check/tmp'
for _tmp_candidate in [_work_tmp, '/tmp', '/var/tmp']:
    try:
        if _tmp_candidate == _work_tmp and not os.path.exists(_tmp_candidate):
            os.makedirs(_tmp_candidate, exist_ok=True)
        if os.path.exists(_tmp_candidate) and os.access(_tmp_candidate, os.W_OK):
            os.environ['TMPDIR'] = _tmp_candidate
            tempfile.tempdir = _tmp_candidate
            break
    except Exception:
        pass

# Patch multiprocessing.util._remove_temp_dir to safely ignore NFS Errno 16 errors
try:
    import multiprocessing.util as _mp_util
    import shutil as _shutil

    def _safe_remove_temp_dir(tempdir):
        try:
            _shutil.rmtree(tempdir, ignore_errors=True)
        except Exception:
            pass

    _mp_util._remove_temp_dir = _safe_remove_temp_dir
except Exception:
    pass


def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'main_percent_new.py'
    print(f"\n[FATAL] Unhandled exception in {script_name}", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _log_uncaught_exception

# ============================================================================
# RUNTIME DEVICE SETUP
# ============================================================================

print("GPU ACCELERATION SETUP (PYTORCH v2 - WITH PHYSICS GPU SUPPORT)")

# GPU visibility is already configured before importing torch.
visible_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[OK] PyTorch GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] Device: {device}")
    print(f"[OK] CUDA_VISIBLE_DEVICES={visible_cuda}")
    print(f"[OK] CUDA Version: {torch.version.cuda}")
else:
    print("[NOTE] No PyTorch GPU detected. Using CPU mode for training")

print(f"[OK] PyTorch version: {torch.__version__}")

# ============================================================================
# CONFIGURATION & PATH SETUP
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

REP_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "rep_code"))
if REP_CODE_DIR not in sys.path:
    sys.path.insert(0, REP_CODE_DIR)

try:
    from library_functions import Load_Data_With_Labels
except ImportError:
    print("[ERROR] Could not find 'library_functions.py'")
    sys.exit(1)

CHECK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.abspath(os.path.join(CHECK_DIR, ".."))

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH or not os.path.exists(DATA_PATH):
    for candidate in [
        os.path.join(CHECK_DIR, "Crack_Shape_Images"),
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
        os.path.join(SCRIPT_DIR, "labels.csv"),
        os.path.join(REP_CODE_DIR, "labels.csv"),
        os.path.join(PAPER_DIR, "labels.csv"),
    ]:
        if os.path.exists(candidate):
            LABELS_PATH = candidate
            break
OUTPUT_ROOT_DIR = os.environ.get(
    "PINN_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "Outputs_cnn_pinn")
)
if not os.path.isabs(OUTPUT_ROOT_DIR):
    OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_ROOT_DIR)
PINN_PROJECT_PATH = SCRIPT_DIR

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
if BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be >= 1")
EPOCHS = int(os.environ.get("EPOCHS", "300"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", os.environ.get("SEED", "42")))
LEARNING_RATE = 0.001
# Total loss target:
#   L = L_clf + ((L_w + L_l + L_d)/3) + alpha * L_pinn
# Recommended default alpha from recent sweep analysis.
ALPHA_INIT = float(os.environ.get("ALPHA_INIT", "0.02"))
if ALPHA_INIT <= 0:
    raise ValueError("ALPHA_INIT must be > 0")
PINN_ACTIVATION_EPOCH = int(os.environ.get("PINN_ACTIVATION_EPOCH", "100"))
if PINN_ACTIVATION_EPOCH < 0:
    raise ValueError("PINN_ACTIVATION_EPOCH must be >= 0")
PINN_BASE_GRAD_CLIP_NORM = float(os.environ.get("PINN_BASE_GRAD_CLIP_NORM", "1.0"))
if PINN_BASE_GRAD_CLIP_NORM <= 0:
    raise ValueError("PINN_BASE_GRAD_CLIP_NORM must be > 0")

# Per-epoch verbose logs (can be enabled via EPOCH_VERBOSE_LOG=1)
EPOCH_VERBOSE_LOG = os.environ.get("EPOCH_VERBOSE_LOG", "0").lower() in ("1", "true", "yes")

# Checkpoint/plot safeguards for long PINN runs.
# - Run heavy checkpoint plotting less frequently.
# - Keep physics visualization off by default during training checkpoints.
# - If enabled later, keep visualization sample count very small.
CHECKPOINT_EVERY = 20
CHECKPOINT_PHYSICS_VIS_ENABLED = False
CHECKPOINT_PHYSICS_NUM_SAMPLES = 1

# PINN update policy:
# - Warm-up phase (before PINN activation): data-loss optimizer step every batch.
# - PINN phase (after activation):
#   1) data-loss optimizer step every batch
#   2) one extra optimizer step at end of epoch from mean PINN loss.

# Train data percentage relative to the FULL dataset size.
# Use env var TRAIN_PERCENT=1..10 for your requested runs.
TRAIN_PERCENT = int(os.environ.get("TRAIN_PERCENT", "10"))
if not 1 <= TRAIN_PERCENT <= 100:
    raise ValueError("TRAIN_PERCENT must be in [1, 100]")

# PINN loss type: log(1 + SSE / sum(B_true^2))
PINN_LOSS_TYPE = "log1p_norm_sse"

# Loss normalization method: 'ema' (default - Exponential Moving Average scale normalization), 'initial', or 'none'
LOSS_NORM_METHOD = os.environ.get("LOSS_NORM_METHOD", "ema").strip().lower()
if LOSS_NORM_METHOD not in ("ema", "initial", "none"):
    raise ValueError(f"LOSS_NORM_METHOD must be one of ['ema', 'initial', 'none'], got: {LOSS_NORM_METHOD}")


# Keep outputs separated by train percentage AND loss type to avoid overwriting results.
RUN_TAG = pd.Timestamp.now().strftime("run_%Y%m%d_%H%M%S")
# Make the run tag configurable externally so the orchestration script can separate versions.
import os
if "CUSTOM_RUN_TAG" in os.environ:
    RUN_TAG = os.environ["CUSTOM_RUN_TAG"] + "_" + RUN_TAG
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, f"loss_{PINN_LOSS_TYPE}", f"train_{TRAIN_PERCENT:02d}pct", RUN_TAG)
final_ckpt_dir = os.path.join(OUTPUT_DIR, 'checkpoint_final')
PINN_LOSS_LOG_PATH = os.path.join(OUTPUT_DIR, "loss_pinn_epoch_log.csv")
DATA_LOSS_LOG_PATH = os.path.join(OUTPUT_DIR, "loss_data_epoch_log.csv")
EPOCH_LOSS_LOG_PATH = os.path.join(OUTPUT_DIR, "epoch_loss_log.csv")
SERIES_LABEL = os.environ.get("SERIES_LABEL", f"PINN alpha={ALPHA_INIT:.12g}")


def _atomic_write_df(df, path, **kwargs):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dirname, prefix='.tmp_', suffix='.csv') as tf:
            tmp = tf.name
            df.to_csv(tf.name, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _atomic_write_pickle(obj, path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dirname, prefix='.tmp_', suffix='.pkl') as tf:
            tmp = tf.name
            pickle.dump(obj, tf)
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _atomic_save_fig(fig_or_plt, path, **savefig_kwargs):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=dirname, prefix='.tmp_', suffix='.png') as tf:
            tmp = tf.name
        try:
            if hasattr(fig_or_plt, 'savefig'):
                fig_or_plt.savefig(tmp, **savefig_kwargs)
            else:
                import matplotlib.pyplot as _plt
                _plt.savefig(tmp, **savefig_kwargs)
        except Exception:
            import matplotlib.pyplot as _plt
            _plt.savefig(tmp, **savefig_kwargs)
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _atomic_write_text(text, path, mode='w', encoding='utf-8'):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    tmp = None
    try:
        # use binary write for robustness with encodings
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dirname, prefix='.tmp_', suffix='.txt', encoding=encoding) as tf:
            tmp = tf.name
            tf.write(text)
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _save_per_class_accuracy_plot(y_true, y_pred, class_labels, output_path, title="Per-class Accuracy"):
    """Compute per-class recall (accuracy) and save bar chart atomically."""
    try:
        from sklearn.metrics import precision_recall_fscore_support
        _, recall, _, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_labels)), zero_division=0)
    except Exception:
        # Fallback: compute manually
        recall = []
        support = []
        for cls_idx in range(len(class_labels)):
            mask = (y_true == cls_idx)
            support.append(int(mask.sum()))
            if mask.sum() == 0:
                recall.append(0.0)
            else:
                recall.append(float((y_pred[mask] == cls_idx).sum()) / float(mask.sum()))

    acc_percent = [r * 100.0 for r in recall]
    fig, ax = plt.subplots(figsize=(max(6, len(class_labels)*0.8), 4))
    bars = ax.bar(class_labels, acc_percent, color='tab:blue', alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(acc_percent):
        ax.text(i, v + 1.5, f"{v:.1f}%\n(n={support[i]})", ha='center', fontsize=9)
    plt.tight_layout()
    _atomic_save_fig(plt, output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _build_class_metrics_df(y_true, y_pred, class_labels, train_percent, pinn_alpha=0.0, pinn_active=False):
    labels = list(range(len(class_labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(np.sum(cm))

    rows = []
    for cls_idx, class_name in enumerate(class_labels):
        tp = int(cm[cls_idx, cls_idx])
        fn = int(cm[cls_idx, :].sum() - tp)
        fp = int(cm[:, cls_idx].sum() - tp)
        tn = int(total - tp - fn - fp)
        n_samples = int(support[cls_idx])
        class_accuracy = float(tp / n_samples) if n_samples else 0.0
        one_vs_rest_accuracy = float((tp + tn) / total) if total else 0.0
        rows.append({
            'class_index': cls_idx,
            'shape': class_name,
            'n_samples': n_samples,
            'n_correct': tp,
            'class_accuracy': class_accuracy,
            'class_accuracy_percent': class_accuracy * 100.0,
            'precision': float(precision[cls_idx]),
            'recall': float(recall[cls_idx]),
            'f1': float(f1[cls_idx]),
            'one_vs_rest_accuracy': one_vs_rest_accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'train_percent': train_percent,
            'pinn_alpha': float(pinn_alpha),
            'pinn_active': bool(pinn_active),
        })
    return pd.DataFrame(rows)


def _build_regression_metrics_by_target_df(y_true_wld, y_pred_wld, target_names, train_percent, pinn_alpha=0.0, pinn_active=False):
    rows = []
    for idx, target_name in enumerate(target_names):
        y_true_col = y_true_wld[:, idx]
        y_pred_col = y_pred_wld[:, idx]
        err = y_pred_col - y_true_col
        mse = mean_squared_error(y_true_col, y_pred_col)
        rmse = float(np.sqrt(mse))
        max_err = float(np.max(np.abs(err)))
        nrmse = float(calculate_nrmse(y_true_col, y_pred_col))
        rows.append({
            'target_index': idx,
            'target': target_name,
            'mae': float(mean_absolute_error(y_true_col, y_pred_col)),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2_score(y_true_col, y_pred_col)),
            'max_error': float(max_err),
            'nrmse': float(nrmse),
            'n_samples': int(len(y_true_col)),
            'true_mean': float(np.mean(y_true_col)),
            'pred_mean': float(np.mean(y_pred_col)),
            'error_mean': float(np.mean(err)),
            'error_std': float(np.std(err)),
            'train_percent': train_percent,
            'pinn_alpha': float(pinn_alpha),
            'pinn_active': bool(pinn_active),
        })
    return pd.DataFrame(rows)


def _flatten_regression_metrics(test_metrics):
    metric_row = {}
    for metric_name, values in test_metrics.items():
        if metric_name.endswith('_avg') or not isinstance(values, (list, tuple, np.ndarray)):
            continue
        for short_name, value in zip(('w', 'l', 'd'), values):
            metric_row[f'{metric_name}_{short_name}'] = float(value)
    return metric_row

def set_global_seed(seed: int):
    """Set deterministic behavior for numpy/torch and Python RNG."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

set_global_seed(RANDOM_STATE)

# ============================================================================
# RESUME TRAINING FROM CHECKPOINT (SET TO None TO TRAIN FROM SCRATCH) 
# ============================================================================
# Set RESUME_CHECKPOINT env var to resume from a checkpoint file path, or leave empty for fresh training.
# Example: RESUME_CHECKPOINT = r"C:\path\to\checkpoint_epoch_0100\checkpoint_full.pth"
# Example: RESUME_CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint_epoch_0100", "checkpoint_full.pth")
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "").strip() or None

# Safety: do not allow resuming from a different TRAIN_PERCENT bucket.
# This guarantees each percent setting is evaluated independently.
if RESUME_CHECKPOINT is not None:
    ckpt_abs = os.path.normcase(os.path.abspath(RESUME_CHECKPOINT))
    # Check within the specific loss type folder
    expected_percent_root = os.path.normcase(
        os.path.abspath(os.path.join(OUTPUT_ROOT_DIR, f"loss_{PINN_LOSS_TYPE}", f"train_{TRAIN_PERCENT:02d}pct"))
    )
    expected_prefix = expected_percent_root + os.sep
    if not ckpt_abs.startswith(expected_prefix):
        print(f"[WARN] Ignoring RESUME_CHECKPOINT from different percent/loss bucket: {RESUME_CHECKPOINT}")
        print(f"[WARN] Expected checkpoint under: {expected_percent_root}")
        print("[INFO] Training will start from scratch to keep evaluation independent.")
        RESUME_CHECKPOINT = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n[OK] Output folder: {OUTPUT_DIR}")
print(f"[CONFIG] Neural Network GPU: {'ENABLED' if device.type == 'cuda' else 'DISABLED'}\n")
print(f"[CONFIG] Global seed fixed: {RANDOM_STATE}")
print(f"[CONFIG] Train percent over FULL dataset: {TRAIN_PERCENT}%")
print(f"[CONFIG] Run tag: {RUN_TAG}")
print(f"[CONFIG] Evaluation isolation mode: ON (no cross-percent resume)")
print(f"[CONFIG] Epoch verbose log: {'ON' if EPOCH_VERBOSE_LOG else 'OFF'}")
print(f"[CONFIG] PINN loss type: {PINN_LOSS_TYPE.upper()}")
print("[CONFIG] PINN update schedule: data per-batch + mean PINN step at end of epoch")
print(f"[CONFIG] Loss mode: L = L_clf + ((L_w + L_l + L_d)/3) + alpha*L_pinn")
print("[CONFIG] PINN physics formula: log(1 + sum((B_true - B_pred)^2) / sum(B_true^2))")
print(f"[CONFIG] Weight init: alpha={ALPHA_INIT:.6f}")
print(f"[CONFIG] PINN activation epoch: {PINN_ACTIVATION_EPOCH}")
print(f"[CONFIG] PINN base grad clip norm: {PINN_BASE_GRAD_CLIP_NORM}")
print(f"[CONFIG] Milestone checkpoint interval: every {CHECKPOINT_EVERY} epochs")
print(f"[CONFIG] Checkpoint physics visualization: {'ON' if CHECKPOINT_PHYSICS_VIS_ENABLED else 'OFF'}")


def _ensure_epoch_loss_log_files():
    """Create per-epoch CSV logs for PINN and data losses if missing."""
    if (not os.path.exists(PINN_LOSS_LOG_PATH)) or os.path.getsize(PINN_LOSS_LOG_PATH) == 0:
        header = "epoch,pinn_active,alpha,train_physics_loss,weighted_pinn_loss,pinn_batches_applied,pinn_grad_norm,checkpoint_score\n"
        _atomic_write_text(header, PINN_LOSS_LOG_PATH)

    if (not os.path.exists(DATA_LOSS_LOG_PATH)) or os.path.getsize(DATA_LOSS_LOG_PATH) == 0:
        header = "epoch,train_clf_loss,train_reg_loss,train_data_loss,val_clf_loss,val_reg_loss,val_data_loss,train_clf_acc,val_clf_acc\n"
        _atomic_write_text(header, DATA_LOSS_LOG_PATH)

    # Unified epoch_loss_log.csv (cùng schema với no_pinn, dùng cho consolidation)
    if (not os.path.exists(EPOCH_LOSS_LOG_PATH)) or os.path.getsize(EPOCH_LOSS_LOG_PATH) == 0:
        import csv as _csv
        buf = io.StringIO()
        _csv.writer(buf).writerow([
            "epoch", "pinn_loss", "data_loss", "total_loss",
            "train_percent", "series_label", "mode", "alpha"
        ])
        _atomic_write_text(buf.getvalue(), EPOCH_LOSS_LOG_PATH)


def _append_epoch_loss_logs(
    epoch,
    alpha,
    pinn_active,
    train_physics_loss,
    pinn_batches_applied,
    pinn_grad_norm,
    checkpoint_score,
    train_clf_loss,
    train_reg_loss,
    train_data_loss,
    val_clf_loss,
    val_reg_loss,
    val_data_loss,
    train_clf_acc,
    val_clf_acc,
):
    """Append one epoch row into both separate CSV logs."""
    weighted_pinn_loss = alpha * train_physics_loss

    with open(PINN_LOSS_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(
            f"{epoch},{int(pinn_active)},{alpha:.12g},{train_physics_loss:.12g},{weighted_pinn_loss:.12g},"
            f"{pinn_batches_applied},{pinn_grad_norm:.12g},{checkpoint_score:.12g}\n"
        )

    with open(DATA_LOSS_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(
            f"{epoch},{train_clf_loss:.12g},{train_reg_loss:.12g},{train_data_loss:.12g},"
            f"{val_clf_loss:.12g},{val_reg_loss:.12g},{val_data_loss:.12g},{train_clf_acc:.12g},{val_clf_acc:.12g}\n"
        )

    # Unified epoch_loss_log.csv (cùng schema với no_pinn, dùng cho consolidation)
    pinn_loss_val = weighted_pinn_loss if pinn_active else 0.0
    total_loss_val = train_data_loss + pinn_loss_val
    import csv as _csv
    with open(EPOCH_LOSS_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        _csv.writer(f).writerow([
            epoch,
            f"{pinn_loss_val:.12g}",
            f"{train_data_loss:.12g}",
            f"{total_loss_val:.12g}",
            TRAIN_PERCENT,
            SERIES_LABEL,
            "pinn",
            f"{alpha:.12g}",
        ])


_ensure_epoch_loss_log_files()
print(f"[OK] PINN loss log file: {PINN_LOSS_LOG_PATH}")
print(f"[OK] Data loss log file: {DATA_LOSS_LOG_PATH}")
print(f"[OK] Epoch loss log file: {EPOCH_LOSS_LOG_PATH}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_nrmse(y_true, y_pred):
    """Normalized Root Mean Squared Error (NRMSE) normalized by target range in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    data_range = float(np.max(y_true) - np.min(y_true))
    if data_range <= 1e-12:
        return float("nan")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float((rmse / data_range) * 100.0)

TRAIN_LINE_COLOR = 'blue'
VAL_LINE_COLOR = 'red'

def save_metrics_barchart(clf_acc, bal_acc, f1_macro, mcc, output_dir, filename="test_classification_metrics_barchart.png", title="Classification Metrics (Q1 Standard)"):
    """Save a bar chart of the final classification metrics."""
    labels = ['Accuracy (%)', 'Balanced Acc (%)', 'F1-Macro (%)', 'MCC']
    values = [clf_acc * 100, bal_acc * 100, f1_macro * 100, mcc]
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Score / Percentage', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(output_dir, filename), dpi=220)
    plt.close()

def save_confusion_matrix(y_true, y_pred, unique_shapes, output_dir, filename="confusion_matrix.png", title="Classification Confusion Matrix"):
    """Save a labeled confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_shapes, yticklabels=unique_shapes, cmap='Blues')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(output_dir, filename), dpi=150)
    plt.close()

def _set_ieee_style():
    """Apply IEEE Journal publication style parameters to matplotlib"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'text.usetex': False,
    })

def plot_fig1_training_curves(history, output_dir, pinn_activated_epoch=None):
    """fig1_training_curves.png - IEEE Publication Quality PINN Training Curves"""
    _set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    markevery = max(1, len(epochs) // 10)
    
    # Subplot (a): Total & Data Loss
    axes[0].plot(epochs, history.get('train_loss', []), label='Train Loss', color='#1f77b4', linewidth=1.8, marker='o', markersize=4, markevery=markevery)
    axes[0].plot(epochs, history.get('val_loss', []), label='Val Loss', color='#d62728', linestyle='--', linewidth=1.8, marker='s', markersize=4, markevery=markevery)
    if pinn_activated_epoch and pinn_activated_epoch <= len(epochs):
        axes[0].axvline(x=pinn_activated_epoch, color='#2ca02c', linestyle=':', linewidth=1.8, label='PINN Active')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('(a) Total Loss Convergence', fontweight='bold', pad=8)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    axes[0].grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    
    # Subplot (b): Classification Accuracy
    if len(history.get('val_clf_acc', [])) > 0:
        axes[1].plot(epochs, [acc * 100 for acc in history.get('train_clf_acc', [])], label='Train Acc', color='#2ca02c', linewidth=1.8, marker='^', markersize=4, markevery=markevery)
        axes[1].plot(epochs, [acc * 100 for acc in history.get('val_clf_acc', [])], label='Val Acc', color='#ff7f0e', linestyle='--', linewidth=1.8, marker='v', markersize=4, markevery=markevery)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('(b) Classification Accuracy', fontweight='bold', pad=8)
    axes[1].set_ylim([0, 105])
    axes[1].legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    axes[1].grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    
    # Subplot (c): PINN Physics Loss
    phys_losses = history.get('physics_loss', [])
    if len(phys_losses) > 0:
        axes[2].plot(epochs, phys_losses, label='Physics Loss', color='#9467bd', linewidth=1.8, marker='D', markersize=3.5, markevery=markevery)
        if pinn_activated_epoch and pinn_activated_epoch <= len(epochs):
            axes[2].axvline(x=pinn_activated_epoch, color='#2ca02c', linestyle=':', linewidth=1.8, label='PINN Active')
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Physics Loss', fontweight='bold')
    axes[2].set_title('(c) PINN Residual Loss', fontweight='bold', pad=8)
    axes[2].legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    axes[2].grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(output_dir, 'fig1_training_curves.png'), dpi=300)
    plt.close(fig)

def plot_fig2_confusion_matrix(y_true, y_pred, unique_shapes, output_dir):
    """fig2_confusion_matrix.png - IEEE Normalized Confusion Matrix"""
    _set_ieee_style()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100.0
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=unique_shapes, yticklabels=unique_shapes,
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax, annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title('Test Set Confusion Matrix (%)', fontweight='bold', fontsize=12, pad=10)
    ax.set_ylabel('True Crack Shape', fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Crack Shape', fontweight='bold', fontsize=11)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(output_dir, 'fig2_confusion_matrix.png'), dpi=300)
    plt.close(fig)

def plot_fig3_regression_scatter(y_true_wld, y_pred_wld, output_dir):
    """fig3_regression_scatter.png - IEEE Ground Truth vs Prediction 1x3 Grid"""
    _set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics_names = ['Width ($W$, mm)', 'Length ($L$, mm)', 'Depth ($D$, %)']
    colors = ['#004c6d', '#c35100', '#4a2c5d']
    
    for i in range(3):
        ax = axes[i]
        yt = y_true_wld[:, i]
        yp = y_pred_wld[:, i]
        
        ax.scatter(yt, yp, alpha=0.55, s=25, color=colors[i], edgecolors='white', linewidth=0.3, label='Predictions')
        
        min_v = min(yt.min(), yp.min())
        max_v = max(yt.max(), yp.max())
        ax.plot([min_v, max_v], [min_v, max_v], color='#d62728', linestyle='--', linewidth=1.8, label='Ideal ($y=x$)')
        
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        max_err = np.max(np.abs(yt - yp))
        nrmse = calculate_nrmse(yt, yp)
        
        # Professional IEEE Text box metrics display (Q1 Standard)
        metric_str = f"MAE = {mae:.4f} mm\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f} mm\nNRMSE = {nrmse:.2f}%\nMax Err = {max_err:.4f} mm"
        ax.text(0.05, 0.95, metric_str, transform=ax.transAxes, fontsize=9.0, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
        
        ax.set_xlabel(f'True {metrics_names[i]}', fontweight='bold')
        ax.set_ylabel(f'Predicted {metrics_names[i]}', fontweight='bold')
        ax.set_title(f'({chr(97+i)}) {metrics_names[i].split()[0]} Regression', fontweight='bold', pad=8)
        ax.legend(fontsize=8.5, loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
        
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(output_dir, 'fig3_regression_scatter.png'), dpi=300)
    plt.close(fig)


def plot_fig4_tsne_latent_space(model, test_loader, y_shape_test, y_test_denorm, unique_shapes, output_dir, title_suffix=""):
    """
    fig4_tsne_latent_space.png / .pdf - Publication-Grade (Q1/Q2 Journal Standard) t-SNE Visualizations
    Produces high-resolution (300 DPI PNG + vector PDF) figures:
      1. Backbone t-SNE by Crack Shape
      2. Regression Embedding t-SNE by Crack Shape
      3. Regression Manifold colored by continuous W, L, D (mm)
      4. 2-Panel Composite (Backbone vs Regression)
      5. 4-Panel Composite Physics Geometry Manifold
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        import inspect

        model.eval()
        backbone_parts = []
        reg_parts = []
        with torch.no_grad():
            for batch_X, _, _ in test_loader:
                batch_X = batch_X.to(device)
                b_feat = model.backbone(batch_X)
                b_feat = b_feat.view(b_feat.size(0), -1)
                r_feat = model.regressor_backbone(b_feat)
                backbone_parts.append(b_feat.cpu().numpy())
                reg_parts.append(r_feat.cpu().numpy())

        backbone_feats = np.concatenate(backbone_parts, axis=0)
        reg_feats = np.concatenate(reg_parts, axis=0)
        class_ids = np.asarray(y_shape_test)
        true_wld = np.asarray(y_test_denorm)

        rng = np.random.default_rng(42)
        keep = np.arange(len(class_ids))
        if len(keep) > 750:
            keep = np.sort(rng.choice(keep, size=750, replace=False))
        class_ids = class_ids[keep]
        true_wld = true_wld[keep]

        pub_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        reg_cmaps = {"W": "viridis", "L": "plasma", "D": "turbo"}
        unit_map = {"W": "mm", "L": "mm", "D": "mm"}

        projections = {}
        metrics_dict = {}

        for fkey, flabel, raw_f in [
            ("backbone", "Shared Backbone", backbone_feats),
            ("regression", "Regression Embedding", reg_feats),
        ]:
            feat_subset = np.asarray(raw_f)[keep]
            scaled = StandardScaler().fit_transform(feat_subset)
            pre_dim = min(50, scaled.shape[1], scaled.shape[0] - 1)
            pre_pca = PCA(n_components=pre_dim, random_state=42).fit_transform(scaled)
            perp = min(30.0, max(5.0, (len(pre_pca) - 1) / 3.0))

            tsne_sig = inspect.signature(TSNE)
            lr_def = tsne_sig.parameters["learning_rate"].default
            tsne_kw = {
                "n_components": 2,
                "perplexity": perp,
                "learning_rate": "auto" if isinstance(lr_def, str) else 200.0,
                "init": "pca",
                "random_state": 42,
                "method": "barnes_hut",
                "n_jobs": -1,
            }
            iter_arg = "max_iter" if "max_iter" in tsne_sig.parameters else "n_iter"
            tsne_kw[iter_arg] = 1000
            tsne_proj = TSNE(**tsne_kw).fit_transform(pre_pca)
            projections[fkey] = tsne_proj

            if len(np.unique(class_ids)) > 1:
                sil = float(silhouette_score(scaled, class_ids))
                cal = float(calinski_harabasz_score(scaled, class_ids))
                dav = float(davies_bouldin_score(scaled, class_ids))
            else:
                sil = cal = dav = float("nan")
            metrics_dict[fkey] = {"silhouette": sil, "calinski": cal, "davies": dav}

            # Individual Large High-Res Plot (fig4)
            fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
            for c_idx, c_name in enumerate(unique_shapes):
                mask = class_ids == c_idx
                ax.scatter(
                    tsne_proj[mask, 0],
                    tsne_proj[mask, 1],
                    s=65,
                    alpha=0.82,
                    color=pub_colors[c_idx % len(pub_colors)],
                    edgecolors="black",
                    linewidth=0.35,
                    label=c_name,
                )
            ax.set_xlabel(r"$\mathbf{t\text{-}SNE\ Dimension\ 1}$", fontsize=16, fontweight="bold")
            ax.set_ylabel(r"$\mathbf{t\text{-}SNE\ Dimension\ 2}$", fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=14)
            t_suf = f" ({title_suffix})" if title_suffix else ""
            ax.set_title(f"{flabel} t-SNE by Crack Shape{t_suf}", fontsize=17, fontweight="bold", pad=12)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
            ax.legend(
                title="Crack Shape",
                title_fontsize=14,
                fontsize=13,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
                framealpha=0.95,
            )

            if np.isfinite(sil):
                ax.text(
                    0.03, 0.97,
                    f"Silhouette (S): {sil:.3f}\nDBI: {dav:.3f}\nCH: {cal:.1f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888", alpha=0.92),
                )
            fig.tight_layout()
            for ext in (".png", ".pdf"):
                _atomic_save_fig(plt, os.path.join(output_dir, f"fig4_tsne_{fkey}_by_class{ext}"), dpi=300)
            plt.close(fig)

        # 2-Panel Composite (Backbone vs Regression)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), dpi=300)
        for s_idx, (fk, fl) in enumerate([("backbone", "Shared Backbone"), ("regression", "Regression Embedding")]):
            ax = axes[s_idx]
            proj = projections[fk]
            for c_idx, c_name in enumerate(unique_shapes):
                mask = class_ids == c_idx
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    s=60,
                    alpha=0.82,
                    color=pub_colors[c_idx % len(pub_colors)],
                    edgecolors="black",
                    linewidth=0.35,
                    label=c_name,
                )
            ax.set_xlabel(r"$\mathbf{t\text{-}SNE\ Dim\ 1}$", fontsize=15, fontweight="bold")
            ax.set_ylabel(r"$\mathbf{t\text{-}SNE\ Dim\ 2}$", fontsize=15, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=13)
            ax.set_title(f"({chr(97 + s_idx)}) {fl} Latent Space", fontsize=17, fontweight="bold", pad=10)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)

            sil = metrics_dict[fk]["silhouette"]
            if np.isfinite(sil):
                ax.text(
                    0.03, 0.97,
                    f"Silhouette: {sil:.3f}\nDBI: {metrics_dict[fk]['davies']:.3f}\nCH: {metrics_dict[fk]['calinski']:.1f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#888888", alpha=0.92),
                )
            if s_idx == 1:
                ax.legend(
                    title="Crack Shape",
                    title_fontsize=13,
                    fontsize=12,
                    loc="upper right",
                    frameon=True,
                    facecolor="white",
                    edgecolor="#cccccc",
                    framealpha=0.95,
                )
        fig.suptitle(f"t-SNE Latent Space Representation{t_suf}", fontsize=19, fontweight="bold", y=0.99)
        fig.tight_layout()
        for ext in (".png", ".pdf"):
            _atomic_save_fig(plt, os.path.join(output_dir, f"fig4_composite_tsne_2panel{ext}"), dpi=300)
        plt.close(fig)

        # 4-Panel Composite Geometry Manifold
        r_proj = projections["regression"]
        fig, axes = plt.subplots(2, 2, figsize=(18, 15), dpi=300)
        ax_sh = axes[0, 0]
        for c_idx, c_name in enumerate(unique_shapes):
            mask = class_ids == c_idx
            ax_sh.scatter(
                r_proj[mask, 0],
                r_proj[mask, 1],
                s=60,
                alpha=0.82,
                color=pub_colors[c_idx % len(pub_colors)],
                edgecolors="black",
                linewidth=0.35,
                label=c_name,
            )
        ax_sh.set_title("(a) Shape Class Distribution", fontsize=16, fontweight="bold", pad=8)
        ax_sh.set_xlabel(r"$\mathbf{t\text{-}SNE\ Dim\ 1}$", fontsize=14, fontweight="bold")
        ax_sh.set_ylabel(r"$\mathbf{t\text{-}SNE\ Dim\ 2}$", fontsize=14, fontweight="bold")
        ax_sh.tick_params(axis="both", which="major", labelsize=13)
        ax_sh.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
        ax_sh.legend(title="Shape", title_fontsize=12, fontsize=11, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

        for t_idx, (t_name, t_ax, p_label) in enumerate([
            ("W", axes[0, 1], "(b) Crack Width $W$ (mm)"),
            ("L", axes[1, 0], "(c) Crack Length $L$ (mm)"),
            ("D", axes[1, 1], "(d) Crack Depth $D$ (mm)"),
        ]):
            sc = t_ax.scatter(
                r_proj[:, 0],
                r_proj[:, 1],
                c=true_wld[:, t_idx],
                cmap=reg_cmaps[t_name],
                s=65,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.3,
            )
            cb = fig.colorbar(sc, ax=t_ax, fraction=0.046, pad=0.04)
            cb.set_label(f"True {t_name} ({unit_map[t_name]})", fontsize=13, fontweight="bold")
            cb.ax.tick_params(labelsize=12)
            t_ax.set_title(p_label, fontsize=16, fontweight="bold", pad=8)
            t_ax.set_xlabel(r"$\mathbf{t\text{-}SNE\ Dim\ 1}$", fontsize=14, fontweight="bold")
            t_ax.set_ylabel(r"$\mathbf{t\text{-}SNE\ Dim\ 2}$", fontsize=14, fontweight="bold")
            t_ax.tick_params(axis="both", which="major", labelsize=13)
            t_ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)

        fig.suptitle(f"Physics Geometry Disentanglement on t-SNE Manifold{t_suf}", fontsize=19, fontweight="bold", y=0.99)
        fig.tight_layout()
        for ext in (".png", ".pdf"):
            _atomic_save_fig(plt, os.path.join(output_dir, f"fig4_composite_manifold_4panel{ext}"), dpi=300)
        plt.close(fig)

        print(f"[OK] Publication t-SNE Figures saved to: {output_dir}")
    except Exception as e:
        print(f"[WARNING] Could not generate t-SNE visualization: {e}")



def generate_comprehensive_metrics_visualizations(test_metrics, clf_acc, prec, rec, f1, 
                                                  y_test_denorm, y_pred_wld_denorm, y_shape_test, 
                                                  unique_shapes, output_dir):
    """Generate paper-ready publication quality metric visualizations and violin plots for test set."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE PAPER-QUALITY METRIC VISUALIZATIONS & VIOLIN PLOTS")
    print("="*80)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })

    metrics_names = ['Width (W)', 'Length (L)', 'Depth (D)']
    paper_colors = ['#004c6d', '#c35100', '#6b2d5c'] # Deep Navy, Amber/Terracotta, Deep Plum
    
    # ---- 1. REGRESSION METRICS COMPARISON (Bar Chart) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Regression Metrics (Q1 Standard) - Test Set', fontsize=16, fontweight='bold')
    
    # MAE comparison
    ax = axes[0, 0]
    mae_vals = test_metrics['mae']
    ax.bar(metrics_names, mae_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('MAE (mm)', fontweight='bold')
    ax.set_title('Mean Absolute Error (MAE)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(mae_vals):
        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    ax = axes[0, 1]
    rmse_vals = test_metrics['rmse']
    ax.bar(metrics_names, rmse_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('RMSE (mm)', fontweight='bold')
    ax.set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(rmse_vals):
        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    ax = axes[0, 2]
    r2_vals = test_metrics['r2']
    ax.bar(metrics_names, r2_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Coefficient of Determination ($R^2$)', fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(r2_vals):
        ax.text(i, max(v * 0.95, 0.05), f'{v:.4f}', ha='center', va='bottom', fontweight='bold', color='white' if v > 0.5 else 'black')
    
    # Max Error comparison
    ax = axes[1, 0]
    max_err_vals = test_metrics.get('max_error', [np.max(np.abs(y_test_denorm[:, j] - y_pred_wld_denorm[:, j])) for j in range(3)])
    ax.bar(metrics_names, max_err_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Max Error (mm)', fontweight='bold')
    ax.set_title('Maximum Absolute Error', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(max_err_vals):
        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    # NRMSE comparison
    ax = axes[1, 1]
    nrmse_vals = test_metrics.get('nrmse', [calculate_nrmse(y_test_denorm[:, j], y_pred_wld_denorm[:, j]) for j in range(3)])
    ax.bar(metrics_names, nrmse_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('NRMSE (%)', fontweight='bold')
    ax.set_title('Normalized RMSE (%)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(nrmse_vals):
        ax.text(i, v * 1.02, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Summary box
    axes[1, 2].axis('off')
    summary_text = "Regression Summary (Q1):\n\n"
    summary_text += f"MAE Avg:       {np.mean(mae_vals):.4f} mm\n"
    summary_text += f"RMSE Avg:      {np.mean(rmse_vals):.4f} mm\n"
    summary_text += f"R² Avg:        {np.mean(r2_vals):.4f}\n"
    summary_text += f"Max Error Avg: {np.mean(max_err_vals):.4f} mm\n"
    summary_text += f"NRMSE Avg:     {np.mean(nrmse_vals):.2f}%\n"
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, fontsize=12,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    regression_metrics_path = os.path.join(output_dir, 'all_regression_metrics_comparison.png')
    _atomic_save_fig(plt, regression_metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved regression metrics comparison: {regression_metrics_path}")
    
    # ---- 2. CLASSIFICATION METRICS (Bar Chart) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bal_acc = float(balanced_accuracy_score(y_shape_test, y_pred_shape))
    f1_macro = float(f1_score(y_shape_test, y_pred_shape, average='macro', zero_division=0))
    mcc = float(matthews_corrcoef(y_shape_test, y_pred_shape))
    metrics = ['Accuracy (%)', 'Balanced Acc (%)', 'F1-Macro (%)', 'MCC']
    values = [clf_acc * 100.0, bal_acc * 100.0, f1_macro * 100.0, mcc]
    colors_clf = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    
    bars = ax.bar(metrics, values, color=colors_clf, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Score / Percentage', fontweight='bold', fontsize=12)
    ax.set_title('Classification Metrics (Q1 Standard) - Test Set', fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height < 1.0 else height + 1.0,
                f'{val:.4f}' if abs(val) <= 1.0 else f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    clf_metrics_path = os.path.join(output_dir, 'all_classification_metrics.png')
    _atomic_save_fig(plt, clf_metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved classification metrics: {clf_metrics_path}")
    
    # ---- 3. PREDICTION vs TRUE SCATTER PLOTS ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Predictions vs Ground Truth - Test Set', fontsize=15, fontweight='bold')
    
    for i, (ax, name, color) in enumerate(zip(axes, metrics_names, paper_colors)):
        y_true = y_test_denorm[:, i]
        y_pred = y_pred_wld_denorm[:, i]
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=25, color=color, edgecolors='k', linewidth=0.3)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2.5, label='Ideal ($y=x$)')
        
        ax.set_xlabel(f'True {name}', fontweight='bold')
        ax.set_ylabel(f'Predicted {name}', fontweight='bold')
        ax.set_title(f'{name}\n$R^2={test_metrics["r2"][i]:.4f}$, RMSE={test_metrics["rmse"][i]:.4f}', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'all_predictions_vs_true.png')
    _atomic_save_fig(plt, scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved prediction scatter plots: {scatter_path}")
    
    # ---- 4. RESIDUAL PLOTS ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Residuals - Test Set', fontsize=15, fontweight='bold')
    
    for i, (ax, name, color) in enumerate(zip(axes, metrics_names, paper_colors)):
        y_true = y_test_denorm[:, i]
        y_pred = y_pred_wld_denorm[:, i]
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, s=25, color=color, edgecolors='k', linewidth=0.3)
        ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=2.5)
        ax.set_xlabel(f'Predicted {name}', fontweight='bold')
        ax.set_ylabel(f'Residual (True - Pred)', fontweight='bold')
        ax.set_title(f'{name} Residuals\nMean={np.mean(residuals):.4f}, Std={np.std(residuals):.4f}', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    residual_path = os.path.join(output_dir, 'all_residuals_plots.png')
    _atomic_save_fig(plt, residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved residual plots: {residual_path}")
    
    # ---- 5. ERROR DISTRIBUTION ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Error Distribution - Test Set', fontsize=15, fontweight='bold')
    
    for i, (ax, name, color) in enumerate(zip(axes, metrics_names, paper_colors)):
        y_true = y_test_denorm[:, i]
        y_pred = y_pred_wld_denorm[:, i]
        errors = np.abs(y_true - y_pred)
        
        ax.hist(errors, bins=30, color=color, alpha=0.85, edgecolor='black', linewidth=1.0)
        ax.axvline(x=np.mean(errors), color='#d62728', linestyle='--', linewidth=2, label=f'Mean={np.mean(errors):.4f}')
        ax.axvline(x=np.median(errors), color='#2ca02c', linestyle=':', linewidth=2, label=f'Median={np.median(errors):.4f}')
        ax.set_xlabel('Absolute Error', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{name} Error Distribution', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    error_dist_path = os.path.join(output_dir, 'all_error_distributions.png')
    _atomic_save_fig(plt, error_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved error distributions: {error_dist_path}")
    
    # ---- 6. HEATMAP: Error by True Value ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Error Heatmap (grouped by True Value)', fontsize=15, fontweight='bold')
    
    for i, (ax, name, color) in enumerate(zip(axes, metrics_names, paper_colors)):
        y_true = y_test_denorm[:, i]
        y_pred = y_pred_wld_denorm[:, i]
        errors = np.abs(y_true - y_pred)
        
        h = ax.hist2d(y_true, errors, bins=[20, 20], cmap='YlOrRd')
        ax.set_xlabel(f'True {name}', fontweight='bold')
        ax.set_ylabel('Absolute Error', fontweight='bold')
        ax.set_title(f'{name} Error vs True Value', fontweight='bold')
        plt.colorbar(h[3], ax=ax, label='Count')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'all_error_heatmaps.png')
    _atomic_save_fig(plt, heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved error heatmaps: {heatmap_path}")

    # ---- 7. VIOLIN PLOT: True vs Predicted Value Distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (ax, name) in enumerate(zip(axes, metrics_names)):
        v_data = pd.DataFrame({
            'Value': np.concatenate([y_test_denorm[:, i], y_pred_wld_denorm[:, i]]),
            'Type': ['True Ground Truth'] * len(y_test_denorm) + ['Model Prediction'] * len(y_pred_wld_denorm)
        })
        sns.violinplot(
            data=v_data, x='Type', y='Value', hue='Type',
            palette={'True Ground Truth': paper_colors[i], 'Model Prediction': '#ffa600'},
            inner='box', ax=ax, legend=False, cut=0
        )
        ax.set_title(f'Distribution Comparison: {name}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(f'{name} Value', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    violin_true_pred_path = os.path.join(output_dir, 'violin_wld_true_vs_pred.png')
    _atomic_save_fig(plt, violin_true_pred_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 8. VIOLIN PLOT: Signed Prediction Errors (Pred - True) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    err_w = y_pred_wld_denorm[:, 0] - y_test_denorm[:, 0]
    err_l = y_pred_wld_denorm[:, 1] - y_test_denorm[:, 1]
    err_d = y_pred_wld_denorm[:, 2] - y_test_denorm[:, 2]
    err_df = pd.DataFrame({
        'Error': np.concatenate([err_w, err_l, err_d]),
        'Parameter': ['Width (W)'] * len(err_w) + ['Length (L)'] * len(err_l) + ['Depth (D)'] * len(err_d)
    })
    sns.violinplot(
        data=err_df, x='Parameter', y='Error', hue='Parameter',
        palette={'Width (W)': paper_colors[0], 'Length (L)': paper_colors[1], 'Depth (D)': paper_colors[2]},
        inner='quartile', ax=ax, legend=False, cut=0
    )
    ax.axhline(0, color='#d62728', linestyle='--', linewidth=2, label='Zero Error Line')
    ax.set_title('Test Set Prediction Signed Error Distributions ($Pred - True$)', fontweight='bold')
    ax.set_xlabel('Target Parameter', fontweight='bold')
    ax.set_ylabel('Signed Error', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    violin_signed_err_path = os.path.join(output_dir, 'violin_wld_signed_errors.png')
    _atomic_save_fig(plt, violin_signed_err_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 9. VIOLIN PLOT BY SHAPE CLASS ----
    if y_shape_test is not None and unique_shapes is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        shape_names_test = [unique_shapes[int(s)] for s in y_shape_test]
        for i, (ax, name) in enumerate(zip(axes, metrics_names)):
            shape_err_df = pd.DataFrame({
                'Signed Error': y_pred_wld_denorm[:, i] - y_test_denorm[:, i],
                'Crack Shape': shape_names_test
            })
            sns.violinplot(
                data=shape_err_df, x='Crack Shape', y='Signed Error', hue='Crack Shape',
                palette='Set2', inner='quartile', ax=ax, legend=False, cut=0
            )
            ax.axhline(0, color='#d62728', linestyle='--', linewidth=1.8)
            ax.set_title(f'Error Distribution by Shape: {name}', fontweight='bold')
            ax.set_xlabel('Crack Shape', fontweight='bold')
            ax.set_ylabel(f'Error ({name})', fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
            ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        violin_by_shape_path = os.path.join(output_dir, 'violin_wld_by_shape.png')
        _atomic_save_fig(plt, violin_by_shape_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        violin_by_shape_path = None
    
    print("\n[OK] All comprehensive metric visualizations & violin plots saved!")
    return {
        'regression_metrics': regression_metrics_path,
        'classification_metrics': clf_metrics_path,
        'predictions_scatter': scatter_path,
        'residuals': residual_path,
        'error_distributions': error_dist_path,
        'error_heatmaps': heatmap_path,
        'violin_true_vs_pred': violin_true_pred_path,
        'violin_signed_errors': violin_signed_err_path,
        'violin_by_shape': violin_by_shape_path
    }

def save_regression_plots(y_true_wld, y_pred_wld, output_dir, metrics_names=['Width (W)', 'Length (L)', 'Depth (D)'], prefix='', title_prefix='', y_shape=None, unique_shapes=None, y_scaler=None):
    """Save W/L/D regression scatter plots with paper-ready publication styling and optional denormalization."""
    paper_colors = ['#004c6d', '#c35100', '#6b2d5c']
    
    # Denormalize if scaler provided
    if y_scaler is not None:
        if np.any(np.isnan(y_true_wld)) or np.any(np.isnan(y_pred_wld)):
            print(f"[WARNING] NaN detected in input data before denormalization")
            y_pred_wld = np.where(np.isnan(y_pred_wld), np.nan_to_num(np.nanmean(y_pred_wld, axis=0), nan=0.0), y_pred_wld)
            y_true_wld = np.where(np.isnan(y_true_wld), np.nan_to_num(np.nanmean(y_true_wld, axis=0), nan=0.0), y_true_wld)
        
        y_true_wld_denorm = y_scaler.inverse_transform(y_true_wld)
        y_pred_wld_denorm = y_scaler.inverse_transform(y_pred_wld)
        
        if np.any(np.isnan(y_pred_wld_denorm)):
            y_pred_wld_denorm = np.where(np.isnan(y_pred_wld_denorm), np.nan_to_num(np.nanmean(y_pred_wld_denorm, axis=0), nan=0.0), y_pred_wld_denorm)
    else:
        y_true_wld_denorm = y_true_wld
        y_pred_wld_denorm = y_pred_wld
    
    sns.set_theme(style="whitegrid")
    for i, (name, color) in enumerate(zip(metrics_names, paper_colors)):
        fig, ax = plt.subplots(figsize=(9, 7))
        
        ax.scatter(y_true_wld_denorm[:, i], y_pred_wld_denorm[:, i], 
                  alpha=0.6, s=35, color=color, edgecolors='black', linewidth=0.4, label='Predictions')
        
        min_val = min(y_true_wld_denorm[:, i].min(), y_pred_wld_denorm[:, i].min())
        max_val = max(y_true_wld_denorm[:, i].max(), y_pred_wld_denorm[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2.5, label='Ideal ($y=x$)', alpha=0.9)
        
        y_true_col = y_true_wld_denorm[:, i]
        y_pred_col = y_pred_wld_denorm[:, i]
        valid_mask = ~(np.isnan(y_true_col) | np.isnan(y_pred_col))
        if np.sum(valid_mask) < len(y_true_col):
            y_true_col = y_true_col[valid_mask]
            y_pred_col = y_pred_col[valid_mask]
        
        if len(y_true_col) > 0:
            mae = mean_absolute_error(y_true_col, y_pred_col)
            rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
            r2 = r2_score(y_true_col, y_pred_col)
            max_err = np.max(np.abs(y_true_col - y_pred_col))
            nrmse = calculate_nrmse(y_true_col, y_pred_col)
        else:
            mae = rmse = r2 = max_err = nrmse = np.nan
        
        ax.set_xlabel(f'True Ground Truth: {name}', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Model Prediction: {name}', fontsize=13, fontweight='bold')
        title_str = f'{title_prefix}{name} Regression\nMAE: {mae:.4f} mm | RMSE: {rmse:.4f} mm | $R^2$: {r2:.4f} | MaxErr: {max_err:.4f} mm | NRMSE: {nrmse:.2f}%'
        ax.set_title(title_str, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        filename = f"{prefix}regression_{name.split()[0].lower()}.png"
        fig.tight_layout()
        _atomic_save_fig(plt, os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)

def compute_physics_loss_autograd(H_true, w_pred, l_pred, d_pred, shape_name, shape_map, y_scaler=None, X_scaler=None, 
                                 w_true=None, l_true=None, d_true=None, log_file=None):
    """
    Differentiable PINN physics loss.
    All operations remain in torch so gradients can flow to model outputs.
    
    Args:
        H_true: True magnetic field (raw from dataset)
        w_pred, l_pred, d_pred: Predicted W, L, D (normalized)
        shape_name: True shape class name
        shape_map: Physics modules mapping
        y_scaler: Scaler for denormalizing W/L/D
        X_scaler: Scaler for restoring field channel scale
        w_true, l_true, d_true: Ground-truth W/L/D (normalized), optional logs
        log_file: Optional path to append debug logs
    """
    import math

    base_zero = (w_pred + l_pred + d_pred) * 0.0
    if shape_name not in shape_map:
        return base_zero

    physics_module = shape_map[shape_name]

    # Keep tensors in float32 for physics kernels.
    w_norm = w_pred.float().reshape(())
    l_norm = l_pred.float().reshape(())
    d_norm = d_pred.float().reshape(())

    if y_scaler is not None:
        # Denormalize with the current target scaling: y_real = y_norm * y_max
        w_max, l_max, d_max = [float(v) for v in y_scaler.data_max_]
        
        w_val = w_norm * w_max
        l_val = l_norm * l_max
        d_val = d_norm * d_max
    else:
        w_val, l_val, d_val = w_norm, l_norm, d_norm

    # Physical bounds as soft safety.
    w_val = torch.clamp(w_val, 0.3, 1.5)
    l_val = torch.clamp(l_val, 2.0, 20.0)
    d_val = torch.clamp(d_val, 0.1, 3.0)

    # Physics constants
    freq, sicma, mu = 5000.0, 35461000.0, 0.0000012566
    csi, K, I, G = 0.085, 1.5, 0.01, 3981.0
    delta = 1.0 / math.sqrt(math.pi * freq * mu * sicma) * 1000.0
    z_lift = 1.0
    N, Res = 16, 0.78

    try:
        if shape_name == 'Ellipse':
            H_tensor = physics_module.compute_map_gpu(w_val, l_val, d_val, delta, z_lift)
            H_raw = (csi / (4.0 * math.pi)) * H_tensor * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

        elif shape_name == 'Rectangular':
            H_tensor, _, _ = physics_module.compute_magnetic_field(
                w_val, l_val, d_val, delta, z_lift, N, Res, angle=0, K=K, I=I, G=G, csi=csi
            )
            H_pred = torch.rot90(H_tensor, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

        elif shape_name == 'Triangular':
            H_tensor = physics_module.compute_triangular_map_gpu(w_val, l_val, d_val, delta, z_lift)
            H_raw = (csi / (4.0 * math.pi)) * H_tensor * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

        elif shape_name == 'Step_R':
            xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
            H_val = physics_module.calculate_field_Step_R(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
            H_raw = H_val * (csi / (4.0 * math.pi)) * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.flip(H_pred, dims=(1,))
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)

        elif shape_name == 'Step_T':
            xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
            H_val = physics_module.calculate_field_Step_T(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
            H_raw = H_val * (csi / (4.0 * math.pi)) * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.flip(H_pred, dims=(1,))
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
        else:
            return base_zero

        H_true_t = H_true.float()
        if H_true_t.ndim == 3:
            H_true_t = H_true_t[:, :, 0]

        # Denormalize the field channel so physics loss compares values in the
        # same scale as H_pred. X_scaler was fit with StandardScaler on the
        # image channels, so channel 0 can be restored directly with mean/std.
        if X_scaler is not None and hasattr(X_scaler, 'mean_') and hasattr(X_scaler, 'scale_'):
            field_mean = float(X_scaler.mean_[0])
            field_scale = float(X_scaler.scale_[0])
            H_true_t = H_true_t * field_scale + field_mean

        # Helper function to log messages
        def log_msg(msg):
            if log_file:
                try:
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                except:
                    pass
        
        # Log true parameters
        if w_true is not None and l_true is not None and d_true is not None:
            w_true_val = float(w_true.item() * y_scaler.data_max_[0]) if y_scaler is not None else float(w_true.item())
            l_true_val = float(l_true.item() * y_scaler.data_max_[1]) if y_scaler is not None else float(l_true.item())
            d_true_val = float(d_true.item() * y_scaler.data_max_[2]) if y_scaler is not None else float(d_true.item())
            
            log_msg(f"\n{'='*80}")
            log_msg(f"[PHYSICS] Shape: {shape_name}")
            log_msg(f"{'='*80}")
            log_msg(f"TRUE: W: {w_true.item():.6f} → {w_true_val:.6f} | L: {l_true.item():.6f} → {l_true_val:.6f} | D: {d_true.item():.6f} → {d_true_val:.6f}")
        
        # Log predicted parameters
        log_msg(f"PRED: W: {w_norm.item():.6f} → {w_val.item():.6f} | L: {l_norm.item():.6f} → {l_val.item():.6f} | D: {d_norm.item():.6f} → {d_val.item():.6f}")
        
        log_msg(f"H_true  - Range: [{H_true_t.min().item():.6f}, {H_true_t.max().item():.6f}] | Mean: {H_true_t.mean().item():.6f} | Std: {H_true_t.std().item():.6f}")
        log_msg(f"H_pred  - Range: [{H_pred.min().item():.6f}, {H_pred.max().item():.6f}] | Mean: {H_pred.mean().item():.6f} | Std: {H_pred.std().item():.6f}")
        
        # Physics loss formula:
        #   L_pinn = log(1 + sum((B_true - B_pred)^2) / sum(B_true^2))
        diff = H_true_t - H_pred
        sse = torch.sum(diff * diff)
        true_energy = torch.sum(H_true_t * H_true_t)
        norm_ratio = sse / (true_energy + 1e-12)
        physics_loss = torch.log1p(norm_ratio)

        log_msg(f"Physics Loss: {physics_loss.item():.8f}")
        log_msg(f"  SSE: {sse.item():.8f} | TrueEnergy: {true_energy.item():.8f} | Ratio: {norm_ratio.item():.8f}")
        
        return physics_loss

    except Exception as exc:
        if log_file:
            try:
                with open(log_file, 'a') as f:
                    f.write(f"[PHYSICS][WARN] Failed physics branch for shape={shape_name}: {exc}\n")
            except Exception:
                pass
        # Keep graph valid even if one physics branch fails.
        return base_zero


# ============================================================================
# LOSS HELPERS & DYNAMIC LOSS NORMALIZATION
# ============================================================================

class LossNormalizer:
    """
    Dynamically track and normalize loss components during training
    so that their values are on comparable scales (~1.0).
    Supported methods:
      - 'ema': Exponential Moving Average loss scaling (L_norm = L / EMA(L))
      - 'initial': Initial scale normalization (L_norm = L / L_initial)
      - 'none': Raw loss without normalization
    """
    def __init__(self, method='ema', momentum=0.99, epsilon=1e-8):
        self.method = str(method).lower()
        self.momentum = momentum
        self.epsilon = epsilon
        self.scales = {}
        self.initial_scales = {}

    def update_and_normalize(self, name, loss_tensor):
        if self.method == 'none' or loss_tensor is None:
            return loss_tensor

        val = float(loss_tensor.detach().item()) if torch.is_tensor(loss_tensor) else float(loss_tensor)
        
        if name not in self.scales:
            self.scales[name] = max(val, self.epsilon)
            self.initial_scales[name] = max(val, self.epsilon)
        else:
            if self.method == 'ema':
                self.scales[name] = self.momentum * self.scales[name] + (1.0 - self.momentum) * val

        scale = self.initial_scales[name] if self.method == 'initial' else self.scales[name]
        return loss_tensor / (scale + self.epsilon)

    def get_scale(self, name):
        if self.method == 'initial':
            return self.initial_scales.get(name, 1.0)
        return self.scales.get(name, 1.0)


# Global loss normalizer instance
loss_normalizer = LossNormalizer(method=LOSS_NORM_METHOD)


def compute_regression_losses(y_pred_wld, y_wld, criterion_reg):
    """
    Compute individual regression losses for W, L, D from one joint output.
    
    Args:
        y_pred_wld: Predicted [W, L, D] with shape (N, 3)
        y_wld: Ground truth [W, L, D] (normalized)
        criterion_reg: Regression loss function (MSELoss)
    
    Returns:
        w_loss, l_loss, d_loss: Individual losses
        avg_reg_loss: Average of 3 regression losses
    """
    w_loss = criterion_reg(y_pred_wld[:, 0], y_wld[:, 0])
    l_loss = criterion_reg(y_pred_wld[:, 1], y_wld[:, 1])
    d_loss = criterion_reg(y_pred_wld[:, 2], y_wld[:, 2])
    
    avg_reg_loss = (w_loss + l_loss + d_loss) / 3.0
    
    return w_loss, l_loss, d_loss, avg_reg_loss


def compute_combined_loss(clf_loss, w_loss, l_loss, d_loss, physics_loss=None,
                         log_var_clf=None, log_var_w=None, log_var_l=None, log_var_d=None,
                         alpha=1.0, normalizer=None):
    """
    Compute combined multi-task loss with Homoscedastic Uncertainty Weighting (Kendall et al., 2018)
    across 4 supervised data losses (clf, W, L, D) and standalone physics loss weighted by alpha.
    """
    if normalizer is not None and normalizer.method != 'none':
        norm_clf = normalizer.update_and_normalize("clf", clf_loss)
        norm_w = normalizer.update_and_normalize("w", w_loss)
        norm_l = normalizer.update_and_normalize("l", l_loss)
        norm_d = normalizer.update_and_normalize("d", d_loss)
    else:
        norm_clf, norm_w, norm_l, norm_d = clf_loss, w_loss, l_loss, d_loss

    if all(v is not None for v in (log_var_clf, log_var_w, log_var_l, log_var_d)):
        precision_clf = torch.exp(-log_var_clf)
        precision_w = torch.exp(-log_var_w)
        precision_l = torch.exp(-log_var_l)
        precision_d = torch.exp(-log_var_d)
        data_loss = (
            precision_clf * norm_clf + 0.5 * log_var_clf +
            precision_w * norm_w + 0.5 * log_var_w +
            precision_l * norm_l + 0.5 * log_var_l +
            precision_d * norm_d + 0.5 * log_var_d
        )
    else:
        data_loss = norm_clf + (norm_w + norm_l + norm_d) / 3.0

    if physics_loss is not None:
        norm_phys = normalizer.update_and_normalize("pinn", physics_loss) if (normalizer is not None and normalizer.method != 'none') else physics_loss
        total_loss = data_loss + alpha * norm_phys
    else:
        total_loss = data_loss

    avg_reg_loss = (w_loss + l_loss + d_loss) / 3.0
    return total_loss, avg_reg_loss





def visualize_physics_comparison(X_field, y_wld, y_shape, unique_shapes, shape_map, y_scaler, 
                                  output_dir, num_samples=5, prefix='', verbose=True):
    """
    Save side-by-side physics comparisons: H_true vs H_pred and error maps.
    Uses the same physical reconstruction steps as compute_physics_loss_autograd.
    
    Args:
        X_field: Original field data, shape (N, 32, 32, 2), not normalized
        y_wld: Predicted W/L/D values, shape (N, 3), normalized
        y_shape: Shape indices (N,)
        unique_shapes: List of shape names
        shape_map: Dict mapping shape name -> physics module
        y_scaler: Target scaler used to denormalize W/L/D
        output_dir: Output directory
        num_samples: Number of samples to visualize
        prefix: Filename prefix
    """
    import math
    
    # Physics constants
    freq, sicma, mu = 5000, 35461000, 0.0000012566
    csi, K, I, G = 0.085, 1.5, 0.01, 3981
    delta = 1.0 / math.sqrt(math.pi * freq * mu * sicma) * 1000.0
    z_lift = 1.0
    N, Res = 16, 0.78
    
    # Select samples per shape for balanced visualization.
    samples_per_shape = max(1, num_samples // len(unique_shapes))
    sample_indices = []
    
    for shape_idx in range(len(unique_shapes)):
        shape_mask = np.where(y_shape == shape_idx)[0]
        if len(shape_mask) > 0:
            selected = np.random.choice(shape_mask, min(samples_per_shape, len(shape_mask)), replace=False)
            sample_indices.extend(selected)
    
    sample_indices = sample_indices[:num_samples]
    
    for idx, sample_id in enumerate(sample_indices):
        try:
            # Get field data (channel 0)
            H_true = X_field[sample_id]
            if H_true.ndim == 3:
                H_true = H_true[:, :, 0]
            
            # Get W, L, D (normalized)
            w_norm, l_norm, d_norm = y_wld[sample_id]
            
            # Denormalize W/L/D with max-based scaling.
            w_max, l_max, d_max = y_scaler.data_max_
            w_val = w_norm * w_max
            l_val = l_norm * l_max
            d_val = d_norm * d_max
            
            # Clamp
            w_val = np.clip(w_val, 0.3, 1.5)
            l_val = np.clip(l_val, 2.0, 20.0)
            d_val = np.clip(d_val, 0.1, 3.0)
            
            # Get shape name
            shape_name = unique_shapes[y_shape[sample_id]]
            
            if shape_name not in shape_map:
                continue
            
            physics_module = shape_map[shape_name]
            
            # ===== COMPUTE H_pred (SAME AS compute_physics_loss) =====
            with torch.no_grad():
                if shape_name == 'Ellipse':
                    H_tensor = physics_module.compute_map_gpu(w_val, l_val, d_val, delta, z_lift)
                    H_raw = H_tensor.detach().cpu().numpy()
                    H_raw = (csi / 4 / np.pi) * H_raw * K * I * G
                    H_pred = np.rot90(H_raw, 2)
                    H_pred = np.diff(H_pred, axis=1)
                    H_pred = np.column_stack([H_pred, H_pred[:, -1:]])
                    
                elif shape_name == 'Rectangular':
                    H_tensor, _, _ = physics_module.compute_magnetic_field(
                        w_val, l_val, d_val, delta, z_lift, N, Res, angle=0, K=K, I=I, G=G, csi=csi
                    )
                    H_pred = np.rot90(H_tensor, 2, axes=(0, 1))
                    H_pred = np.diff(H_pred, axis=1)
                    H_pred = np.column_stack([H_pred, H_pred[:, -1:]])
                    
                elif shape_name == 'Triangular':
                    H_tensor = physics_module.compute_triangular_map_gpu(w_val, l_val, d_val, delta, z_lift)
                    H_raw = H_tensor.detach().cpu().numpy().squeeze()
                    H_raw = (csi / 4 / math.pi) * H_raw * K * I * G
                    H_pred = np.rot90(H_raw, 2)
                    H_pred = np.diff(H_pred, axis=1)
                    H_pred = np.column_stack([H_pred, H_pred[:, -1:]])
                    
                elif shape_name == 'Step_R':
                    xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
                    ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
                    X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
                    H_val = physics_module.calculate_field_Step_R(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
                    H_raw = (H_val * (csi / 4 / math.pi) * K * I * G).cpu().numpy()
                    H_pred = np.rot90(H_raw, 2)
                    H_pred = np.diff(H_pred, axis=1)
                    draft_1 = np.zeros((32, 31))
                    for i in range(32):
                        draft_1[i] = H_pred[i][::-1]
                    H_pred = np.column_stack([draft_1, draft_1[:, -1:]])
                    
                elif shape_name == 'Step_T':
                    xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
                    ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
                    X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
                    H_val = physics_module.calculate_field_Step_T(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
                    H_raw = (H_val * (csi / 4 / math.pi) * K * I * G).cpu().numpy()
                    H_pred = np.rot90(H_raw, 2)
                    H_pred = np.diff(H_pred, axis=1)
                    draft_1 = np.zeros((32, 31))
                    for i in range(32):
                        draft_1[i] = H_pred[i][::-1]
                    H_pred = np.column_stack([draft_1, draft_1[:, -1:]])
                else:
                    continue
            
            # ===== COMPUTE ERROR =====
            diff = H_true - H_pred
            mse = np.mean(diff ** 2)

            # Shared ranges for consistent visualization across true/pred and diff.
            vmin = min(np.min(H_true), np.min(H_pred))
            vmax = max(np.max(H_true), np.max(H_pred))
            vmax_diff = max(abs(np.min(diff)), abs(np.max(diff)))
            if vmax_diff == 0:
                vmax_diff = 1e-8

            # Save standalone images requested at each checkpoint:
            # 1) Ground truth, 2) Prediction, 3) Difference (H_true - H_pred)
            standalone_maps = [
                (H_true, 'RdBu_r', vmin, vmax,
                 f"H_true (Ground Truth) | {shape_name}",
                 f"{prefix}physics_true_{shape_name}_{idx+1}.png"),
                (H_pred, 'RdBu_r', vmin, vmax,
                 f"H_pred (Physics Prediction) | {shape_name}",
                 f"{prefix}physics_pred_{shape_name}_{idx+1}.png"),
                (diff, 'RdBu_r', -vmax_diff, vmax_diff,
                 f"Difference (H_true - H_pred) | {shape_name}",
                 f"{prefix}physics_diff_{shape_name}_{idx+1}.png"),
            ]

            for map_data, cmap_name, map_vmin, map_vmax, title_text, out_name in standalone_maps:
                fig_single, ax_single = plt.subplots(figsize=(6, 5))
                im_single = ax_single.imshow(map_data, cmap=cmap_name, aspect='auto', vmin=map_vmin, vmax=map_vmax)
                ax_single.set_title(title_text, fontsize=11, fontweight='bold')
                ax_single.set_xlabel('X')
                ax_single.set_ylabel('Y')
                plt.colorbar(im_single, ax=ax_single)
                fig_single.tight_layout()
                fig_single.savefig(os.path.join(output_dir, out_name), dpi=150)
                plt.close(fig_single)
            
            # ===== PLOT 2x2: H_true | H_pred | Diff | Error Map =====
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Shared colorbar range for H_true and H_pred
            
            # 1. H_true (Ground Truth Field)
            im1 = axes[0, 0].imshow(H_true, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f'H_true (Ground Truth Field)\n{shape_name}', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # 2. H_pred (Physics Model Prediction)
            im2 = axes[0, 1].imshow(H_pred, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f'H_pred (Physics Model)\nW={w_val:.3f}, L={l_val:.3f}, D={d_val:.3f}', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # 3. Difference (H_true - H_pred)
            im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax_diff, vmax=vmax_diff)
            axes[1, 0].set_title(f'Difference (H_true - H_pred)\nMean: {np.mean(diff):.6f}', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # 4. Error Map (Squared Error)
            error_map = (H_true - H_pred) ** 2
            im4 = axes[1, 1].imshow(error_map, cmap='hot', aspect='auto')
            axes[1, 1].set_title(f'Squared Error Map\nMSE (Physics Loss) = {mse:.8f}', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.suptitle(f'Physics Loss Visualization - Sample {idx+1}\n{shape_name}: W={w_val:.3f}, L={l_val:.3f}, D={d_val:.3f}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f"{prefix}physics_comparison_{shape_name}_{idx+1}.png"
            _atomic_save_fig(plt, os.path.join(output_dir, filename), dpi=150)
            plt.close()
            
            if verbose:
                print(f"  Saved: {filename} | MSE={mse:.8f}")
            
        except Exception as e:
            if verbose:
                print(f"  Error visualizing sample {idx}: {str(e)[:50]}")
            continue
    
    print(f"Physics comparison plots saved to {output_dir}")


# ============================================================================
# LEGACY ATTENTION MODULES (CURRENTLY UNUSED)
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ImprovedMultimodelNet(nn.Module):
    """
    Multimodel network for shape classification and joint W/L/D regression
    with Kendall et al. (2018) Homoscedastic Uncertainty Weighting.
    """
    def __init__(self, num_shapes):
        super(ImprovedMultimodelNet, self).__init__()
        self.num_shapes = num_shapes
        
        # Learnable uncertainty parameters
        self.log_var_clf = nn.Parameter(torch.tensor(0.0))
        self.log_var_w = nn.Parameter(torch.tensor(0.0))
        self.log_var_l = nn.Parameter(torch.tensor(0.0))
        self.log_var_d = nn.Parameter(torch.tensor(0.0))
        
        # ===== SHARED BACKBONE =====
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
        
        # ===== CLASSIFICATION HEAD =====
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
        
        # ===== REGRESSION BACKBONE =====
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
        
        # ===== JOINT REGRESSION HEAD (W, L, D) =====
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        """Forward pass"""
        backbone_feat = self.backbone(x)
        backbone_feat = backbone_feat.reshape(backbone_feat.size(0), -1)

        # Classification branch
        shape_logits = self.classifier(backbone_feat)

        # Regression branch
        reg_feat = self.regressor_backbone(backbone_feat)
        y_pred_wld = self.reg_head(reg_feat)

        return shape_logits, y_pred_wld

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("LOADING DATA")

try:
    X, y, matched_filenames = Load_Data_With_Labels(DATA_PATH, LABELS_PATH)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print(f"Loaded: {len(X)} samples")

if len(X) != len(y) or len(X) != len(matched_filenames):
    raise ValueError("Data-Label mismatch!")
print(f"Alignment verified - {len(matched_filenames)} files matched")

print(f"\nEXTRACTING SHAPE LABELS")

labels_df = pd.read_csv(LABELS_PATH)
filename_to_shape = {}
for idx, row in labels_df.iterrows():
    try:
        filename = str(row['filename']).strip()
        shape = str(row['shape']).strip()
        if 'type' in labels_df.columns and shape == 'Step':
            type_val = str(row['type']).strip()
            if type_val and type_val.lower() != 'nan':
                shape = f'Step_{type_val}'
        if shape and shape.lower() != 'nan':
            filename_to_shape[filename] = shape
    except:
        continue

y_shape_str = []
for filename in matched_filenames:
    if filename in filename_to_shape:
        y_shape_str.append(filename_to_shape[filename])
    else:
        y_shape_str.append(filename.split('_')[0])

unique_shapes = sorted(list(set(y_shape_str)))
shape_to_idx = {shape: i for i, shape in enumerate(unique_shapes)}
y_shape = np.array([shape_to_idx[shape] for shape in y_shape_str])

print(f"Shapes: {unique_shapes}")
for shape in unique_shapes:
    count = sum(np.array(y_shape_str) == shape)
    print(f"  {shape}: {count} samples")

# ============================================================================
# 2. DATA SPLIT
# ============================================================================

print(f"\nSPLITTING DATA (70% train, 20% val, 10% test)")

X_temp, X_test, y_temp, y_test, y_shape_temp, y_shape_test = train_test_split(
    X, y, y_shape, test_size=0.10, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val, y_shape_train, y_shape_val = train_test_split(
    X_temp, y_temp, y_shape_temp, test_size=0.222, random_state=RANDOM_STATE
)

# Subsample TRAIN by class-wise percentage.
total_samples = len(X)
available_train_samples = len(X_train)

class_subset_info = []
subset_parts = []
for class_idx in sorted(np.unique(y_shape_train)):
    class_mask_idx = np.where(y_shape_train == class_idx)[0]
    class_count = len(class_mask_idx)
    class_take = int(np.floor(class_count * (TRAIN_PERCENT / 100.0)))
    class_take = max(1, class_take)
    class_take = min(class_take, class_count)

    # Fixed per-class permutation so percent subsets are nested across runs.
    class_seed = RANDOM_STATE + int(class_idx) * 10007
    class_rng = np.random.default_rng(class_seed)
    class_order = class_rng.permutation(class_mask_idx)
    chosen = class_order[:class_take]

    subset_parts.append(np.sort(chosen))
    class_subset_info.append((int(class_idx), int(class_count), int(class_take)))

subset_indices = np.sort(np.concatenate(subset_parts))
X_train = X_train[subset_indices]
y_train = y_train[subset_indices]
y_shape_train = y_shape_train[subset_indices]
final_train_samples = len(X_train)

print(
    f"Train subset selected: {final_train_samples} samples "
    f"(TRAIN_PERCENT={TRAIN_PERCENT}%, available_before_subset={available_train_samples})"
)
for class_idx, class_count, class_take in class_subset_info:
    class_name = unique_shapes[class_idx] if class_idx < len(unique_shapes) else str(class_idx)
    class_ratio = (class_take / class_count) if class_count > 0 else 0.0
    print(f"  {class_name}: kept {class_take}/{class_count} ({class_ratio:.2%})")

n_train, h, w, ch = X_train.shape
n_val, _, _, _ = X_val.shape
n_test, _, _, _ = X_test.shape

print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
print(f"Data shape: (N, H, W, C) = ({n_train}, {h}, {w}, {ch})")
print(
    f"Sample counts (full/train/val/test): "
    f"{total_samples}/{n_train}/{n_val}/{n_test}"
)

split_info_path = os.path.join(OUTPUT_DIR, "dataset_split_info.txt")
split_txt = "DATASET SPLIT INFO\n"
split_txt += "=" * 80 + "\n"
split_txt += f"run_tag: {RUN_TAG}\n"
split_txt += f"train_percent_requested: {TRAIN_PERCENT}\n"
split_txt += "evaluation_isolation_mode: ON (no cross-percent resume)\n"
split_txt += f"resume_checkpoint_used: {RESUME_CHECKPOINT if RESUME_CHECKPOINT is not None else 'None'}\n"
split_txt += f"total_samples: {total_samples}\n"
split_txt += "train_subset_mode: percent_subset\n"
split_txt += f"available_train_before_subset: {available_train_samples}\n"
split_txt += f"final_train_samples: {n_train}\n"
split_txt += f"val_samples: {n_val}\n"
split_txt += f"test_samples: {n_test}\n"
split_txt += f"train_ratio_of_full: {n_train / total_samples:.6f}\n"
split_txt += f"val_ratio_of_full: {n_val / total_samples:.6f}\n"
split_txt += f"test_ratio_of_full: {n_test / total_samples:.6f}\n"
_atomic_write_text(split_txt, split_info_path)

print(f"[OK] Saved split info: {split_info_path}")

# Ensure normalization is always fit on the post-percent train subset.
assert len(X_train) == final_train_samples, "Train subset size mismatch before normalization"
assert len(X_train) == len(y_train) == len(y_shape_train), "Train arrays misaligned before normalization"
print("[ORDER] Confirmed: split -> train percent subset -> normalization fit on subset")

# Keep raw fields for physics visualization/debug before normalization.
X_train_field = X_train.copy()  # (n_train, 32, 32) - after preprocessing, before normalize
X_val_field = X_val.copy()      # (n_val, 32, 32)
X_test_field = X_test.copy()    # (n_test, 32, 32)

print(f"\nNORMALIZING DATA")

# ========== X NORMALIZATION (Field Images) ==========
# Formula: X_norm = (X - mean_train) / std_train
# Using StandardScaler: fit mean/std on TRAIN, apply to TRAIN/VAL/TEST
# StandardScaler normalizes each pixel channel independently
X_scaler = StandardScaler()  # Scaler for images

X_train_flat_norm = X_scaler.fit_transform(X_train.reshape(n_train * h * w, ch))  # FIT on train
X_val_flat_norm = X_scaler.transform(X_val.reshape(n_val * h * w, ch))              # Apply fitted scaler
X_test_flat_norm = X_scaler.transform(X_test.reshape(n_test * h * w, ch))           # Apply fitted scaler

X_train_norm = X_train_flat_norm.reshape(n_train, h, w, ch)  # Back to (N, H, W, C)
X_val_norm = X_val_flat_norm.reshape(n_val, h, w, ch)
X_test_norm = X_test_flat_norm.reshape(n_test, h, w, ch)

_atomic_write_pickle(X_scaler, os.path.join(OUTPUT_DIR, 'X_scaler.pkl'))
_atomic_write_pickle(X_scaler, os.path.join(OUTPUT_DIR, 'X_val_scaler.pkl'))
_atomic_write_pickle(X_scaler, os.path.join(OUTPUT_DIR, 'X_test_scaler.pkl'))

print("X normalization complete (StandardScaler: mean/std from train, applied to all)")
print(f"  Formula: X_norm = (X - mean_train) / std_train")

print(f"\nNORMALIZING REGRESSION TARGETS (Y) - Divide by Max (SEPARATE for W, L, D)")
print(f"  **Different from X!** y uses DIVIDE BY MAX, not StandardScaler")

# ========== y NORMALIZATION (W, L, D Regression Labels) ==========
# Formula: y_norm = y / max_train  (for each of W, L, D separately)
# Calculate max values from TRAIN set only
w_max = float(np.max(y_train[:, 0]))  # max_W from train
l_max = float(np.max(y_train[:, 1]))  # max_L from train
d_max = float(np.max(y_train[:, 2]))  # max_D from train

# TRAIN: Normalize each target by dividing by max from train set
y_train_w_norm = y_train[:, 0] / w_max  # W_norm = W / max_W
y_train_l_norm = y_train[:, 1] / l_max  # L_norm = L / max_L
y_train_d_norm = y_train[:, 2] / d_max  # D_norm = D / max_D
y_train_norm = np.hstack([y_train_w_norm.reshape(-1, 1), y_train_l_norm.reshape(-1, 1), y_train_d_norm.reshape(-1, 1)])

# VAL: Apply same normalization using TRAIN's max
y_val_w_norm = y_val[:, 0] / w_max     # W_norm = W / max_W (from train)
y_val_l_norm = y_val[:, 1] / l_max     # L_norm = L / max_L (from train)
y_val_d_norm = y_val[:, 2] / d_max     # D_norm = D / max_D (from train)
y_val_norm = np.hstack([y_val_w_norm.reshape(-1, 1), y_val_l_norm.reshape(-1, 1), y_val_d_norm.reshape(-1, 1)])

# TEST: Apply same normalization using TRAIN's max
y_test_w_norm = y_test[:, 0] / w_max   # W_norm = W / max_W (from train)
y_test_l_norm = y_test[:, 1] / l_max   # L_norm = L / max_L (from train)
y_test_d_norm = y_test[:, 2] / d_max   # D_norm = D / max_D (from train)
y_test_norm = np.hstack([y_test_w_norm.reshape(-1, 1), y_test_l_norm.reshape(-1, 1), y_test_d_norm.reshape(-1, 1)])

# Create simple scaler objects that store max values (for compatibility with inverse transform)
class MaxScaler:
    """Simple scaler that normalizes by dividing by max value"""
    def __init__(self, max_val):
        self.max_val = float(max_val)  # Store as scalar
        self.data_min_ = 0.0  # Scalar (min assumed to be 0)
        self.data_max_ = float(max_val)  # Scalar (max for normalization)
    
    def transform(self, X):
        """Normalize by dividing by max - handles both 1D and 2D arrays"""
        return np.asarray(X) / self.max_val
    
    def inverse_transform(self, X_norm):
        """Denormalize by multiplying by max - handles both 1D and 2D arrays"""
        return np.asarray(X_norm) * self.max_val


class SeparateMaxScaler:
    """Scaler using 3 separate MaxScaler objects for W, L, D columns."""
    def __init__(self, w_scaler, l_scaler, d_scaler):
        self.w_scaler = w_scaler
        self.l_scaler = l_scaler
        self.d_scaler = d_scaler
        # Compatibility attributes (as arrays for physics loss access)
        self.data_max_ = np.array([w_scaler.max_val, l_scaler.max_val, d_scaler.max_val])
        self.data_min_ = np.array([0.0, 0.0, 0.0])

    def transform(self, X):
        """Normalize W, L, D columns separately - handles both 1D and 2D arrays"""
        X = np.asarray(X).astype(np.float64)
        if X.ndim == 1:
            # Single sample [W, L, D]
            return np.array([self.w_scaler.transform(X[0]), 
                            self.l_scaler.transform(X[1]), 
                            self.d_scaler.transform(X[2])])
        else:
            # Batch of samples (N, 3)
            X_norm = X.copy()
            X_norm[:, 0] = self.w_scaler.transform(X[:, 0])
            X_norm[:, 1] = self.l_scaler.transform(X[:, 1])
            X_norm[:, 2] = self.d_scaler.transform(X[:, 2])
            return X_norm

    def inverse_transform(self, X_norm):
        """Denormalize W, L, D columns separately - handles both 1D and 2D arrays"""
        X_norm = np.asarray(X_norm).astype(np.float64)
        if X_norm.ndim == 1:
            # Single sample [W_norm, L_norm, D_norm]
            return np.array([self.w_scaler.inverse_transform(X_norm[0]), 
                            self.l_scaler.inverse_transform(X_norm[1]), 
                            self.d_scaler.inverse_transform(X_norm[2])])
        else:
            # Batch of samples (N, 3)
            X = X_norm.copy()
            X[:, 0] = self.w_scaler.inverse_transform(X_norm[:, 0])
            X[:, 1] = self.l_scaler.inverse_transform(X_norm[:, 1])
            X[:, 2] = self.d_scaler.inverse_transform(X_norm[:, 2])
            return X

# Separate scalers for W, L, D
w_scaler = MaxScaler(w_max)
l_scaler = MaxScaler(l_max)
d_scaler = MaxScaler(d_max)
y_scaler = SeparateMaxScaler(w_scaler, l_scaler, d_scaler)
y_test_scaler = y_scaler

# Save max-value scalers (single copy for all sets - same max values from train fit)
_atomic_write_pickle(w_scaler, os.path.join(OUTPUT_DIR, 'w_scaler.pkl'))
_atomic_write_pickle(l_scaler, os.path.join(OUTPUT_DIR, 'l_scaler.pkl'))
_atomic_write_pickle(d_scaler, os.path.join(OUTPUT_DIR, 'd_scaler.pkl'))
_atomic_write_pickle(y_scaler, os.path.join(OUTPUT_DIR, 'y_scaler.pkl'))

print("Y normalization complete (normalize by dividing by max from train set)")
print(f"Max values (from train): W_max={w_max:.4f}, L_max={l_max:.4f}, D_max={d_max:.4f}")
print(f"W range after norm: [{y_train_w_norm.min():.4f}, {y_train_w_norm.max():.4f}]")
print(f"L range after norm: [{y_train_l_norm.min():.4f}, {y_train_l_norm.max():.4f}]")
print(f"D range after norm: [{y_train_d_norm.min():.4f}, {y_train_d_norm.max():.4f}]")

# VERIFICATION: Check val/test normalization against train max values
print(f"\n[VERIFICATION] Checking val/test normalized ranges:")
print(f"Val W range: [{y_val_w_norm.min():.4f}, {y_val_w_norm.max():.4f}] (train fitted on max={w_max:.4f})")
print(f"Val L range: [{y_val_l_norm.min():.4f}, {y_val_l_norm.max():.4f}] (train fitted on max={l_max:.4f})")
print(f"Val D range: [{y_val_d_norm.min():.4f}, {y_val_d_norm.max():.4f}] (train fitted on max={d_max:.4f})")
print(f"Test W range: [{y_test_w_norm.min():.4f}, {y_test_w_norm.max():.4f}]")
print(f"Test L range: [{y_test_l_norm.min():.4f}, {y_test_l_norm.max():.4f}]")
print(f"Test D range: [{y_test_d_norm.min():.4f}, {y_test_d_norm.max():.4f}]")

# Count values > 1.0 (outside train range)
val_over_1_w = np.sum(y_val_w_norm > 1.0)
val_over_1_l = np.sum(y_val_l_norm > 1.0)
val_over_1_d = np.sum(y_val_d_norm > 1.0)
test_over_1_w = np.sum(y_test_w_norm > 1.0)
test_over_1_l = np.sum(y_test_l_norm > 1.0)
test_over_1_d = np.sum(y_test_d_norm > 1.0)

if val_over_1_w > 0 or val_over_1_l > 0 or val_over_1_d > 0:
    print(f"[WARNING] Val has {val_over_1_w} W, {val_over_1_l} L, {val_over_1_d} D values > 1.0 (outside train range)")
if test_over_1_w > 0 or test_over_1_l > 0 or test_over_1_d > 0:
    print(f"[WARNING] Test has {test_over_1_w} W, {test_over_1_l} L, {test_over_1_d} D values > 1.0 (outside train range)")

# Verify denormalize works correctly
y_val_denorm_check = y_scaler.inverse_transform(y_val_norm)
y_val_error = np.abs(y_val_denorm_check - y_val).max()
assert y_val_error < 1e-5, f"[FATAL] Denormalization error too large: {y_val_error:.2e} (should be < 1e-5)"
print(f"[OK] Denormalize verification: max error = {y_val_error:.2e}")
print(f"[OK] y_scaler is ready for training and inference")

print(f"\nPREPARING DATA FOR TRAINING")

# ====== X: FIELD IMAGES (Normalized) ======
# Data already has 2 channels (field + gradient), use as-is
X_train_final = X_train_norm  # (n_train, 32, 32, 2) - field images
X_val_final = X_val_norm      # (n_val, 32, 32, 2)   - field images
X_test_final = X_test_norm    # (n_test, 32, 32, 2)  - field images

# Convert X to PyTorch tensors and rearrange to (N, C, H, W) format
# From (N, H, W, C) to (N, C, H, W)
X_train_tensor = torch.FloatTensor(X_train_final).permute(0, 3, 1, 2)  # X: field image
X_val_tensor = torch.FloatTensor(X_val_final).permute(0, 3, 1, 2)      # X: field image
X_test_tensor = torch.FloatTensor(X_test_final).permute(0, 3, 1, 2)    # X: field image

# ====== y: NORMALIZED REGRESSION LABELS (W, L, D) ======
# y_train_norm shape: (n_train, 3) = [W_norm, L_norm, D_norm]
# y_val_norm shape:   (n_val, 3)   = [W_norm, L_norm, D_norm]
# y_test_norm shape:  (n_test, 3)  = [W_norm, L_norm, D_norm]
y_train_tensor = torch.FloatTensor(y_train_norm)  # y: [W_norm, L_norm, D_norm] labels
y_val_tensor = torch.FloatTensor(y_val_norm)      # y: [W_norm, L_norm, D_norm] labels
y_test_tensor = torch.FloatTensor(y_test_norm)    # y: [W_norm, L_norm, D_norm] labels

y_shape_train_tensor = torch.LongTensor(y_shape_train)
y_shape_val_tensor = torch.LongTensor(y_shape_val)
y_shape_test_tensor = torch.LongTensor(y_shape_test)

# ============================================================================
# 5. CREATE DATALOADERS
# ============================================================================

# TensorDataset: (X: field image, y: [W_norm, L_norm, D_norm], y_shape: class label)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, y_shape_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor, y_shape_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, y_shape_test_tensor)

# Windows note: keep num_workers=0 to avoid multiprocessing issues.
NUM_WORKERS_DEFAULT = "0" if os.name == 'nt' else "4"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", NUM_WORKERS_DEFAULT))
# Use adaptive batch size to avoid empty train_loader on very small TRAIN_PERCENT.
effective_batch_size = min(BATCH_SIZE, max(1, len(train_dataset)))
train_drop_last = (len(train_dataset) % effective_batch_size == 1 and len(train_dataset) > 1)

persistent_workers = (NUM_WORKERS > 0)
train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=train_drop_last,
                         persistent_workers=persistent_workers)
# PINN mean-loss pass should cover the full train subset deterministically.
pinn_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                        persistent_workers=persistent_workers)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=persistent_workers)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=persistent_workers)

print(f"DataLoaders created (workers={NUM_WORKERS}, pin_memory=True, batch_size={effective_batch_size}, drop_last={train_drop_last})")
print(f"[CONFIG] PINN loader: shuffle=False, drop_last=False (full-epoch mean PINN loss)")
if train_drop_last:
    print(f"[INFO] drop_last=True: dropping 1 sample in final training batch to avoid BatchNorm size-1 error")

# ============================================================================
# 6. BUILD MODEL
# ============================================================================

print(f"\nBUILDING IMPROVED MULTIMODEL")

num_shapes = len(unique_shapes)
model = ImprovedMultimodelNet(num_shapes).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ============================================================================
# 7. LOSS FUNCTIONS
# ============================================================================

print(f"\nCONFIGURING LOSS FUNCTIONS")

# Use standard CE for classification and MSE for regression.
criterion_clf = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
)

# ============================================================================
# CHECKPOINT SAVING FUNCTION (FULL STATE)
# ============================================================================

def save_full_checkpoint(filepath, model, optimizer, scheduler, epoch, history, best_val_loss):
    """
    Save full training checkpoint for resuming training later.
    
    Args:
        filepath: Path to save checkpoint
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch number
        history: Training history dict
        best_val_loss: Best validation loss so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'random_state': RANDOM_STATE,
            'train_percent': TRAIN_PERCENT,
            'alpha_init': ALPHA_INIT,
            'pinn_activation_epoch': PINN_ACTIVATION_EPOCH,
            'pinn_base_grad_clip_norm': PINN_BASE_GRAD_CLIP_NORM,
            'pinn_loss_type': PINN_LOSS_TYPE,
            'series_label': SERIES_LABEL,
            'run_tag': RUN_TAG,
            'output_dir': OUTPUT_DIR,
        }
    }
    torch.save(checkpoint, filepath)


def _torch_load_checkpoint(filepath):
    try:
        return torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location=device)


def _safe_load_state_dict(model, state_dict):
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    if isinstance(state_dict, dict) and any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).replace("module.", "", 1): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)


def load_checkpoint(filepath, model, optimizer, scheduler):
    """
    Load checkpoint and restore training state.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer
        scheduler: LR scheduler
        
    Returns:
        start_epoch: Epoch to resume from
        history: Training history
        best_val_loss: Best validation loss
    """
    print(f"\n[RESUME] Loading checkpoint from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = _torch_load_checkpoint(filepath)
    
    # Load model weights
    _safe_load_state_dict(model, checkpoint['model_state_dict'])
    print(f"  [OK] Model weights loaded")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"  [OK] Optimizer state loaded")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"  [OK] Scheduler state loaded")
    
    # Get training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    history = checkpoint['history']
    best_val_loss = checkpoint['best_val_loss']
    
    # Print checkpoint info
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        print(f"  [INFO] Checkpoint config: batch_size={cfg.get('batch_size')}, lr={cfg.get('learning_rate')}")
    
    print(f"  [OK] Resuming from epoch {start_epoch} (trained {checkpoint['epoch']} epochs)")
    print(f"  [OK] Best val loss so far: {best_val_loss:.6f}")
    print(f"  [OK] History length: {len(history.get('train_loss', []))} epochs recorded")
    
    return start_epoch, history, best_val_loss

print(f"\nLoss configuration:")
print(f"  Classification loss: CrossEntropyLoss")
print(f"  Regression losses: MSE (W, L, D)")
print(f"  Physics loss: Custom PINN constraint")
print(f"  Weighting method: fixed warm-up then PINN activation")
print(f"    └─ Loss formula (active): L = L_clf + ((L_w + L_l + L_d)/3) + alpha*L_pinn")
print(f"    └─ Warm-up (epochs 1-{PINN_ACTIVATION_EPOCH}): PINN inactive")
print(f"    └─ PINN phase (epoch {PINN_ACTIVATION_EPOCH + 1}+): PINN active when physics modules are available")
print(f"    └─ alpha init: {ALPHA_INIT:.6f}")
print("    └─ Optimizer update: data per-batch; PINN mean-loss step once per epoch")

# ============================================================================
# 8. LOAD PHYSICS MODULES
# ============================================================================

physics_available_for_training = False
SHAPE_MAP_TRAINING = {}
try:
    sys.path.insert(0, PINN_PROJECT_PATH)
    import ellipse as ellipse_module
    import regtangular as rectangular_module
    import triangular as triangular_module
    import step_r as step_r_module
    import step_t as step_t_module
    
    SHAPE_MAP_TRAINING = {
        'Ellipse': ellipse_module,
        'Rectangular': rectangular_module,
        'Triangular': triangular_module,
        'Step_R': step_r_module,
        'Step_T': step_t_module,
    }
    physics_available_for_training = True
    print(f"\n[OK] Physics modules loaded from local source files (PINN enabled)")
except ImportError as e:
    print(f"\n[WARN] Physics modules not available: {e}")

# ============================================================================
# CHECKPOINT PLOTTING FUNCTION
# ============================================================================

def save_checkpoint_plots(history, epoch, checkpoint_dir, unique_shapes=None,
                          X_test_field=None, y_test=None, y_pred_wld=None, y_shape_test=None, y_pred_shape=None, y_scaler=None):
    """Save checkpoint figures (loss curves, metrics, and confusion matrix)."""
    epoch_str = f"epoch_{epoch+1:04d}"
    
    # Plot 1: Total Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history['val_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Total Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_total.png"), dpi=150)
    plt.close()
    
    # Plot 2: Classification Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_clf_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history['val_clf_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Classification Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_clf.png"), dpi=150)
    plt.close()
    
    # Plot 3: Width Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_w_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history['val_w_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Width Regression Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_width.png"), dpi=150)
    plt.close()
    
    # Plot 4: Length Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_l_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history['val_l_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Length Regression Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_length.png"), dpi=150)
    plt.close()
    
    # Plot 5: Depth Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_d_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history['val_d_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Depth Regression Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_depth.png"), dpi=150)
    plt.close()
    
    # Plot 6: Classification Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_clf_acc'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history['val_clf_acc'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Classification Accuracy (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_acc_clf.png"), dpi=150)
    plt.close()
    
    # Plot 7: Confusion Matrix (test set - kept for reference only)
    # Note: Test confusion matrix is now saved separately in checkpoint during training
    if y_pred_shape is not None and y_shape_test is not None and unique_shapes is not None:
        y_clf_pred_array = y_pred_shape
        y_clf_true_array = y_shape_test
        save_confusion_matrix(y_clf_true_array, y_clf_pred_array, unique_shapes, checkpoint_dir, 
                            filename=f"ckpt_{epoch_str}_confusion_matrix_test.png",
                            title=f"Test Confusion Matrix (Epoch {epoch+1})")
    
    # Plot 8: Physics Loss
    if max(history['physics_loss']) > 0:
        plt.figure(figsize=(10, 6))
        
        # Only plot epochs where physics loss is active (> 0)
        phys_losses = np.array(history['physics_loss'])
        active_idx = np.where(phys_losses > 0)[0]
        
        if len(active_idx) > 0:
            active_epochs = active_idx + 1
            active_losses = phys_losses[active_idx]
            
            plt.plot(active_epochs, active_losses, label='Physics Loss = log(1 + SSE/SSE_true)', linewidth=2.5, marker='o', markersize=3, color='red', alpha=0.8)
            plt.xlabel('Epoch', fontsize=12, fontweight='bold')
            plt.ylabel('log(1 + SSE/SSE_true)', fontsize=12, fontweight='bold')
            plt.title(f'Physics Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_physics.png"), dpi=150)
        plt.close()
        
    # -------------------------------------------------------------------------
    # Plot 10 (NEW): Combined 2x2 Summary Dashboard
    # -------------------------------------------------------------------------
    if y_test is not None and y_pred_wld is not None:
        # 1. Denormalize test predictions
        if y_scaler is not None:
            _y_pred_safe = np.where(np.isnan(y_pred_wld), np.nan_to_num(np.nanmean(y_pred_wld, axis=0), nan=0.0), y_pred_wld)
            _y_test_safe = np.where(np.isnan(y_test), np.nan_to_num(np.nanmean(y_test, axis=0), nan=0.0), y_test)
            y_test_denorm = y_scaler.inverse_transform(_y_test_safe)
            y_pred_denorm = y_scaler.inverse_transform(_y_pred_safe)
        else:
            y_test_denorm = np.nan_to_num(y_test, nan=0.0)
            y_pred_denorm = np.nan_to_num(y_pred_wld, nan=0.0)
            
        metrics_names = ['Width (W)', 'Length (L)', 'Depth (D)']
        grid_mae, grid_rmse, grid_r2, grid_max_err, grid_nrmse = [], [], [], [], []
        for i in range(3):
            grid_mae.append(mean_absolute_error(y_test_denorm[:, i], y_pred_denorm[:, i]))
            grid_rmse.append(np.sqrt(mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i])))
            grid_r2.append(r2_score(y_test_denorm[:, i], y_pred_denorm[:, i]))
            grid_max_err.append(np.max(np.abs(y_test_denorm[:, i] - y_pred_denorm[:, i])))
            grid_nrmse.append(calculate_nrmse(y_test_denorm[:, i], y_pred_denorm[:, i]))
            
        avg_nrmse = np.mean(grid_nrmse)
        avg_mae = np.mean(grid_mae)
        avg_rmse = np.mean(grid_rmse)
        avg_r2 = np.mean(grid_r2)
        
        clf_acc_text = "N/A"
        if y_pred_shape is not None and y_shape_test is not None:
            clf_acc = np.mean(y_pred_shape == y_shape_test)
            clf_acc_text = f"{clf_acc * 100.0:.2f}%"
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # [0,0] Total Loss
        axes[0, 0].plot(history.get('train_loss', []), label='Train', linewidth=2, color=TRAIN_LINE_COLOR)
        axes[0, 0].plot(history.get('val_loss', []), label='Val', linewidth=2, color=VAL_LINE_COLOR)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # [0,1] Classification Accuracy
        if len(history.get('train_clf_acc', [])) > 0:
            axes[0, 1].plot(history['train_clf_acc'], label='Train', linewidth=2, color=TRAIN_LINE_COLOR)
        if len(history.get('val_clf_acc', [])) > 0:
            axes[0, 1].plot(history['val_clf_acc'], label='Val', linewidth=2, color=VAL_LINE_COLOR)
        axes[0, 1].set_title('Classification Accuracy (Train vs Val)', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # [1,0] Physics Loss
        phys_losses = np.array(history.get('physics_loss', []))
        if len(phys_losses) > 0 and max(phys_losses) > 0:
            active_idx = np.where(phys_losses > 0)[0]
            if len(active_idx) > 0:
                axes[1, 0].plot(active_idx + 1, phys_losses[active_idx], label='log(1 + SSE/SSE_true)', linewidth=2, color='red')
            axes[1, 0].set_title('Physics Loss', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].axis('off')
            axes[1, 0].text(0.5, 0.5, 'Physics Loss\n(Not Active Yet / PINN Warm-up)', 
                            transform=axes[1, 0].transAxes, fontsize=14, 
                            ha='center', va='center', color='gray',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                            
        # [1,1] Test Set Metrics table (Q1 Standard)
        metrics_text = "Test Set Metrics (Q1 Standard):\n\n"
        metrics_text += f"Classification Accuracy: {clf_acc_text}\n"
        metrics_text += f"MAE Avg: {avg_mae:.4f} mm | RMSE Avg: {avg_rmse:.4f} mm\n"
        metrics_text += f"R² Avg: {avg_r2:.4f} | NRMSE Avg: {avg_nrmse:.2f}%\n\n"
        metrics_text += "Regression Metrics:\n"
        for i, name in enumerate(metrics_names):
            metrics_text += f"{name}:\n"
            metrics_text += f"  MAE: {grid_mae[i]:.4f} mm | RMSE: {grid_rmse[i]:.4f} mm\n"
            metrics_text += f"  R²:  {grid_r2[i]:.4f} | NRMSE: {grid_nrmse[i]:.2f}%\n"
            metrics_text += f"  MaxErr: {grid_max_err[i]:.4f} mm\n"
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=9.5, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_training_curves_summary.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 9: Field Data Visualizations (if test data provided)
    # DISABLED: Causes shape mismatch error when denormalizing multichannel fields
    # if X_test_field is not None and y_test is not None and y_pred_wld is not None and y_shape_test is not None:
    #     save_field_visualizations(X_test_field, y_test, y_pred_wld, y_shape_test, unique_shapes, 
    #                             checkpoint_dir, num_samples=5, prefix=f"ckpt_{epoch_str}_",
    #                             X_scaler=None, y_scaler=y_scaler)  # Pass y_scaler for denormalization

# ============================================================================
# 9. TRAINING LOOP
# ============================================================================

print(f'\nTRAINING WITH PINN WARM-UP')
print(f'  Warm-up (epochs 1-{PINN_ACTIVATION_EPOCH}): classification + regression (PINN inactive)')
print(f'  PINN phase (epoch {PINN_ACTIVATION_EPOCH + 1}+): data-loss per batch + one mean PINN update at end of epoch')
print(f'  Early stopping: after PINN activation, stop when PINN loss plateaus (patience=300)\n')

# Initialize training state
start_epoch = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'train_clf_acc': [],
    'val_clf_acc': [],
    'physics_loss': [],
    'checkpoint_score': [],
    'train_clf_loss': [],
    'val_clf_loss': [],
    'train_w_loss': [],
    'train_l_loss': [],
    'train_d_loss': [],
    'val_w_loss': [],
    'val_l_loss': [],
    'val_d_loss': [],
}
best_val_loss = float('inf')
best_checkpoint_score = best_val_loss

# ============================================================================
# LOAD CHECKPOINT IF SPECIFIED
# ============================================================================
if RESUME_CHECKPOINT is not None:
    try:
        start_epoch, history, best_val_loss = load_checkpoint(
            RESUME_CHECKPOINT, model, optimizer, scheduler
        )
        best_checkpoint_score = best_val_loss
        print(f"[RESUME] Training will continue from epoch {start_epoch + 1} to {EPOCHS}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        print(f"[INFO] Starting training from scratch...")
        start_epoch = 0
else:
    print(f"[INFO] Starting fresh training from epoch 1")

# Backward compatibility for checkpoints created before newer history keys existed.
for _history_key in [
    'train_loss',
    'val_loss',
    'train_clf_acc',
    'val_clf_acc',
    'physics_loss',
    'checkpoint_score',
    'train_clf_loss',
    'val_clf_loss',
    'train_w_loss',
    'train_l_loss',
    'train_d_loss',
    'val_w_loss',
    'val_l_loss',
    'val_d_loss',
]:
    history.setdefault(_history_key, [])
if history.get('checkpoint_score'):
    finite_checkpoint_scores = [
        float(x) for x in history['checkpoint_score']
        if x is not None and np.isfinite(float(x))
    ]
    if finite_checkpoint_scores:
        best_checkpoint_score = min(finite_checkpoint_scores)

# AMP (Automatic Mixed Precision) for faster training
use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)
print(f"[CONFIG] AMP (Mixed Precision): {'ENABLED' if use_amp else 'DISABLED'}")

import time
start_training = time.time()
DEBUG_TIMING = True
TIMING_EVERY = 10

# ============================================================================
# ADAPTIVE WEIGHT TRACKING
# ============================================================================
pinn_stop_patience = 300
pinn_stop_counter = 0
pinn_stop_best_loss = float('inf')
if start_epoch >= PINN_ACTIVATION_EPOCH and history.get('physics_loss'):
    prior_pinn_losses = [
        float(x)
        for x in history['physics_loss'][PINN_ACTIVATION_EPOCH:]
        if x is not None and np.isfinite(float(x)) and float(x) > 0
    ]
    if prior_pinn_losses:
        pinn_stop_best_loss = min(prior_pinn_losses)
min_pinn_epochs = 0
pinn_activated = start_epoch >= PINN_ACTIVATION_EPOCH
pinn_activated_epoch = (PINN_ACTIVATION_EPOCH + 1) if pinn_activated else None
if pinn_activated:
    print(f"[RESUME] PINN already active from epoch {pinn_activated_epoch}; keep checkpoint tracking from loaded history.")
nan_detected = False
last_valid_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
last_valid_optimizer_state = optimizer.state_dict()
last_valid_scheduler_state = scheduler.state_dict()
last_valid_epoch = start_epoch
last_valid_history = copy.deepcopy(history)

for epoch in range(start_epoch, EPOCHS):
    if nan_detected:
        break
        
    epoch_start_time = time.time()
    last_val_total_loss = history['val_loss'][-1] if history['val_loss'] else None

    # Set up physics loss log file
    physics_log_file = os.path.join(OUTPUT_DIR, f'physics_loss_epoch_{epoch+1:04d}.txt')
    
    # Warm-up before enabling PINN loss.
    is_before_pinn = (epoch < PINN_ACTIVATION_EPOCH)
    
    # Track when PINN is first activated
    if not is_before_pinn and not pinn_activated and (ALPHA_INIT > 0):
        pinn_activated = True
        pinn_activated_epoch = epoch + 1  # 1-based epoch number
        best_checkpoint_score = float('inf')
        print(f"\n>>> PINN ACTIVATED at Epoch {pinn_activated_epoch} <<<\n")
        print("[CKPT] Reset best checkpoint tracking. From now on best is selected in PINN phase.")

    
    # ===== TRAINING =====
    model.train()
    train_clf_loss = 0.0
    train_w_loss = 0.0
    train_l_loss = 0.0
    train_d_loss = 0.0
    train_physics_loss = 0.0
    train_clf_correct = 0
    train_clf_total = 0
    epoch_pinn_batches_applied = 0
    epoch_pinn_samples_applied = 0
    epoch_pinn_loss_sum = 0.0
    epoch_pinn_grad_norm = 0.0
    pinn_loss_active = physics_available_for_training and (not is_before_pinn) and (ALPHA_INIT > 0)
    
    # alpha is the active PINN weight in the current loss formula.
    alpha = ALPHA_INIT
    
    # Progress bar for training batches
    pbar = tqdm(
        train_loader,
        desc=f'Epoch {epoch+1}/{EPOCHS}',
        unit='batch',
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    processed_train_batches = 0
    for batch_idx, (batch_X, batch_y_wld, batch_y_shape) in enumerate(pbar):
        t_batch_start = time.perf_counter()
        if batch_X.size(0) < 2:
            # BatchNorm1d needs at least 2 samples in training mode.
            continue
        batch_X = batch_X.to(device)
        batch_y_wld = batch_y_wld.to(device)
        batch_y_shape = batch_y_shape.to(device)
        t_data = time.perf_counter()
        
        # Forward with AMP autocast
        with autocast('cuda', enabled=use_amp):
            shape_logits, y_pred_wld = model(batch_X)
            
            # Check if model outputs NaN
            if torch.isnan(shape_logits).any() or torch.isnan(y_pred_wld).any():
                print(f"\n[CRITICAL] Model outputs NaN detected (Epoch {epoch+1}, Batch {batch_idx}). Dừng huấn luyện, chuyển sang Test!")
                nan_detected = True
                break
            
            # Compute losses using centralized functions
            clf_loss = criterion_clf(shape_logits, batch_y_shape)
            w_loss, l_loss, d_loss, avg_reg_loss = compute_regression_losses(
                y_pred_wld, batch_y_wld, criterion_reg
            )
        t_forward = time.perf_counter()
        
        # Calculate PINN loss per batch if active
        pinn_loss_val = 0.0
        norm_batch_physics_loss = None
        
        if pinn_loss_active:
            # Physics kernels are sensitive to half precision. Keep the complete PINN graph in float32.
            with autocast('cuda', enabled=False):
                _, y_pred_wld_f32 = model(batch_X.float())
                
                sample_phys_losses = []
                for idx in range(batch_X.size(0)):
                    field_value = batch_X[idx, 0]
                    true_shape_idx = int(batch_y_shape[idx].item())
                    shape_name = unique_shapes[true_shape_idx]
                    
                    try:
                        if shape_name in SHAPE_MAP_TRAINING:
                            H_pred_field = compute_physics_loss_autograd(
                                field_value,
                                y_pred_wld_f32[idx, 0],
                                y_pred_wld_f32[idx, 1],
                                y_pred_wld_f32[idx, 2],
                                shape_name,
                                SHAPE_MAP_TRAINING,
                                y_scaler,
                                X_scaler=X_scaler,
                                w_true=batch_y_wld[idx, 0],
                                l_true=batch_y_wld[idx, 1],
                                d_true=batch_y_wld[idx, 2],
                                log_file=physics_log_file,
                            )
                        else:
                            H_pred_field = y_pred_wld_f32[idx].sum() * 0.0
                    except Exception:
                        H_pred_field = y_pred_wld_f32[idx].sum() * 0.0
                    sample_phys_losses.append(H_pred_field)
                
                if sample_phys_losses:
                    batch_physics_loss = torch.stack(sample_phys_losses).mean()
                    # Normalize PINN loss per batch
                    norm_batch_physics_loss = loss_normalizer.update_and_normalize("pinn", batch_physics_loss)
                    
                    pinn_loss_val = float(batch_physics_loss.detach().item())
                    epoch_pinn_loss_sum += pinn_loss_val * batch_X.size(0)
                    epoch_pinn_batches_applied += 1
                    epoch_pinn_samples_applied += batch_X.size(0)
        
        t_physics = time.perf_counter()
        
        # ============================================================
        # LOSS COMPUTATION & BACKWARD STRATEGY
        # ============================================================
        alpha = ALPHA_INIT
        
        # DATA LOSS with Homoscedastic Uncertainty Weighting (Kendall et al., 2018) - Separate for W, L, D
        norm_clf_loss = loss_normalizer.update_and_normalize("clf", clf_loss)
        norm_w_loss = loss_normalizer.update_and_normalize("w", w_loss)
        norm_l_loss = loss_normalizer.update_and_normalize("l", l_loss)
        norm_d_loss = loss_normalizer.update_and_normalize("d", d_loss)
        
        precision_clf = torch.exp(-model.log_var_clf)
        precision_w = torch.exp(-model.log_var_w)
        precision_l = torch.exp(-model.log_var_l)
        precision_d = torch.exp(-model.log_var_d)
        
        data_loss = (
            precision_clf * norm_clf_loss + 0.5 * model.log_var_clf +
            precision_w * norm_w_loss + 0.5 * model.log_var_w +
            precision_l * norm_l_loss + 0.5 * model.log_var_l +
            precision_d * norm_d_loss + 0.5 * model.log_var_d
        )
        
        if torch.isnan(data_loss):
            print(f"\n[CRITICAL] NaN data_loss detected (Epoch {epoch+1}, Batch {batch_idx}). Dừng huấn luyện, chuyển sang Test!")
            nan_detected = True
            break
            
        optimizer.zero_grad(set_to_none=True)
        
        if norm_batch_physics_loss is not None:
            total_loss = data_loss + alpha * norm_batch_physics_loss
        else:
            total_loss = data_loss
            
        if torch.isnan(total_loss):
            print(f"\n[CRITICAL] NaN total_loss detected (Epoch {epoch+1}, Batch {batch_idx}). Dừng huấn luyện, chuyển sang Test!")
            nan_detected = True
            break

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Clip gradients: if PINN is active, clip norm matches effective pinn clip scale
        clip_limit = max(1.0, alpha * PINN_BASE_GRAD_CLIP_NORM) if (pinn_loss_active and alpha > 0) else 1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_limit)
        
        if pinn_loss_active and grad_norm is not None:
            try:
                epoch_pinn_grad_norm += float(grad_norm.detach().cpu().item())
            except Exception:
                epoch_pinn_grad_norm += float(grad_norm)
                
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t_backward = time.perf_counter()
        
        train_clf_loss += clf_loss.item()
        train_w_loss += w_loss.item()
        train_l_loss += l_loss.item()
        train_d_loss += d_loss.item()
        processed_train_batches += 1
        
        _, predicted = torch.max(shape_logits, 1)
        train_clf_correct += (predicted == batch_y_shape).sum().item()
        train_clf_total += batch_y_shape.size(0)

        # Show running train metrics continuously.
        seen_train_batches = max(1, processed_train_batches)
        running_train_clf = train_clf_loss / seen_train_batches
        running_train_w = train_w_loss / seen_train_batches
        running_train_l = train_l_loss / seen_train_batches
        running_train_d = train_d_loss / seen_train_batches
        running_train_reg = (running_train_w + running_train_l + running_train_d) / 3.0
        running_train_total = running_train_clf + running_train_reg
        running_train_acc = train_clf_correct / train_clf_total if train_clf_total > 0 else 0

        train_postfix = {
            'trL': f"{running_train_total:.4f}",
            'trA': f"{running_train_acc:.4f}",
        }
        if DEBUG_TIMING and (batch_idx % TIMING_EVERY == 0):
            train_postfix.update({
                'data_s': f"{t_data - t_batch_start:.3f}",
                'fwd_s': f"{t_forward - t_data:.3f}",
                'phys_s': f"{t_physics - t_forward:.3f}",
                'bwd_s': f"{t_backward - t_physics:.3f}",
            })
        pbar.set_postfix(train_postfix)
    
    if nan_detected:
        break

    if processed_train_batches == 0:
        raise RuntimeError(
            "No valid training batch processed. Increase TRAIN_PERCENT or reduce BATCH_SIZE so train batches have at least 2 samples."
        )

    # Average PINN gradient norm over applied batches
    if epoch_pinn_batches_applied > 0:
        epoch_pinn_grad_norm /= epoch_pinn_batches_applied

    train_clf_loss /= processed_train_batches
    train_w_loss /= processed_train_batches
    train_l_loss /= processed_train_batches
    train_d_loss /= processed_train_batches
    if epoch_pinn_samples_applied > 0:
        train_physics_loss = epoch_pinn_loss_sum / epoch_pinn_samples_applied
    else:
        train_physics_loss = 0.0
    train_clf_acc = train_clf_correct / train_clf_total if train_clf_total > 0 else 0
    train_reg_loss_avg = (train_w_loss + train_l_loss + train_d_loss) / 3.0
    train_total_loss = train_clf_loss + train_reg_loss_avg
    
    # ===== VALIDATION =====
    model.eval()
    val_clf_loss = 0.0
    val_w_loss = 0.0
    val_l_loss = 0.0
    val_d_loss = 0.0
    val_clf_correct = 0
    val_clf_total = 0
    val_clf_pred_list = []
    val_clf_true_list = []
    
    val_pbar = tqdm(
        val_loader,
        desc=f'Val {epoch+1}/{EPOCHS}',
        unit='batch',
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    with torch.no_grad():
        for val_batch_idx, (batch_X, batch_y_wld, batch_y_shape) in enumerate(val_pbar):
            batch_X = batch_X.to(device)
            batch_y_wld = batch_y_wld.to(device)
            batch_y_shape = batch_y_shape.to(device)
            
            shape_logits, y_pred_wld = model(batch_X)
            
            # Compute losses using centralized functions
            clf_loss = criterion_clf(shape_logits, batch_y_shape)
            w_loss, l_loss, d_loss, avg_reg_loss = compute_regression_losses(
                y_pred_wld, batch_y_wld, criterion_reg
            )
            
            val_clf_loss += clf_loss.item()
            val_w_loss += w_loss.item()
            val_l_loss += l_loss.item()
            val_d_loss += d_loss.item()
            
            _, predicted = torch.max(shape_logits, 1)
            val_clf_correct += (predicted == batch_y_shape).sum().item()
            val_clf_total += batch_y_shape.size(0)
            val_clf_pred_list.append(predicted.cpu().numpy())
            val_clf_true_list.append(batch_y_shape.cpu().numpy())

            # Show running validation metrics continuously on tqdm.
            seen_val_batches = val_batch_idx + 1
            running_val_clf = val_clf_loss / seen_val_batches
            running_val_w = val_w_loss / seen_val_batches
            running_val_l = val_l_loss / seen_val_batches
            running_val_d = val_d_loss / seen_val_batches
            running_val_reg = (running_val_w + running_val_l + running_val_d) / 3.0
            running_val_total = running_val_clf + running_val_reg
            running_val_acc = val_clf_correct / val_clf_total if val_clf_total > 0 else 0
            val_pbar.set_postfix({
                'trL': f"{train_total_loss:.4f}",
                'trA': f"{train_clf_acc:.4f}",
                'vL': f"{running_val_total:.4f}",
                'vA': f"{running_val_acc:.4f}",
                'pv': f"{last_val_total_loss:.4f}" if last_val_total_loss is not None else 'n/a'
            })
    
    val_clf_loss /= len(val_loader)
    val_w_loss /= len(val_loader)
    val_l_loss /= len(val_loader)
    val_d_loss /= len(val_loader)
    val_clf_acc = val_clf_correct / val_clf_total if val_clf_total > 0 else 0
    val_reg_loss_avg = (val_w_loss + val_l_loss + val_d_loss) / 3.0
    val_total_loss = val_clf_loss + val_reg_loss_avg

    # Kiểm tra bất thường sụp đổ loss hoặc NaN
    if (
        np.isnan(val_total_loss)
        or np.isinf(val_total_loss)
        or np.isnan(val_clf_loss)
        or val_total_loss > 1e4
        or (last_val_total_loss is not None and last_val_total_loss > 0 and val_total_loss > 50.0 * last_val_total_loss and (epoch + 1) > 10)
    ):
        print(f"\n[CRITICAL] Phát hiện bất thường / sụp đổ loss / NaN (Epoch {epoch+1}, Val Loss={val_total_loss:.4f})! Dừng huấn luyện ngay, KHÔNG lưu đè checkpoint hỏng!")
        nan_detected = True
        break
    
    # ============================================================================
    # EARLY STOPPING (ONLY AFTER PINN IS ACTIVATED)
    # ============================================================================
    current_physics_loss = float(train_physics_loss)
    if pinn_activated and pinn_activated_epoch is not None and (ALPHA_INIT > 0):
        # After PINN activation, best checkpoint uses PINN-aware objective.
        checkpoint_score = val_total_loss + alpha * current_physics_loss
        checkpoint_score_name = "val_data_loss + alpha*train_physics_loss"
    else:
        checkpoint_score = val_total_loss
        checkpoint_score_name = "val_data_loss"

    if pinn_activated and pinn_activated_epoch is not None and (ALPHA_INIT > 0):
        # Check whether physics loss improved.
        if current_physics_loss < pinn_stop_best_loss * 0.995:  # Improve > 0.5%
            pinn_stop_best_loss = current_physics_loss
            pinn_stop_counter = 0
        else:
            pinn_stop_counter += 1

        # Stop if PINN has run long enough and loss is plateauing.
        epochs_with_pinn = (epoch + 1) - pinn_activated_epoch + 1
        if epochs_with_pinn >= min_pinn_epochs and pinn_stop_counter >= pinn_stop_patience:
            print(f"\n{'='*80}")
            print(f"🛑 PINN LOSS PLATEAU → STOPPING TRAINING")
            print(f"   Stopped at epoch {epoch+1} (PINN active since epoch {pinn_activated_epoch})")
            print(f"   Plateau counter: {pinn_stop_counter}/{pinn_stop_patience}")
            print(f"   Final PINN loss: {current_physics_loss:.6e}")
            print(f"{'='*80}\n")
            break
    else:
        pinn_stop_counter = 0
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    elapsed_total = time.time() - start_training
    eta_remaining = (elapsed_total / (epoch + 1)) * (EPOCHS - epoch - 1)
    
    # Format time nicely
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    # Print progress every epoch with detailed loss breakdown
    reg_loss_avg = train_reg_loss_avg
    data_loss = train_total_loss
    
    # Determine phase info for display
    if pinn_activated and pinn_activated_epoch is not None:
        epochs_with_pinn = (epoch + 1) - pinn_activated_epoch + 1
        phase_info = f"PINN active (from epoch {pinn_activated_epoch})"
        early_stop_info = f" | PINNStop: {pinn_stop_counter}/{pinn_stop_patience} (age: {epochs_with_pinn})"
    else:
        phase_info = "Warm-up (PINN inactive)"
        early_stop_info = ""
    
    if EPOCH_VERBOSE_LOG:
        effective_train_objective = train_total_loss + (
            alpha * train_physics_loss if (pinn_activated and pinn_activated_epoch is not None) else 0.0
        )
        print(f"\n{'='*100}")
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Time: {format_time(epoch_time)} | ETA: {format_time(eta_remaining)} | {phase_info}{early_stop_info}")
        print(f"  Train → Clf: {train_clf_loss:.4f} | W: {train_w_loss:.4f} | L: {train_l_loss:.4f} | D: {train_d_loss:.4f} | Acc: {train_clf_acc:.4f}")
        print(f"  Val   → Clf: {val_clf_loss:.4f} | W: {val_w_loss:.4f} | L: {val_l_loss:.4f} | D: {val_d_loss:.4f} | Acc: {val_clf_acc:.4f}")
        print(f"  Weights → alpha (PINN term): {alpha:.4f} | Loss: {effective_train_objective:.4f}")
        print(f"  Checkpoint metric ({checkpoint_score_name}): {checkpoint_score:.6f}")
    
    # Record history
    history['train_loss'].append(train_total_loss)
    history['val_loss'].append(val_total_loss)
    history['train_clf_acc'].append(train_clf_acc)
    history['val_clf_acc'].append(val_clf_acc)
    history['physics_loss'].append(train_physics_loss)
    history['checkpoint_score'].append(checkpoint_score)
    history['train_clf_loss'].append(train_clf_loss)
    history['val_clf_loss'].append(val_clf_loss)
    history['train_w_loss'].append(train_w_loss)
    history['train_l_loss'].append(train_l_loss)
    history['train_d_loss'].append(train_d_loss)
    history['val_w_loss'].append(val_w_loss)
    history['val_l_loss'].append(val_l_loss)
    history['val_d_loss'].append(val_d_loss)

    _append_epoch_loss_logs(
        epoch=epoch + 1,
        alpha=alpha,
        pinn_active=(pinn_activated and pinn_activated_epoch is not None),
        train_physics_loss=train_physics_loss,
        pinn_batches_applied=epoch_pinn_batches_applied,
        pinn_grad_norm=epoch_pinn_grad_norm,
        checkpoint_score=checkpoint_score,
        train_clf_loss=train_clf_loss,
        train_reg_loss=train_reg_loss_avg,
        train_data_loss=train_total_loss,
        val_clf_loss=val_clf_loss,
        val_reg_loss=val_reg_loss_avg,
        val_data_loss=val_total_loss,
        train_clf_acc=train_clf_acc,
        val_clf_acc=val_clf_acc,
    )

    # Learning rate scheduling
    scheduler.step(val_total_loss)
    
    # ============================================================================
    # CHECKPOINT SYSTEM
    # ============================================================================
    
    # Snapshot valid state after successful epoch
    last_valid_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    last_valid_optimizer_state = optimizer.state_dict()
    last_valid_scheduler_state = scheduler.state_dict()
    last_valid_epoch = epoch + 1
    last_valid_history = copy.deepcopy(history)

    # 1. Save best model ONLY (overwrite in OUTPUT_DIR) when checkpoint_score improves
    if checkpoint_score <= best_checkpoint_score:
        best_checkpoint_score = checkpoint_score
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_pytorch.pth'))

# Training complete
if nan_detected:
    print("\n" + "="*80)
    print(f"[CRITICAL][NAN DETECTED] Training stopped early due to NaN detection!")
    print(f"[ROLLBACK] Reverting model to the last clean checkpoint (Epoch {last_valid_epoch}) as BEST checkpoint.")
    print("="*80 + "\n")
    if last_valid_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in last_valid_model_state.items()})
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_pytorch.pth'))

plot_fig1_training_curves(history, OUTPUT_DIR, pinn_activated_epoch)

# ============================================================================
# 10. EVALUATION
# ============================================================================

print(f"\nEVALUATING ON TEST SET")

# Build prioritized list of checkpoints to try for final evaluation.
# Preference order:
# 1) EVAL_CHECKPOINT if provided via env
# 2) checkpoint_final/checkpoint_full.pth (final saved after training)
# 3) checkpoint_latest/model_latest.pth
# 4) epoch checkpoints in descending order (checkpoint_epoch_XXXX/checkpoint_full.pth)
# 5) best checkpoints (post_pinn and general)

import glob

def collect_epoch_checkpoints(out_dir):
    pattern_dir = os.path.join(out_dir, 'checkpoint_epoch_*')
    dirs = [d for d in sorted(glob.glob(pattern_dir), reverse=True)]
    entries = []
    for d in dirs:
        p = os.path.join(d, 'checkpoint_full.pth')
        if os.path.exists(p):
            try:
                base = os.path.basename(d)
                epoch_num = int(base.split('_')[-1])
            except Exception:
                epoch_num = 0
            entries.append((epoch_num, p))
    entries = sorted(entries, key=lambda x: x[0], reverse=True)
    return [p for _, p in entries]

selected_model_path = None
y_pred_shape = None
y_pred_wld = None
y_pred_wld_denorm = None

priority_list = []
eval_ckpt = os.environ.get('EVAL_CHECKPOINT', '').strip()
if eval_ckpt:
    priority_list.append(eval_ckpt)

# [FIX] Luôn ưu tiên checkpoint_best / best_post_pinn trước
priority_list.extend([
    os.path.join(OUTPUT_DIR, 'checkpoint_best_post_pinn', 'best_model_pytorch.pth'),
    os.path.join(OUTPUT_DIR, 'best_model_pytorch_post_pinn.pth'),
    os.path.join(OUTPUT_DIR, 'checkpoint_best', 'best_model_pytorch.pth'),
    os.path.join(OUTPUT_DIR, 'best_model_pytorch.pth'),
])

# Sau đó mới đến checkpoint_final / latest phòng khi không có best
final_ckpt = os.path.join(OUTPUT_DIR, 'checkpoint_final', 'checkpoint_full.pth')
final_model = os.path.join(OUTPUT_DIR, 'checkpoint_final', 'model_final.pth')
if os.path.exists(final_ckpt):
    priority_list.append(final_ckpt)
elif os.path.exists(final_model):
    priority_list.append(final_model)

latest_path = os.path.join(OUTPUT_DIR, 'checkpoint_latest', 'model_latest.pth')
if os.path.exists(latest_path):
    priority_list.append(latest_path)

epoch_ckpts = collect_epoch_checkpoints(OUTPUT_DIR)
priority_list.extend(epoch_ckpts)

seen = set(); ordered = []
for p in priority_list:
    if p and p not in seen:
        seen.add(p); ordered.append(p)

print(f"[INFO] Evaluation candidate checkpoints (in order):")
for p in ordered:
    print(f"  - {p}")

for candidate_path in ordered:
    if not os.path.exists(candidate_path):
        continue
    try:
        ck = _torch_load_checkpoint(candidate_path)
    except Exception as e:
        print(f"[WARN] Could not load checkpoint file {candidate_path}: {e}")
        continue
    try:
        _safe_load_state_dict(model, ck)
        model.eval()

        pred_shape_list = []
        pred_wld_list = []
        with torch.no_grad():
            for batch_X, _, _ in test_loader:
                batch_X = batch_X.to(device)
                shape_logits, pred_wld = model(batch_X)
                if torch.isnan(shape_logits).any() or torch.isnan(pred_wld).any():
                    raise ValueError('NaN in model outputs')
                pred_shape_list.append(torch.argmax(shape_logits, dim=1).cpu().numpy())
                pred_wld_list.append(pred_wld.cpu().numpy())

        tmp_pred_shape = np.concatenate(pred_shape_list)
        tmp_pred_wld = np.concatenate(pred_wld_list)

        tmp_pred_wld_denorm = y_scaler.inverse_transform(tmp_pred_wld)
        if np.any(np.isnan(tmp_pred_wld_denorm)):
            print(f"[WARN] Checkpoint gives NaN after denorm, skipping: {candidate_path}")
            continue

        selected_model_path = candidate_path
        y_pred_shape = tmp_pred_shape
        y_pred_wld = tmp_pred_wld
        y_pred_wld_denorm = tmp_pred_wld_denorm
        print(f"[OK] Selected checkpoint for Test: {selected_model_path}")
        break
    except Exception as e:
        print(f"[WARN] Checkpoint {candidate_path} produced invalid outputs: {e}")
        continue
    except Exception as e:
        print(f"[WARN] Không thể load/evaluate checkpoint {candidate_path}: {e}")
        continue

if selected_model_path is None:
    print("[CRITICAL] Không tìm được checkpoint hợp lệ để đánh giá (mọi candidate đều lỗi/NaN).")
    sys.exit(1)

y_test_denorm = y_test_scaler.inverse_transform(y_test_norm)

# Classification metrics (Q1 Standard)
clf_acc = float(np.mean(y_pred_shape == y_shape_test))
clf_bal_acc = float(balanced_accuracy_score(y_shape_test, y_pred_shape))
clf_f1_macro = float(f1_score(y_shape_test, y_pred_shape, average='macro', zero_division=0))
clf_mcc = float(matthews_corrcoef(y_shape_test, y_pred_shape))
prec, rec, f1, _ = precision_recall_fscore_support(y_shape_test, y_pred_shape, average='macro', zero_division=0)
precision_score_val = float(prec)
recall_score_val = float(rec)
f1_score_val = float(f1)
balanced_accuracy_val = clf_bal_acc
mcc_val = clf_mcc
print(f"\n--- CLASSIFICATION (Q1 METRICS) ---")
print(f"Accuracy: {clf_acc*100.0:.2f}% | Balanced Acc: {clf_bal_acc*100.0:.2f}% | F1-Macro: {clf_f1_macro*100.0:.2f}% | MCC: {clf_mcc:.4f}")
print(f"\n{classification_report(y_shape_test, y_pred_shape, target_names=unique_shapes, zero_division=0)}")

# ============================================================================
# OVERALL REGRESSION METRICS ON TEST SET (denormalized - Q1 Standard)
# ============================================================================
target_names_reg = ['W', 'L', 'D']
metrics_names = ['Width (W)', 'Length (L)', 'Depth (D)']
test_metrics = {'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'max_error': [], 'nrmse': []}
print("\n--- REGRESSION (Q1 METRICS - Denormalized) ---")
for i, tname in enumerate(target_names_reg):
    mae_i = float(mean_absolute_error(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    mse_i = float(mean_squared_error(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    rmse_i = float(np.sqrt(mse_i))
    r2_i = float(r2_score(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    max_err_i = float(np.max(np.abs(y_test_denorm[:, i] - y_pred_wld_denorm[:, i])))
    nrmse_i = float(calculate_nrmse(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    test_metrics['mae'].append(mae_i)
    test_metrics['mse'].append(mse_i)
    test_metrics['rmse'].append(rmse_i)
    test_metrics['r2'].append(r2_i)
    test_metrics['max_error'].append(max_err_i)
    test_metrics['nrmse'].append(nrmse_i)
    print(f"  {tname}: MAE={mae_i:.4f} mm | RMSE={rmse_i:.4f} mm | R²={r2_i:.4f} | MaxErr={max_err_i:.4f} mm | NRMSE={nrmse_i:.2f}%")

mae_avg = float(np.mean(test_metrics['mae']))
rmse_avg = float(np.mean(test_metrics['rmse']))
r2_avg = float(np.mean(test_metrics['r2']))
nrmse_avg = float(np.mean(test_metrics['nrmse']))
print(f"  Avg MAE: {mae_avg:.4f} mm | Avg RMSE: {rmse_avg:.4f} mm | Avg R²: {r2_avg:.4f} | Avg NRMSE: {nrmse_avg:.2f}%")

# Build per-class and per-target metric DataFrames used later
class_metrics_df = _build_class_metrics_df(
    y_shape_test, y_pred_shape, unique_shapes, TRAIN_PERCENT,
    pinn_alpha=ALPHA_INIT if physics_available_for_training else 0.0,
    pinn_active=physics_available_for_training,
)
regression_metrics_by_target_df = _build_regression_metrics_by_target_df(
    y_test_denorm, y_pred_wld_denorm, target_names_reg, TRAIN_PERCENT,
    pinn_alpha=ALPHA_INIT if physics_available_for_training else 0.0,
    pinn_active=physics_available_for_training,
)
# Save per-class and per-target CSVs to run output dir
_atomic_write_df(class_metrics_df, os.path.join(OUTPUT_DIR, 'test_class_metrics.csv'), index=False)
_atomic_write_df(regression_metrics_by_target_df, os.path.join(OUTPUT_DIR, 'test_regression_metrics_by_target.csv'), index=False)

# Save Standalone Standard Summary Metrics and Predictions CSVs
overall_summary_df = pd.DataFrame([{
    'Model': 'PI-MFLNet (Proposed)',
    'Train_Percent': f"{TRAIN_PERCENT}%",
    'Seed': RANDOM_STATE,
    'Alpha': ALPHA_INIT,
    'Warmup': PINN_ACTIVATION_EPOCH,
    'Accuracy': clf_acc * 100.0,
    'Balanced_Accuracy': clf_bal_acc * 100.0,
    'Precision_Macro': precision_score_val * 100.0,
    'Recall_Macro': recall_score_val * 100.0,
    'F1_Macro': clf_f1_macro * 100.0,
    'MCC': clf_mcc,
    'MAE_W': test_metrics['mae'][0], 'MAE_L': test_metrics['mae'][1], 'MAE_D': test_metrics['mae'][2], 'Overall_MAE': mae_avg,
    'RMSE_W': test_metrics['rmse'][0], 'RMSE_L': test_metrics['rmse'][1], 'RMSE_D': test_metrics['rmse'][2], 'Overall_RMSE': rmse_avg,
    'R2_W': test_metrics['r2'][0], 'R2_L': test_metrics['r2'][1], 'R2_D': test_metrics['r2'][2], 'Overall_R2': r2_avg,
    'NMAE_Percent': (test_metrics['mae'][0] / 1.2 + test_metrics['mae'][1] / 18.0 + test_metrics['mae'][2] / 2.9) / 3.0 * 100.0
}])
_atomic_write_df(overall_summary_df, os.path.join(OUTPUT_DIR, 'test_summary_metrics.csv'), index=False)

detailed_predictions_df = pd.DataFrame({
    'True_W': y_test_denorm[:, 0], 'Pred_W': y_pred_wld_denorm[:, 0],
    'True_L': y_test_denorm[:, 1], 'Pred_L': y_pred_wld_denorm[:, 1],
    'True_D': y_test_denorm[:, 2], 'Pred_D': y_pred_wld_denorm[:, 2],
    'True_Class_ID': y_shape_test, 'Pred_Class_ID': y_pred_shape,
    'True_Shape': [unique_shapes[s] for s in y_shape_test],
    'Pred_Shape': [unique_shapes[s] for s in y_pred_shape]
})
_atomic_write_df(detailed_predictions_df, os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

# ============================================================================
# REAL EXPERIMENTAL INFERENCE (EXPERIMENT_1: 5 kHz, 10 kHz, 20 kHz)
# ============================================================================
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from load_real_experiment_data import evaluate_real_experiment
    evaluate_real_experiment(model, X_scaler, y_scaler, unique_shapes, OUTPUT_DIR, device=device)
except Exception as e:
    print(f"[WARN] Real experiment evaluation encountered an error: {e}")

# Save Publication Quality Visualizations (fig1, fig2, fig3, fig4)
plot_fig1_training_curves(history, OUTPUT_DIR, pinn_activated_epoch)
plot_fig2_confusion_matrix(y_shape_test, y_pred_shape, unique_shapes, OUTPUT_DIR)
plot_fig3_regression_scatter(y_test_denorm, y_pred_wld_denorm, OUTPUT_DIR)
plot_fig4_tsne_latent_space(model, test_loader, y_shape_test, y_test_denorm, unique_shapes, OUTPUT_DIR, f"Train {TRAIN_PERCENT}%")

# ============================================================================
# PER-SHAPE ANALYSIS (Chi tiết cho từng loại vết nứt)
# ============================================================================

print(f"\n{'='*80}")
print("PER-SHAPE DETAILED ANALYSIS (Q1 METRICS)")
print(f"{'='*80}")

per_shape_results = []
shape_names = unique_shapes

for shape_idx, shape_name in enumerate(shape_names):
    shape_mask = (y_shape_test == shape_idx)
    n_shape = np.sum(shape_mask)
    
    if n_shape == 0:
        print(f"\n[SKIP] {shape_name}: no test samples")
        continue
    
    print(f"\n--- Shape: {shape_name} (n={n_shape}) ---")
    
    # Classification metrics per shape
    y_pred_shape_subset = y_pred_shape[shape_mask]
    y_shape_subset = y_shape_test[shape_mask]
    shape_clf_acc = float(np.mean(y_pred_shape_subset == y_shape_subset))
    
    # Regression metrics per shape
    y_test_shape = y_test_denorm[shape_mask]
    y_pred_shape_wld = y_pred_wld_denorm[shape_mask]
    
    shape_metrics = {'mae': [], 'rmse': [], 'r2': [], 'max_error': [], 'nrmse': []}
    
    for i in range(3):
        mae = float(mean_absolute_error(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_test_shape[:, i], y_pred_shape_wld[:, i])))
        r2 = float(r2_score(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        max_err = float(np.max(np.abs(y_test_shape[:, i] - y_pred_shape_wld[:, i])))
        nrmse = float(calculate_nrmse(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        
        shape_metrics['mae'].append(mae)
        shape_metrics['rmse'].append(rmse)
        shape_metrics['r2'].append(r2)
        shape_metrics['max_error'].append(max_err)
        shape_metrics['nrmse'].append(nrmse)
    
    shape_nrmse_avg = float(np.mean(shape_metrics['nrmse']))
    
    print(f"  Clf Acc: {shape_clf_acc*100.0:.2f}%")
    print(f"  MAE (W/L/D): {shape_metrics['mae'][0]:.4f} / {shape_metrics['mae'][1]:.4f} / {shape_metrics['mae'][2]:.4f} mm")
    print(f"  RMSE (W/L/D): {shape_metrics['rmse'][0]:.4f} / {shape_metrics['rmse'][1]:.4f} / {shape_metrics['rmse'][2]:.4f} mm")
    print(f"  R² (W/L/D): {shape_metrics['r2'][0]:.4f} / {shape_metrics['r2'][1]:.4f} / {shape_metrics['r2'][2]:.4f}")
    print(f"  NRMSE Avg: {shape_nrmse_avg:.2f}%")
    
    per_shape_results.append({
        'shape': shape_name,
        'n_samples': n_shape,
        'clf_acc': float(shape_clf_acc),
        'mae_w': float(shape_metrics['mae'][0]),
        'mae_l': float(shape_metrics['mae'][1]),
        'mae_d': float(shape_metrics['mae'][2]),
        'rmse_w': float(shape_metrics['rmse'][0]),
        'rmse_l': float(shape_metrics['rmse'][1]),
        'rmse_d': float(shape_metrics['rmse'][2]),
        'r2_w': float(shape_metrics['r2'][0]),
        'r2_l': float(shape_metrics['r2'][1]),
        'r2_d': float(shape_metrics['r2'][2]),
        'max_error_w': float(shape_metrics['max_error'][0]),
        'max_error_l': float(shape_metrics['max_error'][1]),
        'max_error_d': float(shape_metrics['max_error'][2]),
        'nrmse_w': float(shape_metrics['nrmse'][0]),
        'nrmse_l': float(shape_metrics['nrmse'][1]),
        'nrmse_d': float(shape_metrics['nrmse'][2]),
        'nrmse_avg': float(shape_nrmse_avg),
        'train_percent': TRAIN_PERCENT,
        'pinn_alpha': ALPHA_INIT if physics_available_for_training else 0.0,
        'pinn_active': physics_available_for_training,
    })

# Save per-shape results to CSV
if per_shape_results:
    per_shape_df = pd.DataFrame(per_shape_results)
    per_shape_csv = os.path.join(OUTPUT_DIR, 'test_metrics_per_shape.csv')
    _atomic_write_df(per_shape_df, per_shape_csv, index=False)
    print(f"\n[OK] Per-shape metrics saved: {per_shape_csv}")
    
    # Also save to checkpoint_final
    per_shape_csv_final = os.path.join(final_ckpt_dir, 'test_metrics_per_shape.csv')
    _atomic_write_df(per_shape_df, per_shape_csv_final, index=False)

# Visualize per-shape metrics comparison (Q1 Standard)
if per_shape_results:
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Per-Shape Metrics Comparison (Q1 Standard - Train {TRAIN_PERCENT}%)', fontsize=14, fontweight='bold')
        
        shapes = [r['shape'] for r in per_shape_results]
        
        # Classification accuracy per shape
        ax = axes[0, 0]
        accs = [r['clf_acc'] * 100.0 for r in per_shape_results]
        ax.bar(shapes, accs, color='skyblue', alpha=0.7)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Classification Accuracy per Shape')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(accs):
            ax.text(i, v + 1.0, f'{v:.1f}%', ha='center', fontsize=9)
        
        # MAE per shape (W/L/D avg)
        ax = axes[0, 1]
        maes = [(r['mae_w'] + r['mae_l'] + r['mae_d']) / 3.0 for r in per_shape_results]
        ax.bar(shapes, maes, color='lightcoral', alpha=0.7)
        ax.set_ylabel('MAE Avg (mm)', fontweight='bold')
        ax.set_title('Mean Absolute Error per Shape')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(maes):
            ax.text(i, v * 1.02, f'{v:.4f}', ha='center', fontsize=9)
        
        # RMSE per shape (W/L/D avg)
        ax = axes[1, 0]
        rmses = [(r['rmse_w'] + r['rmse_l'] + r['rmse_d']) / 3.0 for r in per_shape_results]
        ax.bar(shapes, rmses, color='lightgreen', alpha=0.7)
        ax.set_ylabel('RMSE Avg (mm)', fontweight='bold')
        ax.set_title('Root Mean Squared Error per Shape')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(rmses):
            ax.text(i, v * 1.02, f'{v:.4f}', ha='center', fontsize=9)
        
        # NRMSE per shape (avg)
        ax = axes[1, 1]
        nrmses = [r['nrmse_avg'] for r in per_shape_results]
        ax.bar(shapes, nrmses, color='plum', alpha=0.7)
        ax.set_ylabel('NRMSE Avg (%)', fontweight='bold')
        ax.set_title('Normalized RMSE per Shape')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(nrmses):
            ax.text(i, v * 1.02, f'{v:.2f}%', ha='center', fontsize=9)
        
        plt.tight_layout()
        per_shape_plot = os.path.join(OUTPUT_DIR, 'test_metrics_per_shape_comparison.png')
        _atomic_save_fig(plt, per_shape_plot, dpi=220)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed generating per-shape comparison plot: {e}")
        print(f"[OK] Per-shape visualization saved: {per_shape_plot}")
        # Per-class accuracy plot (recall per class)
        try:
            class_labels = shapes
            per_class_plot = os.path.join(OUTPUT_DIR, 'per_class_accuracy.png')
            _save_per_class_accuracy_plot(np.array(y_shape_test), np.array(y_pred_shape), class_labels, per_class_plot, title=f'Per-class Accuracy (Train {TRAIN_PERCENT}%)')
            # also save a copy into final checkpoint folder if exists
            try:
                os.makedirs(final_ckpt_dir, exist_ok=True)
                _save_per_class_accuracy_plot(np.array(y_shape_test), np.array(y_pred_shape), class_labels, os.path.join(final_ckpt_dir, 'per_class_accuracy.png'), title=f'Per-class Accuracy (Final Checkpoint)')
            except Exception:
                pass
            print(f"[OK] Per-class accuracy saved: {per_class_plot}")
        except Exception as e:
            print(f"[WARNING] Failed to save per-class accuracy plot: {e}")
    except Exception as e:
        print(f"[WARNING] Failed to generate per-shape plots: {e}")

# Save results table to CSV (needed for run_eval_all.py compatibility)
results_df = pd.DataFrame([{
    'train_percent': TRAIN_PERCENT,
    'seed': RANDOM_STATE,
    'series_label': SERIES_LABEL,
    'run_tag': RUN_TAG,
    'output_dir': OUTPUT_DIR,
    'mode': 'pinn',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'pinn_alpha': ALPHA_INIT,
    'pinn_warmup': PINN_ACTIVATION_EPOCH,
    'pinn_base_grad_clip_norm': PINN_BASE_GRAD_CLIP_NORM,
    'pinn_loss_type': PINN_LOSS_TYPE,
    'selected_checkpoint': selected_model_path,
    'clf_acc': clf_acc,
    'clf_precision_macro': precision_score_val,
    'clf_recall_macro': recall_score_val,
    'clf_f1_macro': f1_score_val,
    'clf_balanced_acc': balanced_accuracy_val,
    'clf_mcc': mcc_val,
    'mae_avg': mae_avg,
    'rmse_avg': rmse_avg,
    'r2_avg': r2_avg,
    'nrmse_avg': nrmse_avg,
    **_flatten_regression_metrics(test_metrics),
}])
results_csv_path = os.path.join(OUTPUT_DIR, "training_results.csv")
_atomic_write_df(results_df, results_csv_path, index=False)

# ---------- Detailed metrics (regression + classification + loss components + stability)
prec, rec, f1, _ = precision_recall_fscore_support(y_shape_test, y_pred_shape, average='macro', zero_division=0)

# Regression metrics already computed in test_metrics
metrics_row = {
    'train_percent': TRAIN_PERCENT,
    'seed': RANDOM_STATE,
    'series_label': SERIES_LABEL,
    'run_tag': RUN_TAG,
    'output_dir': OUTPUT_DIR,
    'mode': 'pinn',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'pinn_alpha': ALPHA_INIT,
    'pinn_warmup': PINN_ACTIVATION_EPOCH,
    'pinn_base_grad_clip_norm': PINN_BASE_GRAD_CLIP_NORM,
    'pinn_loss_type': PINN_LOSS_TYPE,
    'selected_checkpoint': selected_model_path,
    'clf_acc': float(clf_acc),
    'clf_balanced_acc': float(balanced_accuracy_val),
    'clf_precision_macro': float(prec),
    'clf_recall_macro': float(rec),
    'clf_f1_macro': float(f1),
    'clf_mcc': float(mcc_val),

    'mae_w': float(test_metrics['mae'][0]),
    'mae_l': float(test_metrics['mae'][1]),
    'mae_d': float(test_metrics['mae'][2]),
    'mae_avg': float(mae_avg),
    'mse_w': float(test_metrics['mse'][0]),
    'mse_l': float(test_metrics['mse'][1]),
    'mse_d': float(test_metrics['mse'][2]),
    'rmse_w': float(test_metrics['rmse'][0]),
    'rmse_l': float(test_metrics['rmse'][1]),
    'rmse_d': float(test_metrics['rmse'][2]),
    'rmse_avg': float(rmse_avg),
    'r2_w': float(test_metrics['r2'][0]),
    'r2_l': float(test_metrics['r2'][1]),
    'r2_d': float(test_metrics['r2'][2]),
    'r2_avg': float(r2_avg),
    'max_error_w': float(test_metrics['max_error'][0]),
    'max_error_l': float(test_metrics['max_error'][1]),
    'max_error_d': float(test_metrics['max_error'][2]),
    'nrmse_w': float(test_metrics['nrmse'][0]),
    'nrmse_l': float(test_metrics['nrmse'][1]),
    'nrmse_d': float(test_metrics['nrmse'][2]),
    'nrmse_avg': float(nrmse_avg),
}

# Grab final epoch loss components from history if available
def last_or_nan(lst):
    try:
        return float(lst[-1])
    except Exception:
        return float('nan')

metrics_row.update({
    'train_clf_loss_final': last_or_nan(history.get('train_clf_loss', [])),
    'train_w_loss_final': last_or_nan(history.get('train_w_loss', [])),
    'train_l_loss_final': last_or_nan(history.get('train_l_loss', [])),
    'train_d_loss_final': last_or_nan(history.get('train_d_loss', [])),
    'train_pinn_loss_final': last_or_nan(history.get('physics_loss', [])),
    'train_total_loss_final': last_or_nan(history.get('train_loss', [])),
    'val_clf_loss_final': last_or_nan(history.get('val_clf_loss', [])),
    'val_w_loss_final': last_or_nan(history.get('val_w_loss', [])),
    'val_l_loss_final': last_or_nan(history.get('val_l_loss', [])),
    'val_d_loss_final': last_or_nan(history.get('val_d_loss', [])),
    'val_total_loss_final': last_or_nan(history.get('val_loss', [])),
})

# Stability checks
stability = {
    'nan_detected_during_training': bool(nan_detected),
    'nan_in_test_predictions': bool(np.any(np.isnan(y_pred_wld_denorm))) if y_pred_wld_denorm is not None else True,
    'inf_in_history': any(np.isinf(x) for arr in [history.get('train_loss', []), history.get('val_loss', []), history.get('physics_loss', [])] for x in arr),
}
metrics_row.update(stability)

detailed_metrics_df = pd.DataFrame([metrics_row])
detailed_csv_path = os.path.join(OUTPUT_DIR, 'detailed_metrics.csv')
_atomic_write_df(detailed_metrics_df, detailed_csv_path, index=False)

# Also save to checkpoint_final/ for easy access
detailed_csv_final = os.path.join(final_ckpt_dir, 'test_detailed_final.csv')
_atomic_write_df(detailed_metrics_df, detailed_csv_final, index=False)
print(f"[OK] Final detailed metrics: {detailed_csv_final}")

summary_txt = '\nDETAILED METRICS (CSV): ' + detailed_csv_path + '\n'
summary_txt += 'FINAL CHECKPOINT METRICS: ' + detailed_csv_final + '\n'
summary_txt += 'Stability notes:\n'
summary_txt += f"  nan_detected_during_training: {stability['nan_detected_during_training']}\n"
summary_txt += f"  nan_in_test_predictions: {stability['nan_in_test_predictions']}\n"
summary_txt += f"  inf_in_history: {stability['inf_in_history']}\n"
_atomic_write_text(summary_txt, os.path.join(OUTPUT_DIR, 'summary_results.txt'))

# Write test summary to checkpoint_final/
test_summary_path = os.path.join(final_ckpt_dir, 'test_summary_final.txt')
summary_buf = io.StringIO()
summary_buf.write("="*80 + "\n")
summary_buf.write(f"FINAL TEST RESULTS - {RUN_TAG}\n")
summary_buf.write("="*80 + "\n")
summary_buf.write(f"Selected checkpoint: {selected_model_path}\n\n")
summary_buf.write("--- CLASSIFICATION METRICS ---\n")
summary_buf.write(f"Accuracy: {clf_acc*100.0:.2f}%\n")
summary_buf.write(f"Balanced Accuracy: {balanced_accuracy_val*100.0:.2f}%\n")
summary_buf.write(f"F1 (macro): {f1*100.0:.2f}%\n")
summary_buf.write(f"MCC: {mcc_val:.4f}\n\n")
summary_buf.write("--- REGRESSION METRICS (Denormalized) ---\n")
for i, name in enumerate(metrics_names):
    summary_buf.write(f"{name}:\n")
    summary_buf.write(f"  MAE:       {test_metrics['mae'][i]:.4f} mm\n")
    summary_buf.write(f"  RMSE:      {test_metrics['rmse'][i]:.4f} mm\n")
    summary_buf.write(f"  R²:        {test_metrics['r2'][i]:.4f}\n")
    summary_buf.write(f"  Max Error: {test_metrics['max_error'][i]:.4f} mm\n")
    summary_buf.write(f"  NRMSE:     {test_metrics['nrmse'][i]:.2f}%\n\n")
summary_buf.write(f"Average MAE:   {mae_avg:.4f} mm\n")
summary_buf.write(f"Average RMSE:  {rmse_avg:.4f} mm\n")
summary_buf.write(f"Average R²:    {r2_avg:.4f}\n")
summary_buf.write(f"Average NRMSE: {nrmse_avg:.2f}%\n\n")
summary_buf.write("--- STABILITY CHECKS ---\n")
summary_buf.write(f"NaN detected during training: {stability['nan_detected_during_training']}\n")
summary_buf.write(f"NaN in test predictions: {stability['nan_in_test_predictions']}\n")
summary_buf.write(f"Inf in history: {stability['inf_in_history']}\n\n")
summary_buf.write(f"Configuration:\n")
summary_buf.write(f"  TRAIN_PERCENT: {TRAIN_PERCENT}%\n")
summary_buf.write(f"  RANDOM_STATE: {RANDOM_STATE}\n")
summary_buf.write(f"  ALPHA_INIT: {ALPHA_INIT}\n")
summary_buf.write(f"  PINN_ACTIVATION_EPOCH: {PINN_ACTIVATION_EPOCH}\n")
_atomic_write_text(summary_buf.getvalue(), test_summary_path)
print(f"[OK] Test summary: {test_summary_path}")


print(f"\nGenerating regression scatter plots...")
save_regression_plots(y_test_denorm, y_pred_wld_denorm, OUTPUT_DIR, metrics_names, 
                             prefix="test_", title_prefix="Final Test Set - ", y_scaler=None)

# Summary: 2x2 overview (Loss, Accuracy, Physics, Summary)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.get('train_loss', []), label='Train', linewidth=2, color=TRAIN_LINE_COLOR)
axes[0, 0].plot(history.get('val_loss', []), label='Val', linewidth=2, color=VAL_LINE_COLOR)
axes[0, 0].set_title('Total Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Classification Accuracy
axes[0, 1].plot(history['train_clf_acc'], label='Train', linewidth=2, color=TRAIN_LINE_COLOR)
axes[0, 1].plot(history['val_clf_acc'], label='Val', linewidth=2, color=VAL_LINE_COLOR)
axes[0, 1].set_title('Classification Accuracy (Train vs Val)', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Physics Loss
phys_losses = np.array(history.get('physics_loss', []))
if len(phys_losses) > 0 and max(phys_losses) > 0:
    active_idx = np.where(phys_losses > 0)[0]
    if len(active_idx) > 0:
        axes[1, 0].plot(active_idx + 1, phys_losses[active_idx], label='log(1 + SSE/SSE_true)', linewidth=2, color='red')
    axes[1, 0].set_title('Physics Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
else:
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Physics Loss\n(Not Active Yet / PINN Warm-up)', 
                    transform=axes[1, 0].transAxes, fontsize=14, 
                    ha='center', va='center', color='gray',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# Test metrics summary
metrics_text = f"Test Set Metrics (Q1 Standard):\n\n"
metrics_text += f"Accuracy:     {clf_acc*100.0:.2f}%\n"
metrics_text += f"Bal Accuracy: {balanced_accuracy_val*100.0:.2f}%\n"
metrics_text += f"F1-Macro:     {f1*100.0:.2f}%\n"
metrics_text += f"MCC:          {mcc_val:.4f}\n\n"
metrics_text += f"Avg MAE:      {mae_avg:.4f} mm\n"
metrics_text += f"Avg RMSE:     {rmse_avg:.4f} mm\n"
metrics_text += f"Avg R²:       {r2_avg:.4f}\n"
metrics_text += f"Avg NRMSE:    {nrmse_avg:.2f}%\n\n"
metrics_text += "By Output Target:\n"
for i, name in enumerate(metrics_names):
    metrics_text += f"{name}: MAE={test_metrics['mae'][i]:.3f} | RMSE={test_metrics['rmse'][i]:.3f} | R²={test_metrics['r2'][i]:.3f} | NRMSE={test_metrics['nrmse'][i]:.1f}%\n"

axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "training_summary_v2.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 12. DETAILED LOSS PLOTS
# ============================================================================

# Plot 1: Classification Loss (Train vs Val)
plt.figure(figsize=(10, 6))
plt.plot(history['train_clf_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
plt.plot(history['val_clf_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Classification Loss - Train vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_classification.png"), dpi=150)
plt.close()

# Plot 2: Width Regression Loss (Train vs Val)
plt.figure(figsize=(10, 6))
plt.plot(history['train_w_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
plt.plot(history['val_w_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Width Regression Loss - Train vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_width.png"), dpi=150)
plt.close()

# Plot 3: Length Regression Loss (Train vs Val)
plt.figure(figsize=(10, 6))
plt.plot(history['train_l_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
plt.plot(history['val_l_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Length Regression Loss - Train vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_length.png"), dpi=150)
plt.close()

# Plot 4: Depth Regression Loss (Train vs Val)
plt.figure(figsize=(10, 6))
plt.plot(history['train_d_loss'], label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
plt.plot(history['val_d_loss'], label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Depth Regression Loss - Train vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_depth.png"), dpi=150)
plt.close()

# Plot 5: Combined Regression Loss (W, L, D all together - Train)
plt.figure(figsize=(12, 6))
plt.plot(history['train_w_loss'], label='Width (W)', linewidth=2.5, marker='o', markersize=3, alpha=0.7, color='green')
plt.plot(history['train_l_loss'], label='Length (L)', linewidth=2.5, marker='s', markersize=3, alpha=0.7, color='orange')
plt.plot(history['train_d_loss'], label='Depth (D)', linewidth=2.5, marker='^', markersize=3, alpha=0.7, color='purple')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Regression Loss (W, L, D) - Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_regression_train_combined.png"), dpi=150)
plt.close()

# Plot 5b: Combined Regression Loss (W, L, D all together - Validation)
plt.figure(figsize=(12, 6))
plt.plot(history['val_w_loss'], label='Width (W)', linewidth=2.5, marker='o', markersize=3, alpha=0.7, color='green')
plt.plot(history['val_l_loss'], label='Length (L)', linewidth=2.5, marker='s', markersize=3, alpha=0.7, color='orange')
plt.plot(history['val_d_loss'], label='Depth (D)', linewidth=2.5, marker='^', markersize=3, alpha=0.7, color='purple')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Regression Loss (W, L, D) - Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_regression_val_combined.png"), dpi=150)
plt.close()

# Plot 5c: Average Regression Loss (W+L+D)/3 - Train vs Val
train_avg_reg_loss = [(history['train_w_loss'][i] + history['train_l_loss'][i] + history['train_d_loss'][i]) / 3.0 
                      for i in range(len(history['train_w_loss']))]
val_avg_reg_loss = [(history['val_w_loss'][i] + history['val_l_loss'][i] + history['val_d_loss'][i]) / 3.0 
                    for i in range(len(history['val_w_loss']))]

plt.figure(figsize=(12, 6))
plt.plot(train_avg_reg_loss, label='Train (Avg)', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
plt.plot(val_avg_reg_loss, label='Validation (Avg)', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Average Regression Loss (W+L+D)/3 - Train vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_regression_average.png"), dpi=150)
plt.close()

# Plot 6: Physics Loss
if max(history['physics_loss']) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(history['physics_loss'], label='Physics Loss', linewidth=2.5, marker='o', markersize=3, color='red', alpha=0.8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Physics Loss (PINN)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "loss_physics.png"), dpi=150)
    plt.close()

# ============================================================================
# 12. FINAL PLOTS (No alpha tracking)
# ============================================================================

# Plot 4: Total Loss Comparison
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train (Total)', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
plt.plot(history['val_loss'], label='Validation (Total)', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Total Loss - Train vs Validation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_total.png"), dpi=150)
plt.close()

# Plot 4b: Loss Component Breakdown (Classification vs Avg Regression vs Physics)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Train Loss Components
ax1.plot(history['train_clf_loss'], label='Classification Loss', linewidth=2.5, marker='o', markersize=2.5, alpha=0.7, color='blue')
train_avg_reg = [(history['train_w_loss'][i] + history['train_l_loss'][i] + history['train_d_loss'][i]) / 3.0 
                 for i in range(len(history['train_w_loss']))]
ax1.plot(train_avg_reg, label='Avg Regression Loss (W+L+D)/3', linewidth=2.5, marker='s', markersize=2.5, alpha=0.7, color='green')
if max(history['physics_loss']) > 0:
    ax1.plot(history['physics_loss'], label='Physics Loss (PINN)', linewidth=2.5, marker='^', markersize=2.5, alpha=0.7, color='red')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Training Loss Components Breakdown', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Validation Loss Components
ax2.plot(history['val_clf_loss'], label='Classification Loss', linewidth=2.5, marker='o', markersize=2.5, alpha=0.7, color='blue')
val_avg_reg = [(history['val_w_loss'][i] + history['val_l_loss'][i] + history['val_d_loss'][i]) / 3.0 
               for i in range(len(history['val_w_loss']))]
ax2.plot(val_avg_reg, label='Avg Regression Loss (W+L+D)/3', linewidth=2.5, marker='s', markersize=2.5, alpha=0.7, color='green')
if max(history['physics_loss']) > 0:
    ax2.plot(history['physics_loss'], label='Physics Loss (PINN)', linewidth=2.5, marker='^', markersize=2.5, alpha=0.7, color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Validation Loss Components Breakdown', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_components_breakdown.png"), dpi=150)
plt.close()

summary = []
summary.append("PYTORCH MULTIMODEL v3 - PINN WARM-UP + PHYSICS PLATEAU STOP\n")
summary.append("="*80 + "\n\n")
summary.append("KEY FEATURES:\n")
summary.append("-"*80 + "\n")
summary.append("1. PINN ACTIVATION SCHEDULE:\n")
summary.append(f"   - Warm-up: epochs 1-{PINN_ACTIVATION_EPOCH} (PINN inactive)\n")
summary.append(f"   - PINN phase: epoch {PINN_ACTIVATION_EPOCH + 1}+ (physics term enabled)\n")
summary.append("   - Activation is fixed by epoch schedule (not validation early-stop trigger)\n\n")

summary.append("2. PINN PLATEAU EARLY STOP:\n")
summary.append("   - Monitors PINN loss after PINN activation\n")
summary.append(f"   - Patience: {pinn_stop_patience} epochs without >=0.5% improvement\n")
summary.append("   - Stops training when PINN loss plateaus\n\n")
    
summary.append("IMPROVEMENTS MADE:\n")
summary.append("-"*80 + "\n")
summary.append("1. Y-Normalization (Max-based scaling):\n")
summary.append("   Regression targets normalized as y_norm = y / max_train\n")
summary.append("   Denormalized during visualization and evaluation\n")
summary.append("   Scalers saved: w_scaler.pkl, l_scaler.pkl, d_scaler.pkl (SEPARATE)\n\n")
    
summary.append("2. Field Data Normalization:\n")
summary.append("   Raw vs Predicted field data normalized and denormalized\n")
summary.append("   Better visualization of field differences\n\n")
    
summary.append("3. Loss Weighting Strategy:\n")
summary.append("   Loss formula: L = L_clf + ((L_w + L_l + L_d)/3) + alpha*L_pinn\n")
summary.append(f"   Alpha init: alpha={ALPHA_INIT:.6f}\n")
summary.append("   Before PINN activation: physics term = 0\n")
summary.append("   After PINN activation: physics term weighted by alpha\n\n")
    
summary.append("PINN ACTIVATION:\n")
summary.append("-"*80 + "\n")
summary.append(f"Warm-up epoch threshold: {PINN_ACTIVATION_EPOCH}\n")
summary.append(f"Activated at epoch: {pinn_activated_epoch if pinn_activated_epoch is not None else 'Not activated'}\n")
summary.append(f"PINN plateau patience: {pinn_stop_patience}\n")
summary.append(f"Initial alpha: {ALPHA_INIT:.6f}\n")
summary.append("\n")
    
summary.append("GPU ACCELERATION:\n")
summary.append("-"*80 + "\n")
summary.append(f"Model Training (PyTorch):  {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
summary.append(f"Physics Computation:       GPU (PyTorch CUDA)\n\n")
    
summary.append("TEST RESULTS (Denormalized Values - Q1 Standard):\n")
summary.append("-"*80 + "\n")
summary.append(f"Classification Accuracy: {clf_acc*100.0:.2f}%\n")
summary.append(f"Balanced Accuracy:       {balanced_accuracy_val*100.0:.2f}%\n")
summary.append(f"F1-Macro:                {f1*100.0:.2f}%\n")
summary.append(f"MCC:                     {mcc_val:.4f}\n\n")
for i in range(3):
    summary.append(f"{metrics_names[i]}:\n")
    summary.append(f"  MAE:       {test_metrics['mae'][i]:.4f} mm\n")
    summary.append(f"  RMSE:      {test_metrics['rmse'][i]:.4f} mm\n")
    summary.append(f"  R²:        {test_metrics['r2'][i]:.4f}\n")
    summary.append(f"  Max Error: {test_metrics['max_error'][i]:.4f} mm\n")
    summary.append(f"  NRMSE:     {test_metrics['nrmse'][i]:.2f}%\n\n")
summary.append(f"Average MAE:   {mae_avg:.4f} mm\n")
summary.append(f"Average RMSE:  {rmse_avg:.4f} mm\n")
summary.append(f"Average R²:    {r2_avg:.4f}\n")
summary.append(f"Average NRMSE: {nrmse_avg:.2f}%\n\n")
    
summary.append("DATA NORMALIZATION:\n")
summary.append("-"*80 + "\n")
summary.append(f"X_scaler (StandardScaler):  X_scaler.pkl\n")
summary.append(f"W_scaler (MaxScaler):       w_scaler.pkl (SEPARATE for Width)\n")
summary.append(f"L_scaler (MaxScaler):       l_scaler.pkl (SEPARATE for Length)\n")
summary.append(f"D_scaler (MaxScaler):       d_scaler.pkl (SEPARATE for Depth)\n")
summary.append(f"y_scaler (SeparateMaxScaler): y_scaler.pkl (backward compatibility)\n")

_atomic_write_text('\n'.join(summary) + '\n', os.path.join(OUTPUT_DIR, "summary_results.txt"))

# ============================================================================
# REAL EXPERIMENTAL INFERENCE (EXPERIMENT_1: 5 kHz, 10 kHz, 20 kHz)
# ============================================================================
try:
    from load_real_experiment_data import evaluate_real_experiment
    evaluate_real_experiment(model, X_scaler, y_scaler, unique_shapes, OUTPUT_DIR, device=device)
except Exception as e:
    print(f"[WARN] Real experiment evaluation encountered an error: {e}")

print(f"\n[OK] All results saved to: {OUTPUT_DIR}")
print(f"[OK] Model: best_model_pytorch.pth")
print(f"[OK] Summary: summary_results.txt")
print(f"\n" + "="*80)
print(f"[PINN LOSS WEIGHT SUMMARY]")
print(f"="*80)
print(f"  Alpha init (PINN term weight): {ALPHA_INIT:.6f}")
print(f"  Warm-up (epochs 1-100): PINN inactive")
print(f"  PINN phase (epoch 101+): PINN active")
print(f"\nFormula used:")
print(f"  L = L_clf + ((L_w + L_l + L_d)/3) + alpha*L_pinn")
print(f"="*80)
