import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tempfile
import errno
import io
import csv

# Set default visible CUDA device before importing torch.
# run_eval_all.py may override this per subprocess for parallel GPU scheduling.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import torch.nn as nn
import torch.optim as optim
import inspect
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
import time
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import traceback
import faulthandler

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
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'main_percent_no_pinn.py'
    print(f"\n[FATAL] Unhandled exception in {script_name}", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _log_uncaught_exception

# ============================================================================
# GPU CONFIGURATION - FIXED TO GPU 6
# ============================================================================

print("="*70)
print("GPU ACCELERATION SETUP (PYTORCH v2 - NO PINN)")
print("="*70)

# GPU visibility is already configured before importing torch.
visible_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[GPU] CUDA_VISIBLE_DEVICES={visible_cuda} -> {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("[INFO] Using CPU mode")

print(f"PyTorch version: {torch.__version__}")
print("="*70 + "\n")

try:
    from library_functions import Load_Data_With_Labels
except ImportError:
    print("[ERROR] Could not find 'library_functions.py'")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.abspath(os.path.join(CHECK_DIR, ".."))

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH or not os.path.exists(DATA_PATH):
    for candidate in [
        os.path.join(CHECK_DIR, "Crack_Shape_Images"),
        os.path.join(SCRIPT_DIR, "Crack_Shape_Images"),
        os.path.join(PAPER_DIR, "rep_code", "Crack_Shape_Images"),
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
        os.path.join(PAPER_DIR, "rep_code", "labels.csv"),
        os.path.join(PAPER_DIR, "labels.csv"),
    ]:
        if os.path.exists(candidate):
            LABELS_PATH = candidate
            break

OUTPUT_ROOT_DIR = os.environ.get("NO_PINN_OUTPUT_ROOT", "Outputs_cnn_baseline")
if not os.path.isabs(OUTPUT_ROOT_DIR):
    OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_ROOT_DIR)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
if BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be >= 1")
EPOCHS = int(os.environ.get("EPOCHS", "300"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", os.environ.get("SEED", "42")))
LEARNING_RATE = 0.001
# Keep epoch-level prints off by default so tqdm progress remains clean.
EPOCH_VERBOSE_LOG = False

# Train data percentage relative to the FULL dataset size.
# Use env var TRAIN_PERCENT=1..10 for your requested runs.
TRAIN_PERCENT = int(os.environ.get("TRAIN_PERCENT", "10"))
if not 1 <= TRAIN_PERCENT <= 100:
    raise ValueError("TRAIN_PERCENT must be in [1, 100]")

# Keep outputs separated by train percentage and run timestamp.
RUN_TAG = pd.Timestamp.now().strftime("run_%Y%m%d_%H%M%S")
# Make the run tag configurable externally so the orchestration script can separate versions.
if "CUSTOM_RUN_TAG" in os.environ:
    RUN_TAG = os.environ["CUSTOM_RUN_TAG"] + "_" + RUN_TAG
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, f"train_{TRAIN_PERCENT:02d}pct", RUN_TAG)


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
        # fig_or_plt may be matplotlib.pyplot module or a Figure
        try:
            # if it's plt module
            if hasattr(fig_or_plt, 'savefig'):
                fig_or_plt.savefig(tmp, **savefig_kwargs)
            else:
                fig_or_plt.savefig(tmp, **savefig_kwargs)
        except Exception:
            # fallback to pyplot
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
# TRAINING SETUP (FIXED LOSS)
# ============================================================================
# Fixed no-PINN objective:
#   Loss = loss_clf + loss_reg
# where loss_reg = (loss_w + loss_l + loss_d) / 3
TRAINING_EPOCHS = EPOCHS
RUN_NAME = "fixed_loss"
SERIES_LABEL = os.environ.get("SERIES_LABEL", f"No PINN (E={EPOCHS})")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── EPOCH LOSS LOG (ghi realtime từng epoch) ──────────────────────────────────
EPOCH_LOSS_LOG_PATH = os.path.join(OUTPUT_DIR, "epoch_loss_log.csv")

def _init_epoch_loss_log():
    """Tạo file CSV với header nếu chưa tồn tại."""
    if not os.path.exists(EPOCH_LOSS_LOG_PATH) or os.path.getsize(EPOCH_LOSS_LOG_PATH) == 0:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "epoch", "pinn_loss", "data_loss", "total_loss",
            "train_percent", "series_label", "mode", "alpha"
        ])
        _atomic_write_text(buf.getvalue(), EPOCH_LOSS_LOG_PATH)

def _append_epoch_loss_log(epoch, data_loss, total_loss):
    """Append một dòng vào CSV ngay sau khi epoch kết thúc."""
    with open(EPOCH_LOSS_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,          # epoch
            0.0,            # pinn_loss (luôn = 0 với no-PINN model)
            f"{data_loss:.12g}",
            f"{total_loss:.12g}",
            TRAIN_PERCENT,  # train_percent
            SERIES_LABEL,   # series_label
            "no_pinn",      # mode
            "",             # alpha (không có)
        ])
print(f"Output folder: {OUTPUT_DIR}\n")
print(f"[CONFIG] Global seed fixed: {RANDOM_STATE}")
print(f"[CONFIG] Train percent over FULL dataset: {TRAIN_PERCENT}%")
print(f"[CONFIG] Run tag: {RUN_TAG}")
print(f"[CONFIG] Evaluation isolation mode: ON (run-based output isolation)")
print(f"[CONFIG] Epoch verbose log: {'ON' if EPOCH_VERBOSE_LOG else 'OFF'}")

_init_epoch_loss_log()
print(f"[OK] Epoch loss log: {EPOCH_LOSS_LOG_PATH}")

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

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"

TRAIN_LINE_COLOR = 'blue'
VAL_LINE_COLOR = 'red'

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

def plot_fig1_training_curves(history, output_dir):
    """fig1_training_curves.png - IEEE Publication Quality Training Curves"""
    _set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    
    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    markevery = max(1, len(epochs) // 10)
    
    # Subplot (a): Total & Data Loss
    axes[0].plot(epochs, history.get('train_loss', []), label='Train Loss', color='#1f77b4', linewidth=1.8, marker='o', markersize=4, markevery=markevery)
    axes[0].plot(epochs, history.get('val_loss', []), label='Val Loss', color='#d62728', linestyle='--', linewidth=1.8, marker='s', markersize=4, markevery=markevery)
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
    paper_colors = ['#004c6d', '#c35100', '#6b2d5c']
    
    # 1. REGRESSION METRICS COMPARISON (Q1 Standard)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Regression Metrics (Q1 Standard) - Test Set', fontsize=16, fontweight='bold')
    
    # MAE
    ax = axes[0, 0]
    mae_vals = test_metrics['mae']
    ax.bar(metrics_names, mae_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('MAE (mm)', fontweight='bold')
    ax.set_title('Mean Absolute Error (MAE)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(mae_vals):
        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE
    ax = axes[0, 1]
    rmse_vals = test_metrics['rmse']
    ax.bar(metrics_names, rmse_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('RMSE (mm)', fontweight='bold')
    ax.set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(rmse_vals):
        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    ax = axes[0, 2]
    r2_vals = test_metrics['r2']
    ax.bar(metrics_names, r2_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Coefficient of Determination ($R^2$)', fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(r2_vals):
        ax.text(i, max(v * 0.95, 0.05), f'{v:.4f}', ha='center', va='bottom', fontweight='bold', color='white' if v > 0.5 else 'black')
    
    # Max Error
    ax = axes[1, 0]
    max_err_vals = test_metrics.get('max_error', [np.max(np.abs(y_test_denorm[:, j] - y_pred_wld_denorm[:, j])) for j in range(3)])
    ax.bar(metrics_names, max_err_vals, color=paper_colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Max Error (mm)', fontweight='bold')
    ax.set_title('Maximum Absolute Error', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(max_err_vals):
        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    # NRMSE
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
    
    # 2. CLASSIFICATION METRICS (Q1 Standard)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bal_acc = float(balanced_accuracy_score(y_shape_test, grid_pred_shape if 'grid_pred_shape' in locals() else y_shape_test))
    f1_macro = float(f1_score(y_shape_test, grid_pred_shape if 'grid_pred_shape' in locals() else y_shape_test, average='macro', zero_division=0))
    mcc = float(matthews_corrcoef(y_shape_test, grid_pred_shape if 'grid_pred_shape' in locals() else y_shape_test))
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
    
    # 3. PREDICTIONS vs TRUE
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
    
    # 4. RESIDUALS
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

    # 5. ERROR DISTRIBUTION
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

    # 6. ERROR HEATMAP
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

    # 7. VIOLIN PLOT: True vs Pred
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

    # 8. VIOLIN PLOT: Signed errors
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

    # 9. VIOLIN PLOT BY SHAPE
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


def compute_combined_loss(clf_loss, w_loss, l_loss, d_loss, log_var_clf=None, log_var_w=None, log_var_l=None, log_var_d=None):
    """
    Compute Homoscedastic Uncertainty Weighted combined loss (Kendall et al., 2018).
    Formula: L = exp(-log_var_clf)*L_clf + 0.5*log_var_clf + exp(-log_var_w)*L_w + 0.5*log_var_w + exp(-log_var_l)*L_l + 0.5*log_var_l + exp(-log_var_d)*L_d + 0.5*log_var_d
    """
    reg_loss = (w_loss + l_loss + d_loss) / 3.0
    if all(v is not None for v in (log_var_clf, log_var_w, log_var_l, log_var_d)):
        precision_clf = torch.exp(-log_var_clf)
        precision_w = torch.exp(-log_var_w)
        precision_l = torch.exp(-log_var_l)
        precision_d = torch.exp(-log_var_d)
        total_loss = (
            precision_clf * clf_loss + 0.5 * log_var_clf +
            precision_w * w_loss + 0.5 * log_var_w +
            precision_l * l_loss + 0.5 * log_var_l +
            precision_d * d_loss + 0.5 * log_var_d
        )
    else:
        total_loss = clf_loss + reg_loss
    return total_loss, reg_loss

# ============================================================================
# IMPROVED PYTORCH MODEL (NO PINN - WITH UNCERTAINTY WEIGHTING)
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
        
        shape_logits = self.classifier(backbone_feat)
        reg_feat = self.regressor_backbone(backbone_feat)
        y_pred_wld = self.reg_head(reg_feat)
        
        return shape_logits, y_pred_wld

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

try:
    X, y, matched_filenames = Load_Data_With_Labels(DATA_PATH, LABELS_PATH)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    sys.exit(1)

print(f"[OK] Loaded: {len(X)} samples")

if len(X) != len(y) or len(X) != len(matched_filenames):
    raise ValueError("Data-Label mismatch!")
print(f"[OK] Alignment verified - {len(matched_filenames)} files matched")

# ============================================================================
# EXTRACT SHAPE LABELS
# ============================================================================

print(f"\n{'='*70}")
print("EXTRACTING SHAPE LABELS")
print(f"{'='*70}")

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
# 2. TRAIN/VAL/TEST SPLIT
# ============================================================================

print(f"\n{'='*70}")
print("SPLITTING DATA (70% train, 20% val, 10% test)")
print(f"{'='*70}")

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

    # Fixed per-class permutation for nested percent subsets across runs.
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

print(f"[OK] Train: {n_train} | Val: {n_val} | Test: {n_test}")
print(f"[OK] Data shape: (N, H, W, C) = ({n_train}, {h}, {w}, {ch})")
print(
    f"Sample counts (full/train/val/test): "
    f"{total_samples}/{n_train}/{n_val}/{n_test}"
)

split_info_path = os.path.join(OUTPUT_DIR, "dataset_split_info.txt")
split_txt = "DATASET SPLIT INFO\n"
split_txt += "=" * 80 + "\n"
split_txt += f"run_tag: {RUN_TAG}\n"
split_txt += f"train_percent_requested: {TRAIN_PERCENT}\n"
split_txt += "evaluation_isolation_mode: ON (run-based output isolation)\n"
split_txt += "resume_checkpoint_used: None (run-isolated mode)\n"
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

# ============================================================================
# 3. NORMALIZE DATA
# ============================================================================

print(f"\n{'='*70}")
print("NORMALIZING DATA")
print(f"{'='*70}")

# ===== NORMALIZE X (single train-fitted scaler for train/val/test) =====
X_scaler = StandardScaler()

X_train_flat_norm = X_scaler.fit_transform(X_train.reshape(n_train * h * w, ch))
X_val_flat_norm = X_scaler.transform(X_val.reshape(n_val * h * w, ch))
X_test_flat_norm = X_scaler.transform(X_test.reshape(n_test * h * w, ch))

X_train_norm = X_train_flat_norm.reshape(n_train, h, w, ch)
X_val_norm = X_val_flat_norm.reshape(n_val, h, w, ch)
X_test_norm = X_test_flat_norm.reshape(n_test, h, w, ch)

_atomic_write_pickle(X_scaler, os.path.join(OUTPUT_DIR, 'X_scaler.pkl'))
_atomic_write_pickle(X_scaler, os.path.join(OUTPUT_DIR, 'X_val_scaler.pkl'))
_atomic_write_pickle(X_scaler, os.path.join(OUTPUT_DIR, 'X_test_scaler.pkl'))

print("[OK] X normalization: single train-fitted StandardScaler (train/val/test)")

# ===== NORMALIZE Y (W, L, D with divide-by-max method) =====
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

print(f"\n{'='*70}")
print("PREPARING DATA FOR TRAINING")
print(f"{'='*70}")

# ====== X: FIELD IMAGES (Normalized) ======
X_train_final = X_train_norm  # (n_train, 32, 32, 2) - field images
X_val_final = X_val_norm      # (n_val, 32, 32, 2) - field images
X_test_final = X_test_norm    # (n_test, 32, 32, 2) - field images

# ====== y: NORMALIZED REGRESSION LABELS (W, L, D) ======
# Convert to PyTorch tensors and rearrange X to (N, C, H, W) format
# X: field image tensors
X_train_tensor = torch.FloatTensor(X_train_final).permute(0, 3, 1, 2)  # X: field images
X_val_tensor = torch.FloatTensor(X_val_final).permute(0, 3, 1, 2)      # X: field images
X_test_tensor = torch.FloatTensor(X_test_final).permute(0, 3, 1, 2)    # X: field images

# y: [W_norm, L_norm, D_norm] regression labels
y_train_tensor = torch.FloatTensor(y_train_norm)  # y: [W_norm, L_norm, D_norm] labels
y_val_tensor = torch.FloatTensor(y_val_norm)      # y: [W_norm, L_norm, D_norm] labels
y_test_tensor = torch.FloatTensor(y_test_norm)    # y: [W_norm, L_norm, D_norm] labels

y_shape_train_tensor = torch.LongTensor(y_shape_train)
y_shape_val_tensor = torch.LongTensor(y_shape_val)
y_shape_test_tensor = torch.LongTensor(y_shape_test)

# ============================================================================
# 5. CREATE DATALOADERS
# ============================================================================
# TensorDataset: (X: field images, y: [W_norm, L_norm, D_norm] labels, y_shape: class label)
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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=persistent_workers)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=persistent_workers)

print(f"[OK] DataLoaders created (workers={NUM_WORKERS}, pin_memory=True, batch_size={effective_batch_size}, drop_last={train_drop_last})")
if train_drop_last:
    print(f"[INFO] drop_last=True: dropping 1 sample in final training batch to avoid BatchNorm size-1 error")

# ============================================================================
# 6. BUILD MODEL
# ============================================================================

print(f"\n{'='*70}")
print("BUILDING IMPROVED MULTIMODEL")
print(f"{'='*70}")

num_shapes = len(unique_shapes)
model = ImprovedMultimodelNet(num_shapes).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[OK] Model created")
print(f"  Total params: {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# ============================================================================
# 7. LOSS FUNCTIONS
# ============================================================================

print(f"\n{'='*70}")
print("CONFIGURING LOSS FUNCTIONS")
print(f"{'='*70}")

criterion_clf = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

print(f"[OK] Loss configuration ready")
print(f"  Training mode: fixed loss (no weights)")
print(f"  Loss formula:  L = L_clf + ((L_w + L_l + L_d)/3)")
print(f"  Epochs: {TRAINING_EPOCHS}")
print("="*70)

# Denormalize test data for evaluation (use test scaler for ground-truth labels)
y_test_denorm = y_test_scaler.inverse_transform(y_test_norm)

# ============================================================================
# 8. MAIN TRAINING LOOP (FIXED LOSS)
# ============================================================================

print(f"\n{'='*70}")
print("TRAINING: FIXED LOSS (L_clf + L_reg)")
print(f"{'='*70}\n")

grid_results = []
total_combinations = 1
current_combo = 0
grid_search_start = time.time()

for run_name in [RUN_NAME]:
    for _single in [0]:
        current_combo += 1
        combo_start = time.time()
        
        # Create output folder for this run
        combo_folder = OUTPUT_DIR
        os.makedirs(combo_folder, exist_ok=True)
        
        print(f"[{current_combo}/{total_combinations}] Fixed loss run")
        print(f"  Folder: {combo_folder}")
        
        # ===== RESUME CHECKPOINT SYSTEM =====
        checkpoint_file = os.path.join(combo_folder, "best_model_pytorch.pth")
        history_file = os.path.join(combo_folder, "training_history.txt")
        
        start_epoch = 0
        best_grid_loss = float('inf')
        best_model_state = None
        train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_clf_acc': [],
            'train_clf_acc': []
        }
        
        # Check if checkpoint exists (resume)
        if os.path.exists(checkpoint_file) and os.path.exists(history_file):
            print(f"  Found checkpoint! Resuming from saved state...")
            resume_ok = False
            try:
                # Load model
                grid_model = ImprovedMultimodelNet(num_shapes).to(device)
                checkpoint = torch.load(checkpoint_file, map_location=device)
                grid_model.load_state_dict(checkpoint)
                best_model_state = grid_model.state_dict().copy()
                print(f"    → Model loaded from: {checkpoint_file}")

                # Load training history
                with open(history_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines[1:], 1):  # Skip header
                        parts = line.strip().split('\t')
                        if len(parts) == 5:
                            epoch, train_loss, val_loss, train_acc, val_acc = parts
                            train_history['epoch'].append(int(epoch))
                            train_history['train_loss'].append(float(train_loss))
                            train_history['val_loss'].append(float(val_loss))
                            train_history['train_clf_acc'].append(float(train_acc))
                            train_history['val_clf_acc'].append(float(val_acc))
                        elif len(parts) == 4:  # backward compat
                            epoch, train_loss, val_loss, val_acc = parts
                            train_history['epoch'].append(int(epoch))
                            train_history['train_loss'].append(float(train_loss))
                            train_history['val_loss'].append(float(val_loss))
                            train_history['train_clf_acc'].append(0.0)
                            train_history['val_clf_acc'].append(float(val_acc))

                if train_history['epoch']:
                    start_epoch = train_history['epoch'][-1]
                    best_grid_loss = min(train_history['val_loss'])
                    print(f"    → Resuming from epoch {start_epoch + 1}/{TRAINING_EPOCHS}")
                resume_ok = True
            except Exception as e:
                print(f"    → Checkpoint incompatible with current model/loss structure: {e}")
                print(f"    → Starting fresh training for this combination")

            if not resume_ok:
                start_epoch = 0
                best_grid_loss = float('inf')
                train_history = {
                    'epoch': [],
                    'train_loss': [],
                    'val_loss': [],
                    'val_clf_acc': [],
                    'train_clf_acc': []
                }
                grid_model = ImprovedMultimodelNet(num_shapes).to(device)
                best_model_state = grid_model.state_dict().copy()
        else:
            # Create new model for this combination
            grid_model = ImprovedMultimodelNet(num_shapes).to(device)
            best_model_state = grid_model.state_dict().copy()
            print(f"  → New training starting (fresh model)")
        
        grid_optimizer = optim.Adam(grid_model.parameters(), lr=LEARNING_RATE)
        grid_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            grid_optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )
        
        # Train for remaining epochs
        nan_detected = False
        last_valid_model_state = grid_model.state_dict().copy()
        last_valid_epoch = start_epoch
        last_valid_history = copy.deepcopy(train_history)

        epoch_iter = tqdm(
            range(start_epoch, TRAINING_EPOCHS),
            desc=f"Run {current_combo}/{total_combinations} (fixed loss)",
            unit="epoch",
            leave=False,
            ncols=120,
        )
        for grid_epoch in epoch_iter:
            if nan_detected:
                break
            grid_model.train()
            grid_train_clf_loss = 0.0
            grid_train_total_loss = 0.0
            grid_train_clf_correct = 0
            grid_train_clf_total = 0
            processed_train_batches = 0
            train_batches_for_norm = max(1, len(train_loader))
            grid_optimizer.zero_grad()
            
            for batch_X, batch_y_wld, batch_y_shape in train_loader:
                if batch_X.size(0) < 2:
                    # Safety guard for BatchNorm layers.
                    continue
                batch_X = batch_X.to(device)
                batch_y_wld = batch_y_wld.to(device)
                batch_y_shape = batch_y_shape.to(device)
                
                shape_logits, y_pred_wld = grid_model(batch_X)
                
                # Check if model outputs NaN
                if torch.isnan(shape_logits).any() or torch.isnan(y_pred_wld).any():
                    print(f"\n[CRITICAL] Model outputs NaN detected (Epoch {grid_epoch+1}). Dừng huấn luyện, chuyển sang Test!")
                    nan_detected = True
                    break
                
                clf_loss = criterion_clf(shape_logits, batch_y_shape)
                w_loss, l_loss, d_loss, avg_reg_loss = compute_regression_losses(
                    y_pred_wld, batch_y_wld, criterion_reg
                )
                
                # Combined loss (no PINN with Uncertainty Weighting):
                combined_loss, reg_loss = compute_combined_loss(
                    clf_loss, w_loss, l_loss, d_loss, 
                    grid_model.log_var_clf, grid_model.log_var_w, grid_model.log_var_l, grid_model.log_var_d
                )

                # Accumulate average epoch gradient, then step once per epoch.
                if torch.isnan(combined_loss) or torch.isnan(clf_loss) or torch.isnan(reg_loss):
                    print(f"\n[CRITICAL] NaN loss detected (Epoch {grid_epoch+1})... Stopping training, moving to Evaluation!")
                    nan_detected = True
                    break

                loss_for_backward = combined_loss / train_batches_for_norm
                loss_for_backward.backward()
                
                grid_train_clf_loss += clf_loss.item()
                grid_train_total_loss += combined_loss.item()
                _, predicted = torch.max(shape_logits, 1)
                grid_train_clf_correct += (predicted == batch_y_shape).sum().item()
                grid_train_clf_total += batch_y_shape.size(0)
                processed_train_batches += 1

            if nan_detected:
                break

            if processed_train_batches == 0:
                raise RuntimeError(
                    "Grid-search training epoch processed 0 batches: all train batches were skipped "
                    "(likely batch size < 2 for every batch)."
                )

            grid_optimizer.step()
            
            grid_train_clf_loss /= processed_train_batches
            grid_train_total_loss /= processed_train_batches
            grid_train_clf_acc = grid_train_clf_correct / grid_train_clf_total if grid_train_clf_total > 0 else 0
            
            grid_model.eval()
            grid_val_loss = 0.0
            grid_val_clf_correct = 0
            grid_val_clf_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y_wld, batch_y_shape in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y_wld = batch_y_wld.to(device)
                    batch_y_shape = batch_y_shape.to(device)
                    
                    shape_logits, y_pred_wld = grid_model(batch_X)
                    
                    if torch.isnan(shape_logits).any() or torch.isnan(y_pred_wld).any():
                        print(f"\n[CRITICAL] Model outputs NaN detected during validation (Epoch {grid_epoch+1}). Dừng huấn luyện, chuyển sang Test!")
                        nan_detected = True
                        break

                    clf_loss = criterion_clf(shape_logits, batch_y_shape)
                    w_loss, l_loss, d_loss, avg_reg_loss = compute_regression_losses(
                        y_pred_wld, batch_y_wld, criterion_reg
                    )
                    
                    # Combined loss (no PINN): clf + average regression
                    combined_loss, reg_loss = compute_combined_loss(
                        clf_loss, w_loss, l_loss, d_loss
                    )
                    
                    grid_val_loss += combined_loss.item()
                    _, predicted = torch.max(shape_logits, 1)
                    grid_val_clf_correct += (predicted == batch_y_shape).sum().item()
                    grid_val_clf_total += batch_y_shape.size(0)
            
            if nan_detected:
                break
            
            grid_val_loss /= len(val_loader)
            grid_val_clf_acc = grid_val_clf_correct / grid_val_clf_total if grid_val_clf_total > 0 else 0

            # Kiểm tra bất thường sụp đổ loss hoặc NaN
            if (
                np.isnan(grid_val_loss)
                or np.isinf(grid_val_loss)
                or grid_val_loss > 1e4
                or (len(train_history['val_loss']) > 10 and grid_val_loss > 50.0 * train_history['val_loss'][-1])
            ):
                print(f"\n[CRITICAL] Phát hiện bất thường / sụp đổ loss / NaN (Epoch {grid_epoch+1}, Val Loss={grid_val_loss:.4f})! Dừng huấn luyện ngay, KHÔNG lưu đè checkpoint hỏng!")
                nan_detected = True
                break

            grid_scheduler.step(grid_val_loss)

            elapsed_combo = time.time() - combo_start
            done_epochs = (grid_epoch - start_epoch + 1)
            total_epochs = max(1, TRAINING_EPOCHS - start_epoch)
            avg_epoch_time = elapsed_combo / max(1, done_epochs)
            eta_combo = avg_epoch_time * max(0, total_epochs - done_epochs)
            epoch_iter.set_postfix({
                'val_loss': f"{grid_val_loss:.4f}",
                'val_acc': f"{grid_val_clf_acc:.4f}",
                'elapsed': format_time(elapsed_combo),
                'eta': format_time(eta_combo),
            })
            
            # Track history
            train_history['epoch'].append(grid_epoch + 1)
            train_history['train_loss'].append(grid_train_total_loss)
            train_history['val_loss'].append(grid_val_loss)
            train_history['val_clf_acc'].append(grid_val_clf_acc)
            train_history['train_clf_acc'].append(grid_train_clf_acc)

            # Ghi epoch loss log realtime (no-PINN: pinn_loss = 0)
            _append_epoch_loss_log(
                epoch=grid_epoch + 1,
                data_loss=grid_train_total_loss,
                total_loss=grid_train_total_loss,
            )
            
            # Snapshot valid state after successful epoch
            last_valid_model_state = grid_model.state_dict().copy()
            last_valid_epoch = grid_epoch + 1
            last_valid_history = copy.deepcopy(train_history)
            best_model_state = last_valid_model_state
            
            # GHI ĐÈ LIÊN TỤC MỖI EPOCH VÀO 1 FILE DUY NHẤT
            torch.save(best_model_state, os.path.join(combo_folder, "best_model_pytorch.pth"))
            
            # Optional verbose print every 10 epochs (tqdm already shows live progress)
            if EPOCH_VERBOSE_LOG and (grid_epoch + 1) % 10 == 0:
                print(f"  Epoch {grid_epoch+1:2d}/{TRAINING_EPOCHS} | "
                      f"Val Loss: {grid_val_loss:.6f} | "
                      f"Val Acc: {grid_val_clf_acc:.4f}")
        
        if nan_detected:
            print("\n" + "="*80)
            print(f"[CRITICAL][NAN DETECTED] NoPINN training stopped early due to NaN!")
            print(f"[ROLLBACK] Reverting model to the last clean checkpoint (Epoch {last_valid_epoch}) as BEST checkpoint.")
            print("="*80 + "\n")
            if last_valid_model_state is not None:
                best_model_state = last_valid_model_state
                grid_model.load_state_dict(best_model_state)

        # Save model weights (always save, regardless of completion)
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(combo_folder, "best_model_pytorch.pth"))
        
        # Save training history
        hist_lines = []
        hist_lines.append("Epoch\tTrain_Loss\tVal_Loss\tTrain_Acc\tVal_Acc\n")
        for i, epoch in enumerate(train_history['epoch']):
            train_acc_val = train_history['train_clf_acc'][i] if i < len(train_history['train_clf_acc']) else 0.0
            hist_lines.append(f"{epoch}\t{train_history['train_loss'][i]:.6f}\t{train_history['val_loss'][i]:.6f}\t{train_acc_val:.6f}\t{train_history['val_clf_acc'][i]:.6f}\n")
        _atomic_write_text(''.join(hist_lines), os.path.join(combo_folder, "training_history.txt"))
        
        # Check if training completed or was interrupted
        if len(train_history['epoch']) >= TRAINING_EPOCHS:
            status = "COMPLETED"
        else:
            status = f"INTERRUPTED at epoch {len(train_history['epoch'])}/{TRAINING_EPOCHS}"

        combo_elapsed = time.time() - combo_start
        
        # Evaluate as long as at least one epoch is available.
        if train_history['epoch']:
            if best_model_state is not None:
                grid_model.load_state_dict(best_model_state)
            
            grid_model.eval()
            grid_pred_shape_list = []
            grid_pred_wld_list = []
            
            with torch.no_grad():
                for batch_X, _, _ in test_loader:
                    batch_X = batch_X.to(device)
                    shape_logits, y_pred_wld = grid_model(batch_X)
                    grid_pred_shape_list.append(torch.argmax(shape_logits, dim=1).cpu().numpy())
                    grid_pred_wld_list.append(y_pred_wld.cpu().numpy())
            
            grid_pred_shape = np.concatenate(grid_pred_shape_list)
            grid_pred_wld = np.concatenate(grid_pred_wld_list)
            
            # Check for NaN in predictions
            if np.any(np.isnan(grid_pred_wld)):
                print(f"[CRITICAL] NaN detected in {np.sum(np.isnan(grid_pred_wld))} prediction values from the loaded model.")
                print(f"[CRITICAL] Stopping test evaluation because model outputs are invalid (NaN).")
                sys.exit(1)

            # Denormalize and compute metrics (same scaler space as y_test_denorm)
            grid_pred_wld_denorm = y_test_scaler.inverse_transform(grid_pred_wld)
            
            if np.any(np.isnan(grid_pred_wld_denorm)):
                print(f"[CRITICAL] NaN detected after denormalization.")
                print(f"[CRITICAL] Stopping test evaluation.")
                sys.exit(1)
            
            grid_clf_acc = float(np.mean(grid_pred_shape == y_shape_test))
            grid_clf_bal_acc = float(balanced_accuracy_score(y_shape_test, grid_pred_shape))
            grid_clf_f1_macro = float(f1_score(y_shape_test, grid_pred_shape, average='macro', zero_division=0))
            grid_clf_mcc = float(matthews_corrcoef(y_shape_test, grid_pred_shape))
            
            # Compute full Q1 metrics across W, L, D
            metrics_names_wld = ['Width (W)', 'Length (L)', 'Depth (D)']
            metrics_targets = ['W', 'L', 'D']
            grid_mae_values = []
            grid_mse_values = []
            grid_rmse_values = []
            grid_r2_values = []
            grid_max_error_values = []
            grid_nrmse_values = []
            for i in range(3):
                mse = mean_squared_error(y_test_denorm[:, i], grid_pred_wld_denorm[:, i])
                grid_mae_values.append(float(mean_absolute_error(y_test_denorm[:, i], grid_pred_wld_denorm[:, i])))
                grid_mse_values.append(float(mse))
                grid_rmse_values.append(float(np.sqrt(mse)))
                grid_r2_values.append(float(r2_score(y_test_denorm[:, i], grid_pred_wld_denorm[:, i])))
                grid_max_error_values.append(float(np.max(np.abs(y_test_denorm[:, i] - grid_pred_wld_denorm[:, i]))))
                grid_nrmse_values.append(float(calculate_nrmse(y_test_denorm[:, i], grid_pred_wld_denorm[:, i])))

            grid_nrmse_avg = float(np.mean(grid_nrmse_values))
            grid_mae_avg = float(np.mean(grid_mae_values))
            grid_rmse_avg = float(np.mean(grid_rmse_values))
            grid_r2_avg = float(np.mean(grid_r2_values))

            class_metrics_df = _build_class_metrics_df(
                y_shape_test,
                grid_pred_shape,
                unique_shapes,
                TRAIN_PERCENT,
                pinn_alpha=0.0,
                pinn_active=False,
            )
            class_metrics_csv = os.path.join(combo_folder, 'test_class_metrics.csv')
            _atomic_write_df(class_metrics_df, class_metrics_csv, index=False)
            print(f"[OK] Per-class metrics saved: {class_metrics_csv}")

            regression_metrics_by_target_df = _build_regression_metrics_by_target_df(
                y_test_denorm,
                grid_pred_wld_denorm,
                metrics_targets,
                TRAIN_PERCENT,
                pinn_alpha=0.0,
                pinn_active=False,
            )
            regression_metrics_by_target_csv = os.path.join(combo_folder, 'test_regression_metrics_by_target.csv')
            _atomic_write_df(regression_metrics_by_target_df, regression_metrics_by_target_csv, index=False)
            print(f"[OK] Regression metrics by W/L/D saved: {regression_metrics_by_target_csv}")
            
            # Save test metrics (Q1 Standard)
            tm_lines = []
            tm_lines.append("Configuration:\n")
            tm_lines.append("  Loss formula: L_clf + ((L_w + L_l + L_d)/3)\n\n")
            tm_lines.append(f"Classification Accuracy: {grid_clf_acc * 100.0:.2f}%\n")
            tm_lines.append(f"Classification Balanced Acc: {grid_clf_bal_acc * 100.0:.2f}%\n")
            tm_lines.append(f"Classification F1-Macro: {grid_clf_f1_macro * 100.0:.2f}%\n")
            tm_lines.append(f"Classification MCC: {grid_clf_mcc:.4f}\n\n")
            tm_lines.append(f"Regression MAE Avg: {grid_mae_avg:.4f} mm\n")
            tm_lines.append(f"Regression RMSE Avg: {grid_rmse_avg:.4f} mm\n")
            tm_lines.append(f"Regression R2 Avg: {grid_r2_avg:.4f}\n")
            tm_lines.append(f"Regression NRMSE Avg: {grid_nrmse_avg:.2f}%\n\n")
            tm_lines.append(f"Regression Metrics by Dimension:\n")
            for i, name in enumerate(metrics_names_wld):
                tm_lines.append(f"{name}:\n")
                tm_lines.append(f"  MAE:     {grid_mae_values[i]:.4f} mm\n")
                tm_lines.append(f"  RMSE:    {grid_rmse_values[i]:.4f} mm\n")
                tm_lines.append(f"  R2:      {grid_r2_values[i]:.4f}\n")
                tm_lines.append(f"  MaxErr:  {grid_max_error_values[i]:.4f} mm\n")
                tm_lines.append(f"  NRMSE:   {grid_nrmse_values[i]:.2f}%\n")
            _atomic_write_text(''.join(tm_lines), os.path.join(combo_folder, "test_metrics.txt"))
            
            # Save Publication Quality Visualizations (fig1, fig2, fig3, fig4)
            plot_fig1_training_curves(train_history, combo_folder)
            plot_fig2_confusion_matrix(y_shape_test, grid_pred_shape, unique_shapes, combo_folder)
            plot_fig3_regression_scatter(y_test_denorm, grid_pred_wld_denorm, combo_folder)
            plot_fig4_tsne_latent_space(model, test_loader, y_shape_test, y_test_denorm, unique_shapes, combo_folder, f"No-PINN Train {TRAIN_PERCENT}%")

            # Generate comprehensive metric visualizations (Q1 Standard)
            test_metrics = {
                'mae': grid_mae_values,
                'mse': grid_mse_values,
                'rmse': grid_rmse_values,
                'r2': grid_r2_values,
                'max_error': grid_max_error_values,
                'nrmse': grid_nrmse_values
            }
            precision_score_val = float(precision_score(y_shape_test, grid_pred_shape, average='macro', zero_division=0))
            recall_score_val = float(recall_score(y_shape_test, grid_pred_shape, average='macro', zero_division=0))
            f1_score_val = float(f1_score(y_shape_test, grid_pred_shape, average='macro', zero_division=0))

            try:
                generate_comprehensive_metrics_visualizations(
                    test_metrics=test_metrics,
                    clf_acc=grid_clf_acc,
                    prec=precision_score_val,
                    rec=recall_score_val,
                    f1=f1_score_val,
                    y_test_denorm=y_test_denorm,
                    y_pred_wld_denorm=grid_pred_wld_denorm,
                    y_shape_test=y_shape_test,
                    unique_shapes=unique_shapes,
                    output_dir=combo_folder
                )
            except Exception as e:
                print(f"[WARNING] Failed to generate comprehensive visualizations: {e}")
            
            # ============================================================================
            # PER-SHAPE ANALYSIS (Chi tiết cho từng loại vết nứt - Q1 Standard)
            # ============================================================================
            per_shape_results = []
            shape_names = unique_shapes
            
            for shape_idx, shape_name in enumerate(shape_names):
                shape_mask = (y_shape_test == shape_idx)
                n_shape = np.sum(shape_mask)
                
                if n_shape == 0:
                    continue
                
                # Classification accuracy per shape
                y_pred_shape_subset = grid_pred_shape[shape_mask]
                y_shape_subset = y_shape_test[shape_mask]
                shape_clf_acc = float(np.mean(y_pred_shape_subset == y_shape_subset))
                
                # Regression metrics per shape
                y_test_shape = y_test_denorm[shape_mask]
                y_pred_shape_wld = grid_pred_wld_denorm[shape_mask]
                
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
                })
            
            # Save per-shape results to CSV
            if per_shape_results:
                per_shape_df = pd.DataFrame(per_shape_results)
                per_shape_csv = os.path.join(combo_folder, 'test_metrics_per_shape.csv')
                _atomic_write_df(per_shape_df, per_shape_csv, index=False)
                
                # Visualize per-shape metrics comparison (Q1 Standard)
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
                    
                    # MAE per shape (avg)
                    ax = axes[0, 1]
                    maes = [(r['mae_w'] + r['mae_l'] + r['mae_d']) / 3.0 for r in per_shape_results]
                    ax.bar(shapes, maes, color='lightcoral', alpha=0.7)
                    ax.set_ylabel('MAE Avg (mm)', fontweight='bold')
                    ax.set_title('Mean Absolute Error per Shape')
                    ax.grid(axis='y', alpha=0.3)
                    for i, v in enumerate(maes):
                        ax.text(i, v * 1.02, f'{v:.4f}', ha='center', fontsize=9)
                    
                    # RMSE per shape (avg)
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
                    per_shape_plot = os.path.join(combo_folder, 'test_metrics_per_shape.png')
                    _atomic_save_fig(plt, per_shape_plot, dpi=150, bbox_inches='tight')
                    plt.close()
                    # Per-class accuracy plot (recall per class)
                    try:
                        class_labels = shapes
                        per_class_plot = os.path.join(combo_folder, 'per_class_accuracy.png')
                        _save_per_class_accuracy_plot(y_shape_test, grid_pred_shape, class_labels, per_class_plot, title=f'Per-class Accuracy (Train {TRAIN_PERCENT}%)')
                        # also save overall copy
                        _save_per_class_accuracy_plot(y_shape_test, grid_pred_shape, class_labels, os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'), title=f'Per-class Accuracy (Train {TRAIN_PERCENT}%)')
                        print(f"[OK] Per-class accuracy saved: {per_class_plot}")
                    except Exception as e:
                        print(f"[WARNING] Failed to save per-class accuracy plot: {e}")
                except Exception as e:
                    print(f"[WARNING] Failed to generate per-shape plots: {e}")
            
            grid_results.append({
                'clf_acc': grid_clf_acc,
                'clf_balanced_acc': grid_clf_bal_acc,
                'clf_f1_macro': grid_clf_f1_macro,
                'clf_mcc': grid_clf_mcc,
                'mae_avg': grid_mae_avg,
                'rmse_avg': grid_rmse_avg,
                'r2_avg': grid_r2_avg,
                'nrmse_avg': grid_nrmse_avg,
                'clf_precision_macro': precision_score_val,
                'clf_recall_macro': recall_score_val,
                'clf_f1_macro': f1_score_val,
                **_flatten_regression_metrics(test_metrics),
            })
            
            print(f"  {status} | Time={format_time(combo_elapsed)} | Acc={grid_clf_acc*100.0:.2f}% | BalAcc={grid_clf_bal_acc*100.0:.2f}% | MAE={grid_mae_avg:.4f}mm | RMSE={grid_rmse_avg:.4f}mm | R2={grid_r2_avg:.4f} | NRMSE={grid_nrmse_avg:.2f}%")
        else:
            print(f"  {status} | Time={format_time(combo_elapsed)} | No epochs trained, results not computed")

print(f"\n[OK] Training complete!")
print(f"[TIME] Total training time: {format_time(time.time() - grid_search_start)}")

if len(grid_results) == 0:
    raise RuntimeError("No training result was produced. Please check data and training logs.")

final_result = grid_results[0]

print(f"\n{'='*70}")
print("FINAL RESULT (Q1 METRIC STANDARDS)")
print(f"{'='*70}")
print(f"  Classification Accuracy: {final_result['clf_acc']*100.0:.2f}%")
print(f"  Balanced Accuracy:       {final_result['clf_balanced_acc']*100.0:.2f}%")
print(f"  F1-Macro:                {final_result['clf_f1_macro']*100.0:.2f}%")
print(f"  MCC:                     {final_result['clf_mcc']:.4f}")
print(f"  Avg MAE:                 {final_result['mae_avg']:.4f} mm")
print(f"  Avg RMSE:                {final_result['rmse_avg']:.4f} mm")
print(f"  Avg R²:                  {final_result['r2_avg']:.4f}")
print(f"  Avg NRMSE:               {final_result['nrmse_avg']:.2f}%")

# Save results table to CSV
results_df = pd.DataFrame(grid_results)
results_csv_path = os.path.join(OUTPUT_DIR, "training_results.csv")
_atomic_write_df(results_df, results_csv_path, index=False)
print(f"[OK] Results CSV saved: {results_csv_path}")

if len(results_df) > 0:
    detailed_metrics_df = results_df.copy()
    detailed_metrics_df['train_percent'] = TRAIN_PERCENT
    detailed_metrics_df['pinn_alpha'] = 0.0
    detailed_metrics_df['pinn_active'] = False
    detailed_metrics_df['train_clf_loss_final'] = train_history['train_clf_loss'][-1] if train_history.get('train_clf_loss') else np.nan
    detailed_metrics_df['train_w_loss_final'] = train_history['train_w_loss'][-1] if train_history.get('train_w_loss') else np.nan
    detailed_metrics_df['train_l_loss_final'] = train_history['train_l_loss'][-1] if train_history.get('train_l_loss') else np.nan
    detailed_metrics_df['train_d_loss_final'] = train_history['train_d_loss'][-1] if train_history.get('train_d_loss') else np.nan
    detailed_metrics_df['train_total_loss_final'] = train_history['train_loss'][-1] if train_history.get('train_loss') else np.nan
    detailed_metrics_df['val_clf_loss_final'] = train_history['val_clf_loss'][-1] if train_history.get('val_clf_loss') else np.nan
    detailed_metrics_df['val_w_loss_final'] = train_history['val_w_loss'][-1] if train_history.get('val_w_loss') else np.nan
    detailed_metrics_df['val_l_loss_final'] = train_history['val_l_loss'][-1] if train_history.get('val_l_loss') else np.nan
    detailed_metrics_df['val_d_loss_final'] = train_history['val_d_loss'][-1] if train_history.get('val_d_loss') else np.nan
    detailed_metrics_df['val_total_loss_final'] = train_history['val_loss'][-1] if train_history.get('val_loss') else np.nan
    detailed_csv_path = os.path.join(OUTPUT_DIR, 'detailed_metrics.csv')
    _atomic_write_df(detailed_metrics_df, detailed_csv_path, index=False)
    print(f"[OK] Detailed metrics saved: {detailed_csv_path}")

# Save Standalone Standard Summary Metrics and Predictions CSVs
overall_summary_df = pd.DataFrame([{
    'Model': '2D CNN (NoPINN)',
    'Train_Percent': f"{TRAIN_PERCENT}%",
    'Seed': RANDOM_STATE,
    'Accuracy': final_result['clf_acc'] * 100.0,
    'Balanced_Accuracy': final_result['clf_balanced_acc'] * 100.0,
    'Precision_Macro': final_result['clf_precision_macro'] * 100.0 if 'clf_precision_macro' in final_result else final_result['clf_f1_macro'] * 100.0,
    'Recall_Macro': final_result['clf_recall_macro'] * 100.0 if 'clf_recall_macro' in final_result else final_result['clf_f1_macro'] * 100.0,
    'F1_Macro': final_result['clf_f1_macro'] * 100.0,
    'MCC': final_result['clf_mcc'],
    'MAE_W': final_result['mae_w'] if 'mae_w' in final_result else np.nan,
    'MAE_L': final_result['mae_l'] if 'mae_l' in final_result else np.nan,
    'MAE_D': final_result['mae_d'] if 'mae_d' in final_result else np.nan,
    'Overall_MAE': final_result['mae_avg'],
    'RMSE_W': final_result['rmse_w'] if 'rmse_w' in final_result else np.nan,
    'RMSE_L': final_result['rmse_l'] if 'rmse_l' in final_result else np.nan,
    'RMSE_D': final_result['rmse_d'] if 'rmse_d' in final_result else np.nan,
    'Overall_RMSE': final_result['rmse_avg'],
    'R2_W': final_result['r2_w'] if 'r2_w' in final_result else np.nan,
    'R2_L': final_result['r2_l'] if 'r2_l' in final_result else np.nan,
    'R2_D': final_result['r2_d'] if 'r2_d' in final_result else np.nan,
    'Overall_R2': final_result['r2_avg'],
    'NMAE_Percent': final_result['nmae_avg'] if 'nmae_avg' in final_result else np.nan
}])
_atomic_write_df(overall_summary_df, os.path.join(OUTPUT_DIR, 'test_summary_metrics.csv'), index=False)

if 'grid_pred_shape' in locals() and 'grid_pred_wld_denorm' in locals():
    detailed_predictions_df = pd.DataFrame({
        'True_W': y_test_denorm[:, 0], 'Pred_W': grid_pred_wld_denorm[:, 0],
        'True_L': y_test_denorm[:, 1], 'Pred_L': grid_pred_wld_denorm[:, 1],
        'True_D': y_test_denorm[:, 2], 'Pred_D': grid_pred_wld_denorm[:, 2],
        'True_Class_ID': y_shape_test, 'Pred_Class_ID': grid_pred_shape,
        'True_Shape': [unique_shapes[s] for s in y_shape_test],
        'Pred_Shape': [unique_shapes[s] for s in grid_pred_shape]
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


# Save final results summary
results_txt_path = os.path.join(OUTPUT_DIR, "summary_results.txt")
res_lines = []
res_lines.append("=" * 80 + "\n")
res_lines.append("FINAL RESULTS - NO PINN MODEL (Q1 STANDARDS)\n")
res_lines.append("=" * 80 + "\n\n")

res_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
res_lines.append(f"Total training time: {format_time(time.time() - grid_search_start)}\n\n")

res_lines.append("CONFIGURATION:\n")
res_lines.append("-" * 80 + "\n")
res_lines.append(f"  Batch size:      {BATCH_SIZE}\n")
res_lines.append(f"  Epochs:          {TRAINING_EPOCHS}\n")
res_lines.append(f"  Learning rate:   {LEARNING_RATE}\n")
res_lines.append(f"  Random state:    {RANDOM_STATE}\n")
res_lines.append(f"  Loss formula:    L_clf + ((L_w + L_l + L_d)/3)\n")
res_lines.append(f"  Device:          {device}\n\n")

res_lines.append("METRICS:\n")
res_lines.append("-" * 80 + "\n")
res_lines.append(f"  Classification Accuracy: {final_result['clf_acc']*100.0:.2f}%\n")
res_lines.append(f"  Balanced Accuracy:       {final_result['clf_balanced_acc']*100.0:.2f}%\n")
res_lines.append(f"  F1-Macro:                {final_result['clf_f1_macro']*100.0:.2f}%\n")
res_lines.append(f"  MCC:                     {final_result['clf_mcc']:.4f}\n")
res_lines.append(f"  Average MAE:             {final_result['mae_avg']:.4f} mm\n")
res_lines.append(f"  Average RMSE:            {final_result['rmse_avg']:.4f} mm\n")
res_lines.append(f"  Average R²:              {final_result['r2_avg']:.4f}\n")
res_lines.append(f"  Average NRMSE:           {final_result['nrmse_avg']:.2f}%\n\n")

res_lines.append("DATA NORMALIZATION:\n")
res_lines.append("-" * 80 + "\n")
res_lines.append("  X_scaler (StandardScaler): X_scaler.pkl\n")
res_lines.append("  y_scaler (divide-by-max):  y_scaler.pkl\n")

_atomic_write_text(''.join(res_lines), results_txt_path)
print(f"\n[OK] Final results saved to: {results_txt_path}")

print(f"\n{'='*70}")
print("TRAINING COMPLETED SUCCESSFULLY")
print(f"{'='*70}\n")
