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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, classification_report, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score, f1_score
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
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'xiong_2023_baseline.py'
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from library_functions import Load_Data_With_Labels
except ImportError as e:
    print(f"[ERROR] Could not find 'library_functions.py': {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

REP_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REP_CODE_DIR not in sys.path:
    sys.path.insert(0, REP_CODE_DIR)

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
    "PINN_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "Outputs_xiong_pinn")
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
#   L = ((L_w + L_l + L_d)/3) + alpha * L_pinn
# Recommended default alpha from recent sweep analysis.
ALPHA_INIT = float(os.environ.get("ALPHA_INIT", "1.0")) * 1000.0
if ALPHA_INIT < 0:
    raise ValueError("ALPHA_INIT must be >= 0")
if ALPHA_INIT == 0:
    print("[INFO] ALPHA_INIT=0: Physics loss term disabled (running in regular regression mode without PINN)")
PINN_ACTIVATION_EPOCH = int(os.environ.get("PINN_ACTIVATION_EPOCH", "0"))
if PINN_ACTIVATION_EPOCH < 0:
    raise ValueError("PINN_ACTIVATION_EPOCH must be >= 0")
PINN_BASE_GRAD_CLIP_NORM = float(os.environ.get("PINN_BASE_GRAD_CLIP_NORM", "1.0"))
if PINN_BASE_GRAD_CLIP_NORM <= 0:
    raise ValueError("PINN_BASE_GRAD_CLIP_NORM must be > 0")

# Keep per-epoch verbose logs off by default so tqdm output stays readable.
EPOCH_VERBOSE_LOG = False

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

# PINN loss type: MSE (Mean Squared Error between B_true and B_pred)
PINN_LOSS_TYPE = os.environ.get("PINN_LOSS_TYPE", "mse").strip()

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


def _build_regression_metrics_by_target_df(y_true_wld, y_pred_wld, target_names, train_percent, pinn_alpha=0.0, pinn_active=False):
    rows = []
    for idx, target_name in enumerate(target_names):
        y_true_col = y_true_wld[:, idx]
        y_pred_col = y_pred_wld[:, idx]
        err = y_pred_col - y_true_col
        mse = mean_squared_error(y_true_col, y_pred_col)
        rmse = float(np.sqrt(mse))
        true_range = float(np.max(y_true_col) - np.min(y_true_col))
        nrmse = float(rmse / true_range * 100.0) if true_range > 1e-12 else float('nan')
        max_err = float(max_error(y_true_col, y_pred_col))
        rows.append({
            'target_index': idx,
            'target': target_name,
            'mae': float(mean_absolute_error(y_true_col, y_pred_col)),
            'mse': float(mse),
            'rmse': rmse,
            'r2': float(r2_score(y_true_col, y_pred_col)),
            'max_error': max_err,
            'nrmse': nrmse,
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
        if metric_name.endswith('_avg'):
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
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "").strip() or None

if RESUME_CHECKPOINT is not None:
    ckpt_abs = os.path.normcase(os.path.abspath(RESUME_CHECKPOINT))
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
print(f"[CONFIG] Loss mode: L = ((L_w + L_l + L_d)/3) + alpha*L_pinn (Regression only)")
print("[CONFIG] PINN update schedule: unified single step per batch (Xiong et al. 2023)")
print("[CONFIG] PINN physics formula: MSE = mean((B_true - B_pred)^2)")
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
        header = "epoch,train_reg_loss,train_data_loss,val_reg_loss,val_data_loss\n"
        _atomic_write_text(header, DATA_LOSS_LOG_PATH)

    # Unified epoch_loss_log.csv
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
    train_reg_loss,
    train_data_loss,
    val_reg_loss,
    val_data_loss,
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
            f"{epoch},{train_reg_loss:.12g},{train_data_loss:.12g},"
            f"{val_reg_loss:.12g},{val_data_loss:.12g}\n"
        )

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
    """Normalized Root Mean Squared Error (range-normalized, %)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = np.max(y_true) - np.min(y_true)
    if y_range < 1e-8:
        return 0.0
    return (rmse / y_range) * 100.0

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

def plot_fig1_training_curves(history, output_dir, pinn_activated_epoch=None):
    """fig1_training_curves.png - IEEE Publication Quality Training Curves (Regression Only)"""
    _set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    markevery = max(1, len(epochs) // 10)
    
    # Subplot (a): Total & Data Regression Loss
    axes[0].plot(epochs, history.get('train_loss', []), label='Train Reg Loss', color='#1f77b4', linewidth=1.8, marker='o', markersize=4, markevery=markevery)
    axes[0].plot(epochs, history.get('val_loss', []), label='Val Reg Loss', color='#d62728', linestyle='--', linewidth=1.8, marker='s', markersize=4, markevery=markevery)
    if pinn_activated_epoch and pinn_activated_epoch <= len(epochs):
        axes[0].axvline(x=pinn_activated_epoch, color='#2ca02c', linestyle=':', linewidth=1.8, label='PINN Active')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('(a) Regression Loss Convergence', fontweight='bold', pad=8)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    axes[0].grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    
    # Subplot (b): PINN Physics Loss
    phys_losses = history.get('physics_loss', [])
    if len(phys_losses) > 0 and max(phys_losses) > 0:
        axes[1].plot(epochs, phys_losses, label='Physics Loss', color='#9467bd', linewidth=1.8, marker='D', markersize=3.5, markevery=markevery)
        if pinn_activated_epoch and pinn_activated_epoch <= len(epochs):
            axes[1].axvline(x=pinn_activated_epoch, color='#2ca02c', linestyle=':', linewidth=1.8, label='PINN Active')
    else:
        axes[1].text(0.5, 0.5, 'Physics Loss: Inactive (Pure Data-Driven)', ha='center', va='center', color='gray')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Physics Loss', fontweight='bold')
    axes[1].set_title('(b) PINN Residual Loss', fontweight='bold', pad=8)
    axes[1].legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    axes[1].grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(output_dir, 'fig1_training_curves.png'), dpi=300)
    plt.close(fig)

def plot_fig3_regression_scatter(y_true_wld, y_pred_wld, output_dir):
    """fig3_regression_scatter.png - IEEE Ground Truth vs Prediction 1x3 Grid"""
    _set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics_names = ['Width ($W$, mm)', 'Length ($L$, mm)', 'Depth ($D$, mm)']
    colors = ['#004c6d', '#c35100', '#4a2c5d']
    
    for i in range(3):
        ax = axes[i]
        yt = y_true_wld[:, i]
        yp = y_pred_wld[:, i]
        
        ax.scatter(yt, yp, alpha=0.55, s=25, color=colors[i], edgecolors='white', linewidth=0.3, label='Predictions')
        
        min_v = min(yt.min(), yp.min())
        max_v = max(yt.max(), yp.max())
        ax.plot([min_v, max_v], [min_v, max_v], color='#d62728', linestyle='--', linewidth=1.8, label='Ideal ($y=x$)')
        
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        nrmse = float(calculate_nrmse(yt, yp))
        max_err = float(max_error(yt, yp))
        
        # Professional IEEE Text box metrics display
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


def save_regression_plots(y_true_wld, y_pred_wld, output_dir, metrics_names=['Width (W)', 'Length (L)', 'Depth (D)'], prefix='', title_prefix='', y_shape=None, unique_shapes=None, y_scaler=None):
    """Save W/L/D regression scatter plots with optional denormalization."""
    colors = ['green', 'orange', 'purple']
    
    # Denormalize if scaler provided
    if y_scaler is not None:
        if np.any(np.isnan(y_true_wld)) or np.any(np.isnan(y_pred_wld)):
            print(f"[WARNING] NaN detected in input data before denormalization")
            y_pred_wld = np.where(np.isnan(y_pred_wld), np.nan_to_num(np.nanmean(y_pred_wld, axis=0), nan=0.0), y_pred_wld)
            y_true_wld = np.where(np.isnan(y_true_wld), np.nan_to_num(np.nanmean(y_true_wld, axis=0), nan=0.0), y_true_wld)
        
        y_true_wld_denorm = y_scaler.inverse_transform(y_true_wld)
        y_pred_wld_denorm = y_scaler.inverse_transform(y_pred_wld)
        
        if np.any(np.isnan(y_pred_wld_denorm)):
            print(f"[WARNING] NaN detected after denormalization")
            y_pred_wld_denorm = np.where(np.isnan(y_pred_wld_denorm), np.nan_to_num(np.nanmean(y_pred_wld_denorm, axis=0), nan=0.0), y_pred_wld_denorm)
    else:
        y_true_wld_denorm = y_true_wld
        y_pred_wld_denorm = y_pred_wld
    
    for i, (name, color) in enumerate(zip(metrics_names, colors)):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_true_wld_denorm[:, i], y_pred_wld_denorm[:, i], 
                  alpha=0.5, s=30, color=color, edgecolors='k', linewidth=0.5)
        
        min_val = min(y_true_wld_denorm[:, i].min(), y_pred_wld_denorm[:, i].min())
        max_val = max(y_true_wld_denorm[:, i].max(), y_pred_wld_denorm[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
        
        y_true_col = y_true_wld_denorm[:, i]
        y_pred_col = y_pred_wld_denorm[:, i]
        
        valid_mask = ~(np.isnan(y_true_col) | np.isnan(y_pred_col))
        if np.sum(valid_mask) < len(y_true_col):
            y_true_col = y_true_col[valid_mask]
            y_pred_col = y_pred_col[valid_mask]
        
        if len(y_true_col) > 0:
            mae = float(mean_absolute_error(y_true_col, y_pred_col))
            rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
            r2 = float(r2_score(y_true_col, y_pred_col))
            nrmse = float(calculate_nrmse(y_true_col, y_pred_col))
            max_err = float(max_error(y_true_col, y_pred_col))
        else:
            mae = rmse = r2 = nrmse = max_err = np.nan
        
        ax.set_xlabel('True Value (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted Value (mm)', fontsize=13, fontweight='bold')
        title_str = f'{title_prefix}{name} Regression\nMAE: {mae:.4f} mm | RMSE: {rmse:.4f} mm | R²: {r2:.4f} | NRMSE: {nrmse:.2f}% | Max Err: {max_err:.4f} mm'
        ax.set_title(title_str, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(alpha=0.3)
        
        filename = f"{prefix}regression_{name.split()[0].lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
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
        
        # Physics loss formula (MSE between B_true and B_pred):
        diff = H_true_t - H_pred
        physics_loss = torch.mean(diff * diff)

        log_msg(f"Physics Loss (MSE): {physics_loss.item():.8f}")
        
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
# LOSS HELPERS
# ============================================================================


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


def compute_combined_loss(w_loss, l_loss, d_loss, physics_loss,
                         alpha=1.0):
    """
    Compute combined loss used by the active training path (Regression only).

    Formula:
        total = ((w + l + d)/3) + alpha * physics

    Args:
        w_loss, l_loss, d_loss: Regression losses
        physics_loss: Physics (PINN) loss
        alpha: Weight for PINN loss (user controlled)
    Returns:
        total_loss: Total loss
        reg_loss: Average regression loss ((W+L+D)/3)
    """
    reg_loss = (w_loss + l_loss + d_loss) / 3.0
    total_loss = reg_loss + alpha * physics_loss
    return total_loss, reg_loss




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
# MODEL DEFINITION (XIONG ET AL. 2023 - PURE REGRESSION MLP)
# ============================================================================

class RegressionMLP_PINN(nn.Module):
    """
    MLP-based regression network for joint W/L/D defect size estimation.

    Exact architecture faithful to Xiong et al. 2023:
    "Magnetic flux leakage defect size estimation method based on physics-informed neural network"
    (Phil. Trans. R. Soc. A 382: 20220387, Table 3):
    - Input: flattened sensor feature vector (INPUT_DIM = 2048)
    - Hidden Layer 1: 64 neurons, Tanh activation
    - Hidden Layer 2: 32 neurons, Tanh activation
    - Hidden Layer 3: 16 neurons, Tanh activation
    - Output Layer: 3 neurons (W, L, D), Softplus activation (strictly positive)
    - No BatchNorm, No Dropout (pure MLP formulation as in Table 3)
    """
    INPUT_DIM = 32 * 32 * 2  # 2048

    def __init__(self, input_dim=None, *args, **kwargs):
        super(RegressionMLP_PINN, self).__init__()
        in_dim = input_dim or self.INPUT_DIM

        # ===== 3 HIDDEN LAYERS (64 -> 32 -> 16) WITH TANH [XIONG ET AL. 2023 TABLE 3] =====
        self.backbone = nn.Sequential(
            # Hidden Layer 1: in_dim -> 64
            nn.Linear(in_dim, 64),
            nn.Tanh(),

            # Hidden Layer 2: 64 -> 32
            nn.Linear(64, 32),
            nn.Tanh(),

            # Hidden Layer 3: 32 -> 16
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # ===== OUTPUT LAYER (16 -> 3) WITH SOFTPLUS [XIONG ET AL. 2023 TABLE 3] =====
        self.reg_head = nn.Sequential(
            nn.Linear(16, 3),
            nn.Softplus(),
        )

    def forward(self, x):
        """Forward pass. x shape: (N, 2048) or (N, 32, 32, 2)."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        shared_feat = self.backbone(x)
        y_pred_wld = self.reg_head(shared_feat)
        return y_pred_wld

# Backward-compatible aliases so existing checkpoint loaders still resolve the name.
MultitaskMLP_PINN = RegressionMLP_PINN
ImprovedMultimodelNet = RegressionMLP_PINN

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

# Convert X to PyTorch tensors — flatten to (N, 2048) for MLP input
# From (N, H, W, C) = (N, 32, 32, 2) -> (N, 2048)
X_train_tensor = torch.FloatTensor(X_train_final.reshape(n_train, -1))  # X: flattened field
X_val_tensor = torch.FloatTensor(X_val_final.reshape(n_val, -1))        # X: flattened field
X_test_tensor = torch.FloatTensor(X_test_final.reshape(n_test, -1))     # X: flattened field

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
# ===============================================================print(f"\nBUILDING REGRESSION MLP-PINN MODEL (Xiong et al. 2023)")

model = RegressionMLP_PINN().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ============================================================================
# 7. LOSS FUNCTIONS
# ============================================================================

print(f"\nCONFIGURING LOSS FUNCTIONS")

criterion_reg = nn.MSELoss()

# ===== OPTIMIZERS (XIONG ET AL. 2023 TABLE 3: ADAM (lr=0.001) + L-BFGS (lr=0.8)) =====
ADAM_LR = float(os.environ.get("ADAM_LR", "0.001"))
LBFGS_LR = float(os.environ.get("LBFGS_LR", "0.8"))
ADAM_EPOCHS = int(os.environ.get("ADAM_EPOCHS", "100"))
OPTIMIZER_TYPE = os.environ.get("OPTIMIZER_TYPE", "adam_lbfgs").strip().lower()

optimizer_adam = optim.Adam(model.parameters(), lr=ADAM_LR)
optimizer_lbfgs = optim.LBFGS(
    model.parameters(),
    lr=LBFGS_LR,
    max_iter=20,
    history_size=50,
    line_search_fn="strong_wolfe",
)
optimizer = optimizer_adam
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adam, mode='min', factor=0.5, patience=15, min_lr=1e-6
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


def load_checkpoint(filepath, model, optimizer, scheduler):
    """
    Load checkpoint and restore training state.
    """
    print(f"\n[RESUME] Loading checkpoint from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = _torch_load_checkpoint(filepath)
    
    # Load model weights (strict=False handles older checkpoints gracefully)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    print(f"  [OK] Model weights loaded")
    
    # Load optimizer state if matching
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  [OK] Optimizer state loaded")
    except Exception:
        pass
    
    try:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  [OK] Scheduler state loaded")
    except Exception:
        pass
    
    # Get training state
    start_epoch = checkpoint.get('epoch', 0) + 1 if isinstance(checkpoint, dict) else 1
    history = checkpoint.get('history', {}) if isinstance(checkpoint, dict) else {}
    best_val_loss = checkpoint.get('best_val_loss', float('inf')) if isinstance(checkpoint, dict) else float('inf')
    
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        cfg = checkpoint['config']
        print(f"  [INFO] Checkpoint config: batch_size={cfg.get('batch_size')}, lr={cfg.get('learning_rate')}")
    
    print(f"  [OK] Resuming from epoch {start_epoch}")
    print(f"  [OK] Best val loss so far: {best_val_loss:.6f}")
    
    return start_epoch, history, best_val_loss

print(f"\nLoss configuration:")
print(f"  Regression losses: MSE (W, L, D)")
print(f"  Physics loss: Custom PINN constraint")
print(f"  Weighting method: fixed warm-up then PINN activation")
print(f"    └─ Loss formula (active): L = ((L_w + L_l + L_d)/3) + alpha*L_pinn")
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
                          X_test_field=None, y_test=None, y_pred_wld=None, y_shape_test=None, y_scaler=None):
    """Save checkpoint figures (loss curves and metrics)."""
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
    
    # Plot 2: Width Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.get('train_w_loss', []), label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history.get('val_w_loss', []), label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Width Regression Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_width.png"), dpi=150)
    plt.close()
    
    # Plot 3: Length Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.get('train_l_loss', []), label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history.get('val_l_loss', []), label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Length Regression Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_length.png"), dpi=150)
    plt.close()
    
    # Plot 4: Depth Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.get('train_d_loss', []), label='Train', linewidth=2.5, marker='o', markersize=3, alpha=0.8, color=TRAIN_LINE_COLOR)
    plt.plot(history.get('val_d_loss', []), label='Validation', linewidth=2.5, marker='s', markersize=3, alpha=0.8, color=VAL_LINE_COLOR)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Depth Regression Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_depth.png"), dpi=150)
    plt.close()
    
    # Plot 5: Physics Loss
    if history.get('physics_loss') and max(history['physics_loss']) > 0:
        plt.figure(figsize=(10, 6))
        phys_losses = np.array(history['physics_loss'])
        active_idx = np.where(phys_losses > 0)[0]
        
        if len(active_idx) > 0:
            active_epochs = active_idx + 1
            active_losses = phys_losses[active_idx]
            
            plt.plot(active_epochs, active_losses, label='Physics Loss', linewidth=2.5, marker='o', markersize=3, color='red', alpha=0.8)
            plt.xlabel('Epoch', fontsize=12, fontweight='bold')
            plt.ylabel('Physics Loss (MSE)', fontsize=12, fontweight='bold')
            plt.title(f'Physics Loss (Checkpoint - Epoch {epoch+1})', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_loss_physics.png"), dpi=150)
        plt.close()
        
    # Plot 6: Combined 2x2 Summary Dashboard
    if y_test is not None and y_pred_wld is not None:
        if y_scaler is not None:
            _y_pred_safe = np.where(np.isnan(y_pred_wld), np.nan_to_num(np.nanmean(y_pred_wld, axis=0), nan=0.0), y_pred_wld)
            _y_test_safe = np.where(np.isnan(y_test), np.nan_to_num(np.nanmean(y_test, axis=0), nan=0.0), y_test)
            y_test_denorm = y_scaler.inverse_transform(_y_test_safe)
            y_pred_denorm = y_scaler.inverse_transform(_y_pred_safe)
        else:
            y_test_denorm = np.nan_to_num(y_test, nan=0.0)
            y_pred_denorm = np.nan_to_num(y_pred_wld, nan=0.0)
            
        metrics_names = ['Width (W)', 'Length (L)', 'Depth (D)']
        grid_mae, grid_rmse, grid_r2, grid_nrmse, grid_max_err = [], [], [], [], []
        for i in range(3):
            grid_mae.append(float(mean_absolute_error(y_test_denorm[:, i], y_pred_denorm[:, i])))
            grid_rmse.append(float(np.sqrt(mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i]))))
            grid_r2.append(float(r2_score(y_test_denorm[:, i], y_pred_denorm[:, i])))
            grid_nrmse.append(float(calculate_nrmse(y_test_denorm[:, i], y_pred_denorm[:, i])))
            grid_max_err.append(float(max_error(y_test_denorm[:, i], y_pred_denorm[:, i])))
            
        avg_mae = np.mean(grid_mae)
        avg_rmse = np.mean(grid_rmse)
        avg_r2 = np.mean(grid_r2)
        avg_nrmse = np.mean(grid_nrmse)
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # [0,0] Total Loss
        axes[0, 0].plot(history.get('train_loss', []), label='Train', linewidth=2, color=TRAIN_LINE_COLOR)
        axes[0, 0].plot(history.get('val_loss', []), label='Val', linewidth=2, color=VAL_LINE_COLOR)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # [0,1] Average Regression Loss
        train_w = history.get('train_w_loss', [])
        train_l = history.get('train_l_loss', [])
        train_d = history.get('train_d_loss', [])
        val_w = history.get('val_w_loss', [])
        val_l = history.get('val_l_loss', [])
        val_d = history.get('val_d_loss', [])
        if train_w and train_l and train_d:
            tr_avg = [(train_w[i] + train_l[i] + train_d[i]) / 3.0 for i in range(len(train_w))]
            axes[0, 1].plot(tr_avg, label='Train Reg Avg', linewidth=2, color=TRAIN_LINE_COLOR)
        if val_w and val_l and val_d:
            vl_avg = [(val_w[i] + val_l[i] + val_d[i]) / 3.0 for i in range(len(val_w))]
            axes[0, 1].plot(vl_avg, label='Val Reg Avg', linewidth=2, color=VAL_LINE_COLOR)
        axes[0, 1].set_title('Regression Loss (W+L+D)/3', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # [1,0] Physics Loss
        phys_losses = np.array(history.get('physics_loss', []))
        if len(phys_losses) > 0 and max(phys_losses) > 0:
            active_idx = np.where(phys_losses > 0)[0]
            if len(active_idx) > 0:
                axes[1, 0].plot(active_idx + 1, phys_losses[active_idx], label='PINN Loss', linewidth=2, color='red')
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
                            
        # [1,1] Test Set Metrics table
        metrics_text = "Test Set Regression Metrics (Q1 Standard):\n\n"
        metrics_text += f"Avg MAE:   {avg_mae:.4f} mm\n"
        metrics_text += f"Avg RMSE:  {avg_rmse:.4f} mm\n"
        metrics_text += f"Avg R²:    {avg_r2:.4f}\n"
        metrics_text += f"Avg NRMSE: {avg_nrmse:.2f}%\n\n"
        metrics_text += "By Output Target:\n"
        for i, name in enumerate(metrics_names):
            metrics_text += f"{name}: MAE={grid_mae[i]:.3f} | RMSE={grid_rmse[i]:.3f} | R²={grid_r2[i]:.3f} | NRMSE={grid_nrmse[i]:.1f}%\n"
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        _atomic_save_fig(plt, os.path.join(checkpoint_dir, f"ckpt_{epoch_str}_training_curves_summary.png"), dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# 9. TRAINING LOOP
# ============================================================================

print(f'\nTRAINING WITH PINN WARM-UP (Xiong et al. 2023 — unified single-step)')
print(f'  Warm-up (epochs 1-{PINN_ACTIVATION_EPOCH}): L = L_reg (PINN inactive)')
print(f'  PINN phase (epoch {PINN_ACTIVATION_EPOCH + 1}+): L = L_reg + alpha*L_pinn (single optimizer step per batch)')
print(f'  Early stopping: after PINN activation, stop when PINN loss plateaus (patience=300)\n')

# Initialize training state
start_epoch = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'physics_loss': [],
    'checkpoint_score': [],
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

for _history_key in [
    'train_loss',
    'val_loss',
    'physics_loss',
    'checkpoint_score',
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

use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)
print(f"[CONFIG] AMP (Mixed Precision): {'ENABLED' if use_amp else 'DISABLED'}")

import time
start_training = time.time()
DEBUG_TIMING = True
TIMING_EVERY = 10

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
        pinn_activated_epoch = epoch + 1
        best_checkpoint_score = float('inf')
        print(f"\n>>> PINN ACTIVATED at Epoch {pinn_activated_epoch} <<<\n")
        print("[CKPT] Reset best checkpoint tracking. From now on best is selected in PINN phase.")

    # Active optimizer selection (Xiong et al. 2023: Stage 1 Adam -> Stage 2 L-BFGS)
    is_lbfgs_phase = (OPTIMIZER_TYPE == "lbfgs") or (OPTIMIZER_TYPE == "adam_lbfgs" and (epoch + 1) > ADAM_EPOCHS)
    current_optimizer = optimizer_lbfgs if is_lbfgs_phase else optimizer_adam

    # ===== TRAINING =====
    model.train()
    train_w_loss = 0.0
    train_l_loss = 0.0
    train_d_loss = 0.0
    train_physics_loss = 0.0
    epoch_pinn_batches_applied = 0
    epoch_pinn_samples_applied = 0
    epoch_pinn_loss_sum = 0.0
    epoch_pinn_grad_norm = 0.0
    pinn_loss_active = physics_available_for_training and (not is_before_pinn) and (ALPHA_INIT > 0)
    
    alpha = ALPHA_INIT
    
    pbar = tqdm(
        train_loader,
        desc=f'Epoch {epoch+1}/{EPOCHS} [{"L-BFGS" if is_lbfgs_phase else "Adam"}]',
        unit='batch',
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    processed_train_batches = 0
    for batch_idx, (batch_X, batch_y_wld, batch_y_shape) in enumerate(pbar):
        t_batch_start = time.perf_counter()
        if batch_X.size(0) < 2:
            continue
        batch_X = batch_X.to(device)
        batch_y_wld = batch_y_wld.to(device)
        batch_y_shape = batch_y_shape.to(device)
        t_data = time.perf_counter()

        alpha = ALPHA_INIT
        amp_enabled = use_amp and not pinn_loss_active and not is_lbfgs_phase

        with autocast('cuda', enabled=amp_enabled):
            y_pred_wld = model(batch_X.float() if pinn_loss_active else batch_X)

            if torch.isnan(y_pred_wld).any():
                print(f"\n[CRITICAL] Model outputs NaN detected (Epoch {epoch+1}, Batch {batch_idx}). Stop training!")
                nan_detected = True
                break

            w_loss, l_loss, d_loss, avg_reg_loss = compute_regression_losses(
                y_pred_wld, batch_y_wld, criterion_reg
            )
            data_loss = avg_reg_loss

        t_forward = time.perf_counter()

        # Compute physics loss per sample in this batch (if PINN active)
        batch_physics_loss = torch.tensor(0.0, device=device)
        batch_pinn_count = 0
        if pinn_loss_active:
            sample_phys_losses = []
            for idx in range(batch_X.size(0)):
                field_value = batch_X[idx].view(32, 32, 2)[:, :, 0]
                true_shape_idx = int(batch_y_shape[idx].item())
                shape_name = unique_shapes[true_shape_idx]

                try:
                    if shape_name in SHAPE_MAP_TRAINING:
                        phys_loss_sample = compute_physics_loss_autograd(
                            field_value,
                            y_pred_wld[idx, 0],
                            y_pred_wld[idx, 1],
                            y_pred_wld[idx, 2],
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
                        phys_loss_sample = y_pred_wld[idx].sum() * 0.0
                except Exception:
                    phys_loss_sample = y_pred_wld[idx].sum() * 0.0

                sample_phys_losses.append(phys_loss_sample)

            if sample_phys_losses:
                batch_physics_loss = torch.stack(sample_phys_losses).mean()
                batch_pinn_count = len(sample_phys_losses)
                epoch_pinn_loss_sum += float(batch_physics_loss.detach().item()) * batch_pinn_count
                epoch_pinn_batches_applied += 1
                epoch_pinn_samples_applied += batch_pinn_count

        t_physics = time.perf_counter()

        total_loss = data_loss + alpha * batch_physics_loss

        if torch.isnan(total_loss):
            print(f"\n[CRITICAL] NaN total_loss detected (Epoch {epoch+1}, Batch {batch_idx}). Stop training!")
            nan_detected = True
            break

        if is_lbfgs_phase:
            def closure():
                current_optimizer.zero_grad()
                y_pred_c = model(batch_X.float())
                _, _, _, reg_loss_c = compute_regression_losses(y_pred_c, batch_y_wld, criterion_reg)
                phys_loss_c = torch.tensor(0.0, device=device)
                if pinn_loss_active:
                    c_phys_losses = []
                    for idx_c in range(batch_X.size(0)):
                        f_val = batch_X[idx_c].view(32, 32, 2)[:, :, 0]
                        s_idx = int(batch_y_shape[idx_c].item())
                        s_name = unique_shapes[s_idx]
                        if s_name in SHAPE_MAP_TRAINING:
                            try:
                                p_loss = compute_physics_loss_autograd(
                                    f_val, y_pred_c[idx_c, 0], y_pred_c[idx_c, 1], y_pred_c[idx_c, 2],
                                    s_name, SHAPE_MAP_TRAINING, y_scaler, X_scaler=X_scaler,
                                    w_true=batch_y_wld[idx_c, 0], l_true=batch_y_wld[idx_c, 1], d_true=batch_y_wld[idx_c, 2]
                                )
                                c_phys_losses.append(p_loss)
                            except Exception:
                                pass
                    if c_phys_losses:
                        phys_loss_c = torch.stack(c_phys_losses).mean()
                loss_c = reg_loss_c + alpha * phys_loss_c
                loss_c.backward()
                return loss_c
            current_optimizer.step(closure)
        elif amp_enabled:
            current_optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(current_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(current_optimizer)
            scaler.update()
            current_optimizer.zero_grad(set_to_none=True)
        else:
            current_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            clip_norm = max(1.0, alpha * PINN_BASE_GRAD_CLIP_NORM) if (pinn_loss_active and alpha > 0) else 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            try:
                epoch_pinn_grad_norm = max(epoch_pinn_grad_norm, float(grad_norm.detach().cpu().item()))
            except Exception:
                epoch_pinn_grad_norm = max(epoch_pinn_grad_norm, float(grad_norm))
            current_optimizer.step()
            current_optimizer.zero_grad(set_to_none=True)

        t_backward = time.perf_counter()
        
        train_w_loss += w_loss.item()
        train_l_loss += l_loss.item()
        train_d_loss += d_loss.item()
        processed_train_batches += 1

        seen_train_batches = max(1, processed_train_batches)
        running_train_w = train_w_loss / seen_train_batches
        running_train_l = train_l_loss / seen_train_batches
        running_train_d = train_d_loss / seen_train_batches
        running_train_reg = (running_train_w + running_train_l + running_train_d) / 3.0

        train_postfix = {
            'trRegL': f"{running_train_reg:.4f}",
            'W': f"{running_train_w:.4f}",
            'L': f"{running_train_l:.4f}",
            'D': f"{running_train_d:.4f}",
        }
        if pinn_loss_active and epoch_pinn_samples_applied > 0:
            running_pinn = epoch_pinn_loss_sum / epoch_pinn_samples_applied
            train_postfix['pL'] = f"{running_pinn:.4e}"
        pbar.set_postfix(train_postfix)
    
    if nan_detected:
        break

    if processed_train_batches == 0:
        raise RuntimeError(
            "No valid training batch processed. Increase TRAIN_PERCENT or reduce BATCH_SIZE."
        )

    train_w_loss /= processed_train_batches
    train_l_loss /= processed_train_batches
    train_d_loss /= processed_train_batches
    if epoch_pinn_samples_applied > 0:
        train_physics_loss = epoch_pinn_loss_sum / epoch_pinn_samples_applied
    else:
        train_physics_loss = 0.0
    train_reg_loss_avg = (train_w_loss + train_l_loss + train_d_loss) / 3.0
    train_total_loss = train_reg_loss_avg
    
    # ===== VALIDATION =====
    model.eval()
    val_w_loss = 0.0
    val_l_loss = 0.0
    val_d_loss = 0.0
    
    val_pbar = tqdm(
        val_loader,
        desc=f'Val {epoch+1}/{EPOCHS}',
        unit='batch',
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    with torch.no_grad():
        for val_batch_idx, (batch_X, batch_y_wld, _) in enumerate(val_pbar):
            batch_X = batch_X.to(device)
            batch_y_wld = batch_y_wld.to(device)
            
            y_pred_wld = model(batch_X)
            
            w_loss, l_loss, d_loss, avg_reg_loss = compute_regression_losses(
                y_pred_wld, batch_y_wld, criterion_reg
            )
            
            val_w_loss += w_loss.item()
            val_l_loss += l_loss.item()
            val_d_loss += d_loss.item()

            seen_val_batches = val_batch_idx + 1
            running_val_w = val_w_loss / seen_val_batches
            running_val_l = val_l_loss / seen_val_batches
            running_val_d = val_d_loss / seen_val_batches
            running_val_reg = (running_val_w + running_val_l + running_val_d) / 3.0
            val_pbar.set_postfix({
                'trL': f"{train_total_loss:.4f}",
                'vL': f"{running_val_reg:.4f}",
                'pv': f"{last_val_total_loss:.4f}" if last_val_total_loss is not None else 'n/a'
            })
    
    val_w_loss /= len(val_loader)
    val_l_loss /= len(val_loader)
    val_d_loss /= len(val_loader)
    val_reg_loss_avg = (val_w_loss + val_l_loss + val_d_loss) / 3.0
    val_total_loss = val_reg_loss_avg

    # Kiểm tra bất thường sụp đổ loss hoặc NaN
    if (
        np.isnan(val_total_loss)
        or np.isinf(val_total_loss)
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
        checkpoint_score = val_total_loss + alpha * current_physics_loss
        checkpoint_score_name = "val_reg_loss + alpha*train_physics_loss"
    else:
        checkpoint_score = val_total_loss
        checkpoint_score_name = "val_reg_loss"

    if pinn_activated and pinn_activated_epoch is not None and (ALPHA_INIT > 0):
        if current_physics_loss < pinn_stop_best_loss * 0.995:
            pinn_stop_best_loss = current_physics_loss
            pinn_stop_counter = 0
        else:
            pinn_stop_counter += 1

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
    
    epoch_time = time.time() - epoch_start_time
    elapsed_total = time.time() - start_training
    eta_remaining = (elapsed_total / (epoch + 1)) * (EPOCHS - epoch - 1)
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    if EPOCH_VERBOSE_LOG:
        effective_train_objective = train_total_loss + (
            alpha * train_physics_loss if (pinn_activated and pinn_activated_epoch is not None) else 0.0
        )
        print(f"\n{'='*100}")
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Time: {format_time(epoch_time)} | ETA: {format_time(eta_remaining)}")
        print(f"  Train → W: {train_w_loss:.4f} | L: {train_l_loss:.4f} | D: {train_d_loss:.4f} | Avg: {train_reg_loss_avg:.4f}")
        print(f"  Val   → W: {val_w_loss:.4f} | L: {val_l_loss:.4f} | D: {val_d_loss:.4f} | Avg: {val_reg_loss_avg:.4f}")
        print(f"  Weights → alpha (PINN term): {alpha:.4f} | Loss: {effective_train_objective:.4f}")
        print(f"  Checkpoint metric ({checkpoint_score_name}): {checkpoint_score:.6f}")
    
    # Record history
    history['train_loss'].append(train_total_loss)
    history['val_loss'].append(val_total_loss)
    history['physics_loss'].append(train_physics_loss)
    history['checkpoint_score'].append(checkpoint_score)
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
        train_reg_loss=train_reg_loss_avg,
        train_data_loss=train_total_loss,
        val_reg_loss=val_reg_loss_avg,
        val_data_loss=val_total_loss,
    )

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
    print(f"[CRITICAL][NAN DETECTED] Xiong baseline training stopped early due to NaN detection!")
    print(f"[ROLLBACK] Reverting model to the last clean checkpoint (Epoch {last_valid_epoch}) as BEST checkpoint.")
    print("="*80 + "\n")
    if last_valid_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in last_valid_model_state.items()})
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_pytorch.pth'))

plot_fig1_training_curves(history, OUTPUT_DIR, pinn_activated_epoch)

# ============================================================================
# 10. EVALUATION (REGRESSION ONLY)
# ============================================================================

print(f"\nEVALUATING ON TEST SET (REGRESSION ONLY)")

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
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ck, strict=False)
        model.eval()

        pred_wld_list = []
        with torch.no_grad():
            for batch_X, _, _ in test_loader:
                batch_X = batch_X.to(device)
                pred_wld = model(batch_X)
                if torch.isnan(pred_wld).any():
                    raise ValueError('NaN in model outputs')
                pred_wld_list.append(pred_wld.cpu().numpy())

        tmp_pred_wld = np.concatenate(pred_wld_list)
        tmp_pred_wld_denorm = y_scaler.inverse_transform(tmp_pred_wld)
        if np.any(np.isnan(tmp_pred_wld_denorm)):
            print(f"[WARN] Checkpoint gives NaN after denorm, skipping: {candidate_path}")
            continue

        selected_model_path = candidate_path
        y_pred_wld = tmp_pred_wld
        y_pred_wld_denorm = tmp_pred_wld_denorm
        print(f"[OK] Selected checkpoint for Test: {selected_model_path}")
        break
    except Exception as e:
        print(f"[WARN] Checkpoint {candidate_path} produced invalid outputs: {e}")
        continue

if selected_model_path is None:
    print("[CRITICAL] Could not find a valid checkpoint for evaluation.")
    sys.exit(1)

y_test_denorm = y_test_scaler.inverse_transform(y_test_norm)

# Regression metrics (using DENORMALIZED values)
print(f"\n--- REGRESSION METRICS (Denormalized Values) ---")
metrics_names = ['Width (W)', 'Length (L)', 'Depth (D)']
metrics_targets = ['W', 'L', 'D']
test_metrics = {'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'max_error': [], 'nrmse': []}

for i in range(3):
    mae = float(mean_absolute_error(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    mse = float(mean_squared_error(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    max_err = float(max_error(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    nrmse = float(calculate_nrmse(y_test_denorm[:, i], y_pred_wld_denorm[:, i]))
    
    test_metrics['mae'].append(mae)
    test_metrics['mse'].append(mse)
    test_metrics['rmse'].append(rmse)
    test_metrics['r2'].append(r2)
    test_metrics['max_error'].append(max_err)
    test_metrics['nrmse'].append(nrmse)
    
    print(f"\n{metrics_names[i]}:")
    print(f"  MAE:       {mae:.4f} mm")
    print(f"  RMSE:      {rmse:.4f} mm")
    print(f"  R²:        {r2:.4f}")
    print(f"  Max Error: {max_err:.4f} mm")
    print(f"  NRMSE:     {nrmse:.2f}%")

mae_avg = float(np.mean(test_metrics['mae']))
rmse_avg = float(np.mean(test_metrics['rmse']))
r2_avg = float(np.mean(test_metrics['r2']))
nrmse_avg = float(np.mean(test_metrics['nrmse']))

regression_metrics_by_target_df = _build_regression_metrics_by_target_df(
    y_test_denorm,
    y_pred_wld_denorm,
    metrics_targets,
    TRAIN_PERCENT,
    pinn_alpha=ALPHA_INIT if physics_available_for_training else 0.0,
    pinn_active=physics_available_for_training,
)
regression_metrics_by_target_csv = os.path.join(OUTPUT_DIR, 'test_regression_metrics_by_target.csv')
_atomic_write_df(regression_metrics_by_target_df, regression_metrics_by_target_csv, index=False)
print(f"[OK] Regression metrics by W/L/D saved: {regression_metrics_by_target_csv}")

# Save Standalone Standard Summary Metrics and Predictions CSVs
overall_summary_df = pd.DataFrame([{
    'Model': 'Adapted 1D MLP-PINN (Xiong et al. 2023)',
    'Train_Percent': f"{TRAIN_PERCENT}%",
    'Seed': RANDOM_STATE,
    'Alpha': ALPHA_INIT,
    'Warmup': PINN_ACTIVATION_EPOCH,
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
    'Shape_Class': [unique_shapes[s] for s in y_shape_test]
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

# Save Publication Quality Visualizations (fig1, fig3)
plot_fig1_training_curves(history, OUTPUT_DIR, pinn_activated_epoch)
plot_fig3_regression_scatter(y_test_denorm, y_pred_wld_denorm, OUTPUT_DIR)

# ============================================================================
# PER-SHAPE REGRESSION ANALYSIS (Chi tiết cho từng loại vết nứt)
# ============================================================================

print(f"\n{'='*80}")
print("PER-SHAPE REGRESSION ANALYSIS (Q1 STANDARDS)")
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
    
    y_test_shape = y_test_denorm[shape_mask]
    y_pred_shape_wld = y_pred_wld_denorm[shape_mask]
    
    shape_metrics = {'mae': [], 'rmse': [], 'r2': [], 'max_error': [], 'nrmse': []}
    
    for i in range(3):
        mae = float(mean_absolute_error(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_test_shape[:, i], y_pred_shape_wld[:, i])))
        r2 = float(r2_score(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        max_err = float(max_error(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        nrmse = float(calculate_nrmse(y_test_shape[:, i], y_pred_shape_wld[:, i]))
        
        shape_metrics['mae'].append(mae)
        shape_metrics['rmse'].append(rmse)
        shape_metrics['r2'].append(r2)
        shape_metrics['max_error'].append(max_err)
        shape_metrics['nrmse'].append(nrmse)
    
    shape_mae_avg = float(np.mean(shape_metrics['mae']))
    shape_rmse_avg = float(np.mean(shape_metrics['rmse']))
    shape_r2_avg = float(np.mean(shape_metrics['r2']))
    shape_nrmse_avg = float(np.nanmean(shape_metrics['nrmse']))
    shape_max_err_avg = float(np.mean(shape_metrics['max_error']))
    
    print(f"  MAE Avg:   {shape_mae_avg:.4f} mm")
    print(f"  RMSE Avg:  {shape_rmse_avg:.4f} mm")
    print(f"  R² Avg:    {shape_r2_avg:.4f}")
    print(f"  NRMSE Avg: {shape_nrmse_avg:.2f}%")
    
    per_shape_results.append({
        'shape': shape_name,
        'n_samples': n_shape,
        'mae_w': float(shape_metrics['mae'][0]),
        'mae_l': float(shape_metrics['mae'][1]),
        'mae_d': float(shape_metrics['mae'][2]),
        'mae_avg': shape_mae_avg,
        'rmse_w': float(shape_metrics['rmse'][0]),
        'rmse_l': float(shape_metrics['rmse'][1]),
        'rmse_d': float(shape_metrics['rmse'][2]),
        'rmse_avg': shape_rmse_avg,
        'r2_w': float(shape_metrics['r2'][0]),
        'r2_l': float(shape_metrics['r2'][1]),
        'r2_d': float(shape_metrics['r2'][2]),
        'r2_avg': shape_r2_avg,
        'max_error_w': float(shape_metrics['max_error'][0]),
        'max_error_l': float(shape_metrics['max_error'][1]),
        'max_error_d': float(shape_metrics['max_error'][2]),
        'max_error_avg': shape_max_err_avg,
        'nrmse_w': float(shape_metrics['nrmse'][0]),
        'nrmse_l': float(shape_metrics['nrmse'][1]),
        'nrmse_d': float(shape_metrics['nrmse'][2]),
        'nrmse_avg': shape_nrmse_avg,
        'train_percent': TRAIN_PERCENT,
        'pinn_alpha': ALPHA_INIT if physics_available_for_training else 0.0,
        'pinn_active': physics_available_for_training,
    })

if per_shape_results:
    per_shape_df = pd.DataFrame(per_shape_results)
    per_shape_csv = os.path.join(OUTPUT_DIR, 'test_metrics_per_shape.csv')
    _atomic_write_df(per_shape_df, per_shape_csv, index=False)
    print(f"\n[OK] Per-shape metrics saved: {per_shape_csv}")
    
    per_shape_csv_final = os.path.join(final_ckpt_dir, 'test_metrics_per_shape.csv')
    _atomic_write_df(per_shape_df, per_shape_csv_final, index=False)

# Save results table to CSV
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
    'mae_avg': mae_avg,
    'rmse_avg': rmse_avg,
    'r2_avg': r2_avg,
    'nrmse_avg': nrmse_avg,
    **_flatten_regression_metrics(test_metrics),
}])
results_csv_path = os.path.join(OUTPUT_DIR, "training_results.csv")
_atomic_write_df(results_df, results_csv_path, index=False)

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

def last_or_nan(lst):
    try:
        return float(lst[-1])
    except Exception:
        return float('nan')

metrics_row.update({
    'train_w_loss_final': last_or_nan(history.get('train_w_loss', [])),
    'train_l_loss_final': last_or_nan(history.get('train_l_loss', [])),
    'train_d_loss_final': last_or_nan(history.get('train_d_loss', [])),
    'train_pinn_loss_final': last_or_nan(history.get('physics_loss', [])),
    'train_total_loss_final': last_or_nan(history.get('train_loss', [])),
    'val_w_loss_final': last_or_nan(history.get('val_w_loss', [])),
    'val_l_loss_final': last_or_nan(history.get('val_l_loss', [])),
    'val_d_loss_final': last_or_nan(history.get('val_d_loss', [])),
    'val_total_loss_final': last_or_nan(history.get('val_loss', [])),
})

stability = {
    'nan_detected_during_training': bool(nan_detected),
    'nan_in_test_predictions': bool(np.any(np.isnan(y_pred_wld_denorm))) if y_pred_wld_denorm is not None else True,
    'inf_in_history': any(np.isinf(x) for arr in [history.get('train_loss', []), history.get('val_loss', []), history.get('physics_loss', [])] for x in arr),
}
metrics_row.update(stability)

detailed_metrics_df = pd.DataFrame([metrics_row])
detailed_csv_path = os.path.join(OUTPUT_DIR, 'detailed_metrics.csv')
_atomic_write_df(detailed_metrics_df, detailed_csv_path, index=False)

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

test_summary_path = os.path.join(final_ckpt_dir, 'test_summary_final.txt')
summary_buf = io.StringIO()
summary_buf.write("="*80 + "\n")
summary_buf.write(f"FINAL TEST RESULTS (REGRESSION - Q1 STANDARD) - {RUN_TAG}\n")
summary_buf.write("="*80 + "\n")
summary_buf.write(f"Selected checkpoint: {selected_model_path}\n\n")
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

# Summary Dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history.get('train_loss', []), label='Train', linewidth=2, color=TRAIN_LINE_COLOR)
axes[0, 0].plot(history.get('val_loss', []), label='Val', linewidth=2, color=VAL_LINE_COLOR)
axes[0, 0].set_title('Total Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

train_w = history.get('train_w_loss', [])
train_l = history.get('train_l_loss', [])
train_d = history.get('train_d_loss', [])
val_w = history.get('val_w_loss', [])
val_l = history.get('val_l_loss', [])
val_d = history.get('val_d_loss', [])
if train_w and train_l and train_d:
    tr_avg = [(train_w[i] + train_l[i] + train_d[i]) / 3.0 for i in range(len(train_w))]
    axes[0, 1].plot(tr_avg, label='Train Reg Avg', linewidth=2, color=TRAIN_LINE_COLOR)
if val_w and val_l and val_d:
    vl_avg = [(val_w[i] + val_l[i] + val_d[i]) / 3.0 for i in range(len(val_w))]
    axes[0, 1].plot(vl_avg, label='Val Reg Avg', linewidth=2, color=VAL_LINE_COLOR)
axes[0, 1].set_title('Regression Loss (W+L+D)/3', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

phys_losses = np.array(history.get('physics_loss', []))
if len(phys_losses) > 0 and max(phys_losses) > 0:
    active_idx = np.where(phys_losses > 0)[0]
    if len(active_idx) > 0:
        axes[1, 0].plot(active_idx + 1, phys_losses[active_idx], label='PINN Loss', linewidth=2, color='red')
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

metrics_text = f"Test Set Regression Metrics (Q1 Standard):\n\n"
metrics_text += f"Avg MAE:   {mae_avg:.4f} mm\n"
metrics_text += f"Avg RMSE:  {rmse_avg:.4f} mm\n"
metrics_text += f"Avg R²:    {r2_avg:.4f}\n"
metrics_text += f"Avg NRMSE: {nrmse_avg:.2f}%\n\n"
for i, name in enumerate(metrics_names):
    metrics_text += f"{name}: MAE={test_metrics['mae'][i]:.3f} | RMSE={test_metrics['rmse'][i]:.3f} | R²={test_metrics['r2'][i]:.3f} | NRMSE={test_metrics['nrmse'][i]:.1f}%\n"

axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
_atomic_save_fig(plt, os.path.join(OUTPUT_DIR, "training_summary_v2.png"), dpi=150, bbox_inches='tight')
plt.close()

# Detailed Loss Plots
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

if history.get('physics_loss') and max(history['physics_loss']) > 0:
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

summary = []
summary.append("XIONG ET AL. 2023 BASELINE - REGRESSION ONLY\n")
summary.append("="*80 + "\n\n")
summary.append("KEY FEATURES:\n")
summary.append("-"*80 + "\n")
summary.append("1. Task: Pure Defect Size Estimation (Regression W, L, D)\n")
summary.append(f"2. Loss formula: L = ((L_w + L_l + L_d)/3) + alpha*L_pinn\n")
summary.append(f"3. Warm-up threshold: {PINN_ACTIVATION_EPOCH}\n\n")
summary.append("TEST RESULTS (Denormalized Values - Q1 Standard):\n")
summary.append("-"*80 + "\n")
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
summary.append(f"y_scaler (SeparateMaxScaler): y_scaler.pkl\n")

_atomic_write_text('\n'.join(summary) + '\n', os.path.join(OUTPUT_DIR, "summary_results.txt"))

# ============================================================================
# REAL EXPERIMENTAL INFERENCE (EXPERIMENT_1: 5 kHz, 10 kHz, 20 kHz)
# ============================================================================
try:
    from load_real_experiment_data import evaluate_real_experiment
    evaluate_real_experiment(model, X_scaler, y_scaler, unique_shapes=None, output_dir=OUTPUT_DIR, device=device)
except Exception as e:
    print(f"[WARN] Real experiment evaluation encountered an error: {e}")

print(f"\n[OK] All results saved to: {OUTPUT_DIR}")
print(f"[OK] Model: best_model_pytorch.pth")
print(f"[OK] Summary: summary_results.txt")
print(f"\n" + "="*80)
print(f"[PINN LOSS WEIGHT SUMMARY]")
print(f"="*80)
print(f"  Alpha init (PINN term weight): {ALPHA_INIT:.6f}")
print(f"  Warm-up (epochs 1-{PINN_ACTIVATION_EPOCH}): PINN inactive")
print(f"  PINN phase (epoch {PINN_ACTIVATION_EPOCH + 1}+): PINN active")
print(f"\nFormula used:")
print(f"  L = ((L_w + L_l + L_d)/3) + alpha*L_pinn")
print(f"="*80)

