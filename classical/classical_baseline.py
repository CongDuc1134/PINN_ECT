# -*- coding: utf-8 -*-
"""
Classical Machine Learning Baseline (Random Forest & SVM/SVR)
Extracts hand-crafted statistical/field features from 32x32 MFL signals
and evaluates performance on 5% fixed dataset (seed 42).
"""

import os
import sys
import traceback
import faulthandler
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error, mean_absolute_error, max_error,
    balanced_accuracy_score, f1_score, matthews_corrcoef
)
from tqdm import tqdm

try:
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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
            import tempfile
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
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'classical_baseline.py'
    print(f"\n[FATAL] Unhandled exception in {script_name}", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _log_uncaught_exception

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

CHECK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.abspath(os.path.join(CHECK_DIR, ".."))

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH or not os.path.exists(DATA_PATH):
    candidates = [
        os.path.join(CHECK_DIR, "Crack_Shape_Images"),
        os.path.join(PAPER_DIR, "t-sne", "Crack_Shape_Images"),
        os.path.join(PAPER_DIR, "rep_code", "Crack_Shape_Images"),
    ]
    for c in candidates:
        if os.path.exists(c):
            DATA_PATH = c
            break

LABELS_PATH = os.environ.get("LABELS_PATH")
if not LABELS_PATH or not os.path.exists(LABELS_PATH):
    candidates_labels = [
        os.path.join(CHECK_DIR, "labels.csv"),
        os.path.join(PAPER_DIR, "rep_code", "a", "labels.csv"),
        os.path.join(PAPER_DIR, "t-sne", "labels.csv"),
    ]
    for cl in candidates_labels:
        if os.path.exists(cl):
            LABELS_PATH = cl
            break

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Outputs_classical_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MLP_DIR = os.path.join(CHECK_DIR, "mlp")
if MLP_DIR not in sys.path:
    sys.path.insert(0, MLP_DIR)

try:
    from library_functions import Load_Data_With_Labels
except ImportError as e:
    print(f"[ERROR] Could not import Load_Data_With_Labels: {e}")
    sys.exit(1)

def calculate_nrmse(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_range = np.max(y_true) - np.min(y_true)
    if y_range < eps:
        return 0.0
    return (rmse / y_range) * 100.0

def extract_manual_features(X):
    """
    Extract 8 hand-crafted MFL physical/statistical features from (N, 32, 32, 2) array:
    1. max_field: Peak magnetic field intensity
    2. min_field: Minimum field intensity
    3. std_field: Field standard deviation
    4. mean_field: Mean field value
    5. energy_field: Signal Frobenius norm / energy
    6. max_grad: Peak gradient magnitude
    7. std_grad: Gradient standard deviation
    8. mean_grad: Mean gradient magnitude
    """
    features = []
    for i in range(len(X)):
        H = X[i, :, :, 0]
        grad = X[i, :, :, 1]
        
        f1 = np.max(H)
        f2 = np.min(H)
        f3 = np.std(H)
        f4 = np.mean(H)
        f5 = np.sqrt(np.sum(H ** 2))
        f6 = np.max(grad)
        f7 = np.std(grad)
        f8 = np.mean(grad)
        
        features.append([f1, f2, f3, f4, f5, f6, f7, f8])
    return np.array(features, dtype=np.float32)

def run_classical_baseline():
    raw_csv = os.path.join(OUTPUT_DIR, "classical_baseline_all_raw.csv")
    stability_csv = os.path.join(OUTPUT_DIR, "classical_baseline_stability_5percent.csv")
    force_retrain = os.environ.get("FORCE_RETRAIN", "0").strip() in {"1", "true", "yes"}
    skip_if_exists = os.environ.get("SKIP_IF_EXISTS", "1").strip() in {"1", "true", "yes"}

    if skip_if_exists and not force_retrain and os.path.exists(raw_csv) and os.path.exists(stability_csv):
        print(f"\n[SKIP] Classical baselines already computed at: {OUTPUT_DIR}")
        print(f"[SKIP] Found completed output: {stability_csv}")
        print(f"[SKIP] Set FORCE_RETRAIN=1 to force re-running.\n")
        return

    print("\n" + "="*80)
    print("CLASSICAL MACHINE LEARNING BASELINE (RANDOM FOREST & SVM/SVR) - Q1 STANDARDS")
    print("DATASET: 5% TRAINING RATIO | SEED: 42")
    print("="*80)
    
    # 1. Load Data
    X, y, matched_filenames = Load_Data_With_Labels(DATA_PATH, LABELS_PATH)
    
    shapes_list = []
    for fn in matched_filenames:
        prefix = fn.split('_')[0]
        if prefix == 'Step':
            prefix = fn.split('_')[0] + '_' + fn.split('_')[1]
        shapes_list.append(prefix)
        
    unique_shapes = sorted(list(set(shapes_list)))
    shape_to_idx = {s: i for i, s in enumerate(unique_shapes)}
    y_shape = np.array([shape_to_idx[s] for s in shapes_list], dtype=np.int64)
    
    # 2. Extract Manual Features
    print("Trích xuất 8 đặc trưng thủ công (Peak, Energy, Gradient) từ tín hiệu MFL...")
    X_feats = extract_manual_features(X)
    print(f"X_feats shape: {X_feats.shape}")
    
    # 3. Multi-Seed & Multi-Percent Stability Evaluation
    SEEDS = [42, 123, 456, 789]
    TRAIN_PERCENT_LIST = [1, 3, 5, 7, 10]
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        max_error,
        f1_score,
        balanced_accuracy_score,
        matthews_corrcoef,
    )
    
    def evaluate_comprehensive(model_name, seed, current_train_percent, clf_model, reg_model, X_tr_norm, ys_tr, y_tr, X_te_norm, ys_te, y_te):
        clf_model.fit(X_tr_norm, ys_tr)
        pred_ys = clf_model.predict(X_te_norm)
        
        reg_model.fit(X_tr_norm, y_tr)
        pred_y = reg_model.predict(X_te_norm)
        
        acc = accuracy_score(ys_te, pred_ys) * 100.0
        bal_acc = balanced_accuracy_score(ys_te, pred_ys) * 100.0
        f1_macro = f1_score(ys_te, pred_ys, average='macro', zero_division=0) * 100.0
        mcc = matthews_corrcoef(ys_te, pred_ys)
        
        mae_w = mean_absolute_error(y_te[:, 0], pred_y[:, 0])
        mae_l = mean_absolute_error(y_te[:, 1], pred_y[:, 1])
        mae_d = mean_absolute_error(y_te[:, 2], pred_y[:, 2])
        overall_mae = (mae_w + mae_l + mae_d) / 3.0

        rmse_w = np.sqrt(mean_squared_error(y_te[:, 0], pred_y[:, 0]))
        rmse_l = np.sqrt(mean_squared_error(y_te[:, 1], pred_y[:, 1]))
        rmse_d = np.sqrt(mean_squared_error(y_te[:, 2], pred_y[:, 2]))
        overall_rmse = (rmse_w + rmse_l + rmse_d) / 3.0
        
        r2_w = r2_score(y_te[:, 0], pred_y[:, 0])
        r2_l = r2_score(y_te[:, 1], pred_y[:, 1])
        r2_d = r2_score(y_te[:, 2], pred_y[:, 2])
        overall_r2 = (r2_w + r2_l + r2_d) / 3.0

        max_err_w = max_error(y_te[:, 0], pred_y[:, 0])
        max_err_l = max_error(y_te[:, 1], pred_y[:, 1])
        max_err_d = max_error(y_te[:, 2], pred_y[:, 2])
        overall_max_err = max(max_err_w, max_err_l, max_err_d)

        nrmse_w = calculate_nrmse(y_te[:, 0], pred_y[:, 0])
        nrmse_l = calculate_nrmse(y_te[:, 1], pred_y[:, 1])
        nrmse_d = calculate_nrmse(y_te[:, 2], pred_y[:, 2])
        overall_nrmse = (nrmse_w + nrmse_l + nrmse_d) / 3.0
        
        return {
            'Model': model_name,
            'Seed': seed,
            'Train Percent': f"{current_train_percent}%",
            'Clf Acc (%)': acc,
            'Clf Bal Acc (%)': bal_acc,
            'Clf F1-Macro (%)': f1_macro,
            'Clf MCC': mcc,
            'Overall RMSE (mm)': overall_rmse,
            'Overall MAE (mm)': overall_mae,
            'Overall R2': overall_r2,
            'Overall Max Error (mm)': overall_max_err,
            'Overall NRMSE (%)': overall_nrmse,
            'W RMSE (mm)': rmse_w, 'L RMSE (mm)': rmse_l, 'D RMSE (mm)': rmse_d,
            'W MAE (mm)': mae_w, 'L MAE (mm)': mae_l, 'D MAE (mm)': mae_d,
            'W R2': r2_w, 'L R2': r2_l, 'D R2': r2_d,
            'W Max Err (mm)': max_err_w, 'L Max Err (mm)': max_err_l, 'D Max Err (mm)': max_err_d,
            'W NRMSE (%)': nrmse_w, 'L NRMSE (%)': nrmse_l, 'D NRMSE (%)': nrmse_d,
        }

    all_seed_results = []
    
    for seed in SEEDS:
        X_temp, X_test, y_temp, y_test, ys_temp, ys_test = train_test_split(
            X_feats, y, y_shape, test_size=0.10, random_state=seed
        )
        X_tr_full, X_val, y_tr_full, y_val, ys_tr_full, ys_val = train_test_split(
            X_temp, y_temp, ys_temp, test_size=0.222, random_state=seed
        )
        
        for TRAIN_PERCENT in TRAIN_PERCENT_LIST:
            subset_parts = []
            for class_idx in sorted(np.unique(ys_tr_full)):
                class_mask_idx = np.where(ys_tr_full == class_idx)[0]
                class_count = len(class_mask_idx)
                class_take = max(1, int(np.floor(class_count * (TRAIN_PERCENT / 100.0))))
                class_seed = seed + int(class_idx) * 10007
                class_rng = np.random.default_rng(class_seed)
                chosen = class_rng.permutation(class_mask_idx)[:class_take]
                subset_parts.append(np.sort(chosen))
                
            subset_indices = np.sort(np.concatenate(subset_parts))
            X_tr = X_tr_full[subset_indices]
            y_tr = y_tr_full[subset_indices]
            ys_tr = ys_tr_full[subset_indices]
            X_te, y_te, ys_te = X_test, y_test, ys_test
            
            scaler = StandardScaler()
            X_tr_norm = scaler.fit_transform(X_tr)
            X_te_norm = scaler.transform(X_te)
            
            # RF
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=seed)
            res_rf = evaluate_comprehensive('Random Forest', seed, TRAIN_PERCENT, rf_clf, rf_reg, X_tr_norm, ys_tr, y_tr, X_te_norm, ys_te, y_te)
            all_seed_results.append(res_rf)
            
            # SVM
            svm_clf = SVC(kernel='rbf', random_state=seed)
            svr_reg = MultiOutputRegressor(SVR(kernel='rbf', C=10.0))
            res_svm = evaluate_comprehensive('Support Vector Machine', seed, TRAIN_PERCENT, svm_clf, svr_reg, X_tr_norm, ys_tr, y_tr, X_te_norm, ys_te, y_te)
            all_seed_results.append(res_svm)

    df_seeds = pd.DataFrame(all_seed_results)
    
    # 1. Data Scarcity Trend (Seed = 42, All Percentages)
    df_trend = df_seeds[df_seeds['Seed'] == 42].copy()
    trend_csv_path = os.path.join(OUTPUT_DIR, "classical_baseline_trend_1seed.csv")
    df_trend.to_csv(trend_csv_path, index=False)
    
    # 2. 4-Seed Stability Summary (Fixed at 5% Train Percent)
    summary_rows = []
    for model_name in ['Random Forest', 'Support Vector Machine']:
        sub = df_seeds[(df_seeds['Model'] == model_name) & (df_seeds['Train Percent'] == "5%")]
        if len(sub) == 0: continue
        
        summary_rows.append({
            'Model': model_name,
            'Seed': 'Mean ± Std (4 Seeds)',
            'Train Percent': "5%",
            'Clf Acc (%)': f"{sub['Clf Acc (%)'].mean():.2f} ± {sub['Clf Acc (%)'].std():.2f}",
            'Clf Bal Acc (%)': f"{sub['Clf Bal Acc (%)'].mean():.2f} ± {sub['Clf Bal Acc (%)'].std():.2f}",
            'Clf F1-Macro (%)': f"{sub['Clf F1-Macro (%)'].mean():.2f} ± {sub['Clf F1-Macro (%)'].std():.2f}",
            'Clf MCC': f"{sub['Clf MCC'].mean():.4f} ± {sub['Clf MCC'].std():.4f}",
            'Overall RMSE (mm)': f"{sub['Overall RMSE (mm)'].mean():.4f} ± {sub['Overall RMSE (mm)'].std():.4f}",
            'Overall MAE (mm)': f"{sub['Overall MAE (mm)'].mean():.4f} ± {sub['Overall MAE (mm)'].std():.4f}",
            'Overall R2': f"{sub['Overall R2'].mean():.4f} ± {sub['Overall R2'].std():.4f}",
            'Overall Max Error (mm)': f"{sub['Overall Max Error (mm)'].mean():.4f} ± {sub['Overall Max Error (mm)'].std():.4f}",
            'Overall NRMSE (%)': f"{sub['Overall NRMSE (%)'].mean():.2f} ± {sub['Overall NRMSE (%)'].std():.2f}",
            'W RMSE (mm)': f"{sub['W RMSE (mm)'].mean():.4f} ± {sub['W RMSE (mm)'].std():.4f}",
            'L RMSE (mm)': f"{sub['L RMSE (mm)'].mean():.4f} ± {sub['L RMSE (mm)'].std():.4f}",
            'D RMSE (mm)': f"{sub['D RMSE (mm)'].mean():.4f} ± {sub['D RMSE (mm)'].std():.4f}",
            'W MAE (mm)': f"{sub['W MAE (mm)'].mean():.4f} ± {sub['W MAE (mm)'].std():.4f}",
            'L MAE (mm)': f"{sub['L MAE (mm)'].mean():.4f} ± {sub['L MAE (mm)'].std():.4f}",
            'D MAE (mm)': f"{sub['D MAE (mm)'].mean():.4f} ± {sub['D MAE (mm)'].std():.4f}",
            'W R2': f"{sub['W R2'].mean():.4f} ± {sub['W R2'].std():.4f}",
            'L R2': f"{sub['L R2'].mean():.4f} ± {sub['L R2'].std():.4f}",
            'D R2': f"{sub['D R2'].mean():.4f} ± {sub['D R2'].std():.4f}",
            'W Max Err (mm)': f"{sub['W Max Err (mm)'].mean():.4f} ± {sub['W Max Err (mm)'].std():.4f}",
            'L Max Err (mm)': f"{sub['L Max Err (mm)'].mean():.4f} ± {sub['L Max Err (mm)'].std():.4f}",
            'D Max Err (mm)': f"{sub['D Max Err (mm)'].mean():.4f} ± {sub['D Max Err (mm)'].std():.4f}",
            'W NRMSE (%)': f"{sub['W NRMSE (%)'].mean():.2f} ± {sub['W NRMSE (%)'].std():.2f}",
            'L NRMSE (%)': f"{sub['L NRMSE (%)'].mean():.2f} ± {sub['L NRMSE (%)'].std():.2f}",
            'D NRMSE (%)': f"{sub['D NRMSE (%)'].mean():.2f} ± {sub['D NRMSE (%)'].std():.2f}",
        })
            
    df_stability = pd.DataFrame(summary_rows)
    stability_csv_path = os.path.join(OUTPUT_DIR, "classical_baseline_stability_5percent.csv")
    df_stability.to_csv(stability_csv_path, index=False)
    
    # Save separate clean CSVs per model
    df_rf = df_seeds[df_seeds['Model'] == 'Random Forest'].copy()
    df_rf.to_csv(os.path.join(OUTPUT_DIR, "random_forest_summary_metrics.csv"), index=False)
    
    df_svm = df_seeds[df_seeds['Model'] == 'Support Vector Machine'].copy()
    df_svm.to_csv(os.path.join(OUTPUT_DIR, "svm_summary_metrics.csv"), index=False)
    
    # Save raw results for all combinations just in case
    csv_seeds_path = os.path.join(OUTPUT_DIR, "classical_baseline_all_raw.csv")
    df_seeds.to_csv(csv_seeds_path, index=False)
    
    print("\n" + "="*80)
    print("1. DATA SCARCITY TREND (Seed = 42, 1% -> 10%) - Q1 STANDARDS")
    print("="*80)
    print(df_trend[['Model', 'Train Percent', 'Overall RMSE (mm)', 'Overall MAE (mm)', 'Overall R2', 'Overall NRMSE (%)', 'Clf Acc (%)', 'Clf Bal Acc (%)', 'Clf F1-Macro (%)', 'Clf MCC']].to_string(index=False))
    
    print("\n" + "="*80)
    print("2. STABILITY SUMMARY (4 Seeds, Fixed at 5%) - READY FOR TABLE V")
    print("="*80)
    print(df_stability.to_string(index=False))
    print("="*80)
    print(f"[OK] Trend results saved to: {trend_csv_path}")
    print(f"[OK] Stability results saved to: {stability_csv_path}")

if __name__ == "__main__":
    run_classical_baseline()
