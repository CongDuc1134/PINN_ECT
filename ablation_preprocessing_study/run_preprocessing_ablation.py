# -*- coding: utf-8 -*-
"""
scratch/run_preprocessing_ablation.py
Comprehensive Ablation Study on Input Preprocessing for ECT Sim-to-Real Domain Adaptation:
1. Border Baseline Nulling (sensor DC offset & lift-off drift removal)
2. Spatial Gaussian Smoothing (mechanical probe vibration & jitter filtering)

Evaluates 4 configurations:
- Config 1: Full Preprocessing (Nulling ON, Smoothing ON, sigma=0.5) [Proposed]
- Config 2: No Baseline Nulling (Nulling OFF, Smoothing ON, sigma=0.5)
- Config 3: No Gaussian Smoothing (Nulling ON, Smoothing OFF, sigma=0.0)
- Config 4: Completely Raw Sensor (Nulling OFF, Smoothing OFF) [User Request]

Evaluates on:
- CNN PINN (PI-LGL)
- CNN Baseline (NoPINN)
Under both protocols:
- Protocol 1: Repeat Scan (In-Distribution / Familiar Specimens)
- Protocol 2: 10-Fold LODO (Out-of-Distribution / Blind Unseen Specimens)

All results, plots, and markdown reports saved to:
ablation_preprocessing_study/
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ablation_preprocessing_study")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from domain_adaptation.utils import (
    load_pretrained_checkpoint,
    build_10fold_lodo_splits,
    UNIQUE_SHAPES,
    denormalize_regression_predictions,
)
from domain_adaptation.methods.lora_conv import (
    inject_lora_into_model,
    get_lora_trainable_parameters,
)
from load_real_experiment_data import (
    find_experiment_1_dir,
    extract_crack_no_from_filename,
    TABLE_2_GROUND_TRUTH,
)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_raw_5khz_scan_matrices():
    """Loads raw 5kHz CSV matrices and ground truth without any preprocessing."""
    exp1_dir = find_experiment_1_dir()
    train_dir = os.path.join(exp1_dir, "Trainning")
    csv_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if "5khz" in f.lower() and f.endswith(".csv")])

    labels_csv = os.path.join(train_dir, "labels.csv")
    labels_dict = {}
    if os.path.exists(labels_csv):
        ldf = pd.read_csv(labels_csv)
        for _, row in ldf.iterrows():
            fname = str(row['filename']).strip()
            labels_dict[fname] = {
                'shape': str(row.get('shape', 'Unknown')),
                'width': float(row.get('width', np.nan)),
                'length': float(row.get('length', np.nan)),
                'depth': float(row.get('depth', np.nan)),
            }

    raw_samples = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        raw_mat = pd.read_csv(fpath, header=None).values.astype(np.float32)
        crack_no = extract_crack_no_from_filename(fname)

        if fname in labels_dict:
            gt = labels_dict[fname]
        elif crack_no in TABLE_2_GROUND_TRUTH:
            gt = TABLE_2_GROUND_TRUTH[crack_no]
        else:
            gt = {'shape': 'Unknown', 'width': 0.7, 'length': 10.0, 'depth': 2.0}

        raw_samples.append({
            'filename': fname,
            'crack_no': crack_no,
            'raw_mat': raw_mat,
            'true_shape': gt.get('shape', gt.get('shape_name', 'Unknown')),
            'true_w': float(gt.get('width', gt.get('W', 0.7))),
            'true_l': float(gt.get('length', gt.get('L', 10.0))),
            'true_d': float(gt.get('depth', gt.get('D', 2.0))),
        })
    return raw_samples


def process_scan_matrix(matrix_2d, apply_nulling=True, apply_smoothing=True, sigma=0.5):
    """
    Processes 1 raw matrix according to the ablation parameters:
    - Resize to 32x32
    - apply_nulling: subtract median border
    - apply_smoothing: gaussian filter
    - Recompute spatial gradient
    Returns:
    - two_channel_normed: (32, 32, 2)
    - phys_field: (32, 32) single channel in physical unit for Laplacian calculation
    """
    arr = np.array(matrix_2d, dtype=np.float32)
    if arr.shape != (32, 32):
        zoom_factors = (32.0 / arr.shape[0], 32.0 / arr.shape[1])
        arr = scipy.ndimage.zoom(arr, zoom_factors, order=1).astype(np.float32)

    # 1. Border baseline nulling
    if apply_nulling:
        borders = np.concatenate([
            arr[:3, :].flatten(),
            arr[-3:, :].flatten(),
            arr[:, :3].flatten(),
            arr[:, -3:].flatten()
        ])
        baseline_offset = np.median(borders)
        field = arr - baseline_offset
    else:
        field = arr.copy()

    # 2. Gaussian smoothing
    if apply_smoothing and sigma > 0:
        field = scipy.ndimage.gaussian_filter(field, sigma=sigma)

    # Physical field copy for Laplacian guard
    phys_field = field.copy()

    # 3. Gradient magnitude recomputation
    grad_y = np.gradient(field, axis=0)
    grad_x = np.gradient(field, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    two_channel = np.dstack([field, grad_mag]).astype(np.float32)
    return two_channel, phys_field


def build_ablation_tensors(raw_samples, x_scaler, apply_nulling, apply_smoothing, sigma=0.5):
    """Builds (N, 2, 32, 32) scaled tensor and physical field matrices for all samples."""
    N = len(raw_samples)
    img_list = []
    phys_list = []

    for s in raw_samples:
        two_ch, phys = process_scan_matrix(
            s['raw_mat'],
            apply_nulling=apply_nulling,
            apply_smoothing=apply_smoothing,
            sigma=sigma
        )
        img_list.append(two_ch)
        phys_list.append(phys)

    img_np = np.array(img_list, dtype=np.float32) # (N, 32, 32, 2)
    if x_scaler is not None:
        flat = img_np.reshape(-1, 2)
        scaled = x_scaler.transform(flat)
        img_np = scaled.reshape(N, 32, 32, 2)

    X_tensor = torch.tensor(img_np, dtype=torch.float32).permute(0, 3, 1, 2).contiguous().to(device)
    return X_tensor, phys_list


def compute_max_laplacian(field_2d):
    """Calculates spatial curvature Laplacian ||∇²H|| on physical matrix."""
    d2x = np.gradient(np.gradient(field_2d, axis=0), axis=0)
    d2y = np.gradient(np.gradient(field_2d, axis=1), axis=1)
    lap = np.abs(d2x + d2y)
    return float(np.max(lap))


def run_single_adaptation_eval(
    base_model,
    y_scaler,
    X_tensor,
    phys_list,
    raw_samples,
    train_indices,
    test_indices,
    is_pinn=True,
    epochs=70,
    rank=4,
    alpha=8.0,
    lr_lora=1e-3,
    lr_head=5e-4,
    lap_threshold=0.10
):
    """Runs LoRA adaptation on train_indices and evaluates on test_indices."""
    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    y_shape_train = torch.tensor([shape_to_idx[raw_samples[i]["true_shape"]] for i in train_indices], device=device)

    reg_targets = np.array(
        [[raw_samples[i]["true_w"], raw_samples[i]["true_l"], raw_samples[i]["true_d"]] for i in train_indices],
        dtype=np.float32
    )
    if hasattr(y_scaler, "transform"):
        reg_norm = y_scaler.transform(reg_targets)
    elif hasattr(y_scaler, "data_max_"):
        reg_norm = reg_targets / y_scaler.data_max_
    else:
        reg_norm = reg_targets
    y_reg_train = torch.tensor(reg_norm, dtype=torch.float32, device=device)

    # Fresh LoRA model
    model = copy.deepcopy(base_model).to(device)
    model = inject_lora_into_model(model, rank=rank, alpha=alpha).to(device)
    optimizer = optim.AdamW(
        get_lora_trainable_parameters(model, lr_lora=lr_lora, lr_head=lr_head),
        weight_decay=1e-3
    )

    # Train LoRA
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        clf_out, reg_out = model(X_tensor[train_indices])
        loss = F.cross_entropy(clf_out, y_shape_train) + F.mse_loss(reg_out, y_reg_train)
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    with torch.no_grad():
        clf_test, reg_test = model(X_tensor[test_indices])
        raw_pred_classes = torch.argmax(clf_test, dim=1).cpu().numpy()
        pred_regs = denormalize_regression_predictions(reg_test.cpu().numpy(), y_scaler)

    # Predictions & Laplacian Curvature Gating
    preds = []
    for local_idx, global_idx in enumerate(test_indices):
        s = raw_samples[global_idx]
        raw_idx = raw_pred_classes[local_idx]
        raw_shape = UNIQUE_SHAPES[raw_idx] if raw_idx < len(UNIQUE_SHAPES) else "Unknown"

        phys_2d = phys_list[global_idx]
        max_lap = compute_max_laplacian(phys_2d)

        if is_pinn:
            # PI-LGL Laplacian Curvature Guard
            if raw_shape in ["Step_R", "Step_T"] and max_lap < lap_threshold:
                final_shape = "Ellipse"
                gated = True
            else:
                final_shape = raw_shape
                gated = False
        else:
            final_shape = raw_shape
            gated = False

        preds.append({
            'global_idx': global_idx,
            'filename': s['filename'],
            'crack_no': s['crack_no'],
            'true_shape': s['true_shape'],
            'raw_pred_shape': raw_shape,
            'final_pred_shape': final_shape,
            'max_laplacian': max_lap,
            'gated': int(gated),
            'true_w': s['true_w'],
            'true_l': s['true_l'],
            'true_d': s['true_d'],
            'pred_w': pred_regs[local_idx, 0],
            'pred_l': pred_regs[local_idx, 1],
            'pred_d': pred_regs[local_idx, 2],
            'err_w': abs(s['true_w'] - pred_regs[local_idx, 0]),
            'err_l': abs(s['true_l'] - pred_regs[local_idx, 1]),
            'err_d': abs(s['true_d'] - pred_regs[local_idx, 2]),
        })

    return preds


def evaluate_protocol_repeat_scan(base_model, y_scaler, X_tensor, phys_list, raw_samples, is_pinn):
    """Protocol 1: Repeat Scan (Train on _2, Test on _1)."""
    train_idx = [i for i, s in enumerate(raw_samples) if "_1" not in s["filename"]]
    test_idx = [i for i, s in enumerate(raw_samples) if "_1" in s["filename"]]

    preds = run_single_adaptation_eval(
        base_model, y_scaler, X_tensor, phys_list, raw_samples,
        train_indices=train_idx,
        test_indices=test_idx,
        is_pinn=is_pinn
    )
    return preds


def evaluate_protocol_10fold_lodo(base_model, y_scaler, X_tensor, phys_list, raw_samples, is_pinn):
    """Protocol 2: 10-Fold LODO (Leave-One-Defect-Out across 10 cracks)."""
    folds = build_10fold_lodo_splits(raw_samples)
    all_preds = []

    for fold_id, fold_info in sorted(folds.items()):
        test_idx = fold_info['test_indices']
        train_idx = fold_info['train_indices']

        fold_preds = run_single_adaptation_eval(
            base_model, y_scaler, X_tensor, phys_list, raw_samples,
            train_indices=train_idx,
            test_indices=test_idx,
            is_pinn=is_pinn
        )
        for p in fold_preds:
            p['fold_id'] = fold_id
        all_preds.extend(fold_preds)

    return all_preds


def compute_metrics_from_preds(preds):
    """Computes Accuracy, Macro F1, and MAEs for W, L, D, and Total."""
    y_true = [p['true_shape'] for p in preds]
    y_pred = [p['final_pred_shape'] for p in preds]

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0

    mae_w = np.mean([p['err_w'] for p in preds])
    mae_l = np.mean([p['err_l'] for p in preds])
    mae_d = np.mean([p['err_d'] for p in preds])
    mae_total = (mae_w + mae_l + mae_d) / 3.0

    # NMAE relative to mean dimensions: W=0.7mm, L=10.0mm, D=2.1mm
    nmae_w = (mae_w / 0.7) * 100.0
    nmae_l = (mae_l / 10.0) * 100.0
    nmae_d = (mae_d / 2.1) * 100.0
    nmae_total = (nmae_w + nmae_l + nmae_d) / 3.0

    return {
        'acc': acc,
        'f1': f1,
        'mae_w': mae_w,
        'mae_l': mae_l,
        'mae_d': mae_d,
        'mae_total': mae_total,
        'nmae_w': nmae_w,
        'nmae_l': nmae_l,
        'nmae_d': nmae_d,
        'nmae_total': nmae_total,
    }


def main():
    print("=" * 80)
    print("STARTING COMPREHENSIVE PREPROCESSING ABLATION STUDY")
    print("Investigating: (1) Border Baseline Nulling and (2) Spatial Gaussian Smoothing")
    print(f"Device: {device}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. Load checkpoints
    pinn_ckpt = os.path.join(
        PROJECT_ROOT,
        "cnn", "Outputs_cnn_pinn", "loss_log1p_norm_sse", "train_05pct",
        "PINN_base_a1_W100_E300_seed_42_run_20260819_103124"
    )
    nopinn_ckpt = os.path.join(
        PROJECT_ROOT,
        "cnn", "Outputs_cnn_baseline", "train_05pct",
        "NoPINN_E300_seed_42_run_20260818_135833"
    )

    pinn_model, x_scaler, y_scaler, _ = load_pretrained_checkpoint(pinn_ckpt, device=device)
    nopinn_model, _, _, _ = load_pretrained_checkpoint(nopinn_ckpt, device=device)

    # 2. Load Raw Scan Data
    raw_samples = load_raw_5khz_scan_matrices()
    print(f"[OK] Loaded {len(raw_samples)} raw 5kHz experimental scan matrices.")

    # 3. Define the 4 Preprocessing Ablation Configurations
    configs = [
        {
            "id": "Config_1_Full_Prep",
            "name": "Full Preprocessing (Proposed)",
            "desc": "Nulling ON, Smoothing ON (sigma=0.5)",
            "nulling": True,
            "smoothing": True,
            "sigma": 0.5,
        },
        {
            "id": "Config_2_No_Nulling",
            "name": "Ablation: No Baseline Nulling",
            "desc": "Nulling OFF (Preserve DC drift), Smoothing ON (sigma=0.5)",
            "nulling": False,
            "smoothing": True,
            "sigma": 0.5,
        },
        {
            "id": "Config_3_No_Smoothing",
            "name": "Ablation: No Gaussian Smoothing",
            "desc": "Nulling ON, Smoothing OFF (Preserve sensor jitter)",
            "nulling": True,
            "smoothing": False,
            "sigma": 0.0,
        },
        {
            "id": "Config_4_Raw_Neither",
            "name": "Completely Raw Sensor (No Nulling & No Smoothing)",
            "desc": "Nulling OFF, Smoothing OFF (Full raw noise)",
            "nulling": False,
            "smoothing": False,
            "sigma": 0.0,
        },
    ]

    all_summary_rows = []
    all_prediction_records = []

    for cfg in configs:
        print("\n" + "#" * 80)
        print(f"EVALUATING CONFIGURATION: {cfg['name']}")
        print(f"Details: {cfg['desc']}")
        print("#" * 80)

        # Build tensors for this configuration
        X_tensor, phys_list = build_ablation_tensors(
            raw_samples,
            x_scaler=x_scaler,
            apply_nulling=cfg['nulling'],
            apply_smoothing=cfg['smoothing'],
            sigma=cfg['sigma']
        )

        for model_label, model_obj, is_pinn in [
            ("CNN PINN (PI-LGL)", pinn_model, True),
            ("CNN Baseline (NoPINN)", nopinn_model, False)
        ]:
            # Protocol 1: Repeat Scan
            preds_rep = evaluate_protocol_repeat_scan(
                model_obj, y_scaler, X_tensor, phys_list, raw_samples, is_pinn=is_pinn
            )
            m_rep = compute_metrics_from_preds(preds_rep)

            all_summary_rows.append({
                "Configuration": cfg['name'],
                "Config_ID": cfg['id'],
                "Baseline_Nulling": "Enabled" if cfg['nulling'] else "Disabled",
                "Gaussian_Smoothing": "Enabled" if cfg['smoothing'] else "Disabled",
                "Model": model_label,
                "Protocol": "Repeat Scan (In-Dist)",
                "Accuracy (%)": m_rep['acc'],
                "Macro F1 (%)": m_rep['f1'],
                "MAE W (mm)": m_rep['mae_w'],
                "MAE L (mm)": m_rep['mae_l'],
                "MAE D (mm)": m_rep['mae_d'],
                "MAE Total (mm)": m_rep['mae_total'],
                "NMAE W (%)": m_rep['nmae_w'],
                "NMAE L (%)": m_rep['nmae_l'],
                "NMAE D (%)": m_rep['nmae_d'],
                "NMAE Total (%)": m_rep['nmae_total'],
            })
            for p in preds_rep:
                p_rec = dict(p)
                p_rec.update({
                    "Config_ID": cfg['id'],
                    "Model": model_label,
                    "Protocol": "Repeat Scan",
                })
                all_prediction_records.append(p_rec)

            # Protocol 2: 10-Fold LODO
            preds_lodo = evaluate_protocol_10fold_lodo(
                model_obj, y_scaler, X_tensor, phys_list, raw_samples, is_pinn=is_pinn
            )
            m_lodo = compute_metrics_from_preds(preds_lodo)

            all_summary_rows.append({
                "Configuration": cfg['name'],
                "Config_ID": cfg['id'],
                "Baseline_Nulling": "Enabled" if cfg['nulling'] else "Disabled",
                "Gaussian_Smoothing": "Enabled" if cfg['smoothing'] else "Disabled",
                "Model": model_label,
                "Protocol": "10-Fold LODO (Out-of-Dist)",
                "Accuracy (%)": m_lodo['acc'],
                "Macro F1 (%)": m_lodo['f1'],
                "MAE W (mm)": m_lodo['mae_w'],
                "MAE L (mm)": m_lodo['mae_l'],
                "MAE D (mm)": m_lodo['mae_d'],
                "MAE Total (mm)": m_lodo['mae_total'],
                "NMAE W (%)": m_lodo['nmae_w'],
                "NMAE L (%)": m_lodo['nmae_l'],
                "NMAE D (%)": m_lodo['nmae_d'],
                "NMAE Total (%)": m_lodo['nmae_total'],
            })
            for p in preds_lodo:
                p_rec = dict(p)
                p_rec.update({
                    "Config_ID": cfg['id'],
                    "Model": model_label,
                    "Protocol": "10-Fold LODO",
                })
                all_prediction_records.append(p_rec)

            print(f"--> [{model_label:22s}] Repeat Scan: Acc={m_rep['acc']:5.1f}%, F1={m_rep['f1']:5.1f}%, MAE_L={m_rep['mae_l']:.4f}mm | LODO: Acc={m_lodo['acc']:5.1f}%, F1={m_lodo['f1']:5.1f}%, MAE_L={m_lodo['mae_l']:.4f}mm")

    # Save summary dataframe
    df_summary = pd.DataFrame(all_summary_rows)
    df_preds = pd.DataFrame(all_prediction_records)

    summary_csv = os.path.join(OUTPUT_DIR, "ablation_summary_table.csv")
    preds_csv = os.path.join(OUTPUT_DIR, "ablation_detailed_predictions.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_preds.to_csv(preds_csv, index=False)

    print("\n" + "=" * 80)
    print("PREPROCESSING ABLATION RESULTS SUMMARY TABLE:")
    print("=" * 80)
    cols_show = ["Configuration", "Model", "Protocol", "Accuracy (%)", "Macro F1 (%)", "MAE L (mm)", "MAE Total (mm)"]
    print(df_summary[cols_show].to_string(index=False))

    return df_summary, raw_samples


if __name__ == "__main__":
    main()
