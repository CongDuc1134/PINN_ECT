# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/test_calibrated_peft.py
Evaluates Few-Shot PEFT (Head-Tuning) with Analytical Signal Calibration
(Baseline Nulling + Jitter Smoothing) on Real 5kHz ECT Dataset.
Tests direct head tuning on calibrated feature manifold across PINN and Baseline.
================================================================================
"""

import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from domain_adaptation.utils import (
    load_pretrained_checkpoint,
    load_5khz_real_data,
    build_10fold_lodo_splits,
    compute_pooled_oof_summary,
    denormalize_regression_predictions,
    UNIQUE_SHAPES,
    get_model_tag,
    apply_physics_augmentations,
    KendallMultiTaskLoss,
    find_default_checkpoint_dir,
)
from domain_adaptation.methods.signal_calibration import (
    calibrate_real_tensor,
)


def evaluate_calibrated_peft(
    model_dir: str = None,
    output_dir: str = None,
    epochs: int = 50,
    lr: float = 2e-4,
    sigma: float = 0.5,
    use_aug: bool = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method_name = "Calibrated_PEFT"
    print(f"\n[START] Running {method_name} (Sigma={sigma}, Epochs={epochs}, LR={lr}) | Device: {device}")

    # 1. Load Pretrained Checkpoint
    if model_dir and not os.path.exists(model_dir):
        model_dir = find_default_checkpoint_dir(model_dir)

    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    tag = get_model_tag(loaded_dir)
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "exp_calibration", tag)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load & Calibrate Real 5kHz Data
    X_raw, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    X_calibrated = calibrate_real_tensor(X_raw, x_scaler=x_scaler, sigma=sigma)
    print(f"[OK] Applied Signal Calibration (Sigma={sigma}).")

    folds = build_10fold_lodo_splits(metadata_list)
    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    results = []

    for fold_id, fold_info in folds.items():
        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        model = copy.deepcopy(base_model).to(device)

        # Freeze Backbone completely, tune heads
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
        loss_fn = KendallMultiTaskLoss(
            model.log_var_clf, model.log_var_w, model.log_var_l, model.log_var_d
        )

        X_train = X_calibrated[train_idx]
        y_shape_train = torch.tensor(
            [shape_to_idx.get(metadata_list[i]['true_shape'], 0) for i in train_idx],
            dtype=torch.long,
            device=device
        )
        reg_targets = np.array(
            [[metadata_list[i]['true_w'], metadata_list[i]['true_l'], metadata_list[i]['true_d']] for i in train_idx],
            dtype=np.float32
        )
        if hasattr(y_scaler, 'transform'):
            reg_targets_norm = y_scaler.transform(reg_targets)
        elif hasattr(y_scaler, 'data_max_'):
            reg_targets_norm = reg_targets / y_scaler.data_max_
        else:
            reg_targets_norm = reg_targets
        y_reg_train = torch.tensor(reg_targets_norm, dtype=torch.float32, device=device)

        # Physics Augmentation
        if use_aug:
            X_train_batch, y_shape_batch, y_reg_batch = apply_physics_augmentations(
                X_train, y_shape_train, y_reg_train, noise_std=0.015, dc_shift_std=0.02
            )
        else:
            X_train_batch, y_shape_batch, y_reg_batch = X_train, y_shape_train, y_reg_train

        # Fine-tune heads
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            clf_out, reg_out = model(X_train_batch)
            loss = loss_fn(clf_out, y_shape_batch, reg_out, y_reg_batch)
            loss.backward()
            optimizer.step()

        # OOF Inference
        model.eval()
        with torch.no_grad():
            X_test = X_calibrated[test_idx]
            clf_test, reg_test = model(X_test)
            pred_classes = torch.argmax(clf_test, dim=1).cpu().numpy()
            pred_reg = denormalize_regression_predictions(reg_test.cpu().numpy(), y_scaler)

        for i_local, i_global in enumerate(test_idx):
            meta = metadata_list[i_global]
            pred_cls_idx = pred_classes[i_local]
            pred_shape_name = UNIQUE_SHAPES[pred_cls_idx] if pred_cls_idx < len(UNIQUE_SHAPES) else "Unknown"

            results.append({
                'fold': fold_id,
                'method': method_name,
                'filename': meta['filename'],
                'crack_no': meta['crack_no'],
                'true_shape': meta['true_shape'],
                'pred_shape': pred_shape_name,
                'shape_correct': int(meta['true_shape'] == pred_shape_name),
                'true_w': meta['true_w'],
                'pred_w': pred_reg[i_local, 0],
                'err_w': abs(meta['true_w'] - pred_reg[i_local, 0]),
                'true_l': meta['true_l'],
                'pred_l': pred_reg[i_local, 1],
                'err_l': abs(meta['true_l'] - pred_reg[i_local, 1]),
                'true_d': meta['true_d'],
                'pred_d': pred_reg[i_local, 2],
                'err_d': abs(meta['true_d'] - pred_reg[i_local, 2]),
            })

    df_preds = pd.DataFrame(results)
    df_summary = compute_pooled_oof_summary(df_preds, output_dir=output_dir)

    pred_csv = os.path.join(output_dir, "calibrated_peft_predictions.csv")
    summary_csv = os.path.join(output_dir, "calibrated_peft_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print(f"{method_name.upper()} 10-FOLD LODO SUMMARY ({tag})")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--no-aug", action="store_true")
    args = parser.parse_args()

    evaluate_calibrated_peft(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        sigma=args.sigma,
        use_aug=not args.no_aug,
    )
