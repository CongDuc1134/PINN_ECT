"""
================================================================================
domain_adaptation/5_physics_informed_mmd.py
Direction 5: Physics-Informed Domain Adaptation (PI-MMD)
[Proposed Novel Method]

Solves the 'Rigid Backbone Paradox' by combining:
1. Discriminative Layer-Wise Learning Rates (DL-LR):
   - Backbone Conv layers: Slow adaptation (lr ~ 2e-5) to preserve pre-trained
     simulation physics while absorbing sensor DC-offset and probe tilt.
   - Prediction Heads: Fast adaptation (lr ~ 5e-4) to optimize decision boundaries.
2. Gaussian RBF Maximum Mean Discrepancy (MMD) in RKHS to align feature distributions.
3. ECT Electromagnetic Volume-Perturbation Conservation Loss (L_vol):
   - Enforces Faraday's induction law: Peak magnetic field perturbation ΔB_peak
     scales with defect volume (W * L * D), anchoring regression predictions.
4. Kendall Homoscedastic Multi-Task Loss with 10-Fold LODO cross-validation.
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
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
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
)


def compute_mmd(x, y, sigma=1.0):
    """Gaussian RBF Kernel MMD discrepancy between anchor x and adapted y feature representations."""
    dist_xx = torch.cdist(x, x, p=2) ** 2
    dist_yy = torch.cdist(y, y, p=2) ** 2
    dist_xy = torch.cdist(x, y, p=2) ** 2
    k_xx = torch.exp(-dist_xx / (2.0 * sigma**2)).mean()
    k_yy = torch.exp(-dist_yy / (2.0 * sigma**2)).mean()
    k_xy = torch.exp(-dist_xy / (2.0 * sigma**2)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def evaluate_physics_informed_mmd(
    model_dir=None,
    output_dir=None,
    epochs=60,
    lr_head=5e-4,
    lr_backbone=2e-5,
    align_weight=0.1,
    phys_weight=0.2,
    use_aug=True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method_name = "Physics_Informed_MMD"
    print(f"[START] Running Direction 5: {method_name} (10-Fold LODO | Device: {device})")

    # 1. Load Pretrained Checkpoint & Scalers
    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        sub_folder = "5_physics_informed_mmd"
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Target Samples
    X_target_all, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    results = []

    for fold_id, fold_info in folds.items():
        print(f"  --> Processing Fold {fold_id:2d}/10 (Test Crack: No{fold_info['crack_no']:2d} - {fold_info['true_shape']})...")
        model = copy.deepcopy(base_model).to(device)

        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        X_train_target = X_target_all[train_idx]
        y_shape_target = torch.tensor(
            [shape_to_idx.get(metadata_list[i]['true_shape'], 0) for i in train_idx],
            dtype=torch.long
        ).to(device)

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
        y_reg_target = torch.tensor(reg_targets_norm, dtype=torch.float32).to(device)

        # Physics Augmentation on target training samples
        if use_aug:
            X_target_batch, y_shape_batch, y_reg_batch = apply_physics_augmentations(
                X_train_target, y_shape_target, y_reg_target, noise_std=0.015, dc_shift_std=0.02
            )
        else:
            X_target_batch, y_shape_batch, y_reg_batch = X_train_target, y_shape_target, y_reg_target

        # Setup Discriminative Layer-Wise Learning Rates (DL-LR)
        param_groups = []
        if hasattr(model, "backbone"):
            # Backbone gets slow LR to preserve physics representations while adjusting sensor DC offset
            param_groups.append({'params': model.backbone.parameters(), 'lr': lr_backbone, 'weight_decay': 1e-4})
        if hasattr(model, "classifier"):
            param_groups.append({'params': model.classifier.parameters(), 'lr': lr_head, 'weight_decay': 1e-3})
        if hasattr(model, "regressor_backbone"):
            param_groups.append({'params': model.regressor_backbone.parameters(), 'lr': lr_head * 0.5, 'weight_decay': 1e-3})
        if hasattr(model, "reg_head"):
            param_groups.append({'params': model.reg_head.parameters(), 'lr': lr_head, 'weight_decay': 1e-3})

        # Uncertainty parameters
        uncert_params = [p for name, p in model.named_parameters() if "log_var" in name]
        if uncert_params:
            param_groups.append({'params': uncert_params, 'lr': 1e-3})

        # Fallback if architecture differs
        if not param_groups:
            param_groups = [{'params': model.parameters(), 'lr': lr_head, 'weight_decay': 1e-3}]

        optimizer = optim.AdamW(param_groups)
        loss_fn = KendallMultiTaskLoss(
            model.log_var_clf, model.log_var_w, model.log_var_l, model.log_var_d
        )

        # Pre-extract base model anchor features on clean target training samples
        with torch.no_grad():
            base_anchor_feat = base_model.extract_features(X_train_target)

        # Adaptation loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Target forward pass
            clf_target, reg_target = model(X_target_batch)
            loss_task = loss_fn(clf_target, y_shape_batch, reg_target, y_reg_batch)

            # Feature alignment via MMD against anchor
            curr_clean_feat = model.extract_features(X_train_target)
            loss_align = compute_mmd(base_anchor_feat, curr_clean_feat)

            # Physics Volume-Perturbation Conservation Loss
            # ECT induction law: ΔB_peak perturbation scales monotonically with crack volume (W * L * D)
            pred_vol_proxy = reg_target[:, 0] * reg_target[:, 1] * reg_target[:, 2]
            true_vol_proxy = y_reg_batch[:, 0] * y_reg_batch[:, 1] * y_reg_batch[:, 2]
            loss_vol = F.mse_loss(pred_vol_proxy, true_vol_proxy)

            total_loss = loss_task + align_weight * loss_align + phys_weight * loss_vol

            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # OOF Inference on 2 held-out test samples
        model.eval()
        with torch.no_grad():
            X_test = X_target_all[test_idx]
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
    df_summary = compute_pooled_oof_summary(df_preds)

    pred_csv = os.path.join(output_dir, "physics_informed_mmd_predictions.csv")
    summary_csv = os.path.join(output_dir, "physics_informed_mmd_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("DIRECTION 5: PHYSICS-INFORMED MMD (PI-MMD) SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 5: Physics-Informed MMD Domain Adaptation")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs for adaptation")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for prediction heads")
    parser.add_argument("--lr-backbone", type=float, default=2e-5, help="Learning rate for Conv backbone")
    parser.add_argument("--align-weight", type=float, default=0.1, help="Weight for MMD feature alignment loss")
    parser.add_argument("--phys-weight", type=float, default=0.2, help="Weight for physics volume loss")
    args = parser.parse_args()

    evaluate_physics_informed_mmd(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        align_weight=args.align_weight,
        phys_weight=args.phys_weight,
    )
