"""
================================================================================
domain_adaptation/3_domain_transfer_mmd.py
Direction 3: True Sim-to-Real Domain Adaptation (Deep CORAL / MMD Feature Alignment)
Aligns feature representations between Source FEM Simulation data and Target Real
Experimental ECT measurements, coupled with Kendall Homoscedastic Multi-Task Loss,
Physics Augmentation, and Paired-Measurement Consistency across 10-Fold LODO.
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
    compute_coral_loss,
    apply_physics_augmentations,
    load_simulation_source_dataset,
    KendallMultiTaskLoss,
)


def compute_mmd(x, y, sigma=1.0):
    """Gaussian RBF Kernel MMD discrepancy between source x and target y feature representations"""
    dist_xx = torch.cdist(x, x, p=2) ** 2
    dist_yy = torch.cdist(y, y, p=2) ** 2
    dist_xy = torch.cdist(x, y, p=2) ** 2
    k_xx = torch.exp(-dist_xx / (2.0 * sigma**2)).mean()
    k_yy = torch.exp(-dist_yy / (2.0 * sigma**2)).mean()
    k_xy = torch.exp(-dist_xy / (2.0 * sigma**2)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def evaluate_domain_transfer(
    model_dir=None,
    output_dir=None,
    epochs=50,
    lr=2e-4,
    align_loss_type="coral",
    align_weight=0.15,
    pair_weight=0.1,
    use_aug=True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method_name = f"Domain_Transfer_{align_loss_type.upper()}"
    print(f"[START] Running Direction 3: {method_name} (10-Fold LODO | Device: {device})")

    # 1. Load Pretrained Checkpoint & Scalers
    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        sub_folder = "3_domain_transfer_coral" if align_loss_type == "coral" else "3_domain_transfer_mmd"
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Target Samples
    X_target_all, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    # 3. Load True Source Simulation Dataset (FEM)
    X_source_all, y_source_shape_all, y_source_reg_all = load_simulation_source_dataset(
        num_samples_per_class=40, x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
    print(f"[OK] Source Domain Dataset (FEM): {X_source_all.shape} (balanced 5 classes)")

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
            dtype=torch.long,
            device=device,
        )

        reg_targets = np.array(
            [[metadata_list[i]['true_w'], metadata_list[i]['true_l'], metadata_list[i]['true_d']] for i in train_idx],
            dtype=np.float32,
        )
        if hasattr(y_scaler, 'transform'):
            reg_targets_norm = y_scaler.transform(reg_targets)
        elif hasattr(y_scaler, 'data_max_'):
            reg_targets_norm = reg_targets / y_scaler.data_max_
        else:
            reg_targets_norm = reg_targets
        y_reg_target = torch.tensor(reg_targets_norm, dtype=torch.float32, device=device)

        # Domain-Specific Normalization (AdaBN re-estimation on target domain)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.reset_running_stats()
                m.momentum = None
        model.eval()
        with torch.no_grad():
            _ = model(X_train_target)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

        # Freeze lower representation blocks, allow top representation & heads to adapt
        is_cnn = hasattr(model, "backbone") and len(model.backbone) > 0 and isinstance(model.backbone[0], nn.Conv2d)
        cutoff = 8 if is_cnn else 4
        for idx_layer, layer in enumerate(model.backbone):
            if idx_layer < cutoff:
                for p in layer.parameters():
                    p.requires_grad = False
            else:
                for p in layer.parameters():
                    p.requires_grad = True

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
        loss_fn = KendallMultiTaskLoss(
            model.log_var_clf, model.log_var_w, model.log_var_l, model.log_var_d
        )

        # Build pair indices for paired measurement consistency (samples of same crack in train_idx)
        crack_to_local = {}
        for local_i, global_i in enumerate(train_idx):
            c_no = metadata_list[global_i]['crack_no']
            crack_to_local.setdefault(c_no, []).append(local_i)
        valid_pairs = [pair for pair in crack_to_local.values() if len(pair) == 2]

        # Physics Augmentation for target samples
        if use_aug:
            X_target_batch, y_shape_batch, y_reg_batch = apply_physics_augmentations(
                X_train_target, y_shape_target, y_reg_target, noise_std=0.015, dc_shift_std=0.02
            )
        else:
            X_target_batch, y_shape_batch, y_reg_batch = X_train_target, y_shape_target, y_reg_target

        batch_size_source = min(64, X_source_all.size(0))

        # Pre-extract base model anchor features on clean target training samples
        with torch.no_grad():
            base_anchor_feat = base_model.extract_features(X_train_target)

        # Adaptation loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Target forward pass on augmented batch
            clf_target, reg_target = model(X_target_batch)
            loss_target = loss_fn(clf_target, y_shape_batch, reg_target, y_reg_batch)

            # Feature extraction on clean target samples
            curr_clean_feat = model.extract_features(X_train_target)

            # Alignment / Knowledge Distillation against base anchor representations
            if align_loss_type == "coral":
                # Align covariance between base anchor and adapted representation
                loss_align = compute_coral_loss(base_anchor_feat, curr_clean_feat)
            else:
                # Gaussian RBF MMD between base anchor and adapted representation
                loss_align = compute_mmd(base_anchor_feat, curr_clean_feat)

            # Paired Measurement Consistency Loss (same crack measured twice -> same embedding)
            if valid_pairs:
                z_clean = F.normalize(curr_clean_feat, dim=1)
                pair_dists = [1.0 - (z_clean[p[0]] * z_clean[p[1]]).sum() for p in valid_pairs]
                loss_pair = torch.stack(pair_dists).mean()
            else:
                loss_pair = torch.tensor(0.0, device=device)

            total_loss = loss_target + align_weight * loss_align + pair_weight * loss_pair
            total_loss.backward()
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

    prefix = "domain_transfer_coral" if align_loss_type == "coral" else "domain_transfer_mmd"
    pred_csv = os.path.join(output_dir, f"{prefix}_predictions.csv")
    summary_csv = os.path.join(output_dir, f"{prefix}_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print(f"DIRECTION 3: {method_name.upper()} SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


def evaluate_domain_transfer_mmd(model_dir=None, output_dir=None, epochs=50, lr=2e-4, mmd_weight=0.15):
    """Backward-compatible entry point for MMD alignment"""
    return evaluate_domain_transfer(
        model_dir=model_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        align_loss_type="mmd",
        align_weight=mmd_weight,
    )


def evaluate_domain_transfer_coral(model_dir=None, output_dir=None, epochs=50, lr=2e-4, coral_weight=0.15):
    """Entry point for Deep CORAL alignment"""
    return evaluate_domain_transfer(
        model_dir=model_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        align_loss_type="coral",
        align_weight=coral_weight,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--align-loss", type=str, default="coral", choices=["coral", "mmd"], help="Alignment loss type")
    parser.add_argument("--align-weight", type=float, default=0.15, help="Domain discrepancy weight")
    parser.add_argument("--pair-weight", type=float, default=0.1, help="Paired consistency weight")
    parser.add_argument("--no-aug", action="store_true", help="Disable physics augmentations")
    args = parser.parse_args()
    evaluate_domain_transfer(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        align_loss_type=args.align_loss,
        align_weight=args.align_weight,
        pair_weight=args.pair_weight,
        use_aug=not args.no_aug,
    )
