"""
================================================================================
domain_adaptation/3_domain_transfer_mmd.py
Direction 3: Supervised Domain Transfer Learning (MMD Feature Alignment)
Aligns feature distributions between pre-trained anchor representations and real
measurement features using Maximum Mean Discrepancy (MMD) across 10-Fold LODO.
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
)


def compute_mmd(x, y, sigma=1.0):
    """Gaussian RBF Kernel MMD discrepancy between x and y feature representations"""
    dist_xx = torch.cdist(x, x, p=2) ** 2
    dist_yy = torch.cdist(y, y, p=2) ** 2
    dist_xy = torch.cdist(x, y, p=2) ** 2
    k_xx = torch.exp(-dist_xx / (2.0 * sigma**2)).mean()
    k_yy = torch.exp(-dist_yy / (2.0 * sigma**2)).mean()
    k_xy = torch.exp(-dist_xy / (2.0 * sigma**2)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def evaluate_domain_transfer_mmd(model_dir=None, output_dir=None, epochs=50, lr=2e-4, mmd_weight=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[START] Running Direction 3: Domain Transfer MMD (10-Fold LODO) | Device: {device}")

    # 1. Load Pretrained Checkpoint & Scalers
    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), "3_domain_transfer_mmd")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Samples
    X_tensor, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    # Pre-extract source anchor representations from base model
    base_model.eval()
    with torch.no_grad():
        source_anchor_features = base_model.extract_features(X_tensor).detach()

    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    criterion_clf = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    results = []

    for fold_id, fold_info in folds.items():
        print(f"  --> Processing Fold {fold_id:2d}/10 (Test Crack: No{fold_info['crack_no']:2d} - {fold_info['true_shape']})...")
        model = copy.deepcopy(base_model).to(device)

        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        X_train = X_tensor[train_idx]
        y_shape_train = torch.tensor([shape_to_idx.get(metadata_list[i]['true_shape'], 0) for i in train_idx], dtype=torch.long).to(device)

        # Domain-Specific Normalization (AdaBN initialization):
        # Starts with source normalization, then re-estimates domain-specific BN statistics on target domain
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.reset_running_stats()
                m.momentum = None  # Cumulative average on target training samples
        model.eval()
        with torch.no_grad():
            _ = model(X_train)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

        # Freeze lower convolutional blocks (blocks 1 and 2), allow block 3 and heads to adapt
        for idx_layer, layer in enumerate(model.backbone):
            if idx_layer < 8:
                for p in layer.parameters():
                    p.requires_grad = False
            else:
                for p in layer.parameters():
                    p.requires_grad = True

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)

        reg_targets = np.array([[metadata_list[i]['true_w'], metadata_list[i]['true_l'], metadata_list[i]['true_d']] for i in train_idx], dtype=np.float32)
        if hasattr(y_scaler, 'transform'):
            reg_targets_norm = y_scaler.transform(reg_targets)
        elif hasattr(y_scaler, 'data_max_'):
            reg_targets_norm = reg_targets / y_scaler.data_max_
        else:
            reg_targets_norm = reg_targets
        y_reg_train = torch.tensor(reg_targets_norm, dtype=torch.float32).to(device)

        source_feat_sub = source_anchor_features[train_idx]

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            clf_out, reg_out = model(X_train)
            curr_feat = model.extract_features(X_train)

            loss_task = criterion_clf(clf_out, y_shape_train) + criterion_reg(reg_out, y_reg_train)
            loss_mmd = compute_mmd(source_feat_sub, curr_feat)
            total_loss = loss_task + mmd_weight * loss_mmd

            total_loss.backward()
            optimizer.step()

        # OOF Inference
        model.eval()
        with torch.no_grad():
            X_test = X_tensor[test_idx]
            clf_test, reg_test = model(X_test)
            pred_classes = torch.argmax(clf_test, dim=1).cpu().numpy()
            pred_reg = denormalize_regression_predictions(reg_test.cpu().numpy(), y_scaler)

        for i_local, i_global in enumerate(test_idx):
            meta = metadata_list[i_global]
            pred_cls_idx = pred_classes[i_local]
            pred_shape_name = UNIQUE_SHAPES[pred_cls_idx] if pred_cls_idx < len(UNIQUE_SHAPES) else "Unknown"

            results.append({
                'fold': fold_id,
                'method': 'Domain_Transfer_MMD',
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

    pred_csv = os.path.join(output_dir, "domain_transfer_mmd_predictions.csv")
    summary_csv = os.path.join(output_dir, "domain_transfer_mmd_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("DIRECTION 3: DOMAIN TRANSFER MMD SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--mmd-weight", type=float, default=0.1, help="MMD discrepancy weight")
    args = parser.parse_args()
    evaluate_domain_transfer_mmd(args.model_dir, args.output_dir, epochs=args.epochs, lr=args.lr, mmd_weight=args.mmd_weight)
