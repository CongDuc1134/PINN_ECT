#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_real_only_scratch.py
Ablation Baseline: Real-Only Training from Scratch (10-Fold LODO)
Huấn luyện mô hình khởi tạo ngẫu nhiên hoàn toàn từ đầu (KHÔNG dùng pretrained weights từ mô phỏng FEM)
chỉ trên 18 mẫu thực nghiệm trong mỗi Fold và đánh giá trên 2 mẫu test ngoài mẫu.

Mục đích: Chứng minh giá trị cốt lõi của việc tiền huấn luyện trên 33.060 mẫu mô phỏng FEM.
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

from domain_adaptation.models import ImprovedMultimodelNet
from domain_adaptation.utils import (
    load_pretrained_checkpoint,
    load_5khz_real_data,
    build_10fold_lodo_splits,
    compute_pooled_oof_summary,
    denormalize_regression_predictions,
    UNIQUE_SHAPES,
    get_model_tag,
)


def evaluate_real_only_scratch(model_dir=None, output_dir=None, epochs=100, lr=5e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[START] Running Ablation: Real-Only Training From Scratch (10-Fold LODO) | Device: {device}")

    # 1. We load scalers from checkpoint for standardized evaluation & input scaling
    _, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Reference Checkpoint Scalers: {loaded_dir}")

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), "0_real_only_scratch")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Samples
    X_tensor, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    criterion_clf = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    results = []

    for fold_id, fold_info in folds.items():
        print(f"  --> Processing Fold {fold_id:2d}/10 (Test Crack: No{fold_info['crack_no']:2d} - {fold_info['true_shape']})...")

        # Initialize network from scratch (random initialization, NO pretrained weights!)
        model = ImprovedMultimodelNet(num_shapes=len(UNIQUE_SHAPES)).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        X_train = X_tensor[train_idx]
        y_shape_train = torch.tensor([shape_to_idx.get(metadata_list[i]['true_shape'], 0) for i in train_idx], dtype=torch.long).to(device)

        reg_targets = np.array([[metadata_list[i]['true_w'], metadata_list[i]['true_l'], metadata_list[i]['true_d']] for i in train_idx], dtype=np.float32)
        if hasattr(y_scaler, 'transform'):
            reg_targets_norm = y_scaler.transform(reg_targets)
        elif hasattr(y_scaler, 'data_max_'):
            reg_targets_norm = reg_targets / y_scaler.data_max_
        else:
            reg_targets_norm = reg_targets
        y_reg_train = torch.tensor(reg_targets_norm, dtype=torch.float32).to(device)

        X_test = X_tensor[test_idx]

        # Train from scratch on 18 samples
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            clf_out, reg_out = model(X_train)
            loss = criterion_clf(clf_out, y_shape_train) + criterion_reg(reg_out, y_reg_train)
            loss.backward()
            optimizer.step()

        # Evaluate on the 2 held-out test samples
        model.eval()
        with torch.no_grad():
            clf_test, reg_test = model(X_test)
            pred_classes = torch.argmax(clf_test, dim=1).cpu().numpy()
            pred_reg = denormalize_regression_predictions(reg_test.cpu().numpy(), y_scaler)

        for i_local, i_global in enumerate(test_idx):
            meta = metadata_list[i_global]
            pred_cls_idx = pred_classes[i_local]
            pred_shape_name = UNIQUE_SHAPES[pred_cls_idx] if pred_cls_idx < len(UNIQUE_SHAPES) else "Unknown"

            results.append({
                'fold': fold_id,
                'method': 'Real_Only_Scratch',
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
    pred_csv = os.path.join(output_dir, "real_only_scratch_predictions.csv")
    summary_csv = os.path.join(output_dir, "real_only_scratch_summary.csv")

    df_preds.to_csv(pred_csv, index=False)
    df_summary = compute_pooled_oof_summary(df_preds, output_dir=output_dir)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("ABLATION: REAL-ONLY SCRATCH TRAINING SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")

    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Only Scratch Training Baseline (10-Fold LODO)")
    parser.add_argument("--model-dir", type=str, default=None, help="Reference model dir for scalers")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    evaluate_real_only_scratch(args.model_dir, args.output_dir, epochs=args.epochs, lr=args.lr)
