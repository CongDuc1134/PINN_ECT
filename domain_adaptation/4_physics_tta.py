"""
================================================================================
domain_adaptation/4_physics_tta.py
Direction 4: Physics-Informed Test-Time Adaptation (Physics-TTA)
Adapts BatchNorm parameters online on unseen test samples without ground-truth
labels using entropy minimization, evaluated across 10-Fold LODO.
================================================================================
"""

import os
import sys
import copy
import argparse
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


def evaluate_physics_tta(model_dir=None, output_dir=None, steps=25, lr=1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[START] Running Direction 4: Physics-TTA (10-Fold LODO) | Device: {device}")

    # 1. Load Pretrained Checkpoint & Scalers
    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), "4_physics_tta")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Samples
    X_tensor, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    results = []

    for fold_id, fold_info in folds.items():
        print(f"  --> Processing Fold {fold_id:2d}/10 (Test Crack: No{fold_info['crack_no']:2d} - {fold_info['true_shape']})...")
        test_idx = fold_info['test_indices']
        X_test = X_tensor[test_idx]

        model = copy.deepcopy(base_model).to(device)

        # Freeze all parameters except BatchNorm affine parameters (gamma, beta)
        for param in model.parameters():
            param.requires_grad = False
        bn_params = []
        # TTA: Strictly preserve pretrained model normalization preprocessing and running stats
        # Adapt only affine scale and shift parameters (gamma, beta) without altering normalization arbitrarily
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()  # Retain pretrained source running mean and var (prevents noise from small test batch)
                if m.weight is not None:
                    m.weight.requires_grad = True
                    bn_params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    bn_params.append(m.bias)

        if bn_params:
            optimizer = optim.Adam(bn_params, lr=lr)
            for _ in range(steps):
                optimizer.zero_grad()
                clf_out, reg_out = model(X_test)
                # Entropy minimization (self-supervision without ground truth)
                probs = torch.softmax(clf_out, dim=1)
                loss_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                loss_entropy.backward()
                optimizer.step()

        # Final evaluation
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
                'method': 'Physics_TTA',
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

    pred_csv = os.path.join(output_dir, "physics_tta_predictions.csv")
    summary_csv = os.path.join(output_dir, "physics_tta_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("DIRECTION 4: PHYSICS-TTA SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--steps", type=int, default=25, help="TTA adaptation steps per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for BN adaptation")
    args = parser.parse_args()
    evaluate_physics_tta(args.model_dir, args.output_dir, steps=args.steps, lr=args.lr)
