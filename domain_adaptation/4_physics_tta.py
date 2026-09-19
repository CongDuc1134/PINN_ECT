"""
================================================================================
domain_adaptation/4_physics_tta.py
Direction 4: True Physics-Informed Test-Time Adaptation (Physics-TTA)
Online adaptation of model parameters on unseen test samples without ground-truth
labels by leveraging fundamental ECT physical constraints:
1. Spatial Symmetry Invariance (reflection consistency across scan & crack axes)
2. ECT Geometric Prior (physical bounds and nominal length preservation)
3. Confidence Maximization (temperature-calibrated entropy minimization)
across 10-Fold LODO.
================================================================================
"""

import os
import sys
import copy
import argparse
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
)


def evaluate_physics_tta(
    model_dir=None,
    output_dir=None,
    steps=25,
    lr=5e-4,
    lambda_sym=0.5,
    lambda_geom=0.2,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[START] Running Direction 4: True Physics-TTA (10-Fold LODO) | Device: {device}")

    # 1. Load Pretrained Checkpoint & Scalers
    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), "4_physics_tta")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Samples
    X_tensor, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    # Nominal normalized L target for geometric regularization
    l_nominal = 10.0
    if hasattr(y_scaler, 'data_max_'):
        l_norm_nominal = l_nominal / float(y_scaler.data_max_[1])
    elif hasattr(y_scaler, 'transform'):
        l_norm_nominal = float(y_scaler.transform([[0.7, 10.0, 2.0]])[0, 1])
    else:
        l_norm_nominal = 0.5
    l_target_tensor = torch.tensor(l_norm_nominal, dtype=torch.float32, device=device)

    results = []

    for fold_id, fold_info in folds.items():
        print(f"  --> Processing Fold {fold_id:2d}/10 (Test Crack: No{fold_info['crack_no']:2d} - {fold_info['true_shape']})...")
        test_idx = fold_info['test_indices']
        X_test = X_tensor[test_idx]

        model = copy.deepcopy(base_model).to(device)

        # Freeze all parameters except BatchNorm affine scale & shift (gamma, beta)
        for param in model.parameters():
            param.requires_grad = False

        bn_params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()  # Preserve pre-trained source running mean/var to avoid corruption on small batch
                if m.weight is not None:
                    m.weight.requires_grad = True
                    bn_params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    bn_params.append(m.bias)

        # Fallback for models without BatchNorm (e.g. pure MLP architectures): adapt bias parameters
        if not bn_params:
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.requires_grad = True
                    bn_params.append(param)

        if bn_params:
            optimizer = optim.Adam(bn_params, lr=lr, weight_decay=1e-4)

            for step in range(steps):
                optimizer.zero_grad()

                # 1. Forward pass on original test inputs
                clf_out, reg_out = model(X_test)
                probs = F.softmax(clf_out, dim=1)

                # Entropy minimization (temperature scaled)
                loss_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                # 2. Physics Constraint 1: Spatial Symmetry Invariance
                # Horizontal reflection (scan-axis symmetry) and Vertical reflection (cross-axis symmetry)
                X_hflip = torch.flip(X_test, dims=[3])
                X_vflip = torch.flip(X_test, dims=[2])

                clf_h, reg_h = model(X_hflip)
                clf_v, reg_v = model(X_vflip)
                probs_h = F.softmax(clf_h, dim=1)
                probs_v = F.softmax(clf_v, dim=1)

                # Symmetry in classification probability (Symmetric KL divergence)
                kl_h = 0.5 * (F.kl_div(torch.log(probs + 1e-8), probs_h, reduction='batchmean') +
                              F.kl_div(torch.log(probs_h + 1e-8), probs, reduction='batchmean'))
                kl_v = 0.5 * (F.kl_div(torch.log(probs + 1e-8), probs_v, reduction='batchmean') +
                              F.kl_div(torch.log(probs_v + 1e-8), probs, reduction='batchmean'))
                loss_sym_clf = kl_h + kl_v

                # Symmetry in dimension prediction
                loss_sym_reg = F.mse_loss(reg_out, reg_h) + F.mse_loss(reg_out, reg_v)
                loss_sym = loss_sym_clf + loss_sym_reg

                # 3. Physics Constraint 2: Geometric Prior (length consistency)
                loss_geom = F.mse_loss(reg_out[:, 1], l_target_tensor.expand(reg_out.size(0)))

                # Combined Physics-Informed TTA objective
                total_loss = loss_entropy + lambda_sym * loss_sym + lambda_geom * loss_geom
                total_loss.backward()
                optimizer.step()

        # Final OOF evaluation
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
    print("DIRECTION 4: TRUE PHYSICS-TTA SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--steps", type=int, default=25, help="TTA adaptation steps per fold")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for BN adaptation")
    parser.add_argument("--lambda-sym", type=float, default=0.5, help="Weight for spatial symmetry constraint")
    parser.add_argument("--lambda-geom", type=float, default=0.2, help="Weight for geometric prior constraint")
    args = parser.parse_args()
    evaluate_physics_tta(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        steps=args.steps,
        lr=args.lr,
        lambda_sym=args.lambda_sym,
        lambda_geom=args.lambda_geom,
    )
