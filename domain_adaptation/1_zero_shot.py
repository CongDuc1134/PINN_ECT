"""
================================================================================
domain_adaptation/1_zero_shot.py
Direction 1: Zero-Shot Pretrained Evaluation (Source-Only Baseline)
Loads pre-trained model checkpoint and evaluates directly on real 5kHz samples.
================================================================================
"""

import os
import sys
import argparse
import pandas as pd
import torch

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


def evaluate_zero_shot(model_dir=None, output_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[START] Running Direction 1: Zero-Shot Baseline | Device: {device}")

    # 1. Load Pretrained Checkpoint & Scalers
    model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), "1_zero_shot")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Samples
    X_tensor, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    # 3. Predict without adaptation
    model.eval()
    with torch.no_grad():
        clf_out, reg_out = model(X_tensor)
        pred_classes = torch.argmax(clf_out, dim=1).cpu().numpy()
        pred_reg = denormalize_regression_predictions(reg_out.cpu().numpy(), y_scaler)

    results = []
    for fold_id, fold_info in folds.items():
        for test_idx in fold_info['test_indices']:
            meta = metadata_list[test_idx]
            pred_cls_idx = pred_classes[test_idx]
            pred_shape_name = UNIQUE_SHAPES[pred_cls_idx] if pred_cls_idx < len(UNIQUE_SHAPES) else "Unknown"

            results.append({
                'fold': fold_id,
                'method': 'Source_Only_ZeroShot',
                'filename': meta['filename'],
                'crack_no': meta['crack_no'],
                'true_shape': meta['true_shape'],
                'pred_shape': pred_shape_name,
                'shape_correct': int(meta['true_shape'] == pred_shape_name),
                'true_w': meta['true_w'],
                'pred_w': pred_reg[test_idx, 0],
                'err_w': abs(meta['true_w'] - pred_reg[test_idx, 0]),
                'true_l': meta['true_l'],
                'pred_l': pred_reg[test_idx, 1],
                'err_l': abs(meta['true_l'] - pred_reg[test_idx, 1]),
                'true_d': meta['true_d'],
                'pred_d': pred_reg[test_idx, 2],
                'err_d': abs(meta['true_d'] - pred_reg[test_idx, 2]),
            })

    df_preds = pd.DataFrame(results)
    df_summary = compute_pooled_oof_summary(df_preds)

    pred_csv = os.path.join(output_dir, "zero_shot_predictions.csv")
    summary_csv = os.path.join(output_dir, "zero_shot_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("DIRECTION 1: ZERO-SHOT BASELINE SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    evaluate_zero_shot(args.model_dir, args.output_dir)
