"""
================================================================================
domain_adaptation/2_few_shot_peft.py
Direction 2: Few-Shot Parameter-Efficient Fine-Tuning (PEFT / Head-Tuning)
Loads pre-trained checkpoint, freezes convolutional backbone, and fine-tunes
classifier & regressor heads on 18 real samples per fold using 10-Fold LODO.
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
    apply_physics_augmentations,
    KendallMultiTaskLoss,
)


def evaluate_few_shot_peft(model_dir=None, output_dir=None, epochs=50, lr=2e-4, peft_type="head", use_aug=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method_name = "FewShot_BitFit" if peft_type == "bitfit" else "FewShot_PEFT"
    print(f"[START] Running Direction 2: {method_name} (10-Fold LODO | Aug={use_aug}) | Device: {device}")

    # 1. Load Pretrained Checkpoint & Scalers
    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    if output_dir is None:
        sub_folder = "2_few_shot_bitfit" if peft_type == "bitfit" else "2_few_shot_peft"
        output_dir = os.path.join(SCRIPT_DIR, "results", get_model_tag(loaded_dir), sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load 5kHz Real Samples
    X_tensor, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)

    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    results = []

    for fold_id, fold_info in folds.items():
        print(f"  --> Processing Fold {fold_id:2d}/10 (Test Crack: No{fold_info['crack_no']:2d} - {fold_info['true_shape']})...")
        model = copy.deepcopy(base_model).to(device)

        if peft_type == "bitfit":
            # BitFit (Zaken et al., 2022): Fine-tune only bias parameters across the entire network
            for name, param in model.named_parameters():
                if "bias" in name or "log_var" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Default Head-Tuning PEFT: Freeze backbone, fine-tune heads
            for param in model.backbone.parameters():
                param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
        loss_fn = KendallMultiTaskLoss(
            model.log_var_clf, model.log_var_w, model.log_var_l, model.log_var_d
        )

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

        # Apply physics augmentations on 18 samples to combat low-sample overfitting
        if use_aug:
            X_train_batch, y_shape_batch, y_reg_batch = apply_physics_augmentations(
                X_train, y_shape_train, y_reg_train, noise_std=0.015, dc_shift_std=0.02
            )
        else:
            X_train_batch, y_shape_batch, y_reg_batch = X_train, y_shape_train, y_reg_train

        # Fine-tune
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            clf_out, reg_out = model(X_train_batch)
            loss = loss_fn(clf_out, y_shape_batch, reg_out, y_reg_batch)
            loss.backward()
            optimizer.step()

        # OOF Inference on 2 test samples
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

    prefix = "few_shot_bitfit" if peft_type == "bitfit" else "few_shot_peft"
    pred_csv = os.path.join(output_dir, f"{prefix}_predictions.csv")
    summary_csv = os.path.join(output_dir, f"{prefix}_summary.csv")
    df_preds.to_csv(pred_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print(f"DIRECTION 2: {method_name.upper()} SUMMARY (10-FOLD LODO)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    return df_preds, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs per fold")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--peft-type", type=str, default="head", choices=["head", "bitfit"], help="PEFT type: head or bitfit")
    parser.add_argument("--no-aug", action="store_true", help="Disable physics augmentations")
    args = parser.parse_args()
    evaluate_few_shot_peft(
        args.model_dir,
        args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        peft_type=args.peft_type,
        use_aug=not args.no_aug,
    )
