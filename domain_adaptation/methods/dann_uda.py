# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/dann_uda.py
Domain-Adversarial Neural Network (DANN) for Sim-to-Real ECT Domain Adaptation.
Uses Gradient Reversal Layer (GRL) and a Domain Discriminator to learn
Domain-Invariant Feature Representations between FEM Simulation and Real 5kHz Sensor.
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
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from domain_adaptation.utils import (
    load_pretrained_checkpoint,
    find_default_checkpoint_dir,
    load_5khz_real_data,
    build_10fold_lodo_splits,
    compute_pooled_oof_summary,
    denormalize_regression_predictions,
    UNIQUE_SHAPES,
    get_model_tag,
    apply_physics_augmentations,
    KendallMultiTaskLoss,
    load_simulation_source_dataset,
)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features: int = 128 * 4 * 4, hidden_dim: int = 256):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha=1.0):
        reversed_feat = GradientReversalFunction.apply(x, alpha)
        return self.net(reversed_feat)


def evaluate_dann_adaptation(
    model_dir: str = None,
    output_dir: str = None,
    epochs: int = 80,
    lr_backbone: float = 5e-5,
    lr_head: float = 5e-4,
    lr_disc: float = 1e-3,
    adv_weight: float = 0.2,
    phys_weight: float = 0.2,
    use_aug: bool = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method_name = "DANN_Adversarial_DA"
    print(f"\n[START] Running {method_name} (Adv_Weight={adv_weight}) | Device: {device}")

    # 1. Load Pretrained Checkpoint
    if model_dir and not os.path.exists(model_dir):
        model_dir = find_default_checkpoint_dir(model_dir)

    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    print(f"[OK] Pretrained Checkpoint: {loaded_dir}")

    tag = get_model_tag(loaded_dir)
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "exp_dann_uda", tag)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Real 5kHz Target Data
    X_target_all, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    folds = build_10fold_lodo_splits(metadata_list)
    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}

    # 3. Load Source Simulation Dataset
    X_source_all, y_source_shape_all, y_source_reg_all = load_simulation_source_dataset(
        num_samples_per_class=40, x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
    print(f"[OK] Source Domain Dataset (FEM): {X_source_all.shape}")

    results = []

    for fold_id, fold_info in folds.items():
        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        model = copy.deepcopy(base_model).to(device)
        discriminator = DomainDiscriminator(in_features=128 * 4 * 4).to(device)

        # Discriminative Layer-wise Learning Rates
        optimizer_model = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': lr_backbone, 'weight_decay': 1e-4},
            {'params': model.classifier.parameters(), 'lr': lr_head, 'weight_decay': 1e-3},
            {'params': model.regressor_backbone.parameters(), 'lr': lr_head * 0.5, 'weight_decay': 1e-3},
            {'params': model.reg_head.parameters(), 'lr': lr_head, 'weight_decay': 1e-3},
        ])
        optimizer_disc = optim.AdamW(discriminator.parameters(), lr=lr_disc, weight_decay=1e-3)

        loss_task_fn = KendallMultiTaskLoss(
            model.log_var_clf, model.log_var_w, model.log_var_l, model.log_var_d
        )
        loss_domain_fn = nn.CrossEntropyLoss()

        X_train_target = X_target_all[train_idx]
        y_shape_target = torch.tensor(
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
        y_reg_target = torch.tensor(reg_targets_norm, dtype=torch.float32, device=device)

        # Physics Augmentations for target
        if use_aug:
            X_target_batch, y_shape_batch, y_reg_batch = apply_physics_augmentations(
                X_train_target, y_shape_target, y_reg_target, noise_std=0.015, dc_shift_std=0.02
            )
        else:
            X_target_batch, y_shape_batch, y_reg_batch = X_train_target, y_shape_target, y_reg_target

        batch_size_src = min(64, X_source_all.size(0))
        n_target = X_target_batch.size(0)

        # Training loop
        model.train()
        discriminator.train()
        for epoch in range(epochs):
            # Dynamic GRL schedule: p from 0 to 1, alpha = 2 / (1 + exp(-10*p)) - 1
            p = float(epoch) / float(epochs)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            # Sample source batch
            src_indices = torch.randperm(X_source_all.size(0), device=device)[:batch_size_src]
            x_src = X_source_all[src_indices]
            y_src_shape = y_source_shape_all[src_indices]
            y_src_reg = y_source_reg_all[src_indices]

            # Domain labels: 0 for Source, 1 for Target
            domain_label_src = torch.zeros(x_src.size(0), dtype=torch.long, device=device)
            domain_label_tgt = torch.ones(n_target, dtype=torch.long, device=device)

            optimizer_model.zero_grad()
            optimizer_disc.zero_grad()

            # Forward Source (Task Loss + Domain Loss)
            feat_src = model.extract_features(x_src)
            clf_src = model.classifier(feat_src)
            reg_src = model.reg_head(model.regressor_backbone(feat_src))
            loss_task_src = loss_task_fn(clf_src, y_src_shape, reg_src, y_src_reg)

            # Forward Target (Supervised Target Task Loss + Domain Loss)
            feat_tgt = model.extract_features(X_target_batch)
            clf_tgt = model.classifier(feat_tgt)
            reg_tgt = model.reg_head(model.regressor_backbone(feat_tgt))
            loss_task_tgt = loss_task_fn(clf_tgt, y_shape_batch, reg_tgt, y_reg_batch)

            # Adversarial Domain Predictions
            domain_pred_src = discriminator(feat_src, alpha=alpha)
            domain_pred_tgt = discriminator(feat_tgt, alpha=alpha)
            loss_domain = 0.5 * (loss_domain_fn(domain_pred_src, domain_label_src) + loss_domain_fn(domain_pred_tgt, domain_label_tgt))

            # Electromagnetic Volume Conservation Loss on target
            pred_vol = reg_tgt[:, 0] * reg_tgt[:, 1] * reg_tgt[:, 2]
            signal_peak_amp = torch.amax(torch.abs(X_target_batch[:, 0, :, :]), dim=[1, 2])
            signal_amp_norm = signal_peak_amp / (torch.mean(signal_peak_amp) + 1e-6)
            loss_vol = F.mse_loss(pred_vol, signal_amp_norm * torch.mean(pred_vol).detach())

            total_loss = loss_task_tgt + 0.5 * loss_task_src + adv_weight * loss_domain + phys_weight * loss_vol
            total_loss.backward()

            optimizer_model.step()
            optimizer_disc.step()

        # OOF Inference on Hold-out Test Samples
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
    df_summary = compute_pooled_oof_summary(df_preds, output_dir=output_dir)

    pred_csv = os.path.join(output_dir, "dann_predictions.csv")
    summary_csv = os.path.join(output_dir, "dann_summary.csv")
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--adv-weight", type=float, default=0.2)
    parser.add_argument("--phys-weight", type=float, default=0.2)
    parser.add_argument("--no-aug", action="store_true")
    args = parser.parse_args()

    evaluate_dann_adaptation(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        adv_weight=args.adv_weight,
        phys_weight=args.phys_weight,
        use_aug=not args.no_aug,
    )
