# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/run_prototype_adaptation.py
Trial #11: Physics-Informed Nearest Prototype Classifier (PI-NPC)
with Domain-Shift Corrected Simulation Anchors for Missing Classes.
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
    load_5khz_real_data,
    build_10fold_lodo_splits,
    compute_pooled_oof_summary,
    denormalize_regression_predictions,
    UNIQUE_SHAPES,
    get_model_tag,
    apply_physics_augmentations,
    KendallMultiTaskLoss,
    find_default_checkpoint_dir,
    load_simulation_source_dataset,
)
from domain_adaptation.methods.lora_conv import (
    inject_lora_into_model,
    get_lora_trainable_parameters,
)
from domain_adaptation.methods.signal_calibration import (
    calibrate_real_tensor,
)


def evaluate_pi_npc(
    model_dir: str = None,
    output_dir: str = None,
    epochs: int = 70,
    rank: int = 4,
    alpha: float = 8.0,
    lr_lora: float = 1e-3,
    lr_head: float = 5e-4,
    phys_weight: float = 0.3,
    use_calibration: bool = True,
    use_domain_shift: bool = True,
    use_tta: bool = True,
    use_aug: bool = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method_name = "PI_NPC_Prototype"
    print(f"\n[START] Running {method_name} | Rank={rank}, DomainShift={use_domain_shift}, Device: {device}")

    if model_dir and not os.path.exists(model_dir):
        model_dir = find_default_checkpoint_dir(model_dir)

    base_model, x_scaler, y_scaler, loaded_dir = load_pretrained_checkpoint(model_dir, device=device)
    tag = get_model_tag(loaded_dir)
    print(f"[OK] Checkpoint: {loaded_dir} ({tag})")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "trial_11_pi_npc_prototype", tag)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Real Data
    X_target_raw, metadata_list = load_5khz_real_data(x_scaler=x_scaler, device=device)
    if use_calibration:
        X_target_all = calibrate_real_tensor(X_target_raw, x_scaler=x_scaler, sigma=0.5)
    else:
        X_target_all = X_target_raw

    # 2. Load Simulation Source Dataset for Reference Class Prototypes
    X_source, y_source_shape, y_source_reg = load_simulation_source_dataset(
        num_samples_per_class=40, x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
    print(f"[OK] Simulation Reference Dataset: {X_source.shape}")

    folds = build_10fold_lodo_splits(metadata_list)
    shape_to_idx = {s: i for i, s in enumerate(UNIQUE_SHAPES)}
    results = []

    for fold_id, fold_info in folds.items():
        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        # Clone and inject LoRA into base model
        model = copy.deepcopy(base_model).to(device)
        model = inject_lora_into_model(model, rank=rank, alpha=alpha).to(device)

        optimizer = optim.AdamW(
            get_lora_trainable_parameters(model, lr_lora=lr_lora, lr_head=lr_head),
            weight_decay=1e-3
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        loss_fn = KendallMultiTaskLoss(
            model.log_var_clf, model.log_var_w, model.log_var_l, model.log_var_d
        )

        X_train_target = X_target_all[train_idx]
        train_shapes = [shape_to_idx.get(metadata_list[i]['true_shape'], 0) for i in train_idx]
        y_shape_target = torch.tensor(train_shapes, dtype=torch.long, device=device)
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

        if use_aug:
            X_target_batch, y_shape_batch, y_reg_batch = apply_physics_augmentations(
                X_train_target, y_shape_target, y_reg_target, noise_std=0.015, dc_shift_std=0.02
            )
        else:
            X_target_batch, y_shape_batch, y_reg_batch = X_train_target, y_shape_target, y_reg_target

        crack_to_local = {}
        for local_i, global_i in enumerate(train_idx):
            c_no = metadata_list[global_i]['crack_no']
            crack_to_local.setdefault(c_no, []).append(local_i)
        valid_pairs = [pair for pair in crack_to_local.values() if len(pair) == 2]

        # Stage 1: Fine-tune backbone and regression head with Kendall + Faraday loss
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            clf_out, reg_out = model(X_target_batch)
            loss_task = loss_fn(clf_out, y_shape_batch, reg_out, y_reg_batch)

            pred_vol = reg_out[:, 0] * reg_out[:, 1] * reg_out[:, 2]
            signal_peak_amp = torch.amax(torch.abs(X_target_batch[:, 0, :, :]), dim=[1, 2])
            signal_amp_norm = signal_peak_amp / (torch.mean(signal_peak_amp) + 1e-6)
            loss_vol = F.mse_loss(pred_vol, signal_amp_norm * torch.mean(pred_vol).detach())

            total_loss = loss_task + phys_weight * loss_vol

            if valid_pairs:
                curr_clean_feat = model.extract_features(X_train_target)
                z_clean = F.normalize(curr_clean_feat, dim=1)
                pair_dists = [1.0 - (z_clean[p[0]] * z_clean[p[1]]).sum() for p in valid_pairs]
                loss_pair = torch.stack(pair_dists).mean()
                total_loss = total_loss + 0.1 * loss_pair

            total_loss.backward()
            optimizer.step()
            scheduler.step()

        # Stage 2: Construct Physics Prototypes for all 5 classes
        model.eval()
        with torch.no_grad():
            # A. Simulation reference prototypes
            feat_sim = model.extract_features(X_source)
            norm_feat_sim = F.normalize(feat_sim, dim=1)
            sim_prototypes = {}
            for c_idx in range(len(UNIQUE_SHAPES)):
                mask_c = (y_source_shape == c_idx)
                if mask_c.sum() > 0:
                    sim_prototypes[c_idx] = norm_feat_sim[mask_c].mean(dim=0)
                else:
                    sim_prototypes[c_idx] = torch.zeros(feat_sim.shape[1], device=device)

            # B. Real training prototypes
            feat_train = model.extract_features(X_train_target)
            norm_feat_train = F.normalize(feat_train, dim=1)
            real_prototypes = {}
            present_classes = set(train_shapes)
            domain_shifts = []

            for c_idx in present_classes:
                mask_train_c = torch.tensor([s == c_idx for s in train_shapes], dtype=torch.bool, device=device)
                c_proto = norm_feat_train[mask_train_c].mean(dim=0)
                real_prototypes[c_idx] = c_proto
                if c_idx in sim_prototypes:
                    shift = c_proto - sim_prototypes[c_idx]
                    domain_shifts.append(shift)

            # Mean Sim-to-Real Domain Shift Vector across seen classes
            if domain_shifts and use_domain_shift:
                avg_shift = torch.stack(domain_shifts).mean(dim=0)
            else:
                avg_shift = torch.zeros(feat_sim.shape[1], device=device)

            # Assemble final adapted prototypes for all 5 classes
            final_prototypes = []
            for c_idx in range(len(UNIQUE_SHAPES)):
                if c_idx in real_prototypes:
                    proto = real_prototypes[c_idx]
                else:
                    # Absent class: inherit from simulation and apply domain shift
                    proto = sim_prototypes[c_idx] + avg_shift
                final_prototypes.append(F.normalize(proto, dim=0))
            P_matrix = torch.stack(final_prototypes)  # Shape: (5, feature_dim)

            # C. Inference on test samples using Cosine Prototype Matching
            X_test = X_target_all[test_idx]
            if use_tta:
                X_h = torch.flip(X_test, dims=[3])
                X_v = torch.flip(X_test, dims=[2])
                X_hv = torch.flip(X_test, dims=[2, 3])

                def get_proto_scores(x_in):
                    f = model.extract_features(x_in)
                    f_norm = F.normalize(f, dim=1)
                    # Cosine similarities scaled by temperature 10.0
                    cos_sim = torch.matmul(f_norm, P_matrix.t()) * 10.0
                    probs = F.softmax(cos_sim, dim=1)
                    reg = model.reg_head(model.regressor_backbone(f))
                    return probs, reg

                p0, r0 = get_proto_scores(X_test)
                ph, rh = get_proto_scores(X_h)
                pv, rv = get_proto_scores(X_v)
                phv, rhv = get_proto_scores(X_hv)

                p_avg = (p0 + ph + pv + phv) / 4.0
                reg_avg = (r0 + rh + rv + rhv) / 4.0
                pred_classes = torch.argmax(p_avg, dim=1).cpu().numpy()
                pred_reg = denormalize_regression_predictions(reg_avg.cpu().numpy(), y_scaler)
            else:
                feat_test = model.extract_features(X_test)
                f_norm = F.normalize(feat_test, dim=1)
                cos_sim = torch.matmul(f_norm, P_matrix.t()) * 10.0
                pred_classes = torch.argmax(cos_sim, dim=1).cpu().numpy()
                reg_test = model.reg_head(model.regressor_backbone(feat_test))
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

    pred_csv = os.path.join(output_dir, f"{method_name.lower()}_predictions.csv")
    summary_csv = os.path.join(output_dir, f"{method_name.lower()}_summary.csv")
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
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr-lora", type=float, default=1e-3)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--phys-weight", type=float, default=0.3)
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--no-calib", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-aug", action="store_true")
    args = parser.parse_args()

    evaluate_pi_npc(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        rank=args.rank,
        alpha=args.alpha,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        phys_weight=args.phys_weight,
        use_calibration=not args.no_calib,
        use_domain_shift=not args.no_shift,
        use_tta=not args.no_tta,
        use_aug=not args.no_aug,
    )
