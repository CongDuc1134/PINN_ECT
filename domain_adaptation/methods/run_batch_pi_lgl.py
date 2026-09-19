# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/run_batch_pi_lgl.py
Runs batch benchmark of PI-LGL across all 11 CNN 5% checkpoints:
- 4 Baseline (NoPINN) seeds: 42, 123, 456, 789
- 4 PINN base seeds: 42, 123, 456, 789
- 3 PINN alpha variants: alpha=10, alpha=100, alpha=1000
Generates master benchmark summary CSV and saves individual results into:
domain_adaptation/results/trial_13_pi_lgl_90pct/
================================================================================
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from domain_adaptation.methods.run_pi_lgl_adaptation import evaluate_pi_lgl
from domain_adaptation.utils import get_model_tag

CHECKPOINTS = [
    # 4 Baselines
    ("Baseline", "cnn/Outputs_cnn_baseline/train_05pct/NoPINN_E300_seed_42_run_20260818_135833"),
    ("Baseline", "cnn/Outputs_cnn_baseline/train_05pct/NoPINN_E300_seed_123_run_20260819_010320"),
    ("Baseline", "cnn/Outputs_cnn_baseline/train_05pct/NoPINN_E300_seed_456_run_20260819_044511"),
    ("Baseline", "cnn/Outputs_cnn_baseline/train_05pct/NoPINN_E300_seed_789_run_20260819_082332"),
    # 4 PINN Base Seeds
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_base_a1_W100_E300_seed_42_run_20260819_103124"),
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_base_a1_W100_E300_seed_123_run_20260819_020022"),
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_base_a1_W100_E300_seed_456_run_20260819_054213"),
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_base_a1_W100_E300_seed_789_run_20260821_123448"),
    # 3 PINN Alpha Variants
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_alpha_a10_W100_E300_seed_42_run_20260818_164904"),
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_alpha_a100_W100_E300_seed_42_run_20260818_193407"),
    ("PINN", "cnn/Outputs_cnn_pinn/loss_log1p_norm_sse/train_05pct/PINN_alpha_a1000_W100_E300_seed_42_run_20260818_221917"),
]


def run_master_benchmark():
    base_out_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "trial_13_pi_lgl_90pct")
    os.makedirs(base_out_dir, exist_ok=True)

    summary_rows = []

    for model_type, ckpt in CHECKPOINTS:
        full_ckpt = os.path.join(PROJECT_ROOT, ckpt)
        if not os.path.exists(full_ckpt):
            print(f"[SKIP] Checkpoint not found: {full_ckpt}")
            continue

        tag = get_model_tag(full_ckpt)
        out_subdir = os.path.join(base_out_dir, tag)
        print(f"\n>>> Running PI-LGL for {model_type} ({tag})...")

        df_preds, df_sum, _ = evaluate_pi_lgl(
            model_dir=full_ckpt,
            output_dir=out_subdir,
            epochs=70,
            rank=4,
            alpha=8.0,
            lr_lora=1e-3,
            lr_head=5e-4,
            lap_threshold=0.10,
            use_calibration=True,
        )

        row = df_sum.iloc[0].to_dict()
        row["Model_Tag"] = tag
        row["Checkpoint_Dir"] = ckpt
        row["Type"] = model_type
        summary_rows.append(row)

    df_master = pd.DataFrame(summary_rows)
    master_csv = os.path.join(base_out_dir, "pi_lgl_master_benchmark_summary.csv")
    df_master.to_csv(master_csv, index=False)

    print("\n" + "=" * 100)
    print("MASTER PI-LGL BENCHMARK SUMMARY (ALL 11 CHECKPOINTS)")
    print("=" * 100)
    cols_to_print = ["Type", "Model_Tag", "Clf_Accuracy (%)", "F1_Macro (%)", "Overall_MAE (mm)", "MAE_W (mm)", "MAE_L (mm)", "MAE_D (mm)"]
    print(df_master[cols_to_print].to_string(index=False))
    print("=" * 100)
    print(f"[SAVED] Master summary saved to: {master_csv}")


if __name__ == "__main__":
    run_master_benchmark()
