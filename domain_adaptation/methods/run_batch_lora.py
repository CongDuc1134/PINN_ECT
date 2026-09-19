# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/run_batch_lora.py
Batch Runner for PI-LoRA Calibrated Domain Adaptation across all CNN checkpoints
(Seeds 42, 123, 456, 789 and Alphas 10, 100, 1000).
Compiles a consolidated benchmark CSV with Mean +/- Std for PINN vs NoPINN.
================================================================================
"""

import os
import sys
import glob
import time
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from domain_adaptation.utils import (
    list_all_available_checkpoints,
    get_model_tag,
)
from domain_adaptation.methods.run_lora_adaptation import evaluate_lora_pinn


def run_batch_lora_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[START] Batch Benchmark: PI-LoRA Calibrated across CNN models | Device: {device}")

    all_checkpoints = list_all_available_checkpoints("cnn")
    # Filter 5% models (PINN seeds + NoPINN seeds + PINN alphas)
    target_checkpoints = []
    for d in all_checkpoints:
        if "train_05pct" in d:
            target_checkpoints.append(d)

    print(f"[INFO] Found {len(target_checkpoints)} CNN 5% checkpoints to evaluate.")

    results_dir = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "exp_hybrid_pilor")
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    t_start = time.time()

    for idx, ckpt_dir in enumerate(target_checkpoints, 1):
        tag = get_model_tag(ckpt_dir)
        print(f"\n[{idx}/{len(target_checkpoints)}] Evaluating: {tag}...")

        out_sub_dir = os.path.join(results_dir, tag)
        summary_csv = os.path.join(out_sub_dir, "pi_lora_calibrated_summary.csv")

        # Skip if already computed
        if os.path.exists(summary_csv):
            try:
                df_existing = pd.read_csv(summary_csv)
                df_existing["Model_Tag"] = tag
                df_existing["Checkpoint_Dir"] = ckpt_dir
                summary_rows.append(df_existing)
                print(f"  --> [CACHED] Found existing summary: Acc = {df_existing['Clf_Accuracy (%)'].values[0]:.1f}%")
                continue
            except Exception:
                pass

        try:
            _, df_sum, _ = evaluate_lora_pinn(
                model_dir=ckpt_dir,
                output_dir=out_sub_dir,
                epochs=60,
                rank=4,
                alpha=8.0,
                lr_lora=1e-3,
                lr_head=5e-4,
                phys_weight=0.25,
                use_calibration=True,
                use_source_replay=False,
                use_tta=True,
                use_aug=True,
            )
            df_sum["Model_Tag"] = tag
            df_sum["Checkpoint_Dir"] = ckpt_dir
            summary_rows.append(df_sum)
        except Exception as e:
            print(f"  [ERROR] Failed on {tag}: {e}")

    if summary_rows:
        df_all = pd.concat(summary_rows, ignore_index=True)
        # Classify as PINN or Baseline
        df_all["Type"] = df_all["Model_Tag"].apply(lambda t: "Baseline" if "NoPINN" in t else "PINN")

        master_csv = os.path.join(results_dir, "pilor_master_benchmark_summary.csv")
        df_all.to_csv(master_csv, index=False)
        print("\n" + "=" * 80)
        print("PI-LORA CALIBRATED BENCHMARK SUMMARY")
        print("=" * 80)
        print(df_all[["Type", "Model_Tag", "Clf_Accuracy (%)", "Overall_MAE (mm)", "MAE_W (mm)", "MAE_L (mm)", "MAE_D (mm)"]].to_string())

        print("\n" + "=" * 80)
        print("GROUP-WISE MEAN +/- STD (PINN vs BASELINE)")
        print("=" * 80)
        stats = df_all.groupby("Type")[["Clf_Accuracy (%)", "Overall_MAE (mm)", "MAE_W (mm)", "MAE_L (mm)", "MAE_D (mm)"]].agg(["mean", "std"])
        print(stats.to_string())
        print("=" * 80)

    t_total = time.time() - t_start
    print(f"\n[DONE] Batch benchmark completed in {t_total:.1f}s.")


if __name__ == "__main__":
    run_batch_lora_benchmark()
