"""
================================================================================
domain_adaptation/run_all.py
Master Orchestrator: Runs all Domain Adaptation Directions on 5kHz Real Dataset
using 10-Fold Leave-One-Defect-Out (LODO) and Pooled Out-of-Fold (OOF) Evaluation.
Supports running a single model or benchmarking ALL available checkpoints (--all).
================================================================================
"""

import os
import sys
import argparse
import pandas as pd

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from domain_adaptation.utils import (
    find_default_checkpoint_dir,
    get_model_tag,
    list_all_available_checkpoints,
)
from importlib import import_module


def run_single_model(model_dir, output_dir=None, epochs=50, lr=2e-4, tta_steps=25):
    """Executes the full 10-Fold LODO evaluation pipeline for one model checkpoint."""
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "results")

    # Determine architecture folder: 'cnn' or 'mlp'
    norm_m = os.path.normpath(model_dir).lower()
    if "mlp" in norm_m or "xiong" in norm_m:
        arch_subfolder = "mlp"
        arch_name = "MLP_Xiong" if "xiong" in norm_m else "MLP_Multitask"
    else:
        arch_subfolder = "cnn"
        arch_name = "CNN"

    arch_res_dir = os.path.join(output_dir, arch_subfolder)
    os.makedirs(arch_res_dir, exist_ok=True)

    model_tag = get_model_tag(model_dir)
    model_res_dir = os.path.join(arch_res_dir, model_tag)

    dir_0 = os.path.join(model_res_dir, "0_real_only_scratch")
    dir_1 = os.path.join(model_res_dir, "1_zero_shot")
    dir_2 = os.path.join(model_res_dir, "2_few_shot_peft")
    dir_3 = os.path.join(model_res_dir, "3_domain_transfer_mmd")
    dir_4 = os.path.join(model_res_dir, "4_physics_tta")
    for d in [dir_0, dir_1, dir_2, dir_3, dir_4]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 85)
    print(f"BENCHMARKING MODEL: [{model_tag}] ({arch_name})")
    print(f"Architecture Group: {arch_subfolder.upper()}")
    print(f"Checkpoint Path   : {model_dir}")
    print(f"Results Directory : {model_res_dir}")
    print("=" * 85)

    mod_scratch = import_module("domain_adaptation.0_real_only_scratch")
    mod_zero = import_module("domain_adaptation.1_zero_shot")
    mod_peft = import_module("domain_adaptation.2_few_shot_peft")
    mod_mmd = import_module("domain_adaptation.3_domain_transfer_mmd")
    mod_tta = import_module("domain_adaptation.4_physics_tta")

    print("\n>>> [0/4] Running Baseline 0: Real-Only Scratch Training (Ablation)...")
    df_p0, df_s0 = mod_scratch.evaluate_real_only_scratch(model_dir=model_dir, output_dir=dir_0, epochs=100)

    print("\n>>> [1/4] Running Baseline 1: Source-Only Zero-Shot (Simulation Pretrained)...")
    df_p1, df_s1 = mod_zero.evaluate_zero_shot(model_dir=model_dir, output_dir=dir_1)

    print("\n>>> [2/4] Running Direction 2: Few-Shot PEFT Head-Tuning...")
    df_p2, df_s2 = mod_peft.evaluate_few_shot_peft(model_dir=model_dir, output_dir=dir_2, epochs=epochs, lr=lr)

    print("\n>>> [3/4] Running Direction 3: Supervised Domain Transfer MMD...")
    df_p3, df_s3 = mod_mmd.evaluate_domain_transfer_mmd(model_dir=model_dir, output_dir=dir_3, epochs=epochs, lr=lr)

    print("\n>>> [4/4] Running Direction 4: True Physics-Informed Test-Time Adaptation...")
    df_p4, df_s4 = mod_tta.evaluate_physics_tta(model_dir=model_dir, output_dir=dir_4, steps=tta_steps)

    # Combine Summaries
    df_master_summary = pd.concat([df_s0, df_s1, df_s2, df_s3, df_s4], ignore_index=True)
    df_master_preds = pd.concat([df_p0, df_p1, df_p2, df_p3, df_p4], ignore_index=True)

    master_summary_csv = os.path.join(model_res_dir, "lodo_oof_master_summary.csv")
    master_preds_csv = os.path.join(model_res_dir, "lodo_oof_master_predictions.csv")
    df_master_summary.to_csv(master_summary_csv, index=False)
    df_master_preds.to_csv(master_preds_csv, index=False)

    # 1. Update Architecture-specific summary (results/cnn/cnn_master_summary.csv or results/mlp/mlp_master_summary.csv)
    df_summary_with_tag = df_master_summary.copy()
    df_summary_with_tag.insert(0, 'Architecture', arch_name)
    df_summary_with_tag.insert(1, 'Evaluated_Model', model_tag)

    arch_master_csv = os.path.join(arch_res_dir, f"{arch_subfolder}_master_summary.csv")
    if os.path.exists(arch_master_csv):
        try:
            df_old_arch = pd.read_csv(arch_master_csv)
            df_old_arch = df_old_arch[df_old_arch['Evaluated_Model'] != model_tag]
            df_merged_arch = pd.concat([df_old_arch, df_summary_with_tag], ignore_index=True)
        except Exception:
            df_merged_arch = df_summary_with_tag
    else:
        df_merged_arch = df_summary_with_tag
    df_merged_arch.to_csv(arch_master_csv, index=False)

    # 2. Update Global master summary across all architectures (results/master_all_models_summary.csv)
    global_master_csv = os.path.join(output_dir, "master_all_models_summary.csv")
    if os.path.exists(global_master_csv):
        try:
            df_old = pd.read_csv(global_master_csv)
            df_old = df_old[df_old['Evaluated_Model'] != model_tag]
            df_merged = pd.concat([df_old, df_summary_with_tag], ignore_index=True)
        except Exception:
            df_merged = df_summary_with_tag
    else:
        df_merged = df_summary_with_tag
    df_merged.to_csv(global_master_csv, index=False)

    print("\n" + "=" * 85)
    print(f"FINAL COMPARISON TABLE FOR [{model_tag}] (10-FOLD LODO OOF)")
    print("=" * 85)
    print(df_master_summary.to_string(index=False))
    print("=" * 85)
    print(f"[OK] Model summary saved to      : {master_summary_csv}")
    print(f"[OK] Model predictions saved to   : {master_preds_csv}")
    print(f"[OK] {arch_subfolder.upper()} Master summary updated: {arch_master_csv}")
    print(f"[OK] Global master summary updated : {global_master_csv}")

    # Auto-generate publication plots
    try:
        from domain_adaptation.plot_results import plot_domain_adaptation_results
        plot_domain_adaptation_results(target_dir=model_res_dir)
    except Exception as e:
        print(f"[WARNING] Could not auto-generate plots: {e}")

    return df_master_summary


def main():
    parser = argparse.ArgumentParser(description="Master Runner for Domain Adaptation on Real ECT Data")
    parser.add_argument("--model-dir", type=str, default=None, help="Path or keyword for checkpoint directory (default: auto-detected)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"), help="Base results directory")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for few-shot adaptation (PEFT, MMD)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for few-shot adaptation")
    parser.add_argument("--tta-steps", type=int, default=25, help="TTA adaptation steps")
    parser.add_argument("--all", action="store_true", help="Run benchmark across ALL available checkpoints in repo")
    args = parser.parse_args()

    if args.all:
        all_ckpts = list_all_available_checkpoints()
        print("=" * 85)
        print(f"STARTING COMPREHENSIVE BENCHMARK ACROSS ALL {len(all_ckpts)} AVAILABLE MODELS")
        print("=" * 85)
        for i, ckpt_dir in enumerate(all_ckpts, 1):
            print(f"\n>>>>>> MODEL [{i}/{len(all_ckpts)}]: {get_model_tag(ckpt_dir)} <<<<<<")
            try:
                run_single_model(
                    ckpt_dir,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    lr=args.lr,
                    tta_steps=args.tta_steps,
                )
            except Exception as e:
                print(f"[ERROR] Failed to evaluate model {ckpt_dir}: {e}")

        print("\n" + "=" * 85)
        print("ALL MODELS EVALUATED SUCCESSFULLY!")
        global_master = os.path.join(args.output_dir, "master_all_models_summary.csv")
        if os.path.exists(global_master):
            print(f"Comprehensive master results saved in: {global_master}")
            df_g = pd.read_csv(global_master)
            print(f"Total entries in master summary: {len(df_g)}")
        print("=" * 85)
    else:
        model_dir = find_default_checkpoint_dir(args.model_dir)
        run_single_model(
            model_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            tta_steps=args.tta_steps,
        )


if __name__ == "__main__":
    main()
