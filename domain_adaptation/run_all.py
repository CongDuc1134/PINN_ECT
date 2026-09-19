"""
================================================================================
domain_adaptation/run_all.py
Master Orchestrator: Runs all 4 Domain Adaptation Directions on 5kHz Real Dataset
using 10-Fold Leave-One-Defect-Out (LODO) and Pooled Out-of-Fold (OOF) Evaluation.
================================================================================
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from domain_adaptation.utils import find_default_checkpoint_dir, get_model_tag


def main():
    parser = argparse.ArgumentParser(description="Master Runner for Domain Adaptation on Real ECT Data")
    parser.add_argument("--model-dir", type=str, default=None, help="Path or keyword for checkpoint directory (default: auto-detected)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"), help="Base results directory")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for few-shot adaptation (PEFT, MMD)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for few-shot adaptation")
    parser.add_argument("--tta-steps", type=int, default=25, help="TTA adaptation steps")
    args = parser.parse_args()

    model_dir = find_default_checkpoint_dir(args.model_dir)
    model_tag = get_model_tag(model_dir)

    # Dedicated model subfolder: results/<model_tag>/
    model_res_dir = os.path.join(args.output_dir, model_tag)
    dir_0 = os.path.join(model_res_dir, "0_real_only_scratch")
    dir_1 = os.path.join(model_res_dir, "1_zero_shot")
    dir_2 = os.path.join(model_res_dir, "2_few_shot_peft")
    dir_3 = os.path.join(model_res_dir, "3_domain_transfer_mmd")
    dir_4 = os.path.join(model_res_dir, "4_physics_tta")
    for d in [dir_0, dir_1, dir_2, dir_3, dir_4]:
        os.makedirs(d, exist_ok=True)

    print("=" * 85)
    print("DOMAIN ADAPTATION BENCHMARK (10-FOLD LODO POOLED OOF EVALUATION)")
    print(f"Checkpoint Target: {model_dir}")
    print(f"Model Directory  : {model_res_dir}")
    print(f"Subfolders       : 0_real_only_scratch/, 1_zero_shot/, 2_few_shot_peft/, 3_domain_transfer_mmd/, 4_physics_tta/")
    print("=" * 85)

    from importlib import import_module
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
    df_p2, df_s2 = mod_peft.evaluate_few_shot_peft(model_dir=model_dir, output_dir=dir_2, epochs=args.epochs, lr=args.lr)

    print("\n>>> [3/4] Running Direction 3: Supervised Domain Transfer MMD...")
    df_p3, df_s3 = mod_mmd.evaluate_domain_transfer_mmd(model_dir=model_dir, output_dir=dir_3, epochs=args.epochs, lr=args.lr)

    print("\n>>> [4/4] Running Direction 4: Physics-Informed Test-Time Adaptation...")
    df_p4, df_s4 = mod_tta.evaluate_physics_tta(model_dir=model_dir, output_dir=dir_4, steps=args.tta_steps)

    # Combine All Summaries
    df_master_summary = pd.concat([df_s0, df_s1, df_s2, df_s3, df_s4], ignore_index=True)
    df_master_preds = pd.concat([df_p0, df_p1, df_p2, df_p3, df_p4], ignore_index=True)

    # Save within model-specific folder
    master_summary_csv = os.path.join(model_res_dir, "lodo_oof_master_summary.csv")
    master_preds_csv = os.path.join(model_res_dir, "lodo_oof_master_predictions.csv")
    df_master_summary.to_csv(master_summary_csv, index=False)
    df_master_preds.to_csv(master_preds_csv, index=False)

    # Maintain global consolidated table across all models in results/
    df_summary_with_tag = df_master_summary.copy()
    df_summary_with_tag.insert(0, 'Evaluated_Model', model_tag)
    global_master_csv = os.path.join(args.output_dir, "master_all_models_summary.csv")
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
    print(f"[OK] Model summary saved to   : {master_summary_csv}")
    print(f"[OK] Model predictions saved to: {master_preds_csv}")
    print(f"[OK] Global master summary updated: {global_master_csv}")

    # 5. Auto-generate publication plots
    try:
        from domain_adaptation.plot_results import plot_domain_adaptation_results
        plot_domain_adaptation_results(target_dir=model_res_dir)
    except Exception as e:
        print(f"[WARNING] Could not auto-generate plots: {e}")


if __name__ == "__main__":
    main()
