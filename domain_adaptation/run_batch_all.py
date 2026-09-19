"""
================================================================================
domain_adaptation/run_batch_all.py
Master Batch Runner: Sequentially benchmarks all available models in the repository
(CNN Baseline, CNN PINN, Multitask MLP PINN, and Xiong et al. PINN).

Key Features:
- Supports 43 total checkpoints across CNN and MLP architectures
- Filter by suite: --suite [all | cnn | mlp | xiong | pinn | baseline]
- Resume capability: --skip-completed (skips models already evaluated)
- Pooled 10-Fold LODO Out-of-Fold (OOF) Evaluation across 5 methodologies
- Publication-quality automated plotting (7 IEEE figures per model)
================================================================================
"""

import os
import sys
import argparse
import time
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
    list_all_available_checkpoints,
    get_model_tag,
)
from domain_adaptation.run_all import run_single_model


def is_model_completed(ckpt_dir, output_dir, completed_tags):
    """
    Checks if a model has already been evaluated by:
    1. Matching model tag or run identifier against completed_tags from master summary CSV
    2. Checking if lodo_oof_master_summary.csv exists in either prefixed or legacy results directory
    """
    tag = get_model_tag(ckpt_dir)
    norm = os.path.normpath(ckpt_dir)
    run_name = os.path.basename(norm)
    parts = norm.split(os.sep)
    pct = "model"
    for p in parts:
        if "train_" in p:
            pct = p
            break
    legacy_tag = f"{pct}_{run_name}"

    # Check 1: CSV tags
    if tag in completed_tags or legacy_tag in completed_tags:
        return True
    for c_tag in completed_tags:
        if run_name in c_tag:
            return True

    # Check 2: Results directories on disk (supporting both cnn/ and mlp/ subfolders and legacy root)
    candidate_dirs = [
        os.path.join(output_dir, "cnn", tag),
        os.path.join(output_dir, "cnn", legacy_tag),
        os.path.join(output_dir, "cnn", run_name),
        os.path.join(output_dir, "mlp", tag),
        os.path.join(output_dir, "mlp", legacy_tag),
        os.path.join(output_dir, "mlp", run_name),
        os.path.join(output_dir, tag),
        os.path.join(output_dir, legacy_tag),
        os.path.join(output_dir, run_name),
    ]
    for c_dir in candidate_dirs:
        summary_file = os.path.join(c_dir, "lodo_oof_master_summary.csv")
        if os.path.exists(summary_file):
            try:
                df = pd.read_csv(summary_file)
                if len(df) >= 4:
                    return True
            except Exception:
                pass
    return False


def get_completed_models(summary_csv):
    """Reads the master summary CSV and returns a set of completed model tags."""
    if not os.path.exists(summary_csv):
        return set()
    try:
        df = pd.read_csv(summary_csv)
        if 'Evaluated_Model' in df.columns:
            counts = df['Evaluated_Model'].value_counts()
            completed = set(counts[counts >= 4].index.tolist())
            return completed
    except Exception:
        pass
    return set()


def filter_checkpoints_by_suite(all_ckpts, suite="all"):
    """Filters list of checkpoint directories based on requested suite."""
    suite = suite.lower()
    if suite == "cnn":
        return [c for c in all_ckpts if "Outputs_cnn" in c]
    elif suite == "mlp":
        return [c for c in all_ckpts if "Outputs_mlp_pinn" in c]
    elif suite == "xiong":
        return [c for c in all_ckpts if "Outputs_xiong_pinn" in c]
    elif suite == "all_mlp":
        return [c for c in all_ckpts if "mlp" in c.lower()]
    elif suite == "pinn":
        return [c for c in all_ckpts if "Outputs_cnn_pinn" in c or "Outputs_mlp_pinn" in c or "Outputs_xiong_pinn" in c]
    elif suite == "baseline":
        return [c for c in all_ckpts if "Outputs_cnn_baseline" in c or "Outputs_xiong" in c]
    return all_ckpts


def main():
    parser = argparse.ArgumentParser(description="Master Batch Runner for All Domain Adaptation Models")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "cnn", "mlp", "xiong", "all_mlp", "pinn", "baseline"],
        help="Target suite of models to evaluate (default: all)",
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"), help="Results output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for few-shot adaptation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for few-shot adaptation")
    parser.add_argument("--tta-steps", type=int, default=25, help="TTA adaptation steps")
    parser.add_argument("--skip-completed", action="store_true", default=True, help="Skip models that are already completed (default: True)")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run all models even if already completed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_ckpts = list_all_available_checkpoints()
    selected_ckpts = filter_checkpoints_by_suite(all_ckpts, suite=args.suite)

    global_master_csv = os.path.join(args.output_dir, "master_all_models_summary.csv")
    completed_tags = set()
    if args.skip_completed and not args.force_rerun:
        completed_tags = get_completed_models(global_master_csv)

    to_run = []
    skipped = []
    for ckpt in selected_ckpts:
        tag = get_model_tag(ckpt)
        if (not args.force_rerun) and is_model_completed(ckpt, args.output_dir, completed_tags):
            skipped.append(tag)
        else:
            to_run.append(ckpt)

    print("=" * 85)
    print("MASTER DOMAIN ADAPTATION BATCH BENCHMARK")
    print(f"Target Suite         : {args.suite.upper()}")
    print(f"Total in Repo        : {len(all_ckpts)} models (19 CNN, 12 Multitask MLP, 12 Xiong MLP)")
    print(f"Suite Selected       : {len(selected_ckpts)} models")
    print(f"Already Completed    : {len(skipped)} models (skipped)")
    print(f"Remaining To Execute : {len(to_run)} models")
    print(f"Global Summary CSV   : {global_master_csv}")
    print("=" * 85)

    if not to_run:
        print("\n[INFO] All selected models have already been evaluated! Nothing to run.")
        return

    start_time = time.time()
    for i, ckpt_dir in enumerate(to_run, 1):
        tag = get_model_tag(ckpt_dir)
        print("\n" + "#" * 85)
        print(f"[{i}/{len(to_run)}] EXECUTING MODEL: {tag}")
        print(f"Directory: {ckpt_dir}")
        print("#" * 85)

        m_start = time.time()
        try:
            run_single_model(
                ckpt_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                lr=args.lr,
                tta_steps=args.tta_steps,
            )
            elapsed = time.time() - m_start
            print(f"[OK] Completed {tag} in {elapsed:.1f}s")
        except Exception as e:
            print(f"[ERROR] Failed to run {tag}: {e}")

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print("\n" + "=" * 85)
    print(f"[FINISHED] ALL {len(to_run)} MODELS PROCESSED IN {minutes}m {seconds}s!")
    if os.path.exists(global_master_csv):
        df_master = pd.read_csv(global_master_csv)
        print(f"Total models in master summary: {df_master['Evaluated_Model'].nunique()}")
        print(f"Master summary file: {global_master_csv}")
    print("=" * 85)


if __name__ == "__main__":
    main()
