"""
================================================================================
domain_adaptation/organize_results.py
Organizes benchmark results into clean 'cnn/' and 'mlp/' subdirectories.
Splits and consolidates master summaries:
  - results/cnn/cnn_master_summary.csv
  - results/mlp/mlp_master_summary.csv
  - results/master_all_models_summary.csv
================================================================================
"""

import os
import shutil
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CNN_DIR = os.path.join(RESULTS_DIR, "cnn")
MLP_DIR = os.path.join(RESULTS_DIR, "mlp")


def detect_architecture(folder_or_tag):
    s = folder_or_tag.lower()
    if "mlp" in s or "xiong" in s:
        return "mlp"
    return "cnn"


def organize_results_directory():
    os.makedirs(CNN_DIR, exist_ok=True)
    os.makedirs(MLP_DIR, exist_ok=True)

    moved_count = {"cnn": 0, "mlp": 0}

    # 1. Scan results directory for model folders
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if not os.path.isdir(item_path):
            continue
        if item in ["cnn", "mlp", "__pycache__"]:
            continue

        # Check if it is a model experiment folder
        summary_path = os.path.join(item_path, "lodo_oof_master_summary.csv")
        if os.path.exists(summary_path) or item.startswith("train_") or item.startswith("cnn_") or item.startswith("mlp_") or item.startswith("xiong_"):
            arch = detect_architecture(item)
            dest_dir = CNN_DIR if arch == "cnn" else MLP_DIR
            dest_path = os.path.join(dest_dir, item)

            # Move or replace
            if not os.path.exists(dest_path):
                print(f"[MOVE] {item} -> {arch}/{item}")
                shutil.move(item_path, dest_path)
                moved_count[arch] += 1
            else:
                print(f"[EXISTS] {arch}/{item} already exists")

    # 2. Re-consolidate CSV master summaries
    cnn_summaries = []
    mlp_summaries = []

    # Collect from results/cnn
    for item in os.listdir(CNN_DIR):
        s_file = os.path.join(CNN_DIR, item, "lodo_oof_master_summary.csv")
        if os.path.exists(s_file):
            try:
                df = pd.read_csv(s_file)
                if 'Evaluated_Model' not in df.columns:
                    df.insert(0, 'Evaluated_Model', item)
                if 'Architecture' not in df.columns:
                    df.insert(0, 'Architecture', 'CNN')
                cnn_summaries.append(df)
            except Exception as e:
                print(f"[WARN] Failed to read {s_file}: {e}")

    # Collect from results/mlp
    for item in os.listdir(MLP_DIR):
        s_file = os.path.join(MLP_DIR, item, "lodo_oof_master_summary.csv")
        if os.path.exists(s_file):
            try:
                df = pd.read_csv(s_file)
                if 'Evaluated_Model' not in df.columns:
                    df.insert(0, 'Evaluated_Model', item)
                if 'Architecture' not in df.columns:
                    arch_type = 'MLP_Xiong' if 'xiong' in item.lower() else 'MLP_Multitask'
                    df.insert(0, 'Architecture', arch_type)
                mlp_summaries.append(df)
            except Exception as e:
                print(f"[WARN] Failed to read {s_file}: {e}")

    # Save cnn master summary
    if cnn_summaries:
        df_cnn_master = pd.concat(cnn_summaries, ignore_index=True)
        cnn_csv = os.path.join(CNN_DIR, "cnn_master_summary.csv")
        df_cnn_master.to_csv(cnn_csv, index=False)
        print(f"[SAVED] CNN Master Summary ({len(cnn_summaries)} models): {cnn_csv}")

    # Save mlp master summary
    if mlp_summaries:
        df_mlp_master = pd.concat(mlp_summaries, ignore_index=True)
        mlp_csv = os.path.join(MLP_DIR, "mlp_master_summary.csv")
        df_mlp_master.to_csv(mlp_csv, index=False)
        print(f"[SAVED] MLP Master Summary ({len(mlp_summaries)} models): {mlp_csv}")

    # Save global unified master summary
    all_summaries = cnn_summaries + mlp_summaries
    if all_summaries:
        df_global_master = pd.concat(all_summaries, ignore_index=True)
        global_csv = os.path.join(RESULTS_DIR, "master_all_models_summary.csv")
        df_global_master.to_csv(global_csv, index=False)
        print(f"[SAVED] Global Unified Summary ({len(all_summaries)} models total): {global_csv}")

    print(f"\n[OK] Organization complete: {moved_count['cnn']} CNN models, {moved_count['mlp']} MLP models.")


if __name__ == "__main__":
    organize_results_directory()
