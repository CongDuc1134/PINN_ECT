import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.utils import list_all_available_checkpoints, get_model_tag
from domain_adaptation.methods.run_pi_lgl_adaptation import evaluate_pi_lgl

mlp_dirs = list_all_available_checkpoints(model_type="mlp")
xiong_dirs = list_all_available_checkpoints(model_type="xiong")

print(f"Found {len(mlp_dirs)} Multitask MLP checkpoints and {len(xiong_dirs)} Xiong MLP checkpoints.")

out_base = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "pi_lgl_all_mlp_models")
os.makedirs(out_base, exist_ok=True)

all_results = []

for arch, dirs in [("MLP_Multitask", mlp_dirs), ("MLP_Xiong", xiong_dirs)]:
    for d in dirs:
        tag = get_model_tag(d)
        print(f"\n>>> Running PI-LGL on {arch}: {tag}")
        out_sub = os.path.join(out_base, tag)
        try:
            res_df, summary_df, _ = evaluate_pi_lgl(d, output_dir=out_sub)
            row = summary_df.iloc[0].to_dict()
            row["Architecture"] = arch
            row["Model_Tag"] = tag
            row["Checkpoint_Dir"] = d
            all_results.append(row)
            print(f"[DONE] {tag} -> Acc: {row['Clf_Accuracy (%)']}%, Overall MAE: {row['Overall_MAE (mm)']:.4f} mm, L_MAE: {row['MAE_L (mm)']:.4f} mm")
        except Exception as e:
            print(f"[ERROR] Failed for {tag}: {e}")

sum_df = pd.DataFrame(all_results)
out_csv = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "pi_lgl_all_mlp_models_summary.csv")
sum_df.to_csv(out_csv, index=False)
print(f"\n[SUCCESS] Saved all MLP PI-LGL summary to {out_csv}")
print("Total rows:", len(sum_df))
