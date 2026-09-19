import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "master_all_models_summary.csv")

if not os.path.exists(csv_path):
    print("CSV not found:", csv_path)
    exit(1)

df = pd.read_csv(csv_path)
print("Total rows:", len(df))
print("Architectures:", df["Architecture"].unique())

for arch in df["Architecture"].unique():
    sub = df[df["Architecture"] == arch]
    print(f"\n==================== Architecture: {arch} ====================")
    print(f"Number of Checkpoints: {sub['Evaluated_Model'].nunique()}")
    for m in sub["Method"].unique():
        sm = sub[sub["Method"] == m]
        print(f"Method: {m:24s} | Acc={sm['Clf_Accuracy (%)'].mean():5.1f}% | Overall MAE={sm['Overall_MAE (mm)'].mean():.4f} mm | L_MAE={sm['MAE_L (mm)'].mean():.4f} mm | W_MAE={sm['MAE_W (mm)'].mean():.4f} mm | D_MAE={sm['MAE_D (mm)'].mean():.4f} mm")
