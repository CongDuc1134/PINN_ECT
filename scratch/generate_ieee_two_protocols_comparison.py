import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.plot_ieee_utils import (
    setup_ieee_style,
    save_ieee_figure,
    plot_finetuned_two_protocols_comparison,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "ieee_ablation_study")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Compile Data for Both Protocols
df_p1 = pd.DataFrame([
    {
        "Model": "CNN_NoPINN",
        "Stage": "Finetuned",
        "Test_Samples": 10,
        "Total_Correct": 10,
        "ACC(%)": 100.0,
        "MAE_Overall(mm)": 0.6892,
        "MAE_W(mm)": 0.0596,
        "MAE_L(mm)": 1.3849,
        "MAE_D(mm)": 0.6231,
        "NMAE_Overall(%)": 16.19,
    },
    {
        "Model": "CNN_Proposed",
        "Stage": "Finetuned",
        "Test_Samples": 10,
        "Total_Correct": 9,
        "ACC(%)": 90.0,
        "MAE_Overall(mm)": 0.1602,
        "MAE_W(mm)": 0.0512,
        "MAE_L(mm)": 0.1315,
        "MAE_D(mm)": 0.2980,
        "NMAE_Overall(%)": 7.07,
    },
])

df_p2 = pd.DataFrame([
    {
        "Model": "CNN_NoPINN",
        "Stage": "Finetuned",
        "Test_Samples": 20,
        "Total_Correct": 13,
        "ACC(%)": 65.0,
        "MAE_Overall(mm)": 0.6561,
        "MAE_W(mm)": 0.0659,
        "MAE_L(mm)": 0.9807,
        "MAE_D(mm)": 0.9216,
        "NMAE_Overall(%)": 19.30,
    },
    {
        "Model": "CNN_Proposed",
        "Stage": "Finetuned",
        "Test_Samples": 20,
        "Total_Correct": 11,
        "ACC(%)": 55.0,
        "MAE_Overall(mm)": 0.3883,
        "MAE_W(mm)": 0.0860,
        "MAE_L(mm)": 0.2660,
        "MAE_D(mm)": 0.8130,
        "NMAE_Overall(%)": 16.38,
    },
])

# Save Combined CSV
df_combined = pd.concat([
    df_p1.assign(Protocol="Protocol 1 (Scan Split)"),
    df_p2.assign(Protocol="Protocol 2 (10-Fold LODO)"),
], ignore_index=True)

csv_path = os.path.join(TABLES_DIR, "table_two_protocols_comparison.csv")
df_combined.to_csv(csv_path, index=False)
print(f"[OK] Saved Combined CSV: {csv_path}")

# Save LaTeX Table
latex_table = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Comparison of Baseline CNN\_NoPINN vs. Proposed Physics-Informed CNN across Protocol 1 (Repeat-Scan Repeatability) and Protocol 2 (10-Fold Leave-One-Defect-Out Generalization)}
\label{tab:protocols_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Evaluation Protocol} & \textbf{Model} & \textbf{ACC (\%)} & \textbf{Correct} & \textbf{Overall MAE (mm)} & \textbf{MAE $W$ (mm)} & \textbf{MAE $L$ (mm)} & \textbf{MAE $D$ (mm)} & \textbf{NMAE (\%)} \\
\midrule
\multirow{2}{*}{\textbf{Protocol 1: Scan Split (Scan 1 $\to$ Scan 2)}} 
  & CNN\_NoPINN & 100.0 & 10/10 & 0.6892 & 0.0596 & 1.3849 & 0.6231 & 16.19 \\
  & \textbf{CNN\_Proposed (PINN)} & \textbf{90.0} & \textbf{9/10} & \textbf{0.1602} & \textbf{0.0512} & \textbf{0.1315} & \textbf{0.2980} & \textbf{7.07} \\
\midrule
\multirow{2}{*}{\textbf{Protocol 2: 10-Fold LODO (Generalization)}} 
  & CNN\_NoPINN & 65.0 & 13/20 & 0.6561 & 0.0659 & 0.9807 & 0.9216 & 19.30 \\
  & \textbf{CNN\_Proposed (PINN)} & \textbf{55.0} & \textbf{11/20} & \textbf{0.3883} & \textbf{0.0860} & \textbf{0.2660} & \textbf{0.8130} & \textbf{16.38} \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""

tex_path = os.path.join(TABLES_DIR, "table_two_protocols_ieee.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_table)
print(f"[OK] Saved LaTeX Table: {tex_path}")

# Plot IEEE Comparison Figure
fig_path = os.path.join(FIGURES_DIR, "fig5_two_protocols_finetuned_comparison")
plot_finetuned_two_protocols_comparison(df_p1, df_p2, fig_path)
print(f"[OK] Generated Fig 5: {fig_path}.png and .pdf")
