# -*- coding: utf-8 -*-
"""
===============================================================================
IEEE TRANSACTIONS ABLATION STUDY GENERATOR: NOPINN VS PROPOSED
===============================================================================
Script tạo thư mục độc lập, bảng số liệu chuẩn IEEE (CSV, LaTeX, MD)
và hệ thống đồ thị xuất bản 300 DPI (PNG & PDF) chuẩn IEEE Transactions.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.plot_ieee_utils import setup_ieee_style, save_ieee_figure

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "ieee_ablation_study")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load Data
src_csv = os.path.join(PROJECT_ROOT, "domain_adaptation", "finetune_results", "experimental_ablation_study.csv")
df = pd.read_csv(src_csv)

# Save clean CSV in tables/
clean_csv_path = os.path.join(TABLES_DIR, "ablation_study_summary.csv")
df.to_csv(clean_csv_path, index=False)
print(f"[OK] Saved CSV: {clean_csv_path}")

# 2. Generate IEEE LaTeX Table
latex_content = r"""\begin{table*}[t]
\centering
\caption{Ablation Study and Hyperparameter Calibration of Proposed PINN vs. NoPINN Baseline on Experimental 5\,kHz Repeat-Scan Protocol}
\label{tab:ablation_study}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clcccccccccccc}
\toprule
\textbf{Exp.} & \textbf{Configuration Description} & \textbf{Model} & \textbf{Reset Kendall} & \textbf{Freeze BN} & \textbf{Epochs} & \textbf{LR} & \textbf{Stop Ep.} & \textbf{ACC (\%)} & \textbf{Correct} & \textbf{Overall MAE (mm)} & \textbf{MAE W (mm)} & \textbf{MAE L (mm)} & \textbf{MAE D (mm)} & \textbf{NMAE (\%)} \\
\midrule
"""

for _, r in df.iterrows():
    rk_str = "Yes" if r["Reset_Kendall"] else "No"
    bn_str = "Yes" if r["Freeze_BN"] else "No"
    cfg_name = str(r["Configuration"]).replace("%", r"\%").replace("_", r"\_")
    latex_content += f"{int(r['Experiment_No'])} & {cfg_name} & {r['Model']} & {rk_str} & {bn_str} & {int(r['Epochs'])} & {r['LR']:.0e} & {r['Stop_Epoch']} & {r['ACC(%)']:.1f} & {int(r['Correct'])}/{int(r['Total'])} & {r['MAE_Total(mm)']:.4f} & {r['MAE_W(mm)']:.3f} & {r['MAE_L(mm)']:.3f} & {r['MAE_D(mm)']:.3f} & {r['NMAE(%)']:.2f} \\\\\n"

latex_content += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""

latex_path = os.path.join(TABLES_DIR, "table_ablation_ieee.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex_content)
print(f"[OK] Saved LaTeX table: {latex_path}")


# =============================================================================
# 3. PLOT FIGURES (IEEE STYLE)
# =============================================================================
setup_ieee_style()

# -----------------------------------------------------------------------------
# FIG 1: Key Stages Head-to-Head Comparison (ACC & NMAE)
# -----------------------------------------------------------------------------
key_exp_ids = [1, 2, 6, 9, 18, 20]
df_key = df[df["Experiment_No"].isin(key_exp_ids)].copy().reset_index(drop=True)
key_labels = [
    "NoPINN Raw\n(Baseline, 300 ep)",
    "Proposed Raw\n(EarlyStop @ 66)",
    "Proposed Repo\n(5000 ep, BN train)",
    "NoPINN Freeze BN\n(500 ep)",
    "Proposed Freeze BN\n(1500 ep, lr 2e-4)",
    "Proposed Optimal\n(Frozen BB + Reinit)",
]

fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=300)
x = np.arange(len(df_key))
width = 0.35

c_acc = "#1F77B4"
c_nmae = "#D62728"

rects1 = ax1.bar(x - width/2, df_key["ACC(%)"], width, label="Classification Accuracy (%)", color=c_acc, edgecolor="black", linewidth=0.8, alpha=0.9)
ax1.set_ylabel("Classification Accuracy (%)", color=c_acc, fontsize=10.5, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=c_acc)
ax1.set_ylim(0, 110)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, df_key["NMAE(%)"], width, label="Overall NMAE (%) [Lower is Better]", color=c_nmae, edgecolor="black", linewidth=0.8, alpha=0.85, hatch="//")
ax2.set_ylabel("Overall NMAE (%)", color=c_nmae, fontsize=10.5, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=c_nmae)
ax2.set_ylim(0, 58)

ax1.set_xticks(x)
ax1.set_xticklabels(key_labels, fontsize=8.5)
ax1.set_title("Ablation Study: Sim-to-Real ECT Adaptation Accuracy & Error", fontsize=11, fontweight="bold", pad=12)

# Value annotations
for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=c_acc)

for r in rects2:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=c_nmae)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=9.0)

fig1_base = os.path.join(FIGURES_DIR, "fig1_ablation_acc_nmae_head_to_head")
save_ieee_figure(fig, fig1_base)
plt.close(fig)
print(f"[OK] Generated Fig 1: {fig1_base}")


# -----------------------------------------------------------------------------
# FIG 2: 3D Dimensional Error Breakdown (MAE W, L, D in mm)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
x = np.arange(len(df_key))
width = 0.24

c_w = "#2CA02C"
c_l = "#FF7F0E"
c_d = "#9467BD"

r_w = ax.bar(x - width, df_key["MAE_W(mm)"], width, label=r"Width $W$ Error (mm)", color=c_w, edgecolor="black", linewidth=0.8)
r_l = ax.bar(x, df_key["MAE_L(mm)"], width, label=r"Length $L$ Error (mm)", color=c_l, edgecolor="black", linewidth=0.8)
r_d = ax.bar(x + width, df_key["MAE_D(mm)"], width, label=r"Depth $D$ Error (mm)", color=c_d, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Mean Absolute Error (mm)", fontsize=10.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(key_labels, fontsize=8.5)
ax.set_title("3D Dimensional Quantification Errors Across Calibration Stages", fontsize=11, fontweight="bold", pad=12)
ax.set_ylim(0, 2.1)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate values
for r in r_l:
    h = r.get_height()
    ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#B35806")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fontsize=9.0)

fig2_base = os.path.join(FIGURES_DIR, "fig2_dimensional_mae_breakdown_ablation")
save_ieee_figure(fig, fig2_base)
plt.close(fig)
print(f"[OK] Generated Fig 2: {fig2_base}")


# -----------------------------------------------------------------------------
# FIG 3: Convergence Trajectory of Proposed under Freeze BN (Epochs vs MAE / NMAE)
# -----------------------------------------------------------------------------
df_bn = df[(df["Model"] == "CNN") & (df["Variant"] == "PINN") & (df["Freeze_BN"] == True)].copy()
df_bn_lr1 = df_bn[df_bn["LR"] == 1e-4].sort_values("Epochs")
df_bn_lr2 = df_bn[df_bn["LR"] == 2e-4].sort_values("Epochs")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=300)

# 3.a Overall MAE
ax1.plot(df_bn_lr1["Epochs"], df_bn_lr1["MAE_Total(mm)"], marker="o", linewidth=1.8, color="#1F77B4", label=r"Proposed ($\eta = 1\times 10^{-4}$)")
ax1.plot(df_bn_lr2["Epochs"], df_bn_lr2["MAE_Total(mm)"], marker="s", linewidth=1.8, color="#D62728", label=r"Proposed ($\eta = 2\times 10^{-4}$)")
ax1.axhline(y=0.6308, color="gray", linestyle="--", linewidth=1.2, label="NoPINN Baseline (0.631 mm)")

ax1.set_title(r"(a) Overall 3D Error ($\mathrm{MAE}$)", fontsize=10.0, fontweight="bold")
ax1.set_xlabel("Finetuning Epochs", fontsize=9.5)
ax1.set_ylabel("MAE (mm)", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.set_ylim(0.15, 0.70)

# 3.b Overall NMAE
ax2.plot(df_bn_lr1["Epochs"], df_bn_lr1["NMAE(%)"], marker="o", linewidth=1.8, color="#1F77B4", label=r"Proposed ($\eta = 1\times 10^{-4}$)")
ax2.plot(df_bn_lr2["Epochs"], df_bn_lr2["NMAE(%)"], marker="s", linewidth=1.8, color="#D62728", label=r"Proposed ($\eta = 2\times 10^{-4}$)")
ax2.axhline(y=16.13, color="gray", linestyle="--", linewidth=1.2, label="NoPINN Baseline (16.13%)")

ax2.set_title(r"(b) Normalized Macro Error ($\mathrm{NMAE}$)", fontsize=10.0, fontweight="bold")
ax2.set_xlabel("Finetuning Epochs", fontsize=9.5)
ax2.set_ylabel("NMAE (%)", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.set_ylim(8.0, 21.0)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True, fontsize=8.8)
plt.tight_layout()

fig3_base = os.path.join(FIGURES_DIR, "fig3_effect_of_epochs_and_lr_freeze_bn")
save_ieee_figure(fig, fig3_base)
plt.close(fig)
print(f"[OK] Generated Fig 3: {fig3_base}")


# -----------------------------------------------------------------------------
# FIG 4: Composite IEEE 4-Panel Master Dashboard
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), dpi=300)

# Panel A: Accuracy comparison
ax = axes[0, 0]
acc_data = [90.0, 40.0, 50.0, 80.0]
acc_labels = ["NoPINN\nRaw", "Proposed\nRaw", "Proposed\nRepo", "Proposed\nOptimal"]
b_cols = ["#1F77B4", "#7F7F7F", "#FF7F0E", "#D62728"]
bars = ax.bar(acc_labels, acc_data, color=b_cols, width=0.55, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Accuracy (%)", fontsize=9.0, fontweight="bold")
ax.set_title("(a) Defect Shape Classification (ACC)", fontsize=9.5, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.4)
for b, v in zip(bars, acc_data):
    ax.text(b.get_x() + b.get_width()/2.0, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Panel B: MAE comparison
ax = axes[0, 1]
mae_data = [0.6308, 1.0517, 0.4024, 0.2018]
bars = ax.bar(acc_labels, mae_data, color=b_cols, width=0.55, edgecolor="black", linewidth=0.8)
ax.set_ylabel("MAE (mm)", fontsize=9.0, fontweight="bold")
ax.set_title("(b) 3D Sizing Error (MAE Overall)", fontsize=9.5, fontweight="bold")
ax.set_ylim(0, 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
for b, v in zip(bars, mae_data):
    ax.text(b.get_x() + b.get_width()/2.0, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Panel C: Trajectory MAE
ax = axes[1, 0]
ax.plot(df_bn_lr2["Epochs"], df_bn_lr2["MAE_Total(mm)"], marker="s", color="#D62728", linewidth=1.6, label="Proposed (Freeze BN, lr 2e-4)")
ax.plot(df_bn_lr1["Epochs"], df_bn_lr1["MAE_Total(mm)"], marker="o", color="#1F77B4", linewidth=1.6, label="Proposed (Freeze BN, lr 1e-4)")
ax.axhline(0.6308, color="black", linestyle="--", linewidth=1.1, label="NoPINN Baseline")
ax.set_xlabel("Epochs", fontsize=9.0)
ax.set_ylabel("MAE (mm)", fontsize=9.0)
ax.set_title("(c) Convergence Trajectory of Proposed", fontsize=9.5, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=7.2)

# Panel D: Kendall uncertainty mechanism
ax = axes[1, 1]
components = ["Loss Clf", "Loss W", "Loss L", "Loss D"]
raw_w = [78.58, 0.55, 0.85, 0.59]
reset_w = [1.00, 1.00, 1.00, 1.00]
x_comp = np.arange(len(components))
w_comp = 0.35
ax.bar(x_comp - w_comp/2, raw_w, w_comp, label=r"Raw Pretrained ($\exp(-s)$)", color="#D62728", edgecolor="black", linewidth=0.8)
ax.bar(x_comp + w_comp/2, reset_w, w_comp, label=r"Reset Kendall ($\exp(0) = 1.0$)", color="#2CA02C", edgecolor="black", linewidth=0.8)
ax.set_xticks(x_comp)
ax.set_xticklabels(components, fontsize=8.5)
ax.set_ylabel(r"Gradient Multiplier $\exp(-s)$", fontsize=9.0, fontweight="bold")
ax.set_title(r"(d) Kendall Weighting Re-balancing", fontsize=9.5, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.2, 180)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=7.5)

plt.tight_layout()
fig4_base = os.path.join(FIGURES_DIR, "fig4_unified_ablation_dashboard")
save_ieee_figure(fig, fig4_base)
plt.close(fig)
print(f"[OK] Generated Fig 4: {fig4_base}")

print("\n>>> ALL IEEE ABLATION ARTIFACTS SUCCESSFULLY GENERATED! <<<")
