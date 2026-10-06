import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.plot_ieee_utils import setup_ieee_style, save_ieee_figure

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "pure_physics_study")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. COMPILE DATA FOR BOTH PROTOCOLS (STRICTLY NO REINIT - 100% PRETRAINED)
# -----------------------------------------------------------------------------
df_master = pd.DataFrame([
    {
        "Protocol": "Protocol 1 (Scan Split: Scan 1 -> Scan 2)",
        "Model": "NoPINN (Raw Baseline)",
        "Physics_Preserved": "No Physics",
        "Reinit_Clf": "No",
        "Loss_Function": "CE + MSE (Uncertainty)",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.6308,
        "MAE_W(mm)": 0.0710,
        "MAE_L(mm)": 1.1980,
        "MAE_D(mm)": 0.6230,
        "NMAE(%)": 16.13,
    },
    {
        "Protocol": "Protocol 1 (Scan Split: Scan 1 -> Scan 2)",
        "Model": "Proposed PINN (Raw Baseline)",
        "Physics_Preserved": "Corrupted by 78x Kendall",
        "Reinit_Clf": "No",
        "Loss_Function": "CE (78x) + MSE",
        "ACC(%)": 40.0,
        "Correct": "4/10",
        "MAE_Total(mm)": 1.0517,
        "MAE_W(mm)": 0.6480,
        "MAE_L(mm)": 1.7610,
        "MAE_D(mm)": 0.7460,
        "NMAE(%)": 48.02,
    },
    {
        "Protocol": "Protocol 1 (Scan Split: Scan 1 -> Scan 2)",
        "Model": "Proposed PINN (Standard CE, Freeze BN)",
        "Physics_Preserved": "100% Pretrained Weights",
        "Reinit_Clf": "No",
        "Loss_Function": "Standard CE + MSE",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.1976,
        "MAE_W(mm)": 0.0320,
        "MAE_L(mm)": 0.0980,
        "MAE_D(mm)": 0.4630,
        "NMAE(%)": 8.32,
    },
    {
        "Protocol": "Protocol 1 (Scan Split: Scan 1 -> Scan 2)",
        "Model": "Proposed PINN (Calibrated T=2.0, Optimal)",
        "Physics_Preserved": "100% Pretrained Weights",
        "Reinit_Clf": "No",
        "Loss_Function": "Temp-Scaled CE (T=2.0) + MSE",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.1892,
        "MAE_W(mm)": 0.0330,
        "MAE_L(mm)": 0.1050,
        "MAE_D(mm)": 0.4290,
        "NMAE(%)": 7.94,
    },
    {
        "Protocol": "Protocol 2 (10-Fold LODO: Generalization)",
        "Model": "NoPINN (Raw Baseline)",
        "Physics_Preserved": "No Physics",
        "Reinit_Clf": "No",
        "Loss_Function": "CE + MSE",
        "ACC(%)": 65.0,
        "Correct": "13/20",
        "MAE_Total(mm)": 0.6561,
        "MAE_W(mm)": 0.0659,
        "MAE_L(mm)": 0.9807,
        "MAE_D(mm)": 0.9216,
        "NMAE(%)": 19.30,
    },
    {
        "Protocol": "Protocol 2 (10-Fold LODO: Generalization)",
        "Model": "Proposed PINN (Standard CE, Freeze BN)",
        "Physics_Preserved": "100% Pretrained Weights",
        "Reinit_Clf": "No",
        "Loss_Function": "Standard CE + MSE",
        "ACC(%)": 50.0,
        "Correct": "10/20",
        "MAE_Total(mm)": 0.3804,
        "MAE_W(mm)": 0.0850,
        "MAE_L(mm)": 0.2430,
        "MAE_D(mm)": 0.8120,
        "NMAE(%)": 16.28,
    },
    {
        "Protocol": "Protocol 2 (10-Fold LODO: Generalization)",
        "Model": "Proposed PINN (Calibrated T=2.0, Optimal)",
        "Physics_Preserved": "100% Pretrained Weights",
        "Reinit_Clf": "No",
        "Loss_Function": "Temp-Scaled CE (T=2.0) + MSE",
        "ACC(%)": 55.0,
        "Correct": "11/20",
        "MAE_Total(mm)": 0.3686,
        "MAE_W(mm)": 0.0860,
        "MAE_L(mm)": 0.2430,
        "MAE_D(mm)": 0.7770,
        "NMAE(%)": 15.81,
    },
])

# Save Master CSV
master_csv = os.path.join(TABLES_DIR, "pure_physics_two_protocols_master.csv")
df_master.to_csv(master_csv, index=False)
print(f"[OK] Saved Master CSV: {master_csv}")

# -----------------------------------------------------------------------------
# 2. GENERATE IEEE LATEX TABLE
# -----------------------------------------------------------------------------
latex_table = r"""\begin{table*}[t]
\centering
\caption{Performance Benchmark of Pure Physics-Preserving Adaptation (100\% Pretrained Weights Retained, No Classifier Reinitialization) vs. Baseline NoPINN across Two Experimental Protocols}
\label{tab:pure_physics_benchmark}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Protocol} & \textbf{Model \& Adaptation Strategy} & \textbf{ACC (\%)} & \textbf{Correct} & \textbf{Overall MAE (mm)} & \textbf{MAE $W$ (mm)} & \textbf{MAE $L$ (mm)} & \textbf{MAE $D$ (mm)} & \textbf{NMAE (\%)} \\
\midrule
\multirow{4}{*}{\textbf{\shortstack[l]{Protocol 1:\\Scan Split\\(Repeatability)}}}
  & NoPINN Raw Baseline & 90.0 & 9/10 & 0.6308 & 0.0710 & 1.1980 & 0.6230 & 16.13 \\
  & Proposed PINN Raw (Kendall 78$\times$, EarlyStop) & 40.0 & 4/10 & 1.0517 & 0.6480 & 1.7610 & 0.7460 & 48.02 \\
  & Proposed PINN (Standard CE, Freeze BN) & 90.0 & 9/10 & 0.1976 & 0.0320 & 0.0980 & 0.4630 & 8.32 \\
  & \textbf{Proposed PINN (Calibrated $T=2.0$, Freeze BN)} & \textbf{90.0} & \textbf{9/10} & \textbf{0.1892} & \textbf{0.0330} & \textbf{0.1050} & \textbf{0.4290} & \textbf{7.94} \\
\midrule
\multirow{3}{*}{\textbf{\shortstack[l]{Protocol 2:\\10-Fold LODO\\(Generalization)}}}
  & NoPINN Raw Baseline & 65.0 & 13/20 & 0.6561 & 0.0659 & 0.9807 & 0.9216 & 19.30 \\
  & Proposed PINN (Standard CE, Freeze BN) & 50.0 & 10/20 & 0.3804 & 0.0850 & 0.2430 & 0.8120 & 16.28 \\
  & \textbf{Proposed PINN (Calibrated $T=2.0$, Freeze BN)} & \textbf{55.0} & \textbf{11/20} & \textbf{0.3686} & \textbf{0.0860} & \textbf{0.2430} & \textbf{0.7770} & \textbf{15.81} \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""

latex_path = os.path.join(TABLES_DIR, "table_pure_physics_ieee.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex_table)
print(f"[OK] Saved LaTeX Table: {latex_path}")

# -----------------------------------------------------------------------------
# 3. PLOT FIGURES (IEEE STYLE, 300 DPI, PNG & PDF)
# -----------------------------------------------------------------------------
setup_ieee_style()

# FIG 1: Protocol 1 Head-to-Head Comparison (NoPINN vs Proposed PINN)
p1_df = df_master[df_master["Protocol"].str.contains("Protocol 1")].copy().reset_index(drop=True)
p1_labels = [
    "NoPINN\nRaw",
    "Proposed\nRaw (78x)",
    "Proposed PINN\n(Standard CE)",
    "Proposed PINN\n(Temp Scale T=2)",
]

fig, ax1 = plt.subplots(figsize=(6.8, 3.8), dpi=300)
x = np.arange(len(p1_df))
width = 0.35

c_acc = "#1F77B4"
c_nmae = "#D62728"

rects1 = ax1.bar(x - width/2, p1_df["ACC(%)"], width, label="Classification Accuracy (%)", color=c_acc, edgecolor="black", linewidth=0.8)
ax1.set_ylabel("Classification Accuracy (%)", color=c_acc, fontsize=10, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=c_acc)
ax1.set_ylim(0, 110)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, p1_df["NMAE(%)"], width, label="Overall NMAE (%) [Lower is Better]", color=c_nmae, edgecolor="black", linewidth=0.8, alpha=0.85, hatch="//")
ax2.set_ylabel("Overall NMAE (%)", color=c_nmae, fontsize=10, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=c_nmae)
ax2.set_ylim(0, 58)

ax1.set_xticks(x)
ax1.set_xticklabels(p1_labels, fontsize=8.5)
ax1.set_title("Protocol 1 (Scan Split): Pure Physics-Preserving Adaptation", fontsize=10.5, fontweight="bold", pad=12)

for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=c_acc)

for r in rects2:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=c_nmae)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=8.5)

fig1_path = os.path.join(FIGURES_DIR, "fig1_pure_physics_p1_head_to_head")
save_ieee_figure(fig, fig1_path)
plt.close(fig)
print(f"[OK] Generated Fig 1: {fig1_path}")


# FIG 2: 3D Dimensional Error Breakdown (W, L, D in mm)
fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=300)
width_bar = 0.24
c_w = "#2CA02C"
c_l = "#FF7F0E"
c_d = "#9467BD"

rw = ax.bar(x - width_bar, p1_df["MAE_W(mm)"], width_bar, label=r"Width $W$ Error (mm)", color=c_w, edgecolor="black", linewidth=0.8)
rl = ax.bar(x, p1_df["MAE_L(mm)"], width_bar, label=r"Length $L$ Error (mm)", color=c_l, edgecolor="black", linewidth=0.8)
rd = ax.bar(x + width_bar, p1_df["MAE_D(mm)"], width_bar, label=r"Depth $D$ Error (mm)", color=c_d, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Mean Absolute Error (mm)", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(p1_labels, fontsize=8.5)
ax.set_title("3D Dimensional Error Breakdown (W, L, D in mm)", fontsize=10.5, fontweight="bold", pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 2.05)

for r in [rw, rl, rd]:
    for bar in r:
        h = bar.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.2, rotation=45)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fontsize=8.5)
fig2_path = os.path.join(FIGURES_DIR, "fig2_pure_physics_dimensional_mae_wld")
save_ieee_figure(fig, fig2_path)
plt.close(fig)
print(f"[OK] Generated Fig 2: {fig2_path}")


# FIG 3: Effect of Temperature Scaling on MAE & Generalization
temp_vals = [1.0, 1.3, 1.5, 2.0]
mae_vals = [0.1976, 0.1951, 0.1927, 0.1892]
nmae_vals = [8.32, 8.33, 8.04, 7.94]

fig, ax1 = plt.subplots(figsize=(5.5, 3.4), dpi=300)
c_line1 = "#1F77B4"
c_line2 = "#D62728"

line1 = ax1.plot(temp_vals, mae_vals, marker="o", color=c_line1, linewidth=2.0, markersize=6, label="Overall MAE (mm)")
ax1.set_xlabel(r"Logit Calibration Temperature Parameter $T$", fontsize=10, fontweight="bold")
ax1.set_ylabel("Overall MAE (mm)", color=c_line1, fontsize=10, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=c_line1)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
line2 = ax2.plot(temp_vals, nmae_vals, marker="s", color=c_line2, linewidth=2.0, markersize=6, linestyle="--", label="NMAE (%)")
ax2.set_ylabel("Overall NMAE (%)", color=c_line2, fontsize=10, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=c_line2)

for t, m in zip(temp_vals, mae_vals):
    ax1.annotate(f"{m:.4f}mm", (t, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.0, fontweight="bold", color=c_line1)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True, fontsize=8.5)
ax1.set_title("Effect of Temperature Scaling on Dimension Estimation Accuracy", fontsize=10, fontweight="bold")

fig3_path = os.path.join(FIGURES_DIR, "fig3_temperature_scaling_mae_curve")
save_ieee_figure(fig, fig3_path)
plt.close(fig)
print(f"[OK] Generated Fig 3: {fig3_path}")


# FIG 4: Master 4-Panel Unified Dashboard for Publication
fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32, top=0.92, bottom=0.10)

# (a) Classification Accuracy Head-to-Head
ax_a = fig.add_subplot(gs[0, 0])
ax_a.bar(x, p1_df["ACC(%)"], color=["#AEC7E8", "#FFBB78", "#1F77B4", "#2CA02C"], edgecolor="black", linewidth=0.8)
ax_a.set_xticks(x)
ax_a.set_xticklabels(["NoPINN\nRaw", "Proposed\nRaw", "Proposed\nStandard", "Proposed\nT=2.0"], fontsize=7.8)
ax_a.set_ylabel("ACC (%)", fontsize=9, fontweight="bold")
ax_a.set_ylim(0, 110)
ax_a.set_title("(a) Classification Accuracy (Protocol 1)", fontsize=9.5, fontweight="bold")
ax_a.grid(axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(p1_df["ACC(%)"]):
    ax_a.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8.0, fontweight="bold")

# (b) Overall MAE Reduction
ax_b = fig.add_subplot(gs[0, 1])
ax_b.bar(x, p1_df["MAE_Total(mm)"], color=["#7F7F7F", "#D62728", "#1F77B4", "#2CA02C"], edgecolor="black", linewidth=0.8)
ax_b.set_xticks(x)
ax_b.set_xticklabels(["NoPINN\nRaw", "Proposed\nRaw", "Proposed\nStandard", "Proposed\nT=2.0"], fontsize=7.8)
ax_b.set_ylabel("MAE Total (mm)", fontsize=9, fontweight="bold")
ax_b.set_title("(b) Size MAE Reduction (mm)", fontsize=9.5, fontweight="bold")
ax_b.grid(axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(p1_df["MAE_Total(mm)"]):
    ax_b.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8.0, fontweight="bold")

# (c) Dimensional Breakdown
ax_c = fig.add_subplot(gs[1, 0])
idx_compare = [0, 3] # NoPINN Raw vs Proposed T=2.0
labels_c = ["NoPINN Raw", "Proposed PINN (T=2)"]
w_vals = [p1_df.loc[0, "MAE_W(mm)"], p1_df.loc[3, "MAE_W(mm)"]]
l_vals = [p1_df.loc[0, "MAE_L(mm)"], p1_df.loc[3, "MAE_L(mm)"]]
d_vals = [p1_df.loc[0, "MAE_D(mm)"], p1_df.loc[3, "MAE_D(mm)"]]
xc = np.arange(2)
w_bar = 0.25
ax_c.bar(xc - w_bar, w_vals, w_bar, label="Width W", color=c_w, edgecolor="black")
ax_c.bar(xc, l_vals, w_bar, label="Length L", color=c_l, edgecolor="black")
ax_c.bar(xc + w_bar, d_vals, w_bar, label="Depth D", color=c_d, edgecolor="black")
ax_c.set_xticks(xc)
ax_c.set_xticklabels(labels_c, fontsize=8.5)
ax_c.set_ylabel("MAE (mm)", fontsize=9, fontweight="bold")
ax_c.set_title("(c) 3D Error: NoPINN vs Proposed PINN", fontsize=9.5, fontweight="bold")
ax_c.grid(axis="y", linestyle="--", alpha=0.4)
ax_c.legend(fontsize=7.8, loc="upper right")

# (d) Protocol 2: 10-Fold LODO Error Comparison
ax_d = fig.add_subplot(gs[1, 1])
p2_df = df_master[df_master["Protocol"].str.contains("Protocol 2")].copy().reset_index(drop=True)
labels_d = ["NoPINN\nRaw", "Proposed\nStandard", "Proposed\nT=2.0"]
xd = np.arange(len(p2_df))
ax_d.bar(xd, p2_df["MAE_Total(mm)"], color=["#7F7F7F", "#1F77B4", "#2CA02C"], edgecolor="black", linewidth=0.8)
ax_d.set_xticks(xd)
ax_d.set_xticklabels(labels_d, fontsize=8.5)
ax_d.set_ylabel("10-Fold LODO MAE (mm)", fontsize=9, fontweight="bold")
ax_d.set_title("(d) Generalization Error (Protocol 2)", fontsize=9.5, fontweight="bold")
ax_d.grid(axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(p2_df["MAE_Total(mm)"]):
    ax_d.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.0, fontweight="bold")

fig4_path = os.path.join(FIGURES_DIR, "fig4_pure_physics_master_dashboard")
save_ieee_figure(fig, fig4_path)
plt.close(fig)
print(f"[OK] Generated Fig 4: {fig4_path}")

print("\n>>> ALL PURE PHYSICS ARTIFACTS GENERATED SUCCESSFULLY! <<<")
