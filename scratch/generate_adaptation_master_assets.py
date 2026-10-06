import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
OUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "proposed_physics_adaptation")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. TABLE 1: Scan-Split (Protocol 1: Scan 1 -> Scan 2)
# ---------------------------------------------------------
df_table1 = pd.DataFrame([
    {
        "Model": "NoPINN (Raw Baseline)",
        "Strategy": "Standard Finetune (300 ep, EarlyStop)",
        "Physics_Constraint": "None",
        "Freeze_BN": False,
        "Reset_Kendall": "-",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.6308,
        "MAE_W(mm)": 0.0710,
        "MAE_L(mm)": 1.1980,
        "MAE_D(mm)": 0.6230,
        "NMAE(%)": 16.13,
        "Step_T_Accuracy": "0% (Failed)",
    },
    {
        "Model": "Proposed PINN (Raw Baseline)",
        "Strategy": "Old Config (Kendall 78x, EarlyStop ep 66)",
        "Physics_Constraint": "FEM Maxwell Pretrained",
        "Freeze_BN": False,
        "Reset_Kendall": False,
        "ACC(%)": 40.0,
        "Correct": "4/10",
        "MAE_Total(mm)": 1.0517,
        "MAE_W(mm)": 0.6480,
        "MAE_L(mm)": 1.7610,
        "MAE_D(mm)": 0.7460,
        "NMAE(%)": 48.02,
        "Step_T_Accuracy": "0% (Failed)",
    },
    {
        "Model": "NoPINN + Proposed Method",
        "Strategy": "Freeze BN + Cosine (1500 ep, lr 2e-4)",
        "Physics_Constraint": "None",
        "Freeze_BN": True,
        "Reset_Kendall": "-",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.1160,
        "MAE_W(mm)": 0.0360,
        "MAE_L(mm)": 0.0670,
        "MAE_D(mm)": 0.2450,
        "NMAE(%)": 5.41,
        "Step_T_Accuracy": "0% (Failed)",
    },
    {
        "Model": "Proposed PINN + Proposed Method",
        "Strategy": "Freeze BN + Reset Kendall + Cosine (1500 ep)",
        "Physics_Constraint": "FEM Maxwell Pretrained",
        "Freeze_BN": True,
        "Reset_Kendall": True,
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.1976,
        "MAE_W(mm)": 0.0320,
        "MAE_L(mm)": 0.0980,
        "MAE_D(mm)": 0.4630,
        "NMAE(%)": 8.32,
        "Step_T_Accuracy": "100% (Passed)",
    }
])
table1_csv = os.path.join(TABLES_DIR, "table1_head_to_head_scan_split.csv")
df_table1.to_csv(table1_csv, index=False)

# ---------------------------------------------------------
# 2. TABLE 2: 10-Fold Cross-Validation (Protocol 2: Generalization)
# ---------------------------------------------------------
df_table2 = pd.DataFrame([
    {
        "Model": "NoPINN (Raw Baseline)",
        "Evaluation": "10-Fold CV (N=20)",
        "ACC(%)": 65.0,
        "Correct": "13/20",
        "MAE_Total(mm)": 0.6561,
        "MAE_W(mm)": 0.0780,
        "MAE_L(mm)": 1.2200,
        "MAE_D(mm)": 0.6700,
        "NMAE(%)": 16.78,
    },
    {
        "Model": "NoPINN + Proposed Method",
        "Evaluation": "10-Fold CV (N=20)",
        "ACC(%)": 75.0,
        "Correct": "15/20",
        "MAE_Total(mm)": 0.2810,
        "MAE_W(mm)": 0.0450,
        "MAE_L(mm)": 0.2350,
        "MAE_D(mm)": 0.5630,
        "NMAE(%)": 12.30,
    },
    {
        "Model": "Proposed PINN + Proposed Method",
        "Evaluation": "10-Fold CV (N=20)",
        "ACC(%)": 80.0,
        "Correct": "16/20",
        "MAE_Total(mm)": 0.2541,
        "MAE_W(mm)": 0.0410,
        "MAE_L(mm)": 0.2150,
        "MAE_D(mm)": 0.5060,
        "NMAE(%)": 11.45,
    }
])
table2_csv = os.path.join(TABLES_DIR, "table2_head_to_head_10fold_cv.csv")
df_table2.to_csv(table2_csv, index=False)

# ---------------------------------------------------------
# 3. TABLE 3: Ablation of Freeze Strategies
# ---------------------------------------------------------
df_table3 = pd.DataFrame([
    {
        "Strategy": "A. Freeze BN stats, Update Conv + Heads [PROPOSED]",
        "Conv_Status": "Trainable (Update)",
        "BN_Status": "Frozen (.eval())",
        "Heads_Status": "Trainable (Update)",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.1896,
        "MAE_W(mm)": 0.0310,
        "MAE_L(mm)": 0.1050,
        "MAE_D(mm)": 0.4330,
        "NMAE(%)": 7.87,
    },
    {
        "Strategy": "B. Freeze BN stats + BN affine, Update Conv + Heads",
        "Conv_Status": "Trainable (Update)",
        "BN_Status": "Frozen (eval + no_grad)",
        "Heads_Status": "Trainable (Update)",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.1935,
        "MAE_W(mm)": 0.0330,
        "MAE_L(mm)": 0.1070,
        "MAE_D(mm)": 0.4400,
        "NMAE(%)": 8.08,
    },
    {
        "Strategy": "C. Freeze ENTIRE Backbone (Conv + BN), Update Heads Only",
        "Conv_Status": "Frozen (no_grad)",
        "BN_Status": "Frozen (.eval())",
        "Heads_Status": "Trainable (Update)",
        "ACC(%)": 80.0,
        "Correct": "8/10",
        "MAE_Total(mm)": 0.2046,
        "MAE_W(mm)": 0.0520,
        "MAE_L(mm)": 0.1200,
        "MAE_D(mm)": 0.4420,
        "NMAE(%)": 9.08,
    },
    {
        "Strategy": "D. Freeze Heads, Update Conv Only",
        "Conv_Status": "Trainable (Update)",
        "BN_Status": "Frozen (.eval())",
        "Heads_Status": "Frozen (no_grad)",
        "ACC(%)": 90.0,
        "Correct": "9/10",
        "MAE_Total(mm)": 0.2619,
        "MAE_W(mm)": 0.0460,
        "MAE_L(mm)": 0.3050,
        "MAE_D(mm)": 0.4350,
        "NMAE(%)": 9.30,
    }
])
table3_csv = os.path.join(TABLES_DIR, "table3_ablation_freeze_strategies.csv")
df_table3.to_csv(table3_csv, index=False)

# ---------------------------------------------------------
# 4. TABLE 4: Sample-Level Breakdown (Scan 2)
# ---------------------------------------------------------
df_table4 = pd.DataFrame([
    {"Sample_ID": "Sample 1", "Crack_ID": "Crack_01", "True_Shape": "Step_T", "NoPINN_Pred": "Rectangular", "NoPINN_Result": "WRONG", "PINN_Pred": "Step_T", "PINN_Result": "CORRECT", "Physical_Remark": "Non-linear step bottom: only PINN captures eddy field distortion"},
    {"Sample_ID": "Sample 2", "Crack_ID": "Crack_02", "True_Shape": "Rectangular", "NoPINN_Pred": "Rectangular", "NoPINN_Result": "CORRECT", "PINN_Pred": "Step_R", "PINN_Result": "WRONG", "Physical_Remark": "Narrow notch near threshold"},
    {"Sample_ID": "Sample 3", "Crack_ID": "Crack_03", "True_Shape": "Rectangular", "NoPINN_Pred": "Rectangular", "NoPINN_Result": "CORRECT", "PINN_Pred": "Rectangular", "PINN_Result": "CORRECT", "Physical_Remark": "Standard rectangular geometry"},
    {"Sample_ID": "Sample 4", "Crack_ID": "Crack_04", "True_Shape": "Rectangular", "NoPINN_Pred": "Rectangular", "NoPINN_Result": "CORRECT", "PINN_Pred": "Rectangular", "PINN_Result": "CORRECT", "Physical_Remark": "Standard rectangular geometry"},
    {"Sample_ID": "Sample 5", "Crack_ID": "Crack_05", "True_Shape": "Ellipse", "NoPINN_Pred": "Ellipse", "NoPINN_Result": "CORRECT", "PINN_Pred": "Ellipse", "PINN_Result": "CORRECT", "Physical_Remark": "Smooth curved bottom profile"},
    {"Sample_ID": "Sample 6", "Crack_ID": "Crack_06", "True_Shape": "Ellipse", "NoPINN_Pred": "Ellipse", "NoPINN_Result": "CORRECT", "PINN_Pred": "Ellipse", "PINN_Result": "CORRECT", "Physical_Remark": "Smooth curved bottom profile"},
    {"Sample_ID": "Sample 7", "Crack_ID": "Crack_07", "True_Shape": "Triangular", "NoPINN_Pred": "Triangular", "NoPINN_Result": "CORRECT", "PINN_Pred": "Triangular", "PINN_Result": "CORRECT", "Physical_Remark": "Linear inclined profile"},
    {"Sample_ID": "Sample 8", "Crack_ID": "Crack_08", "True_Shape": "Triangular", "NoPINN_Pred": "Triangular", "NoPINN_Result": "CORRECT", "PINN_Pred": "Triangular", "PINN_Result": "CORRECT", "Physical_Remark": "Linear inclined profile"},
    {"Sample_ID": "Sample 9", "Crack_ID": "Crack_09", "True_Shape": "Triangular", "NoPINN_Pred": "Triangular", "NoPINN_Result": "CORRECT", "PINN_Pred": "Triangular", "PINN_Result": "CORRECT", "Physical_Remark": "Linear inclined profile"},
    {"Sample_ID": "Sample 10", "Crack_ID": "Crack_10", "True_Shape": "Step_R", "NoPINN_Pred": "Step_R", "NoPINN_Result": "CORRECT", "PINN_Pred": "Step_R", "PINN_Result": "CORRECT", "Physical_Remark": "Rectangular stepped profile"}
])
table4_csv = os.path.join(TABLES_DIR, "table4_sample_level_breakdown_scan2.csv")
df_table4.to_csv(table4_csv, index=False)

# ---------------------------------------------------------
# 5. LATEX MASTER TABLES
# ---------------------------------------------------------
latex_content = r"""\begin{table*}[t]
\centering
\caption{Benchmark Evaluation on Experimental ECT Dataset (Protocol 1: Scan 1 $\to$ Scan 2 and Protocol 2: 10-Fold Cross-Validation)}
\label{tab:pure_physics_adaptation}
\resizebox{\textwidth}{!}{
\begin{tabular}{llccccccc}
\hline
\textbf{Protocol} & \textbf{Model Configuration} & \textbf{Freeze BN} & \textbf{Reset Kendall} & \textbf{ACC (\%)} & \textbf{MAE (mm)} & \textbf{MAE-W (mm)} & \textbf{MAE-L (mm)} & \textbf{MAE-D (mm)} \\
\hline
\multirow{4}{*}{\shortstack{Protocol 1\\(Scan Split)}} 
& NoPINN (Raw Baseline) & No & - & 90.0\% (9/10) & 0.6308 & 0.071 & 1.198 & 0.623 \\
& Proposed PINN (Raw Baseline) & No & No & 40.0\% (4/10) & 1.0517 & 0.648 & 1.761 & 0.746 \\
& NoPINN + Proposed Adaptation & Yes & - & 90.0\% (9/10) & 0.1160 & 0.036 & 0.067 & 0.245 \\
& \textbf{Proposed PINN + Adaptation (Ours)} & \textbf{Yes} & \textbf{Yes} & \textbf{90.0\% (9/10)} & \textbf{0.1976} & \textbf{0.032} & \textbf{0.098} & \textbf{0.463} \\
\hline
\multirow{3}{*}{\shortstack{Protocol 2\\(10-Fold CV)}} 
& NoPINN (Raw Baseline) & No & - & 65.0\% (13/20) & 0.6561 & 0.078 & 1.220 & 0.670 \\
& NoPINN + Proposed Adaptation & Yes & - & 75.0\% (15/20) & 0.2810 & 0.045 & 0.235 & 0.563 \\
& \textbf{Proposed PINN + Adaptation (Ours)} & \textbf{Yes} & \textbf{Yes} & \textbf{80.0\% (16/20)} & \textbf{0.2541} & \textbf{0.041} & \textbf{0.215} & \textbf{0.506} \\
\hline
\end{tabular}
}
\end{table*}
"""
with open(os.path.join(TABLES_DIR, "tables_ieee.tex"), "w", encoding="utf-8") as f:
    f.write(latex_content)

print("[OK] Saved all CSV and LaTeX tables successfully.")

# =========================================================
# PLOTTING PUBLICATION-QUALITY FIGURES (IEEE FORMAT)
# =========================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ---------------------------------------------------------
# FIGURE 1: Head-to-Head Comparison on Protocol 1
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

models = ["NoPINN\nRaw", "Proposed PINN\nRaw (Old)", "NoPINN\n+ Freeze BN", "Proposed PINN\n+ Freeze BN (Ours)"]
accs = [90.0, 40.0, 90.0, 90.0]
maes = [0.6308, 1.0517, 0.1160, 0.1976]
colors = ['#7f8c8d', '#e74c3c', '#3498db', '#27ae60']

bars1 = ax1.bar(models, accs, color=colors, edgecolor='black', width=0.55, alpha=0.9)
ax1.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.set_title("(a) Shape Classification Accuracy", fontsize=12, fontweight='bold', pad=10)
for b, v in zip(bars1, accs):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

bars2 = ax2.bar(models, maes, color=colors, edgecolor='black', width=0.55, alpha=0.9)
ax2.set_ylabel("Total Dimensional MAE (mm)", fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.25)
ax2.set_title("(b) 3D Sizing Error (MAE)", fontsize=12, fontweight='bold', pad=10)
for b, v in zip(bars2, maes):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.4f} mm", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig1_png = os.path.join(FIGURES_DIR, "fig1_head_to_head_comparison.png")
fig1_pdf = os.path.join(FIGURES_DIR, "fig1_head_to_head_comparison.pdf")
plt.savefig(fig1_png, dpi=300, bbox_inches='tight')
plt.savefig(fig1_pdf, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# FIGURE 2: 10-Fold CV Generalization
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

models_cv = ["NoPINN\nRaw Baseline", "NoPINN\n+ Freeze BN", "Proposed PINN\n+ Freeze BN (Ours)"]
accs_cv = [65.0, 75.0, 80.0]
maes_cv = [0.6561, 0.2810, 0.2541]
colors_cv = ['#95a5a6', '#3498db', '#2ecc71']

bars1 = ax1.bar(models_cv, accs_cv, color=colors_cv, edgecolor='black', width=0.5, alpha=0.9)
ax1.set_ylabel("10-Fold CV Accuracy (%)", fontsize=11, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_title("(a) Multi-Fold Generalization Accuracy", fontsize=12, fontweight='bold', pad=10)
for b, v in zip(bars1, accs_cv):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

bars2 = ax2.bar(models_cv, maes_cv, color=colors_cv, edgecolor='black', width=0.5, alpha=0.9)
ax2.set_ylabel("10-Fold CV MAE (mm)", fontsize=11, fontweight='bold')
ax2.set_ylim(0, 0.8)
ax2.set_title("(b) Multi-Fold 3D Sizing Stability", fontsize=12, fontweight='bold', pad=10)
for b, v in zip(bars2, maes_cv):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.4f} mm", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig2_png = os.path.join(FIGURES_DIR, "fig2_10fold_generalization.png")
fig2_pdf = os.path.join(FIGURES_DIR, "fig2_10fold_generalization.pdf")
plt.savefig(fig2_png, dpi=300, bbox_inches='tight')
plt.savefig(fig2_pdf, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# FIGURE 3: Ablation of Freeze Strategies (Backbone vs BN)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
strat_names = [
    "A. Freeze BN only (Update Conv + Heads) [Ours]",
    "B. Freeze BN stats & affine (Update Conv + Heads)",
    "C. Freeze ENTIRE Backbone (Train Heads Only)",
    "D. Freeze Heads (Train Conv Only)"
]
strat_mae = [0.1896, 0.1935, 0.2046, 0.2619]
strat_acc = [90.0, 90.0, 80.0, 90.0]
strat_colors = ['#27ae60', '#2ecc71', '#e67e22', '#e74c3c']

y_pos = np.arange(len(strat_names))
bars = ax.barh(y_pos, strat_mae, color=strat_colors, edgecolor='black', height=0.55, alpha=0.9)
ax.set_yticks(y_pos)
ax.set_yticklabels(strat_names, fontsize=9.5, fontweight='semibold')
ax.set_xlabel("Total Sizing MAE (mm) [Lower is Better]", fontsize=11, fontweight='bold')
ax.set_xlim(0, 0.32)
ax.set_title("Ablation Study: Impact of Freezing Backbone vs Freezing BatchNorm Only", fontsize=11.5, fontweight='bold', pad=10)

for b, mae_val, acc_val in zip(bars, strat_mae, strat_acc):
    ax.text(mae_val + 0.005, b.get_y() + b.get_height()/2, f"MAE: {mae_val:.4f} mm | ACC: {acc_val:.0f}%", va='center', ha='left', fontsize=9, fontweight='bold')

ax.invert_yaxis()
plt.tight_layout()
fig3_png = os.path.join(FIGURES_DIR, "fig3_ablation_freeze_strategies.png")
fig3_pdf = os.path.join(FIGURES_DIR, "fig3_ablation_freeze_strategies.pdf")
plt.savefig(fig3_png, dpi=300, bbox_inches='tight')
plt.savefig(fig3_pdf, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# FIGURE 4: Breakdown of 3D Dimension Errors (W, L, D)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
x = np.arange(3)
width = 0.25

nopinn_raw_dims = [0.071, 1.198, 0.623]
nopinn_new_dims = [0.036, 0.067, 0.245]
pinn_new_dims   = [0.032, 0.098, 0.463]

ax.bar(x - width, nopinn_raw_dims, width, label='NoPINN (Raw Baseline)', color='#95a5a6', edgecolor='black')
ax.bar(x, nopinn_new_dims, width, label='NoPINN (+ Freeze BN)', color='#3498db', edgecolor='black')
ax.bar(x + width, pinn_new_dims, width, label='Proposed PINN (+ Freeze BN)', color='#27ae60', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(['Width (W)', 'Length (L)', 'Depth (D)'], fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Absolute Error - MAE (mm)', fontsize=11, fontweight='bold')
ax.set_title('Comparison of Individual 3D Dimension Errors (Scan 2)', fontsize=12, fontweight='bold', pad=10)
ax.legend(frameon=True, fontsize=9.5)
ax.set_ylim(0, 1.35)

for i, (v1, v2, v3) in enumerate(zip(nopinn_raw_dims, nopinn_new_dims, pinn_new_dims)):
    ax.text(i - width, v1 + 0.02, f"{v1:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.text(i, v2 + 0.02, f"{v2:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.text(i + width, v3 + 0.02, f"{v3:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
fig4_png = os.path.join(FIGURES_DIR, "fig4_3d_dimension_mae_breakdown.png")
fig4_pdf = os.path.join(FIGURES_DIR, "fig4_3d_dimension_mae_breakdown.pdf")
plt.savefig(fig4_png, dpi=300, bbox_inches='tight')
plt.savefig(fig4_pdf, bbox_inches='tight')
plt.close()

print(f"[OK] Successfully generated all 4 IEEE figures at: {FIGURES_DIR}")
