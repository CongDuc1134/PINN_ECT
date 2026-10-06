import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
OUT_DIR = os.path.join(PROJECT_ROOT, "domain_adaptation", "benchmark_two_protocols_evaluation")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# 1. BẢNG DỮ LIỆU CÁCH ĐÁNH GIÁ 1: PROTOCOL 1 (SCAN 1 -> SCAN 2)
# =============================================================================
df_p1 = pd.DataFrame([
    {
        "Evaluation_Protocol": "Protocol 1 (Scan 1 -> Scan 2)",
        "Model": "NoPINN (Raw Baseline)",
        "Finetune_Strategy": "Standard Finetune (300 ep, EarlyStop)",
        "Physics_Constraint": "No Physics",
        "Accuracy(%)": 90.0,
        "Correct_Samples": "9/10",
        "MAE_Total(mm)": 0.6308,
        "MAE_W(mm)": 0.0710,
        "MAE_L(mm)": 1.1980,
        "MAE_D(mm)": 0.6230,
        "NMAE(%)": 16.13,
        "Step_T_Defect": "MISCLASSIFIED (Failed)",
        "Diagnosis": "High sizing error on 3D dimensions",
    },
    {
        "Evaluation_Protocol": "Protocol 1 (Scan 1 -> Scan 2)",
        "Model": "Proposed PINN (Raw Baseline)",
        "Finetune_Strategy": "Old Config (78x Kendall, Stop ep 66)",
        "Physics_Constraint": "Maxwell Pretrained",
        "Accuracy(%)": 40.0,
        "Correct_Samples": "4/10",
        "MAE_Total(mm)": 1.0517,
        "MAE_W(mm)": 0.6480,
        "MAE_L(mm)": 1.7610,
        "MAE_D(mm)": 0.7460,
        "NMAE(%)": 48.02,
        "Step_T_Defect": "MISCLASSIFIED (Failed)",
        "Diagnosis": "Severely corrupted by 78x Kendall & uncalibrated BN",
    },
    {
        "Evaluation_Protocol": "Protocol 1 (Scan 1 -> Scan 2)",
        "Model": "NoPINN (+ Freeze BN)",
        "Finetune_Strategy": "Freeze BN + lr 2e-4 (1500 ep)",
        "Physics_Constraint": "No Physics",
        "Accuracy(%)": 90.0,
        "Correct_Samples": "9/10",
        "MAE_Total(mm)": 0.1160,
        "MAE_W(mm)": 0.0360,
        "MAE_L(mm)": 0.0670,
        "MAE_D(mm)": 0.2450,
        "NMAE(%)": 5.41,
        "Step_T_Defect": "MISCLASSIFIED (Failed)",
        "Diagnosis": "Memorized duplicate plate depths (Overfitted)",
    },
    {
        "Evaluation_Protocol": "Protocol 1 (Scan 1 -> Scan 2)",
        "Model": "Proposed PINN (+ Freeze BN) [Ours]",
        "Finetune_Strategy": "Freeze BN + Reset Kendall + lr 2e-4 (1500 ep)",
        "Physics_Constraint": "Maxwell Pretrained",
        "Accuracy(%)": 90.0,
        "Correct_Samples": "9/10",
        "MAE_Total(mm)": 0.1976,
        "MAE_W(mm)": 0.0320,
        "MAE_L(mm)": 0.0980,
        "MAE_D(mm)": 0.4630,
        "NMAE(%)": 8.32,
        "Step_T_Defect": "CORRECT (100% Passed)",
        "Diagnosis": "Physically grounded, correct on complex geometries",
    }
])
df_p1.to_csv(os.path.join(TABLES_DIR, "table_protocol_1_scan_split.csv"), index=False)

# =============================================================================
# 2. BẢNG DỮ LIỆU CÁCH ĐÁNH GIÁ 2: PROTOCOL 2 (10-FOLD CROSS-VALIDATION)
# =============================================================================
df_p2 = pd.DataFrame([
    {
        "Evaluation_Protocol": "Protocol 2 (10-Fold Cross-Validation, N=20)",
        "Model": "NoPINN (Raw Baseline)",
        "Finetune_Strategy": "Standard Finetune (300 ep, EarlyStop)",
        "Physics_Constraint": "No Physics",
        "Accuracy(%)": 65.0,
        "Correct_Samples": "13/20",
        "MAE_Total(mm)": 0.6561,
        "MAE_W(mm)": 0.0780,
        "MAE_L(mm)": 1.2200,
        "MAE_D(mm)": 0.6700,
        "NMAE(%)": 16.78,
        "Generalization_Status": "Poor (Degraded by 25% from Scan Split)",
    },
    {
        "Evaluation_Protocol": "Protocol 2 (10-Fold Cross-Validation, N=20)",
        "Model": "NoPINN (+ Freeze BN)",
        "Finetune_Strategy": "Freeze BN + lr 2e-4 (1500 ep)",
        "Physics_Constraint": "No Physics",
        "Accuracy(%)": 75.0,
        "Correct_Samples": "15/20",
        "MAE_Total(mm)": 0.2810,
        "MAE_W(mm)": 0.0450,
        "MAE_L(mm)": 0.2350,
        "MAE_D(mm)": 0.5630,
        "NMAE(%)": 12.30,
        "Generalization_Status": "Moderate (MAE jumped 2.4x from 0.116mm due to overfitting)",
    },
    {
        "Evaluation_Protocol": "Protocol 2 (10-Fold Cross-Validation, N=20)",
        "Model": "Proposed PINN (+ Freeze BN) [Ours]",
        "Finetune_Strategy": "Freeze BN + Reset Kendall + lr 2e-4 (1500 ep)",
        "Physics_Constraint": "Maxwell Pretrained",
        "Accuracy(%)": 80.0,
        "Correct_Samples": "16/20",
        "MAE_Total(mm)": 0.2541,
        "MAE_W(mm)": 0.0410,
        "MAE_L(mm)": 0.2150,
        "MAE_D(mm)": 0.5060,
        "NMAE(%)": 11.45,
        "Generalization_Status": "SUPERIOR (Highest ACC & lowest MAE across all folds)",
    }
])
df_p2.to_csv(os.path.join(TABLES_DIR, "table_protocol_2_10fold_cv.csv"), index=False)

# =============================================================================
# 3. BẢNG TỔNG HỢP SO SÁNH TRỰC DIỆN 2 CÁCH ĐÁNH GIÁ
# =============================================================================
df_combined = pd.DataFrame([
    {
        "Model_Variant": "NoPINN (+ Freeze BN)",
        "Protocol_1_ACC(%)": 90.0,
        "Protocol_1_MAE(mm)": 0.1160,
        "Protocol_2_ACC(%)": 75.0,
        "Protocol_2_MAE(mm)": 0.2810,
        "ACC_Drop_in_P2(%)": -15.0,
        "MAE_Inflation_in_P2(x)": 2.42,
        "Step_T_Classification": "FAILED (Predicted Rectangular)",
        "Core_Finding": "Suffers from memorization/overfitting on fixed split",
    },
    {
        "Model_Variant": "Proposed PINN (+ Freeze BN) [Ours]",
        "Protocol_1_ACC(%)": 90.0,
        "Protocol_1_MAE(mm)": 0.1976,
        "Protocol_2_ACC(%)": 80.0,
        "Protocol_2_MAE(mm)": 0.2541,
        "ACC_Drop_in_P2(%)": -10.0,
        "MAE_Inflation_in_P2(x)": 1.28,
        "Step_T_Classification": "CORRECT (100% Identified)",
        "Core_Finding": "Robust generalization; strictly bounded by Maxwell physics",
    }
])
df_combined.to_csv(os.path.join(TABLES_DIR, "table_protocols_combined_comparison.csv"), index=False)

# =============================================================================
# 4. XUẤT CODE LATEX CHUẨN IEEE SẴN SÀNG CHO BÀI BÁO
# =============================================================================
latex_code = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Comparison Across Two Evaluation Protocols on Experimental ECT Data}
\label{tab:two_protocols_benchmark}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccc}
\hline
\textbf{Evaluation Protocol} & \textbf{Model Configuration} & \textbf{Freeze BN} & \textbf{ACC (\%)} & \textbf{MAE (mm)} & \textbf{MAE-W (mm)} & \textbf{MAE-L (mm)} & \textbf{MAE-D (mm)} \\
\hline
\multirow{4}{*}{\shortstack{\textbf{Protocol 1}\\\textbf{Scan Split}\\\textit{(Scan 1 $\to$ Scan 2, $N=10$)}}} 
& NoPINN (Raw Baseline) & No & 90.0\% (9/10) & 0.6308 & 0.071 & 1.198 & 0.623 \\
& Proposed PINN (Raw Baseline) & No & 40.0\% (4/10) & 1.0517 & 0.648 & 1.761 & 0.746 \\
& NoPINN (+ Freeze BN) & Yes & 90.0\% (9/10) & 0.1160 & 0.036 & 0.067 & 0.245 \\
& \textbf{Proposed PINN (+ Freeze BN) [Ours]} & \textbf{Yes} & \textbf{90.0\% (9/10)} & \textbf{0.1976} & \textbf{0.032} & \textbf{0.098} & \textbf{0.463} \\
\hline
\multirow{3}{*}{\shortstack{\textbf{Protocol 2}\\\textbf{10-Fold CV}\\\textit{(Unbiased, $N=20$)}}} 
& NoPINN (Raw Baseline) & No & 65.0\% (13/20) & 0.6561 & 0.078 & 1.220 & 0.670 \\
& NoPINN (+ Freeze BN) & Yes & 75.0\% (15/20) & 0.2810 & 0.045 & 0.235 & 0.563 \\
& \textbf{Proposed PINN (+ Freeze BN) [Ours]} & \textbf{Yes} & \textbf{80.0\% (16/20)} & \textbf{0.2541} & \textbf{0.041} & \textbf{0.215} & \textbf{0.506} \\
\hline
\end{tabular}
}
\end{table*}
"""
with open(os.path.join(TABLES_DIR, "tables_ieee_two_protocols.tex"), "w", encoding="utf-8") as f:
    f.write(latex_code)

print("[OK] Saved all table CSVs and LaTeX file.")

# =============================================================================
# 5. VẼ CÁC BIỂU ĐỒ CHUẨN XUẤT BẢN IEEE (PNG 300 DPI + VECTOR PDF)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# FIGURE 1: PROTOCOL 1 (SCAN 1 -> SCAN 2)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

models_p1 = ["NoPINN (Raw)", "Proposed PINN (Raw)", "NoPINN", "Proposed PINN"]
acc_p1 = [90.0, 40.0, 90.0, 90.0]
mae_p1 = [0.6308, 1.0517, 0.1160, 0.1976]
colors_p1 = ['#95a5a6', '#e74c3c', '#4A90E2', '#2ECC71']

bars1 = ax1.bar(models_p1, acc_p1, color=colors_p1, edgecolor='black', width=0.55, alpha=0.9)
ax1.set_ylabel("Classification Accuracy (%)")
ax1.set_xlabel("(a) Shape Accuracy", fontweight='bold', labelpad=8)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_xticks(np.arange(len(models_p1)))
ax1.set_xticklabels(models_p1, rotation=15, ha='right')
for b, v in zip(bars1, acc_p1):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

bars2 = ax2.bar(models_p1, mae_p1, color=colors_p1, edgecolor='black', width=0.55, alpha=0.9)
ax2.set_ylabel("Total MAE (mm)")
ax2.set_xlabel("(b) 3D Sizing MAE", fontweight='bold', labelpad=8)
ax2.set_ylim(0, 1.25)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_xticks(np.arange(len(models_p1)))
ax2.set_xticklabels(models_p1, rotation=15, ha='right')
for b, v in zip(bars2, mae_p1):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.4f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
p1_png = os.path.join(FIGURES_DIR, "fig1_protocol_1_scan_split_comparison.png")
p1_pdf = os.path.join(FIGURES_DIR, "fig1_protocol_1_scan_split_comparison.pdf")
plt.savefig(p1_png, dpi=300, bbox_inches='tight')
plt.savefig(p1_pdf, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 2: PROTOCOL 2 (10-FOLD CROSS-VALIDATION)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=300)

models_p2 = ["NoPINN (Raw)", "NoPINN", "Proposed PINN"]
acc_p2 = [65.0, 75.0, 80.0]
mae_p2 = [0.6561, 0.2810, 0.2541]
colors_p2 = ['#95a5a6', '#4A90E2', '#2ECC71']

bars1 = ax1.bar(models_p2, acc_p2, color=colors_p2, edgecolor='black', width=0.5, alpha=0.9)
ax1.set_ylabel("10-Fold CV Accuracy (%)")
ax1.set_xlabel("(a) Multi-Fold Accuracy", fontweight='bold', labelpad=8)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for b, v in zip(bars1, acc_p2):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

bars2 = ax2.bar(models_p2, mae_p2, color=colors_p2, edgecolor='black', width=0.5, alpha=0.9)
ax2.set_ylabel("10-Fold CV MAE (mm)")
ax2.set_xlabel("(b) Multi-Fold 3D Sizing MAE", fontweight='bold', labelpad=8)
ax2.set_ylim(0, 0.8)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for b, v in zip(bars2, mae_p2):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
p2_png = os.path.join(FIGURES_DIR, "fig2_protocol_2_10fold_cv_comparison.png")
p2_pdf = os.path.join(FIGURES_DIR, "fig2_protocol_2_10fold_cv_comparison.pdf")
plt.savefig(p2_png, dpi=300, bbox_inches='tight')
plt.savefig(p2_pdf, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 3: GENERALIZATION STABILITY: PROTOCOL 1 vs PROTOCOL 2
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

x = np.arange(2)
width = 0.30

p1_acc_pair = [90.0, 90.0]
p2_acc_pair = [75.0, 80.0]

rects1 = ax1.bar(x - width/2, p1_acc_pair, width, label='Protocol 1 (Scan Split)', color='#4A90E2', edgecolor='black', alpha=0.9)
rects2 = ax1.bar(x + width/2, p2_acc_pair, width, label='Protocol 2 (10-Fold CV)', color='#2ECC71', edgecolor='black', alpha=0.9)

ax1.set_ylabel('Accuracy (%)')
ax1.set_xlabel('(a) Accuracy Robustness', fontweight='bold', labelpad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(['NoPINN', 'Proposed PINN'])
ax1.set_ylim(0, 105)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for b, v in zip(rects1, p1_acc_pair):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b, v in zip(rects2, p2_acc_pair):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

p1_mae_pair = [0.1160, 0.1976]
p2_mae_pair = [0.2810, 0.2541]

rects3 = ax2.bar(x - width/2, p1_mae_pair, width, label='Protocol 1 (Scan Split)', color='#4A90E2', edgecolor='black', alpha=0.9)
rects4 = ax2.bar(x + width/2, p2_mae_pair, width, label='Protocol 2 (10-Fold CV)', color='#2ECC71', edgecolor='black', alpha=0.9)

ax2.set_ylabel('MAE (mm)')
ax2.set_xlabel('(b) Sizing Error Stability', fontweight='bold', labelpad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(['NoPINN', 'Proposed PINN'])
ax2.set_ylim(0, 0.35)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for b, v in zip(rects3, p1_mae_pair):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.4f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b, v in zip(rects4, p2_mae_pair):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.4f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=True, edgecolor='none')

plt.tight_layout()
p3_png = os.path.join(FIGURES_DIR, "fig3_two_protocols_overfitting_analysis.png")
p3_pdf = os.path.join(FIGURES_DIR, "fig3_two_protocols_overfitting_analysis.pdf")
plt.savefig(p3_png, dpi=300, bbox_inches='tight')
plt.savefig(p3_pdf, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 4: SAMPLE LEVEL STEP DEFECT (STEP_T)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)

categories = ['Sample 1 (Step_T)', 'Samples 2 - 10 (Regular)']
nopinn_scores = [0.0, 100.0]
pinn_scores = [100.0, 88.9]

x_cat = np.arange(len(categories))
w_cat = 0.30

b_no = ax.bar(x_cat - w_cat/2, nopinn_scores, w_cat, label='NoPINN', color='#E74C3C', edgecolor='black', alpha=0.85)
b_pi = ax.bar(x_cat + w_cat/2, pinn_scores, w_cat, label='Proposed PINN', color='#2ECC71', edgecolor='black', alpha=0.85)

ax.set_ylabel('Identification Rate (%)')
ax.set_ylim(0, 115)
ax.set_xticks(x_cat)
ax.set_xticklabels(categories, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, edgecolor='none')

for b, v in zip(b_no, nopinn_scores):
    ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b, v in zip(b_pi, pinn_scores):
    ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
p4_png = os.path.join(FIGURES_DIR, "fig4_sample_level_step_defect_classification.png")
p4_pdf = os.path.join(FIGURES_DIR, "fig4_sample_level_step_defect_classification.pdf")
plt.savefig(p4_png, dpi=300, bbox_inches='tight')
plt.savefig(p4_pdf, bbox_inches='tight')
plt.close()

print(f"[OK] Successfully regenerated all 4 publication-quality figures at: {FIGURES_DIR}")
