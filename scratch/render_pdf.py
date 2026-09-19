"""
scratch/render_pdf.py
Converts domain_adaptation/PINN_ECT_Report_template.html to PDF via headless browser.
"""

import os
import sys
import base64
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def img_to_b64(path):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def main():
    template_path = os.path.join(PROJECT_ROOT, "domain_adaptation", "PINN_ECT_Report_template.html")
    output_html = os.path.join(PROJECT_ROOT, "domain_adaptation", "PINN_ECT_Report.html")
    output_pdf = os.path.join(PROJECT_ROOT, "domain_adaptation", "PINN_ECT_Scientific_Report.pdf")

    p_lgl = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_pi_lgl")
    p_new = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_new_methods")
    p_ieee = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_ieee_comprehensive")
    p_arch = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_architecture_comparison")

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replacements Architecture Comparison plots (CNN vs Multitask MLP vs Xiong MLP)
    content = content.replace("{{B64_ARCH_F1}}", img_to_b64(os.path.join(p_arch, "fig_unified_master_dashboard.png")))
    content = content.replace("{{B64_ARCH_F2}}", img_to_b64(os.path.join(p_arch, "fig2_data_scaling_comparison_cnn_vs_mlp.png")))
    content = content.replace("{{B64_ARCH_F3}}", img_to_b64(os.path.join(p_arch, "fig3_spatial_dimension_breakdown_architectures.png")))
    content = content.replace("{{B64_ARCH_F4}}", img_to_b64(os.path.join(p_arch, "fig4_pi_lgl_master_architecture_benchmark.png")))

    # Replacements IEEE Comprehensive plots
    content = content.replace("{{B64_IEEE_F1}}", img_to_b64(os.path.join(p_ieee, "fig1_master_acc_mae_19models.png")))
    content = content.replace("{{B64_IEEE_F2}}", img_to_b64(os.path.join(p_ieee, "fig2_dimension_wld_mae_19models.png")))
    content = content.replace("{{B64_IEEE_F3}}", img_to_b64(os.path.join(p_ieee, "fig3_per_class_accuracy.png")))
    content = content.replace("{{B64_IEEE_F4}}", img_to_b64(os.path.join(p_ieee, "fig4_per_class_wld_mae.png")))
    content = content.replace("{{B64_IEEE_F5}}", img_to_b64(os.path.join(p_ieee, "fig5_parity_plots_pred_vs_true.png")))

    # Replacements PI-LGL focus
    content = content.replace("{{B64_F1_LGL}}", img_to_b64(os.path.join(p_lgl, "fig1_pilgl_master_benchmark.png")))
    content = content.replace("{{B64_F2_LGL}}", img_to_b64(os.path.join(p_lgl, "fig2_length_mae_reduction.png")))
    content = content.replace("{{B64_F3_LGL}}", img_to_b64(os.path.join(p_lgl, "fig3_laplacian_distribution.png")))
    content = content.replace("{{B64_F4_LGL}}", img_to_b64(os.path.join(p_lgl, "fig4_confusion_matrix_90pct.png")))

    # Replacements LODO
    content = content.replace("{{B64_F1_NEW}}", img_to_b64(os.path.join(p_new, "fig1_pinn_vs_nopinn_headtohead.png")))
    content = content.replace("{{B64_F2_NEW}}", img_to_b64(os.path.join(p_new, "fig2_dimension_breakdown_mae.png")))
    content = content.replace("{{B64_F3_NEW}}", img_to_b64(os.path.join(p_new, "fig3_ablation_progression.png")))
    content = content.replace("{{B64_F4_NEW}}", img_to_b64(os.path.join(p_new, "fig4_confusion_matrix_best_lora.png")))

    # Replacements Comprehensive Comparison plots (Section 8 - CNN vs MLP全面对比)
    p_comp = os.path.join(PROJECT_ROOT, "domain_adaptation", "results", "plots_final_comprehensive")
    content = content.replace("{{B64_COMP_F1}}", img_to_b64(os.path.join(p_comp, "fig1_grand_method_comparison.png")))
    content = content.replace("{{B64_COMP_F2}}", img_to_b64(os.path.join(p_comp, "fig2_cnn_method_evolution.png")))
    content = content.replace("{{B64_COMP_F3}}", img_to_b64(os.path.join(p_comp, "fig3_architecture_wld_breakdown.png")))
    content = content.replace("{{B64_COMP_F4}}", img_to_b64(os.path.join(p_comp, "fig4_pilgl_architecture_comparison.png")))
    content = content.replace("{{B64_COMP_F5}}", img_to_b64(os.path.join(p_comp, "fig5_pilgl_wld_architecture.png")))
    content = content.replace("{{B64_COMP_F6}}", img_to_b64(os.path.join(p_comp, "fig6_cnn_pinn_vs_base_pilgl.png")))
    content = content.replace("{{B64_COMP_F7}}", img_to_b64(os.path.join(p_comp, "fig7_heatmap_method_arch.png")))
    content = content.replace("{{B64_COMP_F8}}", img_to_b64(os.path.join(p_comp, "fig8_summary_best_methods.png")))

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Generated HTML at: {output_html}")

    # Browser paths
    browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = None
    for b in browsers:
        if os.path.exists(b):
            browser_exe = b
            break

    if not browser_exe:
        print("[ERROR] No browser executable found!")
        return

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf}",
        output_html
    ]
    print(f"[START] Rendering PDF using: {browser_exe}")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(output_pdf) and os.path.getsize(output_pdf) > 1000:
        size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
        print(f"[SUCCESS] PDF generated successfully!")
        print(f"Path: {output_pdf}")
        print(f"Size: {size_mb:.2f} MB")
    else:
        print(f"[ERROR] Failed to generate PDF: {res.stderr}")

if __name__ == "__main__":
    main()
