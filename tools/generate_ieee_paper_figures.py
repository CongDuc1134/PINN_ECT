# -*- coding: utf-8 -*-
"""
===============================================================================
IEEE TRANSACTIONS PUBLICATION FIGURES GENERATOR FOR ECT DOMAIN ADAPTATION
===============================================================================
Công cụ dòng lệnh tạo lập toàn bộ biểu đồ khoa học chuẩn mực IEEE Transactions (Q1/Q2)
từ kết quả thực nghiệm Domain Adaptation (Sim-to-Real), xuất thành cả 300 DPI PNG và vector PDF.

Sử dụng thư viện chuẩn: domain_adaptation.plot_ieee_utils

CẤU TRÚC THƯ MỤC XUẤT HÌNH:
figures/
├── 01_protocol1_scan_split/
│   ├── confusion_matrices/
│   ├── regression_scatter/
│   ├── loss_convergence/
│   └── performance_metrics/
├── 02_protocol2_10fold_lodo/
│   ├── confusion_matrices/
│   ├── regression_scatter/
│   └── performance_metrics/
└── 03_cross_protocol_comparison/
===============================================================================
"""

import os
import sys
import argparse

# Thiết lập đường dẫn gốc dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.plot_ieee_utils import (
    setup_ieee_style,
    save_dual_figure,
    save_ieee_figure,
    plot_confusion_matrix_ieee,
    plot_regression_scatter_ieee,
    plot_separate_loss_curve_single_model,
    plot_all_methods_loss_grid,
    plot_protocol_metrics_ieee,
    plot_finetuned_two_protocols_comparison,
    generate_all_organized_figures_from_run,
    get_protocol_subdirs,
    get_cross_protocol_dir,
)

# Bí danh tương thích ngược
generate_all_organized_figures = generate_all_organized_figures_from_run


def main():
    parser = argparse.ArgumentParser(
        description="Tái tạo toàn bộ hệ thống biểu đồ IEEE Transactions vào các thư mục riêng lẻ."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="domain_adaptation/finetune_results/latest",
        help="Đường dẫn đến thư mục kết quả finetuning chứa subfolder tables/ (mặc định: domain_adaptation/finetune_results/latest)",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.exists(run_dir):
        print(f"[Lỗi] Thư mục không tồn tại: {run_dir}")
        sys.exit(1)

    print(f"\n[Figures Tool] Bắt đầu kết xuất biểu đồ cho thư mục: {run_dir}")
    generate_all_organized_figures_from_run(run_dir)


if __name__ == "__main__":
    main()
