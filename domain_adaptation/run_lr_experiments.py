#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT NGHIÊN CỨU ẢNH HƯỞNG TỐC ĐỘ HỌC (LEARNING RATE ABLATION STUDY)
ECT 5kHz DOMAIN ADAPTATION BENCHMARK: CNN_Proposed (PINN) vs CNN_NoPINN

Mục đích:
- Khảo sát các mức learning rate (mặc định: 1e-3, 5e-4, 1e-4) để tìm mức LR tối ưu.
- Huấn luyện & Đánh giá ĐỒNG THỜI cả hai mô hình đối chứng cùng một phương pháp:
    1. CNN_Proposed (Có ràng buộc vật lý Physics PINN Loss)
    2. CNN_NoPINN   (Thuần dữ liệu Supervised Data Loss)
- Lưu trữ TÁCH BẠCH mỗi LR vào một thư mục con riêng biệt, ngăn nắp, dễ trích xuất bảng & hình:
    experiments_lr_study/
    ├── LR_1e-3/             (Figures, tables, checkpoints, summary cho cả PINN & NoPINN)
    ├── LR_5e-4/             (Figures, tables, checkpoints, summary cho cả PINN & NoPINN)
    ├── LR_1e-4/             (Figures, tables, checkpoints, summary cho cả PINN & NoPINN)
    └── SUMMARY_COMPARISON/  (Bảng đối chiếu PINN vs NoPINN CSV, báo cáo TXT, biểu đồ cột nhóm IEEE)
================================================================================
"""

import os
import sys
import argparse
import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Định tuyến đường dẫn gốc dự án
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
from domain_adaptation.benchmark_evaluation_protocols import (
    MODELS_SUITE,
    run_protocol_1_scan_split,
    run_protocol_2_kfold_defect_split,
    set_reproducible_seed,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Khảo sát các mức Learning Rate & So sánh đối chứng PINN vs NoPINN cùng phương pháp"
    )
    parser.add_argument(
        "--lrs",
        type=str,
        default="1e-3,5e-4,1e-4",
        help="Danh sách các mức Learning Rate cách nhau bởi dấu phẩy (mặc định: '1e-3,5e-4,1e-4')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Số epochs huấn luyện cho mỗi mức LR (mặc định: 500)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["1", "2"],
        default="1",
        help="Giao thức thực thi: '1' (Scan Split: Scan 1 -> Test mù Scan 2), hoặc '2' (10-Fold LODO). Mặc định: '1'",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cnn_proposed,cnn_nopinn",
        help="Mô hình khảo sát: 'cnn_proposed,cnn_nopinn', 'cnn_proposed', 'all' (mặc định: 'cnn_proposed,cnn_nopinn')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "experiments_lr_study"),
        help="Thư mục mẹ chứa toàn bộ kết quả khảo sát LR",
    )
    parser.add_argument(
        "--save_final",
        action="store_true",
        default=False,
        help="Lưu mô hình tại epoch cuối cùng thay vì lấy checkpoint tốt nhất (mặc định: False - lấy Best Val Loss)",
    )
    parser.add_argument(
        "--freeze_backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Đóng băng Backbone khi finetuning",
    )
    parser.add_argument(
        "--no_freeze_backbone",
        dest="freeze_backbone",
        action="store_false",
        default=False,
        help="Không đóng băng Backbone, finetune toàn bộ mô hình (mặc định: False - finetune toàn bộ)",
    )
    parser.add_argument(
        "--reset_kendall",
        dest="reset_kendall",
        action="store_true",
        default=True,
        help="Reset tham số Kendall về 0.0 khi finetuning (mặc định: True)",
    )
    parser.add_argument(
        "--no_reset_kendall",
        dest="reset_kendall",
        action="store_false",
        help="Giữ nguyên tham số Kendall từ pretrained checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán (cuda/cpu)",
    )
    return parser.parse_args()


def format_lr_folder_name(lr: float) -> str:
    """Tạo tên folder ngắn gọn, chuẩn mực theo chuẩn IEEE (ví dụ: LR_1e-3, LR_5e-4)."""
    if lr >= 1e-2:
        return f"LR_{lr:g}"
    elif lr >= 1e-3:
        if abs(lr - 1e-3) < 1e-7:
            return "LR_1e-3"
        return f"LR_{lr:.1e}".replace(".0e", "e")
    elif lr >= 1e-4:
        if abs(lr - 5e-4) < 1e-7:
            return "LR_5e-4"
        elif abs(lr - 1e-4) < 1e-7:
            return "LR_1e-4"
        return f"LR_{lr:.1e}".replace(".0e", "e")
    else:
        return f"LR_{lr:.1e}".replace(".0e", "e")


def plot_grouped_lr_comparison_figure(df_pinn_vs_nopinn: pd.DataFrame, out_png: str, out_pdf: str):
    """
    Vẽ đồ thị cột nhóm chuẩn xuất bản IEEE so sánh song song giữa CNN_Proposed (PINN) và CNN_NoPINN
    trên từng mức Learning Rate.
    """
    try:
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.autolayout": True,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

        lr_labels = df_pinn_vs_nopinn["LR_Label"].tolist()
        x = np.arange(len(lr_labels))
        bar_width = 0.35

        # Dữ liệu Accuracy
        pinn_acc = df_pinn_vs_nopinn["PINN_ACC(%)"].tolist()
        nopinn_acc = df_pinn_vs_nopinn["NoPINN_ACC(%)"].tolist()

        # Dữ liệu MAE
        pinn_mae = df_pinn_vs_nopinn["PINN_MAE(mm)"].tolist()
        nopinn_mae = df_pinn_vs_nopinn["NoPINN_MAE(mm)"].tolist()

        # =========================================================================
        # (a) Subplot 1: Shape Classification Accuracy (%)
        # =========================================================================
        rects1 = ax1.bar(
            x - bar_width / 2, pinn_acc, bar_width,
            label="CNN_Proposed (PINN)", color="#1f77b4", edgecolor="black", linewidth=0.9
        )
        rects2 = ax1.bar(
            x + bar_width / 2, nopinn_acc, bar_width,
            label="CNN_NoPINN (Data-only)", color="#ff7f0e", edgecolor="black", linewidth=0.9
        )
        ax1.set_title("(a) Shape Classification Accuracy", fontweight="bold", pad=10)
        ax1.set_ylabel("Accuracy (%)", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(lr_labels, fontweight="bold")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.9)

        for rect in rects1:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        # =========================================================================
        # (b) Subplot 2: Overall Sizing MAE (mm)
        # =========================================================================
        rects3 = ax2.bar(
            x - bar_width / 2, pinn_mae, bar_width,
            label="CNN_Proposed (PINN)", color="#2ca02c", edgecolor="black", linewidth=0.9
        )
        rects4 = ax2.bar(
            x + bar_width / 2, nopinn_mae, bar_width,
            label="CNN_NoPINN (Data-only)", color="#d62728", edgecolor="black", linewidth=0.9
        )
        ax2.set_title("(b) 3D Sizing Overall MAE", fontweight="bold", pad=10)
        ax2.set_ylabel("Mean Absolute Error (mm)", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(lr_labels, fontweight="bold")
        max_val = max(max(pinn_mae), max(nopinn_mae)) if (pinn_mae and nopinn_mae) else 1.0
        ax2.set_ylim(0, max_val * 1.25)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left", framealpha=0.9)

        for rect in rects3:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2.0, h + (max_val * 0.02), f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        for rect in rects4:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2.0, h + (max_val * 0.02), f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        plt.close()
    except Exception as e:
        print(f"[WARN] Không thể vẽ biểu đồ so sánh LR: {e}")


def main():
    args = parse_arguments()
    set_reproducible_seed(42)

    # 1. Xử lý danh sách các mức Learning Rate
    lr_list = []
    for s in args.lrs.split(","):
        s_clean = s.strip()
        if s_clean:
            try:
                lr_list.append(float(s_clean))
            except ValueError:
                raise ValueError(f"Giá trị LR không hợp lệ: '{s_clean}'")

    if not lr_list:
        raise ValueError("Danh sách LR trống!")

    # 2. Xử lý danh sách mô hình mục tiêu (Mặc định cả Proposed và NoPINN)
    alias_map = {
        "cnn_proposed": "CNN_Proposed",
        "proposed": "CNN_Proposed",
        "pinn": "CNN_Proposed",
        "cnn_nopinn": "CNN_NoPINN",
        "nopinn": "CNN_NoPINN",
        "mlp": "MLP",
        "xiong": "XIONG",
    }
    req_keys = [alias_map.get(k.strip().lower(), k.strip()) for k in args.models.split(",")]
    models_to_run = [m for m in MODELS_SUITE if m["key"].lower() in [k.lower() for k in req_keys]]
    if not models_to_run:
        models_to_run = [m for m in MODELS_SUITE if m["key"] in ["CNN_Proposed", "CNN_NoPINN"]]

    use_final_epoch = args.save_final
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("CHƯƠNG TRÌNH KHẢO SÁT LEARNING RATE & SO SÁNH ĐỐI CHỨNG PINN VS NoPINN")
    print("=" * 80)
    print(f"Danh sách Learning Rates : {lr_list}")
    print(f"Số Epochs mỗi lần chạy  : {args.epochs}")
    print(f"Giao thức thực thi       : Protocol {args.protocol}")
    print(f"Mô hình đối chứng        : {[m['key'] for m in models_to_run]}")
    print(f"Chế độ lấy kết quả       : {'Epoch cuối cùng' if use_final_epoch else 'Best Val Loss (Mặc định)'}")
    print(f"Đóng băng Backbone       : {args.freeze_backbone} {'(Freeze)' if args.freeze_backbone else '(Finetune toàn bộ mô hình)'}")
    print(f"Reset Kendall Uncertainty: {args.reset_kendall}")
    print(f"Thư mục lưu trữ mẹ       : {args.output_dir}")
    print("=" * 80 + "\n")

    all_detailed_records = []
    pinn_vs_nopinn_by_lr = []

    # 3. Lặp tuần tự qua từng mức LR
    for idx, lr in enumerate(lr_list, start=1):
        lr_folder_name = format_lr_folder_name(lr)
        lr_dir = os.path.join(args.output_dir, lr_folder_name)
        os.makedirs(lr_dir, exist_ok=True)
        os.makedirs(os.path.join(lr_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(lr_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(lr_dir, "checkpoints"), exist_ok=True)

        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"[{idx}/{len(lr_list)}] BẮT ĐẦU CHẠY CẢ 2 MÔ HÌNH VỚI LEARNING RATE: {lr:g} ({lr_folder_name})")
        print(f"Mô hình đối chứng: {[m['key'] for m in models_to_run]}")
        print(f"Thư mục lưu riêng biệt: {lr_dir}")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

        start_time = datetime.datetime.now()

        # Thực thi giao thức tương ứng cho toàn bộ danh sách mô hình
        if args.protocol == "1":
            df_res, hists = run_protocol_1_scan_split(
                models_to_test=models_to_run,
                epochs=args.epochs,
                lr=lr,
                freeze_backbone=args.freeze_backbone,
                device=args.device,
                output_dir=lr_dir,
                reset_kendall=args.reset_kendall,
                early_stopping=False,
                use_final_epoch=use_final_epoch,
                fixed_lr=True,
            )
        else:
            df_res, _ = run_protocol_2_kfold_defect_split(
                models_to_test=models_to_run,
                epochs=args.epochs,
                lr=lr,
                freeze_backbone=args.freeze_backbone,
                device=args.device,
                output_dir=lr_dir,
                reset_kendall=args.reset_kendall,
                early_stopping=False,
                use_final_epoch=use_final_epoch,
                fixed_lr=True,
            )

        duration_sec = (datetime.datetime.now() - start_time).total_seconds()

        # Thu thập thông số cho từng mô hình trong lượt chạy này
        lr_summary_lines = [
            "=" * 80,
            f"KẾT QUẢ ĐỐI CHỨNG TẠI MỨC TỐC ĐỘ HỌC: {lr_folder_name} (LR = {lr})",
            "=" * 80,
            f"Thời gian chạy       : {duration_sec:.1f} giây",
            f"Số epochs            : {args.epochs}",
            f"Đóng băng Backbone   : {args.freeze_backbone}",
            f"Chế độ lấy kết quả   : {'Epoch cuối cùng' if use_final_epoch else 'Best Val Loss'}",
            "-" * 80,
        ]

        # Trích xuất kết quả chi tiết từng mô hình
        pinn_metrics = {}
        nopinn_metrics = {}

        if df_res is not None and not df_res.empty:
            sub = df_res[df_res["Stage"] == "Finetuned"] if "Stage" in df_res.columns else df_res
            for _, row in sub.iterrows():
                m_name = str(row.get("Model", "Unknown"))
                acc = float(row.get("ACC(%)", np.nan))
                correct_str = str(row.get("Total_Correct", "N/A"))
                total_s = str(row.get("Test_Samples", "10"))
                mae_avg = float(row.get("MAE_Overall(mm)", np.nan))
                mae_w = float(row.get("MAE_W(mm)", np.nan))
                mae_l = float(row.get("MAE_L(mm)", np.nan))
                mae_d = float(row.get("MAE_D(mm)", np.nan))

                record = {
                    "LR_Label": lr_folder_name,
                    "Learning_Rate": lr,
                    "Model": m_name,
                    "Accuracy(%)": acc,
                    "Correct": f"{correct_str}/{total_s}",
                    "MAE_Overall(mm)": mae_avg,
                    "MAE_W(mm)": mae_w,
                    "MAE_L(mm)": mae_l,
                    "MAE_D(mm)": mae_d,
                    "Duration(s)": round(duration_sec, 1),
                }
                all_detailed_records.append(record)

                lr_summary_lines.append(f"\n[Mô hình: {m_name}]")
                lr_summary_lines.append(f"  * Độ chính xác (ACC)    : {acc:.1f}% ({correct_str}/{total_s})")
                lr_summary_lines.append(f"  * Sai số trung bình MAE : {mae_avg:.4f} mm")
                lr_summary_lines.append(f"    - MAE Chiều rộng (W)  : {mae_w:.4f} mm")
                lr_summary_lines.append(f"    - MAE Chiều dài (L)   : {mae_l:.4f} mm")
                lr_summary_lines.append(f"    - MAE Độ sâu (D)      : {mae_d:.4f} mm")

                if "Proposed" in m_name:
                    pinn_metrics = record
                elif "NoPINN" in m_name:
                    nopinn_metrics = record

        # So sánh chênh lệch giữa PINN và NoPINN tại LR này
        if pinn_metrics and nopinn_metrics:
            delta_acc = pinn_metrics["Accuracy(%)"] - nopinn_metrics["Accuracy(%)"]
            delta_mae = nopinn_metrics["MAE_Overall(mm)"] - pinn_metrics["MAE_Overall(mm)"]
            mae_improv_pct = (delta_mae / nopinn_metrics["MAE_Overall(mm)"] * 100.0) if nopinn_metrics["MAE_Overall(mm)"] > 0 else 0.0

            lr_summary_lines.append("\n" + "-" * 80)
            lr_summary_lines.append(f">> SO SÁNH TRỰC DIỆN: CNN_Proposed (PINN) vs CNN_NoPINN tại {lr_folder_name}:")
            lr_summary_lines.append(f"  * Chênh lệch Accuracy   : {delta_acc:+.1f}% ({pinn_metrics['Accuracy(%)']:.1f}% vs {nopinn_metrics['Accuracy(%)']:.1f}%)")
            lr_summary_lines.append(f"  * Giảm sai số MAE       : {delta_mae:.4f} mm (Tốt hơn {mae_improv_pct:.1f}%)")
            lr_summary_lines.append(f"    - PINN MAE = {pinn_metrics['MAE_Overall(mm)']:.4f} mm | NoPINN MAE = {nopinn_metrics['MAE_Overall(mm)']:.4f} mm")

            pinn_vs_nopinn_by_lr.append({
                "LR_Label": lr_folder_name,
                "Learning_Rate": lr,
                "PINN_ACC(%)": pinn_metrics["Accuracy(%)"],
                "NoPINN_ACC(%)": nopinn_metrics["Accuracy(%)"],
                "Delta_ACC(%)": round(delta_acc, 2),
                "PINN_MAE(mm)": pinn_metrics["MAE_Overall(mm)"],
                "NoPINN_MAE(mm)": nopinn_metrics["MAE_Overall(mm)"],
                "MAE_Improvement(%)": round(mae_improv_pct, 2),
                "PINN_MAE_W(mm)": pinn_metrics["MAE_W(mm)"],
                "NoPINN_MAE_W(mm)": nopinn_metrics["MAE_W(mm)"],
                "PINN_MAE_L(mm)": pinn_metrics["MAE_L(mm)"],
                "NoPINN_MAE_L(mm)": nopinn_metrics["MAE_L(mm)"],
                "PINN_MAE_D(mm)": pinn_metrics["MAE_D(mm)"],
                "NoPINN_MAE_D(mm)": nopinn_metrics["MAE_D(mm)"],
            })

        lr_summary_lines.append("\n" + "=" * 80)
        summary_txt_path = os.path.join(lr_dir, "run_summary.txt")
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lr_summary_lines))

    # 4. TỔNG HỢP VÀ SO SÁNH ĐỐI CHỨNG VÀO FOLDER SUMMARY_COMPARISON
    summary_dir = os.path.join(args.output_dir, "SUMMARY_COMPARISON")
    os.makedirs(summary_dir, exist_ok=True)

    df_all_details = pd.DataFrame(all_detailed_records)
    csv_all_path = os.path.join(summary_dir, "comparison_all_models_by_lr.csv")
    df_all_details.to_csv(csv_all_path, index=False)

    df_pinn_vs_nopinn = pd.DataFrame(pinn_vs_nopinn_by_lr)
    csv_comparison_path = os.path.join(summary_dir, "comparison_pinn_vs_nopinn_by_lr.csv")
    df_pinn_vs_nopinn.to_csv(csv_comparison_path, index=False)

    # Vẽ biểu đồ nhóm IEEE so sánh PINN vs NoPINN trên các mức LR
    fig_png_path = os.path.join(summary_dir, "fig_comparison_pinn_vs_nopinn_by_lr.png")
    fig_pdf_path = os.path.join(summary_dir, "fig_comparison_pinn_vs_nopinn_by_lr.pdf")
    if not df_pinn_vs_nopinn.empty:
        plot_grouped_lr_comparison_figure(df_pinn_vs_nopinn, fig_png_path, fig_pdf_path)

    # Xuất báo cáo tổng quan học thuật
    report_lines = [
        "=" * 85,
        "BÁO CÁO ĐỐI CHỨNG HỌC THUẬT: SO SÁNH CNN_Proposed (PINN) vs CNN_NoPINN CÙNG PHƯƠNG PHÁP",
        "=" * 85,
        f"Giao thức thực thi : Protocol {args.protocol}",
        f"Số Epochs          : {args.epochs}",
        f"Đóng băng Backbone : {args.freeze_backbone} {'(Freeze)' if args.freeze_backbone else '(Finetune toàn bộ mô hình)'}",
        f"Chế độ lấy kết quả : {'Epoch cuối cùng' if use_final_epoch else 'Best Val Loss (Mặc định)'}",
        f"Thư mục lưu trữ    : {args.output_dir}",
        "-" * 85,
        "\nBẢNG SỐ LIỆU ĐỐI CHIẾU TRỰC DIỆN (PINN vs NoPINN THEO TỪNG MỨC LR - DÙNG CHO BÀI BÁO IEEE):",
    ]

    if not df_pinn_vs_nopinn.empty:
        disp_cols = ["LR_Label", "PINN_ACC(%)", "NoPINN_ACC(%)", "Delta_ACC(%)", "PINN_MAE(mm)", "NoPINN_MAE(mm)", "MAE_Improvement(%)"]
        report_lines.append(df_pinn_vs_nopinn[disp_cols].to_string(index=False))

        best_pinn_row = df_pinn_vs_nopinn.loc[df_pinn_vs_nopinn["PINN_ACC(%)"].idxmax()]
        best_mae_row = df_pinn_vs_nopinn.loc[df_pinn_vs_nopinn["PINN_MAE(mm)"].idxmin()]

        report_lines.extend([
            "\n" + "-" * 85,
            f">> MỨC LR TỐI ƯU CỦA PINN (ĐỘ CHÍNH XÁC CAO NHẤT): {best_pinn_row['LR_Label']} (PINN = {best_pinn_row['PINN_ACC(%)']:.1f}% vs NoPINN = {best_pinn_row['NoPINN_ACC(%)']:.1f}%)",
            f">> MỨC LR TỐI ƯU CỦA PINN (SAI SỐ MAE THẤP NHẤT)   : {best_mae_row['LR_Label']} (PINN = {best_mae_row['PINN_MAE(mm)']:.4f} mm vs NoPINN = {best_mae_row['NoPINN_MAE(mm)']:.4f} mm)",
            "=" * 85,
        ])
    else:
        report_lines.append(df_all_details.to_string(index=False))

    report_lines.append("\nCHỈ MỤC CÁC THƯ MỤC KẾT QUẢ RIÊNG BIỆT:")
    for lr in lr_list:
        folder_n = format_lr_folder_name(lr)
        report_lines.append(f"  * {folder_n}: {os.path.join(args.output_dir, folder_n)}")
    report_lines.append(f"  * Tổng hợp đối chiếu: {summary_dir}")
    report_lines.append("=" * 85)

    report_text = "\n".join(report_lines)
    report_file_path = os.path.join(summary_dir, "comparison_pinn_vs_nopinn_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n[HOÀN TẤT] Bảng đối chiếu PINN vs NoPINN: {csv_comparison_path}")
    print(f"[HOÀN TẤT] Biểu đồ nhóm IEEE: {fig_png_path}")


if __name__ == "__main__":
    main()
