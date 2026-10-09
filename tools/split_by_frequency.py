# -*- coding: utf-8 -*-
"""
===============================================================================
SPLIT REAL EXPERIMENT DATA BY FREQUENCY (5kHz, 10kHz, 20kHz)
===============================================================================
Script tự động đọc toàn bộ file mẫu từ một thư mục (mặc định là Experiment_1)
và phân loại/chia tách thành 3 thư mục con độc lập theo tần số:
- 5khz/   : chứa toàn bộ mẫu 5kHz kèm labels.csv tương ứng
- 10khz/  : chứa toàn bộ mẫu 10kHz kèm labels.csv tương ứng
- 20khz/  : chứa toàn bộ mẫu 20kHz kèm labels.csv tương ứng

Tính năng:
1. Quét đệ quy tìm kiếm toàn bộ các file .csv mẫu và labels.csv.
2. Tự động nhận diện tần số từ tên file (5khz, 10khz, 20khz) hoặc nội dung labels.
3. Sao chép các file sample vào thư mục tương ứng một cách an toàn.
4. Tách và tạo file labels.csv độc lập cho từng tần số.
5. In báo cáo tổng hợp chi tiết số lượng mẫu và dạng vết nứt.
"""

import os
import sys
import shutil
import argparse
import pandas as pd

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def detect_frequency(filename: str) -> str:
    """Xác định tần số ('5khz', '10khz', '20khz') từ tên file."""
    f_lower = filename.lower()
    if "5khz" in f_lower:
        return "5khz"
    elif "10khz" in f_lower:
        return "10khz"
    elif "20khz" in f_lower:
        return "20khz"
    return "unknown"


def split_data_by_frequency(input_dir: str, output_dir: str = None, copy_mode: str = "copy"):
    """
    Đọc input_dir và chia thành các folder 5khz, 10khz, 20khz.

    Tham số:
        input_dir: Thư mục nguồn chứa dữ liệu thực nghiệm (VD: Experiment_1).
        output_dir: Thư mục đích. Nếu None, tạo trực tiếp bên trong input_dir.
        copy_mode: 'copy' để sao chép file an toàn.
    """
    input_dir = os.path.abspath(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = os.path.abspath(output_dir)

    print("=" * 80)
    print(f"BẮT ĐẦU CHIA TÁCH DỮ LIỆU THEO TẦN SỐ (5kHz, 10kHz, 20kHz)")
    print(f"Thư mục nguồn: {input_dir}")
    print(f"Thư mục đích : {output_dir}")
    print("=" * 80)

    # 1. Thu thập tất cả các file labels.csv có sẵn để ghép lại thành từ điển nhãn
    combined_labels = {}
    for root, _, files in os.walk(input_dir):
        if "labels.csv" in files:
            lbl_path = os.path.join(root, "labels.csv")
            try:
                df = pd.read_csv(lbl_path)
                for _, row in df.iterrows():
                    fname = str(row["filename"]).strip()
                    combined_labels[fname] = row.to_dict()
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể đọc {lbl_path}: {e}")

    # 2. Quét toàn bộ file CSV mẫu
    freq_buckets = {
        "5khz": [],
        "10khz": [],
        "20khz": [],
    }

    # Tránh quét lặp vào chính các thư mục đích nếu chúng nằm trong input_dir
    skip_dirs = {os.path.join(output_dir, f) for f in ["5khz", "10khz", "20khz"]}

    for root, _, files in os.walk(input_dir):
        if os.path.abspath(root) in skip_dirs:
            continue
        for fname in files:
            if not fname.endswith(".csv") or fname == "labels.csv":
                continue
            
            freq = detect_frequency(fname)
            if freq in freq_buckets:
                src_path = os.path.join(root, fname)
                freq_buckets[freq].append((fname, src_path))
            else:
                print(f"[BỎ QUA] Không xác định được tần số của file: {fname}")

    # 3. Tạo các thư mục đích và sao chép file
    summary_stats = []

    for freq, file_list in freq_buckets.items():
        target_freq_dir = os.path.join(output_dir, freq)
        os.makedirs(target_freq_dir, exist_ok=True)

        copied_count = 0
        freq_label_rows = []

        for fname, src_path in file_list:
            dst_path = os.path.join(target_freq_dir, fname)
            
            # Tránh tự copy chính nó nếu file đã ở sẵn trong thư mục đích
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)
            copied_count += 1

            # Lấy thông tin nhãn tương ứng
            if fname in combined_labels:
                lbl_row = dict(combined_labels[fname])
                lbl_row["frequency"] = f"{freq[:freq.find('khz')] if 'khz' in freq else freq}kHz"
                lbl_row["split"] = f"{freq[:freq.find('khz')] if 'khz' in freq else freq}kHz"
                freq_label_rows.append(lbl_row)

        # Ghi file labels.csv cho riêng thư mục tần số này
        if freq_label_rows:
            labels_df = pd.DataFrame(freq_label_rows)
            # Sắp xếp theo tên file hoặc crack_no nếu có
            if "crack_no" in labels_df.columns:
                labels_df = labels_df.sort_values(by=["crack_no", "filename"])
            else:
                labels_df = labels_df.sort_values(by=["filename"])
            
            out_label_csv = os.path.join(target_freq_dir, "labels.csv")
            labels_df.to_csv(out_label_csv, index=False)
            labels_info = f"Đã tạo {out_label_csv} ({len(labels_df)} dòng nhãn)"
        else:
            labels_info = "Không có labels.csv"

        summary_stats.append({
            "Frequency": freq,
            "Target_Directory": target_freq_dir,
            "Sample_Count": copied_count,
            "Labels_Status": labels_info,
        })

    # 4. In báo cáo tổng kết
    print("\n" + "=" * 80)
    print("KẾT QUẢ CHIA TÁCH THƯ MỤC THEO TẦN SỐ:")
    print("=" * 80)
    for stat in summary_stats:
        print(f"\n📁 Thư mục: [{stat['Frequency']}]")
        print(f"   Đường dẫn: {stat['Target_Directory']}")
        print(f"   Số lượng sample : {stat['Sample_Count']} files .csv")
        print(f"   File labels     : {stat['Labels_Status']}")

    print("\n" + "=" * 80)
    print("HOÀN TẤT CHIA TÁCH DỮ LIỆU!")
    print("=" * 80)
    return summary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chia dữ liệu thực nghiệm thành các folder 5khz, 10khz, 20khz")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Experiment_1"),
        help="Đường dẫn tới thư mục nguồn (mặc định: Experiment_1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục đích (mặc định: tạo 5khz, 10khz, 20khz bên trong input_dir)",
    )

    args = parser.parse_args()
    split_data_by_frequency(args.input_dir, args.output_dir)
