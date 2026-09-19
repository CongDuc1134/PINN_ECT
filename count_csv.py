import os
import argparse
from collections import defaultdict

def count_csv_files(root_dir):
    total_csv = 0
    folder_counts = defaultdict(int)
    
    print(f"{'='*60}")
    print(f"Bắt đầu quét file CSV trong: {os.path.abspath(root_dir)}")
    print(f"{'='*60}")
    
    # Quét đệ quy toàn bộ thư mục
    for dirpath, _, filenames in os.walk(root_dir):
        # Đếm số file có đuôi .csv trong thư mục hiện tại
        csv_files = [f for f in filenames if f.endswith('.csv')]
        count = len(csv_files)
        
        if count > 0:
            rel_path = os.path.relpath(dirpath, root_dir)
            if rel_path == '.':
                rel_path = '(Thư mục gốc)'
            
            folder_counts[rel_path] = count
            total_csv += count
    
    # In kết quả chi tiết
    if total_csv == 0:
        print("Không tìm thấy file .csv nào!")
    else:
        # Sắp xếp theo số lượng file giảm dần để dễ nhìn
        sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\n[Chi tiết theo thư mục]")
        for folder, count in sorted_folders:
            print(f" ├── {count:>4} files : {folder}")
            
        print(f"\n{'-'*60}")
        print(f"👉 TỔNG CỘNG: {total_csv} file CSV")
        print(f"{'-'*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đếm số lượng file CSV trong thư mục")
    parser.add_argument("--path", type=str, default=".", help="Đường dẫn tới thư mục cần quét (mặc định: thư mục hiện tại)")
    
    args = parser.parse_args()
    count_csv_files(args.path)
