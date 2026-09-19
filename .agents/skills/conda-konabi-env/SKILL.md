---
name: conda-konabi-env
description: Hướng dẫn và quy chuẩn bắt buộc kích hoạt môi trường Conda `konabi` (`conda activate konabi` hoặc `conda run -n konabi python ...`) mỗi khi chạy code Python, chạy script huấn luyện/đánh giá, cài đặt thư viện hoặc kiểm thử trong dự án.
---

# Conda Konabi Environment Activation Skill

Skill này quy định chuẩn thực thi và kích hoạt môi trường ảo Conda **`konabi`** cho toàn bộ các tác vụ chạy code, kiểm thử, huấn luyện và đánh giá mô hình trong dự án.

---

## 1. Nguyên Tắc Bắt Buộc

Mọi lệnh thực thi liên quan đến Python (chạy script, cài đặt package, chạy test, chạy benchmark) **BẮT BUỘC** phải được thực thi trong môi trường Conda **`konabi`**.

---

## 2. Cú Pháp Thực Thi Chuẩn Theo Nền Tảng

### A. Windows (PowerShell / Command Prompt)

#### Cách 1: Sử dụng `conda run` (Khuyên dùng cho lệnh đơn - Chạy trực tiếp không cần thay đổi shell state)
```powershell
conda run -n konabi python <duong_dan_file.py> [tham_so]
```
*Ví dụ:*
```powershell
conda run -n konabi python load_real_experiment_data.py
conda run -n konabi python count_csv.py
conda run -n konabi python cnn/main_percent_new.py
conda run -n konabi python cnn/run_eval_all.py
```

#### Cách 2: Kích hoạt môi trường trước khi chạy chuỗi lệnh (`conda activate`)
```powershell
conda activate konabi
python <duong_dan_file.py> [tham_so]
```
*Hoặc trên cùng một dòng lệnh trong PowerShell:*
```powershell
conda activate konabi; python <duong_dan_file.py>
```

---

### B. Linux / macOS (Bash / Zsh)

#### Cách 1: Sử dụng `conda run`
```bash
conda run -n konabi python <duong_dan_file.py>
```

#### Cách 2: Kích hoạt trong subshell hoặc script
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate konabi
python <duong_dan_file.py>
```

---

## 3. Quản Lý Gói & Thư Viện (Package Management)

Khi cài đặt hoặc kiểm tra các package (PyTorch, Scikit-learn, Pandas, v.v.):
```powershell
# Cài đặt qua pip trong môi trường konabi
conda run -n konabi pip install <package_name>

# Cài đặt qua conda
conda install -n konabi <package_name>

# Kiểm tra danh sách thư viện hiện có
conda run -n konabi pip list
# hoặc
conda list -n konabi
```

---

## 4. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

1. **Lỗi `CommandNotFoundError` hoặc `conda: command not found`**:
   - Xác định đường dẫn Python trực tiếp của env `konabi`:
     - Windows: `C:\Users\<Username>\anaconda3\envs\konabi\python.exe` hoặc `C:\Users\<Username>\miniconda3\envs\konabi\python.exe`
     - Chạy trực tiếp bằng binary:
       ```powershell
       & "$env:USERPROFILE\anaconda3\envs\konabi\python.exe" <file.py>
       ```

2. **Lỗi `Your shell has not been properly configured to use 'conda activate'`**:
   - Chạy lệnh khởi tạo cho shell:
     ```powershell
     conda init powershell
     ```
   - Hoặc chuyển sang dùng `conda run -n konabi python <file.py>` (không phụ thuộc vào shell hook).
