# Domain Adaptation & Test-Time Adaptation for ECT-NDE

Thư mục này chứa toàn bộ các mô-đun thực thi các hướng **Chuyển giao miền (Domain Adaptation)**, **Fine-Tuning** và **Test-Time Adaptation (TTA)** để khắc phục hiện tượng **Sim-to-Real Gap** khi áp dụng mô hình đã huấn luyện từ dữ liệu mô phỏng sang dữ liệu đo thực nghiệm (tập 5kHz gồm 20 mẫu từ 10 khuyết tật No1 – No10).

---

## 1. Cấu Trúc Thư Mục

```
domain_adaptation/
├── models.py                     # Định nghĩa kiến trúc ImprovedMultimodelNet khớp 100% với checkpoint
├── utils.py                      # Bộ công cụ: load checkpoint, load scalers, load data thực 5kHz, chia 10-Fold LODO
├── 1_zero_shot.py               # Hướng 1: Đánh giá mô hình Pretrained gốc không thích ứng (Baseline)
├── 2_few_shot_peft.py           # Hướng 2: Few-Shot PEFT (Head Fine-tuning trên 18 mẫu / Fold)
├── 3_domain_transfer_mmd.py     # Hướng 3: Supervised Domain Transfer với MMD Feature Alignment
├── 4_physics_tta.py             # Hướng 4: Physics-Informed Test-Time Adaptation (Self-Supervised BN Adaptation)
├── run_all.py                   # Script tổng điều phối chạy lần lượt cả 4 hướng và xuất bảng so sánh
├── run_all.bat                  # File chạy 1-click trên Windows
├── run_all.sh                   # File chạy trên Linux HPC
└── results/                     # Thư mục lưu kết quả chi tiết và bảng tổng hợp
    ├── lodo_oof_master_summary.csv
    └── lodo_oof_master_predictions.csv
```

---

## 2. Giao Thức Đánh Giá: 10-Fold LODO & Pooled OOF Evaluation

* **10-Fold Leave-One-Defect-Out (LODO):**
  * Chia 10 Folds từ 20 mẫu (10 khuyết tật No1 đến No10):
  * Mỗi Fold bỏ riêng **2 file** của cùng một khuyết tật (ví dụ: `5khz_No1.csv` & `5khz_No1_1.csv`) làm tập Test độc lập.
  * **18 file** còn lại được dùng cho quá trình Fine-tuning / Thích ứng.
* **Pooled Out-of-Fold (OOF) Evaluation:**
  * Toàn bộ 20 dự đoán ngoài mẫu (OOF) được gom về 1 bảng duy nhất.
  * Tính toán **Ma trận nhầm lẫn toàn cục 5×5**, Overall Accuracy $\frac{\sum \text{TP}}{20}$, Macro-F1, và sai số $MAE (W, L, D)$ trực tiếp trên 20 mẫu thực tế, loại bỏ hoàn toàn hiện tượng méo mó số liệu của các fold nhỏ.

---

## 3. Cách Sử Dụng & Thực Thi

### Chạy toàn bộ 4 hướng cùng lúc (Khuyên dùng):
* **Trên Windows:**
  ```cmd
  domain_adaptation\run_all.bat
  ```
  *Hoặc qua môi trường conda konabi:*
  ```cmd
  conda run -n konabi python domain_adaptation/run_all.py
  ```
* **Trên Linux:**
  ```bash
  bash domain_adaptation/run_all.sh
  ```

### Chạy từng hướng độc lập:
```bash
# 1. Zero-shot Baseline
conda run -n konabi python domain_adaptation/1_zero_shot.py

# 2. Few-Shot PEFT Head Fine-Tuning
conda run -n konabi python domain_adaptation/2_few_shot_peft.py --epochs 50 --lr 0.0002

# 3. Supervised Domain Transfer MMD
conda run -n konabi python domain_adaptation/3_domain_transfer_mmd.py --epochs 50 --lr 0.0002

# 4. Physics-Informed TTA
conda run -n konabi python domain_adaptation/4_physics_tta.py --steps 25
```

### Chỉ định đường dẫn Checkpoint khác (nếu có model mới):
```bash
conda run -n konabi python domain_adaptation/run_all.py --model-dir <duong_dan_thu_muc_chua_best_model_pytorch.pth>
```
*(Mặc định nếu không truyền tham số, script sẽ tự động tìm và nạp checkpoint tốt nhất trong `cnn/Outputs_cnn_pinn/`)*.
