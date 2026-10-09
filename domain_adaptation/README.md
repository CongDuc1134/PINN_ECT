# GÓI THÍCH ỨNG MIỀN (SIM-TO-REAL DOMAIN ADAPTATION) - PINN ECT

Mô-đun thực hiện nghiên cứu thích ứng miền từ dữ liệu mô phỏng phần tử hữu hạn (Simulation / Finite Element Model) sang dữ liệu đo thực nghiệm cảm biến dòng xoáy ECT (Real Eddy Current Testing Data) cho bài toán nhận dạng hình học và định lượng kích thước 3D khuyết tật nứt kim loại.

---

## 1. Cấu trúc thư mục tinh gọn (Clean Architecture)

```
domain_adaptation/
│
├── __init__.py                           # Export API chính của gói domain_adaptation
├── README.md                             # Tài liệu tổng quan, hướng dẫn sử dụng & chỉ mục khoa học
├── benchmark_evaluation_protocols.py     # Engine thực thi Protocol 1 & Protocol 2 (Finetuning & Đánh giá)
├── data_loader.py                        # Dataset PyTorch, tiền xử lý, nạp dữ liệu 5k/10k/20k & chia fold
├── model_loader.py                       # Nạp checkpoints tiền huấn luyện, nạp scalers & suy luận
├── load_real_experiment_data.py          # Ground truth chuẩn Table 2, phát hiện thư mục ảnh & scalers
├── plot_ieee_utils.py                    # Tiện ích trực quan hóa kết quả theo chuẩn xuất bản IEEE
│
├── cnn/                                  # Mô hình 2D Convolutional Multitask
│   ├── models.py                         # Kiến trúc ImprovedMultimodelNet (Backbone + 2 Heads)
│   ├── losses.py                         # Hàm loss đa nhiệm tích hợp PINN & độ bất định Kendall
│   └── config.py                         # Cấu hình siêu tham số mặc định cho CNN
│
├── mlp/                                  # Mô hình Multitask MLP 1D Baseline
│   ├── models.py                         # Kiến trúc MultitaskMLP_PINN
│   ├── losses.py                         # Hàm loss đa nhiệm MLP
│   └── config.py                         # Cấu hình siêu tham số cho MLP
│
├── xiong/                                # Mô hình 1D Regression Baseline (Xiong et al. 2023)
│   ├── models.py                         # Kiến trúc Single-task RegressionMLP_PINN
│   ├── losses.py                         # Hàm loss hồi quy Xiong
│   └── config.py                         # Cấu hình siêu tham số cho Xiong
│
├── finetune_results/                     # Kết quả huấn luyện và kiểm thử thích ứng miền
│   ├── runs_history.csv                  # Nhật ký tổng hợp theo dõi toàn bộ các lần chạy/finetune theo thời gian
│   ├── experimental_ablation_study.csv   # Dữ liệu phân tích thực nghiệm tổng hợp
│   ├── latest/                           # Tự động đồng bộ bản sao kết quả đợt chạy mới nhất
│   ├── runs/                             # Thư mục lưu trữ riêng biệt cho từng lần chạy / finetuning
│   │   └── run_<timestamp>_<config>/     # Thư mục riêng cho mỗi lần finetune
│   │       ├── run_config.json           # Cấu hình siêu tham số và môi trường của đợt chạy
│   │       ├── run_summary.txt          # Báo cáo tóm tắt kết quả dễ đọc trực quan
│   │       ├── checkpoints/              # Checkpoint mô hình sau finetuning
│   │       ├── figures/                  # Đồ thị chuẩn IEEE (PNG 300 DPI & vector PDF)
│   │       └── tables/                   # Bảng số liệu chi tiết (CSV & LaTeX)
│   └── protocols/                        # Thư mục tương thích ngược cho kết quả mặc định
```

---

## 2. Chiến lược Đóng băng Trọng số (Option 1: Freeze Backbone)

Khi thích ứng mô hình sang tập thực nghiệm quy mô nhỏ (20 mẫu thực nghiệm), chiến lược tối ưu để tránh hiện tượng quên thảm họa (Catastrophic Forgetting) và quá khớp (Overfitting) là **Freeze Backbone** (`--freeze_backbone`):

### Bảng phân bổ tham số (Parameter Budget) - Mô hình Proposed 2D CNN:
| Thành phần Mạng | Trạng thái Finetuning | Số lượng Tham số | Tỉ lệ (%) | Chi tiết Kiến trúc & Vai trò |
| :--- | :---: | :---: | :---: | :--- |
| **Convolutional Backbone** | **FROZEN** (`requires_grad=False`) | **93,408** | **84.38%** | 3 khối Conv2D (32 $\to$ 64 $\to$ 128) + BatchNorm2d (eval mode) + SiLU + Pooling. Bảo toàn tri thức trích xuất đặc trưng vật lý từ miền mô phỏng. |
| **Classification Head** | **TRAINABLE** (`requires_grad=True`) | **8,709** | **7.87%** | Linear(128, 64) $\to$ BN1d $\to$ SiLU $\to$ Dropout(0.1) $\to$ Linear(64, 5). Hiệu chỉnh phân loại 5 hình thái vết nứt trên miền thực nghiệm. |
| **Regression Head** | **TRAINABLE** (`requires_grad=True`) | **8,579** | **7.75%** | Linear(128, 64) $\to$ BN1d $\to$ SiLU $\to$ Dropout(0.1) $\to$ Linear(64, 3) $\to$ **Sigmoid()**. Giới hạn đầu ra $[W, L, D] \in (0, 1)$ khớp chuẩn với Scaler. |
| **Kendall Uncertainty** | **TRAINABLE** (`requires_grad=True`) | **4** | **< 0.01%** | 4 tham số vô hướng ($s_{\text{clf}}, s_w, s_l, s_d$), tự động học trọng số loss đa nhiệm hoàn toàn tự nhiên. |
| **Tổng cộng** | — | **110,700** | **100%** | **Đóng băng 93,408 tham số (84.38%), chỉ cập nhật 17,292 tham số (15.62%).** |

---

## 3. Hàm Loss và Cơ chế Tự động Học Trọng số Kendall (Loss Audit)

### A. Cơ chế Phân phối Hàm Loss Bản thể ("Mô hình nào gọi đúng Loss tương ứng"):
Thay vì ép buộc tất cả các mô hình dùng chung 1 hàm loss, hệ thống tự động phân phối hàm loss chuẩn theo đúng bản thể kiến trúc của từng mô hình:
1. **Mô hình PINN (`CNN_Proposed`, `MLP`, `Xiong`)**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \alpha \mathcal{L}_{\text{physics}}$$
   - **Physics Loss ($\mathcal{L}_{\text{physics}}$)**: Tính toán thông qua hàm đạo hàm tự động `compute_physics_loss_autograd`. Với mỗi hình dạng khuyết tật (Rectangular, Ellipse, Triangular, Step), bộ giải trường từ tiến (forward solver) tính toán trường phân bố từ $H_{\text{pred}}(\hat{W}, \hat{L}, \hat{D})$, so sánh độ lệch với trường đo đạc thực tế $H_{\text{true}}$. Đạo hàm vi phân PDE $\frac{\partial \mathcal{L}_{\text{physics}}}{\partial [\hat{W}, \hat{L}, \hat{D}]}$ lan truyền trực tiếp về Regression Head, ép buộc các kích thước dự đoán phải tuân thủ nghiêm ngặt định luật điện từ cảm ứng.
   - Trọng số $\alpha$: Được tự động nạp từ checkpoint pre-trained gốc (mặc định $\alpha = 1.0$).
2. **Mô hình NoPINN (`CNN_NoPINN`)**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}$$
   - Thuần túy hàm supervised data loss đa nhiệm (CrossEntropy + Kendall MSE), hoàn toàn không tính thành phần Physics Loss ($\alpha = 0.0$).
3. **Công thức Data Loss đa nhiệm $\mathcal{L}_{\text{data}}$**:
   $$\mathcal{L}_{\text{data}} = \exp(-s_{\text{clf}})\mathcal{L}_{\text{CE}}(\hat{c}, c) + \frac{1}{2}s_{\text{clf}} + \sum_{k \in \{W, L, D\}} \left(\exp(-s_k)\mathcal{L}_{\text{MSE}, k}(\hat{y}_k, y_k) + \frac{1}{2}s_k\right)$$

### B. Cơ chế Tham số Kendall (Mặc định: RESET Kendall = True):
- **Khởi tạo lại Tham số Kendall (`reset_kendall = True`)**: Khi bắt đầu finetuning trên tập thực nghiệm ECT, toàn bộ tham số độ bất định Kendall ($s_{\text{clf}}, s_w, s_l, s_d, s_{\text{reg}}$) được reset về `0.0`.
- **Hệ quả Toán học**:
  - Hệ số nhân trọng số: $\exp(-s) = \exp(0) = 1.0$.
  - Số hạng điều hòa: $\frac{1}{2}s = 0.0$.
  - Cân bằng hoàn toàn giữa nhiệm vụ phân loại hình thái ($\mathcal{L}_{\text{CE}}$) và nhiệm vụ hồi quy 3 chiều $[W, L, D]$ ($\mathcal{L}_{\text{MSE}}$).
  - Đưa hàm Train Loss về thang đo tự nhiên chuẩn hóa ($\approx 25 \sim 28$), loại bỏ hoàn toàn hiện tượng Loss bị phóng đại lên $184,000+$ do hệ số checkpoint mô phỏng cũ ($s_{\text{clf}} \approx -8.88 \implies \exp(-s) \approx 7,200$).
- **Huấn luyện Tự nhiên (Không ép buộc / Không kẹp biên)**: Trong quá trình finetuning, các tham số Kendall $s$ được cập nhật tự do theo gradient của miền thực nghiệm ECT qua `loss.backward()` và `optimizer.step()`, hoàn toàn không áp đặt kẹp biên nhân tạo.
- **Tùy chọn Bảo tồn Checkpoint Cũ**: Nếu muốn giữ nguyên tham số bất định từ pre-trained checkpoint, người dùng có thể truyền cờ `--no_reset_kendall`.

### C. Tốc độ học Cố định (Fixed Learning Rate = $10^{-4}$):
- Mặc định `--fixed_lr` (`fixed_lr=True`) với `--lr 1e-4`: Khi tham số Kendall đã được RESET về 0.0 ($\exp(-s) = 1.0$), tốc độ học $10^{-4}$ giúp các tầng Classification Head và Regression Head hội tụ nhanh và tối ưu trong chu kỳ 200 epochs.
- Người dùng có thể tùy biến mức LR khác hoặc bật lại scheduler (`--use_scheduler`) nếu muốn.

---

## 4. Hai Giao thức Đánh giá Khoa học (Evaluation Protocols)

1. **Protocol 1 (Scan Split: Scan 1 $\to$ Scan 2)**:
   - *Mục tiêu*: Đánh giá độ bền lặp (repeatability) và khả năng thích ứng trước nhiễu trôi của cảm biến thực tế.
   - *Cấu hình*: Huấn luyện trên 10 mẫu quét đợt 1 (Scan 1), kiểm thử mù trên 10 mẫu quét đợt 2 (Scan 2) cùng phôi.
2. **Protocol 2 (10-Fold Leave-One-Defect-Out - LODO)**:
   - *Mục tiêu*: Đánh giá khả năng tổng quát hóa trên khuyết tật hoàn toàn mới (Zero-shot / Novel Defect Generalization).
   - *Cấu hình*: Chia 10 fold theo từng phôi khuyết tật (mỗi fold giữ lại 2 mẫu Scan 1 & Scan 2 của 1 phôi làm test, 18 mẫu còn lại làm train).

---

## 5. Hướng dẫn Thực thi (Quickstart Commands)

> **Lưu ý**: Tất cả lệnh thực thi tự động sử dụng môi trường Conda `konabi` (`C:\ProgramData\miniconda3\envs\konabi\python.exe`).

### A. Khởi chạy nhanh bằng PowerShell Script:
```powershell
# Mở menu tương tác lựa chọn giao thức:
.\run_domain_adaptation.ps1

# Hoặc chạy trực tiếp theo từng chế độ:
.\run_domain_adaptation.ps1 all      # Chạy cả 2 Giao thức (Protocol 1 + Protocol 2)
.\run_domain_adaptation.ps1 p1       # Chạy Giao thức 1 (Scan 1 Train / Scan 2 Test)
.\run_domain_adaptation.ps1 p2       # Chạy Giao thức 2 (10-Fold Leave-One-Defect-Out)
.\run_domain_adaptation.ps1 cnn      # Chạy riêng cho 2 mô hình CNN (Proposed + NoPINN)
.\run_domain_adaptation.ps1 test     # Chạy Quick Smoke Test (1 epoch)
```

### B. Chạy Benchmark Protocol 1 (Scan Split):
```bash
# Chạy mặc định (Fixed LR = 1e-4, Freeze Backbone, Reset Kendall)
python domain_adaptation/benchmark_evaluation_protocols.py \
    --protocol 1 \
    --models cnn_proposed cnn_nopinn \
    --epochs 200 \
    --lr 1e-4 \
    --fixed_lr \
    --freeze_backbone

# Hoặc đặt tên folder riêng tùy chọn bằng --run_name hoặc --tag
python domain_adaptation/benchmark_evaluation_protocols.py \
    --protocol 1 \
    --models cnn_proposed \
    --epochs 200 \
    --lr 1e-4 \
    --fixed_lr \
    --freeze_backbone \
    --run_name exp_p1_freeze_bb_lr1e4
```

### B. Chạy Benchmark Protocol 2 (10-Fold LODO):
```bash
python domain_adaptation/benchmark_evaluation_protocols.py \
    --protocol 2 \
    --models cnn_proposed cnn_nopinn \
    --epochs 200 \
    --lr 1e-4 \
    --fixed_lr \
    --freeze_backbone \
    --tag lodo_lr1e4
```

---

## 6. Tiêu chuẩn Đồ thị Xuất bản IEEE (`figures_ieee`)
Toàn bộ đồ thị xuất ra từ quy trình đều tuân thủ nghiêm ngặt 6 tiêu chuẩn xuất bản quốc tế:
1. Font chữ đồng nhất: **Times New Roman** (8 - 10 pt cho nhãn và tick marks).
2. Không sử dụng tiêu đề nội bộ (`plt.title`) trong đồ thị; chú thích được chuyển hoàn toàn vào LaTeX Figure Caption.
3. Hộp chú giải (Legend) đặt ngoài khung dữ liệu hoặc trong góc thoáng, không che lấp đường đồ thị.
4. Bảng màu chuyên nghiệp: `colorblind`/`tableau-colorblind`, có ký hiệu marker phân biệt khi in đen trắng.
5. Kích thước chuẩn cột IEEE: Cột đơn 3.5 inch, hai cột 7.0 inch.
6. Xuất đồng thời 2 định dạng: **PNG (300 DPI)** cho xem nhanh và **vector PDF** cho biên tập Overleaf.
