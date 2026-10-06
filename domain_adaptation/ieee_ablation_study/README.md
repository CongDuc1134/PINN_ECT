# GÓI BÁO CÁO THỰC NGHIỆM ĐỐI CHỨNG VÀ ĐỒ THỊ CHUẨN IEEE TRANSACTIONS
## SIM-TO-REAL ABLATION STUDY: NOPINN BASELINE VS. PROPOSED PINN

Thư mục này được tạo lập độc lập nhằm lưu trữ toàn bộ dữ liệu, bảng số liệu LaTeX/CSV và các đồ thị chất lượng cao (300 DPI PNG & vector PDF) chuẩn tạp chí khoa học quốc tế **IEEE Transactions on Instrumentation and Measurement / Industrial Electronics**.

---

### 1. Cấu trúc thư mục

```
domain_adaptation/ieee_ablation_study/
├── generate_ieee_ablation_artifacts.py   # Script tạo tự động toàn bộ bảng và biểu đồ
├── README.md                             # Tài liệu giải thích khoa học và chỉ mục
├── tables/
│   ├── ablation_study_summary.csv        # Bảng số liệu chi tiết 17 cấu hình thử nghiệm
│   └── table_ablation_ieee.tex           # Bảng mã nguồn LaTeX chuẩn IEEE (\toprule, \midrule, \bottomrule)
└── figures/
    ├── fig1_ablation_acc_nmae_head_to_head.png (.pdf)   # So sánh Head-to-Head ACC & NMAE
    ├── fig2_dimensional_mae_breakdown_ablation.png (.pdf) # Phân rã sai số kích thước 3 chiều W, L, D (mm)
    ├── fig3_effect_of_epochs_and_lr_freeze_bn.png (.pdf)  # Động học hội tụ theo Epochs và Learning Rate
    └── fig4_unified_ablation_dashboard.png (.pdf)       # Dashboard tổng hợp 4-Panel chuẩn bài báo IEEE
```

---

### 2. Danh mục Bảng biểu và Đồ thị chuẩn IEEE

#### A. Bảng số liệu (Tables)
* [ablation_study_summary.csv](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/tables/ablation_study_summary.csv): Chứa 17 cấu hình đo đạc trên 10 mẫu kiểm thử thực nghiệm mù (Scan 2), bao gồm các thông số:
  * Phân loại: Độ chính xác phân loại hình dạng ($\text{ACC}$), số mẫu phân loại đúng ($9/10$, $8/10$, $4/10$).
  * Hồi quy 3D: Sai số tổng hợp $\text{MAE}$, sai số chiều rộng $\text{MAE}_W$, chiều dài $\text{MAE}_L$, chiều sâu $\text{MAE}_D$ (đơn vị mm) và sai số chuẩn hóa vĩ mô $\text{NMAE}$ (%).
* [table_ablation_ieee.tex](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/tables/table_ablation_ieee.tex): Bảng mã nguồn LaTeX sẵn sàng nhúng trực tiếp vào bản thảo bài báo Overleaf.

#### B. Hệ thống Đồ thị chuẩn IEEE (Figures)
* [fig1_ablation_acc_nmae_head_to_head.png](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/figures/fig1_ablation_acc_nmae_head_to_head.png):
  So sánh trực diện hai chỉ số cốt lõi $\text{ACC}$ và $\text{NMAE}$ qua 6 mốc tiến hóa mô hình.
* [fig2_dimensional_mae_breakdown_ablation.png](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/figures/fig2_dimensional_mae_breakdown_ablation.png):
  Phân rã sai số 3 chiều không gian. Thể hiện rõ mô hình Proposed sau khi hiệu chỉnh đã giảm mạnh sai số chiều dài $L$ từ $1.761\text{ mm} \to 0.142\text{ mm}$ và chiều sâu $D$ từ $0.746\text{ mm} \to 0.392\text{ mm}$.
* [fig3_effect_of_epochs_and_lr_freeze_bn.png](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/figures/fig3_effect_of_epochs_and_lr_freeze_bn.png):
  Khảo sát động học hội tụ khi đóng băng BatchNorm (`Freeze BN`) theo số epoch từ 300 đến 1500 cho hai mức learning rate $\eta = 1\times 10^{-4}$ và $\eta = 2\times 10^{-4}$.
* [fig4_unified_ablation_dashboard.png](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/figures/fig4_unified_ablation_dashboard.png):
  Bảng điều khiển tổng hợp 4-Panel tiêu chuẩn IEEE Transactions:
  * (a) Tỉ lệ phân loại hình thái vết nứt.
  * (b) Sai số kích thước trung bình 3D.
  * (c) Quỹ đạo hội tụ sai số theo số epochs.
  * (d) Cơ chế cân bằng tham số bất định Kendall.
* [fig5_two_protocols_finetuned_comparison.png](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/ieee_ablation_study/figures/fig5_two_protocols_finetuned_comparison.png):
  Đồ thị so sánh trực diện giữa **Giao thức 1 (Scan Split: Scan 1 $\to$ Scan 2)** và **Giao thức 2 (10-Fold LODO: Leave-One-Defect-Out)** cho cả hai chỉ số phân loại (ACC) và sai số kích thước 3 chiều $W, L, D$.

---

### 3. Tóm tắt kết quả khoa học trên cả 2 Giao thức (Protocols)

#### A. Đối chiếu kết quả chi tiết giữa Giao thức 1 và Giao thức 2:

| Giao thức Đánh giá | Mô hình | ACC (%) | Số mẫu đúng | MAE Tổng (mm) | MAE $W$ (mm) | MAE $L$ (mm) | MAE $D$ (mm) | NMAE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Giao thức 1 (Scan 1 $\to$ Scan 2)** | CNN\_NoPINN | 100.0% | 10/10 | 0.6892 | 0.0596 | 1.3849 | 0.6231 | 16.19% |
| *(Kiểm tra độ bền lặp và trôi sensor)* | **CNN\_Proposed (PINN)** | **90.0%** | **9/10** | **0.1602** | **0.0512** | **0.1315** | **0.2980** | **7.07%** |
| **Giao thức 2 (10-Fold LODO)** | CNN\_NoPINN | 65.0% | 13/20 | 0.6561 | 0.0659 | 0.9807 | 0.9216 | 19.30% |
| *(Kiểm tra tổng quát hóa mẫu nứt mới)* | **CNN\_Proposed (PINN)** | **55.0%** | **11/20** | **0.3883** | **0.0860** | **0.2660** | **0.8130** | **16.38%** |

#### B. Các phát hiện cốt lõi:
1. **Ở Giao thức 1 (Scan Split)**:
   - Mô hình Proposed PINN đạt hiệu năng kích thước **vượt trội tuyệt đối** (MAE $0.1602\text{ mm}$ so với $0.6892\text{ mm}$ của NoPINN, giảm **76.8% sai số**; NMAE giảm từ $16.19\% \to 7.07\%$).
   - Về phân loại, Proposed nhận diện chính xác vết nứt bậc phức tạp `Step_T` với độ tin cậy $79.4\%$.
2. **Ở Giao thức 2 (10-Fold Leave-One-Defect-Out)**:
   - Trong 10 khuyết tật phôi thực tế, hai dạng khuyết tật bậc `Step_R` (mẫu số 9) và `Step_T` (mẫu số 10) **chỉ có duy nhất 1 phôi trong toàn bộ tập dữ liệu**. Khi rút phôi đó ra làm kiểm thử mù, tập huấn luyện có **0 mẫu** thuộc lớp này (Zero-Shot Generalization).
   - Mô hình Proposed PINN duy trì độ chính xác kích thước 3D **vượt trội NoPINN** (MAE $0.3883\text{ mm}$ vs. $0.6561\text{ mm}$, sai số chiều dài $L$ giảm mạnh từ $0.981\text{ mm} \to 0.266\text{ mm}$).
