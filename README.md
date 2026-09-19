# PINN-ECT: Physics-Informed Neural Networks & Domain Adaptation for Eddy Current Testing NDE

Hệ thống mã nguồn nghiên cứu bài báo khoa học chuẩn quốc tế Q1 (IEEE Transactions / Elsevier / NDT&E) về ứng dụng **Mạng Nơ-ron Thông tin Vật lý (PINN)** và **Chuyển giao miền (Domain Adaptation)** trong kiểm tra không phá hủy bằng dòng điện xoáy (ECT).

---

## 📌 ĐIỂM NHẤN CÔNG TRÌNH VÀ HIỆN TRẠNG HỆ THỐNG

- **Dữ liệu thực nghiệm**: 20 mẫu đo thực tế 5kHz từ 10 mẫu khuyết tật kim loại chuẩn (No1 – No10: Chữ nhật, Elip, Tam giác, Bậc R, Bậc T).
- **Quy mô Benchmark**: Đã đánh giá **43 Checkpoint hoàn chỉnh (215 thực nghiệm 10-Fold LODO độc lập)**:
  - 19 mô hình CNN (CNN Baselines, CNN PINN Base, CNN PINN Alpha).
  - 12 mô hình Multitask MLP PINN.
  - 12 mô hình Xiong et al. (2023) PINN.
- **Tài liệu đặc tả chi tiết & Hướng dẫn tái lập**: Vui lòng xem toàn văn tại **[domain_adaptation/README.md](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/README.md)**.

---

## 🚀 HƯỚNG DẪN CHẠY 1-CLICK TRÊN WINDOWS

Hệ thống hỗ trợ file batch menu tương tác thông minh tại thư mục gốc:

```cmd
run_domain_adaptation.bat
```

*Hoặc truyền tham số dòng lệnh trực tiếp qua môi trường Conda `konabi`:*
```cmd
# Chạy toàn bộ 43 models qua 5 phương pháp
run_domain_adaptation.bat all

# Chỉ chạy nhóm 19 mô hình CNN
run_domain_adaptation.bat cnn

# Kiểm tra trạng thái tiến độ
run_domain_adaptation.bat status
```

---

## 📊 TÓM TẮT KẾT QUẢ THEN CHỐT

1. **CNN áp đảo hoàn toàn MLP khi thích ứng thực tế**:
   - Chuyển giao miền bằng MMD trên CNN đạt **Acc 50.53%**, sai số định lượng kích thước **MAE chỉ 0.67 mm**.
   - Trong khi đó, cả hai kiến trúc MLP (Multitask và Xiong et al.) bị sụp đổ không gian biểu diễn (Acc bão hòa ở mức 12% - 18%, sai số MAE gấp nhiều lần).
2. **Khám phá cơ chế & Đột phá PI-MMD**:
   - Khi áp dụng Fine-tuning đóng băng xương sống (Standard PEFT), Baseline tạm thời dẫn trước PINN do xương sống PINN bị khóa cứng không triệt tiêu được trôi cảm biến.
   - Khi áp dụng **Physics-Informed MMD (PI-MMD)** kết hợp **Discriminative Learning Rate** và **Ràng buộc thể tích vật lý**, **PINN lập kỷ lục độ chính xác mới toàn hệ thống với MAE chỉ 0.463 mm** (vượt trội hoàn toàn so với 0.511 mm của Baseline).

---

## 📂 ĐƯỜNG DẪN KẾT QUẢ ĐÃ TỔNG HỢP

- **Master Summary toàn bộ 43 models**: [domain_adaptation/results/master_all_models_summary.csv](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/results/master_all_models_summary.csv)
- **Master Summary CNN (19 models)**: [domain_adaptation/results/cnn/cnn_master_summary.csv](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/results/cnn/cnn_master_summary.csv)
- **Master Summary MLP (24 models)**: [domain_adaptation/results/mlp/mlp_master_summary.csv](file:///c:/Users/Admin/Documents/paper/PINN_ECT/domain_adaptation/results/mlp/mlp_master_summary.csv)
