---
name: code-integrity-guardian
description: Chuyên gia kiểm toán toàn vẹn mã nguồn (Code Integrity & Anti-Hallucination Auditor). Tự động rà soát, phát hiện và ngăn chặn triệt để các lỗi thiếu hàm, tự bịa hàm/API không tồn tại, sai chữ ký hàm (signature mismatch), lỗi import, bẫy thực thi trễ (late-execution traps) và lỗi shape tensor trước khi chạy.
---

# Code Integrity Guardian & Anti-Hallucination Audit Skill

Skill này cung cấp quy trình và bộ tiêu chuẩn kiểm toán mã nguồn nghiêm ngặt, loại bỏ hoàn toàn tình trạng **"code chạy được một nửa / giữa chừng mới văng lỗi do thiếu hàm, bịa hàm, sai API hoặc lệch tham số"**.

---

## 1. Các "Bẫy Tử Thần" Thường Gặp (Root Causes of Mid-Run Failures)

1. **Hallucination / Bịa API không tồn tại**:
   - Gọi các hàm/thuộc tính "tưởng là có" của thư viện (ví dụ: `torch.nn.functional.smooth_l1` thay vì `smooth_l1_loss`, `model.evaluate_generator`, `cv2.COLOR_BGR2GRAYSCALE`).
2. **Missing Function / NameError**:
   - Gọi hàm phụ trợ (helper) nhưng quên định nghĩa `def ...` hoặc quên import từ module utils/library.
3. **Call-Site & Signature Mismatch**:
   - Định nghĩa `def train_epoch(model, loader, optimizer, device, loss_fn)` nhưng lúc gọi chỉ truyền 4 đối số hoặc sai thứ tự.
   - Sửa tên hàm/đối số ở file định nghĩa nhưng không cập nhật ở các file gọi.
4. **Bẫy Thực Thi Trễ (Deferred / Late-Execution Traps)**:
   - Các khối code chỉ chạy khi kết thúc vòng lặp hoặc theo chu kỳ: `if epoch % 10 == 0:`, `if is_best: save_checkpoint(...)`, `def evaluate(...)` chứa biến/hàm chưa từng được định nghĩa. Code chạy 50 epoch (~1-2 tiếng) mới nhảy vào khối này và sụp đổ.
5. **Tensor Shape & Device Traps**:
   - Tensor nằm trên CPU tính toán với Tensor trên GPU (`Expected all tensors to be on the same device`).
   - Mismatch shape ở bước `torch.cat`, `np.concatenate` hoặc `Dense(in_features)` sau phép `Flatten()`.

---

## 2. Quy Trình Kiểm Toán 5 Tầng Bắt Buộc (5-Tier Verification Checklist)

Trước khi xác nhận hoàn thành code hoặc chạy các script huấn luyện/xử lý nặng, PHẢI thực hiện kiểm toán theo 5 tầng sau:

### 🛡️ Tầng 1: Rà Soát Định Nghĩa Thực Thể (Symbol & Definition Audit)
- [ ] **Mọi hàm được gọi đều phải có nguồn gốc rõ ràng**:
  - Tự định nghĩa trong file (`def func_name(...)`)
  - Hoặc được import tường minh từ module nội bộ (`from library_functions import func_name`)
  - Hoặc là built-in / thư viện chuẩn.
- [ ] **Cấm tuyệt đối Import Wildcard (`from module import *`)**: Luôn import rõ tên hàm hoặc import namespace (`import numpy as np`, `import torch.nn as nn`) để tránh ghi đè tên biến và dễ dàng truy vết.
- [ ] **Xác thực API thư viện**: Không suy diễn API. Đối chiếu chính xác theo phiên bản thư viện hiện hành (PyTorch, TensorFlow/Keras, Scikit-learn, OpenCV, Pandas, Numpy).

### 🛡️ Tầng 2: Nhất Quán Chữ Ký Hàm (Signature & Parameter Match)
- [ ] Kiểm tra số lượng và tên đối số (`args`, `kwargs`) giữa định nghĩa (`def`) và nơi gọi (`call-site`).
- [ ] Kiểm tra giá trị trả về (`return`): Nếu hàm trả về 3 giá trị `return a, b, c`, nơi gọi phải unpack đúng 3 biến hoặc nhận 1 tuple.
- [ ] **Kiểm tra tác động chéo (Cross-File Refactoring Check)**: Khi sửa tên hàm hoặc tham số trong file thư viện (vd: `library_functions.py`), BẮT BUỘC dùng grep tìm và cập nhật TẤT CẢ các file phụ thuộc (`main.py`, `eval.py`, `notebooks`).

### 🛡️ Tầng 3: Khử Bẫy Thực Thi Trễ (Deferred & Branch Sweep)
- [ ] Rà soát toàn bộ các nhánh điều kiện ít kích hoạt:
  - Khối `if epoch % val_interval == 0:`
  - Khối `if is_test:` / `if eval_mode:`
  - Khối lưu mô hình `save_model()`, `torch.save()`, `export_results()`
  - Khối vẽ biểu đồ kết quả `plot_loss()`, `plot_confusion_matrix()`
  - Khối xử lý ngoại lệ `except Exception as e:`
- [ ] Đảm bảo các biến dùng trong các khối này đã được khởi tạo giá trị mặc định ở đầu hàm/vòng lặp, không bị phụ thuộc vào thứ tự rẽ nhánh.

### 🛡️ Tầng 4: Tính Toán Luồng Dữ Liệu & Thiết Bị (Device & Shape Tracing)
- [ ] **Device Sync**: Mọi Tensor đầu vào, nhãn và mô hình đều phải cùng nằm trên `device` (`.to(device)`).
- [ ] **Data Types**: Đồng nhất `torch.float32` cho đầu vào / trọng số, `torch.long` cho nhãn phân loại (CrossEntropyLoss), `torch.float32` cho hồi quy (MSE/L1).
- [ ] **Dimension Flow Trace**: Tính toán shape qua từng bước biến đổi:
  $$\text{Input: } (B, C, H, W) \xrightarrow{\text{Conv}} (B, C', H', W') \xrightarrow{\text{Flatten}} (B, D) \xrightarrow{\text{FC}} (B, \text{num\_classes})$$

### 🛡️ Tầng 5: Chạy Thử Nghiệm Kiểm Tra Cú Pháp & Smoke Test (1-Step Dry Run)
- [ ] **Kiểm tra AST / Syntax Parsing**: Chạy parser tĩnh để bắt `SyntaxError` và `IndentationError`.
- [ ] **Dry-run với Dummy Data (Smoke Test)**:
  - Chạy mô hình với 1 batch dữ liệu giả lập (kích thước nhỏ: batch size = 2, max_epochs = 1, val_epochs = 1).
  - Đảm bảo toàn bộ chu trình đi qua từ `Forward pass` $\to$ `Loss calculation` $\to$ `Backward pass` $\to$ `Validation step` $\to$ `Metric computation` $\to$ `Checkpoint saving` thành công mà không ném ngoại lệ.

---

## 3. Công Cụ Hỗ Trợ Tự Động (Built-in Helper Scripts)

Skill này đi kèm công cụ kiểm toán tự động bằng AST (Abstract Syntax Tree) trong thư mục `scripts/`:

### 🚀 Cách sử dụng:
1. **Kiểm tra 1 file script bất kỳ**:
   ```bash
   python .agents/skills/code-integrity-guardian/scripts/verify_code_integrity.py path/to/script.py
   ```
2. **Kiểm tra toàn bộ thư mục dự án**:
   ```bash
   python .agents/skills/code-integrity-guardian/scripts/verify_code_integrity.py . --recursive
   ```

Script sẽ tự động quét:
- Tất cả các biến / hàm gọi chưa được định nghĩa hoặc chưa import (`Undefined Variables & Functions`).
- Phát hiện các hàm import nhưng không tồn tại trong module nội bộ.
- Cảnh báo các khối điều kiện có nguy cơ gây lỗi trễ (late execution).

---

## 4. Nguyên Tắc Cốt Lõi Khi Viết / Chỉnh Sửa Code (Golden Rules)

1. **"Never assume, always verify"**: Không bao giờ đoán cú pháp hàm của thư viện. Luôn kiểm tra định nghĩa hàm gốc hoặc tài liệu chuẩn.
2. **"No partial refactoring"**: Đã sửa định nghĩa hàm thì phải sửa toàn bộ các điểm gọi trên toàn bộ codebase.
3. **"Test the tail end"**: Code lưu trữ, tính metric và xuất báo cáo phải được test ngay từ đầu, không để đến khi train xong mới phát hiện lỗi.
