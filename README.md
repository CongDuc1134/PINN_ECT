# Quy Trình Xử Lý Dữ Liệu & Kiến Trúc Mô Hình PINN-CNN (ECT)

> **Tài liệu kỹ thuật nội bộ:** Tổng hợp chi tiết dòng chảy dữ liệu (End-to-End Pipeline), chuẩn hóa (Normalization), kiến trúc mạng nơ-ron đa nhiệm (`ImprovedMultimodelNet`), dự đoán và giải chuẩn hóa (Denormalization) trong dự án Eddy Current Testing (ECT).

---

## 🗺️ Sơ Đồ Luồng Dữ Liệu Tổng Thể (End-to-End Flow)

```
[DỮ LIỆU THÔ BAN ĐẦU]
  • Ảnh trường từ cảm ứng: (32, 32, 2)
  • Nhãn kích thước thật: [W, L, D] (mm)
  • Nhãn hình dạng thật: "Ellipse", "Step_T", "Rectangular",...
               │
               ▼
[BƯỚC 1: CHUẨN HÓA DỮ LIỆU (NORMALIZATION)]
  • X_norm = (X - mean_train) / std_train       --> StandardScaler (Fit trên Train)
  • y_norm = [W / W_max, L / L_max, D / D_max]   --> SeparateMaxScaler (Chia Max Train)
  • Shape_encoded = {0, 1, 2, 3, 4}              --> Label Mapping (Từ điển)
               │
               ▼
[BƯỚC 2: MÔ HÌNH HỌC SÂU (ImprovedMultimodelNet)]
  • Đầu vào Tensor: (Batch, 2, 32, 32)
  • Shared Backbone: 3 Block Conv2D + BatchNorm + SiLU + Global Avg Pool (thuần tuần tự, không Residual)
  • Không gian ẩn (Latent Bottleneck): (Batch, 128) [Nén 16x so với Input 2048]
  • 4 Learnable Uncertainty Parameters: log_var_clf, log_var_w, log_var_l, log_var_d (Kendall 2018)
  • Rẽ 2 Header tinh gọn:
       ├── Nhánh 1 (Classification Header) -> shape_logits (Batch, 5) [Logits thô]
       └── Nhánh 2 (Regression Header)     -> y_pred_norm  (Batch, 3) [Norm W, L, D]
               │
               ▼
[BƯỚC 3: GIẢI CHUẨN HÓA (DENORMALIZATION / INVERSE TRANSFORM)]
  • Phân loại: argmax(shape_logits)  --> Ánh xạ ngược về Tên hình học (vd: "Ellipse")
  • Hồi quy:   y_pred_norm * Max     --> Thu lại kích thước thực tế [W, L, D] (mm)
  • (Tùy chọn): np.clip(..., min=0)  --> Đảm bảo tính nhất quán vật lý (không âm)
```

---

## 1. Dữ Liệu Thô Ban Đầu (Raw Input Data)

Mỗi mẫu khuyết tật kim loại quét qua hệ thống đo dòng xoáy (ECT) bao gồm 3 trường thông tin:

1. **Ảnh trường từ cảm ứng ($X$):**
   * Kích thước mảng: `(32, 32, 2)` (tương ứng lưới quét không gian $32 \times 32$ điểm đo, gồm 2 kênh từ thông/tín hiệu ECT).
   * Giá trị thực tế dao động tự do theo biên độ điện từ trường cảm ứng.
2. **Kích thước khuyết tật thật ($y$):**
   * Mảng 3 phần tử liên tục `[W, L, D]` đo bằng **milimet (mm)**:
     * $W$ (Width): Bề rộng khuyết tật nứt (khoảng $0.3 \to 1.5\text{ mm}$).
     * $L$ (Length): Chiều dài vết nứt (khoảng $2.0 \to 20.0\text{ mm}$).
     * $D$ (Depth): Độ sâu vết nứt (khoảng $0.1 \to 3.0\text{ mm}$).
3. **Nhãn hình dạng hình học (`y_shape`):**
   * Chuỗi định danh gồm 5 lớp khuyết tật:
     `'Ellipse'`, `'Rectangular'`, `'Triangular'`, `'Step_R'`, `'Step_T'`.

---

## 2. Chiến Lược Chuẩn Hóa Dữ Liệu (Normalization)

Dự án áp dụng các kỹ thuật chuẩn hóa độc lập phù hợp với từng bản chất vật lý:

| Thành phần dữ liệu | Scaler áp dụng | Công thức chuẩn hóa | Dải giá trị sau khi Norm |
| :--- | :--- | :--- | :--- |
| **Ảnh trường từ ($X$)** | `StandardScaler()` *(dòng 1150-1160)* | $X_{\text{norm}} = \dfrac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$ | Phân phối chuẩn $\approx [-3, +3]$, $\text{mean}=0, \text{std}=1$. Chuyển trục PyTorch `(Batch, 2, 32, 32)`. |
| **Kích thước ($W, L, D$)** | `SeparateMaxScaler()` *(dòng 1175-1260)* | $W_{\text{norm}} = \dfrac{W}{W_{\max}}$<br>$L_{\text{norm}} = \dfrac{L}{L_{\max}}$<br>$D_{\text{norm}} = \dfrac{D}{D_{\max}}$ | Khoảng xấp xỉ **$[0, 1]$** *(với $W_{\max}, L_{\max}, D_{\max}$ là cực đại tìm thấy trên tập Train)*. |
| **Hình dạng khuyết tật** | Label Encoding *(dòng 1068-1069)* | `shape_to_idx = {name: i}` | Số nguyên rời rạc: `0, 1, 2, 3, 4`. |

---

## 3. Cấu Trúc Mô Hình `ImprovedMultimodelNet`

Mô hình đa nhiệm (multi-task) kết hợp phân loại hình dạng + hồi quy kích thước, sử dụng Homoscedastic Uncertainty Weighting (Kendall et al., 2018).

### A. Chi tiết các khối mạng:
1. **Compact Shared Backbone (thuần tuần tự `nn.Sequential`, không có Residual/Skip connection):**
   * **Block 1:** `Conv2d(2 -> 32, 3x3, pad=1)` $\to$ `BatchNorm2d` $\to$ `SiLU` $\to$ `MaxPool2d(2x2)` $\to$ `Dropout(0.05)` *(Ra: $32 \times 16 \times 16$)*
   * **Block 2:** `Conv2d(32 -> 64, 3x3, pad=1)` $\to$ `BatchNorm2d` $\to$ `SiLU` $\to$ `MaxPool2d(2x2)` $\to$ `Dropout(0.05)` *(Ra: $64 \times 8 \times 8$)*
   * **Block 3:** `Conv2d(64 -> 128, 3x3, pad=1)` $\to$ `BatchNorm2d` $\to$ `SiLU` $\to$ `AdaptiveAvgPool2d((1, 1))` *(Ra: $128 \times 1 \times 1$)*
2. **Không gian ẩn nén (Latent Bottleneck):**
   * Làm phẳng: $\text{view}(B, -1) \implies$ **Vector Latent có kích thước `(Batch, 128)`**.
   * *Ý nghĩa:* Nén **$16\times$** so với input $2 \times 32 \times 32 = 2048$, tạo nút cổ chai ép mạng học biểu diễn trừu tượng cốt lõi.
3. **Classification Header (Phân loại):**
   * `Linear(128 -> 64)` $\to$ `BatchNorm1d` $\to$ `SiLU` $\to$ `Dropout(0.1)` $\to$ `Linear(64 -> 5)`.
   * Đầu ra: Logits thô (chưa qua Softmax).
4. **Regression Header (Hồi quy $W, L, D$):**
   * `regressor_backbone`: `Linear(128 -> 64)` $\to$ `BatchNorm1d` $\to$ `SiLU` $\to$ `Dropout(0.1)`
   * `reg_head`: `Linear(64 -> 3)` $\to$ `Sigmoid()`
   * Đầu ra: Bị chặn chặt chẽ trong dải $(0, 1)$ nhờ `Sigmoid()`, tương thích hoàn hảo với nhãn mục tiêu đã chuẩn hóa theo $y / y_{\max}$.
5. **4 Learnable Uncertainty Parameters (Kendall et al., 2018):**
   * `log_var_clf`, `log_var_w`, `log_var_l`, `log_var_d` — mỗi tham số khởi tạo $= 0.0$.
   * Dùng để cân bằng tự động trọng số giữa 4 thành phần loss (classification + 3 regression targets).
   * Công thức: $\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \frac{1}{2}\log \sigma_i^2$ với $\sigma_i^2 = e^{\text{log\_var}_i}$.

### B. Bảng phân bổ tham số:
* **Tổng số tham số:** **$\sim 110.700$** tham số (bao gồm 4 scalar uncertainty parameters).
* Giảm tới **$95\%$** so với kiến trúc cũ ($2.1$ triệu tham số), hạn chế tối đa hiện tượng học vẹt (overfitting).
* Đã gỡ bỏ toàn bộ code dư thừa (dead code: `ChannelAttention`, `SpatialAttention`) không sử dụng.

---

## 4. Cấu Hình Huấn Luyện (Training Configuration)

### A. Loss Functions:
* **Phân loại:** `nn.CrossEntropyLoss()` (đã tích hợp sẵn LogSoftmax).
* **Hồi quy:** `nn.MSELoss()` tính riêng cho từng target $W$, $L$, $D$.
* **Tổng loss dữ liệu (Kendall weighting):**
  $$\mathcal{L}_{\text{data}} = e^{-s_{\text{clf}}} \mathcal{L}_{\text{clf}} + \tfrac{1}{2}s_{\text{clf}} + e^{-s_w} \mathcal{L}_w + \tfrac{1}{2}s_w + e^{-s_l} \mathcal{L}_l + \tfrac{1}{2}s_l + e^{-s_d} \mathcal{L}_d + \tfrac{1}{2}s_d$$
  với $s_i = \text{log\_var}_i$ là tham số học được.
* **Tổng loss cuối cùng:** $\mathcal{L} = \mathcal{L}_{\text{data}} + \alpha \cdot \mathcal{L}_{\text{PINN}}$

### B. Optimizer & Scheduler:
* **Optimizer:** `Adam(lr=0.001)`
* **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=15, min_lr=1e-6)`

### C. Chiến lược PINN 2 pha (Warmup):
* **Pha 1 (Epoch 0 → 99):** Chỉ tối ưu $\mathcal{L}_{\text{data}}$ (không có PINN), để mạng hội tụ cơ bản trước.
* **Pha 2 (Epoch 100 →):** Bật $\mathcal{L}_{\text{PINN}}$ với trọng số $\alpha = 0.02$ (mặc định). Mỗi epoch: (1) data-loss optimizer step mỗi batch, (2) thêm 1 optimizer step cuối epoch từ mean PINN loss.
* **Loss normalization:** EMA (Exponential Moving Average) chuẩn hóa scale giữa các thành phần loss.
* **Gradient clipping:** `clip_grad_norm_(model.parameters(), 1.0)` khi PINN active.

---

## 5. Đầu Ra Dự Đoán Sau Mô Hình (Model Outputs)

Khi thực thi `shape_logits, y_pred_wld = model(batch_X)`:
* **`shape_logits` (Batch, 5):** Là các số thực unnormalized logits (chưa qua Softmax), ví dụ: `[2.4, -1.1, 0.3, -0.8, 5.1]`.
* **`y_pred_wld` (Batch, 3):** Là các số thực $[\hat{W}_{\text{norm}}, \hat{L}_{\text{norm}}, \hat{D}_{\text{norm}}]$ nằm chặt chẽ trong dải **$(0, 1)$** nhờ kích hoạt `Sigmoid()` cuối regression head, ví dụ: `[0.62, 0.45, 0.70]`. Mô hình đảm bảo tính nhất quán vật lý (không bị âm) ngay từ kiến trúc mạng.

---

## 6. Quy Trình Giải Chuẩn Hóa (Denormalization / Unnorm)

Để khôi phục nhãn phân loại và kích thước thực tế hiển thị trong báo cáo:

### A. Giải chuẩn hóa Phân loại:
```python
# 1. Tìm vị trí class có giá trị logit cực đại:
predicted_idx = torch.argmax(shape_logits, dim=1).cpu().numpy()

# 2. Tra từ điển ra tên khuyết tật bằng chữ:
predicted_shape = unique_shapes[predicted_idx]  # Ví dụ: 'Triangular'

# (Tùy chọn) Tính xác suất % để hiển thị/báo cáo:
confidence_pct = torch.softmax(shape_logits, dim=1).cpu().numpy() * 100.0
```

### B. Giải chuẩn hóa Hồi quy ($W, L, D$):
Thực hiện nhân ngược lại với giá trị Max ban đầu của tập huấn luyện:
$$\hat{W} = \hat{W}_{\text{norm}} \times W_{\max}, \quad \hat{L} = \hat{L}_{\text{norm}} \times L_{\max}, \quad \hat{D} = \hat{D}_{\text{norm}} \times D_{\max}$$

Code thực thi:
```python
# Nhân ngược qua scaler:
y_pred_denorm = y_scaler.inverse_transform(y_pred_wld.cpu().numpy())

# Ví dụ kết quả thực tế sau unnorm:
# Width: 0.93 mm | Length: 9.00 mm | Depth: 2.10 mm
```

---

## 7. Phân Tích: Ép Khoảng (Bounding) Đầu Ra Regression

* **Tại kiến trúc mạng (`self.reg_head`):**
  * **CÓ ÉP KHOẢNG:** Đầu ra qua lớp `nn.Sigmoid()` ép trực tiếp về dải $(0, 1)$.
  * *Lý do:* Vì target hồi quy được chuẩn hóa qua `SeparateMaxScaler` ($y / y_{\max} \in [0, 1]$), hàm `Sigmoid()` đảm bảo giá trị dự đoán không bao giờ bị âm ($< 0$) hay vượt ngưỡng phi lý mà không cần can thiệp heuristic từ bên ngoài.
* **Khi tính mất mát vật lý PINN (`compute_physics_loss_autograd`):**
  * **CÓ BẢO VỆ CLAMP:** Dùng `torch.clamp(w, 0.3, 1.5)`, `torch.clamp(l, 2.0, 20.0)`, `torch.clamp(d, 0.1, 3.0)` mm.
  * *Lý do:* Bảo vệ nghiệm giải tích lưỡng cực (Dipole kernel) không bị lỗi chia cho 0 hay căn bậc hai số âm trong quá trình autograd đạo hàm tự động.
* **Khi kiểm thử / Inference:**
  * Nhờ có `Sigmoid()`, giá trị $\hat{y}_{\text{denorm}} = \hat{y}_{\text{pred}} \times y_{\max}$ luôn đảm bảo $\ge 0$ và $\le y_{\max}$. Không cần lo lắng về kích thước nứt âm phi vật lý.

---

## 8. Vì Sao TUYỆT ĐỐI KHÔNG DÙNG Softmax ở Header?

1. **Xung đột với `nn.CrossEntropyLoss()`:**
   * Trong PyTorch, `nn.CrossEntropyLoss()` đã tích hợp sẵn toán tử `LogSoftmax + NLLLoss` qua thuật toán ổn định số học *Log-Sum-Exp*.
   * Đặt thêm `Softmax` ở header sẽ làm Softmax bị chạy **2 lần liên tiếp**, gây triệt tiêu gradient (vanishing gradient) và khiến loss không thể tối ưu.
2. **Nhánh Hồi quy kích thước ($W, L, D$):**
   * Softmax có ràng buộc toán học $\sum \text{Softmax}_i = 1$ (tổng bằng 1).
   * Kích thước $W, L, D$ là các đại lượng hình học độc lập (một khuyết tật có thể vừa rộng tối đa, vừa dài tối đa, vừa sâu tối đa). Dùng Softmax sẽ ép buộc $W+L+D=1$, làm sai lệch hoàn toàn bản chất vật lý.
