import numpy as np
import time
import matplotlib.pyplot as plt
import torch

# -------------------------------------------------
# Device Configuration
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def linspace_with_grad(start, end, steps, *, device=device, dtype=torch.float32):
    """Like torch.linspace, but keeps gradient to tensor start/end values."""
    ref = start if torch.is_tensor(start) else end if torch.is_tensor(end) else None
    out_device = ref.device if ref is not None else device
    out_dtype = ref.dtype if ref is not None and ref.is_floating_point() else dtype
    start_t = start.to(device=out_device, dtype=out_dtype) if torch.is_tensor(start) else torch.tensor(start, device=out_device, dtype=out_dtype)
    end_t = end.to(device=out_device, dtype=out_dtype) if torch.is_tensor(end) else torch.tensor(end, device=out_device, dtype=out_dtype)
    t = torch.linspace(0.0, 1.0, steps, device=out_device, dtype=out_dtype)
    return start_t + (end_t - start_t) * t

# -------------------------------------------------
# Simulation Functions (InteE)
# -------------------------------------------------
def InteE_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = -wc / 2.0, wc / 2.0
    c, d = -dc, 0.0

    x_s = linspace_with_grad(a, b, m + 1)
    z_s = linspace_with_grad(c, d, n + 1)
    XX, ZZ = torch.meshgrid(x_s, z_s, indexing="ij")

    XX = XX[None, None, :, :]
    ZZ = ZZ[None, None, :, :]
    X_exp = X[:, :, None, None]
    Y_exp = Y[:, :, None, None]

    k = (b - a) / m
    h = (d - c) / n

    ellipse_term = torch.clamp(1.0 - (ZZ / dc) ** 2, min=1e-8)
    yy = (lc / 2.0) * torch.sqrt(ellipse_term)

    dist_sq_1 = (X_exp - XX) ** 2 + (Y_exp + yy) ** 2 + (z - ZZ) ** 2
    dist_sq_2 = (X_exp - XX) ** 2 + (Y_exp - yy) ** 2 + (z - ZZ) ** 2

    F = torch.exp(ZZ / delta) * (z - ZZ) * (
        1.0 / (dist_sq_1 ** 1.5) - 1.0 / (dist_sq_2 ** 1.5)
    )

    W = torch.ones_like(F)
    W[:, :, 0, :] *= 0.5
    W[:, :, -1, :] *= 0.5
    W[:, :, :, 0] *= 0.5
    W[:, :, :, -1] *= 0.5

    return k * h * torch.sum(W * F, dim=(2, 3))

def InteE_add1_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = -lc / 2.0, 0.0
    y_lin = linspace_with_grad(a, b, m + 1)
    ellipse_term = torch.clamp(1.0 - (2.0 * y_lin / lc) ** 2, min=1e-8)
    c_depth = -dc * torch.sqrt(ellipse_term)
    t = torch.linspace(0.0, 1.0, n + 1, device=device)

    y_source = y_lin[:, None].expand(m + 1, n + 1)
    z_source = c_depth[:, None] + (0.0 - c_depth[:, None]) * t[None, :]
    Hh = (0.0 - c_depth) / n

    X_exp = X[:, :, None, None]
    Y_exp = Y[:, :, None, None]
    YY = y_source[None, None, :, :]
    ZZ = z_source[None, None, :, :]
    HH_step = Hh[None, None, :, None]

    tangalfa = YY * (2.0 * dc / lc) ** 2 / (ZZ - 1e-12)
    dist_sq_plus = (X_exp + wc / 2) ** 2 + (Y_exp - YY) ** 2 + (z - ZZ) ** 2
    dist_sq_minus = (X_exp - wc / 2) ** 2 + (Y_exp - YY) ** 2 + (z - ZZ) ** 2

    F = (
        tangalfa / torch.sqrt(1.0 + tangalfa**2)
        * torch.exp(ZZ / delta) * (z - ZZ) * (-2.0 * YY / lc)
        * (1.0 / (dist_sq_plus ** 1.5) + 1.0 / (dist_sq_minus ** 1.5))
    )

    W = torch.ones_like(F)
    W[:, :, 0, :] *= 0.5; W[:, :, -1, :] *= 0.5
    W[:, :, :, 0] *= 0.5; W[:, :, :, -1] *= 0.5

    dy = (b - a) / m
    return torch.sum(W * F * HH_step, dim=(2, 3)) * dy

def InteE_add2_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = 0.0, lc / 2.0
    y_lin = linspace_with_grad(a, b, m + 1)
    ellipse_term = torch.clamp(1.0 - (2.0 * y_lin / lc) ** 2, min=1e-8)
    c_depth = -dc * torch.sqrt(ellipse_term)
    t = torch.linspace(0.0, 1.0, n + 1, device=device)

    y_source = y_lin[:, None].expand(m + 1, n + 1)
    z_source = c_depth[:, None] + (0.0 - c_depth[:, None]) * t[None, :]
    Hh = (0.0 - c_depth) / n

    X_exp = X[:, :, None, None]
    Y_exp = Y[:, :, None, None]
    YY = y_source[None, None, :, :]
    ZZ = z_source[None, None, :, :]
    HH_step = Hh[None, None, :, None]

    tangalfa = -YY * (2.0 * dc / lc) ** 2 / (ZZ - 1e-12)
    dist_sq_plus = (X_exp + wc / 2) ** 2 + (Y_exp - YY) ** 2 + (z - ZZ) ** 2
    dist_sq_minus = (X_exp - wc / 2) ** 2 + (Y_exp - YY) ** 2 + (z - ZZ) ** 2

    F = (
        tangalfa / torch.sqrt(1.0 + tangalfa**2)
        * torch.exp(ZZ / delta) * (z - ZZ) * (2.0 * YY / lc)
        * (1.0 / (dist_sq_plus ** 1.5) + 1.0 / (dist_sq_minus ** 1.5))
    )

    W = torch.ones_like(F)
    W[:, :, 0, :] *= 0.5; W[:, :, -1, :] *= 0.5
    W[:, :, :, 0] *= 0.5; W[:, :, :, -1] *= 0.5

    dy = (b - a) / m
    return torch.sum(W * F * HH_step, dim=(2, 3)) * dy

def compute_map_gpu(wc, lc, dc, delta, z, N=16, Res=0.78):
    xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
    ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    H = (
        InteE_map_torch(X, Y, z, wc, lc, dc, delta)
        + InteE_add1_map_torch(X, Y, z, wc, lc, dc, delta)
        - InteE_add2_map_torch(X, Y, z, wc, lc, dc, delta)
    )
    return H

# -------------------------------------------------
# DATA PROCESSING FUNCTION (Adapted from Load_Data)
# -------------------------------------------------
def process_simulation_data(H_input):
    """
    Hàm này thay thế cho Load_Data.
    Input: H_input (Ma trận numpy 32x32 từ mô phỏng)
    Output: Ma trận (32, 32, 2) chứa [Processed_Field, Gradient_Magnitude]
    """

    # 1. Rotated image (Tương đương np.rot90(draft, 2))
    # Xoay 180 độ
    rotated_image = np.rot90(H_input, 2)

    # 2. Diff (Tương đương np.diff(rotated_image))
    # Tính vi phân (mô phỏng đầu dò vi sai), mặc định axis=-1 (theo chiều ngang)
    # Kết quả shape sẽ giảm từ 32 -> 31 cột
    diff_image = np.diff(rotated_image, axis=1)

    # Lưu ý: Trong hàm Load_Data cũ của bạn có đoạn check 'Step_R'/'Step_T' để đảo chiều dòng (zigzag scan).
    # Vì dữ liệu mô phỏng là lưới chuẩn (Grid), ta không cần bước đảo dòng này.

    # 3. Padding (Copy cột cuối cùng sang cột 31)
    # Khôi phục shape từ 32x31 -> 32x32
    last_column = diff_image[:, -1:]
    processed_image = np.column_stack([diff_image, last_column])

    # 4. Tính đạo hàm Gradient (∇H)
    grad_y = np.gradient(processed_image, axis=0) # Đạo hàm dọc
    grad_x = np.gradient(processed_image, axis=1) # Đạo hàm ngang

    # Độ lớn gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 5. Stack dữ liệu
    # Kết quả trả về gồm 2 lớp: Lớp 0 là tín hiệu sau diff, Lớp 1 là Gradient Magnitude
    # Shape: (32, 32, 2)
    result = np.dstack([processed_image, gradient_magnitude])

    return result

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    # --- 1. SET PARAMETERS ---
    wc, lc, dc = 0.9, 20.0, 3.0
    freq = 5000
    sicma = 35461000
    mu = 0.0000012566
    delta = 1 / np.sqrt(np.pi * freq * mu * sicma) * 1000
    z_lift = 1.0
    csi = 0.085
    K = 1.5
    I = 0.01
    G = 3981

    # --- 2. RUN SIMULATION ---
    print("Running simulation...")
    # Warmup
    _ = compute_map_gpu(wc, lc, dc, delta, z_lift)

    t0 = time.perf_counter()
    H_tensor = compute_map_gpu(wc, lc, dc, delta, z_lift)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Convert to Physical Value
    H_np = H_tensor.detach().cpu().numpy()
    H_np = (csi / 4 / np.pi) * H_np
    H_np = K * I * G * H_np

    print(f"Simulation done in {t1-t0:.4f}s. Shape: {H_np.shape}")

    # --- 3. APPLY DATA PROCESSING (Load_Data Logic) ---
    print("Processing data (Rotate -> Diff -> Gradient)...")
    final_data = process_simulation_data(H_np)

    print(f"Final Data Shape: {final_data.shape}") # (32, 32, 2)

    # Tách ra để vẽ
    processed_field = final_data[:, :, 0] # Kênh 0: Ảnh sau khi diff
    grad_magnitude = final_data[:, :, 1]  # Kênh 1: Độ lớn Gradient

    # --- 4. VISUALIZATION ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Setup extent
    N, Res = 16, 0.78
    extent = [-N*Res, N*Res, -N*Res, N*Res]

    # Ảnh 1: Tín hiệu gốc H (Raw Simulation)
    im1 = axs[0].imshow(H_np, cmap="jet", origin="lower", extent=extent)
    axs[0].set_title("Original Simulation (H)")
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    # Ảnh 2: Tín hiệu sau xử lý (Rotated + Diff)
    # Lưu ý: diff làm thay đổi ý nghĩa vật lý, thường dùng để mô phỏng cảm biến vi sai
    im2 = axs[1].imshow(processed_field, cmap="jet", origin="lower", extent=extent)
    axs[1].set_title("Processed (Rot180 + Diff)")
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    # Ảnh 3: Gradient Magnitude
    im3 = axs[2].imshow(grad_magnitude, cmap="inferno", origin="lower", extent=extent)
    axs[2].set_title("Gradient Magnitude ||∇H||")
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Nếu bạn muốn tạo dataset X như trong Load_Data:
    # X = []
    # X.append(final_data)
    # X = np.array(X)
