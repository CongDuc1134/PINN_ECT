import torch
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Phần Mô Phỏng (Simulation - Giữ nguyên của bạn)
# -----------------------------
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

# Physics Constants
FREQ = 5000
SICMA = 35461000
MU = 0.0000012566
CSI = 0.085
K = 1.5
I = 0.01
G = 3981
PI = math.pi

# Kernel Functions
def calT_torch(x, y, z, lc, dc, delta, xx, zz):
    term1 = (x - xx)**2 + (y + lc / 2 / dc * zz + lc / 2)**2 + (z - zz)**2
    term2 = (x - xx)**2 + (y - lc / 2 / dc * zz - lc / 2)**2 + (z - zz)**2
    term1 = torch.clamp(term1, min=1e-12)
    term2 = torch.clamp(term2, min=1e-12)
    out = torch.exp(zz / delta) * (z - zz) * (1.0 / term1**1.5 - 1.0 / term2**1.5)
    return out

def calT_add3_torch(x, y, z, wc, lc, dc, delta, yy, zz):
    term1 = (x + wc / 2)**2 + (y - yy)**2 + (z - zz)**2
    term2 = (x - wc / 2)**2 + (y - yy)**2 + (z - zz)**2
    term1 = torch.clamp(term1, min=1e-12)
    term2 = torch.clamp(term2, min=1e-12)
    out = (-2 * yy / lc) * torch.exp(zz / delta) * (z - zz) * \
          (1.0 / term1**1.5 + 1.0 / term2**1.5)
    return out

def calT_add4_torch(x, y, z, wc, lc, dc, delta, yy, zz):
    term1 = (x + wc / 2)**2 + (y - yy)**2 + (z - zz)**2
    term2 = (x - wc / 2)**2 + (y - yy)**2 + (z - zz)**2
    term1 = torch.clamp(term1, min=1e-12)
    term2 = torch.clamp(term2, min=1e-12)
    out = (2 * yy / lc) * torch.exp(zz / delta) * (z - zz) * \
          (1.0 / term1**1.5 + 1.0 / term2**1.5)
    return out

def get_trapezoidal_weights(m, n):
    W = torch.ones((m + 1, n + 1), device=device)
    W[0, :] *= 0.5; W[-1, :] *= 0.5; W[:, 0] *= 0.5; W[:, -1] *= 0.5
    return W

def InteT_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = -wc / 2, wc / 2
    c, d = -dc, 0
    xx = linspace_with_grad(a, b, m + 1)
    zz = linspace_with_grad(c, d, n + 1)
    XX, ZZ = torch.meshgrid(xx, zz, indexing='ij')
    XX_b, ZZ_b = XX[:, :, None, None], ZZ[:, :, None, None]
    F = calT_torch(X, Y, z, lc, dc, delta, XX_b, ZZ_b)
    W = get_trapezoidal_weights(m, n)[:, :, None, None]
    h, k = (d - c) / n, (b - a) / m
    return torch.sum(F * W, dim=(0, 1)) * h * k

def InteT_add3_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    sinalfa = dc / ((dc**2 + (lc / 2)**2)**0.5)
    a, b = -lc / 2, 0
    k = (b - a) / m
    xx_1d = linspace_with_grad(a, b, m + 1)
    c_1d = -xx_1d * 2 * dc / lc - dc
    d_1d = torch.zeros_like(c_1d)
    h_1d = (d_1d - c_1d) / n
    t = torch.linspace(0, 1, n + 1, device=device)
    ZZ = c_1d[:, None] + (d_1d - c_1d)[:, None] * t[None, :]
    XX = xx_1d[:, None].expand(-1, n + 1)
    XX_b, ZZ_b = XX[:, :, None, None], ZZ[:, :, None, None]
    F = calT_add3_torch(X, Y, z, wc, lc, dc, delta, XX_b, ZZ_b)
    W = get_trapezoidal_weights(m, n)[:, :, None, None]
    H_val = h_1d[:, None, None, None]
    integral = torch.sum(F * W * H_val, dim=(0, 1)) * k
    return sinalfa * integral

def InteT_add4_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    sinalfa = dc / ((dc**2 + (lc / 2)**2)**0.5)
    a, b = 0, lc / 2
    k = (b - a) / m
    xx_1d = linspace_with_grad(a, b, m + 1)
    c_1d = xx_1d * 2 * dc / lc - dc
    d_1d = torch.zeros_like(c_1d)
    h_1d = (d_1d - c_1d) / n
    t = torch.linspace(0, 1, n + 1, device=device)
    ZZ = c_1d[:, None] + (d_1d - c_1d)[:, None] * t[None, :]
    XX = xx_1d[:, None].expand(-1, n + 1)
    XX_b, ZZ_b = XX[:, :, None, None], ZZ[:, :, None, None]
    F = calT_add4_torch(X, Y, z, wc, lc, dc, delta, XX_b, ZZ_b)
    W = get_trapezoidal_weights(m, n)[:, :, None, None]
    H_val = h_1d[:, None, None, None]
    integral = torch.sum(F * W * H_val, dim=(0, 1)) * k
    return sinalfa * integral

def compute_triangular_map_gpu(wc, lc, dc, delta, z, N=16, Res=0.78):
    xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
    ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
    X_grid, Y_grid = torch.meshgrid(xs, ys, indexing='ij')
    X, Y = X_grid[None, None, :, :], Y_grid[None, None, :, :]

    term_main = InteT_map_torch(X, Y, z, wc, lc, dc, delta, n=40, m=40)
    term_add3 = InteT_add3_map_torch(X, Y, z, wc, lc, dc, delta, n=40, m=40)
    term_add4 = InteT_add4_map_torch(X, Y, z, wc, lc, dc, delta, n=40, m=40)

    H_map = (term_main + term_add3 - term_add4)
    return H_map

# -----------------------------
# 2. Phần Xử Lý Dữ Liệu (Theo Logic Load_Data của bạn)
# -----------------------------

def process_simulation_output(H_simulation_np, flip_rows=False):
    """
    Hàm này thực hiện các bước giống hệt trong Load_Data:
    1. Rot90
    2. np.diff (Mô phỏng đầu dò vi sai)
    3. Padding (Bù lại cột bị mất)
    4. Gradient Magnitude
    """
    # -- Step 1: Rotation (Giống: rotated_image = np.rot90(draft,2)) --
    rotated_image = np.rot90(H_simulation_np, 2)

    # -- Step 2: Diff (Giống: rotated_image = np.diff(rotated_image)) --
    # np.diff mặc định axis=-1 (theo cột), làm giảm chiều rộng đi 1
    # Đây là mô phỏng tín hiệu vi sai (differential signal)
    diff_image = np.diff(rotated_image, axis=1)

    # -- Step 3: Flip Rows (Tùy chọn, giống đoạn if 'Step_R' in i...) --
    # Với dữ liệu mô phỏng chuẩn, thường không cần lật, nhưng tôi để đây nếu cần
    if flip_rows:
        Draft_1 = np.zeros_like(diff_image)
        for i in range(diff_image.shape[0]):
            Draft_1[i] = diff_image[i][::-1] # Đảo ngược hàng
        diff_image = Draft_1

    # -- Step 4: Padding (Giống: copy last column) --
    # Để khôi phục kích thước ban đầu sau khi diff
    last_column = diff_image[:, -1:]
    processed_image = np.column_stack([diff_image, last_column])

    # -- Step 5: Gradient Calculation (Giống đoạn tính ∇H) --
    # Tính đạo hàm riêng
    grad_y = np.gradient(processed_image, axis=0) # ∂H/∂y
    grad_x = np.gradient(processed_image, axis=1) # ∂H/∂x

    # Tính độ lớn Gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return processed_image, gradient_magnitude

# -----------------------------
# 3. Main Execution
# -----------------------------

if __name__ == "__main__":
    # --- Input Parameters ---
    wc = 0.7
    lc = 20.0
    dc = 3.0
    z = 1.0

    print(">>> 1. Running Simulation...")
    # Calculate delta
    delta = 1 / np.sqrt(PI * FREQ * MU * SICMA) * 1000
    # Tính toán trên GPU
    H_tensor = compute_triangular_map_gpu(wc, lc, dc, delta, z)
    # Apply physics constants
    H_tensor = (CSI / 4 / PI) * H_tensor * K * I * G
    if torch.cuda.is_available(): torch.cuda.synchronize()

    # Chuyển về Numpy [H, W]
    H_raw = H_tensor.detach().cpu().numpy().squeeze()
    print(f"Raw Simulation Shape: {H_raw.shape}")

    print(">>> 2. Processing Data (Diff & Gradient)...")
    # Áp dụng logic xử lý của Load_Data
    # processed_H: là ảnh sau khi rot90 và diff
    # grad_mag: là ảnh độ lớn gradient (cái bạn cần vẽ)
    processed_H, grad_mag = process_simulation_output(H_raw, flip_rows=False)

    print(f"Processed Shape: {processed_H.shape}")
    print(f"Gradient Mag Shape: {grad_mag.shape}")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # Thiết lập trục toạ độ (giả sử N=16, Res=0.78)
    N, Res = 16, 0.78
    extent = [-N*Res, N*Res, -N*Res, N*Res]

    # Hình 1: Raw Simulation (Kết quả mô phỏng gốc)
    plt.subplot(1, 3, 1)
    plt.imshow(H_raw, cmap="jet", origin="lower", extent=extent)
    plt.title("1. Raw Simulation ($H_z$)")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Hình 2: Processed Signal (Sau khi Rot90 + Diff)
    # Đây là tín hiệu tương đương đầu dò vi sai
    plt.subplot(1, 3, 2)
    plt.imshow(processed_H, cmap="jet", origin="lower", extent=extent)
    plt.title("2. Differential Signal\n(Rot90 + Diff)")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Hình 3: Gradient Magnitude (Kết quả cuối cùng)
    plt.subplot(1, 3, 3)
    plt.imshow(grad_mag, cmap="inferno", origin="lower", extent=extent)
    plt.title("3. Gradient Magnitude\n($|\\nabla H|$)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
