import torch
import time
import matplotlib.pyplot as plt
import math
import numpy as np

# -----------------------------
# 1. CẤU HÌNH THIẾT BỊ
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


def linspace_with_grad(start, end, steps, *, device=device, dtype=torch.float32):
    """Like torch.linspace, but keeps gradient to tensor start/end values."""
    ref = start if torch.is_tensor(start) else end if torch.is_tensor(end) else None
    out_device = ref.device if ref is not None else device
    out_dtype = ref.dtype if ref is not None and ref.is_floating_point() else dtype
    start_t = start.to(device=out_device, dtype=out_dtype) if torch.is_tensor(start) else torch.tensor(start, device=out_device, dtype=out_dtype)
    end_t = end.to(device=out_device, dtype=out_dtype) if torch.is_tensor(end) else torch.tensor(end, device=out_device, dtype=out_dtype)
    t = torch.linspace(0.0, 1.0, steps, device=out_device, dtype=out_dtype)
    return start_t + (end_t - start_t) * t

# -----------------------------
# 2. CÁC HÀM KERNEL (GIỮ NGUYÊN)
# -----------------------------
def InteR_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = -wc / 2, wc / 2
    c, d = -dc, 0.0
    xx = linspace_with_grad(a, b, m + 1)
    zz = linspace_with_grad(c, d, n + 1)
    XX, ZZ = torch.meshgrid(xx, zz, indexing="ij")
    XX, ZZ = XX[:, :, None, None], ZZ[:, :, None, None]
    k, h = (b - a) / m, (d - c) / n
    F = (torch.exp(ZZ / delta) * (z - ZZ) *
        (1.0 / ((X - XX/2)**2 + (Y + lc/2)**2 + (z - ZZ)**2)**1.5
       - 1.0 / ((X - XX/2)**2 + (Y - lc/2)**2 + (z - ZZ)**2)**1.5))
    W = torch.ones_like(F)
    W[0,:,:,:] *= 0.5; W[-1,:,:,:] *= 0.5; W[:,0,:,:] *= 0.5; W[:,-1,:,:] *= 0.5
    return k * h * torch.sum(W * F, dim=(0, 1)) / (4.0 * math.pi)

def InteR_add1_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = -lc / 2, -lc / 2 + dc
    c, d = -dc, 0.0
    xx = linspace_with_grad(a, b, m + 1)
    zz = linspace_with_grad(c, d, n + 1)
    XX, ZZ = torch.meshgrid(xx, zz, indexing="ij")
    XX, ZZ = XX[:, :, None, None], ZZ[:, :, None, None]
    k, h, yy = (b - a) / m, (d - c) / n, XX
    F = (torch.exp(ZZ / delta) * (z - ZZ) * (-yy - lc/2 + dc) / dc *
        (1.0 / ((X + wc/2)**2 + (Y - yy)**2 + (z - ZZ)**2)**1.5 +
         1.0 / ((X - wc/2)**2 + (Y - yy)**2 + (z - ZZ)**2)**1.5))
    W = torch.ones_like(F)
    W[0,:,:,:] *= 0.5; W[-1,:,:,:] *= 0.5; W[:,0,:,:] *= 0.5; W[:,-1,:,:] *= 0.5
    return k * h * torch.sum(W * F, dim=(0, 1)) / (4.0 * math.pi)

def InteR_add2_map_torch(X, Y, z, wc, lc, dc, delta, n=20, m=20):
    a, b = lc / 2 - dc, lc / 2
    c, d = -dc, 0.0
    xx = linspace_with_grad(a, b, m + 1)
    zz = linspace_with_grad(c, d, n + 1)
    XX, ZZ = torch.meshgrid(xx, zz, indexing="ij")
    XX, ZZ = XX[:, :, None, None], ZZ[:, :, None, None]
    k, h, yy = (b - a) / m, (d - c) / n, XX
    F = (torch.exp(ZZ / delta) * (z - ZZ) * (yy - lc/2 + dc) / dc *
        (1.0 / ((X + wc/2)**2 + (Y - yy)**2 + (z - ZZ)**2)**1.5 +
         1.0 / ((X - wc/2)**2 + (Y - yy)**2 + (z - ZZ)**2)**1.5))
    W = torch.ones_like(F)
    W[0,:,:,:] *= 0.5; W[-1,:,:,:] *= 0.5; W[:,0,:,:] *= 0.5; W[:,-1,:,:] *= 0.5
    return k * h * torch.sum(W * F, dim=(0, 1)) / (4.0 * math.pi)

# -----------------------------
# 3. HÀM TÍNH TOÁN (SIMULATION)
# -----------------------------
def compute_magnetic_field(wc, lc, dc, delta, z, N=16, Res=0.78, angle=0, K=1.5, I=0.01, G=3981, csi=0.085):
    xs = torch.linspace(-N*Res, N*Res, 2*N, device=device)
    ys = torch.linspace(-N*Res, N*Res, 2*N, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    rad = angle * math.pi / 180
    X_rot = X * math.cos(rad) - Y * math.sin(rad)
    Y_rot = X * math.sin(rad) + Y * math.cos(rad)

    term1 = InteR_map_torch(X_rot, Y_rot, z, wc, lc, dc, delta)
    term2 = InteR_add1_map_torch(X_rot, Y_rot, z, wc, lc, dc, delta)
    term3 = InteR_add2_map_torch(X_rot, Y_rot, z, wc, lc, dc, delta)

    H = (term1 + term2 - term3) * math.cos(rad) * csi * K * I * G
    return H, xs, ys

# -----------------------------
# 4. HÀM XỬ LÝ (PROCESSING)
# -----------------------------
def process_simulation_output(H_simulation_np):
    # 1. Rot90
    rotated_image = np.rot90(H_simulation_np, 2)
    # 2. Diff (Mô phỏng vi sai)
    diff_image = np.diff(rotated_image, axis=1)
    # 3. Padding (Bù cột)
    last_column = diff_image[:, -1:]
    processed_image = np.column_stack([diff_image, last_column])
    # 4. Gradient Magnitude
    grad_y = np.gradient(processed_image, axis=0)
    grad_x = np.gradient(processed_image, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return processed_image, gradient_magnitude

# -----------------------------
# 5. MAIN (CHẠY 1 THAM SỐ)
# -----------------------------
if __name__ == "__main__":
    # === NHẬP THAM SỐ Ở ĐÂY ===
    wc_in = 0.41    # Chiều rộng
    lc_in = 2.0    # Chiều dài
    dc_in = 0.1     # Chiều sâu
    # ==========================

    # Các hằng số vật lý
    z = 1.0
    freq, sicma, mu = 5000, 35461000, 0.0000012566
    csi, K, I, G = 0.085, 1.5, 0.01, 3981
    delta_val = 1 / math.sqrt(math.pi * freq * mu * sicma) * 1000
    angle = 0

    print(f"\n>>> Running Single Case: wc={wc_in}, lc={lc_in}, dc={dc_in}")

    # --- 1. MÔ PHỎNG ---
    t0 = time.perf_counter()
    H_tensor, xs, ys = compute_magnetic_field(
        wc_in, lc_in, dc_in, delta_val, z, angle=angle, K=K, I=I, G=G, csi=csi
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    H_raw = H_tensor.detach().cpu().numpy() # Chuyển về Numpy

    # --- 2. XỬ LÝ (Diff + Gradient) ---
    processed_sig, grad_mag = process_simulation_output(H_raw)

    print(f"Done in {t1 - t0:.4f}s")
    print(f"Output Shapes -> Raw: {H_raw.shape}, Processed: {grad_mag.shape}")

    # --- 3. VẼ ĐỒ THỊ ---
    extent = [xs[0].item(), xs[-1].item(), ys[0].item(), ys[-1].item()]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Hình 1: Raw Simulation
    im1 = ax1.imshow(H_raw, cmap="jet", origin="lower", extent=extent)
    ax1.set_title(f"Raw Simulation ($H_z$)\nwc={wc_in}, dc={dc_in}")
    plt.colorbar(im1, ax=ax1)

    # Hình 2: Differential
    im2 = ax2.imshow(processed_sig, cmap="jet", origin="lower", extent=extent)
    ax2.set_title("Differential Signal\n(Rot90 + Diff)")
    plt.colorbar(im2, ax=ax2)

    # Hình 3: Gradient Magnitude
    im3 = ax3.imshow(grad_mag, cmap="inferno", origin="lower", extent=extent)
    ax3.set_title("Gradient Magnitude $|\\nabla H|$\n(Final Input)")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()
