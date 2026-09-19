import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# ... (Giữ nguyên các hàm calSR1 -> calSR6 và generic_integration từ code trước) ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def linspace_with_grad(start, end, steps, *, device=device, dtype=dtype):
    """Like torch.linspace, but keeps gradient to tensor start/end values."""
    ref = start if torch.is_tensor(start) else end if torch.is_tensor(end) else None
    out_device = ref.device if ref is not None else device
    out_dtype = ref.dtype if ref is not None and ref.is_floating_point() else dtype
    start_t = start.to(device=out_device, dtype=out_dtype) if torch.is_tensor(start) else torch.tensor(start, device=out_device, dtype=out_dtype)
    end_t = end.to(device=out_device, dtype=out_dtype) if torch.is_tensor(end) else torch.tensor(end, device=out_device, dtype=out_dtype)
    t = torch.linspace(0.0, 1.0, steps, device=out_device, dtype=out_dtype)
    return start_t + (end_t - start_t) * t

def get_trapz_weights(m, n, device):
    W = torch.ones((m + 1, n + 1), device=device, dtype=dtype)
    W[0, :] *= 0.5; W[-1, :] *= 0.5; W[:, 0] *= 0.5; W[:, -1] *= 0.5
    return W[:, :, None, None]

def calSR1(X, Y, z, lc, delta, xx, zz):
    return torch.exp(zz / delta) * (z - zz) / ((X - xx) ** 2 + (Y + lc / 2) ** 2 + (z - zz) ** 2) ** 1.5

def calSR2(X, Y, z, wc, lc, dc, delta, yy, zz):
    term = (1 / ((X + wc / 2) ** 2 + (Y - yy) ** 2 + (z - zz) ** 2) ** 1.5 +
            1 / ((X - wc / 2) ** 2 + (Y - yy) ** 2 + (z - zz) ** 2) ** 1.5)
    return torch.exp(zz / delta) * (z - zz) * ((-2 * yy - lc + dc) / dc) * term

def calSR3(X, Y, z, delta, xx, zz):
    return torch.exp(zz / delta) * (z - zz) / ((X - xx) ** 2 + Y**2 + (z - zz) ** 2) ** 1.5

def calSR4(X, Y, z, wc, dc, delta, yy, zz):
    term = (1 / ((X + wc / 2) ** 2 + (Y - yy) ** 2 + (z - zz) ** 2) ** 1.5 +
            1 / ((X - wc / 2) ** 2 + (Y - yy) ** 2 + (z - zz) ** 2) ** 1.5)
    return torch.exp(zz / delta) * (z - zz) * ((-2 * yy + dc) / dc) * term

def calSR5(X, Y, z, lc, delta, xx, zz):
    return torch.exp(zz / delta) * (z - zz) / ((X - xx) ** 2 + (Y - lc / 2) ** 2 + (z - zz) ** 2) ** 1.5

def calSR6(X, Y, z, wc, lc, dc, delta, yy, zz):
    term = (1 / ((X + wc / 2) ** 2 + (Y - yy) ** 2 + (z - zz) ** 2) ** 1.5 +
            1 / ((X - wc / 2) ** 2 + (Y - yy) ** 2 + (z - zz) ** 2) ** 1.5)
    return torch.exp(zz / delta) * (z - zz) * ((yy - lc / 2 + dc) / dc) * term

def generic_integration(func, X, Y, z, args_dict, x_range, z_range, m=20, n=20):
    xx = linspace_with_grad(x_range[0], x_range[1], m + 1)
    zz = linspace_with_grad(z_range[0], z_range[1], n + 1)
    XX, ZZ = torch.meshgrid(xx, zz, indexing="ij")
    XX_b, ZZ_b = XX[:, :, None, None], ZZ[:, :, None, None]
    F = func(X, Y, z, **args_dict, xx=XX_b, zz=ZZ_b)
    k, h = (x_range[1] - x_range[0]) / m, (z_range[1] - z_range[0]) / n
    W = get_trapz_weights(m, n, device)
    return k * h * torch.sum(W * F, dim=(0, 1))

def generic_integration_v2(func, X, Y, z, args_dict, x_range, z_range, m=20, n=20):
    xx = linspace_with_grad(x_range[0], x_range[1], m + 1)
    zz = linspace_with_grad(z_range[0], z_range[1], n + 1)
    XX, ZZ = torch.meshgrid(xx, zz, indexing="ij")
    YY_b, ZZ_b = XX[:, :, None, None], ZZ[:, :, None, None]
    F = func(X, Y, z, **args_dict, yy=YY_b, zz=ZZ_b)
    k, h = (x_range[1] - x_range[0]) / m, (z_range[1] - z_range[0]) / n
    W = get_trapz_weights(m, n, device)
    return k * h * torch.sum(W * F, dim=(0, 1))

def calculate_field_Step_R(X, Y, z, wc, lc, dc, delta):
    v1 = generic_integration(calSR1, X, Y, z, {'lc': lc, 'delta': delta}, (-wc/2, wc/2), (-dc/2, 0))
    v2 = generic_integration_v2(calSR2, X, Y, z, {'wc': wc, 'lc': lc, 'dc': dc, 'delta': delta}, (-lc/2, (-lc+dc)/2), (-dc/2, 0))
    v5 = generic_integration(calSR5, X, Y, z, {'lc': lc, 'delta': delta}, (-wc/2, wc/2), (-dc, 0))
    v6 = generic_integration_v2(calSR6, X, Y, z, {'wc': wc, 'lc': lc, 'dc': dc, 'delta': delta}, (lc/2 - dc, lc/2), (-dc, 0))
    v3 = generic_integration(calSR3, X, Y, z, {'delta': delta}, (-wc/2, wc/2), (-dc, -dc/2))
    v4 = generic_integration_v2(calSR4, X, Y, z, {'wc': wc, 'dc': dc, 'delta': delta}, (0, dc/2), (-dc, -dc/2))
    return v1 + v2 - v5 - v6 + v3 + v4

def process_like_load_data(draft_np):
    rotated_image = np.rot90(draft_np, 2)
    rotated_image = np.diff(rotated_image, axis=1)
    draft_1 = np.zeros((32, 31))
    for i in range(32):
        draft_1[i] = rotated_image[i][::-1]
    tam_1 = np.zeros((32, 1))
    final_image = np.insert(draft_1, [31], tam_1, axis=1)
    return final_image

if __name__ == "__main__":
    wc, lc, dc = 1.0, 15.0, 3.0
    N, Res, z_lift = 16, 0.78, 1.0
    freq, mu, sicma = 5000.0, 0.0000012566, 35461000.0
    csi, K, I, G = 0.085, 1.5, 0.01, 3981.0

    delta = 1.0 / math.sqrt(math.pi * freq * mu * sicma) * 1000.0
    xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device, dtype=dtype)
    ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device, dtype=dtype)
    X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")

    with torch.no_grad():
        H_val = calculate_field_Step_R(X_grid, Y_grid, z_lift, wc, lc, dc, delta)
        H_final = H_val * (csi / 4 / math.pi) * K * I * G

    H_draft = H_final.cpu().numpy()
    final_output = process_like_load_data(H_draft)

    # --- VẼ ĐỒ THỊ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Heatmap
    extent = [ys[0].item(), ys[-1].item(), xs[0].item(), xs[-1].item()]
    im = ax1.imshow(final_output, cmap='jet', origin='lower', extent=extent, aspect='auto')
    plt.colorbar(im, ax=ax1)
    ax1.set_title("Step_R Heatmap")
    ax1.set_xlabel("Length (mm)"); ax1.set_ylabel("Width (mm)")

    # 2. Section View (Cắt ngang tại Width = 0)
    mid_idx = final_output.shape[0] // 2
    section_data = final_output[mid_idx, :]
    dist_length = np.linspace(ys[0].item(), ys[-1].item(), final_output.shape[1])

    ax2.plot(dist_length, section_data, 'r-', linewidth=2)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title(f"Section View at Centerline (Width ≈ 0)")
    ax2.set_xlabel("Length (mm)"); ax2.set_ylabel("H Intensity")

    plt.tight_layout()
    plt.show()
