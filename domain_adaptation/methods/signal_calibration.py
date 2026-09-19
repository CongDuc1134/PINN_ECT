# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/signal_calibration.py
Analytical Pre-processing & Sensor Calibration for Real ECT 5kHz Measurements.
Removes probe baseline DC drift, filters high-frequency sensor noise,
and calibrates gradient magnitude channel to match simulation FEM distributions.
================================================================================
"""

import numpy as np
import scipy.ndimage
import torch


def calibrate_single_scan_matrix(matrix_2d: np.ndarray, apply_smoothing: bool = True, sigma: float = 0.6) -> np.ndarray:
    """
    Calibrates a single 32x32 raw ECT scan matrix:
    1. Robust background nulling using border quartile estimation.
    2. Spatial Gaussian smoothing (sigma) to remove high-frequency probe vibration.
    3. Gradient magnitude recomputation.
    Returns: (32, 32, 2) calibrated two-channel representation.
    """
    arr = np.array(matrix_2d, dtype=np.float32)

    # 1. Resize if needed
    if arr.shape != (32, 32):
        zoom_factors = (32.0 / arr.shape[0], 32.0 / arr.shape[1])
        arr = scipy.ndimage.zoom(arr, zoom_factors, order=1).astype(np.float32)

    # 2. Border baseline nulling (estimate sensor zero offset from 4 outer rows/cols)
    borders = np.concatenate([
        arr[:3, :].flatten(),
        arr[-3:, :].flatten(),
        arr[:, :3].flatten(),
        arr[:, -3:].flatten()
    ])
    baseline_offset = np.median(borders)
    calibrated_field = arr - baseline_offset

    # 3. Spatial Gaussian filter for sensor jitter suppression
    if apply_smoothing and sigma > 0:
        calibrated_field = scipy.ndimage.gaussian_filter(calibrated_field, sigma=sigma)

    # 4. Recompute spatial gradient magnitude (∇H)
    grad_y = np.gradient(calibrated_field, axis=0)
    grad_x = np.gradient(calibrated_field, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 5. Stack into (32, 32, 2)
    return np.dstack([calibrated_field, grad_mag]).astype(np.float32)


def calibrate_real_tensor(X_tensor: torch.Tensor, x_scaler=None, sigma: float = 0.6) -> torch.Tensor:
    """
    Calibrates a batch of real measurement tensors (N, 2, 32, 32).
    Applies baseline nulling and re-scales with x_scaler if provided.
    """
    device = X_tensor.device
    X_np = X_tensor.cpu().numpy()
    N = X_np.shape[0]

    calibrated_list = []
    for i in range(N):
        # Extract channel 0 (field)
        field = X_np[i, 0, :, :]
        cal_2ch = calibrate_single_scan_matrix(field, apply_smoothing=True, sigma=sigma)
        calibrated_list.append(cal_2ch)

    cal_arr = np.array(calibrated_list, dtype=np.float32)  # (N, 32, 32, 2)

    if x_scaler is not None:
        flat = cal_arr.reshape(-1, 2)
        scaled = x_scaler.transform(flat)
        cal_arr = scaled.reshape(N, 32, 32, 2)

    cal_tensor = torch.tensor(cal_arr, dtype=torch.float32).permute(0, 3, 1, 2).contiguous().to(device)
    return cal_tensor
