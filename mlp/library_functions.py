# -*- coding: utf-8 -*-
"""
Library functions for ECT MFL Data Processing.
"""

import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm


def Load_Data_With_Labels(data_folder_path, labels_csv_path):
    """
    Load MFL signal data from CSV files in data_folder_path and align with labels_csv_path.

    Returns:
        X: numpy array of data (N, 32, 32, 2) - [Field value channel, Gradient magnitude channel]
        y: numpy array of labels (N, 3) - [width, length, depth]
        matched_filenames: list of successfully loaded filenames
    """
    # 1. Load labels from CSV
    labels_df = pd.read_csv(labels_csv_path)
    labels_dict = {}
    for idx, row in labels_df.iterrows():
        try:
            filename = str(row['filename']).strip()
            w = float(row['width'])
            l = float(row['length'])
            d = float(row['depth'])
            labels_dict[filename] = [w, l, d]
        except (ValueError, KeyError):
            continue

    # 2. Get all CSV files from data folder, sorted for consistency
    csv_files = sorted([f for f in os.listdir(data_folder_path) if f.endswith('.csv')])

    # 3. Load data and match with labels
    X = []
    y = []
    matched_filenames = []

    for csv_file in tqdm(csv_files, desc="📊 Loading Data with Labels", unit="file", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        if csv_file not in labels_dict:
            print(f"  [WARNING] {csv_file} not found in labels CSV")
            continue

        try:
            filepath = os.path.join(data_folder_path, csv_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                tam = []
                line = [row for row in reader]
                for k in range(4, len(line)):
                    tam1 = []
                    for l in range(len(line[k])):
                        tam1.append(float(line[k][l]))
                    tam.append(tam1)

                draft = np.array(tam)
                rotated_image = np.rot90(draft, 2)
                rotated_image = np.diff(rotated_image)

                if ('Step_R' in csv_file) or ('Step_T' in csv_file):
                    Draft_1 = np.zeros((32, 31))
                    for i in range(32):
                        Draft_1[i] = rotated_image[i][::-1]
                    rotated_image = Draft_1

                # Copy last column to fill 32x32
                last_column = rotated_image[:, -1:]
                rotated_image = np.column_stack([rotated_image, last_column])

                # Gradient calculation (∇H)
                grad_y = np.gradient(rotated_image, axis=0)
                grad_x = np.gradient(rotated_image, axis=1)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Stack field value + gradient magnitude
                field_with_gradient = np.dstack([rotated_image, gradient_magnitude])

                X.append(field_with_gradient)
                y.append(labels_dict[csv_file])
                matched_filenames.append(csv_file)

        except Exception as e:
            print(f"  ERROR loading {csv_file}: {str(e)}")
            continue

    X = np.array(X)
    y = np.array(y)

    print(f"\n{'='*70}")
    print("DATA-LABEL ALIGNMENT VERIFICATION")
    print(f"{'='*70}")
    print(f"Total files loaded: {len(matched_filenames)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if len(X) != len(y):
        print(f"  ERROR: Mismatch! X has {len(X)} samples but y has {len(y)} labels")
    elif len(X) != len(matched_filenames):
        print(f"  ERROR: Mismatch! X has {len(X)} samples but only {len(matched_filenames)} filenames")
    else:
        print(f"  OK: All {len(X)} samples, labels, and filenames are aligned")

    print(f"\nFirst 3 samples (for manual verification):")
    n_show = 3 if len(matched_filenames) >= 3 else len(matched_filenames)
    for i in range(n_show):
        print(f"  {i}: {matched_filenames[i]:<40} | W={y[i,0]:.4f}, L={y[i,1]:.4f}, D={y[i,2]:.4f} | Shape={X[i].shape}")
    print(f"{'='*70}\n")

    return X, y, matched_filenames


def Load_Real_Experiment_Data(target_path, labels_csv_path=None):
    """
    Nạp dữ liệu thực tế (Experiment_1) cho MLP, trả về mảng X (N, 32, 32, 2) và danh sách tên file.
    """
    if os.path.isdir(target_path):
        files = sorted([f for f in os.listdir(target_path) if f.endswith('.csv') and f != 'labels.csv'])
    elif os.path.isfile(target_path):
        files = [os.path.basename(target_path)]
        target_path = os.path.dirname(target_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy: {target_path}")

    X = []
    for f in files:
        fpath = os.path.join(target_path, f)
        raw = pd.read_csv(fpath, header=None).values.astype(np.float32)
        if raw.shape == (31, 31):
            raw = np.vstack([raw, raw[-1:, :]])
            raw = np.hstack([raw, raw[:, -1:]])
        elif raw.shape[0] == 32 and raw.shape[1] == 31:
            raw = np.hstack([raw, raw[:, -1:]])
        grad_y = np.gradient(raw, axis=0)
        grad_x = np.gradient(raw, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        two_ch = np.dstack([raw, grad_mag])
        X.append(two_ch)

    return np.array(X, dtype=np.float32), files

