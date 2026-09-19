# -*- coding: utf-8 -*-
"""
===============================================================================
DEDICATED REAL EXPERIMENTAL DATA LOADER & SIM-TO-REAL EVALUATION MODULE
===============================================================================
Module chuyên dụng để nạp và tiền xử lý dữ liệu thực nghiệm (Experiment_1)
phục vụ chạy Inference (suy luận) và đánh giá Sim-to-Real trên mô hình AI (CNN / MLP).

Các tính năng chính:
1. Đọc file CSV thực tế từ Experiment_1 (Trainning/5kHz, Testing/10kHz, Testing/20kHz).
2. Tự động kiểm tra và chuẩn hóa ma trận về đúng kích thước 32x32.
3. Tự động tính kênh độ lớn Gradient không gian (∇H).
4. Ghép thành Tensor 2 kênh chuẩn: [Trường vi sai H, Độ lớn Gradient ||∇H||] shape (N, 2, 32, 32) hoặc (N, 2048).
5. Trích xuất Ground Truth chính xác từ labels.csv và Table 2 (Le et al., 2013).
6. Đánh giá tự động toàn diện trên cả 3 tập tần số (5kHz, 10kHz, 20kHz) và xuất báo cáo CSV chi tiết.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import scipy.ndimage
import torch

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# =============================================================================
# SCALER CLASSES & COMPATIBILITY LAYER
# =============================================================================
class MaxScaler:
    """Scaler chuẩn hóa chia cho giá trị cực đại."""
    def __init__(self, max_val=1.0):
        self.max_val = float(max_val)
        self.data_min_ = 0.0
        self.data_max_ = float(max_val)

    def transform(self, X):
        return np.asarray(X, dtype=np.float32) / (self.max_val if self.max_val != 0 else 1.0)

    def inverse_transform(self, X_norm):
        return np.asarray(X_norm, dtype=np.float32) * self.max_val


class SeparateMaxScaler:
    """Scaler gồm 3 MaxScaler độc lập cho Width, Length, Depth."""
    def __init__(self, w_scaler=None, l_scaler=None, d_scaler=None):
        self.w_scaler = w_scaler if w_scaler is not None else MaxScaler(1.0)
        self.l_scaler = l_scaler if l_scaler is not None else MaxScaler(1.0)
        self.d_scaler = d_scaler if d_scaler is not None else MaxScaler(1.0)
        self.data_max_ = np.array([self.w_scaler.max_val, self.l_scaler.max_val, self.d_scaler.max_val])
        self.data_min_ = np.array([0.0, 0.0, 0.0])

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            return np.array([self.w_scaler.transform(X[0]), self.l_scaler.transform(X[1]), self.d_scaler.transform(X[2])])
        X_norm = X.copy()
        X_norm[:, 0] = self.w_scaler.transform(X[:, 0])
        X_norm[:, 1] = self.l_scaler.transform(X[:, 1])
        X_norm[:, 2] = self.d_scaler.transform(X[:, 2])
        return X_norm

    def inverse_transform(self, X_norm):
        X_norm = np.asarray(X_norm, dtype=np.float32)
        if X_norm.ndim == 1:
            return np.array([self.w_scaler.inverse_transform(X_norm[0]), self.l_scaler.inverse_transform(X_norm[1]), self.d_scaler.inverse_transform(X_norm[2])])
        X_denorm = X_norm.copy()
        X_denorm[:, 0] = self.w_scaler.inverse_transform(X_norm[:, 0])
        X_denorm[:, 1] = self.l_scaler.inverse_transform(X_norm[:, 1])
        X_denorm[:, 2] = self.d_scaler.inverse_transform(X_norm[:, 2])
        return X_denorm


# Tự động đăng ký các lớp scaler vào __main__ để pickle.load không bị lỗi AttributeError
if '__main__' in sys.modules:
    main_mod = sys.modules['__main__']
    if not hasattr(main_mod, 'SeparateMaxScaler'):
        setattr(main_mod, 'SeparateMaxScaler', SeparateMaxScaler)
    if not hasattr(main_mod, 'MaxScaler'):
        setattr(main_mod, 'MaxScaler', MaxScaler)

# Bảng thông số kích thước thật chuẩn từ Table 2 (Le et al., 2013)
# unique_shapes chuẩn: ['Ellipse', 'Rectangular', 'Step_R', 'Step_T', 'Triangular'] -> ID: 0, 1, 2, 3, 4
DEFAULT_UNIQUE_SHAPES = ['Ellipse', 'Rectangular', 'Step_R', 'Step_T', 'Triangular']

TABLE_2_GROUND_TRUTH = {
    1:  {'shape': 'Rectangular', 'shape_name': 'Rectangular', 'shape_id': 1, 'width': 0.7, 'length': 10.0, 'depth': 1.0, 'W': 0.7, 'L': 10.0, 'D': 1.0},
    2:  {'shape': 'Rectangular', 'shape_name': 'Rectangular', 'shape_id': 1, 'width': 0.7, 'length': 10.0, 'depth': 2.0, 'W': 0.7, 'L': 10.0, 'D': 2.0},
    3:  {'shape': 'Rectangular', 'shape_name': 'Rectangular', 'shape_id': 1, 'width': 0.7, 'length': 10.0, 'depth': 3.0, 'W': 0.7, 'L': 10.0, 'D': 3.0},
    4:  {'shape': 'Ellipse',     'shape_name': 'Ellipse',     'shape_id': 0, 'width': 0.7, 'length': 10.0, 'depth': 3.0, 'W': 0.7, 'L': 10.0, 'D': 3.0},
    5:  {'shape': 'Ellipse',     'shape_name': 'Ellipse',     'shape_id': 0, 'width': 0.9, 'length': 10.0, 'depth': 3.0, 'W': 0.9, 'L': 10.0, 'D': 3.0},
    6:  {'shape': 'Triangular',  'shape_name': 'Triangular',  'shape_id': 4, 'width': 0.7, 'length': 10.0, 'depth': 1.0, 'W': 0.7, 'L': 10.0, 'D': 1.0},
    7:  {'shape': 'Triangular',  'shape_name': 'Triangular',  'shape_id': 4, 'width': 0.7, 'length': 10.0, 'depth': 2.0, 'W': 0.7, 'L': 10.0, 'D': 2.0},
    8:  {'shape': 'Triangular',  'shape_name': 'Triangular',  'shape_id': 4, 'width': 0.7, 'length': 10.0, 'depth': 3.0, 'W': 0.7, 'L': 10.0, 'D': 3.0},
    9:  {'shape': 'Step_R',      'shape_name': 'Step_R',      'shape_id': 2, 'width': 0.5, 'length': 10.0, 'depth': 3.0, 'W': 0.5, 'L': 10.0, 'D': 3.0},
    10: {'shape': 'Step_T',      'shape_name': 'Step_T',      'shape_id': 3, 'width': 0.5, 'length': 10.0, 'depth': 3.0, 'W': 0.5, 'L': 10.0, 'D': 3.0},
}


def find_experiment_1_dir():
    """Tự động tìm kiếm thư mục Experiment_1 trong các đường dẫn khả dụng."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.environ.get("EXP1_DIR", ""),
        os.path.abspath(os.path.join(script_dir, "Experiment_1")),
        os.path.abspath(os.path.join(os.getcwd(), "Experiment_1")),
        os.path.abspath(os.path.join(script_dir, "..", "Experiment_1")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "Experiment_1")),
        r"c:\Users\Admin\Documents\paper\PINN-MFL-Crack-Quantification\Experiment_1",
        r"c:\Users\Admin\Documents\paper\new\PINN-MFL-Crack-Quantification\Experiment_1",
        r"c:\Users\Admin\Documents\paper\new\new\Experiment_1",
        r"c:\Users\Admin\Documents\paper\Experiment_1",
        "/work/dat.lt19010205/Cong_Duc/Experiment_1",
    ]
    for cand in candidate_dirs:
        if cand and os.path.isdir(cand):
            return cand
    return None


def extract_crack_no_from_filename(filename):
    """
    Trích xuất số hiệu vết nứt (1 -> 10) từ tên file CSV.
    Ví dụ: '5khz_No1_1.csv' -> 1, '10khz_No10.csv' -> 10.
    """
    match = re.search(r'No(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def preprocess_single_real_image(matrix_2d, scale_factor=1.0):
    """
    Tiền xử lý 1 ma trận ảnh thực tế 2D:
    - Ép kích thước về 32x32 (padding nếu 31x31).
    - Nhân hệ số co dãn biên độ (scale_factor nếu có).
    - Tính kênh Gradient không gian (∇H).
    - Trả về mảng 2 kênh (32, 32, 2) [Field value, Gradient magnitude].
    """
    arr = np.array(matrix_2d, dtype=np.float32)
    
    # 1. Đảm bảo kích thước 32x32 qua nội suy song tuyến (Bilinear Interpolation) bảo toàn tâm và đối xứng
    if arr.shape != (32, 32):
        zoom_factors = (32.0 / arr.shape[0], 32.0 / arr.shape[1])
        res = scipy.ndimage.zoom(arr, zoom_factors, order=1)
        if res.shape != (32, 32):
            out = np.zeros((32, 32), dtype=np.float32)
            h, w = min(32, res.shape[0]), min(32, res.shape[1])
            out[:h, :w] = res[:h, :w]
            arr = out
        else:
            arr = res.astype(np.float32)
        
    # 2. Nhân hệ số co dãn biên độ (nếu cần)
    field_data = arr * scale_factor
    
    # 3. Tính độ lớn Gradient không gian
    grad_y = np.gradient(field_data, axis=0)
    grad_x = np.gradient(field_data, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. Ghép 2 kênh (32, 32, 2)
    two_channel_image = np.dstack([field_data, grad_mag])
    return two_channel_image


def load_real_experiment_for_inference(target_path, labels_csv_path=None, x_scaler=None, scale_factor=1.0, device='cpu', freq_filter=None):
    """
    Hàm nạp dữ liệu thực nghiệm chuyên dùng cho Inference.

    Tham số:
        target_path (str): Đường dẫn đến 1 file .csv hoặc 1 thư mục chứa các file .csv.
        labels_csv_path (str, optional): Đường dẫn đến file labels.csv.
        x_scaler (StandardScaler, optional): Bộ scaler chuẩn hóa X_scaler từ tập huấn luyện.
        scale_factor (float, optional): Hệ số co dãn biên độ (mặc định = 1.0).
        device (str or torch.device): Thiết bị chạy PyTorch ('cpu' hoặc 'cuda').
        freq_filter (str, optional): Lọc theo tần số ('5khz', '10khz', '20khz').

    Trả về:
        X_tensor (torch.FloatTensor): Tensor đầu vào cho model với shape (N, 2, 32, 32).
        metadata_list (list of dict): Thông tin chi tiết của từng mẫu (tên file, nhãn thật, kích thước thật).
    """
    if os.path.isfile(target_path):
        csv_files = [os.path.abspath(target_path)]
        data_dir = os.path.dirname(target_path)
    elif os.path.isdir(target_path):
        data_dir = target_path
        csv_files = sorted([
            os.path.join(target_path, f)
            for f in os.listdir(target_path)
            if f.endswith('.csv') and f != 'labels.csv'
        ])
    else:
        raise FileNotFoundError(f"Không tìm thấy đường dẫn: {target_path}")

    if freq_filter:
        freq_lower = freq_filter.lower()
        csv_files = [f for f in csv_files if freq_lower in os.path.basename(f).lower()]

    if len(csv_files) == 0:
        raise ValueError(f"Không có file .csv dữ liệu nào trong: {target_path} (freq_filter='{freq_filter}')")

    # Đọc nhãn từ file labels.csv nếu có
    labels_dict = {}
    if labels_csv_path is None:
        auto_lbl = os.path.join(data_dir, 'labels.csv')
        if os.path.exists(auto_lbl):
            labels_csv_path = auto_lbl
            
    if labels_csv_path and os.path.exists(labels_csv_path):
        ldf = pd.read_csv(labels_csv_path)
        for _, row in ldf.iterrows():
            fname = str(row['filename']).strip()
            labels_dict[fname] = {
                'shape': str(row.get('shape', 'Unknown')),
                'shape_id': int(row.get('shape_id', -1)) if 'shape_id' in row and not pd.isna(row['shape_id']) else -1,
                'width': float(row.get('width', np.nan)),
                'length': float(row.get('length', np.nan)),
                'depth': float(row.get('depth', np.nan)),
            }

    processed_images = []
    metadata_list = []

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            raw_matrix = pd.read_csv(fpath, header=None).values
            img_2ch = preprocess_single_real_image(raw_matrix, scale_factor=scale_factor)
            processed_images.append(img_2ch)
            
            # Lấy thông tin nhãn thật: ưu tiên file labels.csv, sau đó fallback sang TABLE_2
            crack_no = extract_crack_no_from_filename(fname)
            if fname in labels_dict:
                gt_info = labels_dict[fname]
            elif crack_no in TABLE_2_GROUND_TRUTH:
                gt_info = TABLE_2_GROUND_TRUTH[crack_no]
            else:
                gt_info = {'shape': 'Unknown', 'shape_id': -1, 'width': np.nan, 'length': np.nan, 'depth': np.nan}
            
            metadata_list.append({
                'filepath': fpath,
                'filename': fname,
                'crack_no': crack_no,
                'true_shape': gt_info.get('shape', gt_info.get('shape_name', 'Unknown')),
                'true_shape_id': gt_info.get('shape_id', -1),
                'true_w': gt_info.get('width', gt_info.get('W', np.nan)),
                'true_l': gt_info.get('length', gt_info.get('L', np.nan)),
                'true_d': gt_info.get('depth', gt_info.get('D', np.nan)),
                'raw_min': float(raw_matrix.min()),
                'raw_max': float(raw_matrix.max()),
                'raw_p2p': float(raw_matrix.max() - raw_matrix.min()),
            })
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi đọc file {fname}: {e}")
            continue

    processed_np = np.array(processed_images, dtype=np.float32) # (N, 32, 32, 2)
    
    # Chuẩn hóa nếu có x_scaler
    if x_scaler is not None:
        n_samples = processed_np.shape[0]
        reshaped = processed_np.reshape(-1, 2)
        scaled = x_scaler.transform(reshaped)
        processed_np = scaled.reshape(n_samples, 32, 32, 2)

    # Chuyển đổi sang định dạng Tensor PyTorch (N, C, H, W) = (N, 2, 32, 32)
    X_tensor = torch.tensor(processed_np, dtype=torch.float32).permute(0, 3, 1, 2).contiguous().to(device)

    print(f"[OK] Đã nạp {len(metadata_list)} mẫu thực nghiệm từ: {os.path.basename(data_dir)}")
    print(f"     Tensor đầu vào: {X_tensor.shape} (N={X_tensor.shape[0]}, C=2, H=32, W=32) | Device: {device}")

    return X_tensor, metadata_list


def denormalize_regression_predictions(pred_wld_norm, y_scaler):
    """
    Giải chuẩn hóa đầu ra hồi quy (W, L, D) hỗ trợ đa dạng cấu trúc scaler:
    - SeparateMaxScaler (object với .inverse_transform)
    - Dict of scalers: {'w': w_scaler, 'l': l_scaler, 'd': d_scaler}
    - Tuple/List of 3 scalers: (w_scaler, l_scaler, d_scaler)
    - StandardScaler / MinMaxScaler của sklearn
    - Mảng numpy / hệ số max nhân trực tiếp
    """
    if isinstance(pred_wld_norm, torch.Tensor):
        raw = pred_wld_norm.detach().cpu().numpy()
    else:
        raw = np.array(pred_wld_norm)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if y_scaler is None:
        return raw

    # 1. Scaler có phương thức inverse_transform
    if hasattr(y_scaler, 'inverse_transform'):
        try:
            return y_scaler.inverse_transform(raw)
        except Exception:
            pass

    # 2. Tuple/List gồm 3 scaler riêng biệt
    if isinstance(y_scaler, (list, tuple)) and len(y_scaler) == 3:
        cols = []
        for i, sc in enumerate(y_scaler):
            col_data = raw[:, i:i+1]
            if hasattr(sc, 'inverse_transform'):
                cols.append(sc.inverse_transform(col_data))
            elif isinstance(sc, (int, float, np.number)):
                cols.append(col_data * sc)
            else:
                cols.append(col_data)
        return np.hstack(cols)

    # 3. Dict gồm các key 'w', 'l', 'd'
    if isinstance(y_scaler, dict) and all(k in y_scaler for k in ('w', 'l', 'd')):
        cols = []
        for i, k in enumerate(('w', 'l', 'd')):
            sc = y_scaler[k]
            col_data = raw[:, i:i+1]
            if hasattr(sc, 'inverse_transform'):
                cols.append(sc.inverse_transform(col_data))
            elif isinstance(sc, (int, float, np.number)):
                cols.append(col_data * sc)
            else:
                cols.append(col_data)
        return np.hstack(cols)

    # 4. Mảng số hoặc scalar
    if isinstance(y_scaler, (int, float, np.number, np.ndarray)):
        return raw * y_scaler

    return raw


# =============================================================================
# CHỨC NĂNG INFERENCE VÀ ĐÁNH GIÁ TỰ ĐỘNG TẬP THỰC NGHIỆM (SIM-TO-REAL)
# =============================================================================
def evaluate_real_experiment(model, x_scaler=None, y_scaler=None, unique_shapes=None, output_dir="./", device='cpu'):
    """
    Tự động suy luận trên toàn bộ dữ liệu thực nghiệm (Experiment_1):
    - 5 kHz (Trainning)
    - 10 kHz (Testing)
    - 20 kHz (Testing)
    Lưu kết quả chi tiết từng file và bảng tổng hợp vào output_dir/real_experiment_results/
    """
    if unique_shapes is None:
        unique_shapes = DEFAULT_UNIQUE_SHAPES

    exp_base = find_experiment_1_dir()
    if exp_base is None:
        raise FileNotFoundError("Không tìm thấy thư mục Experiment_1 trong các đường dẫn khả dụng.")

    train_dir = os.path.join(exp_base, "Trainning")
    test_dir = os.path.join(exp_base, "Testing")
    
    out_exp_dir = os.path.join(output_dir, "real_experiment_results")
    os.makedirs(out_exp_dir, exist_ok=True)
    
    splits = [
        ('5kHz_Trainning', train_dir, '5khz'),
        ('10kHz_Testing', test_dir, '10khz'),
        ('20kHz_Testing', test_dir, '20khz')
    ]
    
    model.eval()
    summary_rows = []
    
    for split_label, s_dir, freq_filter in splits:
        if not os.path.exists(s_dir):
            continue
            
        csv_files = [
            f for f in os.listdir(s_dir)
            if f.endswith('.csv') and f != 'labels.csv' and freq_filter in f.lower()
        ]
        if not csv_files:
            continue
            
        # Nạp dữ liệu thực qua load_real_experiment_for_inference
        X_real_tensor, meta = load_real_experiment_for_inference(
            s_dir, x_scaler=x_scaler, device=device, freq_filter=freq_filter
        )
        if len(meta) == 0:
            continue
        
        with torch.no_grad():
            try:
                outputs = model(X_real_tensor)
            except Exception as e_direct:
                try:
                    # Fallback cho kiến trúc 1D MLP (N, 2048)
                    X_real_1d = X_real_tensor.reshape(X_real_tensor.size(0), -1)
                    outputs = model(X_real_1d)
                except Exception:
                    raise e_direct

            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 2:
                    pred_logits, pred_wld_norm = outputs
                    has_clf = True
                    has_reg = True
                else:
                    pred_logits = outputs[0]
                    has_clf = True
                    has_reg = False
            else:
                if hasattr(outputs, 'shape') and outputs.shape[1] == len(unique_shapes):
                    pred_logits = outputs
                    has_clf = True
                    has_reg = False
                else:
                    pred_wld_norm = outputs
                    has_clf = False
                    has_reg = True
                    
            if has_clf:
                pred_shape_ids = torch.argmax(pred_logits, dim=1).cpu().numpy()
                pred_shapes = [unique_shapes[sid] if (0 <= sid < len(unique_shapes)) else 'Unknown' for sid in pred_shape_ids]
            else:
                pred_shapes = ['N/A'] * len(meta)
                
            if has_reg:
                pred_wld_denorm = denormalize_regression_predictions(pred_wld_norm, y_scaler)
            else:
                pred_wld_denorm = np.full((len(meta), 3), np.nan)
                
        pred_records = []
        for i, m in enumerate(meta):
            shape_match = (m['true_shape'] == pred_shapes[i]) if has_clf else np.nan
            err_w = abs(m['true_w'] - pred_wld_denorm[i, 0]) if has_reg and not np.isnan(m['true_w']) else np.nan
            err_l = abs(m['true_l'] - pred_wld_denorm[i, 1]) if has_reg and not np.isnan(m['true_l']) else np.nan
            err_d = abs(m['true_d'] - pred_wld_denorm[i, 2]) if has_reg and not np.isnan(m['true_d']) else np.nan
            
            pred_records.append({
                'Filename': m['filename'],
                'Frequency': freq_filter,
                'True_Shape': m['true_shape'],
                'Pred_Shape': pred_shapes[i],
                'Shape_Match': shape_match,
                'True_W': m['true_w'], 'Pred_W': round(float(pred_wld_denorm[i, 0]), 4) if has_reg else np.nan, 'Err_W': round(float(err_w), 4) if not np.isnan(err_w) else np.nan,
                'True_L': m['true_l'], 'Pred_L': round(float(pred_wld_denorm[i, 1]), 4) if has_reg else np.nan, 'Err_L': round(float(err_l), 4) if not np.isnan(err_l) else np.nan,
                'True_D': m['true_d'], 'Pred_D': round(float(pred_wld_denorm[i, 2]), 4) if has_reg else np.nan, 'Err_D': round(float(err_d), 4) if not np.isnan(err_d) else np.nan,
            })
            
        df_preds = pd.DataFrame(pred_records)
        pred_csv = os.path.join(out_exp_dir, f"real_{split_label}_predictions.csv")
        df_preds.to_csv(pred_csv, index=False)
        
        # Tính toán các chỉ số tóm tắt
        acc = df_preds['Shape_Match'].mean() * 100.0 if has_clf else np.nan
        mae_w = df_preds['Err_W'].mean() if has_reg else np.nan
        mae_l = df_preds['Err_L'].mean() if has_reg else np.nan
        mae_d = df_preds['Err_D'].mean() if has_reg else np.nan
        mae_avg = (mae_w + mae_l + mae_d) / 3.0 if has_reg else np.nan
        
        summary_rows.append({
            'Split': split_label,
            'Frequency': freq_filter,
            'Num_Samples': len(df_preds),
            'Clf_Accuracy (%)': round(acc, 2) if not np.isnan(acc) else 'N/A',
            'MAE_W (mm)': round(mae_w, 4) if not np.isnan(mae_w) else 'N/A',
            'MAE_L (mm)': round(mae_l, 4) if not np.isnan(mae_l) else 'N/A',
            'MAE_D (mm)': round(mae_d, 4) if not np.isnan(mae_d) else 'N/A',
            'Overall_MAE (mm)': round(mae_avg, 4) if not np.isnan(mae_avg) else 'N/A',
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_exp_dir, "real_experiment_summary_metrics.csv")
    df_summary.to_csv(summary_csv, index=False)
    
    print("\n" + "="*80)
    print("REAL EXPERIMENT (EXPERIMENT_1) SIM-TO-REAL EVALUATION SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print(f"\n[OK] Real experiment results saved to: {out_exp_dir}")
    print("="*80 + "\n")
    return df_summary


# =============================================================================
# KIỂM THỬ ĐỘC LẬP
# =============================================================================
if __name__ == "__main__":
    exp_dir = find_experiment_1_dir()
    if exp_dir is None:
        print("[LỖI] Không tìm thấy thư mục Experiment_1.")
    else:
        print(f"[THỬ NGHIỆM] Nạp dữ liệu từ: {exp_dir}")
        for sub in ["Trainning", "Testing"]:
            sub_path = os.path.join(exp_dir, sub)
            if os.path.exists(sub_path):
                X, meta = load_real_experiment_for_inference(sub_path)
                print(f"--- Mẫu đại diện từ {sub} ---")
                for i in range(min(2, len(meta))):
                    m = meta[i]
                    print(f"  [{m['filename']}] Crack No.{m['crack_no']} | Dạng: {m['true_shape']} (ID={m['true_shape_id']}) | Kích thước: W={m['true_w']}mm, L={m['true_l']}mm, D={m['true_d']}mm | Đỉnh-Đỉnh: {m['raw_p2p']:.4f}V")
