# -*- coding: utf-8 -*-
"""
===============================================================================
DEMO: LOAD REAL EXPERIMENT DATA VÀO CÁC MÔ HÌNH TRONG DOMAIN ADAPTATION
===============================================================================
Minh họa thực tế cách tải dữ liệu từ Experiment_1 (thông qua load_real_experiment_data.py
và domain_adaptation.data_loader) để truyền vào:
1. CNN Multitask (Đầu vào không gian 2D: (B, 2, 32, 32))
2. Multitask MLP (Đầu vào vector 1D: (B, 2048))
3. Xiong et al. 2023 Single-Task (Đầu vào vector 1D: (B, 2048))
Đồng thời minh họa cách dùng DataLoader cho Unsupervised Domain Adaptation (UDA).
"""

import os
import sys
import torch

# Đảm bảo import được các module từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.cnn import ImprovedMultimodelNet, CNNTotalLoss
from domain_adaptation.mlp import MultitaskMLP_PINN, MultitaskMLPTotalLoss
from domain_adaptation.xiong import RegressionMLP_PINN, XiongTotalLoss

from domain_adaptation.data_loader import (
    load_real_data_for_model,
    get_real_dataloader,
    RealExperimentDataset,
)
from load_real_experiment_data import (
    load_real_experiment_for_inference,
    evaluate_real_experiment,
    DEFAULT_UNIQUE_SHAPES,
)


def demo_load_into_cnn():
    print("\n" + "=" * 75)
    print("1. LOAD DỮ LIỆU THỰC NGHIỆM VÀO MÔ HÌNH CNN (2D SPATIAL TENSOR)")
    print("=" * 75)
    
    # Cách A: Dùng load_real_data_for_model một dòng tiện lợi
    X_cnn, y_clf, y_wld, meta = load_real_data_for_model(model_type='cnn', split='5khz')
    print(f"[*] Đã nạp {len(meta)} mẫu thực nghiệm 5kHz cho CNN:")
    print(f"    - X shape:     {X_cnn.shape} (N, C=2, H=32, W=32)")
    print(f"    - y_clf shape: {y_clf.shape} (Nhãn hình dạng 0..4)")
    print(f"    - y_wld shape: {y_wld.shape} (Kích thước [W, L, D] thật)")

    # Khởi tạo mô hình CNN
    model = ImprovedMultimodelNet(num_shapes=5)
    model.eval()

    # Forward pass trực tiếp
    with torch.no_grad():
        logits, pred_wld = model(X_cnn)
    
    print(f"[+] Output Logits shape: {logits.shape}")
    print(f"[+] Output WLD shape:    {pred_wld.shape}")
    
    # Dự đoán hình dạng và kích thước mẫu đầu tiên
    pred_class_id = torch.argmax(logits[0]).item()
    pred_shape = DEFAULT_UNIQUE_SHAPES[pred_class_id]
    print(f"    -> Mẫu 0 [{meta[0]['filename']}]: Thật = {meta[0]['true_shape']} (W={meta[0]['true_w']}, L={meta[0]['true_l']}, D={meta[0]['true_d']})")
    print(f"       Dự đoán CNN: Shape = {pred_shape} | [W, L, D] = {pred_wld[0].numpy().round(3)}")


def demo_load_into_mlp():
    print("\n" + "=" * 75)
    print("2. LOAD DỮ LIỆU THỰC NGHIỆM VÀO MÔ HÌNH MULTITASK MLP (1D VECTOR)")
    print("=" * 75)
    
    # Nạp dữ liệu tự động làm phẳng thành vector 2048 chiều
    X_mlp, y_clf, y_wld, meta = load_real_data_for_model(model_type='mlp', split='10khz')
    print(f"[*] Đã nạp {len(meta)} mẫu thực nghiệm 10kHz cho MLP:")
    print(f"    - X shape:     {X_mlp.shape} (N, 2048)")
    print(f"    - y_clf shape: {y_clf.shape}")
    print(f"    - y_wld shape: {y_wld.shape}")

    model = MultitaskMLP_PINN(num_shapes=5)
    model.eval()

    with torch.no_grad():
        logits, pred_wld = model(X_mlp)

    print(f"[+] Output Logits shape: {logits.shape}")
    print(f"[+] Output WLD shape:    {pred_wld.shape} (Softplus kích hoạt > 0)")
    
    pred_class_id = torch.argmax(logits[0]).item()
    print(f"    -> Mẫu 0 [{meta[0]['filename']}]: Thật = {meta[0]['true_shape']}")
    print(f"       Dự đoán MLP: Shape = {DEFAULT_UNIQUE_SHAPES[pred_class_id]} | [W, L, D] = {pred_wld[0].numpy().round(3)}")


def demo_load_into_xiong():
    print("\n" + "=" * 75)
    print("3. LOAD DỮ LIỆU THỰC NGHIỆM VÀO MÔ HÌNH XIONG ET AL. 2023 (1D VECTOR, NO CLF)")
    print("=" * 75)
    
    X_xiong, _, y_wld, meta = load_real_data_for_model(model_type='xiong', split='20khz')
    print(f"[*] Đã nạp {len(meta)} mẫu thực nghiệm 20kHz cho Xiong et al.:")
    print(f"    - X shape:     {X_xiong.shape} (N, 2048)")
    print(f"    - y_wld shape: {y_wld.shape}")

    model = RegressionMLP_PINN()
    model.eval()

    with torch.no_grad():
        pred_wld = model(X_xiong)

    print(f"[+] Output WLD shape:    {pred_wld.shape} (Duy nhất 1 Tensor [W, L, D], KHÔNG có Logits!)")
    print(f"    -> Mẫu 0 [{meta[0]['filename']}]: Thật [W, L, D] = [{meta[0]['true_w']}, {meta[0]['true_l']}, {meta[0]['true_d']}]")
    print(f"       Dự đoán Xiong: [W, L, D] = {pred_wld[0].numpy().round(3)}")


def demo_dataloader_for_domain_adaptation():
    print("\n" + "=" * 75)
    print("4. DEMO TẠO PYTORCH DATALOADER CHO TARGET DOMAIN (UDA TRAINING)")
    print("=" * 75)
    
    # Tạo Target Domain DataLoader dạng 2D (cho CNN) hoặc 1D (cho MLP)
    target_loader_cnn = get_real_dataloader(split='5khz', model_type='cnn', batch_size=4, shuffle=True)
    print(f"[*] Số batch Target Domain (CNN 2D, batch_size=4): {len(target_loader_cnn)}")
    
    for batch_idx, (batch_x, batch_yclf, batch_ywld, batch_meta) in enumerate(target_loader_cnn):
        print(f"    - Batch {batch_idx}: X = {batch_x.shape}, y_clf = {batch_yclf.shape}, y_wld = {batch_ywld.shape}")
        # Trong UDA (Unsupervised Domain Adaptation):
        # Target data chỉ dùng batch_x để tối ưu hóa khoảng cách phân phối (MMD, Coral, DANN discriminator)
        break


if __name__ == "__main__":
    demo_load_into_cnn()
    demo_load_into_mlp()
    demo_load_into_xiong()
    demo_dataloader_for_domain_adaptation()
    print("\n" + "=" * 75)
    print("[HOÀN TẤT] NẠP DỮ LIỆU THỰC NGHIỆM VÀO CẢ 3 MÔ HÌNH THÀNH CÔNG RỰC RỠ!")
    print("=" * 75)
