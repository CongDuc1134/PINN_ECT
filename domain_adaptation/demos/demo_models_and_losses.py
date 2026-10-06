"""
===============================================================================
DEMO: HUONG DAN GOI MO HINH THUAN TUY, HAM LOSS VA THONG SO TRONG DOMAIN ADAPTATION
===============================================================================
Script nay minh hoa truc quan va ro rang cach goi tung mo hinh, truyen du lieu dau vao,
tinh ham loss dac thu va truy xuat cac tham so (parameters) cho ca 3 phan he:
1. CNN (2D Input, Dual Head, 4 Kendall Uncertainties)
2. MLP (1D Input, Dual Head, 2 Kendall Uncertainties)
3. Xiong et al. (1D Input, Single-Task 3D Regression, Strictly NO Classifier)
"""

import torch
import sys
import os

# Them thu muc goc vao sys.path de import truc tiep
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 1. Import tu folder cnn/
from domain_adaptation.cnn import (
    ImprovedMultimodelNet as CNN_Multitask,
    CNN_SingleTask_Classification,
    CNN_SingleTask_Regression,
    CNNTotalLoss,
    CNNModelConfig,
    CNNTrainingConfig,
)

# 2. Import tu folder mlp/
from domain_adaptation.mlp import (
    MultitaskMLP_PINN as MLP_Multitask,
    MultitaskMLPTotalLoss,
    MLPModelConfig,
    MLPTrainingConfig,
)

# 3. Import tu folder xiong/
from domain_adaptation.xiong import (
    RegressionMLP_PINN as Xiong_Regression,
    XiongTotalLoss,
    XiongModelConfig,
    XiongTrainingConfig,
)


def demo_cnn():
    print("\n" + "=" * 75)
    print("1. DEMO MO HINH CNN (2D SPATIAL TENSOR & 4-VARIABLE KENDALL LOSS)")
    print("=" * 75)
    
    # A. Khoi tao config & mo hinh
    model_cfg = CNNModelConfig()
    train_cfg = CNNTrainingConfig()
    model = CNN_Multitask(num_shapes=model_cfg.num_shapes, config=model_cfg)
    loss_fn = CNNTotalLoss(alpha=train_cfg.alpha_pinn, use_pinn=train_cfg.use_pinn_loss)
    
    print(f"[*] Thong so kien truc: channels={model_cfg.input_channels}, classes={model_cfg.num_shapes}, dropout={model_cfg.dropout_rate}")
    print(f"[*] Thong so huan luyen: lr={train_cfg.learning_rate}, batch_size={train_cfg.batch_size}, alpha_pinn={train_cfg.alpha_pinn}")
    print(f"[*] Tong so tham so hoc duoc: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[*] Cac tham so bat dinh (Kendall): log_var_clf, log_var_w, log_var_l, log_var_d")

    # B. Du lieu gia lap (Batch 4 mau, 2 kenh vi sai & gradient, anh 32x32)
    x = torch.randn(4, 2, 32, 32)
    y_true_clf = torch.tensor([0, 1, 4, 2], dtype=torch.long)
    y_true_wld = torch.tensor([[0.7, 10.0, 1.0], [0.7, 10.0, 2.0], [0.9, 10.0, 3.0], [0.5, 10.0, 3.0]], dtype=torch.float32)

    # C. Forward Pass
    logits, pred_wld = model(x)
    print(f"\n[+] Input shape:         {x.shape}")
    print(f"[+] Output Logits shape: {logits.shape} (Phan loai 5 dang hinh)")
    print(f"[+] Output WLD shape:    {pred_wld.shape} (Hoi quy [W, L, D])")

    # D. Tinh Loss & Backward
    total_loss, metrics = loss_fn(
        shape_logits=logits,
        shape_targets=y_true_clf,
        pred_wld=pred_wld,
        true_wld=y_true_wld,
        log_var_clf=model.log_var_clf,
        log_var_w=model.log_var_w,
        log_var_l=model.log_var_l,
        log_var_d=model.log_var_d,
    )
    total_loss.backward()

    print(f"\n[+] Total Loss:          {total_loss.item():.4f}")
    print(f"[+] Chi tiet cac thanh phan Loss:")
    for k, v in metrics.items():
        print(f"    - {k:<12}: {v:.4f}")


def demo_mlp():
    print("\n" + "=" * 75)
    print("2. DEMO MO HINH MULTITASK MLP (1D VECTOR & 2-VARIABLE KENDALL LOSS)")
    print("=" * 75)
    
    # A. Khoi tao config & mo hinh
    model_cfg = MLPModelConfig()
    train_cfg = MLPTrainingConfig()
    model = MLP_Multitask(num_shapes=model_cfg.num_shapes, config=model_cfg)
    loss_fn = MultitaskMLPTotalLoss(alpha=train_cfg.alpha_pinn, use_pinn=train_cfg.use_pinn_loss)
    
    print(f"[*] Thong so kien truc: input_dim={model_cfg.input_dim}, hidden_dims={model_cfg.hidden_dims}")
    print(f"[*] Thong so huan luyen: lr={train_cfg.learning_rate}, batch_size={train_cfg.batch_size}, alpha_pinn={train_cfg.alpha_pinn}")
    print(f"[*] Tong so tham so hoc duoc: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[*] Cac tham so bat dinh (Kendall): log_var_clf, log_var_reg")

    # B. Du lieu gia lap (Batch 4 mau, vector phang 2048 = 2*32*32)
    x = torch.randn(4, 2048)
    y_true_clf = torch.tensor([0, 1, 4, 2], dtype=torch.long)
    y_true_wld = torch.tensor([[0.7, 10.0, 1.0], [0.7, 10.0, 2.0], [0.9, 10.0, 3.0], [0.5, 10.0, 3.0]], dtype=torch.float32)

    # C. Forward Pass
    logits, pred_wld = model(x)
    print(f"\n[+] Input shape:         {x.shape}")
    print(f"[+] Output Logits shape: {logits.shape} (Phan loai 5 dang hinh)")
    print(f"[+] Output WLD shape:    {pred_wld.shape} (Hoi quy [W, L, D])")

    # D. Tinh Loss & Backward
    total_loss, metrics = loss_fn(
        shape_logits=logits,
        shape_targets=y_true_clf,
        pred_wld=pred_wld,
        true_wld=y_true_wld,
        log_var_clf=model.log_var_clf,
        log_var_reg=model.log_var_reg,
    )
    total_loss.backward()

    print(f"\n[+] Total Loss:          {total_loss.item():.4f}")
    print(f"[+] Chi tiet cac thanh phan Loss:")
    for k, v in metrics.items():
        print(f"    - {k:<12}: {v:.4f}")


def demo_xiong():
    print("\n" + "=" * 75)
    print("3. DEMO MO HINH XIONG ET AL. (SINGLE-TASK 3D REGRESSION & PURE MSE LOSS)")
    print("=" * 75)
    
    # A. Khoi tao config & mo hinh
    model_cfg = XiongModelConfig()
    train_cfg = XiongTrainingConfig()
    model = Xiong_Regression(config=model_cfg)
    loss_fn = XiongTotalLoss(alpha=train_cfg.alpha_pinn, use_pinn=train_cfg.use_pinn_loss)
    
    print(f"[*] Thong so kien truc: input_dim={model_cfg.input_dim}, hidden_dims={model_cfg.hidden_dims} (Tanh + Softplus)")
    print(f"[*] Thong so huan luyen: lr={train_cfg.learning_rate}, batch_size={train_cfg.batch_size}, alpha_pinn={train_cfg.alpha_pinn}")
    print(f"[*] Tong so tham so hoc duoc: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[*] LUU Y: Mo hinh don nhiem, HOAN TOAN KHONG co dau phan loai logits va khong dung CE Loss!")

    # B. Du lieu gia lap (Batch 4 mau, vector phang 2048)
    x = torch.randn(4, 2048)
    y_true_wld = torch.tensor([[0.7, 10.0, 1.0], [0.7, 10.0, 2.0], [0.9, 10.0, 3.0], [0.5, 10.0, 3.0]], dtype=torch.float32)

    # C. Forward Pass (Tra ve duy nhat 1 Tensor hoi quy)
    pred_wld = model(x)
    print(f"\n[+] Input shape:         {x.shape}")
    print(f"[+] Output WLD shape:    {pred_wld.shape} (Duy nhat Tensor [W, L, D], KHONG co Logits!)")

    # D. Tinh Loss & Backward (Chi truyen pred_wld va y_true_wld, khong truyen nhan phan loai)
    total_loss, metrics = loss_fn(
        pred_wld=pred_wld,
        true_wld=y_true_wld,
    )
    total_loss.backward()

    print(f"\n[+] Total Loss:          {total_loss.item():.4f}")
    print(f"[+] Chi tiet cac thanh phan Loss:")
    for k, v in metrics.items():
        print(f"    - {k:<12}: {v:.4f}")


if __name__ == "__main__":
    demo_cnn()
    demo_mlp()
    demo_xiong()
    print("\n" + "=" * 75)
    print("[HOAN TAT] DA GOI THUAN TUY CA 3 MO HINH, HAM LOSS VA THONG SO THANH CONG!")
    print("=" * 75)
