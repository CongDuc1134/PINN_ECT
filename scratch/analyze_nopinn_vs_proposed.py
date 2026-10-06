import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.model_loader import load_5khz_fully_prepared, predict_and_denormalize
from domain_adaptation.data_loader import split_5khz_scan1_scan2

b_no = load_5khz_fully_prepared(model_type="cnn", variant="nopinn")
b_pi = load_5khz_fully_prepared(model_type="cnn", variant="pinn")

m_no = b_no["model"]
m_pi = b_pi["model"]

print("=== PRETRAINED MODEL LOG_VARS & KENDALL MULTIPLIERS ===")
print("CNN_NoPINN:")
print(f"  log_var_clf: {m_no.log_var_clf.item():.4f} -> multiplier exp(-s): {torch.exp(-m_no.log_var_clf).item():.4f}")
print(f"  log_var_w:   {m_no.log_var_w.item():.4f} -> multiplier exp(-s): {torch.exp(-m_no.log_var_w).item():.4f}")
print(f"  log_var_l:   {m_no.log_var_l.item():.4f} -> multiplier exp(-s): {torch.exp(-m_no.log_var_l).item():.4f}")
print(f"  log_var_d:   {m_no.log_var_d.item():.4f} -> multiplier exp(-s): {torch.exp(-m_no.log_var_d).item():.4f}")

print("\nCNN_Proposed (PINN):")
print(f"  log_var_clf: {m_pi.log_var_clf.item():.4f} -> multiplier exp(-s): {torch.exp(-m_pi.log_var_clf).item():.4f}")
print(f"  log_var_w:   {m_pi.log_var_w.item():.4f} -> multiplier exp(-s): {torch.exp(-m_pi.log_var_w).item():.4f}")
print(f"  log_var_l:   {m_pi.log_var_l.item():.4f} -> multiplier exp(-s): {torch.exp(-m_pi.log_var_l).item():.4f}")
print(f"  log_var_d:   {m_pi.log_var_d.item():.4f} -> multiplier exp(-s): {torch.exp(-m_pi.log_var_d).item():.4f}")

# Check scalers
print("\n=== SCALERS ===")
print("NoPINN y_scaler:", b_no["y_scaler"])
print("Proposed y_scaler:", b_pi["y_scaler"])

# Check predictions on Scan 1 (Train) and Scan 2 (Test)
train_set_no, test_set_no = split_5khz_scan1_scan2(b_no)
train_set_pi, test_set_pi = split_5khz_scan1_scan2(b_pi)

p_sh_no_tr, p_wld_no_tr, _, _ = predict_and_denormalize(m_no, train_set_no["X"], y_scaler=b_no["y_scaler"])
p_sh_no_te, p_wld_no_te, _, _ = predict_and_denormalize(m_no, test_set_no["X"], y_scaler=b_no["y_scaler"])

p_sh_pi_tr, p_wld_pi_tr, _, _ = predict_and_denormalize(m_pi, train_set_pi["X"], y_scaler=b_pi["y_scaler"])
p_sh_pi_te, p_wld_pi_te, _, _ = predict_and_denormalize(m_pi, test_set_pi["X"], y_scaler=b_pi["y_scaler"])

true_sh_tr = [m["true_shape"] for m in train_set_no["metadata"]]
true_sh_te = [m["true_shape"] for m in test_set_no["metadata"]]

print("\n=== ZERO-SHOT SHAPE MATCHES ===")
print("Scan 1 (Train set):")
print("  NoPINN:", sum(1 for t, p in zip(true_sh_tr, p_sh_no_tr) if t == p), "/ 10")
print("  Proposed:", sum(1 for t, p in zip(true_sh_tr, p_sh_pi_tr) if t == p), "/ 10")

print("Scan 2 (Test set):")
print("  NoPINN:", sum(1 for t, p in zip(true_sh_te, p_sh_no_te) if t == p), "/ 10")
print("  Proposed:", sum(1 for t, p in zip(true_sh_te, p_sh_pi_te) if t == p), "/ 10")

# Inspect finetuned checkpoint if exists
ckpt_raw_no = "domain_adaptation/finetune_results_raw/protocols/checkpoints/finetuned_cnn_nopinn.pth"
ckpt_raw_pi = "domain_adaptation/finetune_results_raw/protocols/checkpoints/finetuned_cnn_proposed.pth"
ckpt_new_pi = "domain_adaptation/finetune_results/protocols/checkpoints/finetuned_cnn_proposed.pth"

for name, path in [("Raw NoPINN", ckpt_raw_no), ("Raw Proposed", ckpt_raw_pi), ("New Proposed", ckpt_new_pi)]:
    if os.path.exists(path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        sd = c.get("model_state_dict", c)
        print(f"\nCheckpoint {name}:")
        for k in ["log_var_clf", "log_var_w", "log_var_l", "log_var_d"]:
            if k in sd:
                val = sd[k].item()
                print(f"  {k}: {val:.4f} (exp(-s)={np.exp(-val):.4f})")
