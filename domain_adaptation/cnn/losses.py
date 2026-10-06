"""
Loss functions for CNN Multitask Architectures.
Faithfully extracted from cnn/main_percent_new.py and cnn/main_percent_no_pinn.py.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class CNNRegressionLoss(nn.Module):
    """
    Computes individual regression losses for W, L, D from joint output.
    """
    def __init__(self):
        super(CNNRegressionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, y_pred_wld: torch.Tensor, y_wld: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y_pred_wld: Predicted [W, L, D], shape (B, 3)
            y_wld: Ground truth [W, L, D], shape (B, 3) (normalized to [0, 1])
        Returns:
            w_loss, l_loss, d_loss, avg_reg_loss
        """
        w_loss = self.criterion(y_pred_wld[:, 0], y_wld[:, 0])
        l_loss = self.criterion(y_pred_wld[:, 1], y_wld[:, 1])
        d_loss = self.criterion(y_pred_wld[:, 2], y_wld[:, 2])
        avg_reg_loss = (w_loss + l_loss + d_loss) / 3.0
        return w_loss, l_loss, d_loss, avg_reg_loss


class CNNClassificationLoss(nn.Module):
    """
    Cross Entropy loss for 5-class defect shape classification.
    """
    def __init__(self):
        super(CNNClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class CNNTotalLoss(nn.Module):
    """
    Combines Classification and 3D Regression with:
    1. Kendall et al. (2018) Homoscedastic Uncertainty Weighting (4 learnable variables)
    2. Optional standalone Physics PINN loss weighted by alpha.
    """
    def __init__(self, alpha: float = 1.0, use_pinn: bool = True):
        super(CNNTotalLoss, self).__init__()
        self.alpha = alpha
        self.use_pinn = use_pinn
        self.clf_fn = CNNClassificationLoss()
        self.reg_fn = CNNRegressionLoss()

    def forward(
        self,
        shape_logits: torch.Tensor,
        shape_targets: torch.Tensor,
        pred_wld: torch.Tensor,
        true_wld: torch.Tensor,
        log_var_clf: Optional[torch.Tensor] = None,
        log_var_w: Optional[torch.Tensor] = None,
        log_var_l: Optional[torch.Tensor] = None,
        log_var_d: Optional[torch.Tensor] = None,
        physics_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes the complete combined multi-task loss.
        """
        loss_clf = self.clf_fn(shape_logits, shape_targets)
        w_loss, l_loss, d_loss, avg_reg_loss = self.reg_fn(pred_wld, true_wld)

        # Kendall Uncertainty Weighting across 4 supervised data tasks
        if all(v is not None for v in (log_var_clf, log_var_w, log_var_l, log_var_d)):
            precision_clf = torch.exp(-log_var_clf)
            precision_w = torch.exp(-log_var_w)
            precision_l = torch.exp(-log_var_l)
            precision_d = torch.exp(-log_var_d)
            
            data_loss = (
                precision_clf * loss_clf + 0.5 * log_var_clf +
                precision_w * w_loss + 0.5 * log_var_w +
                precision_l * l_loss + 0.5 * log_var_l +
                precision_d * d_loss + 0.5 * log_var_d
            )
        else:
            data_loss = loss_clf + avg_reg_loss

        # Physics loss regularizer
        if self.use_pinn and physics_loss is not None:
            total_loss = data_loss + self.alpha * physics_loss
        else:
            total_loss = data_loss

        metrics = {
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "loss_clf": loss_clf.item(),
            "w_loss": w_loss.item(),
            "l_loss": l_loss.item(),
            "d_loss": d_loss.item(),
            "avg_reg_loss": avg_reg_loss.item(),
            "physics_loss": physics_loss.item() if physics_loss is not None else 0.0
        }
        return total_loss, metrics


class CNN_SingleTask_ClassificationTotalLoss(nn.Module):
    """
    Loss for Single-Task 2D CNN Shape Classification.
    Extracted from cnn/cnn_single_task_classification_pinn.py.
    """
    def __init__(self, alpha: float = 1.0, use_pinn: bool = True):
        super(CNN_SingleTask_ClassificationTotalLoss, self).__init__()
        self.alpha = alpha
        self.use_pinn = use_pinn
        self.clf_fn = CNNClassificationLoss()

    def forward(
        self,
        shape_logits: torch.Tensor,
        shape_targets: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        data_loss = self.clf_fn(shape_logits, shape_targets)
        if self.use_pinn and physics_loss is not None:
            total_loss = data_loss + self.alpha * physics_loss
        else:
            total_loss = data_loss

        metrics = {
            "total_loss": total_loss.item(),
            "loss_clf": data_loss.item(),
            "physics_loss": physics_loss.item() if physics_loss is not None else 0.0,
        }
        return total_loss, metrics


class CNN_SingleTask_RegressionTotalLoss(nn.Module):
    """
    Loss for Single-Task 2D CNN 3D Regression (W, L, D).
    With 3 Kendall uncertainty parameters (w, l, d).
    Extracted from cnn/cnn_single_task_regression_pinn.py.
    """
    def __init__(self, alpha: float = 1.0, use_pinn: bool = True):
        super(CNN_SingleTask_RegressionTotalLoss, self).__init__()
        self.alpha = alpha
        self.use_pinn = use_pinn
        self.reg_fn = CNNRegressionLoss()

    def forward(
        self,
        pred_wld: torch.Tensor,
        true_wld: torch.Tensor,
        log_var_w: Optional[torch.Tensor] = None,
        log_var_l: Optional[torch.Tensor] = None,
        log_var_d: Optional[torch.Tensor] = None,
        physics_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        w_loss, l_loss, d_loss, avg_reg_loss = self.reg_fn(pred_wld, true_wld)

        if all(v is not None for v in (log_var_w, log_var_l, log_var_d)):
            precision_w = torch.exp(-log_var_w)
            precision_l = torch.exp(-log_var_l)
            precision_d = torch.exp(-log_var_d)
            data_loss = (
                precision_w * w_loss + 0.5 * log_var_w +
                precision_l * l_loss + 0.5 * log_var_l +
                precision_d * d_loss + 0.5 * log_var_d
            )
        else:
            data_loss = avg_reg_loss

        if self.use_pinn and physics_loss is not None:
            total_loss = data_loss + self.alpha * physics_loss
        else:
            total_loss = data_loss

        metrics = {
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "w_loss": w_loss.item(),
            "l_loss": l_loss.item(),
            "d_loss": d_loss.item(),
            "avg_reg_loss": avg_reg_loss.item(),
            "physics_loss": physics_loss.item() if physics_loss is not None else 0.0,
        }
        return total_loss, metrics


# =============================================================================
# PHYSICS FORWARD SOLVER FOR REAL DATA PINN LOSS
# =============================================================================
_SHAPE_PHYSICS_MAP = None

def get_shape_physics_map():
    global _SHAPE_PHYSICS_MAP
    if _SHAPE_PHYSICS_MAP is None:
        import sys, os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cnn_dir = os.path.join(project_root, "cnn")
        if cnn_dir not in sys.path:
            sys.path.insert(0, cnn_dir)
        try:
            import ellipse, regtangular, triangular, step_r, step_t
            _SHAPE_PHYSICS_MAP = {
                "Ellipse": ellipse,
                "Rectangular": regtangular,
                "Triangular": triangular,
                "Step_R": step_r,
                "Step_T": step_t,
            }
        except Exception as e:
            print(f"[WARN] Failed to load shape physics modules: {e}")
            _SHAPE_PHYSICS_MAP = {}
    return _SHAPE_PHYSICS_MAP


def compute_physics_loss_autograd(
    H_true: torch.Tensor,
    w_pred: torch.Tensor,
    l_pred: torch.Tensor,
    d_pred: torch.Tensor,
    shape_name: str,
    shape_map: dict = None,
    y_scaler = None,
    X_scaler = None,
) -> torch.Tensor:
    """
    Differentiable PINN physics loss.
    Faithfully extracted from cnn/main_percent_new.py lines 1335-1493.
    """
    import math
    if shape_map is None:
        shape_map = get_shape_physics_map()

    device = H_true.device if torch.is_tensor(H_true) else "cpu"
    base_zero = (w_pred + l_pred + d_pred) * 0.0
    if shape_name not in shape_map:
        return base_zero

    physics_module = shape_map[shape_name]
    target_device = getattr(physics_module, "device", (H_true.device if torch.is_tensor(H_true) else "cpu"))

    w_norm = w_pred.float().reshape(()).to(target_device)
    l_norm = l_pred.float().reshape(()).to(target_device)
    d_norm = d_pred.float().reshape(()).to(target_device)
    device = target_device

    if y_scaler is not None and hasattr(y_scaler, "data_max_"):
        w_max, l_max, d_max = [float(v) for v in y_scaler.data_max_]
        w_val = w_norm * w_max
        l_val = l_norm * l_max
        d_val = d_norm * d_max
    else:
        w_val, l_val, d_val = w_norm, l_norm, d_norm

    w_val = torch.clamp(w_val, 0.3, 1.5)
    l_val = torch.clamp(l_val, 2.0, 20.0)
    d_val = torch.clamp(d_val, 0.1, 3.0)

    freq, sicma, mu = 5000.0, 35461000.0, 0.0000012566
    csi, K, I, G = 0.085, 1.5, 0.01, 3981.0
    delta = 1.0 / math.sqrt(math.pi * freq * mu * sicma) * 1000.0
    z_lift = 1.0
    N, Res = 16, 0.78

    try:
        if shape_name == "Ellipse":
            H_tensor = physics_module.compute_map_gpu(w_val, l_val, d_val, delta, z_lift)
            H_raw = (csi / (4.0 * math.pi)) * H_tensor * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
        elif shape_name == "Rectangular":
            H_tensor, _, _ = physics_module.compute_magnetic_field(
                w_val, l_val, d_val, delta, z_lift, N, Res, angle=0, K=K, I=I, G=G, csi=csi
            )
            H_pred = torch.rot90(H_tensor, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
        elif shape_name == "Triangular":
            H_tensor = physics_module.compute_triangular_map_gpu(w_val, l_val, d_val, delta, z_lift)
            H_raw = (csi / (4.0 * math.pi)) * H_tensor * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
        elif shape_name == "Step_R":
            xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
            H_val = physics_module.calculate_field_Step_R(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
            H_raw = H_val * (csi / (4.0 * math.pi)) * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.flip(H_pred, dims=(1,))
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
        elif shape_name == "Step_T":
            xs = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            ys = torch.linspace(-N * Res, N * Res, 2 * N, device=device)
            X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
            H_val = physics_module.calculate_field_Step_T(X_grid, Y_grid, z_lift, w_val, l_val, d_val, delta)
            H_raw = H_val * (csi / (4.0 * math.pi)) * K * I * G
            H_pred = torch.rot90(H_raw, 2, dims=(0, 1))
            H_pred = torch.diff(H_pred, dim=1)
            H_pred = torch.flip(H_pred, dims=(1,))
            H_pred = torch.cat([H_pred, H_pred[:, -1:]], dim=1)
        else:
            return base_zero

        H_true_t = H_true.float().to(target_device)
        if H_true_t.ndim == 3:
            H_true_t = H_true_t[:, :, 0]

        if X_scaler is not None and hasattr(X_scaler, "mean_") and hasattr(X_scaler, "scale_"):
            field_mean = float(X_scaler.mean_[0])
            field_scale = float(X_scaler.scale_[0])
            H_true_t = H_true_t * field_scale + field_mean

        diff = H_true_t - H_pred
        sse = torch.sum(diff * diff)
        true_energy = torch.sum(H_true_t * H_true_t)
        norm_ratio = sse / (true_energy + 1e-12)
        physics_loss = torch.log1p(norm_ratio)
        orig_device = H_true.device if torch.is_tensor(H_true) else "cpu"
        return physics_loss.to(orig_device)
    except Exception as exc:
        return base_zero


