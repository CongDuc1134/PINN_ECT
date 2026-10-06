"""
Loss functions for Xiong et al. (2023) Single-Task Architecture.
Faithfully extracted from mlp/xiong_2023_baseline.py.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class XiongRegressionLoss(nn.Module):
    """
    Computes individual and mean regression losses across W, L, D for Xiong baseline.
    """
    def __init__(self):
        super(XiongRegressionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, y_pred_wld: torch.Tensor, y_wld: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_loss = self.criterion(y_pred_wld[:, 0], y_wld[:, 0])
        l_loss = self.criterion(y_pred_wld[:, 1], y_wld[:, 1])
        d_loss = self.criterion(y_pred_wld[:, 2], y_wld[:, 2])
        avg_reg_loss = (w_loss + l_loss + d_loss) / 3.0
        return w_loss, l_loss, d_loss, avg_reg_loss


class XiongTotalLoss(nn.Module):
    """
    Single-Task Total Loss for Xiong et al. (2023):
    Total Loss = MSE(W, L, D) + alpha * Physics_Loss (PINN).
    Notice: Strictly NO classification loss and NO Kendall uncertainty parameters!
    """
    def __init__(self, alpha: float = 1.0, use_pinn: bool = True):
        super(XiongTotalLoss, self).__init__()
        self.alpha = alpha
        self.use_pinn = use_pinn
        self.reg_fn = XiongRegressionLoss()

    def forward(
        self,
        pred_wld: torch.Tensor,
        true_wld: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        w_loss, l_loss, d_loss, avg_reg_loss = self.reg_fn(pred_wld, true_wld)

        if self.use_pinn and physics_loss is not None:
            total_loss = avg_reg_loss + self.alpha * physics_loss
        else:
            total_loss = avg_reg_loss

        metrics = {
            "total_loss": total_loss.item(),
            "w_loss": w_loss.item(),
            "l_loss": l_loss.item(),
            "d_loss": d_loss.item(),
            "avg_reg_loss": avg_reg_loss.item(),
            "physics_loss": physics_loss.item() if physics_loss is not None else 0.0
        }
        return total_loss, metrics
