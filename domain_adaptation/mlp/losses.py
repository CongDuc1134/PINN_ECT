"""
Loss functions for Multitask MLP PINN Architecture.
Faithfully extracted from mlp/mlp.py.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultitaskMLPClassificationLoss(nn.Module):
    def __init__(self):
        super(MultitaskMLPClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class MultitaskMLPRegressionLoss(nn.Module):
    """
    Computes individual and mean regression losses across W, L, D.
    """
    def __init__(self):
        super(MultitaskMLPRegressionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, y_pred_wld: torch.Tensor, y_wld: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_loss = self.criterion(y_pred_wld[:, 0], y_wld[:, 0])
        l_loss = self.criterion(y_pred_wld[:, 1], y_wld[:, 1])
        d_loss = self.criterion(y_pred_wld[:, 2], y_wld[:, 2])
        avg_reg_loss = (w_loss + l_loss + d_loss) / 3.0
        return w_loss, l_loss, d_loss, avg_reg_loss


class MultitaskMLPTotalLoss(nn.Module):
    """
    Kendall et al. (2018) 2-variable Homoscedastic Uncertainty Loss for MLP:
    Balances Classification Loss vs Mean 3D Regression Loss.
    """
    def __init__(self, alpha: float = 1.0, use_pinn: bool = True):
        super(MultitaskMLPTotalLoss, self).__init__()
        self.alpha = alpha
        self.use_pinn = use_pinn
        self.clf_fn = MultitaskMLPClassificationLoss()
        self.reg_fn = MultitaskMLPRegressionLoss()

    def forward(
        self,
        shape_logits: torch.Tensor,
        shape_targets: torch.Tensor,
        pred_wld: torch.Tensor,
        true_wld: torch.Tensor,
        log_var_clf: Optional[torch.Tensor] = None,
        log_var_reg: Optional[torch.Tensor] = None,
        physics_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        loss_clf = self.clf_fn(shape_logits, shape_targets)
        w_loss, l_loss, d_loss, avg_reg_loss = self.reg_fn(pred_wld, true_wld)

        # 2-variable Kendall formulation (Clf vs Mean Reg)
        if log_var_clf is not None and log_var_reg is not None:
            precision_clf = torch.exp(-log_var_clf)
            precision_reg = torch.exp(-log_var_reg)
            data_loss = (
                precision_clf * loss_clf + 0.5 * log_var_clf +
                precision_reg * avg_reg_loss + 0.5 * log_var_reg
            )
        else:
            data_loss = loss_clf + avg_reg_loss

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
