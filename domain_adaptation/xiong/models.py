"""
Single-Task 3D Regression MLP Architecture of Xiong et al. (2023).
Faithfully extracted from mlp/xiong_2023_baseline.py:
"Magnetic flux leakage defect size estimation method based on physics-informed neural network"
(Phil. Trans. R. Soc. A 382: 20220387, Table 3).
"""

import torch
import torch.nn as nn
from .config import XiongModelConfig

class RegressionMLP_PINN(nn.Module):
    """
    MLP-based regression network for joint W/L/D defect size estimation.
    Strictly SINGLE-TASK: does NOT have a classification head.
    
    Architecture (Table 3):
    - Input: flattened vector (2048)
    - Hidden Layer 1: 64 neurons, Tanh
    - Hidden Layer 2: 32 neurons, Tanh
    - Hidden Layer 3: 16 neurons, Tanh
    - Output: 3 neurons (W, L, D), Softplus (strictly positive)
    - No BatchNorm, No Dropout (pure MLP formulation)
    """
    INPUT_DIM = 32 * 32 * 2  # 2048

    def __init__(self, input_dim: int = None, config: XiongModelConfig = None, *args, **kwargs):
        super(RegressionMLP_PINN, self).__init__()
        self.config = config or XiongModelConfig()
        in_dim = input_dim or self.config.input_dim

        # ===== 3 HIDDEN LAYERS (64 -> 32 -> 16) WITH TANH [TABLE 3] =====
        self.backbone = nn.Sequential(
            # Hidden Layer 1: in_dim -> 64
            nn.Linear(in_dim, self.config.hidden_dims[0]),
            nn.Tanh(),

            # Hidden Layer 2: 64 -> 32
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
            nn.Tanh(),

            # Hidden Layer 3: 32 -> 16
            nn.Linear(self.config.hidden_dims[1], self.config.hidden_dims[2]),
            nn.Tanh(),
        )

        # ===== OUTPUT LAYER (16 -> 3) WITH SOFTPLUS [TABLE 3] =====
        self.reg_head = nn.Sequential(
            nn.Linear(self.config.hidden_dims[2], self.config.regression_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: shape (B, 2048) or (B, 2, 32, 32)
        Returns:
            y_pred_wld: shape (B, 3) representing [W, L, D] strictly positive
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        shared_feat = self.backbone(x)
        y_pred_wld = self.reg_head(shared_feat)
        return y_pred_wld

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-level latent features (16D) before regression head."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.backbone(x)


# Backward-compatible aliases
MultitaskMLP_PINN = RegressionMLP_PINN
ImprovedMultimodelNet = RegressionMLP_PINN
