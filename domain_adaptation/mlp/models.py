"""
Multitask MLP Architecture for Shape Classification and 3D Regression.
Faithfully extracted from mlp/mlp.py.
"""

import torch
import torch.nn as nn
from .config import MLPModelConfig

class MultitaskMLP_PINN(nn.Module):
    """
    MLP-based multitask network for shape classification + joint W/L/D regression.
    Adapted from Xiong et al. 2023 Table 3 backbone:
    - Input: flattened 32x32x2 = 2048 features
    - Shared MLP backbone (3 hidden layers): 2048 -> 64 -> 32 -> 16 with Tanh
    - Classification head: 16 -> num_shapes
    - Direct Regression head: 16 -> 3 (W, L, D) with Softplus activation
    - Pure MLP formulation without BatchNorm/Dropout matching Table 3
    """
    INPUT_DIM = 32 * 32 * 2  # 2048

    def __init__(self, num_shapes: int = 5, input_dim: int = None, config: MLPModelConfig = None):
        super(MultitaskMLP_PINN, self).__init__()
        self.config = config or MLPModelConfig()
        self.num_shapes = num_shapes
        in_dim = input_dim or self.config.input_dim

        # Learnable uncertainty parameters (Classification vs Mean Regression)
        self.log_var_clf = nn.Parameter(torch.tensor(0.0))
        self.log_var_reg = nn.Parameter(torch.tensor(0.0))

        # ===== SHARED MLP BACKBONE (3 HIDDEN LAYERS) =====
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

        # ===== DIRECT CLASSIFICATION HEAD =====
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dims[2], num_shapes),
        )

        # ===== DIRECT REGRESSION HEAD (W, L, D) WITH SOFTPLUS =====
        self.reg_head = nn.Sequential(
            nn.Linear(self.config.hidden_dims[2], self.config.regression_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: shape (B, 2048) or (B, 2, 32, 32)
        Returns:
            shape_logits: (B, num_shapes)
            y_pred_wld: (B, 3) representing [W, L, D] strictly positive
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        shared_feat = self.backbone(x)

        # Classification branch
        shape_logits = self.classifier(shared_feat)

        # Regression branch
        y_pred_wld = self.reg_head(shared_feat)

        return shape_logits, y_pred_wld

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-level latent features (16D) before task heads."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.backbone(x)


# Backward-compatible alias
ImprovedMultimodelNet = MultitaskMLP_PINN
