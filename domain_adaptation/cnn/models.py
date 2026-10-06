"""
CNN Multitask Architectures for Shape Classification and 3D Regression.
Faithfully extracted from cnn/main_percent_new.py and cnn/main_percent_no_pinn.py.
Compact architecture (~110k parameters) with Sigmoid-bounded regression head in (0, 1).
"""

import torch
import torch.nn as nn
from .config import CNNModelConfig

class ImprovedMultimodelNet(nn.Module):
    """
    Multitask CNN network for shape classification and joint W/L/D regression
    with Kendall et al. (2018) Homoscedastic Uncertainty Weighting.
    Used for PINN models (incorporating physical regularizers).
    Compact architecture: 128-dim compressed latent bottleneck and simplified 2 heads.
    Regression head uses Sigmoid to bound output to (0, 1) matching normalized targets.
    """
    def __init__(self, num_shapes: int = 5, config: CNNModelConfig = None, latent_dim: int = 128):
        super(ImprovedMultimodelNet, self).__init__()
        self.config = config or CNNModelConfig()
        self.num_shapes = num_shapes
        self.latent_dim = latent_dim
        
        # Learnable uncertainty parameters (4 variables: 1 for clf, 3 for W, L, D)
        self.log_var_clf = nn.Parameter(torch.tensor(0.0))
        self.log_var_w = nn.Parameter(torch.tensor(0.0))
        self.log_var_l = nn.Parameter(torch.tensor(0.0))
        self.log_var_d = nn.Parameter(torch.tensor(0.0))
        
        # ===== COMPACT SHARED 2D CONVOLUTIONAL BACKBONE =====
        # Input: (B, 2, 32, 32) -> Compressed Latent Feature: (B, 128)
        self.backbone = nn.Sequential(
            # Block 1 (32x32 -> 16x16)
            nn.Conv2d(self.config.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),

            # Block 2 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),

            # Block 3 (8x8 -> 1x1 Global Average Pool)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # ===== SIMPLE CLASSIFICATION HEAD =====
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_shapes)
        )
        
        # ===== SIMPLE REGRESSION BACKBONE & HEAD (W, L, D) =====
        # Sigmoid at the end bounds output to (0, 1) matching normalized targets
        self.regressor_backbone = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (B, 2, 32, 32)
        Returns:
            shape_logits: (B, num_shapes) raw logits
            y_pred_wld: (B, 3) representing [W, L, D] bounded to (0, 1)
        """
        backbone_feat = self.backbone(x)
        backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)

        # Classification branch
        shape_logits = self.classifier(backbone_feat)

        # Regression branch
        reg_feat = self.regressor_backbone(backbone_feat)
        y_pred_wld = self.reg_head(reg_feat)

        return shape_logits, y_pred_wld

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-level latent features before task heads (128-dim)."""
        backbone_feat = self.backbone(x)
        return backbone_feat.view(backbone_feat.size(0), -1)


class ImprovedMultimodelNet_NoPINN(ImprovedMultimodelNet):
    """
    Multitask CNN network baseline (NoPINN).
    Identical architectural backbone to ImprovedMultimodelNet, but trained without
    PDE physics loss (purely supervised data loss with Kendall uncertainty).
    """
    def __init__(self, num_shapes: int = 5, config: CNNModelConfig = None, latent_dim: int = 128):
        super(ImprovedMultimodelNet_NoPINN, self).__init__(num_shapes=num_shapes, config=config, latent_dim=latent_dim)


class CNN_SingleTask_Classification(nn.Module):
    """
    Single-Task 2D CNN for 5-class defect shape classification only.
    """
    def __init__(self, num_shapes: int = 5, config: CNNModelConfig = None, latent_dim: int = 128):
        super(CNN_SingleTask_Classification, self).__init__()
        self.config = config or CNNModelConfig()
        self.num_shapes = num_shapes
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(self.config.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_shapes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)


class CNN_SingleTask_Regression(nn.Module):
    """
    Single-Task 2D CNN for 3D crack sizing regression (W, L, D) only.
    With Kendall Homoscedastic Uncertainty across 3 dimensions (W, L, D).
    Regression head uses Sigmoid to bound output to (0, 1).
    """
    def __init__(self, config: CNNModelConfig = None, latent_dim: int = 128):
        super(CNN_SingleTask_Regression, self).__init__()
        self.config = config or CNNModelConfig()
        self.latent_dim = latent_dim

        self.log_var_w = nn.Parameter(torch.tensor(0.0))
        self.log_var_l = nn.Parameter(torch.tensor(0.0))
        self.log_var_d = nn.Parameter(torch.tensor(0.0))

        self.backbone = nn.Sequential(
            nn.Conv2d(self.config.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor_backbone = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        reg_feat = self.regressor_backbone(feat)
        return self.reg_head(reg_feat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)
