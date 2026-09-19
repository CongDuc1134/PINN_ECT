"""
================================================================================
domain_adaptation/models.py
Architecture definition for Multimodal ECT Net (matching pretrained checkpoints).
================================================================================
"""

import torch
import torch.nn as nn


class ImprovedMultimodelNet(nn.Module):
    """
    Multimodal network for shape classification and joint W/L/D regression
    with Kendall et al. (2018) Homoscedastic Uncertainty Weighting.
    Exactly matches the checkpoint weights in Outputs_cnn_pinn / Outputs_cnn_baseline.
    """
    def __init__(self, num_shapes=5):
        super(ImprovedMultimodelNet, self).__init__()
        self.num_shapes = num_shapes
        
        # Learnable uncertainty parameters
        self.log_var_clf = nn.Parameter(torch.tensor(0.0))
        self.log_var_w = nn.Parameter(torch.tensor(0.0))
        self.log_var_l = nn.Parameter(torch.tensor(0.0))
        self.log_var_d = nn.Parameter(torch.tensor(0.0))
        
        # ===== SHARED BACKBONE =====
        self.backbone = nn.Sequential(
            # Block 1 (32x32 -> 16x16)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 2 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 3 (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Dropout(0.1)
        )
        
        # ===== CLASSIFICATION HEAD =====
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_shapes)
        )
        
        # ===== REGRESSION BACKBONE =====
        self.regressor_backbone = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # ===== JOINT REGRESSION HEAD (W, L, D) =====
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        """Forward pass returning logits and W/L/D predictions"""
        backbone_feat = self.backbone(x)
        feat_flat = backbone_feat.reshape(backbone_feat.size(0), -1)

        shape_logits = self.classifier(feat_flat)
        reg_feat = self.regressor_backbone(feat_flat)
        y_pred_wld = self.reg_head(reg_feat)

        return shape_logits, y_pred_wld

    def extract_features(self, x):
        """Extract flat representation for feature alignment (MMD / CORAL)"""
        backbone_feat = self.backbone(x)
        return backbone_feat.reshape(backbone_feat.size(0), -1)
