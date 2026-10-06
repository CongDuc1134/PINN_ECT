"""
Configuration parameters for Xiong et al. (2023) Baseline PINN Architecture.
Faithfully extracted from mlp/xiong_2023_baseline.py:
"Magnetic flux leakage defect size estimation method based on physics-informed neural network"
(Phil. Trans. R. Soc. A 382: 20220387, Table 3).
"""

from dataclasses import dataclass
from typing import List

@dataclass
class XiongModelConfig:
    # Input dimension: flattened 32x32x2 = 2048
    input_dim: int = 32 * 32 * 2
    
    # 3-Layer Hidden Backbone [Table 3]
    hidden_dims: List[int] = (64, 32, 16)
    activation: str = "Tanh"
    
    # Strictly SINGLE-TASK Regression: No Classification Head!
    regression_dim: int = 3  # [W, L, D]
    regression_activation: str = "Softplus"

@dataclass
class XiongTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    batch_size: int = 32
    
    # Alpha PINN weighting
    alpha_pinn: float = 1.0
    use_pinn_loss: bool = True
    
    # Notice: NO Kendall Uncertainty Weighting (Single-Task Regression uses direct MSE)
    use_uncertainty_weighting: bool = False
