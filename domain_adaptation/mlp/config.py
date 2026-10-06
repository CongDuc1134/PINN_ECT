"""
Configuration parameters for Multitask MLP PINN Architecture.
Extracted from mlp/mlp.py.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class MLPModelConfig:
    # Input dimension: flattened 32x32x2 = 2048
    input_dim: int = 32 * 32 * 2
    num_shapes: int = 5
    
    # 3-Layer Hidden Backbone [Xiong et al. 2023 Table 3]
    hidden_dims: List[int] = (64, 32, 16)
    activation: str = "Tanh"
    
    # Output heads
    regression_dim: int = 3  # [W, L, D]
    regression_activation: str = "Softplus"  # strictly positive output

@dataclass
class MLPTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    batch_size: int = 32
    
    # Alpha PINN weighting
    alpha_pinn: float = 1.0
    use_pinn_loss: bool = True
    
    # Kendall et al. (2018) 2-variable uncertainty weighting (Clf vs Mean Reg)
    use_uncertainty_weighting: bool = True
