"""
Configuration parameters for CNN Multitask Architectures (PINN and NoPINN).
Synchronized with cnn/main_percent_new.py and cnn/main_percent_no_pinn.py.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class CNNModelConfig:
    # Input field dimensions: 2 channels (Re/Im or Bx/Bz), 32x32 spatial grid
    input_channels: int = 2
    input_size: Tuple[int, int] = (32, 32)
    num_shapes: int = 5
    
    # Feature Extractor Backbone specifications (Compact Architecture ~110k params)
    conv_channels: Tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 3
    padding: int = 1
    dropout_rate: float = 0.05
    adaptive_pool_size: Tuple[int, int] = (1, 1)
    latent_dim: int = 128
    
    # Classification Head dimensions
    clf_hidden_dim: int = 64
    clf_dropout: float = 0.1
    
    # Regression Head dimensions
    reg_hidden_dim: int = 64
    reg_dropout: float = 0.1
    reg_output_dim: int = 3  # [W, L, D]

@dataclass
class CNNTrainingConfig:
    # Optimizer settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 32
    
    # Physics Loss (PINN) settings
    alpha_pinn: float = 1.0
    use_pinn_loss: bool = True
    
    # Homoscedastic Uncertainty Weighting (Kendall et al., 2018)
    use_uncertainty_weighting: bool = True
    initial_log_var: float = 0.0
