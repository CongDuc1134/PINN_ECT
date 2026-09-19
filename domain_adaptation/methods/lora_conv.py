# -*- coding: utf-8 -*-
"""
================================================================================
domain_adaptation/methods/lora_conv.py
Parameter-Efficient Fine-Tuning via Low-Rank Adaptation (LoRA) for Conv2d layers.
Preserves simulation PDE inductive bias while allowing low-rank manifold adaptation
to real sensor artifacts (probe tilt, DC drift, lift-off variations).
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """
    LoRA Conv2d layer wrapping an existing pre-trained Conv2d layer.
    W = W_0 + (alpha / rank) * (B @ A)
    - W_0: Frozen pre-trained simulation weights
    - A: Down-projection conv (kernel_size x kernel_size), Kaiming normal init
    - B: Up-projection conv (1x1), Zero init (ensures Delta W = 0 at step 0)
    """
    def __init__(self, original_conv: nn.Conv2d, rank: int = 4, alpha: float = 8.0):
        super(LoRAConv2d, self).__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        # 1. Preserve and freeze original pre-trained simulation weights
        self.weight = nn.Parameter(original_conv.weight.data.clone(), requires_grad=False)
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # 2. LoRA parameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank) if rank > 0 else 1.0

        if rank > 0:
            device = original_conv.weight.device
            dtype = original_conv.weight.dtype
            # Down-projection with original spatial receptive field
            self.lora_A = nn.Conv2d(
                self.in_channels,
                rank,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=False,
                device=device,
                dtype=dtype
            )
            # Up-projection (1x1 conv)
            self.lora_B = nn.Conv2d(
                rank,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                device=device,
                dtype=dtype
            )
            # Initialize A with Kaiming uniform and B with zeros
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass using frozen pre-trained simulation weights
        base_out = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        # Add LoRA adaptation term if rank > 0
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.lora_B(self.lora_A(x)) * self.scaling
            return base_out + lora_out
        return base_out


def inject_lora_into_model(model: nn.Module, rank: int = 4, alpha: float = 8.0):
    """
    Recursively finds and replaces all nn.Conv2d layers in model.backbone with LoRAConv2d.
    Freezes all non-LoRA parameters in the backbone.
    Returns: modified model with LoRA adapters.
    """
    if not hasattr(model, "backbone"):
        return model

    # Traverse sequential backbone and replace Conv2d layers
    for i, layer in enumerate(model.backbone):
        if isinstance(layer, nn.Conv2d) and not isinstance(layer, LoRAConv2d):
            lora_layer = LoRAConv2d(layer, rank=rank, alpha=alpha)
            model.backbone[i] = lora_layer

    # Freeze non-LoRA backbone parameters
    for name, param in model.backbone.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def get_lora_trainable_parameters(model: nn.Module, lr_lora: float = 1e-3, lr_head: float = 5e-4):
    """
    Builds distinct parameter groups for optimizer:
    - LoRA conv adapters (lr_lora)
    - Classifier and Regression heads (lr_head)
    - Uncertainty parameters (1e-3)
    """
    lora_params = []
    head_params = []
    uncert_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif "log_var" in name:
            uncert_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': lr_lora, 'weight_decay': 1e-4})
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': 1e-3})
    if uncert_params:
        param_groups.append({'params': uncert_params, 'lr': 1e-3})

    return param_groups
