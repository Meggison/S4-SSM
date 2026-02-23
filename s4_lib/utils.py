"""
Utility helpers for the S4 library.
"""

import torch
import torch.nn as nn
from typing import Dict, List


def get_ssm_param_groups(
    model: nn.Module,
    ssm_lr: float = 1e-3,
    ssm_wd: float = 0.0,
    default_lr: float = 1e-3,
    default_wd: float = 0.01,
) -> List[Dict]:
    """
    Create optimizer parameter groups with special SSM learning rates.

    The S4 paper recommends lr=0.001 and weight_decay=0 for the SSM
    kernel parameters (A, B, log_dt), while other parameters use the
    default optimizer settings.

    Usage::

        groups = get_ssm_param_groups(model, ssm_lr=1e-3)
        optimizer = torch.optim.AdamW(groups, lr=1e-3)

    Args:
        model: S4 model.
        ssm_lr: Learning rate for SSM parameters.
        ssm_wd: Weight decay for SSM parameters (usually 0).
        default_lr: Learning rate for other parameters.
        default_wd: Weight decay for other parameters.

    Returns:
        List of parameter-group dicts.
    """
    ssm_names = {
        "log_dt", "log_A_real", "A_imag",
        "B_real", "B_imag", "C_real", "C_imag",
        "Lambda_real", "Lambda_imag",
        "inv_A_real",
    }
    ssm_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (ssm_params if name.split(".")[-1] in ssm_names else other_params).append(p)
    return [
        {"params": ssm_params, "lr": ssm_lr, "weight_decay": ssm_wd},
        {"params": other_params, "lr": default_lr, "weight_decay": default_wd},
    ]


class DropoutNd(nn.Module):
    """Channel-wise dropout (same mask across the sequence dimension)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, features, length).

        Returns:
            Dropped-out tensor.
        """
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(
            torch.full(x.shape[:2] + (1,), 1 - self.p, device=x.device)
        ) / (1 - self.p)
        return x * mask
