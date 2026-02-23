"""
S4 Sequence Model — stack of S4/S4D blocks with encoder/decoder heads.

Provides a complete model that can be used for classification, regression,
or generation tasks on sequential data.

Usage::

    # Sequence classification (e.g. sCIFAR, LRA)
    model = S4SequenceModel(
        d_input=1, d_model=128, d_output=10, n_layers=4, task="classification"
    )
    logits = model(x)   # x: (batch, length, 1)  →  logits: (batch, 10)

    # Forecasting
    model = S4SequenceModel(
        d_input=3, d_model=64, d_output=3, n_layers=3, task="regression"
    )
    preds = model(x)    # x: (batch, length, 3)  →  preds: (batch, length, 3)
"""

import torch
import torch.nn as nn
from typing import Optional

from .s4d_layer import S4DBlock


class S4SequenceModel(nn.Module):
    """
    Full sequence model built from stacked S4D blocks.

    Architecture:
        Linear encoder → N × S4DBlock → pooling/projection → Linear decoder

    Args:
        d_input: Input feature dimension (e.g. 1 for pixel-by-pixel, 3 for RGB).
        d_model: Internal feature dimension.
        d_output: Output dimension (num_classes for classification).
        n_layers: Number of S4D blocks.
        d_state: SSM state dimension per block. Default 64.
        dropout: Dropout probability. Default 0.1.
        task: "classification" (pool over time → single output) or
              "regression" (per-timestep output).
        disc: Discretization method passed to S4D blocks.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.1,
        task: str = "classification",
        disc: str = "zoh",
    ):
        super().__init__()
        self.task = task

        # Encoder: project input features to model dimension
        self.encoder = nn.Linear(d_input, d_model)

        # Stack of S4D blocks
        self.blocks = nn.ModuleList([
            S4DBlock(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                disc=disc,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, d_input).

        Returns:
            classification: (batch, d_output) — pooled logits.
            regression: (batch, length, d_output) — per-step output.
        """
        # Encode
        x = self.encoder(x)  # (B, L, d_model)

        # S4D blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.task == "classification":
            # Global average pooling over time
            x = x.mean(dim=1)  # (B, d_model)

        return self.decoder(x)
