"""
s4_lib — Structured State Spaces for Sequence Modeling
======================================================

A self-contained PyTorch implementation of S4 and S4D from:

* **S4**: "Efficiently Modeling Long Sequences with Structured State Spaces"
  (Gu, Goel, Ré — ICLR 2022)
* **S4D**: "On the Parameterization and Initialization of Diagonal State Space Models"
  (Gu, Gupta, Goel, Ré — NeurIPS 2022)

Quick start::

    from s4_lib import S4DLayer, S4SequenceModel

    # Single S4D layer
    layer = S4DLayer(d_model=128, d_state=64)
    y = layer(x)   # x: (batch, length, 128)

    # Full classification model
    model = S4SequenceModel(
        d_input=1, d_model=128, d_output=10,
        n_layers=4, task="classification",
    )
    logits = model(x)   # x: (batch, length, 1)
"""

from .s4_layer import S4Layer
from .s4d_layer import S4DLayer, S4DBlock
from .model import S4SequenceModel
from .utils import get_ssm_param_groups

__all__ = [
    "S4Layer",
    "S4DLayer",
    "S4DBlock",
    "S4SequenceModel",
    "get_ssm_param_groups",
]
