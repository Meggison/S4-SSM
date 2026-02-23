"""
S4 Layer — Full DPLR (Diagonal Plus Low-Rank) Parameterization.

Implements the original S4 layer from:
    "Efficiently Modeling Long Sequences with Structured State Spaces"
    (Gu, Goel, Ré — ICLR 2022)

Dual modes:
    CNN (training)  — compute kernel K, then y = FFT_conv(u, K)
    RNN (inference) — step-by-step recurrence for autoregressive generation

Usage::

    layer = S4Layer(d_model=128, d_state=64)
    y = layer(u)                 # u: (batch, length, d_model)
    y, state = layer.step(u_t, state)   # single-step RNN
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple

from .hippo import make_hippo_legs, make_hippo_b, dplr
from .kernels import kernel_dplr, fft_conv


class S4Layer(nn.Module):
    """
    Full S4 layer with DPLR parameterization.

    Processes H independent SSM channels in parallel (one per feature).

    Args:
        d_model: Number of input/output features (H).
        d_state: SSM state dimension (N). Default 64.
        dt_min: Minimum initial step size. Default 0.001.
        dt_max: Maximum initial step size. Default 0.1.
        disc: Discretization method ("bilinear" or "zoh").
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        disc: str = "bilinear",
    ):
        super().__init__()
        self.H = d_model
        self.N = d_state
        self.disc = disc

        # --- HiPPO + DPLR init ---
        A = make_hippo_legs(d_state)
        B = make_hippo_b(d_state, "legs")
        Lambda, P, B_t, _V = dplr(A, B)

        # Store complex params as paired real/imag (avoids complex-param issues)
        self.Lambda_real = nn.Parameter(torch.from_numpy(Lambda.real).float())
        self.Lambda_imag = nn.Parameter(torch.from_numpy(Lambda.imag).float())
        self.P_real = nn.Parameter(torch.from_numpy(P.real).float())
        self.P_imag = nn.Parameter(torch.from_numpy(P.imag).float())
        self.B_real = nn.Parameter(torch.from_numpy(B_t.real).float())
        self.B_imag = nn.Parameter(torch.from_numpy(B_t.imag).float())

        # C: per-channel output projection, random init — (H, N)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

        # log Δ — uniform in log-space between [dt_min, dt_max]
        log_dt = torch.rand(d_model) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # D — skip / direct feed-through
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

    # ---- helpers ----

    def _cx(self):
        """Reconstruct complex parameters."""
        L = torch.complex(self.Lambda_real, self.Lambda_imag)
        P = torch.complex(self.P_real, self.P_imag)
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)
        return L, P, B, C

    def _kernel(self, length: int) -> torch.Tensor:
        """Compute (H, length) kernel — one per feature channel."""
        Lambda, P, B, C = self._cx()
        dt = torch.exp(self.log_dt)
        Ks = []
        for h in range(self.H):
            Ks.append(kernel_dplr(Lambda, P, P, B, C[h], dt[h], length, self.disc))
        return torch.stack(Ks, dim=0)

    # ---- forward (CNN mode) ----

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        CNN-mode forward pass (parallel, used during training).

        Args:
            u: (batch, length, d_model).

        Returns:
            (batch, length, d_model).
        """
        # u -> (B, H, L)
        x = u.transpose(-1, -2)
        K = self._kernel(x.shape[-1])       # (H, L)
        y = fft_conv(x, K)                  # (B, H, L)
        y = y + x * self.D[None, :, None]   # skip connection
        return y.transpose(-1, -2)           # (B, L, H)

    # ---- step (RNN mode) ----

    def step(
        self, u_t: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single RNN step for autoregressive inference.

        Args:
            u_t: (batch, d_model) — input at one timestep.
            state: (batch, d_model, d_state) complex hidden state.

        Returns:
            y_t: (batch, d_model) output.
            new_state: updated hidden state.
        """
        Lambda, _P, B, C = self._cx()
        dt = torch.exp(self.log_dt)

        # Diagonal discretization (approximate for RNN speed)
        Ab = torch.exp(Lambda.unsqueeze(0) * dt.unsqueeze(-1))       # (H, N)
        Bb = B.unsqueeze(0) * (Ab - 1.0) / (Lambda.unsqueeze(0) + 1e-8)

        u_exp = u_t.unsqueeze(-1).to(state.dtype)  # (B, H, 1)
        new_state = Ab.unsqueeze(0) * state + Bb.unsqueeze(0) * u_exp
        y_t = (C.unsqueeze(0) * new_state).sum(-1).real + u_t * self.D
        return y_t, new_state

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Zero-initialize RNN state: (batch, d_model, d_state) complex."""
        return torch.zeros(
            batch_size, self.H, self.N,
            dtype=torch.cfloat, device=self.log_dt.device,
        )
