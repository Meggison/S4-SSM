"""
S4D Layer — Diagonal State Space Model.

Simplified variant from:
    "On the Parameterization and Initialization of Diagonal State Space Models"
    (Gu, Gupta, Goel, Ré — NeurIPS 2022)

S4D restricts A to a diagonal matrix initialized from HiPPO eigenvalues.
The kernel computation reduces to ~2 lines of code yet performs comparably
to the full DPLR S4 on nearly all benchmarks.

Usage::

    layer = S4DLayer(d_model=128, d_state=64)
    y = layer(u)                           # (batch, length, d_model)
    y_t, state = layer.step(u_t, state)    # single-step RNN
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple

from .hippo import diagonal_init
from .kernels import kernel_diagonal_fft, fft_conv


class S4DLayer(nn.Module):
    """
    Diagonal State Space layer (S4D).

    Args:
        d_model: Feature dimension (H).
        d_state: State dimension (N). Default 64.
        dt_min: Min initial step size. Default 0.001.
        dt_max: Max initial step size. Default 0.1.
        disc: Discretization method ("zoh" or "bilinear"). Default "zoh".
        init: Initialization ("legs", "legt", "fout", "inv"). Default "legs".
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        disc: str = "zoh",
        init: str = "legs",
    ):
        super().__init__()
        self.H = d_model
        self.N = d_state
        self.disc = disc

        # --- Diagonal A init from HiPPO eigenvalues ---
        if init == "inv":
            from .hippo import s4d_inv_init
            Lambda = s4d_inv_init(d_state)
        else:
            Lambda = diagonal_init(d_state, method=init)

        # Store as real / imag
        self.Lambda_real = nn.Parameter(torch.from_numpy(Lambda.real).float())
        self.Lambda_imag = nn.Parameter(torch.from_numpy(Lambda.imag).float())

        # B: (H, N) — per-channel input projection, real-valued
        self.B_real = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))
        self.B_imag = nn.Parameter(torch.zeros(d_model, d_state))

        # C: (H, N) — per-channel output projection, complex random init
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

        # log Δ — per-channel, uniform in log-space
        log_dt = torch.rand(d_model) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # D — skip connection
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

    # ---- helpers ----

    def _cx(self):
        L = torch.complex(self.Lambda_real, self.Lambda_imag)  # (N,)
        B = torch.complex(self.B_real, self.B_imag)            # (H, N)
        C = torch.complex(self.C_real, self.C_imag)            # (H, N)
        return L, B, C

    def _kernel(self, length: int) -> torch.Tensor:
        """Compute (H, length) real kernel."""
        Lambda, B, C = self._cx()
        dt = torch.exp(self.log_dt)  # (H,)
        Ks = []
        for h in range(self.H):
            Ks.append(
                kernel_diagonal_fft(Lambda, B[h], C[h], dt[h], length, self.disc)
            )
        return torch.stack(Ks, dim=0)

    # ---- forward (CNN mode) ----

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        CNN-mode forward (training).

        Args:
            u: (batch, length, d_model).

        Returns:
            (batch, length, d_model).
        """
        x = u.transpose(-1, -2)              # (B, H, L)
        K = self._kernel(x.shape[-1])         # (H, L)
        y = fft_conv(x, K)                    # (B, H, L)
        y = y + x * self.D[None, :, None]     # skip
        return y.transpose(-1, -2)

    # ---- step (RNN mode) ----

    def step(
        self, u_t: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single RNN step.

        Args:
            u_t: (batch, d_model).
            state: (batch, d_model, d_state) complex.

        Returns:
            y_t: (batch, d_model).
            new_state.
        """
        Lambda, B, C = self._cx()
        dt = torch.exp(self.log_dt)

        # Discretize (diagonal, per-channel)
        Ab = torch.exp(Lambda.unsqueeze(0) * dt.unsqueeze(-1))  # (H, N)
        Bb = B * (Ab - 1.0) / (Lambda.unsqueeze(0) + 1e-8)

        u_exp = u_t.unsqueeze(-1).to(state.dtype)
        new_state = Ab.unsqueeze(0) * state + Bb.unsqueeze(0) * u_exp
        y_t = (C.unsqueeze(0) * new_state).sum(-1).real + u_t * self.D
        return y_t, new_state

    def init_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.H, self.N,
            dtype=torch.cfloat, device=self.log_dt.device,
        )


class S4DBlock(nn.Module):
    """
    Pre-norm residual block:  x → LayerNorm → S4D → residual → LN → FFN → residual

    Drop-in replacement for a Transformer block.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        disc: str = "zoh",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.s4d = S4DLayer(d_model=d_model, d_state=d_state, disc=disc)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, d_model).

        Returns:
            (batch, length, d_model).
        """
        x = x + self.drop1(self.s4d(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x
