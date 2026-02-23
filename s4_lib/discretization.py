"""
Discretization Methods for State Space Models.

Converts continuous-time SSM parameters (A, B) into discrete-time
parameters (A_bar, B_bar) using a learnable step size Δ.

Continuous:  x'(t) = A x(t) + B u(t),  y(t) = C x(t) + D u(t)
Discrete:    x[k]  = Ā x[k-1] + B̄ u[k],  y[k] = C x[k] + D u[k]

Two methods:
    1. Bilinear (Tustin) — used in the original S4 paper.
    2. Zero-Order Hold (ZOH) — used in S4D, Mamba, and later variants.
"""

import torch
from typing import Tuple


def bilinear(
    A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear (Tustin) discretization.

    Preserves stability: left half-plane maps to inside unit circle.

    For diagonal A (S4D):
        Ā = (1 + Δ/2 · A) / (1 - Δ/2 · A)
        B̄ = Δ · B / (1 - Δ/2 · A)

    For dense A:
        Ā = (I - Δ/2 A)^{-1} (I + Δ/2 A)
        B̄ = (I - Δ/2 A)^{-1} Δ B

    Args:
        A: (N,) diagonal or (N, N) dense state matrix.
        B: (N,) or (N, 1) input matrix.
        dt: Step size — scalar or broadcastable.

    Returns:
        (A_bar, B_bar): discretized parameters.
    """
    if A.dim() == 1:
        dA = dt * A
        A_bar = (1 + dA / 2) / (1 - dA / 2)
        B_bar = dt * B / (1 - dA / 2)
    else:
        N = A.shape[0]
        I = torch.eye(N, dtype=A.dtype, device=A.device)
        A_bar = torch.linalg.solve(I - (dt / 2) * A, I + (dt / 2) * A)
        B_bar = torch.linalg.solve(I - (dt / 2) * A, dt * B)
    return A_bar, B_bar


def zoh(
    A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Order Hold (ZOH) discretization.

    Assumes input is held constant between samples.

    For diagonal A:
        Ā = exp(Δ A)
        B̄ = (exp(Δ A) - 1) / A · B

    Args:
        A: (N,) diagonal or (N, N) dense state matrix.
        B: (N,) or (N, 1) input matrix.
        dt: Step size.

    Returns:
        (A_bar, B_bar): discretized parameters.
    """
    if A.dim() == 1:
        dA = dt * A
        A_bar = torch.exp(dA)
        B_bar = (A_bar - 1.0) / A * B
    else:
        dA = dt * A
        A_bar = torch.matrix_exp(dA)
        N = A.shape[0]
        I = torch.eye(N, dtype=A.dtype, device=A.device)
        B_bar = torch.linalg.solve(A, (A_bar - I) @ B)
    return A_bar, B_bar


def discretize(
    A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor, method: str = "bilinear"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize continuous SSM parameters.

    Args:
        A: Continuous state matrix.
        B: Continuous input matrix.
        dt: Step size (learnable in S4).
        method: "bilinear" or "zoh".

    Returns:
        (A_bar, B_bar).
    """
    if method == "bilinear":
        return bilinear(A, B, dt)
    elif method == "zoh":
        return zoh(A, B, dt)
    raise ValueError(f"Unknown method: {method}. Use 'bilinear' or 'zoh'.")
