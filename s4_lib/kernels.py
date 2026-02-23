"""
S4 Kernel Computations.

The core bottleneck in SSMs is computing the convolution kernel:
    K = (C B̄, C Ā B̄, C Ā² B̄, ..., C Ā^{L-1} B̄)

For general (N×N) A, this is O(N²L).  S4 reduces it to O((N+L) log(N+L))
via the DPLR parameterization + Cauchy kernel + FFT.

For S4D (diagonal), the kernel simplifies to just two lines of code.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Cauchy kernel
# ---------------------------------------------------------------------------

def cauchy_naive(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Naive O(NL) Cauchy kernel: sum_i v_i / (z_j - w_i).

    Args:
        v: (..., N) numerator coefficients.
        z: (..., L) evaluation points.
        w: (..., N) poles.

    Returns:
        (..., L) result.
    """
    # (..., L, N)
    cauchy = 1.0 / (z.unsqueeze(-1) - w.unsqueeze(-2))
    return torch.einsum("...ln,...n->...l", cauchy, v)


# ---------------------------------------------------------------------------
# DPLR kernel  (full S4)
# ---------------------------------------------------------------------------

def kernel_dplr(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    L: int,
    method: str = "bilinear",
) -> torch.Tensor:
    """
    S4 convolution kernel via DPLR parameterization.

    Uses the Woodbury identity so the resolvent of A = diag(Λ) - P Q^T
    decomposes into four Cauchy dot-products.

    Args:
        Lambda: (N,) complex diagonal eigenvalues.
        P, Q: (N,) low-rank factors (Q = P for symmetric case).
        B: (N,) transformed input vector.
        C: (N,) output vector.
        dt: Scalar step size.
        L: Sequence length.
        method: "bilinear" or "zoh".

    Returns:
        (L,) real convolution kernel.
    """
    Omega = torch.exp(2j * math.pi * torch.arange(L, device=Lambda.device) / L)

    if method == "bilinear":
        z = 2.0 * (1.0 + Omega) / (1.0 - Omega + 1e-7) / dt
    else:
        z = torch.log(Omega + 1e-7) / dt

    # Helper: Cauchy dot product  sum_i a_i / (z_j - w_i)
    def cdot(a, z, w):
        return (a.unsqueeze(-2) / (z.unsqueeze(-1) - w.unsqueeze(-2))).sum(-1)

    k00 = cdot(B * C, z, Lambda)
    k01 = cdot(B * Q, z, Lambda)
    k10 = cdot(P * C, z, Lambda)
    k11 = cdot(P * Q, z, Lambda)

    K_hat = k00 - k01 * k10 / (1.0 + k11)

    if method == "bilinear":
        K_hat = K_hat * 2.0 / (1.0 - Omega + 1e-7)

    return torch.fft.ifft(K_hat, n=L).real


# ---------------------------------------------------------------------------
# Diagonal kernel  (S4D — "two lines of code")
# ---------------------------------------------------------------------------

def kernel_diagonal(
    Lambda: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    L: int,
    method: str = "zoh",
) -> torch.Tensor:
    """
    S4D kernel: direct Vandermonde-style computation.

    K[l] = Re[ Σ_n C_n · Ā_n^l · B̄_n ]

    Args:
        Lambda: (N,) complex continuous eigenvalues.
        B, C: (N,) complex vectors.
        dt: Step size.
        L: Sequence length.
        method: "zoh" or "bilinear".

    Returns:
        (L,) real kernel.
    """
    if method == "zoh":
        Ab = torch.exp(Lambda * dt)
        Bb = B * (Ab - 1.0) / Lambda
    else:
        d = Lambda * dt / 2
        Ab = (1 + d) / (1 - d)
        Bb = B * dt / (1 - d)

    # (N, L):  Ab^0, Ab^1, ..., Ab^{L-1}
    powers = Ab.unsqueeze(-1) ** torch.arange(L, device=Lambda.device).float()
    return ((C * Bb).unsqueeze(-1) * powers).sum(0).real


def kernel_diagonal_fft(
    Lambda: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    L: int,
    method: str = "zoh",
) -> torch.Tensor:
    """
    Memory-efficient diagonal kernel via z-transform + IFFT.

    K̂[k] = Σ_n C_n B̄_n / (1 - Ā_n exp(-2πik/L))

    Args:
        Lambda, B, C, dt, L, method: same as kernel_diagonal.

    Returns:
        (L,) real kernel.
    """
    if method == "zoh":
        Ab = torch.exp(Lambda * dt)
        Bb = B * (Ab - 1.0) / Lambda
    else:
        d = Lambda * dt / 2
        Ab = (1 + d) / (1 - d)
        Bb = B * dt / (1 - d)

    omega = torch.exp(-2j * math.pi * torch.arange(L, device=Lambda.device) / L)
    CB = (C * Bb).unsqueeze(-1)           # (N, 1)
    denom = 1.0 - Ab.unsqueeze(-1) * omega.unsqueeze(0)  # (N, L)
    K_hat = (CB / denom).sum(0)           # (L,)
    return torch.fft.ifft(K_hat, n=L).real


# ---------------------------------------------------------------------------
# FFT convolution
# ---------------------------------------------------------------------------

def fft_conv(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Causal convolution via FFT with zero-padding.

    Args:
        u: (..., L) input.
        K: (L,) or (..., L) kernel.

    Returns:
        (..., L) output.
    """
    L = u.shape[-1]
    fft_size = 1 << (2 * L - 1).bit_length()   # next power of 2
    u_f = torch.fft.rfft(u, n=fft_size)
    K_f = torch.fft.rfft(K, n=fft_size)
    return torch.fft.irfft(u_f * K_f, n=fft_size)[..., :L]
