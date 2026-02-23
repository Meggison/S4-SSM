"""
HiPPO (High-order Polynomial Projection Operators) Matrix Constructions.

Implements the special state matrices from:
- "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (Gu et al., 2020)
- "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2022)
- "How to Train Your HiPPO" (Gu et al., 2023)

The HiPPO framework provides principled initializations for the state matrix A
that optimally compress a continuous signal history into a finite-dimensional
state vector using orthogonal polynomial bases.

Key variants:
    LegS (Scaled Legendre) — default for S4, captures long-range dependencies.
    LegT (Truncated Legendre) — fixed-window approximation.
    FouT (Truncated Fourier) — Fourier basis alternative.
"""

import numpy as np
from typing import Tuple


def make_hippo_legs(N: int) -> np.ndarray:
    """
    Construct the HiPPO-LegS (Scaled Legendre) matrix.

    This is the primary matrix used in S4. It projects the input signal
    onto scaled Legendre polynomials, enabling the state to optimally
    approximate the entire input history.

    Matrix entries:
        A[n,k] = -(2n+1)^{1/2} (2k+1)^{1/2}   if n > k
        A[n,k] = -(n+1)                          if n = k
        A[n,k] = 0                                if n < k

    Args:
        N: State dimension (number of Legendre coefficients).

    Returns:
        (N, N) HiPPO-LegS matrix.
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_hippo_legt(N: int) -> np.ndarray:
    """
    Construct the HiPPO-LegT (Truncated Legendre) matrix.

    Uses truncated Legendre polynomials on a fixed window [0, 1].
    Better suited for bounded-context tasks.

    Args:
        N: State dimension.

    Returns:
        (N, N) HiPPO-LegT matrix.
    """
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, np.newaxis]
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    return A


def make_hippo_fout(N: int) -> np.ndarray:
    """
    Construct the HiPPO-FouT (Truncated Fourier) matrix.

    Fourier basis alternative introduced in "How to Train Your HiPPO".

    Args:
        N: State dimension (should be even for sin/cos pairs).

    Returns:
        (N, N) HiPPO-FouT matrix.
    """
    freqs = np.arange(N // 2)
    d = np.stack([freqs, freqs], axis=-1).reshape(-1)[1:]
    A = 2 * np.pi * (np.diag(d, 1) - np.diag(d, -1))
    A = A - np.eye(N)
    return A[:N, :N]


def make_hippo_b(N: int, method: str = "legs") -> np.ndarray:
    """
    Construct the HiPPO input matrix B.

    Args:
        N: State dimension.
        method: One of "legs", "legt", "fout".

    Returns:
        (N, 1) input matrix.
    """
    if method == "legs":
        B = np.sqrt(1 + 2 * np.arange(N))[:, np.newaxis]
    elif method == "legt":
        B = (2 * np.arange(N) + 1)[:, np.newaxis]
    elif method == "fout":
        B = np.ones((N, 1))
        B[0::2, 0] = 2**0.5
    else:
        raise ValueError(f"Unknown HiPPO method: {method}")
    return B


def nplr(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a HiPPO matrix into Normal Plus Low-Rank (NPLR) form.

    The HiPPO-LegS matrix A is decomposed as:
        A = V Λ V^* - P P^T
    where Λ is diagonal (eigenvalues of the normal part) and P captures
    the low-rank correction. This is the structural insight that makes
    S4's kernel computation efficient via Cauchy kernels.

    Args:
        A: (N, N) HiPPO matrix.
        B: (N, 1) input matrix.

    Returns:
        (Lambda, P, B_tilde, V):
            Lambda: (N,) complex eigenvalues of the normal part.
            P: (N,) low-rank correction in the eigenbasis.
            B_tilde: (N,) transformed B in the eigenbasis.
            V: (N, N) eigenvector matrix.
    """
    N = A.shape[0]

    # The HiPPO-LegS matrix is close to normal + rank-1.
    # Extract the low-rank factor from the symmetric part.
    S = A + A.T
    S_diag = np.diagonal(S)
    P = np.sqrt(np.abs(S_diag) / 2.0 + 0.5)

    # Normal part: A + P P^T
    A_normal = A + np.outer(P, P)

    # Diagonalize
    Lambda, V = np.linalg.eig(A_normal)

    # Sort by imaginary part for reproducibility
    idx = np.argsort(np.imag(Lambda))
    Lambda = Lambda[idx]
    V = V[:, idx]

    # Transform P, B into eigenbasis
    V_inv = np.linalg.inv(V)
    P_tilde = V_inv @ P
    B_tilde = V_inv @ B.squeeze(-1)

    return (
        Lambda.astype(np.complex64),
        P_tilde.astype(np.complex64),
        B_tilde.astype(np.complex64),
        V.astype(np.complex64),
    )


def dplr(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Diagonal Plus Low-Rank (DPLR) representation.

    Equivalent to NPLR — returns components for S4's efficient Cauchy
    kernel computation.

    Args:
        A: (N, N) HiPPO matrix.
        B: (N, 1) input matrix.

    Returns:
        (Lambda, P, B_tilde, V) — same as nplr().
    """
    return nplr(A, B)


def diagonal_init(N: int, method: str = "legs") -> np.ndarray:
    """
    Initialize diagonal state matrix from HiPPO eigenvalues (S4D-Lin).

    S4D uses a diagonal approximation: just the eigenvalues of the HiPPO
    matrix, discarding the low-rank correction. This is the default "S4D-Lin"
    initialization.

    Args:
        N: State dimension.
        method: HiPPO variant ("legs", "legt", "fout").

    Returns:
        (N,) complex eigenvalues.
    """
    if method == "legs":
        A = make_hippo_legs(N)
    elif method == "legt":
        A = make_hippo_legt(N)
    elif method == "fout":
        A = make_hippo_fout(N)
    else:
        raise ValueError(f"Unknown method: {method}")

    eigs = np.linalg.eigvals(A)
    idx = np.argsort(np.imag(eigs))
    return eigs[idx].astype(np.complex64)


def s4d_inv_init(N: int) -> np.ndarray:
    """
    S4D-Inv initialization: deterministic, no eigendecomposition needed.

    Lambda_n = -1/2 + i * pi * n

    Args:
        N: State dimension.

    Returns:
        (N,) complex eigenvalues.
    """
    return (-0.5 + 1j * np.pi * np.arange(N)).astype(np.complex64)
