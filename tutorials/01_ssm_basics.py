#!/usr/bin/env python3
"""
Tutorial 1 — SSM Basics
========================

This tutorial walks through the foundational concepts behind S4:

1. What is a State Space Model (SSM)?
2. The HiPPO matrix — why initialization matters
3. Discretization — bridging continuous and discrete worlds
4. CNN vs RNN duality — why S4 is efficient

Run:  python tutorials/01_ssm_basics.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# 1. What is a State Space Model?
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. WHAT IS A STATE SPACE MODEL?")
print("=" * 60)
print("""
A State Space Model (SSM) maps an input signal u(t) to an output y(t)
through a hidden state x(t):

  Continuous form:
    x'(t) = A x(t) + B u(t)      (state equation)
    y(t)  = C x(t) + D u(t)      (output equation)

  Where:
    A ∈ R^{N×N}  — state matrix (controls dynamics)
    B ∈ R^{N×1}  — input matrix
    C ∈ R^{1×N}  — output matrix
    D ∈ R        — skip connection
    N            — state dimension

Key insight: the choice of A determines what the model can learn.
Random A → poor long-range performance.
HiPPO A  → optimal polynomial approximation of input history.
""")

# ═══════════════════════════════════════════════════════════════════════════
# 2. The HiPPO Matrix
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("2. THE HiPPO MATRIX")
print("=" * 60)

from s4_lib.hippo import make_hippo_legs, make_hippo_b, diagonal_init

N = 16  # small state dim for visualization
A = make_hippo_legs(N)
B = make_hippo_b(N, "legs")

print(f"HiPPO-LegS matrix A shape: {A.shape}")
print(f"A is lower-triangular (plus diagonal): {np.allclose(A, np.tril(A))}")
print(f"Eigenvalues have negative real parts (stable):")
eigs = np.linalg.eigvals(A)
print(f"  Re(λ) range: [{eigs.real.min():.2f}, {eigs.real.max():.2f}]")
print(f"  All stable (Re < 0): {(eigs.real < 0).all()}")
print()

# Visualize the matrix
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im = axes[0].imshow(A, cmap="RdBu_r", vmin=-A.max(), vmax=A.max())
axes[0].set_title("HiPPO-LegS Matrix (N=16)")
axes[0].set_xlabel("Column k")
axes[0].set_ylabel("Row n")
plt.colorbar(im, ax=axes[0])

# Plot eigenvalues
axes[1].scatter(eigs.real, eigs.imag, c="steelblue", edgecolors="k", s=50)
axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_title("Eigenvalues of HiPPO-LegS")
axes[1].set_xlabel("Real")
axes[1].set_ylabel("Imaginary")
plt.tight_layout()
plt.savefig("tutorials/hippo_matrix.png", dpi=100)
print("→ Saved tutorials/hippo_matrix.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Discretization
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("3. DISCRETIZATION")
print("=" * 60)
print("""
To use SSMs on discrete sequences (text, audio, pixels), we discretize:

  x[k] = Ā x[k-1] + B̄ u[k]
  y[k] = C x[k] + D u[k]

Two methods:
  • Bilinear (Tustin):  Ā = (I - Δ/2 A)⁻¹(I + Δ/2 A)
  • Zero-Order Hold:    Ā = exp(Δ A)

The step size Δ is learnable — it controls the "resolution" of the model.
""")

from s4_lib.discretization import bilinear, zoh

# Example: discretize a small diagonal system
Lambda = torch.tensor([-1.0 + 2j, -3.0 - 1j], dtype=torch.cfloat)
B_ex = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.cfloat)
dt = torch.tensor(0.01)

Ab_bil, Bb_bil = bilinear(Lambda, B_ex, dt)
Ab_zoh, Bb_zoh = zoh(Lambda, B_ex, dt)

print(f"Continuous A:   {Lambda.tolist()}")
print(f"Bilinear Ā:     {Ab_bil.tolist()}")
print(f"ZOH Ā:          {Ab_zoh.tolist()}")
print(f"|Ā| < 1 (stable): bilinear={all(abs(a) < 1 for a in Ab_bil.tolist())}, "
      f"zoh={all(abs(a) < 1 for a in Ab_zoh.tolist())}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 4. CNN vs RNN Duality
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("4. CNN vs RNN DUALITY")
print("=" * 60)
print("""
The discrete SSM has two equivalent views:

  RNN view (sequential):
    for k in range(L):
        x[k] = Ā x[k-1] + B̄ u[k]
        y[k] = C x[k]

  CNN view (parallel):
    K = [C B̄, C Ā B̄, C Ā² B̄, ..., C Ā^{L-1} B̄]
    y = conv(u, K)

The CNN view allows parallel training via FFT: O(L log L).
The RNN view allows O(1)-per-step inference for generation.

S4's key contribution: compute the kernel K *efficiently* via
the DPLR parameterization and Cauchy kernels.
""")

# Demonstrate equivalence
from s4_lib.hippo import s4d_inv_init
from s4_lib.kernels import kernel_diagonal, fft_conv

N_demo = 8
L_demo = 32
Lambda_d = torch.from_numpy(s4d_inv_init(N_demo))
B_d = torch.randn(N_demo, dtype=torch.cfloat) * 0.1
C_d = torch.randn(N_demo, dtype=torch.cfloat) * 0.1
dt_d = torch.tensor(0.05)

# CNN mode: compute kernel, convolve
K = kernel_diagonal(Lambda_d, B_d, C_d, dt_d, L_demo, method="zoh")
u = torch.randn(1, L_demo)
y_cnn = fft_conv(u, K)

# RNN mode: step-by-step
Ab_d = torch.exp(Lambda_d * dt_d)
Bb_d = B_d * (Ab_d - 1.0) / Lambda_d
state = torch.zeros(N_demo, dtype=torch.cfloat)
y_rnn = []
for k in range(L_demo):
    state = Ab_d * state + Bb_d * u[0, k]
    y_rnn.append((C_d * state).sum().real.item())
y_rnn = torch.tensor(y_rnn).unsqueeze(0)

error = (y_cnn - y_rnn).abs().max().item()
print(f"CNN vs RNN max absolute error: {error:.2e}")
print(f"They are equivalent: {error < 1e-4}")
print()
print("✓ Tutorial complete!")
