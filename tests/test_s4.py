"""
Tests for the s4_lib package.

Run:  pytest tests/test_s4.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest

# ── HiPPO tests ──────────────────────────────────────────────────────────

class TestHiPPO:
    def test_legs_shape(self):
        from s4_lib.hippo import make_hippo_legs
        A = make_hippo_legs(64)
        assert A.shape == (64, 64)

    def test_legs_lower_triangular(self):
        from s4_lib.hippo import make_hippo_legs
        A = make_hippo_legs(32)
        assert np.allclose(A, np.tril(A))

    def test_legs_stable(self):
        """All eigenvalues should have negative real parts."""
        from s4_lib.hippo import make_hippo_legs
        A = make_hippo_legs(64)
        eigs = np.linalg.eigvals(A)
        assert (eigs.real < 0).all()

    def test_hippo_b_shape(self):
        from s4_lib.hippo import make_hippo_b
        for method in ["legs", "legt", "fout"]:
            B = make_hippo_b(32, method)
            assert B.shape == (32, 1)

    def test_nplr_shapes(self):
        from s4_lib.hippo import make_hippo_legs, make_hippo_b, nplr
        A = make_hippo_legs(16)
        B = make_hippo_b(16)
        Lambda, P, B_t, V = nplr(A, B)
        assert Lambda.shape == (16,)
        assert P.shape == (16,)
        assert B_t.shape == (16,)
        assert V.shape == (16, 16)

    def test_diagonal_init(self):
        from s4_lib.hippo import diagonal_init
        eigs = diagonal_init(32, "legs")
        assert eigs.shape == (32,)
        assert np.issubdtype(eigs.dtype, np.complexfloating)

    def test_s4d_inv_init(self):
        from s4_lib.hippo import s4d_inv_init
        eigs = s4d_inv_init(16)
        assert eigs.shape == (16,)
        assert np.allclose(eigs.real, -0.5)
        assert np.allclose(eigs.imag, np.pi * np.arange(16))


# ── Discretization tests ─────────────────────────────────────────────────

class TestDiscretization:
    def test_bilinear_stability(self):
        """Bilinear should map stable eigenvalues to |λ̄| < 1."""
        from s4_lib.discretization import bilinear
        A = torch.tensor([-1.0 + 2j, -3.0 - 1j], dtype=torch.cfloat)
        B = torch.ones(2, dtype=torch.cfloat)
        Ab, Bb = bilinear(A, B, torch.tensor(0.01))
        assert (Ab.abs() < 1).all()

    def test_zoh_stability(self):
        from s4_lib.discretization import zoh
        A = torch.tensor([-1.0 + 2j, -3.0 - 1j], dtype=torch.cfloat)
        B = torch.ones(2, dtype=torch.cfloat)
        Ab, Bb = zoh(A, B, torch.tensor(0.01))
        assert (Ab.abs() < 1).all()


# ── Kernel tests ─────────────────────────────────────────────────────────

class TestKernels:
    def test_diagonal_kernel_shape(self):
        from s4_lib.kernels import kernel_diagonal
        N, L = 8, 32
        Lambda = torch.randn(N, dtype=torch.cfloat) - 1.0
        B = torch.randn(N, dtype=torch.cfloat)
        C = torch.randn(N, dtype=torch.cfloat)
        K = kernel_diagonal(Lambda, B, C, torch.tensor(0.01), L)
        assert K.shape == (L,)
        assert K.dtype == torch.float32

    def test_diagonal_fft_matches_direct(self):
        from s4_lib.kernels import kernel_diagonal, kernel_diagonal_fft
        N, L = 8, 64
        # Use well-damped eigenvalues so |Ā^L| ≈ 0 (the FFT method
        # assumes the geometric series has effectively converged).
        Lambda = torch.randn(N, dtype=torch.cfloat) - 5.0
        B = torch.randn(N, dtype=torch.cfloat)
        C = torch.randn(N, dtype=torch.cfloat)
        dt = torch.tensor(0.1)
        K1 = kernel_diagonal(Lambda, B, C, dt, L)
        K2 = kernel_diagonal_fft(Lambda, B, C, dt, L)
        assert torch.allclose(K1, K2, atol=1e-3)

    def test_fft_conv_shape(self):
        from s4_lib.kernels import fft_conv
        u = torch.randn(2, 3, 64)
        K = torch.randn(3, 64)
        y = fft_conv(u, K)
        assert y.shape == (2, 3, 64)

    def test_cnn_rnn_equivalence(self):
        """CNN-mode kernel conv should match step-by-step RNN."""
        from s4_lib.hippo import s4d_inv_init
        from s4_lib.kernels import kernel_diagonal, fft_conv

        N, L = 8, 32
        Lambda = torch.from_numpy(s4d_inv_init(N))
        B = torch.randn(N, dtype=torch.cfloat) * 0.1
        C = torch.randn(N, dtype=torch.cfloat) * 0.1
        dt = torch.tensor(0.05)

        K = kernel_diagonal(Lambda, B, C, dt, L, method="zoh")
        u = torch.randn(1, L)
        y_cnn = fft_conv(u, K)

        Ab = torch.exp(Lambda * dt)
        Bb = B * (Ab - 1.0) / Lambda
        state = torch.zeros(N, dtype=torch.cfloat)
        y_rnn = []
        for k in range(L):
            state = Ab * state + Bb * u[0, k]
            y_rnn.append((C * state).sum().real.item())
        y_rnn = torch.tensor(y_rnn).unsqueeze(0)

        assert torch.allclose(y_cnn, y_rnn, atol=1e-4)


# ── Layer tests ──────────────────────────────────────────────────────────

class TestLayers:
    def test_s4d_forward_shape(self):
        from s4_lib.s4d_layer import S4DLayer
        layer = S4DLayer(d_model=16, d_state=8)
        x = torch.randn(2, 32, 16)
        y = layer(x)
        assert y.shape == (2, 32, 16)

    def test_s4d_step_shape(self):
        from s4_lib.s4d_layer import S4DLayer
        layer = S4DLayer(d_model=16, d_state=8)
        state = layer.init_state(2)
        u_t = torch.randn(2, 16)
        y_t, new_state = layer.step(u_t, state)
        assert y_t.shape == (2, 16)
        assert new_state.shape == state.shape

    def test_s4_forward_shape(self):
        from s4_lib.s4_layer import S4Layer
        layer = S4Layer(d_model=16, d_state=8)
        x = torch.randn(2, 32, 16)
        y = layer(x)
        assert y.shape == (2, 32, 16)

    def test_s4d_backward(self):
        from s4_lib.s4d_layer import S4DLayer
        layer = S4DLayer(d_model=16, d_state=8)
        x = torch.randn(2, 32, 16, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None

    def test_s4d_block_shape(self):
        from s4_lib.s4d_layer import S4DBlock
        block = S4DBlock(d_model=16, d_state=8)
        x = torch.randn(2, 32, 16)
        y = block(x)
        assert y.shape == (2, 32, 16)


# ── Model tests ──────────────────────────────────────────────────────────

class TestModel:
    def test_classification(self):
        from s4_lib.model import S4SequenceModel
        model = S4SequenceModel(
            d_input=1, d_model=16, d_output=10,
            n_layers=2, d_state=8, task="classification",
        )
        x = torch.randn(2, 64, 1)
        logits = model(x)
        assert logits.shape == (2, 10)

    def test_regression(self):
        from s4_lib.model import S4SequenceModel
        model = S4SequenceModel(
            d_input=3, d_model=16, d_output=3,
            n_layers=2, d_state=8, task="regression",
        )
        x = torch.randn(2, 64, 3)
        y = model(x)
        assert y.shape == (2, 64, 3)

    def test_backward(self):
        from s4_lib.model import S4SequenceModel
        model = S4SequenceModel(
            d_input=1, d_model=16, d_output=2,
            n_layers=2, d_state=8, task="classification",
        )
        x = torch.randn(4, 32, 1)
        target = torch.randint(0, 2, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        # Check all params got gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
