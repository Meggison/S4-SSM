#!/usr/bin/env python3
"""
Tutorial 4 — Time-Series Forecasting
=====================================

Uses S4 to forecast a synthetic multi-variate time series
(sum of sinusoids with noise).

Run:  python tutorials/04_time_series.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from s4_lib import S4SequenceModel, get_ssm_param_groups


# ── Dataset: noisy sinusoids ──────────────────────────────────────────────

def make_sinusoid_data(n_samples: int, seq_len: int, n_features: int = 3):
    """
    Generate multivariate sinusoidal sequences.

    Each feature is a sum of 2-3 sinusoids with random frequencies/phases
    plus Gaussian noise.  The task: given x[0:T], predict x[1:T+1].
    """
    t = torch.linspace(0, 4 * torch.pi, seq_len + 1).unsqueeze(0).unsqueeze(-1)
    t = t.expand(n_samples, -1, n_features)

    freqs = 0.5 + 2.0 * torch.rand(n_samples, 1, n_features)
    phases = 2 * torch.pi * torch.rand(n_samples, 1, n_features)
    freqs2 = 1.0 + 3.0 * torch.rand(n_samples, 1, n_features)
    phases2 = 2 * torch.pi * torch.rand(n_samples, 1, n_features)

    signal = torch.sin(freqs * t + phases) + 0.5 * torch.sin(freqs2 * t + phases2)
    signal = signal + 0.05 * torch.randn_like(signal)

    x = signal[:, :-1, :]   # input:  [0, T-1]
    y = signal[:, 1:, :]    # target: [1, T]
    return x, y


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 200
    n_features = 3
    batch_size = 32
    epochs = 20
    lr = 1e-3

    print("Generating sinusoidal time-series data...")
    x_train, y_train = make_sinusoid_data(1000, seq_len, n_features)
    x_test, y_test = make_sinusoid_data(200, seq_len, n_features)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size
    )

    print(f"Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"Sequence length: {seq_len}, Features: {n_features}\n")

    # Build model (regression mode — per-timestep output)
    model = S4SequenceModel(
        d_input=n_features,
        d_model=64,
        d_output=n_features,
        n_layers=3,
        d_state=32,
        dropout=0.0,
        task="regression",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    param_groups = get_ssm_param_groups(model, ssm_lr=lr, default_lr=lr)
    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                test_loss += criterion(preds, yb).item() * xb.size(0)

        print(
            f"Epoch {epoch:2d}/{epochs}  "
            f"Train MSE: {total_loss / len(x_train):.6f}  "
            f"Test MSE: {test_loss / len(x_test):.6f}"
        )

    print(f"\nFinal test MSE: {test_loss / len(x_test):.6f}")
    print("✓ Forecasting tutorial complete!")


if __name__ == "__main__":
    main()
