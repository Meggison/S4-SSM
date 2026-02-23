#!/usr/bin/env python3
"""
Tutorial 3 — Sequence Classification
=====================================

Trains an S4 model on a **synthetic delayed-XOR task** to show
end-to-end training on a long-range dependency problem.

Task: given a binary sequence of length L with two marked positions,
predict the XOR of the values at those positions.
This requires the model to remember information across the full sequence.

Run:  python tutorials/03_sequence_classification.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from s4_lib import S4SequenceModel, get_ssm_param_groups


# ── Dataset: Delayed XOR ──────────────────────────────────────────────────

def make_delayed_xor_dataset(n_samples: int, seq_len: int = 256):
    """
    Generate a delayed-XOR dataset.

    Each sample is a sequence of length `seq_len` with 3 channels:
      - Channel 0: random binary noise (0 or 1)
      - Channel 1: marker at position p1 (1.0, else 0)
      - Channel 2: marker at position p2 (1.0, else 0)

    Label = XOR of the noise values at positions p1 and p2.
    """
    x = torch.zeros(n_samples, seq_len, 3)
    y = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        noise = torch.randint(0, 2, (seq_len,)).float()
        x[i, :, 0] = noise

        # Pick two random positions (at least 50 apart for difficulty)
        p1 = torch.randint(0, seq_len // 3, (1,)).item()
        p2 = torch.randint(2 * seq_len // 3, seq_len, (1,)).item()
        x[i, p1, 1] = 1.0
        x[i, p2, 2] = 1.0

        # Label: XOR of noise at p1 and p2
        y[i] = int(noise[p1].item()) ^ int(noise[p2].item())

    return x, y


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 256
    batch_size = 32
    epochs = 15
    lr = 1e-3

    print("Generating delayed-XOR dataset...")
    x_train, y_train = make_delayed_xor_dataset(2000, seq_len)
    x_test, y_test = make_delayed_xor_dataset(500, seq_len)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size
    )

    print(f"Train: {len(x_train)}, Test: {len(x_test)}, Seq length: {seq_len}")
    print(f"Baseline (random): 50% accuracy\n")

    # Build model
    model = S4SequenceModel(
        d_input=3,        # 3 channels: noise + 2 markers
        d_model=64,
        d_output=2,       # binary classification
        n_layers=3,
        d_state=32,
        dropout=0.1,
        task="classification",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # Optimizer with SSM-specific groups
    param_groups = get_ssm_param_groups(model, ssm_lr=lr, default_lr=lr)
    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(-1) == yb).sum().item()
            total += xb.size(0)

        train_acc = correct / total

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(-1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        test_acc = correct / total

        print(
            f"Epoch {epoch:2d}/{epochs}  "
            f"Loss: {total_loss / len(x_train):.4f}  "
            f"Train acc: {train_acc:.1%}  "
            f"Test acc: {test_acc:.1%}"
        )

    print(f"\nFinal test accuracy: {test_acc:.1%}")
    if test_acc > 0.7:
        print("✓ Model learned the long-range XOR dependency!")
    else:
        print("△ Try more epochs or larger model for better accuracy.")


if __name__ == "__main__":
    main()
