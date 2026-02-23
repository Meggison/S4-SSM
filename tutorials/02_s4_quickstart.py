#!/usr/bin/env python3
"""
Tutorial 2 — S4 Quick Start
============================

Shows how to use the library in < 5 minutes:
  • Import a layer or model
  • Run a forward pass
  • Switch between CNN and RNN modes
  • Set up the optimizer with SSM-specific learning rates

Run:  python tutorials/02_s4_quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from s4_lib import S4DLayer, S4Layer, S4SequenceModel, get_ssm_param_groups

# ── 1. Single S4D layer ──────────────────────────────────────────────────
print("=" * 60)
print("1. SINGLE S4D LAYER")
print("=" * 60)

layer = S4DLayer(d_model=32, d_state=16)
x = torch.randn(2, 100, 32)          # (batch=2, length=100, features=32)
y = layer(x)                          # CNN mode
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Same shape ✓: {x.shape == y.shape}")
print()

# ── 2. RNN step mode ─────────────────────────────────────────────────────
print("=" * 60)
print("2. RNN STEP MODE")
print("=" * 60)

state = layer.init_state(batch_size=2)
print(f"Initial state shape: {state.shape}  dtype: {state.dtype}")

u_t = torch.randn(2, 32)              # single timestep input
y_t, state = layer.step(u_t, state)
print(f"Step input:   {u_t.shape}")
print(f"Step output:  {y_t.shape}")
print(f"New state:    {state.shape}")
print()

# ── 3. Full S4 layer (DPLR) ─────────────────────────────────────────────
print("=" * 60)
print("3. FULL S4 LAYER (DPLR)")
print("=" * 60)

s4 = S4Layer(d_model=32, d_state=16)
y_s4 = s4(x)
print(f"S4 output shape: {y_s4.shape}")
n_params = sum(p.numel() for p in s4.parameters())
print(f"S4 parameters:   {n_params:,}")
print()

# ── 4. Complete sequence model ────────────────────────────────────────────
print("=" * 60)
print("4. COMPLETE SEQUENCE MODEL")
print("=" * 60)

model = S4SequenceModel(
    d_input=1,
    d_model=64,
    d_output=10,
    n_layers=4,
    d_state=32,
    task="classification",
)

x_seq = torch.randn(4, 1024, 1)      # e.g. 1024 pixel values
logits = model(x_seq)
print(f"Input:   {x_seq.shape}  (batch=4, length=1024, features=1)")
print(f"Logits:  {logits.shape}  (batch=4, classes=10)")

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
print()

# ── 5. Optimizer with SSM param groups ────────────────────────────────────
print("=" * 60)
print("5. OPTIMIZER SETUP")
print("=" * 60)

param_groups = get_ssm_param_groups(
    model,
    ssm_lr=1e-3,      # SSM params: lr=0.001, wd=0
    default_lr=1e-3,   # Other params: lr=0.001, wd=0.01
    default_wd=0.01,
)
optimizer = torch.optim.AdamW(param_groups)

n_ssm = len(param_groups[0]["params"])
n_other = len(param_groups[1]["params"])
print(f"SSM parameters:     {n_ssm} tensors  (lr=0.001, wd=0)")
print(f"Other parameters:   {n_other} tensors  (lr=0.001, wd=0.01)")
print()

# ── 6. Quick training step ────────────────────────────────────────────────
print("=" * 60)
print("6. QUICK TRAINING STEP")
print("=" * 60)

target = torch.randint(0, 10, (4,))
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")
print("Backward pass + optimizer step ✓")
print()
print("✓ Quick start complete!")
