# Structured State Spaces for Sequence Modeling

A self-contained PyTorch implementation of **S4** and **S4D** from:

- **S4**: [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) (Gu, Goel, Ré — ICLR 2022)
- **S4D**: [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893) (Gu, Gupta, Goel, Ré — NeurIPS 2022)

## What is S4?

S4 is a sequence model based on **state space models (SSMs)** — linear dynamical systems that map inputs to outputs through a hidden state:

```
x'(t) = A x(t) + B u(t)    (state equation)
y(t)  = C x(t) + D u(t)    (output equation)
```

Key innovations:
- **HiPPO initialization** — a mathematically principled state matrix that enables long-range dependency modeling
- **DPLR parameterization** — decomposes the state matrix to enable O((N+L) log(N+L)) kernel computation via Cauchy kernels
- **CNN/RNN duality** — trains as a CNN (parallel, efficient) but can run as an RNN at inference (O(1) per step)

**S4D** simplifies S4 by using a diagonal state matrix (initialized from HiPPO eigenvalues), reducing the kernel to ~2 lines of code while matching S4's performance.

## Installation

```bash
# Clone and install
git clone https://github.com/Meggison/S4-Implementation.git
cd S4-Implementation
pip install -e ".[dev]"
```

## Quick Start

```python
from s4_lib import S4DLayer, S4SequenceModel, get_ssm_param_groups

# Single S4D layer
layer = S4DLayer(d_model=128, d_state=64)
y = layer(x)   # x: (batch, length, 128)

# Full classification model (e.g. sequential CIFAR-10)
model = S4SequenceModel(
    d_input=1, d_model=128, d_output=10,
    n_layers=4, task="classification",
)
logits = model(x)   # x: (batch, 1024, 1) → logits: (batch, 10)

# Optimizer with SSM-specific learning rates (important!)
param_groups = get_ssm_param_groups(model, ssm_lr=1e-3)
optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
```

## Library Structure

```
s4_lib/
  __init__.py          # Public API
  hippo.py             # HiPPO matrix constructions (LegS, LegT, FouT)
  discretization.py    # Bilinear & ZOH discretization
  kernels.py           # Cauchy kernel, DPLR kernel, diagonal kernel, FFT conv
  s4_layer.py          # Full S4 layer (DPLR parameterization)
  s4d_layer.py         # S4D layer (diagonal) + S4DBlock
  model.py             # S4SequenceModel — stacked blocks with encoder/decoder
  utils.py             # Optimizer param groups, DropoutNd
tutorials/
  01_ssm_basics.py               # SSM concepts, HiPPO, discretization, CNN/RNN duality
  02_s4_quickstart.py            # Import, forward pass, RNN step, optimizer setup
  03_sequence_classification.py  # Train on synthetic delayed-XOR task
  04_time_series.py              # Multivariate time-series forecasting
tests/
  test_s4.py           # Unit tests (pytest)
```

## Tutorials

```bash
python tutorials/01_ssm_basics.py               # Learn the theory
python tutorials/02_s4_quickstart.py             # Use the library
python tutorials/03_sequence_classification.py   # Train a classifier
python tutorials/04_time_series.py               # Forecast time series
```

## Running Tests

```bash
pytest tests/test_s4.py -v
```

## Key Implementation Details

- **HiPPO-LegS** matrix used for initialization (captures long-range dependencies via Legendre polynomial projections)
- **Dual compute modes**: CNN mode for training (FFT convolution), RNN mode for generation (step-by-step recurrence)
- **SSM-specific optimizer groups**: A, B, and Δ parameters use lr=0.001 with weight_decay=0 (as recommended in the paper)
- **S4D-Lin initialization** by default: eigenvalues of the HiPPO matrix, which is simpler than DPLR while performing comparably

## Citation

If you use this implementation, please cite the original papers:

```bibtex
@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle={The International Conference on Learning Representations (ICLR)},
  year={2022}
}

@article{gu2022s4d,
  title={On the Parameterization and Initialization of Diagonal State Space Models},
  author={Gu, Albert and Gupta, Ankit and Goel, Karan and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
