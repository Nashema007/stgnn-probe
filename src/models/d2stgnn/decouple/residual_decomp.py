"""Residual decomposition module used by both the diffusion and inherent blocks.

Ported verbatim from D2STGNN (Shao et al., VLDB 2022),
``new_models/D2STGNN/models/decouple/residual_decomp.py``.
"""

import torch.nn as nn


class ResidualDecomp(nn.Module):
    """Residual decomposition."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u
