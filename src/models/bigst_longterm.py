"""BigST long-term pretraining branch — ``linear_transformer``.

Ported from ``new_models/BigST/preprocess/model.py`` (VLDB 2024 long-term
feature extractor). This is a *separate, offline* model from the main
``BigST`` model in ``src/models/bigst.py``: per explicit project decision, the
long-term branch is run once as a standalone preprocessing step
(``scripts/preprocess_bigst_features.py``) rather than wired into the live
training adapter path. Shared random-feature math lives in
``src/models/bigst_common.py`` and is imported here rather than duplicated.

forward(x) -> (prediction, feat)
    x:          (B, N, T, D)   T = input_length (a long window; this
                                framework uses 288 = one day at 5-min
                                resolution by default, not the original
                                paper's 2016 = one week, since tsl-loaded
                                METR-LA/PEMS-BAY are shorter than the
                                paper's long-term corpus)
    prediction: (B, N, output_length)
    feat:       (B, N, nhid)   flattened per-node hidden representation —
                                this is the artifact
                                ``scripts/preprocess_bigst_features.py`` saves
                                to disk for the main BigST model's
                                ``use_long=True`` path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bigst_common import LinearizedAttention

__all__ = ["LinearTransformer"]


class LinearTransformer(nn.Module):
    def __init__(
        self,
        input_length: int,
        output_length: int,
        in_dim: int,
        num_nodes: int,
        nhid: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.tau = 1.0
        self.layer_num = 3
        self.random_feature_dim = nhid * 2

        self.use_residual = True
        self.use_bn = False
        self.use_act = True

        self.dropout = dropout
        self.activation = nn.ReLU()

        self.fc_convs = nn.ModuleList()
        self.transformer_layer = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.context_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=nhid, kernel_size=(12, 1), stride=(12, 1)
        )

        self.temporal_embedding = nn.Parameter(
            torch.empty(int(input_length / 12), nhid), requires_grad=True
        )  # (C, nhid)
        nn.init.xavier_uniform_(self.temporal_embedding)

        for _ in range(self.layer_num):
            self.transformer_layer.append(
                LinearizedAttention(nhid, nhid, self.dropout, self.random_feature_dim, self.tau)
            )
            self.bn.append(nn.LayerNorm(nhid))

        self.regression_layer = nn.Linear(nhid, output_length)

    def forward(self, x: torch.Tensor):
        # input: (B, N, T, D), T must be a multiple of 12 (context_conv kernel/stride)
        B, N, T, D = x.size()
        pe = self.temporal_embedding.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, T/12, nhid)

        x = x.reshape(B * N, T, D)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B*N, T, D) -> (B*N, D, T, 1)

        # convolution layer
        x = self.context_conv(x)  # (B*N, D, T, 1) -> (B*N, nhid, T/12, 1)
        x = x.squeeze(-1)  # (B*N, nhid, T/12)

        # temporal embedding layer
        x = x.permute(0, 2, 1)  # (B*N, T/12, nhid)
        x = x + pe  # (B*N, T/12, nhid)

        # linearized attention
        for num in range(self.layer_num):
            residual = x  # (B*N, T/12, nhid)
            x = self.transformer_layer[num](x)  # (B*N, T/12, nhid)
            x = self.bn[num](x)
            x = x + residual  # (B*N, T/12, nhid)

        x = self.activation(x)  # (B*N, T/12, nhid)
        x = x[:, -1, :]
        feat = x.view(B, N, -1)  # (B, N, nhid)
        x = self.regression_layer(feat)  # (B, N, output_length)
        return x, feat


# Backwards/forwards-compatible alias matching the original repo's class name.
linear_transformer = LinearTransformer
