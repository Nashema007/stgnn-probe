"""BigST: linear-complexity spatial convolution via random-feature attention.

Ported from ``new_models/BigST/model.py`` (Han et al., "BigST: Linear Complexity
Spatio-Temporal Graph Neural Network for Traffic Forecasting", VLDB 2024).

Native shapes (node-major, NOT the framework's canonical channel-major layout):
    forward(x, feat=None)
        x:    (B, N, T, D)        D = in_dim channels: [speed, tod_fraction, dow_int]
        feat: (B, N, F) optional  long-term feature embedding (only if use_long=True)
    returns (prediction, spatial_loss)
        prediction:   (B, N, output_length)
        spatial_loss: scalar tensor if use_spatial else 0

Bug fix vs. the original repo: ``input_emb_layer`` was sized with
``args.output_length * args.in_dim`` input channels, but ``forward()`` flattens
the actual input window (``x.contiguous().view(B, N, -1)`` over the real T it
receives) — i.e. the *input* window length, not the output horizon. The
original repo always trained with input_length == output_length == 12 so this
never surfaced. This framework trains the same architecture at multiple output
horizons (e.g. 6) while the input window stays fixed at 12, so the two diverge
and the bug must be fixed: use ``args.input_length * args.in_dim`` instead.

BUGFIX-2026-07-02: ``linear_conv``/``regression_layer`` were sized as
``hid_dim*4``, but ``forward()`` concatenates channels of width
``hid_dim + node_dim + time_dim*2`` — only equal to ``hid_dim*4`` if
``node_dim == time_dim == hid_dim``. Added an explicit ``__init__``-time
assertion instead of computing the concat width dynamically (the lower-risk
fix — avoids touching the residual/regression-layer plumbing). This is a
no-op under the shipped ``bigst_base.yaml`` (all three already equal 32) and
only changes behavior for a hyperparameter combination that previously
crashed deep inside ``LinearizedConv``'s Conv2d anyway — same
``ValueError``-instead-of-``RuntimeError`` trade as the other 2026-07-02
fixes in this codebase (tag: BUGFIX-2026-07-02). Also removed the dead,
never-read ``self.supports_len`` (vestigial from a GCN-family sibling model;
BigST's linear-complexity design intentionally has no explicit
multi-support graph-diffusion channel concat).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from .bigst_common import LinearizedConv, spatial_loss

__all__ = ["BigST", "make_bigst"]


class BigST(nn.Module):
    def __init__(
        self,
        args: Any,
        supports: list[torch.Tensor] | None = None,
        edge_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.tau = args.tau
        self.num_layers = args.num_layers
        self.random_feature_dim = args.random_feature_dim

        self.use_residual = args.use_residual
        self.use_bn = args.use_bn
        self.use_spatial = args.use_spatial
        self.use_long = args.use_long

        self.dropout = args.dropout
        self.activation = nn.ReLU()
        self.supports = supports
        self.edge_indices = edge_indices

        self.time_num = args.time_num
        self.week_num = args.week_num

        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(args.num_nodes, args.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, args.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, args.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        # NOTE bug fix: use args.input_length (the actual input window T), not
        # args.output_length — see module docstring.
        self.input_emb_layer = nn.Conv2d(
            args.input_length * args.in_dim, args.hid_dim, kernel_size=(1, 1), bias=True
        )

        w_in = args.node_dim + args.time_dim * 2
        self.W_1 = nn.Conv2d(w_in, args.hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(w_in, args.hid_dim, kernel_size=(1, 1), bias=True)

        # BUGFIX-2026-07-02: forward() concatenates input_emb(hid_dim) + node_emb(node_dim) +
        # time_emb(time_dim) + week_emb(time_dim); linear_conv/regression_layer
        # below are sized as hid_dim*4, which is only correct if these three
        # dims coincide.
        if not (args.node_dim == args.hid_dim and args.time_dim == args.hid_dim):
            raise ValueError(
                "BigST requires node_dim == time_dim == hid_dim (layer sizing in "
                f"__init__ assumes a uniform hid_dim*4 concat width); got "
                f"hid_dim={args.hid_dim}, node_dim={args.node_dim}, time_dim={args.time_dim}."
            )

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.linear_conv.append(
                LinearizedConv(
                    args.hid_dim * 4,
                    args.hid_dim * 4,
                    self.dropout,
                    self.tau,
                    self.random_feature_dim,
                )
            )
            self.bn.append(nn.LayerNorm(args.hid_dim * 4))

        if self.use_long:
            self.long_feat_dim = args.long_feat_dim
            self.regression_layer = nn.Conv2d(
                args.hid_dim * 4 * 2 + args.long_feat_dim,
                args.output_length,
                kernel_size=(1, 1),
                bias=True,
            )
        else:
            self.regression_layer = nn.Conv2d(
                args.hid_dim * 4 * 2, args.output_length, kernel_size=(1, 1), bias=True
            )

    def forward(self, x: torch.Tensor, feat: torch.Tensor | None = None):
        # input: (B, N, T, D)
        B, N, T, D = x.size()

        # NOTE: original used `.type(torch.LongTensor)`, which always builds a
        # CPU long tensor regardless of `x`'s device. `.long()` keeps the
        # index tensor on the same device as the input (same fix as D2STGNN).
        time_emb = self.time_emb_layer[(x[:, :, -1, 1] * self.time_num).long()]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).long()]

        # input embedding
        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1)
        node_emb = node_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)

        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)

        # linearized spatial convolution
        x_pool = [x]  # (B, dim*4, N, 1)
        node_vec1 = self.W_1(x_g)  # (B, dim, N, 1)
        node_vec2 = self.W_2(x_g)  # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        node_vec1_prime = node_vec2_prime = None
        for i in range(self.num_layers):
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)

            if self.use_residual:
                x = x + residual

            if self.use_bn:
                x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)  # (B, dim*4, N, 1)

        x = self.activation(x)  # (B, dim*4, N, 1)

        if self.use_long:
            if feat is None:
                raise ValueError("BigST(use_long=True) requires a `feat` tensor at forward time.")
            if feat.shape[-1] != self.long_feat_dim:
                raise ValueError(
                    f"BigST(use_long=True) expected feat dim {self.long_feat_dim}, "
                    f"got {feat.shape[-1]}."
                )
            feat_emb = feat.permute(0, 2, 1).unsqueeze(-1)  # (B, F, N, 1)
            x = torch.cat([x, feat_emb], dim=1)
            x = self.regression_layer(x)  # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)
        else:
            x = self.regression_layer(x)  # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)

        if self.use_spatial:
            assert node_vec1_prime is not None and node_vec2_prime is not None
            assert self.supports is not None and self.edge_indices is not None
            s_loss = spatial_loss(
                node_vec1_prime, node_vec2_prime, self.supports, self.edge_indices
            )
            return x, s_loss
        else:
            return x, 0


def make_bigst(
    num_nodes: int,
    in_dim: int = 3,
    hid_dim: int = 32,
    node_dim: int = 32,
    time_dim: int = 32,
    num_layers: int = 3,
    random_feature_dim: int = 64,
    input_length: int = 12,
    output_length: int = 12,
    tau: float = 1.0,
    dropout: float = 0.3,
    use_residual: bool = True,
    use_bn: bool = True,
    use_spatial: bool = False,
    use_long: bool = False,
    long_feat_dim: int | None = None,
    time_num: int = 288,
    week_num: int = 7,
    supports: list[torch.Tensor] | None = None,
    edge_indices: torch.Tensor | None = None,
) -> BigST:
    """Build a ``BigST`` model from plain keyword arguments.

    Hides the argparse-namespace plumbing the original repo's constructor
    expects and keeps that compatibility concern outside the model class.

    ``use_long`` defaults to False so the model (and its adapter) work even
    when no precomputed long-term feature array exists. Pass
    ``use_long=True`` only when the caller has a precomputed ``(N, F)``
    long-term feature array to feed as ``feat`` at forward time (see
    ``scripts/preprocess_bigst_features.py`` / ``src/models/bigst_longterm.py``).
    ``long_feat_dim`` defines ``F`` and defaults to ``hid_dim``.

    ``supports``/``edge_indices`` are only consumed when ``use_spatial=True``
    (they parameterise the optional spatial regularisation loss); pass dense
    ``(N, N)`` torch tensors for ``supports`` (e.g.
    ``data.adjacency.build_supports(adj_np, "doubletransition")`` converted to
    tensors) and an ``(E, 2)`` long tensor for ``edge_indices``.
    """
    args = SimpleNamespace(
        tau=tau,
        num_layers=num_layers,
        random_feature_dim=random_feature_dim,
        use_residual=use_residual,
        use_bn=use_bn,
        use_spatial=use_spatial,
        use_long=use_long,
        long_feat_dim=long_feat_dim if long_feat_dim is not None else hid_dim,
        dropout=dropout,
        time_num=time_num,
        week_num=week_num,
        num_nodes=num_nodes,
        node_dim=node_dim,
        time_dim=time_dim,
        input_length=input_length,
        output_length=output_length,
        in_dim=in_dim,
        hid_dim=hid_dim,
    )
    return BigST(args, supports=supports, edge_indices=edge_indices)
