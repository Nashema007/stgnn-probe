"""Spatial-Temporal Adaptive WaveNet (STAWnet).

Ported from:
    https://github.com/CYBruce/STAWnet

Original paper:
    Hang Tan, Guangliang Cheng, Shupeng Wang, Yuanlong Ge, Weixiong Rao.
    "Spatial-Temporal Adaptive Graph Convolutional Networks for Traffic Flow
    Forecasting." IEEE Access, 2021.

Changes from the original:
  - Replaced nn.Conv1d (with 2D kernel, applied to 4D input) with nn.Conv2d
    throughout gate_convs, residual_convs, and skip_convs. Not a behavior
    change: on PyTorch <=1.9.1, Conv1d.forward had no input-rank check, so a
    (1, 1) kernel (always a 4D weight regardless of declared "1D"-ness) ran
    against a 4D tensor identically to the equivalent Conv2d (verified
    bit-identical on torch 1.9.1). PyTorch added an explicit input.dim() in
    {2, 3} guard to Conv1d.forward between 1.9.1 and 1.13.1, which rejects
    this same call on every version since — see src/models/gwn_v2.py's
    docstring for the full investigation (same fix, same root cause, ported
    from a sibling WaveNet-family repo).
  - Removed unused imports (Variable, sys, numpy).
  - Removed unused nconv class.
  - Replaced is_test flag with return_attention for clarity.
  - Replaced try/except skip with isinstance(skip, int) guard.

BUGFIX-2026-07-02 (tag: BUGFIX-2026-07-02) — ACTUAL BEHAVIOR CHANGE, unlike
the other 2026-07-02 fixes across this codebase (which are all no-ops under
their shipped base configs): previously, ``graph_attention=True,
adaptive_adjacency_matrix=False`` silently built real ``self.gat`` attention
weights but never called them in ``forward()`` (it fell back to a plain 1x1
``residual_convs`` conv, i.e. no spatial mixing at all despite requesting
graph attention) — and ``self.supports`` was accepted but never read
anywhere. Now, that combination builds a fixed (non-learnable) node
embedding via SVD of ``supports`` (mirrors ``gwn_v2.GWNv2._init_nodevecs``)
and ``forward()`` actually runs graph attention with it. The shipped
``stawnet_base.yaml`` sets ``adaptive_adjacency_matrix: true`` and the sweep
never varies these flags, so no existing results are affected — but if you
ever see this config combination in a run or sweep predating this commit,
its outputs are not comparable to runs after it (it went from "silently no
spatial mixing" to "real graph attention"), and should be re-run rather than
treated as a regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GraphAttention(nn.Module):
    """Self-attention over nodes using learnable node embeddings."""

    def __init__(
        self,
        c_in,
        c_out,
        dropout,
        d=16,
        emb_length=0,
        use_node_embedding_only=False,
        use_node_embedding=False,
    ):
        super().__init__()
        self.d = d
        self.use_node_embedding_only = use_node_embedding_only
        self.use_node_embedding = use_node_embedding
        self.mlp = _Linear(c_in * 2, c_out)
        self.dropout = dropout
        self.emb_length = emb_length

        if use_node_embedding_only:
            self.qm = _Linear(self.emb_length, d)
            self.km = _Linear(self.emb_length, d)
        elif use_node_embedding:
            self.qm = _Linear(c_in, d)
            self.km = _Linear(c_in, d)
        else:
            self.qm = _Linear(c_in + self.emb_length, d)
            self.km = _Linear(c_in + self.emb_length, d)

    def forward(self, x, embedding):
        # x: (B, C, N, T)
        # embedding: (emb_length, N)
        out = [x]
        emb = embedding.repeat(x.shape[0], x.shape[-1], 1, 1)  # (B, T, emb_length, N)
        emb = emb.permute(0, 2, 3, 1)  # (B, emb_length, N, T)

        if self.use_node_embedding_only:
            query = self.qm(emb).permute(0, 3, 2, 1)  # (B, T, N, d)
            key = self.km(emb).permute(0, 3, 2, 1)
        elif self.use_node_embedding:
            query = self.qm(x).permute(0, 3, 2, 1)
            key = self.km(x).permute(0, 3, 2, 1)
        else:
            x_emb = torch.cat([x, emb], dim=1)  # (B, C+emb, N, T)
            query = self.qm(x_emb).permute(0, 3, 2, 1)
            key = self.km(x_emb).permute(0, 3, 2, 1)

        attention = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.d**0.5)
        attention = F.softmax(attention, dim=-1)  # (B, T, N, N)

        x = torch.matmul(x.permute(0, 3, 1, 2), attention).permute(0, 2, 3, 1)
        out.append(x)

        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training), attention


class _Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.fc(x)


class STAWnet(nn.Module):
    """Spatial-Temporal Adaptive WaveNet.

    WaveNet dilated TCN with per-layer node self-attention using learnable
    node embeddings as keys/queries.

    Input shape:  (B, in_dim, num_nodes, seq_len)
    Output shape: (B, out_dim, num_nodes, 1)
    """

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        graph_attention=True,
        adaptive_adjacency_matrix=True,
        use_node_embedding_only=False,
        use_node_embedding=False,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        emb_length=16,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.graph_attention = graph_attention
        self.adaptive_adjacency_matrix = adaptive_adjacency_matrix
        self.supports = supports

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gat = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))

        receptive_field = 1
        if graph_attention and adaptive_adjacency_matrix:
            self.embedding = nn.Parameter(
                torch.randn(emb_length, num_nodes).to(device), requires_grad=True
            )
        elif graph_attention and not adaptive_adjacency_matrix:
            # BUGFIX-2026-07-02 (behavior change, not a no-op): see module docstring.
            if supports is None:
                raise ValueError(
                    "STAWnet(graph_attention=True, adaptive_adjacency_matrix=False) "
                    "requires `supports` (the real graph adjacency) to build a fixed "
                    "node embedding; got supports=None."
                )
            if emb_length > num_nodes:
                raise ValueError(
                    f"STAWnet(adaptive_adjacency_matrix=False) derives the node "
                    f"embedding via SVD of the (num_nodes, num_nodes) adjacency, which "
                    f"yields at most num_nodes={num_nodes} singular vectors; got "
                    f"emb_length={emb_length} > num_nodes."
                )
            # Non-adaptive: derive a fixed (non-learnable) node embedding from the
            # real graph via SVD, instead of a freely learned nn.Parameter (mirrors
            # gwn_v2.GWNv2._init_nodevecs's aptinit-based node-vector init).
            adj_for_svd = torch.stack([s.to(device) for s in supports]).mean(dim=0)
            _, singular_values, right_vecs = torch.svd(adj_for_svd)
            fixed_embedding = torch.mm(
                torch.diag(singular_values[:emb_length] ** 0.5),
                right_vecs[:, :emb_length].t(),
            )  # (emb_length, num_nodes)
            self.register_buffer("embedding", fixed_embedding)

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))

                if graph_attention:
                    self.gat.append(
                        _GraphAttention(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            emb_length=emb_length,
                            use_node_embedding_only=use_node_embedding_only,
                            use_node_embedding=use_node_embedding,
                        )
                    )

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input, return_attention=False):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0
        attentions = []

        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = self.skip_convs[i](x)
            skip = s if isinstance(skip, int) else s + skip[:, :, :, -s.size(3) :]

            # BUGFIX-2026-07-02: was `if self.graph_attention and self.adaptive_adjacency_matrix`;
            # see module docstring (behavior change, not a no-op).
            if self.graph_attention:
                x, att = self.gat[i](x, self.embedding)
                if return_attention:
                    attentions.append(att.mean(dim=(0, 1)).cpu().detach().numpy())
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        out = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))

        if return_attention:
            return out, attentions
        return out
