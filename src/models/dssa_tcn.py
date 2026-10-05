"""DSSA-TCN: Dual-Stream Sparse Attention Temporal Convolutional Network.

Ported from the DSSA-TCN baseline in the BasicTS framework repository:
    https://github.com/GestaltCogTeam/BasicTS (baselines/DSSA-TCN/)

Original paper:
    "DSSA-TCN: Exploiting Adaptive Sparse Attention and Diffusion Graph
    Convolutions in Temporal Convolutional Networks for Traffic Flow
    Forecasting."

Changes from the original:
  - Renamed TCNBlock → DSSATCN; internal helpers prefixed with _.
  - Removed BasicTS-specific forward args (future_data, batch_seen, epoch,
    train) — none were used by the model body.
  - Fixed end_conv_2: hardcoded out_channels=12 replaced with out_dim param.
  - Replaced try/except skip-slice with isinstance guard (same pattern as GWN).
  - Removed Chinese inline comments.

BUGFIX-2026-07-02 (tag: BUGFIX-2026-07-02, see git log for this commit): three
shape bugs where a constructor's channel-sizing formula didn't match what
forward() actually computed. All three are no-ops under the shipped base
configs (num_layers=1, residual_channels==dilation_channels) and only change
behavior for previously-crashing configs (e.g. this repo's W&B sweeps), so if
results on the base config regress, these are NOT the cause — revert and
re-check elsewhere first:
  - _GCN.forward now actually performs `order`-hop diffusion (it silently did
    only 1 hop before, regardless of `order`).
  - _SpatialAttentionStack.forward now returns the final stacked layer's
    output instead of concatenating every intermediate layer's output
    (previously multiplied channel width by num_layers).
  - spatial_attn/gcn_layers are now sized off dilation_channels (what the
    gated TCN output actually is), not residual_channels.
If re-investigating: these were latent because the DSSA-TCN sweep varies
`num_layers`/`residual_channels`/`dilation_channels` independently but the
base config keeps them at safe defaults. Sweep trials that previously
crashed on these combos will now run to completion instead — there are no
prior valid results for those combos to compare against (they errored out),
so this only adds new trials, it doesn't invalidate old ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _AttentionLayer(nn.Module):
    def __init__(
        self, model_dim, num_heads=4, mask=False, use_topk=False, topk_ratio_range=(1 / 6, 1 / 3)
    ):
        super().__init__()
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.mask = mask
        self.use_topk = use_topk

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        if use_topk:
            self.topk_lower, self.topk_upper = topk_ratio_range
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, query, key, value, return_attention: bool = False):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Split heads: (num_heads * B, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        attn = (query @ key.transpose(-1, -2)) / self.head_dim**0.5

        if self.use_topk:
            attn = self._apply_adaptive_topk(attn)

        if self.mask:
            causal = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            attn.masked_fill_(~causal, -torch.inf)

        attn_weights = torch.softmax(attn, dim=-1)  # (heads*B, ..., N, N)
        out = attn_weights @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        if return_attention:
            return out, attn_weights
        return out

    def _apply_adaptive_topk(self, attn):
        N = attn.shape[-1]
        ratio = self.topk_lower + (self.topk_upper - self.topk_lower) * torch.sigmoid(self.alpha)
        k = max(int(ratio * N), 1)
        flat = attn.reshape(-1, N)
        kth = torch.topk(flat, k, dim=-1).values[:, -1].unsqueeze(-1)
        flat = flat.masked_fill(flat < kth, float("-inf"))
        return flat.reshape_as(attn)


class _SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        feed_forward_dim=128,
        num_heads=4,
        dropout=0.1,
        mask=False,
        use_topk=False,
        topk_ratio_range=(1 / 6, 1 / 3),
    ):
        super().__init__()
        self.attn = _AttentionLayer(model_dim, num_heads, mask, use_topk, topk_ratio_range)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2, return_attention: bool = False):
        x = x.transpose(dim, -2)
        residual = x
        if return_attention:
            attn_out, attn_weights = self.attn(x, x, x, return_attention=True)
        else:
            attn_out = self.attn(x, x, x)
            attn_weights = None
        out = self.ln1(residual + self.dropout1(attn_out))
        out = self.ln2(out + self.dropout2(self.feed_forward(out)))
        out = out.transpose(dim, -2)
        if return_attention:
            return out, attn_weights
        return out


class _SpatialAttentionStack(nn.Module):
    def __init__(
        self,
        model_dim,
        feed_forward_dim=128,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        mask=False,
        use_topk=False,
        topk_ratio_range=(1 / 6, 1 / 3),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _SelfAttentionLayer(
                    model_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                    mask,
                    use_topk,
                    topk_ratio_range,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, return_attention: bool = False):
        # x: (B, T, N, C) — spatial axis is dim=2
        # BUGFIX-2026-07-02: was `outputs.append(x)` + `torch.cat(outputs, dim=-1)`,
        # multiplying channel width by num_layers instead of returning the stack's
        # final output. See module docstring.
        all_attn = []
        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, dim=2, return_attention=True)
                all_attn.append(attn_weights)
            else:
                x = layer(x, dim=2)
        if return_attention:
            return x, all_attn  # list of (heads*B, T, N, N)
        return x


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class _EmbeddingLayer(nn.Module):
    """Multi-source embedding: value projection + time-of-day + day-of-week
    + optional spatial + optional adaptive node embeddings.

    Input:  history_data (B, T, N, C) — channel 0 is the target feature,
            channels 1 and 2 are normalised ToD and DoW indices.
    Output: (x_embed, x_embed_no_adaptive) both of shape (B, T, N, model_dim).
    """

    def __init__(
        self,
        input_dim=3,
        steps_per_day=288,
        num_nodes=207,
        input_embedding_dim=12,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=12,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_emb = nn.Parameter(torch.empty(1, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_emb)

    def forward(self, history_data):
        # history_data: (B, T, N, C)
        batch_size, T, N, _ = history_data.shape
        x = history_data[..., : self.input_dim]
        tod = (history_data[..., 1] * self.steps_per_day).long()
        dow = (history_data[..., 2] * 7).long()

        parts = [self.input_proj(x)]
        if self.tod_embedding_dim > 0:
            parts.append(self.tod_embedding(tod))
        if self.dow_embedding_dim > 0:
            parts.append(self.dow_embedding(dow))
        if self.spatial_embedding_dim > 0:
            parts.append(self.node_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, T, -1, -1))

        x_no_adaptive = torch.cat(parts, dim=-1)

        if self.adaptive_embedding_dim > 0:
            parts.append(self.adaptive_emb.expand(batch_size, T, -1, -1))

        return torch.cat(parts, dim=-1), x_no_adaptive


# ---------------------------------------------------------------------------
# Diffusion GCN
# ---------------------------------------------------------------------------


class _GCN(nn.Module):
    """Diffusion-style graph convolution operating on (B, T, N, C) tensors."""

    def __init__(self, c_in, c_out, dropout=0.1, support_len=2, order=2):
        super().__init__()
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, (1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports):
        # x: (B, T, N, C_in) → permute to (B, C_in, N, T) for Conv2d
        x = x.permute(0, 3, 2, 1)

        # BUGFIX-2026-07-02: was a single hop (order-hop loop below missing
        # entirely) regardless of self.order, mismatching self.mlp's sizing
        # (order*support_len+1)*c_in. See module docstring.
        out = [x]
        for a in supports:
            x1 = torch.einsum("ncvl,vw->ncwl", x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum("ncvl,vw->ncwl", x1, a)
                out.append(x1)

        h = self.mlp(torch.cat(out, dim=1))
        h = F.dropout(h, self.dropout, training=self.training)
        return h.permute(0, 3, 2, 1)  # (B, T, N, C_out)


# ---------------------------------------------------------------------------
# DSSA-TCN
# ---------------------------------------------------------------------------


class DSSATCN(nn.Module):
    """Dual-Stream Sparse Attention Temporal Convolutional Network.

    Combines a WaveNet-style dilated TCN backbone with per-layer spatial
    attention (sparse multi-head, optionally top-k gated) and diffusion
    graph convolution.

    Input shape:  (B, T, N, C)  — time-major, consistent with BasicTS convention.
    Output shape: (B, out_dim, N, 1)  — same layout as other models in this repo.

    Note on input channels: channel 0 is the traffic feature; channels 1 and 2
    should be normalised time-of-day (0–1) and day-of-week (0–1) indices for
    the temporal embeddings to work correctly.
    """

    def __init__(
        self,
        input_dim=3,
        output_dim=1,
        out_dim=12,
        steps_per_day=288,
        num_nodes=207,
        input_embedding_dim=12,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        feed_forward_dim=128,
        num_heads=4,
        num_layers=1,
        mask=False,
        use_topk=True,
        topk_ratio_range=(1 / 6, 1 / 3),
        adjs=None,
        gcn_order=2,
        dropout=0.3,
    ):
        super().__init__()

        self.blocks = blocks
        self.layers = layers

        self.embedding = _EmbeddingLayer(
            input_dim=input_dim,
            steps_per_day=steps_per_day,
            num_nodes=num_nodes,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
        )
        model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.start_conv = nn.Conv2d(model_dim, residual_channels, (1, 1))

        # Register pre-normalized supports passed in from outside (e.g. doubletransition)
        if adjs is not None:
            self.register_buffer("support", torch.stack(adjs, dim=0).clone().detach())
            support_len = len(self.support)
        else:
            self.support = None
            support_len = 2  # default for GCN sizing

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.spatial_attn = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels, dilation_channels, (1, kernel_size), dilation=dilation
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels, dilation_channels, (1, kernel_size), dilation=dilation
                    )
                )
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, (1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                # BUGFIX-2026-07-02: model_dim/c_in/c_out match dilation_channels, not
                # residual_channels: the filter/gate convs above map residual_channels
                # -> dilation_channels, and this tensor is what spatial_attn/gcn_layers
                # actually receive (and must return unchanged, since skip_convs/
                # residual_convs below expect dilation_channels input).
                self.spatial_attn.append(
                    _SpatialAttentionStack(
                        model_dim=dilation_channels,
                        feed_forward_dim=feed_forward_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout=dropout,
                        mask=mask,
                        use_topk=use_topk,
                        topk_ratio_range=topk_ratio_range,
                    )
                )
                self.gcn_layers.append(
                    _GCN(
                        c_in=dilation_channels,
                        c_out=dilation_channels,
                        dropout=dropout,
                        support_len=support_len,
                        order=gcn_order,
                    )
                )

        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """
        Args:
            x: (B, T, N, C) — time-major input with value + ToD + DoW channels.
            return_attention: if True, also return a list of sparse spatial
                attention matrices (one per SSA sub-layer across all TCN blocks),
                each of shape (heads*B, T, N, N).

        Returns:
            (B, out_dim, N, 1), or ((B, out_dim, N, 1), list) when return_attention=True.
        """
        x, _ = self.embedding(x)  # (B, T, N, model_dim)
        x = x.permute(0, 3, 2, 1)  # (B, model_dim, N, T)

        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        x = self.start_conv(x)  # (B, residual_channels, N, T)
        skip: torch.Tensor | None = None
        all_block_attn: list = []

        supports = self.support

        for i in range(self.blocks * self.layers):
            residual = x
            x = torch.tanh(self.filter_convs[i](residual)) * torch.sigmoid(
                self.gate_convs[i](residual)
            )

            # Spatial attention + diffusion GCN (operate in B, T, N, C layout)
            x_bnct = x.permute(0, 3, 2, 1)  # (B, T, N, C)
            if return_attention:
                x_bnct, block_attn = self.spatial_attn[i](x_bnct, return_attention=True)
                all_block_attn.extend(block_attn)  # each: (heads*B, T, N, N)
            else:
                x_bnct = self.spatial_attn[i](x_bnct)
            if supports is not None:
                x_bnct = self.gcn_layers[i](x_bnct, supports)
            x = x_bnct.permute(0, 3, 2, 1)  # (B, C, N, T)

            s = self.skip_convs[i](x)
            skip = s if skip is None else skip[:, :, :, -s.size(3) :] + s

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        if skip is None:
            raise RuntimeError("DSSATCN requires at least one temporal layer.")
        x = F.relu(self.end_conv_1(F.relu(skip)))
        out = self.end_conv_2(x)[..., -1:]
        if return_attention:
            return out, all_block_attn
        return out
