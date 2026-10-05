"""Model adapters that present a uniform interface to the shared trainer.

Every adapter accepts canonical input ``x: (B, C, N, T)`` and returns
``(B, N, H)`` — the only shape the trainer and metrics layer ever see.

Optional keyword arguments (passed through for future extensibility; most adapters ignore them):
    y_full       : (B, C, N, H)  multi-channel target window
    batches_seen : int           global training step
    task_level   : int           active forecast horizon
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.typing import Adj, OptTensor

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GraphModelAdapter(nn.Module):
    """Wraps a graph model and normalises its I/O to the canonical layout."""

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        raise NotImplementedError

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Return an (N, N) learned adjacency matrix, or None if unavailable.

        Adapters for models with static learned graphs (GWN, GWNv2, STAEformer,
        DSSA-TCN) compute adjacency directly from parameters and ignore
        ``loader``. The STAWnet attention adapter requires a DataLoader and
        raises ``ValueError`` when one is not provided.
        """
        return None


def _validate_canonical_input(x: Tensor, adapter_name: str) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"{adapter_name} expected canonical input x with shape (B, C, N, T); "
            f"got {tuple(x.shape)}."
        )


def _validate_native_rank4_output(out: Tensor, adapter_name: str) -> None:
    if out.ndim != 4 or out.shape[-1] != 1:
        raise ValueError(
            f"{adapter_name} native output must have shape (B, H, N, 1); got {tuple(out.shape)}."
        )


def _validate_canonical_output(out: Tensor, adapter_name: str) -> None:
    if out.ndim != 3:
        raise ValueError(
            f"{adapter_name} canonical output must have shape (B, N, H); got {tuple(out.shape)}."
        )


# ---------------------------------------------------------------------------
# GWN  (B, C, N, T) → permute → (B, T, N, C) → tsl GraphWaveNetModel → (B, H, N, 1) → (B, N, H)
# ---------------------------------------------------------------------------


class GWNAdapter(GraphModelAdapter):
    """Adapter for tsl's ``GraphWaveNetModel`` (Wu et al., IJCAI 2019).

    The underlying model needs a static graph (``edge_index``/``edge_weight``)
    in addition to its own learned adaptive adjacency, so both are captured
    once at construction time and threaded through every forward call.
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("edge_index", edge_index, persistent=False)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight, persistent=False)
        else:
            self.edge_weight = None

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "GWNAdapter")
        x_tsl = x.permute(0, 3, 2, 1)  # (B, T, N, C)
        out = self.model(x_tsl, self.edge_index, self.edge_weight)  # (B, H, N, 1)
        _validate_native_rank4_output(out, "GWNAdapter")
        pred = out.squeeze(-1).permute(0, 2, 1)  # (B, N, H)
        _validate_canonical_output(pred, "GWNAdapter")
        return pred

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Learned adaptive adjacency via tsl's own ``get_learned_adj()``."""
        # cast: nn.Module.__getattr__ is typed Tensor | Module, so mypy can't
        # know the runtime-only attributes these third-party models expose
        # (guarded by the hasattr() checks below).
        m = cast(Any, self.model)
        if not hasattr(m, "get_learned_adj"):
            return None
        with torch.no_grad():
            adj = m.get_learned_adj()
        return adj.cpu().numpy()


# ---------------------------------------------------------------------------
# TCN  (B, C, N, T) → permute → (B, T, N, C) → tsl TCNModel → (B, H, N, 1) → (B, N, H)
# ---------------------------------------------------------------------------


class TCNAdapter(GraphModelAdapter):
    """Adapter for tsl's ``TCNModel`` — a graph-free temporal baseline.

    ``num_inputs`` controls whether the model sees the full canonical channel
    set or only channel 0 (speed), matching the legacy univariate/multivariate
    config knob in ``tcn_base.yaml``.
    """

    def __init__(self, model: nn.Module, num_inputs: int = 1) -> None:
        super().__init__()
        self.model = model
        self.num_inputs = num_inputs

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "TCNAdapter")
        x_in = x[:, : self.num_inputs] if self.num_inputs < x.shape[1] else x
        x_tsl = x_in.permute(0, 3, 2, 1)  # (B, T, N, C)
        out = self.model(x_tsl)  # (B, H, N, 1)
        _validate_native_rank4_output(out, "TCNAdapter")
        pred = out.squeeze(-1).permute(0, 2, 1)  # (B, N, H)
        _validate_canonical_output(pred, "TCNAdapter")
        return pred


# ---------------------------------------------------------------------------
# GWNv2  (B, C, N, T) → model → (B, H, N, 1) → (B, N, H)
# ---------------------------------------------------------------------------


class GWNv2Adapter(GraphModelAdapter):
    """Adapter for Graph WaveNet v2 (GWNv2)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "GWNv2Adapter")
        out = self.model(x)
        _validate_native_rank4_output(out, "GWNv2Adapter")
        pred = out.squeeze(-1).permute(0, 2, 1)
        _validate_canonical_output(pred, "GWNv2Adapter")
        return pred

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Same as GWNAdapter — adaptive adjacency from node vectors."""
        m = cast(Any, self.model)
        if not (hasattr(m, "nodevec1") and hasattr(m, "nodevec2")):
            return None
        with torch.no_grad():
            adj = F.softmax(F.relu(torch.mm(m.nodevec1, m.nodevec2)), dim=1)
        return adj.cpu().numpy()


# ---------------------------------------------------------------------------
# STAWnet  (B, C, N, T) → model → (B, H, N, 1) → (B, N, H)
# ---------------------------------------------------------------------------


class STAWnetAdapter(GraphModelAdapter):
    """Adapter for STAWnet."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "STAWnetAdapter")
        out = self.model(x)
        _validate_native_rank4_output(out, "STAWnetAdapter")
        pred = out.squeeze(-1).permute(0, 2, 1)
        _validate_canonical_output(pred, "STAWnetAdapter")
        return pred

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Spatial attention weights averaged over test batches and blocks.

        Requires a DataLoader.  STAWnet's forward supports
        ``return_attention=True`` which returns per-block (N, N) matrices
        already averaged over batch and time dimensions.
        """
        if loader is None:
            raise ValueError("STAWnetAdapter.get_adjacency requires a DataLoader.")
        dev = device or next(self.model.parameters()).device
        self.model.eval()
        all_attn: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(dev)  # (B, C, N, T)
                result = self.model(x, return_attention=True)
                if isinstance(result, tuple):
                    _, attentions = result
                    # attentions: list of (N, N) arrays, one per GAT block
                    all_attn.append(np.mean(attentions, axis=0))
        if not all_attn:
            return None
        return np.mean(all_attn, axis=0).astype(np.float32)  # (N, N)


# ---------------------------------------------------------------------------
# STAEformer  (B, C, N, T) → permute → (B, T, N, C) → model → (B, out_steps, N, 1) → (B, N, H)
# ---------------------------------------------------------------------------


class STAEformerAdapter(GraphModelAdapter):
    """Adapter for STAEformer.

    STAEformer expects *(B, T, N, C)* input and returns *(B, out_steps, N, 1)*.
    Requires at least 3 input channels: [speed, time-of-day, day-of-week].
    The ToD channel must be normalized to [0, 1) so that
    ``(tod * steps_per_day).long()`` produces valid embedding indices.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "STAEformerAdapter")
        x_stae = x.permute(0, 3, 2, 1)  # (B, T, N, C)
        out = self.model(x_stae)  # (B, out_steps, N, 1)
        pred = out.squeeze(-1).permute(0, 2, 1)  # (B, N, H)
        _validate_canonical_output(pred, "STAEformerAdapter")
        return pred

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Node-similarity matrix derived from Z = E_f || E_p || E_a.

        STAEformer replaces an explicit adjacency with the spatio-temporal
        adaptive embedding (Liu et al., CIKM 2023 §3.2).  For analysis we
        recover an (N, N) proxy by capturing Z (the full concatenated hidden
        representation, shape B×T×N×d_h) just before the attention layers,
        averaging over batch and time to get (N, d_h), then computing
        softmax(relu(Z_mean @ Z_mean.T)).

        The Z-mean is accumulated over the **entire** test set (summing over
        every batch's B and T, divided by the total sample count), not a single
        batch — matching STAWnet, which averages its attention over all test
        batches. This is a consistency fix:
        a one-batch Z-mean and a full-test Z-mean would otherwise place these
        two models on a different footing from the attention-based ones.

        Requires a DataLoader.  Falls back to E_a alone if none is provided.
        """
        m = cast(Any, self.model)
        dev = device or next(m.parameters()).device

        if loader is not None:
            captured: dict[str, torch.Tensor] = {}

            def _hook(module: nn.Module, args: tuple) -> None:  # noqa: ARG001
                captured["z"] = args[0].detach()  # (B, T, N, d_h)

            handle = m.attn_layers_t[0].register_forward_pre_hook(_hook)
            z_sum: torch.Tensor | None = None
            count = 0
            try:
                m.eval()
                with torch.no_grad():
                    for batch in loader:
                        x_canonical = batch[0].to(dev)  # (B, C, N, T)
                        x_stae = x_canonical.permute(0, 3, 2, 1)  # (B, T, N, C)
                        m(x_stae)
                        z = captured["z"]  # (B, T, N, d_h)
                        b, t = z.shape[0], z.shape[1]
                        batch_sum = z.sum(dim=(0, 1))  # (N, d_h)
                        z_sum = batch_sum if z_sum is None else z_sum + batch_sum
                        count += b * t
            finally:
                handle.remove()

            if z_sum is not None and count > 0:
                z_mean = z_sum / count  # (N, d_h), averaged over the full test set
                adj = F.softmax(F.relu(z_mean @ z_mean.T), dim=1)
                return adj.cpu().numpy().astype(np.float32)

        # Fallback: use E_a alone (input-independent, purely learned)
        if not (hasattr(m, "adaptive_embedding") and m.adaptive_embedding_dim > 0):
            return None
        with torch.no_grad():
            emb = m.adaptive_embedding.mean(0)  # (N, d_a)
            adj = F.softmax(F.relu(emb @ emb.T), dim=1)
        return adj.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# DSSA-TCN  (B, C, N, T) → permute → (B, T, N, C) → model → (B, H, N, 1) → (B, N, H)
# ---------------------------------------------------------------------------


class DSSATCNAdapter(GraphModelAdapter):
    """Adapter for DSSA-TCN.

    DSSA-TCN expects time-major input *(B, T, N, C)* due to its embedding
    layer extracting ToD/DoW features from specific channel indices.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "DSSATCNAdapter")
        if x.shape[1] < 3:
            raise ValueError(
                f"DSSATCNAdapter requires at least 3 channels for value, time-of-day, "
                f"and day-of-week; got {x.shape[1]}."
            )
        # (B, C, N, T) → (B, T, N, C)
        x_dssa = x.permute(0, 3, 2, 1)
        out = self.model(x_dssa)  # (B, out_dim, N, 1)
        _validate_native_rank4_output(out, "DSSATCNAdapter")
        pred = out.squeeze(-1).permute(0, 2, 1)  # (B, N, H)
        _validate_canonical_output(pred, "DSSATCNAdapter")
        return pred

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Sparse spatial attention averaged over SSA sub-layers, heads, batch, and time.

        Runs the **full test set** through DSSATCN with return_attention=True to
        collect the sparse A_sparse = softmax(topk(Q @ K^T / √d)) matrices from
        every SSA sub-layer across all TCN blocks, averages them to a per-batch
        (N, N) proxy, then takes the unweighted mean over batches — matching
        STAWnet. This is a consistency fix:
        the previous implementation used only a single evaluation batch, placing
        DSSA-TCN on a different footing from the models averaged over all
        batches.

        Requires a DataLoader.
        """
        if loader is None:
            raise ValueError("DSSATCNAdapter.get_adjacency requires a DataLoader.")
        m = self.model
        dev = device or next(m.parameters()).device
        m.eval()
        per_batch: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x_canonical = batch[0].to(dev)  # (B, C, N, T)
                x_dssa = x_canonical.permute(0, 3, 2, 1)  # (B, T, N, C)
                _, all_block_attn = m(x_dssa, return_attention=True)
                if not all_block_attn:
                    continue
                # all_block_attn: list of (heads*B, T, N, N)
                B = x_canonical.shape[0]
                adjs = []
                for attn in all_block_attn:
                    heads_b = attn.shape[0]
                    num_heads = heads_b // B
                    a = attn.view(num_heads, B, *attn.shape[1:])  # (heads, B, T, N, N)
                    adjs.append(a.mean(dim=(0, 1, 2)))  # (N, N)
                adj = torch.stack(adjs).mean(0)  # (N, N) mean over blocks
                per_batch.append(adj.cpu().numpy().astype(np.float32))

        if not per_batch:
            return None
        return np.mean(per_batch, axis=0).astype(np.float32)  # (N, N)


# ---------------------------------------------------------------------------
# BigST  (B, C, N, T) → permute → (B, N, T, C) → model → (B, N, H)
# ---------------------------------------------------------------------------


class BigSTAdapter(GraphModelAdapter):
    """Adapter for BigST (Han et al., VLDB 2024).

    BigST expects node-major input *(B, N, T, C)* (not the channel-major
    layout most other adapters convert to) and returns a
    *(prediction, spatial_loss)* tuple. The day-of-week channel convention
    also differs: the framework's canonical channel 2 is ``weekday / 7.0`` in
    ``[0, 6/7]``, but BigST's forward indexes an embedding table with a raw
    integer weekday (``x[:, :, -1, 2]).long()``), so this adapter multiplies
    that channel by 7 (and rounds) before calling the model. This only feeds
    an embedding lookup index, so the rounding doesn't need to be
    differentiable.

    If constructed with a precomputed long-term feature array (see
    ``scripts/preprocess_bigst_features.py`` / ``src/models/bigst_longterm.py``),
    it is registered as a buffer and broadcast over the batch dimension at
    forward time so the wrapped model's ``use_long=True`` path can consume it.
    """

    long_term_feat: Tensor | None

    def __init__(self, model: nn.Module, long_term_feat: Tensor | None = None) -> None:
        super().__init__()
        self.model = model
        if long_term_feat is not None:
            # (N, F) -> registered once; broadcast over batch at forward time.
            self.register_buffer("long_term_feat", long_term_feat, persistent=False)
        else:
            self.register_buffer("long_term_feat", None, persistent=False)

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "BigSTAdapter")
        if x.shape[1] < 3:
            raise ValueError(
                f"BigSTAdapter requires at least 3 channels for value, time-of-day, "
                f"and day-of-week; got {x.shape[1]}."
            )
        # (B, C, N, T) -> (B, N, T, C)
        x_bigst = x.permute(0, 2, 3, 1).clone()
        # Canonical day-of-week channel is weekday/7.0 in [0, 6/7]; BigST's
        # forward indexes an embedding table with a raw integer weekday
        # (x[:, :, -1, 2]).long() — undo the normalisation here, not in the model.
        x_bigst[:, :, :, 2] = (x_bigst[:, :, :, 2] * 7.0).round()

        feat = None
        if self.long_term_feat is not None:
            feat = self.long_term_feat.unsqueeze(0).expand(x_bigst.shape[0], -1, -1)  # (B, N, F)

        pred, _spatial_loss = self.model(x_bigst, feat=feat)  # (B, N, H)
        # _spatial_loss: BigST's optional spatial regulariser (only non-zero
        # when use_spatial=True). The trainer's loss computation
        # (GraphTrainer._train_epoch) only accepts a single prediction tensor
        # from the adapter — there is no auxiliary-loss hook today — so this
        # is discarded for now rather than wired into the training loss.
        _validate_canonical_output(pred, "BigSTAdapter")
        return pred

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Static learned graph via softmax(relu(E @ E^T)) over the single node embedding.

        BigST has one node-embedding parameter (``node_emb_layer``, shape
        (N, node_dim)) rather than the two-vector factorisation GWNv2/D2STGNN
        use, so the closest faithful equivalent is the self-similarity of
        that single embedding table.
        """
        m = cast(Any, self.model)
        if not hasattr(m, "node_emb_layer"):
            return None
        with torch.no_grad():
            e = m.node_emb_layer
            adj = F.softmax(F.relu(e @ e.T), dim=1)
        return adj.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# D2STGNN  (B, C, N, T) → permute + build (B, T, N, num_feat+2) → model → (B, N, H)
# ---------------------------------------------------------------------------


class D2STGNNAdapter(GraphModelAdapter):
    """Adapter for D2STGNN (Shao et al., VLDB 2022).

    D2STGNN expects time-major input *(B, T, N, num_feat + 2)* where the
    *last two* channels are the time-of-day fraction in [0, 1) and the raw
    day-of-week integer in [0, 6]. This framework's canonical channel 2
    (day-of-week) is normalised to ``weekday / 7.0`` in ``[0, 6/7]``, so it is
    multiplied by 7 here before being passed to the model;
    ``D2STGNN._prepare_inputs`` immediately calls ``.long()`` on it for an
    embedding-table lookup, so this conversion does not need to be
    differentiable.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        y_full: Tensor | None = None,
        batches_seen: int | None = None,
        task_level: int = 12,
    ) -> Tensor:
        _validate_canonical_input(x, "D2STGNNAdapter")
        if x.shape[1] < 3:
            raise ValueError(
                f"D2STGNNAdapter requires at least 3 channels for value, time-of-day, "
                f"and day-of-week; got {x.shape[1]}."
            )
        # (B, C, N, T) → (B, T, N, C)
        x_d2 = x.permute(0, 3, 2, 1).clone()
        # Canonical day-of-week channel is weekday/7.0 in [0, 6/7]; D2STGNN
        # expects the raw integer weekday in [0, 6] — undo the normalisation
        # here, not in the model.
        x_d2[..., 2] = (x_d2[..., 2] * 7.0).round()
        out = self.model(x_d2)  # (B, N, H)
        _validate_canonical_output(out, "D2STGNNAdapter")
        return out

    def get_adjacency(
        self,
        loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray | None:
        """Static adaptive graph from the model's own node embeddings.

        Same math as ``D2STGNN._graph_constructor``'s static-graph branch:
        ``softmax(relu(node_emb_u @ node_emb_d.T))``. No DataLoader needed —
        this is a static-graph model, like GWN/GWNv2.
        """
        m = cast(Any, self.model)
        if not (hasattr(m, "node_emb_u") and hasattr(m, "node_emb_d")):
            return None
        with torch.no_grad():
            adj = F.softmax(F.relu(torch.mm(m.node_emb_u, m.node_emb_d.T)), dim=1)
        return adj.cpu().numpy().astype(np.float32)
