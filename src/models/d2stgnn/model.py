"""D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network.

Ported from D2STGNN (Shao et al., "Decoupled Dynamic Spatial-Temporal Graph
Neural Network for Traffic Forecasting", VLDB 2022),
``new_models/D2STGNN/models/model.py``.

Changes from the original:
  - ``_prepare_inputs`` used ``(...).type(torch.LongTensor)``, which always
    creates a *CPU* long tensor regardless of the input's device, silently
    breaking GPU training (cross-device indexing into ``self.T_i_D_emb`` /
    ``self.D_i_W_emb``). Replaced with ``.long()`` on the tensor itself, which
    preserves the source device.
  - Cleaned up unused/no-op self-assignment.
  - ``num_modalities`` from the original METR-LA.yaml config is intentionally
    *not* accepted here: it is not referenced anywhere in the model body
    (verified by grepping ``new_models/D2STGNN/models/*.py``).
  - Channel convention: this faithful port still expects
    ``history_data[..., num_feat]`` to be the time-of-day fraction in [0, 1)
    and ``history_data[..., num_feat + 1]`` to be the *raw* day-of-week
    integer in [0, 6] (not normalised), exactly as upstream D2STGNN does. The
    framework's canonical day-of-week channel is normalised to
    ``weekday / 7.0`` -- the conversion back to a raw integer is handled in
    ``D2STGNNAdapter`` (src/training/adapters.py), not here, so this class
    stays a faithful, reusable port of the original model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decouple.estimation_gate import EstimationGate
from .diffusion_block import DifBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .inherent_block import InhBlock


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args["node_hidden"],
            time_emb_dim=model_args["time_emb_dim"],
            hidden_dim=64,
        )
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(
        self,
        history_data: torch.Tensor,
        dynamic_graph: torch.Tensor,
        static_graph,
        node_embedding_u,
        node_embedding_d,
        time_in_day_feat,
        day_in_week_feat,
    ):
        """decouple layer

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency
                matrix with shape (B, N, k_t * N)
            static_graph (list of torch.Tensor): the self-adaptive transition
                matrix with shape (N, N)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            torch.Tensor: the un-decoupled signal in this layer, i.e. the
                X^{l+1} fed to the next layer. Shape [B, L', N, D].
            torch.Tensor: the output of the Diffusion Block's forecast
                branch, shape (B, L'', N, D), where
                L'' = output_seq_len / model_args['gap'] to avoid error
                accumulation in auto-regression.
            torch.Tensor: the output of the Inherent Block's forecast
                branch, shape (B, L'', N, D), where
                L'' = output_seq_len / model_args['gap'] to avoid error
                accumulation in auto-regression.
        """

        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data
        )
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data,
            gated_history_data=gated_history_data,
            dynamic_graph=dynamic_graph,
            static_graph=static_graph,
        )
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    """Decoupled Dynamic Spatial-Temporal Graph Neural Network.

    Native I/O (faithful to the original implementation):
        forward(history_data) where history_data: (B, L, N, num_feat + 2)
        returns: (B, N, L_out) where L_out = model_args['seq_length']
        (requires model_args['seq_length'] % model_args['gap'] == 0).

    Required ``model_args`` keys: num_feat, num_hidden, node_hidden,
    time_emb_dim, dropout, seq_length, k_t, k_s, gap, num_nodes, adjs (list of
    dense (N, N) torch tensors). ``use_pre``, ``dy_graph``, ``sta_graph`` are
    set internally (hardcoded to False, True, True) and must not be passed.

    Optional ``model_args`` key: ``in_seq_length``. Upstream D2STGNN hardcodes
    the input window length to equal ``seq_length`` (both input and output
    windows are 12 in the original METR-LA/PEMS-BAY configs) because
    ``DynamicGraphConstructor``'s ``DistanceFunction`` submodule has a
    ``nn.Linear`` whose ``in_features`` must match the actual input window
    length T. This framework allows T (input window) and H (forecast
    horizon, still ``seq_length``) to differ by setting ``in_seq_length = T``
    explicitly; it defaults to ``seq_length`` to preserve byte-for-byte
    upstream behaviour when omitted.
    """

    def __init__(self, **model_args):
        super().__init__()
        seq_length = model_args["seq_length"]
        gap = model_args["gap"]
        if gap <= 0:
            raise ValueError(f"D2STGNN requires a positive gap; got {gap}.")
        if seq_length % gap != 0:
            raise ValueError(
                "D2STGNN requires seq_length to be divisible by gap "
                f"because each decoder state predicts gap steps; got "
                f"seq_length={seq_length}, gap={gap}."
            )
        # attributes
        self._in_feat = model_args["num_feat"]
        self._hidden_dim = model_args["num_hidden"]
        self._node_dim = model_args["node_hidden"]
        self._forecast_dim = 256
        self._output_hidden = 512
        self._output_dim = seq_length

        self._num_nodes = model_args["num_nodes"]
        self._k_s = model_args["k_s"]
        self._k_t = model_args["k_t"]
        self._num_layers = 5

        model_args["use_pre"] = False
        model_args["dy_graph"] = True
        model_args["sta_graph"] = True

        self._model_args = model_args

        # start embedding layer
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args["time_emb_dim"]))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args["time_emb_dim"]))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)]
        )
        for _ in range(self._num_layers - 1):
            self.layers.append(
                DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)
            )

        # dynamic and static hidden graph constructor
        if model_args["dy_graph"]:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # output layer
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args["gap"])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs["node_embedding_u"]
        E_u = inputs["node_embedding_d"]
        if self._model_args["sta_graph"]:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args["dy_graph"]:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat = self._model_args["num_feat"]
        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
        # time slot embedding
        # NOTE: original used `.type(torch.LongTensor)`, which always builds a
        # CPU tensor regardless of `history_data`'s device. `.long()` keeps
        # the index tensor on the same device as the input.
        time_in_day_feat = self.T_i_D_emb[
            (history_data[:, :, :, num_feat] * 288).long()
        ]  # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[
            (history_data[:, :, :, num_feat + 1]).long()
        ]  # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        """Feed forward of D2STGNN.

        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]

        Returns:
            torch.Tensor: prediction data with shape: [B, N, L]
        """

        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = (
            self._prepare_inputs(history_data)
        )

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat,
        )

        # Start embedding layer
        history_data = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(
                inh_backcast_seq_res,
                dynamic_graph,
                static_graph,
                node_embedding_u,
                node_embedding_d,
                time_in_day_feat,
                day_in_week_feat,
            )
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        # regression layer
        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = (
            forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        )

        return forecast


def make_d2stgnn(
    device,
    num_nodes: int,
    adj_mx,
    num_feat: int = 1,
    num_hidden: int = 32,
    node_hidden: int = 10,
    time_emb_dim: int = 10,
    dropout: float = 0.1,
    seq_length: int = 12,
    in_seq_length: int | None = None,
    k_t: int = 3,
    k_s: int = 2,
    gap: int = 3,
) -> D2STGNN:
    """Build a D2STGNN model from a list of dense adjacency supports.

    Threads precomputed dense adjacency supports (e.g. from
    ``data.adjacency.build_supports``) into the model's constructor. ``adj_mx``
    should be a list of dense (N, N) arrays
    or tensors (e.g. the ``"doubletransition"`` pair
    ``[asym_adj(adj), asym_adj(adj.T)]``); they are converted to float
    tensors on ``device`` and passed through as ``model_args['adjs']``.

    ``in_seq_length`` is the input window length T (defaults to
    ``seq_length`` if the input window and forecast horizon are equal, as in
    upstream D2STGNN's own configs); pass it explicitly when T != H (the
    forecast horizon, still controlled by ``seq_length``). ``seq_length`` must
    be divisible by ``gap`` because each decoder state predicts ``gap`` output
    steps.
    """
    adjs = [a if isinstance(a, torch.Tensor) else torch.from_numpy(a) for a in adj_mx]
    adjs = [a.to(device=device, dtype=torch.float32) for a in adjs]

    model_args = dict(
        num_nodes=num_nodes,
        num_feat=num_feat,
        num_hidden=num_hidden,
        node_hidden=node_hidden,
        time_emb_dim=time_emb_dim,
        dropout=dropout,
        seq_length=seq_length,
        k_t=k_t,
        k_s=k_s,
        gap=gap,
        adjs=adjs,
    )
    if in_seq_length is not None:
        model_args["in_seq_length"] = in_seq_length

    model = D2STGNN(**model_args)
    return model.to(device)
