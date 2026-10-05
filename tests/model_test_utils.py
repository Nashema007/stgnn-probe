from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from tsl.nn.models.stgn import GraphWaveNetModel
from tsl.nn.models.temporal import TCNModel

from models.dssa_tcn import DSSATCN
from models.gwn_v2 import GWNv2
from models.staeformer import STAEformer
from models.stawnet import STAWnet
from training.adapters import (
    DSSATCNAdapter,
    GraphModelAdapter,
    GWNAdapter,
    GWNv2Adapter,
    STAEformerAdapter,
    STAWnetAdapter,
    TCNAdapter,
)

B = 2
N = 5
T = 12
H = 3
DEVICE = torch.device("cpu")


def seed_all(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ring_adjacency(num_nodes: int = N) -> torch.Tensor:
    adj = torch.eye(num_nodes, dtype=torch.float32)
    adj += torch.roll(torch.eye(num_nodes, dtype=torch.float32), shifts=1, dims=1)
    return adj


def canonical_x(channels: int = 1) -> torch.Tensor:
    seed_all()
    return torch.randn(B, channels, N, T, device=DEVICE)


def canonical_target() -> torch.Tensor:
    seed_all(11)
    return torch.randn(B, N, H, device=DEVICE)


def dssa_x() -> torch.Tensor:
    seed_all(17)
    value = torch.randn(B, 1, N, T, device=DEVICE)
    tod = torch.linspace(0, (T - 1) / 288, T, device=DEVICE).view(1, 1, 1, T)
    tod = tod.expand(B, 1, N, T)
    dow = torch.zeros(B, 1, N, T, device=DEVICE)
    return torch.cat([value, tod, dow], dim=1)


def assert_finite_float_output(out: torch.Tensor, shape: tuple[int, ...]) -> None:
    assert out.shape == shape
    assert out.dtype.is_floating_point
    assert torch.isfinite(out).all()


def clone_trainable_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def assert_has_nonzero_finite_gradient(model: nn.Module) -> None:
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert any(torch.isfinite(g).all() and torch.any(g != 0) for g in grads)


def assert_parameter_changed(model: nn.Module, before: list[torch.Tensor]) -> None:
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    assert len(after) == len(before)
    assert any(not torch.allclose(old, new) for old, new in zip(before, after, strict=True))


def graph_model_specs() -> list[
    tuple[str, Callable[[], GraphModelAdapter], Callable[[], torch.Tensor]]
]:
    return [
        ("gwn", build_gwn_adapter, lambda: canonical_x(1)),
        ("gwn_v2", build_gwn_v2_adapter, lambda: canonical_x(1)),
        ("stawnet", build_stawnet_adapter, lambda: canonical_x(1)),
        ("staeformer", build_staeformer_adapter, dssa_x),
        ("dssa_tcn", build_dssa_tcn_adapter, dssa_x),
        ("tcn", build_tcn_adapter, lambda: canonical_x(1)),
    ]


def torch_model_specs() -> list[
    tuple[str, Callable[[], nn.Module], Callable[[nn.Module], torch.Tensor]]
]:
    return [
        ("gwn", build_gwn, lambda model: gwn_forward(model, canonical_x(1))),
        ("gwn_v2", build_gwn_v2_adapter, lambda model: model(canonical_x(1))),
        ("stawnet", build_stawnet_adapter, lambda model: model(canonical_x(1))),
        ("staeformer", build_staeformer_adapter, lambda model: model(dssa_x())),
        ("dssa_tcn", build_dssa_tcn_adapter, lambda model: model(dssa_x())),
        ("tcn", build_tcn, lambda model: tcn_forward(model, canonical_x(1))),
    ]


def build_gwn() -> GraphWaveNetModel:
    seed_all()
    return GraphWaveNetModel(
        input_size=1,
        output_size=1,
        horizon=H,
        n_nodes=N,
        hidden_size=4,
        ff_size=8,
        n_layers=2,
        emb_size=2,
    )


def gwn_forward(model: GraphWaveNetModel, x: torch.Tensor) -> torch.Tensor:
    edge_index, edge_weight = dense_to_sparse(ring_adjacency())
    x_tsl = x.permute(0, 3, 2, 1)  # (B, T, N, C)
    out = model(x_tsl, edge_index, edge_weight)  # (B, H, N, 1)
    return out.squeeze(-1).permute(0, 2, 1)  # (B, N, H)


def build_gwn_adapter() -> GWNAdapter:
    seed_all()
    edge_index, edge_weight = dense_to_sparse(ring_adjacency())
    return GWNAdapter(build_gwn(), edge_index, edge_weight)


def build_gwn_v2_adapter() -> GWNv2Adapter:
    seed_all()
    return GWNv2Adapter(
        GWNv2(
            DEVICE,
            N,
            dropout=0.0,
            supports=[ring_adjacency()],
            in_dim=1,
            out_dim=H,
            residual_channels=2,
            dilation_channels=2,
            skip_channels=4,
            end_channels=8,
            blocks=4,
            layers=2,
            apt_size=2,
        )
    )


def build_stawnet_adapter() -> STAWnetAdapter:
    seed_all()
    return STAWnetAdapter(
        STAWnet(
            DEVICE,
            N,
            dropout=0.0,
            in_dim=1,
            out_dim=H,
            residual_channels=2,
            dilation_channels=2,
            skip_channels=4,
            end_channels=8,
            blocks=4,
            layers=2,
            emb_length=2,
        )
    )


def build_staeformer_adapter() -> STAEformerAdapter:
    seed_all()
    model = STAEformer(
        num_nodes=N,
        in_steps=T,
        out_steps=H,
        steps_per_day=288,
        input_dim=1,
        output_dim=1,
        input_embedding_dim=2,
        tod_embedding_dim=2,
        dow_embedding_dim=2,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=2,
        feed_forward_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    return STAEformerAdapter(model)


def build_dssa_tcn_adapter() -> DSSATCNAdapter:
    seed_all()
    model = DSSATCN(
        input_dim=3,
        out_dim=H,
        num_nodes=N,
        residual_channels=2,
        dilation_channels=2,
        skip_channels=4,
        end_channels=8,
        blocks=1,
        layers=1,
        input_embedding_dim=2,
        tod_embedding_dim=2,
        dow_embedding_dim=2,
        adaptive_embedding_dim=2,
        feed_forward_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        adjs=[ring_adjacency()],
        gcn_order=1,
        use_topk=False,
    )
    return DSSATCNAdapter(model)


def build_tcn() -> TCNModel:
    seed_all()
    return TCNModel(input_size=1, output_size=1, horizon=H, hidden_size=4, ff_size=4, n_layers=2)


def tcn_forward(model: TCNModel, x: torch.Tensor) -> torch.Tensor:
    x_tsl = x.permute(0, 3, 2, 1)  # (B, T, N, C)
    out = model(x_tsl)  # (B, H, N, 1)
    return out.squeeze(-1).permute(0, 2, 1)  # (B, N, H)


def build_tcn_adapter() -> TCNAdapter:
    return TCNAdapter(build_tcn(), num_inputs=1)
