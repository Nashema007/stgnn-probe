from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from model_test_utils import B, H, N, T, dssa_x, ring_adjacency, seed_all

from models import make_bigst, make_d2stgnn
from training.adapters import BigSTAdapter, D2STGNNAdapter


class _BigSTInputRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, feat: torch.Tensor | None = None):
        self.seen = x.detach().clone()
        return x.new_zeros(x.shape[0], x.shape[1], H), x.new_tensor(0.0)


class _D2STGNNInputRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x.detach().clone()
        return x.new_zeros(x.shape[0], x.shape[2], H)


def _weekday_probe_x() -> torch.Tensor:
    x = dssa_x()
    weekdays = torch.arange(T, device=x.device, dtype=x.dtype).remainder(7) / 7.0
    x[:, 2, :, :] = weekdays.view(1, 1, T).expand(B, N, T)
    return x


def test_bigst_adapter_converts_weekday_channel_to_integer_range() -> None:
    recorder = _BigSTInputRecorder()
    adapter = BigSTAdapter(recorder)

    out = adapter(_weekday_probe_x())

    assert out.shape == (B, N, H)
    assert recorder.seen is not None
    weekdays = recorder.seen[..., 2]
    assert torch.equal(weekdays, weekdays.round())
    assert int(weekdays.min()) == 0
    assert int(weekdays.max()) == 6


def test_d2stgnn_adapter_converts_weekday_channel_to_integer_range() -> None:
    recorder = _D2STGNNInputRecorder()
    adapter = D2STGNNAdapter(recorder)

    out = adapter(_weekday_probe_x())

    assert out.shape == (B, N, H)
    assert recorder.seen is not None
    weekdays = recorder.seen[..., 2]
    assert torch.equal(weekdays, weekdays.round())
    assert int(weekdays.min()) == 0
    assert int(weekdays.max()) == 6


def test_bigst_forward_shape_and_gradients() -> None:
    seed_all()
    adapter = BigSTAdapter(
        make_bigst(
            num_nodes=N,
            in_dim=3,
            hid_dim=2,
            node_dim=2,
            time_dim=2,
            num_layers=1,
            random_feature_dim=4,
            input_length=T,
            output_length=H,
            dropout=0.0,
            use_bn=False,
        )
    )

    out = adapter(dssa_x())
    loss = out.square().mean()
    loss.backward()

    grads = [p.grad for p in adapter.parameters() if p.requires_grad and p.grad is not None]
    assert out.shape == (B, N, H)
    assert grads
    assert any(torch.isfinite(g).all() and torch.any(g != 0) for g in grads)


def test_d2stgnn_forward_shape_and_gradients() -> None:
    seed_all()
    model = make_d2stgnn(
        torch.device("cpu"),
        num_nodes=N,
        adj_mx=[ring_adjacency(), ring_adjacency().T],
        num_feat=1,
        num_hidden=4,
        node_hidden=2,
        time_emb_dim=2,
        dropout=0.0,
        seq_length=H,
        in_seq_length=T,
        k_t=2,
        k_s=1,
        gap=H,
    )
    adapter = D2STGNNAdapter(model)

    out = adapter(dssa_x())
    loss = out.square().mean()
    loss.backward()

    grads = [p.grad for p in adapter.parameters() if p.requires_grad and p.grad is not None]
    assert out.shape == (B, N, H)
    assert grads
    assert any(torch.isfinite(g).all() and torch.any(g != 0) for g in grads)


def test_d2stgnn_rejects_horizon_not_divisible_by_gap() -> None:
    with pytest.raises(ValueError, match="seq_length to be divisible by gap"):
        make_d2stgnn(
            torch.device("cpu"),
            num_nodes=N,
            adj_mx=[ring_adjacency()],
            seq_length=5,
            gap=3,
        )
