"""Smoke tests for the shared trainer infrastructure.

All tests run on CPU with tiny synthetic data. No W&B login required
(wandb_mode="offline" is used when W&B is exercised, but use_wandb=False
by default so wandb is not needed at all).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.dataset import SlidingWindowDataset, load_dataset
from data.scaler import StandardScaler
from evaluation.metrics import (
    masked_mae,
    metric_global,
    metric_per_horizon,
    metric_per_node,
)
from training.adapters import (
    DSSATCNAdapter,
    GWNAdapter,
    STAWnetAdapter,
)

DEVICE = torch.device("cpu")
B, C, N, T, H = 2, 2, 3, 4, 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_data(t=90, n=N, c=C) -> np.ndarray:
    rng = np.random.default_rng(0)
    data = rng.random((t, n, c)).astype(np.float32)
    data[:, :, 0] *= 60  # speed-like values
    return data


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


@pytest.fixture
def scaler_and_data(synthetic_data):
    scaler = StandardScaler().fit(synthetic_data[:140])
    norm = scaler.transform(synthetic_data)
    return scaler, norm


@pytest.fixture
def dataloaders_and_scaler(tmp_path, synthetic_data):
    path = tmp_path / "data.npz"
    np.savez(str(path), data=synthetic_data)
    dl, sc = load_dataset(str(path), in_len=T, out_len=H, batch_size=B, num_workers=0)
    return dl, sc


# ---------------------------------------------------------------------------
# 1. Data module
# ---------------------------------------------------------------------------


def test_data_module_shapes(synthetic_data):
    scaler = StandardScaler().fit(synthetic_data[:140])
    norm = scaler.transform(synthetic_data)
    ds = SlidingWindowDataset(norm, in_len=T, out_len=H)
    x, y, y_full = ds[0]
    assert x.shape == (C, N, T)
    assert y.shape == (N, H)
    assert y_full.shape == (C, N, H)


def test_data_module_load_dataset(dataloaders_and_scaler):
    dl, sc = dataloaders_and_scaler
    assert set(dl.keys()) == {"train", "val", "test"}
    x, y, y_full = next(iter(dl["train"]))
    assert x.shape[1:] == (C, N, T)
    assert y.shape[1:] == (N, H)
    assert y_full.shape[1:] == (C, N, H)


# ---------------------------------------------------------------------------
# 2. Scaler
# ---------------------------------------------------------------------------


def test_scaler_roundtrip(synthetic_data):
    scaler = StandardScaler().fit(synthetic_data)
    norm = scaler.transform(synthetic_data)
    # Channel 0 is normalised; check mean ≈ 0
    assert abs(norm[:, :, 0].mean()) < 0.1
    # inverse_transform on a (B, N, H) tensor
    y = torch.from_numpy(norm[:H, :, 0].T.copy()).unsqueeze(0)  # (1, N, H)
    y_back = scaler.inverse_transform(y)
    orig = torch.from_numpy(synthetic_data[:H, :, 0].T.copy()).unsqueeze(0)
    assert torch.allclose(y_back, orig, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------


def test_metrics_shapes():
    pred = torch.rand(B, N, H)
    true = torch.rand(B, N, H)
    mae, mape, rmse = metric_global(pred, true)
    assert isinstance(mae, float)
    per_h = metric_per_horizon(pred, true)
    assert per_h["mae"].shape == (H,)
    per_n = metric_per_node(pred, true)
    assert per_n["mae"].shape == (N,)


def test_masked_mae_ignores_zeros():
    pred = torch.ones(2, 3, 4)
    true = torch.zeros(2, 3, 4)
    # All targets are zero → all masked → loss should be 0 (or nan-free)
    loss = masked_mae(pred, true, null_val=0.0)
    assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# 4. Adapters — all return (B, N, H)
# ---------------------------------------------------------------------------


def _dummy_x():
    return torch.randn(B, C, N, T)


def _dummy_y_full():
    return torch.randn(B, C, N, H)


class _FakeGWN(nn.Module):
    """Minimal stand-in: output (B, H, N, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, edge_index=None, edge_weight=None):
        return x.new_ones(x.size(0), H, N, 1) * self.bias


class _FakeDSSATCN(nn.Module):
    """Input (B, T, N, C), output (B, H, N, 1)."""

    def forward(self, x):
        return torch.zeros(x.size(0), H, N, 1)


class _FakeSTAWnet(nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), H, N, 1)


def test_gwn_adapter():
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    adapter = GWNAdapter(_FakeGWN(), no_edges)
    out = adapter(_dummy_x())
    assert out.shape == (B, N, H)


def test_dssa_tcn_adapter():
    adapter = DSSATCNAdapter(_FakeDSSATCN())
    out = adapter(torch.randn(B, 3, N, T))
    assert out.shape == (B, N, H)


def test_stawnet_adapter():
    adapter = STAWnetAdapter(_FakeSTAWnet())
    out = adapter(_dummy_x())
    assert out.shape == (B, N, H)
