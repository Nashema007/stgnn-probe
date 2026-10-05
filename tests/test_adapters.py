from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from model_test_utils import (
    B,
    H,
    N,
    T,
    build_staeformer_adapter,
    canonical_x,
    dssa_x,
    graph_model_specs,
)

from training.adapters import DSSATCNAdapter, GWNAdapter, STAEformerAdapter

_NO_EDGES = torch.zeros(2, 0, dtype=torch.long)


class _ShapeRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, *args: object) -> torch.Tensor:
        self.seen_shape = tuple(x.shape)
        return torch.zeros(B, H, N, 1)


class _BadNativeOutput(nn.Module):
    def forward(self, x: torch.Tensor, *args: object) -> torch.Tensor:
        return torch.zeros(B, H, N)


@pytest.mark.parametrize(("name", "build_adapter", "make_x"), graph_model_specs())
def test_graph_adapters_return_canonical_shape(name, build_adapter, make_x) -> None:
    adapter = build_adapter()
    adapter.eval()
    x = make_x()
    out = adapter(x)
    assert out.shape == (B, N, H)
    assert torch.isfinite(out).all()


def test_adapter_rejects_non_canonical_input_rank() -> None:
    adapter = GWNAdapter(_ShapeRecorder(), _NO_EDGES)

    with pytest.raises(ValueError, match="canonical input"):
        adapter(torch.randn(B, N, T))


def test_adapter_rejects_native_output_without_final_singleton_dim() -> None:
    adapter = GWNAdapter(_BadNativeOutput(), _NO_EDGES)

    with pytest.raises(ValueError, match="native output"):
        adapter(canonical_x(1))


def test_dssa_adapter_converts_canonical_input_to_time_major_layout() -> None:
    recorder = _ShapeRecorder()
    adapter = DSSATCNAdapter(recorder)

    out = adapter(dssa_x())

    assert recorder.seen_shape == (B, T, N, 3)
    assert out.shape == (B, N, H)


def test_dssa_adapter_requires_time_feature_channels() -> None:
    adapter = DSSATCNAdapter(_ShapeRecorder())

    with pytest.raises(ValueError, match="at least 3 channels"):
        adapter(canonical_x(2))


def test_staeformer_adapter_converts_canonical_input_to_time_major_layout() -> None:
    recorder = _ShapeRecorder()
    adapter = STAEformerAdapter(recorder)

    out = adapter(dssa_x())

    assert recorder.seen_shape == (B, T, N, 3)
    assert out.shape == (B, N, H)


def test_staeformer_adapter_ignores_optional_args() -> None:
    adapter = build_staeformer_adapter()
    adapter.eval()

    out = adapter(dssa_x(), y_full=None, batches_seen=None, task_level=H)

    assert out.shape == (B, N, H)
