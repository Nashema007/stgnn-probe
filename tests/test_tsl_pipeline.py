"""Tests for the canonical-scaler-reuse contract in src/data/tsl_pipeline.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import data.tsl_pipeline as tsl_pipeline
from data.scaler import StandardScaler


def _fake_load_tsl_dataset_and_graph(
    dataset_name: str, root: object
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    raw = (rng.random((200, 4, 3)).astype(np.float32) * 50.0) + 10.0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.ones(4, dtype=torch.float32)
    return raw, edge_index, edge_weight


@pytest.fixture(autouse=True)
def _stub_dataset_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid downloading/loading a real tsl dataset; the dataset/graph content
    is irrelevant to this module's own scaler-handling logic."""
    monkeypatch.setattr(
        tsl_pipeline, "_load_tsl_dataset_and_graph", _fake_load_tsl_dataset_and_graph
    )


def test_build_tsl_pipeline_reuses_provided_scaler() -> None:
    """A passed-in scaler is used as-is, never refit at this pipeline's own out_len."""
    canonical_scaler = StandardScaler().fit(np.zeros((1, 4, 3), dtype=np.float32))

    pipeline = tsl_pipeline.build_tsl_pipeline(
        "METR-LA", in_len=4, out_len=6, batch_size=2, scaler=canonical_scaler
    )

    assert pipeline.scaler is canonical_scaler
    np.testing.assert_array_equal(pipeline.scaler.mean, canonical_scaler.mean)
    np.testing.assert_array_equal(pipeline.scaler.std, canonical_scaler.std)


def test_build_tsl_pipeline_without_scaler_fits_its_own_out_len_specific_one() -> None:
    """Default (no scaler passed) behaviour fits independently per out_len.

    This is exactly the cross-horizon mismatch run_training.py's canonical
    `eval_pipeline.scaler` reuse exists to avoid: two pipelines built at
    different `out_len` values, with no shared scaler passed, fit different
    statistics from `_ratio_split_train_len`'s out_len-dependent train
    boundary.
    """
    pipeline_a = tsl_pipeline.build_tsl_pipeline("METR-LA", in_len=4, out_len=6, batch_size=2)
    pipeline_b = tsl_pipeline.build_tsl_pipeline("METR-LA", in_len=4, out_len=40, batch_size=2)

    assert not np.allclose(pipeline_a.scaler.mean, pipeline_b.scaler.mean)


def test_build_tsl_pipeline_with_shared_scaler_normalises_identically_across_out_len() -> None:
    """Passing the same scaler to two different out_len pipelines makes their
    normalised target data byte-for-byte identical — the actual property
    run_training.py depends on for evaluation to be valid."""
    canonical_scaler = StandardScaler().fit(np.zeros((1, 4, 3), dtype=np.float32))

    pipeline_a = tsl_pipeline.build_tsl_pipeline(
        "METR-LA", in_len=4, out_len=6, batch_size=2, scaler=canonical_scaler
    )
    pipeline_b = tsl_pipeline.build_tsl_pipeline(
        "METR-LA", in_len=4, out_len=40, batch_size=2, scaler=canonical_scaler
    )

    np.testing.assert_array_equal(pipeline_a.scaler.mean, pipeline_b.scaler.mean)
    np.testing.assert_array_equal(pipeline_a.scaler.std, pipeline_b.scaler.std)


class _FakeTslDataset:
    """Stand-in for tsl.datasets.MetrLA/PemsBay — only the attributes/methods
    export_probe_dataset_inputs_from_tsl actually touches."""

    def __init__(self, root: str) -> None:
        self.root_dir = root
        self.nodes = pd.Index(["20", "10", "30"])  # deliberately not sorted
        # (T, N, 1) — matches dataset.numpy()'s real ndim
        self._value = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
        # node order matches self.nodes: ["20", "10", "30"]
        self.dist = np.array(
            [
                [0.0, 5.0, np.inf],
                [5.0, 0.0, np.inf],
                [np.inf, np.inf, 0.0],
            ]
        )

    def numpy(self) -> np.ndarray:
        return self._value

    def get_connectivity(self, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        # One directed edge 20->10 (node indices 0->1), matching self.dist.
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_weight = torch.tensor([0.9], dtype=torch.float32)
        return edge_index, edge_weight


def test_export_probe_dataset_inputs_from_tsl_aligns_coordinates_to_dataset_node_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinates must be reindexed to dataset.nodes's order — not assumed
    from locations.csv's row order, which this test deliberately scrambles."""
    cache_dir = tmp_path / "tsl_cache"
    cache_dir.mkdir()
    # Row order deliberately differs from _FakeTslDataset.nodes ("20","10","30").
    pd.DataFrame(
        {
            "sensor_id": ["10", "30", "20"],
            "latitude": [1.1, 3.3, 2.2],
            "longitude": [-1.1, -3.3, -2.2],
        }
    ).to_csv(cache_dir / "locations.csv", index=False)

    monkeypatch.setattr(tsl_pipeline, "_resolve_tsl_dataset_cls", lambda name: _FakeTslDataset)

    output_dir = tmp_path / "probe_inputs"
    paths = tsl_pipeline.export_probe_dataset_inputs_from_tsl(
        "METR-LA", output_dir, tsl_cache_dir=str(cache_dir)
    )

    # read_csv parses numeric-looking sensor ids back as int64 — the same
    # round-trip quirk the legacy exporter's output has; cast for comparison.
    coords = pd.read_csv(paths["coordinates"], dtype={"sensor_id": str})
    assert list(coords["sensor_id"]) == ["20", "10", "30"]
    np.testing.assert_allclose(coords["latitude"], [2.2, 1.1, 3.3])
    assert list(coords["node_id"]) == [0, 1, 2]

    raw = np.load(paths["raw_data"])
    assert raw.shape == (2, 3)  # (T, N) — squeezed from (T, N, 1)

    edges = pd.read_csv(paths["distance_edges"], dtype={"from_sensor_id": str, "to_sensor_id": str})
    # dist has 4 finite (incl. diagonal) entries among the off-diagonal-finite
    # pair plus 3 self-loops: (0,0),(0,1),(1,0),(1,1),(2,2) = 5 finite entries.
    assert len(edges) == 5
    assert set(zip(edges["from_sensor_id"], edges["to_sensor_id"], strict=True)) == {
        ("20", "20"),
        ("20", "10"),
        ("10", "20"),
        ("10", "10"),
        ("30", "30"),
    }

    road_adj = np.load(paths["road_adjacency"])
    assert road_adj.shape == (3, 3)
    assert road_adj[0, 1] == pytest.approx(0.9)  # node 0 ("20") -> node 1 ("10")
    assert road_adj.sum() == pytest.approx(0.9)  # the only edge get_connectivity returned


def test_export_probe_dataset_inputs_from_tsl_raises_when_locations_csv_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tsl_pipeline, "_resolve_tsl_dataset_cls", lambda name: _FakeTslDataset)

    with pytest.raises(FileNotFoundError, match="locations.csv"):
        tsl_pipeline.export_probe_dataset_inputs_from_tsl(
            "METR-LA", tmp_path / "probe_inputs", tsl_cache_dir=str(tmp_path / "empty_cache")
        )
