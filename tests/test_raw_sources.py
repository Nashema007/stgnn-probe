from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.raw_sources import (
    build_probe_dataset_config,
    export_probe_dataset_inputs,
    load_distance_edges,
    load_raw_traffic_frame,
    load_sensor_coordinates,
)


def _write_adj(path: Path, sensor_ids: list[str], adj: np.ndarray | None = None) -> None:
    matrix = adj if adj is not None else np.eye(len(sensor_ids), dtype=np.float32)
    mapping = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    with path.open("wb") as f:
        pickle.dump((sensor_ids, mapping, matrix), f)


def test_gitignore_keeps_raw_payloads_local_and_allows_gitkeep() -> None:
    gitignore = Path(".gitignore").read_text().splitlines()

    assert "src/data/raw-data/*" in gitignore
    assert "!src/data/raw-data/.gitkeep" in gitignore


def test_metr_la_style_locations_are_ordered_by_adjacency_sensor_ids(tmp_path) -> None:
    raw_dir = tmp_path
    _write_adj(raw_dir / "adj_mx.pkl", ["b", "a"])
    pd.DataFrame(
        {
            "index": [0, 1],
            "sensor_id": ["a", "b"],
            "latitude": [34.1, 34.2],
            "longitude": [-118.1, -118.2],
        }
    ).to_csv(raw_dir / "graph_sensor_locations.csv", index=False)

    coords = load_sensor_coordinates("METR-LA", raw_dir=raw_dir)

    assert list(coords.columns) == ["node_id", "sensor_id", "latitude", "longitude"]
    assert coords.to_dict("records") == [
        {"node_id": 0, "sensor_id": "b", "latitude": 34.2, "longitude": -118.2},
        {"node_id": 1, "sensor_id": "a", "latitude": 34.1, "longitude": -118.1},
    ]


def test_pems_bay_style_locations_are_normalized(tmp_path) -> None:
    raw_dir = tmp_path
    _write_adj(raw_dir / "adj_mx_bay.pkl", ["400017", "400001"])
    pd.DataFrame(
        {
            "sensor_id": [400001, 400017],
            "latitude": [37.1, 37.2],
            "longitude": [-121.1, -121.2],
        }
    ).to_csv(raw_dir / "graph_sensor_locations_bay.csv", index=False)

    coords = load_sensor_coordinates("PEMS-BAY", raw_dir=raw_dir)

    assert coords.to_dict("records") == [
        {"node_id": 0, "sensor_id": "400017", "latitude": 37.2, "longitude": -121.2},
        {"node_id": 1, "sensor_id": "400001", "latitude": 37.1, "longitude": -121.1},
    ]


def test_missing_sensor_coordinates_raise_with_missing_ids(tmp_path) -> None:
    raw_dir = tmp_path
    _write_adj(raw_dir / "adj_mx.pkl", ["a", "missing"])
    pd.DataFrame(
        {
            "sensor_id": ["a"],
            "latitude": [34.1],
            "longitude": [-118.1],
        }
    ).to_csv(raw_dir / "graph_sensor_locations.csv", index=False)

    with pytest.raises(ValueError, match="missing"):
        load_sensor_coordinates("METR-LA", raw_dir=raw_dir)


def test_distance_edges_are_normalized(tmp_path) -> None:
    raw_dir = tmp_path
    pd.DataFrame(
        {
            "from": [1, 2],
            "to": [2, 1],
            "cost": [10.5, 11.5],
        }
    ).to_csv(raw_dir / "distances_bay_2017.csv", index=False)

    edges = load_distance_edges("PEMS-BAY", raw_dir=raw_dir)

    assert list(edges.columns) == ["from_sensor_id", "to_sensor_id", "cost"]
    assert edges.to_dict("records") == [
        {"from_sensor_id": "1", "to_sensor_id": "2", "cost": 10.5},
        {"from_sensor_id": "2", "to_sensor_id": "1", "cost": 11.5},
    ]


def test_load_raw_traffic_frame_aligns_columns_to_adjacency_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path
    _write_adj(raw_dir / "adj_mx.pkl", ["b", "a"])
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    monkeypatch.setattr(pd, "read_hdf", lambda path: frame)

    loaded = load_raw_traffic_frame("METR-LA", raw_dir=raw_dir)

    assert list(loaded.columns) == ["b", "a"]
    np.testing.assert_allclose(loaded.to_numpy(), np.array([[3.0, 1.0], [4.0, 2.0]]))


def test_export_probe_dataset_inputs_writes_standardized_files(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_adj(raw_dir / "adj_mx.pkl", ["b", "a"], np.array([[0.0, 2.0], [3.0, 0.0]]))
    pd.DataFrame(
        {
            "sensor_id": ["a", "b"],
            "latitude": [34.1, 34.2],
            "longitude": [-118.1, -118.2],
        }
    ).to_csv(raw_dir / "graph_sensor_locations.csv", index=False)
    pd.DataFrame({"from": ["a"], "to": ["b"], "cost": [12.5]}).to_csv(
        raw_dir / "distances_la_2012.csv",
        index=False,
    )
    monkeypatch.setattr(
        pd,
        "read_hdf",
        lambda path: pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
    )

    paths = export_probe_dataset_inputs("METR-LA", tmp_path / "probe", raw_dir=raw_dir)
    config = build_probe_dataset_config(
        "METR-LA",
        exported_dir=tmp_path / "probe",
        predictions_dir=tmp_path / "predictions",
        adjacency_dir=tmp_path / "model_adjacency",
        ground_truth_path=tmp_path / "ground_truth.npy",
    )

    assert paths["raw_data"] == tmp_path / "probe" / "metr-la_raw.npy"
    assert paths["coordinates"] == tmp_path / "probe" / "metr-la_coords.csv"
    assert paths["distance_edges"] == tmp_path / "probe" / "metr-la_distance_edges.csv"
    assert (
        paths["road_adjacency"] == tmp_path / "probe" / "adjacency" / "metr-la_road_adjacency.npy"
    )
    np.testing.assert_allclose(np.load(paths["raw_data"]), np.array([[3.0, 1.0], [4.0, 2.0]]))
    assert pd.read_csv(paths["coordinates"])["sensor_id"].astype(str).tolist() == ["b", "a"]
    np.testing.assert_allclose(np.load(paths["road_adjacency"]), np.array([[0.0, 2.0], [3.0, 0.0]]))
    assert config.raw_data == str(paths["raw_data"])
    assert config.coordinates == str(paths["coordinates"])
    assert config.ground_truth == str(tmp_path / "ground_truth.npy")
    assert config.predictions_dir == str(tmp_path / "predictions")
    assert config.adjacency_dir == str(tmp_path / "model_adjacency")
