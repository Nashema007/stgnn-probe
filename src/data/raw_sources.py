"""Adapters from repo-local raw traffic files to STGNN-Probe inputs.

This module is intentionally outside ``analysis``.  STGNN-Probe consumes
standardized ``.npy`` and ``.csv`` files; only this data layer knows about the
legacy METR-LA / PEMS-BAY raw file names and schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import DatasetConfig

from .adjacency import load_adj

RAW_DATA_DIR = Path(__file__).resolve().parent / "raw-data"


@dataclass(frozen=True)
class RawDatasetSpec:
    name: str
    traffic_filename: str
    adjacency_filename: str
    locations_filename: str
    distances_filename: str
    sensor_ids_filename: str | None = None

    @property
    def slug(self) -> str:
        return self.name.lower()


RAW_DATASETS: dict[str, RawDatasetSpec] = {
    "METR-LA": RawDatasetSpec(
        name="METR-LA",
        traffic_filename="metr-la.h5",
        adjacency_filename="adj_mx.pkl",
        locations_filename="graph_sensor_locations.csv",
        distances_filename="distances_la_2012.csv",
        sensor_ids_filename="graph_sensor_ids.txt",
    ),
    "PEMS-BAY": RawDatasetSpec(
        name="PEMS-BAY",
        traffic_filename="pems-bay.h5",
        adjacency_filename="adj_mx_bay.pkl",
        locations_filename="graph_sensor_locations_bay.csv",
        distances_filename="distances_bay_2017.csv",
    ),
}

_ALIASES = {
    "METRLA": "METR-LA",
    "METR_LA": "METR-LA",
    "METR-LA": "METR-LA",
    "PEMSBAY": "PEMS-BAY",
    "PEMS_BAY": "PEMS-BAY",
    "PEMS-BAY": "PEMS-BAY",
}


def get_raw_dataset_spec(dataset_name: str) -> RawDatasetSpec:
    key = dataset_name.upper().replace(" ", "-")
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unknown raw dataset {dataset_name!r}. Expected one of {sorted(RAW_DATASETS)}."
        )
    return RAW_DATASETS[canonical]


def _path(raw_dir: str | Path, filename: str) -> Path:
    return Path(raw_dir) / filename


def _string_ids(values) -> list[str]:
    return [str(value) for value in values]


def load_raw_adjacency(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Load legacy adjacency pickle and normalize sensor ids to strings."""
    spec = get_raw_dataset_spec(dataset_name)
    sensor_ids, sensor_id_to_ind, adjacency = load_adj(str(_path(raw_dir, spec.adjacency_filename)))
    ordered_ids = _string_ids(sensor_ids)
    id_to_index = {str(sensor_id): int(index) for sensor_id, index in sensor_id_to_ind.items()}
    if set(id_to_index) != set(ordered_ids):
        id_to_index = {sensor_id: idx for idx, sensor_id in enumerate(ordered_ids)}
    return ordered_ids, id_to_index, np.asarray(adjacency, dtype=np.float32)


def load_sensor_coordinates(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load and align sensor coordinates to adjacency order.

    Returns columns ``node_id``, ``sensor_id``, ``latitude``, ``longitude``.
    """
    spec = get_raw_dataset_spec(dataset_name)
    sensor_ids, _, _ = load_raw_adjacency(dataset_name, raw_dir=raw_dir)
    coords = pd.read_csv(_path(raw_dir, spec.locations_filename))
    required = {"sensor_id", "latitude", "longitude"}
    missing_columns = required - set(coords.columns)
    if missing_columns:
        raise ValueError(
            "Sensor coordinate file is missing columns: " + ", ".join(sorted(missing_columns))
        )

    coords = coords.assign(sensor_id=coords["sensor_id"].astype(str))
    indexed = coords.set_index("sensor_id", drop=False)
    missing_ids = [sensor_id for sensor_id in sensor_ids if sensor_id not in indexed.index]
    if missing_ids:
        raise ValueError(
            "Sensor coordinate file is missing sensor ids: " + ", ".join(missing_ids[:10])
        )

    ordered = indexed.loc[sensor_ids, ["sensor_id", "latitude", "longitude"]].reset_index(drop=True)
    ordered.insert(0, "node_id", np.arange(len(ordered), dtype=np.int32))
    return ordered[["node_id", "sensor_id", "latitude", "longitude"]]


def load_distance_edges(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load road-distance edge CSV with canonical column names."""
    spec = get_raw_dataset_spec(dataset_name)
    edges = pd.read_csv(_path(raw_dir, spec.distances_filename))
    required = {"from", "to", "cost"}
    missing_columns = required - set(edges.columns)
    if missing_columns:
        raise ValueError(
            "Distance edge file is missing columns: " + ", ".join(sorted(missing_columns))
        )
    return pd.DataFrame(
        {
            "from_sensor_id": edges["from"].astype(str),
            "to_sensor_id": edges["to"].astype(str),
            "cost": edges["cost"].astype(float),
        }
    )


def load_raw_traffic_frame(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load traffic HDF and align columns to adjacency sensor order."""
    spec = get_raw_dataset_spec(dataset_name)
    sensor_ids, _, _ = load_raw_adjacency(dataset_name, raw_dir=raw_dir)
    try:
        frame = pd.read_hdf(_path(raw_dir, spec.traffic_filename))
    except ImportError as exc:
        raise ImportError(
            "Reading raw traffic HDF files requires PyTables. "
            "Install the optional raw-data dependencies with: "
            'uv pip install -e ".[raw-data]"'
        ) from exc

    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    if set(sensor_ids).issubset(frame.columns):
        return frame.loc[:, sensor_ids]
    if frame.shape[1] == len(sensor_ids):
        frame.columns = sensor_ids
        return frame
    missing_ids = [sensor_id for sensor_id in sensor_ids if sensor_id not in frame.columns]
    raise ValueError(
        "Raw traffic frame columns do not match adjacency sensor ids. "
        f"Missing examples: {', '.join(missing_ids[:10])}"
    )


def export_probe_dataset_inputs(
    dataset_name: str,
    output_dir: str | Path,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> dict[str, Path]:
    """Export raw legacy files to STGNN-Probe's standardized input files."""
    spec = get_raw_dataset_spec(dataset_name)
    out = Path(output_dir)
    adjacency_dir = out / "adjacency"
    adjacency_dir.mkdir(parents=True, exist_ok=True)

    raw_traffic = load_raw_traffic_frame(spec.name, raw_dir=raw_dir)
    coords = load_sensor_coordinates(spec.name, raw_dir=raw_dir)
    distance_edges = load_distance_edges(spec.name, raw_dir=raw_dir)
    _, _, road_adjacency = load_raw_adjacency(spec.name, raw_dir=raw_dir)

    paths = {
        "raw_data": out / f"{spec.slug}_raw.npy",
        "coordinates": out / f"{spec.slug}_coords.csv",
        "distance_edges": out / f"{spec.slug}_distance_edges.csv",
        "road_adjacency": adjacency_dir / f"{spec.slug}_road_adjacency.npy",
    }

    np.save(paths["raw_data"], raw_traffic.to_numpy(dtype=np.float32))
    coords.to_csv(paths["coordinates"], index=False)
    distance_edges.to_csv(paths["distance_edges"], index=False)
    np.save(paths["road_adjacency"], road_adjacency)
    return paths


def build_probe_dataset_config(
    dataset_name: str,
    exported_dir: str | Path,
    predictions_dir: str | Path,
    adjacency_dir: str | Path,
    ground_truth_path: str | Path,
) -> DatasetConfig:
    """Build a ``DatasetConfig`` from exported STGNN-Probe raw inputs."""
    spec = get_raw_dataset_spec(dataset_name)
    exported = Path(exported_dir)
    coords_path = exported / f"{spec.slug}_coords.csv"
    if coords_path.exists():
        num_nodes = len(pd.read_csv(coords_path))
    else:
        sensor_ids, _, _ = load_raw_adjacency(spec.name)
        num_nodes = len(sensor_ids)
    return DatasetConfig(
        name=spec.name,
        num_nodes=num_nodes,
        raw_data=str(exported / f"{spec.slug}_raw.npy"),
        coordinates=str(coords_path),
        ground_truth=str(Path(ground_truth_path)),
        predictions_dir=str(Path(predictions_dir)),
        adjacency_dir=str(Path(adjacency_dir)),
    )
