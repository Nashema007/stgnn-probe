"""Data pipeline built on tsl.datasets + tsl.data abstractions.

Loads ``tsl.datasets.MetrLA``/``PemsBay`` (auto-downloaded by tsl on first
use — a separate copy of the data from the local ``data/metr-la.npz``/
``data/pems-bay.npz`` files this framework used previously), builds a
``SpatioTemporalDataset`` with the dataset's own graph connectivity, and
bridges each batch into the framework's existing canonical ``(x, y, y_full)``
3-tuple contract so the unmodified ``GraphTrainer`` can drive training.

Every model trains on the same 3-channel layout the local npz files used to
provide: ``[value, time-of-day fraction, day-of-week fraction]``, derived
from each tsl dataset's own ``DatetimeIndex``. ``time-of-day`` is the
fraction of the day elapsed (``[0, 1)``) and ``day-of-week`` is the weekday
index divided by 7 — the exact convention ``models/staeformer.py`` indexes
its ``tod``/``dow`` embeddings with (``(tod * steps_per_day).long()`` and
``(dow * 7).long()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.datasets import MetrLA, PemsBay

from .dataset import DEFAULT_WALK_FORWARD_BOUNDARIES
from .scaler import StandardScaler

_TSL_DATASETS = {"METR-LA": MetrLA, "PEMS-BAY": PemsBay}

DEFAULT_VAL_LEN = 0.1
DEFAULT_TEST_LEN = 0.2


@dataclass(frozen=True)
class TslPipeline:
    """Dataloaders, scaler, and static graph for a tsl-backed training run."""

    dataloaders: dict[str, _BridgeLoader]
    scaler: StandardScaler
    edge_index: Tensor
    edge_weight: Tensor | None


@dataclass(frozen=True)
class TslWalkForwardFold:
    """Dataloaders and scaler for one tsl-backed walk-forward validation fold."""

    index: int
    boundaries: tuple[int, int, int]
    dataloaders: dict[str, _BridgeLoader]
    scaler: StandardScaler


class _BridgeLoader:
    """Iterates a tsl ``StaticGraphLoader``, yielding canonical 3-tuples.

    Duck-types the ``GraphTrainer`` dataloader contract (``__iter__`` and
    ``__len__``) without being a literal ``torch.utils.data.DataLoader``.
    """

    def __init__(self, tsl_loader: Any) -> None:
        self._loader = tsl_loader

    def __iter__(self):
        for batch in self._loader:
            x = batch.input.x.permute(0, 3, 2, 1)  # (B,T,N,C) -> (B,C,N,T)
            y_full = batch.target.y.permute(0, 3, 2, 1)  # (B,H,N,C) -> (B,C,N,H)
            y = y_full[:, 0]  # (B,N,H)
            yield x, y, y_full

    def __len__(self) -> int:
        return len(self._loader)


def _resolve_tsl_dataset_cls(dataset_name: str) -> type:
    key = dataset_name.upper()
    if key not in _TSL_DATASETS:
        raise ValueError(
            f"No tsl dataset mapped for {dataset_name!r}. Expected one of {sorted(_TSL_DATASETS)}."
        )
    return _TSL_DATASETS[key]


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace("-", "_")


def export_probe_dataset_inputs_from_tsl(
    dataset_name: str,
    output_dir: str | Path,
    tsl_cache_dir: str | Path = "data/tsl_cache",
) -> dict[str, Path]:
    """Export STGNN-Probe's standardized raw inputs from tsl's own cache.

    Produces the same output contract as
    ``data.raw_sources.export_probe_dataset_inputs`` (same keys/filenames —
    ``analysis.probe``/``build_probe_dataset_config`` need no changes), but
    sources everything from tsl's already-downloaded dataset (the same one
    every model trains against) instead of requiring a separate manual
    download of the legacy DCRNN raw files. tsl's own ``build()`` step
    happens to leave ``locations.csv`` (sensor coordinates) in its cache
    directory even though it isn't exposed as a dataset attribute, so that's
    the one piece still read from a file rather than a tsl API.

    ``road_adjacency`` is built via the dataset's own ``get_connectivity``
    call — the exact same method/threshold ``_load_tsl_dataset_and_graph``
    uses to build the adjacency every spatial model actually trains on, so
    probe analysis and training are guaranteed to agree on the graph rather
    than risk two independent reconstructions silently diverging.

    Node ordering is taken from ``dataset.nodes`` and used to reindex
    ``locations.csv`` explicitly (not assumed from row order), the same
    defensive alignment ``data.raw_sources.load_sensor_coordinates`` already
    does for the legacy path.
    """
    import pandas as pd

    dataset_cls = _resolve_tsl_dataset_cls(dataset_name)
    dataset = dataset_cls(root=str(tsl_cache_dir))

    node_ids = [str(n) for n in dataset.nodes]
    n = len(node_ids)

    value = dataset.numpy()
    if value.ndim == 3:
        value = value[:, :, 0]
    raw_traffic = np.asarray(value, dtype=np.float32)  # (T, N)

    locations_path = Path(dataset.root_dir) / "locations.csv"
    if not locations_path.exists():
        raise FileNotFoundError(
            f"Expected tsl to have left {locations_path} after downloading/building "
            f"{dataset_name} — found none. Try deleting {tsl_cache_dir} and re-running "
            "training once to force a fresh download/build."
        )
    raw_locations = pd.read_csv(locations_path)
    if "sensor_id" not in raw_locations.columns:
        # tsl's PEMS-BAY archive ships sensor_locations_bay.csv with no header row
        # (unlike METR-LA's sensor_locations_la.csv, which has one) — pandas' default
        # header=0 otherwise swallows the first sensor as a header.
        raw_locations = pd.read_csv(
            locations_path, header=None, names=["sensor_id", "latitude", "longitude"]
        )
    raw_locations = raw_locations.assign(sensor_id=raw_locations["sensor_id"].astype(str))
    indexed = raw_locations.set_index("sensor_id", drop=False)
    missing = [node_id for node_id in node_ids if node_id not in indexed.index]
    if missing:
        raise ValueError(
            f"{locations_path} is missing sensor ids present in the dataset: {missing[:10]}"
        )
    coords = indexed.loc[node_ids, ["sensor_id", "latitude", "longitude"]].reset_index(drop=True)
    coords.insert(0, "node_id", np.arange(len(coords), dtype=np.int32))

    dist = np.asarray(dataset.dist, dtype=np.float64)
    finite_i, finite_j = np.where(np.isfinite(dist))
    distance_edges = pd.DataFrame(
        {
            "from_sensor_id": [node_ids[i] for i in finite_i],
            "to_sensor_id": [node_ids[j] for j in finite_j],
            "cost": dist[finite_i, finite_j],
        }
    )

    edge_index, edge_weight = dataset.get_connectivity(
        method="distance",
        threshold=0.1,
        include_self=False,
        normalize_axis=1,
        layout="edge_index",
    )
    ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
    ew = edge_weight.numpy() if hasattr(edge_weight, "numpy") else np.asarray(edge_weight)
    road_adjacency = np.zeros((n, n), dtype=np.float32)
    road_adjacency[ei[0], ei[1]] = ew

    slug = _dataset_slug(dataset_name)
    out = Path(output_dir)
    adjacency_dir = out / "adjacency"
    adjacency_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw_data": out / f"{slug}_raw.npy",
        "coordinates": out / f"{slug}_coords.csv",
        "distance_edges": out / f"{slug}_distance_edges.csv",
        "road_adjacency": adjacency_dir / f"{slug}_road_adjacency.npy",
    }
    np.save(paths["raw_data"], raw_traffic)
    coords.to_csv(paths["coordinates"], index=False)
    distance_edges.to_csv(paths["distance_edges"], index=False)
    np.save(paths["road_adjacency"], road_adjacency)
    return paths


def _load_tsl_dataset_and_graph(
    dataset_name: str,
    root: str | Path,
) -> tuple[np.ndarray, Any, Any]:
    """Load a tsl dataset, returning ``(raw, edge_index, edge_weight)``.

    ``raw`` is shaped ``(T, N, 3)``: ``[value, time-of-day, day-of-week]``.
    """
    dataset_cls = _resolve_tsl_dataset_cls(dataset_name)
    dataset = dataset_cls(root=str(root))

    value = dataset.numpy()  # (T, N) or (T, N, 1)
    if value.ndim == 2:
        value = value[:, :, None]
    n_steps, n_nodes, _ = value.shape

    index = dataset.index
    tod = (index.hour * 3600 + index.minute * 60 + index.second).to_numpy(
        dtype=np.float32
    ) / 86400.0
    dow = index.dayofweek.to_numpy(dtype=np.float32) / 7.0
    tod = np.broadcast_to(tod[:, None, None], (n_steps, n_nodes, 1))
    dow = np.broadcast_to(dow[:, None, None], (n_steps, n_nodes, 1))
    raw = np.concatenate([value, tod, dow], axis=-1).astype(np.float32)

    edge_index, edge_weight = dataset.get_connectivity(
        method="distance",
        threshold=0.1,
        include_self=False,
        normalize_axis=1,
        layout="edge_index",
    )
    return raw, edge_index, edge_weight


def _ratio_split_train_len(
    n_steps: int,
    window: int,
    horizon: int,
    val_len: float,
    test_len: float,
) -> int:
    """Raw-row boundary safe for fitting the scaler without leaking val/test rows.

    Replicates ``TemporalSplitter.fit``'s ratio math exactly: ratios are
    taken over the *windowed sample count* (``n_samples``), matching tsl's
    own ``len(dataset)`` basis, not raw timesteps — and a
    ``samples_offset == window`` gap (``SpatioTemporalDataset.samples_offset``
    with the default ``stride=1`` ``build_tsl_pipeline`` uses) is subtracted
    from the train boundary, the same gap tsl uses to guarantee no train
    window overlaps the val split. Without that gap, fitting the scaler on
    ``raw[:train_len]`` would include rows only reachable through a val
    window — a small leak from val into the scaler's fit data.
    """
    n_samples = n_steps - (window + horizon) + 1
    samples_offset = window
    t = int(test_len) if test_len >= 1 else int(test_len * n_samples)
    v = int(val_len) if val_len >= 1 else int(val_len * (n_samples - t))
    test_start = n_samples - t
    val_start = test_start - v
    return max(val_start - samples_offset, 0)


def build_tsl_pipeline(
    dataset_name: str,
    in_len: int = 12,
    out_len: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    root: str | Path = "data/tsl_cache",
    val_len: float = DEFAULT_VAL_LEN,
    test_len: float = DEFAULT_TEST_LEN,
    scaler: StandardScaler | None = None,
) -> TslPipeline:
    """Build dataloaders/scaler/graph for any model from tsl's own dataset classes.

    ``scaler``:
        Optional pre-fitted scaler to reuse instead of fitting a new one.
        Callers that train multiple models at different ``out_len`` values
        but evaluate them all on one canonical shared test-window set (see
        ``scripts/run_training.py``) must pass the *same* scaler — fit once,
        at the largest ``out_len`` used — to every one of those calls.
        ``_ratio_split_train_len``'s train boundary is monotonically
        non-increasing in ``out_len``, so the largest-``out_len`` boundary is
        always a safe (no val/test leakage) choice to reuse for smaller
        ``out_len`` values too. Without this, each ``out_len`` would fit its
        own slightly different scaler, but every model would still be
        *evaluated* through one shared canonical loader normalised with
        whichever scaler built that loader — silently feeding inputs
        normalised with the wrong statistics into models trained on a
        different normalisation.
    """
    raw, edge_index, edge_weight = _load_tsl_dataset_and_graph(dataset_name, root)

    if scaler is None:
        train_len = _ratio_split_train_len(raw.shape[0], in_len, out_len, val_len, test_len)
        scaler = StandardScaler().fit(raw[:train_len])
    normalized = scaler.transform(raw)

    torch_dataset = SpatioTemporalDataset(
        target=normalized,
        connectivity=(edge_index, edge_weight),
        window=in_len,
        horizon=out_len,
    )

    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)
    dm = SpatioTemporalDataModule(
        torch_dataset,
        splitter=splitter,
        batch_size=batch_size,
        workers=num_workers,
    )
    dm.setup()

    dataloaders = {
        "train": _BridgeLoader(dm.train_dataloader(shuffle=True)),
        "val": _BridgeLoader(dm.val_dataloader(shuffle=False)),
        "test": _BridgeLoader(dm.test_dataloader(shuffle=False)),
    }

    return TslPipeline(
        dataloaders=dataloaders,
        scaler=scaler,
        edge_index=torch.as_tensor(torch_dataset.edge_index),
        edge_weight=(
            torch.as_tensor(torch_dataset.edge_weight)
            if torch_dataset.edge_weight is not None
            else None
        ),
    )


def compute_test_window_bounds(
    n_steps: int,
    in_len: int,
    out_len: int,
    val_len: float = DEFAULT_VAL_LEN,
    test_len: float = DEFAULT_TEST_LEN,
) -> tuple[int, int]:
    """Return ``(test_start, test_count)`` for the tsl test split, in raw-row space.

    ``test_start`` is the raw-timestep row index of the first test window's
    input-window start; ``test_count`` is the number of test windows.
    Replicates ``tsl.data.datamodule.TemporalSplitter``'s ratio math directly
    against the raw timeline (assuming the default ``stride=1``, ``delay=0``
    used by ``build_tsl_pipeline``'s ``SpatioTemporalDataset``), so callers
    that need to align to the exact same test windows tsl produces — without
    constructing a full ``SpatioTemporalDataset`` — can do so. This is used to
    align ARIMA's forecast origins to the same canonical test windows the
    neural models are evaluated on.
    """
    n_samples = n_steps - (in_len + out_len) + 1
    test_count = int(test_len) if test_len >= 1 else int(test_len * n_samples)
    test_start = n_samples - test_count
    return test_start, test_count


def load_tsl_raw_array(dataset_name: str, root: str | Path = "data/tsl_cache") -> np.ndarray:
    """Return the raw ``(T, N, 3)`` ``[value, tod, dow]`` array for *dataset_name*.

    Used by ARIMA and ground-truth construction, which need the unwindowed
    array rather than a dataloader.
    """
    raw, _, _ = _load_tsl_dataset_and_graph(dataset_name, root)
    return raw


def build_tsl_walk_forward_pipeline(
    dataset_name: str,
    boundaries: list[int] | tuple[int, ...] | None = None,
    in_len: int = 12,
    out_len: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    root: str | Path = "data/tsl_cache",
) -> tuple[list[TslWalkForwardFold], Tensor, Tensor | None]:
    """Build walk-forward folds from tsl's own dataset.

    Mirrors ``data.dataset.load_walk_forward_datasets``'s boundary semantics
    (each fold uses three consecutive boundaries ``(a, b, c)``: train
    ``data[:a]``, validation ``data[a:b]``, test ``data[b:c]``) but sources
    the raw array and graph connectivity from tsl instead of a local file.
    """
    raw, edge_index, edge_weight = _load_tsl_dataset_and_graph(dataset_name, root)

    resolved_boundaries = list(boundaries) if boundaries is not None else None
    if resolved_boundaries is None:
        key = dataset_name.upper()
        if key not in DEFAULT_WALK_FORWARD_BOUNDARIES:
            raise ValueError(
                f"No default walk-forward boundaries for dataset {dataset_name!r}. "
                "Pass explicit boundaries instead."
            )
        resolved_boundaries = DEFAULT_WALK_FORWARD_BOUNDARIES[key]
    if len(resolved_boundaries) < 3:
        raise ValueError("At least three walk-forward boundaries are required.")
    if resolved_boundaries[-1] > raw.shape[0]:
        raise ValueError(
            f"Last boundary ({resolved_boundaries[-1]}) exceeds data length ({raw.shape[0]})."
        )

    folds: list[TslWalkForwardFold] = []
    for fold_idx in range(len(resolved_boundaries) - 2):
        a, b, c = resolved_boundaries[fold_idx : fold_idx + 3]
        splits = {"train": raw[:a], "val": raw[a:b], "test": raw[b:c]}
        scaler = StandardScaler().fit(splits["train"])

        dataloaders: dict[str, _BridgeLoader] = {}
        for split_name, split_raw in splits.items():
            normalized = scaler.transform(split_raw)
            ds = SpatioTemporalDataset(
                target=normalized,
                connectivity=(edge_index, edge_weight),
                window=in_len,
                horizon=out_len,
            )
            dm = SpatioTemporalDataModule(ds, batch_size=batch_size, workers=num_workers)
            loader = dm.get_dataloader(split=None, shuffle=(split_name == "train"))
            dataloaders[split_name] = _BridgeLoader(loader)

        folds.append(
            TslWalkForwardFold(
                index=fold_idx,
                boundaries=(int(a), int(b), int(c)),
                dataloaders=dataloaders,
                scaler=scaler,
            )
        )

    return (
        folds,
        torch.as_tensor(edge_index),
        torch.as_tensor(edge_weight) if edge_weight is not None else None,
    )
