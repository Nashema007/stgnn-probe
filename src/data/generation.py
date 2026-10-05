"""Graph WaveNet-style data generation utilities.

The legacy Graph WaveNet scripts generated two data layouts:

* pre-windowed ``train.npz`` / ``val.npz`` / ``test.npz`` files for the
  standard validation setup;
* a raw feature ``<dataset>.npz`` file for walk-forward validation, where
  folds are windowed later from contiguous time ranges.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PathLike = str | os.PathLike[str]


def build_time_series_features(
    df: pd.DataFrame,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
) -> np.ndarray:
    """Convert a dataframe to Graph WaveNet feature layout ``(T, N, C)``.

    Channel ``0`` is the raw traffic value. Optional channels match the
    original Graph WaveNet scripts: fractional time-of-day and integer
    day-of-week repeated for every node.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_time_series_features requires a pandas DatetimeIndex.")

    num_samples, num_nodes = df.shape
    values = df.to_numpy(dtype=np.float32)
    feature_list = [np.expand_dims(values, axis=-1)]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(
            1, "D"
        )
        time_in_day = np.tile(time_ind.astype(np.float32), [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        dow = df.index.dayofweek.to_numpy(dtype=np.float32) / 7.0
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1).astype(np.float32)
    if data.shape[0] != num_samples:
        raise RuntimeError("Generated feature data has an unexpected time dimension.")
    return data


def graph_wavenet_offsets(
    in_len: int,
    out_len: int,
    y_start: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy Graph WaveNet x/y offsets."""
    if in_len <= 0:
        raise ValueError("in_len must be positive.")
    if out_len <= 0:
        raise ValueError("out_len must be positive.")
    if y_start <= 0:
        raise ValueError("y_start must be positive.")
    if y_start > out_len:
        raise ValueError("y_start must be less than or equal to out_len.")

    x_offsets = np.sort(np.arange(-(in_len - 1), 1, 1, dtype=np.int64))
    y_offsets = np.sort(np.arange(y_start, out_len + 1, 1, dtype=np.int64))
    return x_offsets, y_offsets


def generate_graph_wavenet_windows(
    data: np.ndarray,
    in_len: int,
    out_len: int,
    y_start: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate legacy Graph WaveNet windows from ``data: (T, N, C)``.

    Returns ``x: (B, in_len, N, C)``, ``y: (B, H, N, C)``, and the two offset
    arrays. ``H`` is ``out_len - y_start + 1``.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"data must have shape (T, N, C); got {data.shape}.")

    x_offsets, y_offsets = graph_wavenet_offsets(in_len, out_len, y_start)
    num_samples = data.shape[0]
    min_t = abs(int(x_offsets.min()))
    max_t = num_samples - abs(int(y_offsets.max()))

    if max_t <= min_t:
        raise ValueError(
            f"Not enough timesteps ({num_samples}) for in_len={in_len}, "
            f"out_len={out_len}, y_start={y_start}."
        )

    x = [data[t + x_offsets, ...] for t in range(min_t, max_t)]
    y = [data[t + y_offsets, ...] for t in range(min_t, max_t)]
    return (
        np.stack(x, axis=0).astype(np.float32),
        np.stack(y, axis=0).astype(np.float32),
        x_offsets,
        y_offsets,
    )


def save_graph_wavenet_splits(
    df: pd.DataFrame,
    output_dir: PathLike,
    in_len: int = 12,
    out_len: int = 12,
    y_start: int = 1,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
) -> dict[str, tuple[int, ...]]:
    """Write legacy pre-windowed ``train/val/test.npz`` files.

    The validation fraction is the remaining samples after train and test,
    matching the original Graph WaveNet split behavior.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")
    if train_ratio + test_ratio >= 1:
        raise ValueError("train_ratio + test_ratio must be less than 1.")

    data = build_time_series_features(df, add_time_in_day, add_day_in_week)
    x, y, x_offsets, y_offsets = generate_graph_wavenet_windows(
        data,
        in_len,
        out_len,
        y_start,
    )

    num_samples = x.shape[0]
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * train_ratio)
    num_val = num_samples - num_test - num_train
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            "Split ratios produced an empty train, val, or test split. "
            f"Got train={num_train}, val={num_val}, test={num_test}."
        )

    splits = {
        "train": (x[:num_train], y[:num_train]),
        "val": (x[num_train : num_train + num_val], y[num_train : num_train + num_val]),
        "test": (x[-num_test:], y[-num_test:]),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, tuple[int, ...]] = {}
    for split_name, (split_x, split_y) in splits.items():
        np.savez_compressed(
            out / f"{split_name}.npz",
            x=split_x,
            y=split_y,
            x_offsets=x_offsets.reshape((*x_offsets.shape, 1)),
            y_offsets=y_offsets.reshape((*y_offsets.shape, 1)),
        )
        shapes[split_name] = split_x.shape
    return shapes


def save_graph_wavenet_splits_from_hdf(
    traffic_df_filename: PathLike,
    output_dir: PathLike,
    in_len: int = 12,
    out_len: int = 12,
    y_start: int = 1,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
) -> dict[str, tuple[int, ...]]:
    """Read an HDF dataframe and write pre-windowed Graph WaveNet split files."""
    df = _read_hdf_dataframe(traffic_df_filename)
    return save_graph_wavenet_splits(
        df,
        output_dir,
        in_len=in_len,
        out_len=out_len,
        y_start=y_start,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )


def save_walk_forward_features(
    df: pd.DataFrame,
    output_dir: PathLike,
    dataset_name: str,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
) -> Path:
    """Write raw walk-forward features to ``<dataset_name>.npz``."""
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty.")
    data = build_time_series_features(df, add_time_in_day, add_day_in_week)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dataset_name.lower()}.npz"
    np.savez_compressed(path, data=data)
    return path


def _read_hdf_dataframe(path: PathLike) -> pd.DataFrame:
    """Read a pandas DataFrame from an HDF5 file written by legacy pandas.

    pd.read_hdf fails on pandas 3.x when the file was written by an older
    pandas that stored the `pandas_type` attribute as bytes rather than str,
    causing a TypeError inside _create_storer. We fall back to reading the
    PyTables arrays directly and reconstructing the DataFrame.
    """
    try:
        return pd.read_hdf(path)
    except TypeError:
        import tables  # PyTables is already a pandas dependency

        with tables.open_file(str(path), "r") as f:
            root = f.root
            # Discover the first group under root that looks like a DataFrame.
            group = next(n for n in root._f_iter_nodes("Group"))
            raw_cols = group.axis0[:]
            columns = (
                raw_cols.astype(str).tolist() if raw_cols.dtype.kind == "S" else raw_cols.tolist()
            )
            # axis1 stores the DatetimeIndex as int64 nanoseconds since epoch.
            index = pd.to_datetime(group.axis1[:], unit="ns")
            # block0_values shape is (len_index, len_columns).
            values = group.block0_values[:]

        return pd.DataFrame(values, index=index, columns=columns)


def save_walk_forward_features_from_hdf(
    traffic_df_filename: PathLike,
    output_dir: PathLike,
    dataset_name: str,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
) -> Path:
    """Read an HDF dataframe and write raw walk-forward feature data."""
    df = _read_hdf_dataframe(traffic_df_filename)
    return save_walk_forward_features(
        df,
        output_dir,
        dataset_name,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
    )
