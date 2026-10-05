"""Sliding-window datasets and Graph WaveNet-compatible loader factories."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .generation import generate_graph_wavenet_windows
from .scaler import StandardScaler

DEFAULT_WALK_FORWARD_BOUNDARIES: dict[str, list[int]] = {
    "METR-LA": [
        6624,
        8640,
        10656,
        12672,
        14688,
        16704,
        18720,
        20736,
        22752,
        24768,
        26784,
        28800,
        30816,
        32832,
        34249,
    ],
    "PEMS-BAY": [
        8928,
        10944,
        12960,
        14976,
        16992,
        19008,
        21024,
        23040,
        25056,
        27072,
        29088,
        31104,
        33120,
        35136,
        37152,
        39168,
        41184,
        43200,
        45216,
        47232,
        49248,
        51264,
        52093,
    ],
}


@dataclass(frozen=True)
class WalkForwardFold:
    """Dataloaders and scaler for one walk-forward validation fold."""

    index: int
    boundaries: tuple[int, int, int]
    dataloaders: dict[str, DataLoader]
    scaler: StandardScaler


class SlidingWindowDataset(Dataset):
    """Sliding-window pairs from a normalised time-series array.

    Parameters
    ----------
    data:
        Array of shape *(T, N, C)* — already normalised.
    in_len:
        Number of input timesteps.
    out_len:
        Forecast horizon.

    Each sample returns:
        x      : Tensor *(C, N, in_len)*  — canonical input layout
        y      : Tensor *(N, out_len)*    — speed/flow channel only (channel 0)
        y_full : Tensor *(C, N, out_len)* — all channels of the output window
    """

    def __init__(self, data: np.ndarray, in_len: int, out_len: int) -> None:
        self.data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        self.in_len = in_len
        self.out_len = out_len
        self.n_samples = len(data) - in_len - out_len + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough timesteps ({len(data)}) for in_len={in_len} + out_len={out_len}."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        start = idx + self.in_len
        end = start + self.out_len
        x = self.data[idx : idx + self.in_len]  # (T_in, N, C)
        y_win = self.data[start:end]  # (H, N, C)
        y = y_win[:, :, 0]  # (H, N)
        # (C, N, T_in), (N, H), (C, N, H)
        return x.permute(2, 1, 0), y.permute(1, 0), y_win.permute(2, 1, 0)


class PrecomputedWindowDataset(Dataset):
    """Dataset for legacy Graph WaveNet windows.

    Parameters
    ----------
    x:
        Windowed inputs shaped ``(B, T_in, N, C)``.
    y:
        Windowed targets shaped ``(B, H, N, C)``.

    Each item is converted to the framework's canonical trainer contract:
    ``x: (C, N, T_in)``, ``y: (N, H)``, ``y_full: (C, N, H)``.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B, T, N, C); got {x.shape}.")
        if y.ndim != 4:
            raise ValueError(f"y must have shape (B, H, N, C); got {y.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if x.shape[2] != y.shape[2]:
            raise ValueError("x and y must contain the same number of nodes.")
        if x.shape[3] != y.shape[3]:
            raise ValueError("x and y must contain the same number of channels.")

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        x = self.x[idx]  # (T_in, N, C)
        y_win = self.y[idx]  # (H, N, C)
        y = y_win[:, :, 0]  # (H, N)
        return x.permute(2, 1, 0), y.permute(1, 0), y_win.permute(2, 1, 0)


def load_dataset(
    data_path: str,
    in_len: int = 12,
    out_len: int = 12,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], StandardScaler]:
    """Load a .npz dataset and build train/val/test DataLoaders.

    The .npz file must contain a key ``"data"`` with shape *(T, N, C)*.
    Walk-forward (time-contiguous) splits are used; the scaler is fitted on
    the training split only.

    Parameters
    ----------
    data_path:
        Path to a ``.npz`` file with key ``"data"``.
    in_len:
        Input sequence length.
    out_len:
        Forecast horizon.
    val_ratio:
        Fraction of total timesteps for validation.
    test_ratio:
        Fraction of total timesteps for test.
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader worker processes.

    Returns
    -------
    dataloaders:
        Dict with keys ``"train"``, ``"val"``, ``"test"``.
    scaler:
        Fitted StandardScaler (use for inverse-transforming predictions).
    """
    raw = np.load(data_path)["data"].astype(np.float32)  # (T, N, C)
    T = len(raw)

    n_test = int(T * test_ratio)
    n_val = int(T * val_ratio)
    n_train = T - n_val - n_test

    train_data = raw[:n_train]
    val_data = raw[n_train : n_train + n_val]
    test_data = raw[n_train + n_val :]

    scaler = StandardScaler().fit(train_data)

    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)

    train_ds = SlidingWindowDataset(train_norm, in_len, out_len)
    val_ds = SlidingWindowDataset(val_norm, in_len, out_len)
    test_ds = SlidingWindowDataset(test_norm, in_len, out_len)

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return dataloaders, scaler


def load_precomputed_graph_wavenet_dataset(
    dataset_dir: str | Path,
    batch_size: int = 64,
    valid_batch_size: int | None = None,
    test_batch_size: int | None = None,
    num_workers: int = 0,
    drop_last_train: bool = True,
) -> tuple[dict[str, DataLoader], StandardScaler]:
    """Load legacy Graph WaveNet ``train/val/test.npz`` window files.

    Files must contain ``x: (B, T, N, C)`` and ``y: (B, H, N, C)``. Channel
    ``0`` is normalised with per-node statistics fitted from training inputs.
    """
    dataset_path = Path(dataset_dir)
    arrays = _load_precomputed_split_arrays(dataset_path)
    scaler = _fit_scaler_from_windows(arrays["train"][0])

    normalized = {
        split: (
            _transform_window_channel0(x, scaler),
            _transform_window_channel0(y, scaler),
        )
        for split, (x, y) in arrays.items()
    }
    return _build_window_dataloaders(
        normalized,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        drop_last_train=drop_last_train,
    ), scaler


def load_walk_forward_datasets(
    data_path: str | Path,
    boundaries: list[int] | tuple[int, ...] | None = None,
    dataset_name: str | None = None,
    in_len: int = 12,
    out_len: int = 12,
    y_start: int = 1,
    batch_size: int = 64,
    valid_batch_size: int | None = None,
    test_batch_size: int | None = None,
    num_workers: int = 0,
    drop_last_train: bool = True,
) -> list[WalkForwardFold]:
    """Build walk-forward dataloaders from a raw ``data: (T, N, C)`` npz file.

    Each fold uses three consecutive boundaries ``(a, b, c)``:
    train ``data[:a]``, validation ``data[a:b]``, and test ``data[b:c]``.
    Pass explicit ``boundaries`` for custom datasets, or ``dataset_name`` for
    the built-in METR-LA / PEMS-BAY legacy boundaries.
    """
    raw = np.load(data_path)["data"].astype(np.float32)
    if raw.ndim != 3:
        raise ValueError(f"data must have shape (T, N, C); got {raw.shape}.")

    split_boundaries = _resolve_walk_forward_boundaries(boundaries, dataset_name)
    if len(split_boundaries) < 3:
        raise ValueError("At least three walk-forward boundaries are required.")
    if split_boundaries[-1] > raw.shape[0]:
        raise ValueError(
            f"Last boundary ({split_boundaries[-1]}) exceeds data length ({raw.shape[0]})."
        )

    folds: list[WalkForwardFold] = []
    for fold_idx in range(len(split_boundaries) - 2):
        a, b, c = split_boundaries[fold_idx : fold_idx + 3]
        split_data = {
            "train": raw[:a],
            "val": raw[a:b],
            "test": raw[b:c],
        }
        scaler = StandardScaler().fit(split_data["train"])
        windows = {
            split: _make_normalized_windows(
                data,
                scaler=scaler,
                in_len=in_len,
                out_len=out_len,
                y_start=y_start,
            )
            for split, data in split_data.items()
        }
        dataloaders = _build_window_dataloaders(
            windows,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            drop_last_train=drop_last_train,
        )
        folds.append(
            WalkForwardFold(
                index=fold_idx,
                boundaries=(int(a), int(b), int(c)),
                dataloaders=dataloaders,
                scaler=scaler,
            )
        )
    return folds


def _load_precomputed_split_arrays(dataset_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        split_path = dataset_dir / f"{split}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing precomputed Graph WaveNet split: {split_path}")
        with np.load(split_path) as data:
            arrays[split] = (
                data["x"].astype(np.float32),
                data["y"].astype(np.float32),
            )
    return arrays


def _fit_scaler_from_windows(x_train: np.ndarray) -> StandardScaler:
    if x_train.ndim != 4:
        raise ValueError(f"x_train must have shape (B, T, N, C); got {x_train.shape}.")
    _, _, num_nodes, channels = x_train.shape
    flattened = x_train.reshape(-1, num_nodes, channels)
    return StandardScaler().fit(flattened)


def _transform_window_channel0(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scaler._check_fitted()
    assert scaler.mean is not None
    assert scaler.std is not None
    arr = np.asarray(data, dtype=np.float32).copy()
    mean = scaler.mean[np.newaxis, np.newaxis, :, np.newaxis]
    std = scaler.std[np.newaxis, np.newaxis, :, np.newaxis]
    arr[..., 0:1] = (arr[..., 0:1] - mean) / std
    return arr


def _make_normalized_windows(
    data: np.ndarray,
    scaler: StandardScaler,
    in_len: int,
    out_len: int,
    y_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, _, _ = generate_graph_wavenet_windows(
        data,
        in_len=in_len,
        out_len=out_len,
        y_start=y_start,
    )
    return _transform_window_channel0(x, scaler), _transform_window_channel0(y, scaler)


def _build_window_dataloaders(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    valid_batch_size: int | None,
    test_batch_size: int | None,
    num_workers: int,
    drop_last_train: bool,
) -> dict[str, DataLoader]:
    valid_bs = valid_batch_size or batch_size
    test_bs = test_batch_size or batch_size
    datasets = {split: PrecomputedWindowDataset(x, y) for split, (x, y) in arrays.items()}
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last_train,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=valid_bs,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=test_bs,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def _resolve_walk_forward_boundaries(
    boundaries: list[int] | tuple[int, ...] | None,
    dataset_name: str | None,
) -> list[int]:
    if boundaries is not None:
        resolved = [int(boundary) for boundary in boundaries]
    elif dataset_name is not None:
        key = dataset_name.upper()
        if key not in DEFAULT_WALK_FORWARD_BOUNDARIES:
            raise ValueError(
                f"No default walk-forward boundaries for dataset '{dataset_name}'. "
                "Pass explicit boundaries instead."
            )
        resolved = DEFAULT_WALK_FORWARD_BOUNDARIES[key]
    else:
        raise ValueError("Pass either explicit boundaries or a dataset_name with defaults.")

    if any(boundary <= 0 for boundary in resolved):
        raise ValueError("Walk-forward boundaries must be positive.")
    if any(a >= b for a, b in pairwise(resolved)):
        raise ValueError("Walk-forward boundaries must be strictly increasing.")
    return resolved
