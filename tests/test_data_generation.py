from __future__ import annotations

import numpy as np
import pandas as pd

from data.dataset import (
    DEFAULT_WALK_FORWARD_BOUNDARIES,
    load_precomputed_graph_wavenet_dataset,
    load_walk_forward_datasets,
)
from data.generation import (
    build_time_series_features,
    generate_graph_wavenet_windows,
    save_graph_wavenet_splits,
    save_walk_forward_features,
)


def _make_frame(t: int = 36, n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=t, freq="5min")
    values = np.arange(t * n, dtype=np.float32).reshape(t, n)
    return pd.DataFrame(values, index=index, columns=[f"node_{i}" for i in range(n)])


def test_build_time_series_features_adds_legacy_time_channels() -> None:
    df = _make_frame(t=3, n=2)

    data = build_time_series_features(df, add_time_in_day=True, add_day_in_week=True)

    assert data.shape == (3, 2, 3)
    np.testing.assert_allclose(data[:, :, 0], df.to_numpy(dtype=np.float32))
    np.testing.assert_allclose(data[0, :, 1], np.array([0.0, 0.0]))
    np.testing.assert_allclose(data[1, :, 1], np.array([5 / 1440, 5 / 1440]))
    np.testing.assert_allclose(data[:, :, 2], np.zeros((3, 2)))


def test_generate_graph_wavenet_windows_uses_legacy_offsets() -> None:
    df = _make_frame(t=8, n=2)
    data = build_time_series_features(df, add_time_in_day=False)

    x, y, x_offsets, y_offsets = generate_graph_wavenet_windows(
        data,
        in_len=3,
        out_len=2,
        y_start=1,
    )

    assert x.shape == (4, 3, 2, 1)
    assert y.shape == (4, 2, 2, 1)
    np.testing.assert_array_equal(x_offsets, np.array([-2, -1, 0]))
    np.testing.assert_array_equal(y_offsets, np.array([1, 2]))
    np.testing.assert_allclose(x[0, :, :, 0], df.to_numpy(dtype=np.float32)[:3])
    np.testing.assert_allclose(y[0, :, :, 0], df.to_numpy(dtype=np.float32)[3:5])


def test_save_and_load_precomputed_graph_wavenet_splits(tmp_path) -> None:
    output_dir = tmp_path / "normal"
    save_graph_wavenet_splits(
        _make_frame(t=30, n=3),
        output_dir,
        in_len=4,
        out_len=3,
        add_time_in_day=True,
    )

    train_npz = np.load(output_dir / "train.npz")
    assert set(train_npz.files) == {"x", "y", "x_offsets", "y_offsets"}
    assert train_npz["x"].shape[1:] == (4, 3, 2)
    assert train_npz["y"].shape[1:] == (3, 3, 2)
    np.testing.assert_array_equal(train_npz["x_offsets"].reshape(-1), np.array([-3, -2, -1, 0]))
    np.testing.assert_array_equal(train_npz["y_offsets"].reshape(-1), np.array([1, 2, 3]))

    dataloaders, scaler = load_precomputed_graph_wavenet_dataset(
        output_dir,
        batch_size=2,
        num_workers=0,
    )
    x, y, y_full = next(iter(dataloaders["train"]))

    assert x.shape[1:] == (2, 3, 4)
    assert y.shape[1:] == (3, 3)
    assert y_full.shape[1:] == (2, 3, 3)
    assert scaler.mean is not None
    assert scaler.mean.shape == (3,)


def test_save_and_load_walk_forward_datasets(tmp_path) -> None:
    output_dir = tmp_path / "walk"
    saved_path = save_walk_forward_features(
        _make_frame(t=36, n=3),
        output_dir,
        dataset_name="SYNTH",
        add_time_in_day=True,
    )

    folds = load_walk_forward_datasets(
        saved_path,
        boundaries=[12, 24, 36],
        in_len=4,
        out_len=3,
        batch_size=2,
        num_workers=0,
    )

    assert len(folds) == 1
    fold = folds[0]
    x, y, y_full = next(iter(fold.dataloaders["train"]))

    assert fold.index == 0
    assert fold.boundaries == (12, 24, 36)
    assert x.shape[1:] == (2, 3, 4)
    assert y.shape[1:] == (3, 3)
    assert y_full.shape[1:] == (2, 3, 3)


def test_default_walk_forward_boundaries_include_legacy_datasets() -> None:
    assert DEFAULT_WALK_FORWARD_BOUNDARIES["METR-LA"][0] == 6624
    assert DEFAULT_WALK_FORWARD_BOUNDARIES["METR-LA"][-1] == 34249
    assert DEFAULT_WALK_FORWARD_BOUNDARIES["PEMS-BAY"][0] == 8928
    assert DEFAULT_WALK_FORWARD_BOUNDARIES["PEMS-BAY"][-1] == 52093
