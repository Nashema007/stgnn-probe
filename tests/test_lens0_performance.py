"""Tests for Lens 0 — Forecasting Performance Benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from analysis.config import DatasetConfig, PerformanceConfig, ProbeConfig
from analysis.lens0_performance import (
    _build_model_registry,
    _linear_regression,
    _mae,
    _mape,
    _rmse,
    standardise_ground_truth,
    standardise_predictions,
)
from analysis.lens0_performance import run_lens0 as _run_lens0_with_figures
from analysis.probe import ProbeRunner


def run_lens0(*args, **kwargs):
    """Test-local wrapper: skip figure generation by default.

    PNG export needs a working Chrome/kaleido install and can be slow (or
    hang, depending on the sandbox) when one isn't available — every test
    in this file except ``test_output_figures_written`` only cares about
    the CSV/JSON outputs, so default to ``save_figures=False`` here and let
    that one test override it explicitly.
    """
    kwargs.setdefault("save_figures", False)
    return _run_lens0_with_figures(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, H, R = 5, 7, 3
HORIZONS = [30, 60, 90, 120, 150, 180, 210]
MODEL_GROUPS = {
    "statistical_baseline": ["arima"],
    "temporal_baseline": ["tcn"],
    "graph_wavenet_based": ["gwn"],
}
BASELINES = {"statistical": "arima", "temporal": "tcn"}


def test_model_registry_excludes_removed_models() -> None:
    registry = _build_model_registry().set_index("model")

    assert registry.loc["tcn", "display_name"] == "TCN-PerNode"
    assert "tcn_shared" not in registry.index
    assert "astgcn" not in registry.index


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def ground_truth(rng: np.random.Generator) -> np.ndarray:
    """Shape (W, H, N) — repo default, single test window."""
    return rng.random((1, H, N)).astype(np.float64) + 1.0


@pytest.fixture()
def predictions_whnr(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Three models in (W, H, N, R) layout, single test window."""
    return {
        "arima": rng.random((1, H, N, R)).astype(np.float64),
        "tcn": rng.random((1, H, N, R)).astype(np.float64),
        "gwn": rng.random((1, H, N, R)).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# 1 — MAE, RMSE, MAPE correctness
# ---------------------------------------------------------------------------


def test_mae_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert _mae(y, y) == pytest.approx(0.0)


def test_mae_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert _mae(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_known_value() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # errors: 3, 4  → mean sq: (9+16)/2=12.5  → sqrt=3.535...
    assert _rmse(y_true, y_pred) == pytest.approx(math.sqrt(12.5))


def test_mape_known_value() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # |errors| / |true|: 0.2, 0.1  → mean=0.15  → *100 = 15
    assert _mape(y_true, y_pred) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 2 — MAPE zero-value handling
# ---------------------------------------------------------------------------


def test_mape_zero_true_values_masked_out() -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([1.0, 2.0])
    result = _mape(y_true, y_pred)
    # The zero-coded entry is masked; only |1-2|/1 = 100% contributes.
    assert math.isfinite(result)
    assert result == pytest.approx(100.0)


def test_mape_all_zeros_masked_returns_nan() -> None:
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    result = _mape(y_true, y_pred)
    # Every entry is zero-coded (missing) and masked out; with no valid
    # ground truth MAPE is undefined (nan) — not a 1/eps blow-up.
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# 3 — Shape validation
# ---------------------------------------------------------------------------


def test_standardise_predictions_passthrough(rng: np.random.Generator) -> None:
    """(W, H, N, R) passes through unchanged."""
    W = 4
    arr = rng.random((W, H, N, R))
    out = standardise_predictions(arr)
    assert out.shape == (W, H, N, R)
    np.testing.assert_array_equal(out, arr)


@pytest.mark.parametrize("shape", [(N, H, R), (H, N), (2, 3, 4, 5, 6)])
def test_standardise_predictions_rejects_wrong_ndim(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="ndim"):
        standardise_predictions(rng.random(shape))


def test_standardise_ground_truth_passthrough(rng: np.random.Generator) -> None:
    """(W, H, N) passes through unchanged."""
    W = 4
    arr = rng.random((W, H, N))
    out = standardise_ground_truth(arr)
    assert out.shape == (W, H, N)
    np.testing.assert_array_equal(out, arr)


@pytest.mark.parametrize("shape", [(N, H), (2, 3, 4, 5)])
def test_standardise_ground_truth_rejects_wrong_ndim(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="ndim"):
        standardise_ground_truth(rng.random(shape))


# ---------------------------------------------------------------------------
# 4 — metrics_by_run column check
# ---------------------------------------------------------------------------


def test_metrics_by_run_columns(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    expected = {"dataset", "model", "model_group", "run", "horizon_minutes", "mae", "rmse", "mape"}
    assert set(result.metrics_by_run.columns) == expected


def test_metrics_by_run_row_count(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    # 3 models × 3 runs × 7 horizons = 63
    assert len(result.metrics_by_run) == 3 * R * H


# ---------------------------------------------------------------------------
# 5 — metrics_by_horizon aggregation
# ---------------------------------------------------------------------------


def test_metrics_by_horizon_aggregation_correctness(tmp_path: Path) -> None:
    """Mean MAE across runs matches manual calculation."""
    gt = np.ones((1, H, N)) * 2.0
    # pred always 3.0 → mae per run = 1.0 for all horizons
    pred = np.ones((1, H, N, R)) * 3.0
    preds = {"tcn": pred, "arima": pred.copy()}
    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions=preds,
        horizons=HORIZONS,
        model_groups={"temporal_baseline": ["tcn"], "statistical_baseline": ["arima"]},
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    tcn_rows = result.metrics_by_horizon[result.metrics_by_horizon["model"] == "tcn"]
    np.testing.assert_allclose(tcn_rows["mae_mean"].to_numpy(), 1.0)
    np.testing.assert_allclose(tcn_rows["mae_std"].to_numpy(), 0.0, atol=1e-12)


def test_metrics_average_per_window_error_not_error_of_average(tmp_path: Path) -> None:
    """MAE is the average of per-window error, not the error of the per-window average."""
    gt = np.zeros((2, H, N))
    gt[1] = 2.0  # window 0 true=0, window 1 true=2
    pred = {"gwn": np.ones((2, H, N, 1))}  # constant prediction of 1.0 in both windows
    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions=pred,
        horizons=HORIZONS,
        model_groups={"graph_wavenet_based": ["gwn"]},
        baselines={},
        output_dir=tmp_path,
    )
    # average of per-window error: mean(|1-0|, |1-2|) = 1.0
    # error of the per-window average (the old, biased way): |1 - mean(0,2)| = 0.0
    np.testing.assert_allclose(result.metrics_by_horizon["mae_mean"].to_numpy(), 1.0)


def test_metrics_by_horizon_columns(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    expected = {
        "dataset",
        "model",
        "model_group",
        "horizon_minutes",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "mape_mean",
        "mape_std",
    }
    assert set(result.metrics_by_horizon.columns) == expected


# ---------------------------------------------------------------------------
# 6 — metrics_by_node node-level values
# ---------------------------------------------------------------------------


def test_metrics_by_node_columns(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    expected = {
        "dataset",
        "model",
        "model_group",
        "node_id",
        "horizon_minutes",
        "mae",
        "rmse",
        "mape",
    }
    assert set(result.metrics_by_node.columns) == expected


def test_metrics_by_node_count(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    # 3 models × 5 nodes × 7 horizons = 105
    assert len(result.metrics_by_node) == 3 * N * H


def test_metrics_by_node_mae_is_nonnegative(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    assert (result.metrics_by_node["mae"] >= 0).all()


# ---------------------------------------------------------------------------
# 7 — baseline improvement formula
# ---------------------------------------------------------------------------


def test_baseline_improvement_formula(tmp_path: Path) -> None:
    """((baseline_mae - model_mae) / baseline_mae) * 100."""
    gt = np.ones((1, H, N)) * 2.0
    # arima: pred = 4.0 → mae = 2.0
    arima_pred = np.ones((1, H, N, 1)) * 4.0
    # gwn: pred = 2.5 → mae = 0.5
    gwn_pred = np.ones((1, H, N, 1)) * 2.5
    tcn_pred = np.ones((1, H, N, 1)) * 4.0

    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions={"arima": arima_pred, "gwn": gwn_pred, "tcn": tcn_pred},
        horizons=HORIZONS,
        model_groups={
            "statistical_baseline": ["arima"],
            "graph_wavenet_based": ["gwn"],
            "temporal_baseline": ["tcn"],
        },
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    gwn_arima = result.baseline_improvements[
        (result.baseline_improvements["model"] == "gwn")
        & (result.baseline_improvements["baseline_model"] == "arima")
    ]
    assert not gwn_arima.empty
    # baseline_mae=2.0, model_mae=0.5  → improvement=(2.0-0.5)/2.0*100=75
    np.testing.assert_allclose(gwn_arima["improvement_pct"].to_numpy(), 75.0, rtol=1e-5)


def test_missing_baseline_skipped_gracefully(tmp_path: Path) -> None:
    """Run completes even when a baseline model is absent from predictions."""
    gt = np.ones((1, H, N))
    pred = {"gwn": np.ones((1, H, N, R))}
    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions=pred,
        horizons=HORIZONS,
        model_groups={"graph_wavenet_based": ["gwn"]},
        baselines={"statistical": "arima"},  # arima not in predictions
        output_dir=tmp_path,
    )
    assert result.baseline_improvements.empty


# ---------------------------------------------------------------------------
# 8 — model ranking (lower MAE = rank 1)
# ---------------------------------------------------------------------------


def test_model_ranking_lower_mae_first(tmp_path: Path) -> None:
    gt = np.ones((1, H, N)) * 2.0
    # arima: mae ≈ 3.0, gwn: mae ≈ 0.1
    arima_pred = np.ones((1, H, N, 1)) * 5.0
    gwn_pred = np.ones((1, H, N, 1)) * 2.1
    tcn_pred = np.ones((1, H, N, 1)) * 3.5

    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions={"arima": arima_pred, "gwn": gwn_pred, "tcn": tcn_pred},
        horizons=HORIZONS,
        model_groups={
            "statistical_baseline": ["arima"],
            "graph_wavenet_based": ["gwn"],
            "temporal_baseline": ["tcn"],
        },
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    h30 = result.model_rankings_by_horizon[
        result.model_rankings_by_horizon["horizon_minutes"] == 30
    ].set_index("model")["rank"]
    assert h30["gwn"] < h30["tcn"] < h30["arima"]


# ---------------------------------------------------------------------------
# 9 — degradation slope is positive when MAE increases with horizon
# ---------------------------------------------------------------------------


def test_degradation_slope_positive_when_mae_increases(tmp_path: Path) -> None:
    gt = np.ones((1, H, N)) * 1.0
    # Build predictions where error grows with horizon index
    arr = np.zeros((1, H, N, 1))
    for h_idx in range(H):
        arr[:, h_idx, :, :] = 1.0 + (h_idx + 1) * 0.5  # growing error
    preds = {"gwn": arr, "arima": arr.copy(), "tcn": arr.copy()}
    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions=preds,
        horizons=HORIZONS,
        model_groups={
            "graph_wavenet_based": ["gwn"],
            "statistical_baseline": ["arima"],
            "temporal_baseline": ["tcn"],
        },
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    gwn_mae_rate = next(
        r for r in result.degradation_rates if r["model"] == "gwn" and r["metric"] == "mae"
    )
    assert gwn_mae_rate["slope"] > 0.0


def test_degradation_rate_keys_are_metric_specific(tmp_path: Path) -> None:
    """first_horizon_X and last_horizon_X keys match the metric field."""
    gt = np.ones((1, H, N))
    pred = {"gwn": np.ones((1, H, N, 1)) * 1.5}
    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions=pred,
        horizons=HORIZONS,
        model_groups={"graph_wavenet_based": ["gwn"]},
        baselines={},
        output_dir=tmp_path,
    )
    for entry in result.degradation_rates:
        metric = entry["metric"]
        assert f"first_horizon_{metric}" in entry, f"Missing key for metric={metric}"
        assert f"last_horizon_{metric}" in entry, f"Missing key for metric={metric}"
        assert "first_horizon_mae" not in entry or metric == "mae"


def test_linear_regression_known_values() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 1 + 2x
    slope, intercept, r2 = _linear_regression(x, y)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10 — output files written to expected folder
# ---------------------------------------------------------------------------


def test_output_files_written(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    run_lens0(
        dataset="METR-LA",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    base = tmp_path / "lens0_performance" / "METR-LA"
    expected_csvs = [
        "metrics_by_run.csv",
        "metrics_by_horizon.csv",
        "metrics_by_node.csv",
        "horizon_group_summary.csv",
        "baseline_improvements.csv",
        "model_rankings_by_horizon.csv",
        "model_rankings_by_group.csv",
        "model_registry.csv",
    ]
    for fname in expected_csvs:
        assert (base / fname).exists(), f"Missing output file: {fname}"

    assert (base / "degradation_rates.json").exists()
    assert (base / "summary.json").exists()


def test_output_figures_written(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    _run_lens0_with_figures(
        dataset="METR-LA",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
        save_figures=True,
    )
    fig_dir = tmp_path / "lens0_performance" / "METR-LA" / "figures"
    expected_html = [
        "mae_by_horizon.html",
        "rmse_by_horizon.html",
        "mape_by_horizon.html",
        "short_boundary_long_mae.html",
        "improvement_over_arima.html",
        "improvement_over_tcn.html",
        "model_rank_heatmap.html",
        "degradation_rate_bar.html",
    ]
    for fname in expected_html:
        assert (fig_dir / fname).exists(), f"Missing figure: {fname}"


def test_summary_json_is_valid(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    run_lens0(
        dataset="METR-LA",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    path = tmp_path / "lens0_performance" / "METR-LA" / "summary.json"
    with open(path) as f:
        data = json.load(f)
    assert "best_by_horizon" in data
    assert isinstance(data["best_by_horizon"], dict)
    assert len(data["best_by_horizon"]) == len(HORIZONS)


def test_result_summary_method(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    s = result.summary()
    assert isinstance(s, dict)
    assert "best_by_horizon" in s
    assert "num_models" in s


def test_best_by_horizon_has_one_entry_per_horizon(
    ground_truth: np.ndarray,
    predictions_whnr: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    result = run_lens0(
        dataset="TEST",
        ground_truth=ground_truth,
        predictions=predictions_whnr,
        horizons=HORIZONS,
        model_groups=MODEL_GROUPS,
        baselines=BASELINES,
        output_dir=tmp_path,
    )
    bbh = result.summary()["best_by_horizon"]
    assert set(bbh.keys()) == {str(h) for h in HORIZONS}
    for entry in bbh.values():
        assert "model" in entry
        assert "model_group" in entry
        assert "mae" in entry


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_run_std_is_zero(tmp_path: Path) -> None:
    gt = np.ones((1, H, N))
    pred = {"gwn": np.ones((1, H, N, 1)) * 1.5}
    result = run_lens0(
        dataset="TEST",
        ground_truth=gt,
        predictions=pred,
        horizons=HORIZONS,
        model_groups={"graph_wavenet_based": ["gwn"]},
        baselines={},
        output_dir=tmp_path,
    )
    np.testing.assert_allclose(result.metrics_by_horizon["mae_std"].to_numpy(), 0.0, atol=1e-12)


def test_empty_predictions_raises(tmp_path: Path) -> None:
    gt = np.ones((1, H, N))
    with pytest.raises(ValueError, match="empty"):
        run_lens0(
            dataset="TEST",
            ground_truth=gt,
            predictions={},
            horizons=HORIZONS,
            model_groups={},
            baselines={},
            output_dir=tmp_path,
        )


def test_window_count_mismatch_raises(tmp_path: Path) -> None:
    gt = np.ones((2, H, N))
    pred = {"gwn": np.ones((3, H, N, 1))}
    with pytest.raises(ValueError, match="windows"):
        run_lens0(
            dataset="TEST",
            ground_truth=gt,
            predictions=pred,
            horizons=HORIZONS,
            model_groups={"graph_wavenet_based": ["gwn"]},
            baselines={},
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# ProbeRunner — run_performance and run_all_performance_from_config
# ---------------------------------------------------------------------------


def _write_performance_inputs(
    base: Path,
    n: int,
    h: int,
    r: int,
    models: list[str],
) -> tuple[Path, Path]:
    """Write ground truth and prediction files; return (gt_path, predictions_dir)."""
    rng = np.random.default_rng(0)
    data_dir = base / "data"
    pred_dir = data_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    gt_path = data_dir / "ground_truth.npy"
    np.save(gt_path, rng.random((1, h, n)).astype(np.float64) + 1.0)

    for model in models:
        np.save(pred_dir / f"{model}_predictions.npy", rng.random((1, h, n, r)).astype(np.float64))

    return gt_path, pred_dir


def _make_perf_config(
    tmp_path: Path,
    dataset_name: str,
    gt_path: Path,
    pred_dir: Path,
    horizon_minutes: list[int],
    model_groups: dict[str, list[str]],
) -> ProbeConfig:
    return ProbeConfig(
        datasets=[
            DatasetConfig(
                name=dataset_name,
                num_nodes=N,
                ground_truth=str(gt_path),
                predictions_dir=str(pred_dir),
                horizon_minutes=horizon_minutes,
            )
        ],
        performance=PerformanceConfig(
            model_groups=model_groups,
            baselines={"statistical": "arima", "temporal": "tcn"},
        ),
    )


def test_run_performance_returns_lens0_result(tmp_path: Path) -> None:
    from analysis.lens0_performance import Lens0Result

    models = ["arima", "tcn", "gwn"]
    gt_path, pred_dir = _write_performance_inputs(tmp_path, N, H, R, models)
    config = _make_perf_config(tmp_path, "TEST", gt_path, pred_dir, HORIZONS, MODEL_GROUPS)
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)
    result = runner.run_performance("TEST", save_figures=False)

    assert isinstance(result, Lens0Result)
    out_dir = tmp_path / "outputs" / "lens0_performance" / "TEST"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics_by_horizon.csv").exists()


def test_run_performance_skips_missing_model_file(tmp_path: Path) -> None:
    """gwn prediction file is absent; gwn should be absent from output."""
    models = ["arima", "tcn"]  # gwn intentionally omitted
    gt_path, pred_dir = _write_performance_inputs(tmp_path, N, H, R, models)
    config = _make_perf_config(tmp_path, "TEST", gt_path, pred_dir, HORIZONS, MODEL_GROUPS)
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)
    result = runner.run_performance("TEST", save_figures=False)

    assert "gwn" not in result.metrics_by_run["model"].values
    assert set(result.metrics_by_run["model"].unique()) == {"arima", "tcn"}


def test_run_all_performance_from_config_returns_all_datasets(tmp_path: Path) -> None:
    """Both datasets in the config appear as keys in the returned dict."""
    from analysis.lens0_performance import Lens0Result

    models = ["arima", "tcn", "gwn"]

    gt_a, pred_a = _write_performance_inputs(tmp_path / "ds_a", N, H, R, models)
    gt_b, pred_b = _write_performance_inputs(tmp_path / "ds_b", N, H, R, models)

    config = ProbeConfig(
        datasets=[
            DatasetConfig(
                name="DS_A",
                num_nodes=N,
                ground_truth=str(gt_a),
                predictions_dir=str(pred_a),
                horizon_minutes=HORIZONS,
            ),
            DatasetConfig(
                name="DS_B",
                num_nodes=N,
                ground_truth=str(gt_b),
                predictions_dir=str(pred_b),
                horizon_minutes=HORIZONS,
            ),
        ],
        performance=PerformanceConfig(
            model_groups=MODEL_GROUPS,
            baselines=BASELINES,
        ),
    )
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)
    all_results = runner.run_all_performance_from_config(save_figures=False)

    assert set(all_results.keys()) == {"DS_A", "DS_B"}
    for result in all_results.values():
        assert isinstance(result, Lens0Result)
