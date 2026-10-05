from __future__ import annotations

import numpy as np

from scripts.run_dummy_e2e import (
    ALL_MODELS,
    HORIZONS,
    IN_LEN,
    MAX_HORIZON,
    NUM_NODES,
    SPATIAL_MODELS,
    build_parser,
    run_dummy_e2e,
)

# Test-window count shared by every model: split_len - IN_LEN - MAX_HORIZON + 1,
# the same formula used internally for both the SlidingWindowDataset test split
# and ARIMA's aligned origins (see run_dummy_e2e.py).
_SPLIT_LEN = IN_LEN + MAX_HORIZON + 6
_NUM_WINDOWS = _SPLIT_LEN - IN_LEN - MAX_HORIZON + 1


def test_dummy_e2e_runs_all_models_and_writes_probe_artifacts(tmp_path) -> None:
    result = run_dummy_e2e(output_dir=tmp_path / "dummy", epochs=1, save_figures=True)

    assert result.horizons == HORIZONS
    assert result.models == ALL_MODELS
    assert result.probe_results

    predictions_dir = result.probe_inputs_dir / "predictions"
    adjacency_dir = result.probe_inputs_dir / "adjacency"

    for model_name in ALL_MODELS:
        preds_path = predictions_dir / f"{model_name}_predictions.npy"
        assert preds_path.exists()
        preds = np.load(preds_path)
        assert preds.shape == (_NUM_WINDOWS, len(HORIZONS), NUM_NODES, 1)
        assert np.isfinite(preds).all()

    arima_preds = np.load(predictions_dir / "arima_predictions.npy")
    assert arima_preds.shape == (_NUM_WINDOWS, len(HORIZONS), NUM_NODES, 1)
    assert np.isfinite(arima_preds).all()

    for model_name in SPATIAL_MODELS:
        adj_path = adjacency_dir / f"{model_name}_adjacency.npy"
        assert adj_path.exists()
        adjacency = np.load(adj_path)
        assert adjacency.shape == (NUM_NODES, NUM_NODES)
        assert np.isfinite(adjacency).all()

        out = result.output_dir / "probe_outputs" / "per_model" / f"{model_name}_DUMMY"
        expected = {
            "sgs_matrix.npy",
            "alignment_scores.json",
            "community_assignments.csv",
            "centrality_metrics.csv",
            "degree_comparison.csv",
            "degradation_rates.json",
            "spatial_gain_map.png",
            "community_map.png",
            "degree_comparison.png",
            "combined_dashboard.png",
        }
        assert expected.issubset({path.name for path in out.iterdir()})

    comparative = result.output_dir / "probe_outputs" / "comparative" / "dummy"
    assert (comparative / "comparative_summary.json").exists()
    assert (comparative / "lens5_combined_dashboard.png").exists()


def test_dummy_e2e_is_deterministic_for_all_model_outputs(tmp_path) -> None:
    first = run_dummy_e2e(
        output_dir=tmp_path / "first",
        epochs=1,
        save_figures=False,
    )
    second = run_dummy_e2e(
        output_dir=tmp_path / "second",
        epochs=1,
        save_figures=False,
    )

    for model_name in ALL_MODELS:
        first_preds = np.load(
            first.probe_inputs_dir / "predictions" / f"{model_name}_predictions.npy"
        )
        second_preds = np.load(
            second.probe_inputs_dir / "predictions" / f"{model_name}_predictions.npy"
        )
        np.testing.assert_allclose(first_preds, second_preds, rtol=1e-5, atol=1e-5)

    for model_name in SPATIAL_MODELS:
        first_adj = np.load(first.probe_inputs_dir / "adjacency" / f"{model_name}_adjacency.npy")
        second_adj = np.load(second.probe_inputs_dir / "adjacency" / f"{model_name}_adjacency.npy")
        np.testing.assert_allclose(first_adj, second_adj, rtol=1e-5, atol=1e-5)


def test_dummy_e2e_parser_exposes_runtime_controls() -> None:
    parser = build_parser()

    args = parser.parse_args(["--output-dir", "out", "--epochs", "3", "--no-figures"])

    assert args.output_dir == "out"
    assert args.epochs == 3
    assert args.no_figures is True
