from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.config import load_config
from analysis.io import (
    adjacency_path,
    model_predictions_path,
    temporal_baseline_predictions_path,
)
from analysis.probe import ProbeRunner, compute_degree_comparison, normalize_adjacency


def _write_probe_inputs(root: Path) -> Path:
    data_dir = root / "data"
    predictions_dir = data_dir / "predictions"
    adjacency_dir = root / "adjacency"
    predictions_dir.mkdir(parents=True)
    adjacency_dir.mkdir()

    raw = np.column_stack(
        [
            np.sin(np.arange(20, dtype=np.float64)),
            np.cos(np.arange(20, dtype=np.float64)),
            np.sin(np.arange(20, dtype=np.float64) / 2),
        ]
    )
    # (H, N): H=0 row [1.0, 2.0, 3.0], H=1 row [1.2, 2.2, 3.2]
    gt_hn = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.2, 2.2, 3.2],
        ]
    )
    ground_truth = gt_hn[np.newaxis]  # (W=1, H, N)
    tcn_predictions = np.stack([gt_hn + 0.5, gt_hn + 0.4], axis=2)[np.newaxis]  # (1, H, N, R=2)
    model_predictions = np.stack([gt_hn + 0.1, gt_hn + 0.2], axis=2)[np.newaxis]  # (1, H, N, R=2)
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ]
    )
    coords = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "latitude": [34.0, 34.01, 34.02],
            "longitude": [-118.0, -118.01, -118.02],
        }
    )

    np.save(data_dir / "raw.npy", raw)
    np.save(data_dir / "ground_truth.npy", ground_truth)
    np.save(predictions_dir / "tcn_predictions.npy", tcn_predictions)
    np.save(predictions_dir / "gwn_predictions.npy", model_predictions)
    np.save(adjacency_dir / "gwn_adjacency.npy", adjacency)
    coords.to_csv(data_dir / "coords.csv", index=False)

    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
datasets:
  - name: SYNTH
    num_nodes: 3
    raw_data: {data_dir / "raw.npy"}
    coordinates: {data_dir / "coords.csv"}
    ground_truth: {data_dir / "ground_truth.npy"}
    predictions_dir: {predictions_dir}
    adjacency_dir: {adjacency_dir}
    horizons: [1, 2]
    horizon_minutes: [5, 10]
models:
  temporal_baselines: [arima, tcn, patchtst]
  spatial_models: [gwn]
granger:
  max_lag: 1
  significance: 0.05
  n_jobs: 1
community:
  num_runs: 1
  random_seed: 4
alignment:
  threshold: 0.1
  sweep_steps: 3
sgs_threshold: 0.1
""",
    )
    return config_path


def test_load_config_reads_datasets_paths_and_model_lists(tmp_path) -> None:
    config_path = _write_probe_inputs(tmp_path)

    config = load_config(config_path)

    assert config.models.temporal_baselines == ["arima", "tcn", "patchtst"]
    assert config.models.spatial_models == ["gwn"]
    assert config.datasets[0].raw_data == str(tmp_path / "data" / "raw.npy")
    assert config.datasets[0].coordinates == str(tmp_path / "data" / "coords.csv")
    assert config.datasets[0].ground_truth == str(tmp_path / "data" / "ground_truth.npy")
    assert config.datasets[0].predictions_dir == str(tmp_path / "data" / "predictions")
    assert config.datasets[0].adjacency_dir == str(tmp_path / "adjacency")


def test_standardized_probe_input_paths_resolve_from_dataset_config(tmp_path) -> None:
    config = load_config(_write_probe_inputs(tmp_path))
    dataset = config.datasets[0]

    assert (
        model_predictions_path(dataset, "gwn")
        == tmp_path / "data" / "predictions" / "gwn_predictions.npy"
    )
    assert adjacency_path(dataset, "gwn") == tmp_path / "adjacency" / "gwn_adjacency.npy"
    assert (
        temporal_baseline_predictions_path(dataset, "tcn")
        == tmp_path / "data" / "predictions" / "tcn_predictions.npy"
    )


def test_normalize_adjacency_row_normalizes_and_rejects_invalid_values() -> None:
    normalized = normalize_adjacency(np.array([[0.0, 2.0], [0.0, 0.0]]), expected_nodes=2)

    np.testing.assert_allclose(normalized, np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        normalize_adjacency(np.array([[0.0, np.nan], [0.0, 0.0]]), expected_nodes=2)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_adjacency(np.array([[0.0, -1.0], [0.0, 0.0]]), expected_nodes=2)


def test_compute_degree_comparison_ignores_diagonal_and_handles_zero_ratios() -> None:
    adjacency = np.array(
        [
            [0.9, 0.5, 0.0],
            [0.2, 0.7, 0.3],
            [0.0, 0.0, 0.8],
        ]
    )
    gcg_matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    comparison = compute_degree_comparison(adjacency, gcg_matrix)

    assert list(comparison.columns) == [
        "node_id",
        "learned_out_degree",
        "learned_in_degree",
        "gcg_out_degree",
        "gcg_in_degree",
        "out_degree_delta",
        "in_degree_delta",
        "out_degree_ratio",
        "in_degree_ratio",
    ]
    np.testing.assert_allclose(comparison["learned_out_degree"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(comparison["learned_in_degree"], [0.2, 0.5, 0.3])
    np.testing.assert_allclose(comparison["gcg_out_degree"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(comparison["gcg_in_degree"], [0.0, 1.0, 1.0])
    assert np.isfinite(comparison.filter(like="_ratio").to_numpy()).all()
    assert comparison.loc[0, "in_degree_ratio"] == pytest.approx(0.0)


def test_run_from_config_writes_standard_per_model_artifacts(tmp_path) -> None:
    config = load_config(_write_probe_inputs(tmp_path))
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)

    result = runner.run_from_config("SYNTH", "gwn", save_figures=True)

    assert result.model_name == "gwn"
    out = tmp_path / "outputs" / "per_model" / "gwn_SYNTH"
    expected = {
        "sgs_matrix.npy",
        "sgs_mean.npy",
        "node_classification.csv",
        "alignment_scores.json",
        "tp_fp_fn_matrix.npy",
        "community_assignments.csv",
        "centrality_metrics.csv",
        "modularity_score.json",
        "gcs_per_community.json",
        "gcs_overall.json",
        "degradation_rates.json",
        "degree_comparison.csv",
        "degree_comparison.json",
        "spatial_gain_map.png",
        "sgs_distribution.png",
        "threshold_sweep.png",
        "degree_comparison.png",
        "community_map.png",
        "centrality_heatmap.png",
        "centrality_by_community.png",
        "adjacency_heatmap.png",
        "sgs_degradation.png",
        "aas_degradation.png",
        "correlation_plot.png",
        "combined_dashboard.png",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    classifications = pd.read_csv(out / "node_classification.csv")
    assert list(classifications.columns) == ["node_id", "label", "sgs_mean"]
    centrality = pd.read_csv(out / "centrality_metrics.csv")
    assert list(centrality.columns) == [
        "node_id",
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
    ]
    assert (out / "lens4_betweenness_centrality.npy").exists()
    assert (out / "lens4_closeness_centrality.npy").exists()
    assert (out / "lens4_eigenvector_centrality.npy").exists()
    degree_comparison = pd.read_csv(out / "degree_comparison.csv")
    assert np.isfinite(degree_comparison.drop(columns=["node_id"]).to_numpy()).all()


def test_pooled_within_group_sd_excludes_between_group_variation() -> None:
    """The pooled within-horizon seed SD ignores horizon-to-horizon mean drift."""
    from analysis.probe import _pooled_within_group_sd

    # Two horizons, identical within-seed spread but very different means.
    groups = [[1.0, 3.0], [10.0, 12.0]]  # each var(ddof=1) = 2.0
    assert _pooled_within_group_sd(groups) == pytest.approx(np.sqrt(2.0))
    # Far below the total spread, which the between-horizon gap dominates.
    assert _pooled_within_group_sd(groups) < float(np.std([1.0, 3.0, 10.0, 12.0], ddof=1))
    # Unequal seed counts use df-weighted pooling: (1·2 + 2·1)/(1+2) = 4/3.
    assert _pooled_within_group_sd([[1.0, 3.0], [10.0, 11.0, 12.0]]) == pytest.approx(
        np.sqrt(4.0 / 3.0)
    )
    # Degenerate/non-finite groups contribute no degrees of freedom.
    assert _pooled_within_group_sd([[5.0]]) == 0.0
    assert _pooled_within_group_sd([]) == 0.0
    assert _pooled_within_group_sd([[1.0, float("nan"), 3.0]]) == pytest.approx(np.sqrt(2.0))


def test_run_from_config_separates_seed_and_horizon_structure_spread(tmp_path) -> None:
    """Structure pools real per-horizon seed runs; seed vs horizon spread stay apart."""
    config = load_config(_write_probe_inputs(tmp_path))
    adjacency_dir = tmp_path / "adjacency"

    h1 = np.stack(
        [
            np.array([[0, 2, 0], [1, 0, 1], [0, 3, 0]], dtype=np.float32),
            np.array([[0, 1, 0], [2, 0, 1], [0, 2, 0]], dtype=np.float32),
            np.array([[0, 3, 0], [1, 0, 2], [0, 1, 0]], dtype=np.float32),
        ]
    )
    h2 = h1 * 0.5 + 0.1  # a genuinely different per-horizon graph set
    # A stale cross-horizon global stack must be ignored (seeds differ per horizon).
    np.save(adjacency_dir / "gwn_adjacency_seeds.npy", h1[:1])
    np.save(adjacency_dir / "gwn_adjacency_h1_seeds.npy", h1)
    np.save(adjacency_dir / "gwn_adjacency_h2_seeds.npy", h2)

    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)
    result = runner.run_from_config("SYNTH", "gwn", save_figures=False)

    ss = result.structure_seed_summary
    assert ss is not None
    assert ss.horizons == [1, 2]
    assert ss.per_horizon_num_seeds == [3, 3]
    assert ss.num_seed_evaluations == 6
    assert len(ss.per_horizon_aas_mean) == 2 and len(ss.per_horizon_aas_seed_sd) == 2
    # Grand mean == mean of per-horizon means when seed counts are equal.
    assert ss.aas_evaluation_grand_mean == pytest.approx(float(np.mean(ss.per_horizon_aas_mean)))
    # Seed spread never exceeds the horizon+seed total spread.
    assert ss.aas_seed_sd_pooled <= ss.aas_total_sd + 1e-9
    assert ss.modularity_seed_sd_pooled <= ss.modularity_total_sd + 1e-9
    assert ss.modularity_z_seed_sd_pooled <= ss.modularity_z_total_sd + 1e-9
    assert np.isfinite(
        [
            ss.aas_seed_sd_pooled,
            ss.aas_horizon_mean_sd,
            ss.modularity_seed_sd_pooled,
            ss.modularity_z_seed_sd_pooled,
        ]
    ).all()

    # Nested JSON keeps the seed vs horizon spreads explicit and per-horizon.
    summary_json = tmp_path / "outputs" / "per_model" / "gwn_SYNTH" / "structure_seed_summary.json"
    saved = json.loads(summary_json.read_text())
    assert saved["num_seed_evaluations"] == 6
    assert set(saved["aas"]) >= {
        "evaluation_grand_mean",
        "seed_sd_pooled",
        "horizon_mean_sd",
        "per_horizon",
    }
    assert set(saved["aas"]["per_horizon"]) == {"1", "2"}
    assert set(saved["modularity_zscore"]) >= {
        "evaluation_grand_mean",
        "seed_sd_pooled",
        "horizon_mean_sd",
        "total_sd",
        "per_horizon",
    }
    assert set(saved["modularity_zscore"]["per_horizon"]) == {"1", "2"}

    # summary() exposes explicit names and drops the ambiguous bare `aas`.
    s = result.summary()
    assert "aas" not in s
    assert "aas_consensus_graph" in s
    assert "aas_evaluation_grand_mean" in s and "aas_seed_sd_pooled" in s
    assert "community_modularity_evaluation_grand_mean" in s

    full_metrics_path = tmp_path / "outputs" / "per_model" / "gwn_SYNTH" / "full_metrics.json"
    full_metrics = json.loads(full_metrics_path.read_text())
    assert "lens3" not in full_metrics and "lens4" not in full_metrics
    assert "aas_consensus_graph" in full_metrics["lens3_consensus_graph"]
    assert "structure_evaluations" in full_metrics
    assert set(full_metrics["structure_evaluations"]["modularity_zscore"]["per_horizon"]) == {
        "1",
        "2",
    }


def test_run_all_from_config_writes_comparative_and_dataset_outputs(tmp_path) -> None:
    config = load_config(_write_probe_inputs(tmp_path))
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)

    results = runner.run_all_from_config(save_figures=True)

    assert set(results) == {"SYNTH:gwn"}
    assert (tmp_path / "outputs" / "comparative" / "synth" / "comparative_summary.json").exists()
    assert (
        tmp_path / "outputs" / "dataset" / "SYNTH" / "gcg_communities" / "community_assignments.csv"
    ).exists()


def test_probe_runner_raises_on_nonfinite_adjacency(tmp_path) -> None:
    config_path = _write_probe_inputs(tmp_path)
    nan_adj = np.full((3, 3), float("nan"), dtype=np.float32)
    np.save(tmp_path / "adjacency" / "gwn_adjacency.npy", nan_adj)

    config = load_config(config_path)
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)

    with pytest.raises(ValueError, match="finite"):
        runner.run_from_config("SYNTH", "gwn", save_figures=False)


def test_probe_runner_raises_on_missing_ground_truth(tmp_path) -> None:
    config_path = _write_probe_inputs(tmp_path)
    (tmp_path / "data" / "ground_truth.npy").unlink()

    config = load_config(config_path)
    runner = ProbeRunner(output_dir=tmp_path / "outputs", config=config)

    with pytest.raises((FileNotFoundError, OSError)):
        runner.run_from_config("SYNTH", "gwn", save_figures=False)


def test_run_probe_cli_accepts_single_model_and_all_modes(tmp_path) -> None:
    config_path = _write_probe_inputs(tmp_path)

    single = subprocess.run(
        [
            sys.executable,
            "run_probe.py",
            "--config",
            str(config_path),
            "--dataset",
            "SYNTH",
            "--model",
            "gwn",
            "--output-dir",
            str(tmp_path / "single_outputs"),
            "--no-figures",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert single.returncode == 0, single.stderr
    assert "STGNN-Probe complete for gwn on SYNTH" in single.stdout

    all_models = subprocess.run(
        [
            sys.executable,
            "run_probe.py",
            "--config",
            str(config_path),
            "--all",
            "--output-dir",
            str(tmp_path / "all_outputs"),
            "--no-figures",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert all_models.returncode == 0, all_models.stderr
    assert "STGNN-Probe complete for 1 model/dataset runs" in all_models.stdout
