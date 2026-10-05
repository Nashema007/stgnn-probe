from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import lens2_granger
from analysis.io import load_granger_result, save_granger_result
from analysis.lens1_spatial_utility import run_lens1
from analysis.lens2_granger import build_gcg_topk, run_lens2
from analysis.lens3_alignment import run_lens3
from analysis.lens4_community import run_lens4
from analysis.lens5_degradation import run_lens5


def test_lens1_allows_mismatched_run_counts_between_tcn_and_model() -> None:
    """R need not match between tcn/model — each is averaged over its own runs."""
    rng = np.random.default_rng(0)
    tcn_predictions = rng.random((1, 2, 3, 2))  # (W=1, H=2, N=3, R=2)
    model_predictions = rng.random((1, 2, 3, 5))  # (W=1, H=2, N=3, R=5)
    ground_truth = rng.random((1, 2, 3))

    result = run_lens1(tcn_predictions, model_predictions, ground_truth)

    assert result.sgs_matrix.shape == (3, 2)


def test_lens1_rejects_mismatched_node_or_horizon_shapes() -> None:
    tcn_predictions = np.zeros((1, 2, 3, 2))  # (W=1, H=2, N=3, R=2)
    model_predictions = np.zeros((1, 2, 4, 3))  # (W=1, H=2, N=4, R=3)
    ground_truth = np.zeros((1, 2, 3))

    with pytest.raises(ValueError, match="same shape"):
        run_lens1(tcn_predictions, model_predictions, ground_truth)


def test_lens1_rejects_window_count_mismatch() -> None:
    ground_truth = np.zeros((4, 2, 3))  # (W=4, H=2, N=3)
    tcn_predictions = np.zeros((4, 2, 3, 2))  # (W=4, H=2, N=3, R=2) — matches gt
    model_predictions = np.zeros((5, 2, 3, 2))  # (W=5, ...) — mismatched

    with pytest.raises(ValueError, match="windows"):
        run_lens1(tcn_predictions, model_predictions, ground_truth)


def test_lens1_computes_sgs_from_repo_default_window_format() -> None:
    ground_truth = np.zeros((2, 2, 2))  # (W=2, H=2, N=2)
    tcn_predictions = np.ones((2, 2, 2, 1))  # constant error of 1 everywhere
    model_predictions = np.zeros((2, 2, 2, 1))  # perfect predictions

    result = run_lens1(tcn_predictions, model_predictions, ground_truth)

    np.testing.assert_allclose(result.mae_tcn, np.ones((2, 2)))
    np.testing.assert_allclose(result.mae_model, np.zeros((2, 2)))
    np.testing.assert_allclose(result.sgs_matrix, np.ones((2, 2)))
    assert result.mean_sgs == pytest.approx(1.0)
    assert result.pct_beneficial == pytest.approx(1.0)


def test_lens1_averages_per_window_error_across_real_windows() -> None:
    """SGS is computed from genuinely paired per-window error, not error-of-means."""
    ground_truth = np.zeros((2, 1, 1))  # (W=2, H=1, N=1): window 0 true=0, window 1 true=2
    ground_truth[1] = 2.0
    tcn_predictions = np.ones((2, 1, 1, 1))  # constant prediction of 1.0 in both windows
    model_predictions = np.zeros((2, 1, 1, 1))

    result = run_lens1(tcn_predictions, model_predictions, ground_truth)

    # average of per-window error: mean(|1-0|, |1-2|) = 1.0, not |1 - mean(0,2)| = 0.0
    np.testing.assert_allclose(result.mae_tcn, np.array([[1.0]]))


def test_lens3_ignores_diagonal_self_loops_for_alignment_scores() -> None:
    adjacency = np.array(
        [
            [0.9, 0.8, 0.0],
            [0.0, 0.9, 0.7],
            [0.0, 0.0, 0.9],
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

    result = run_lens3(adjacency, gcg_matrix, threshold=0.5)

    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.weighted_precision == pytest.approx(1.0)


def test_lens5_uses_diagonal_masked_alignment_for_horizon_aas() -> None:
    sgs_matrix = np.ones((3, 2))
    adjacency = np.array(
        [
            [0.9, 0.8, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ]
    )
    gcg_matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = run_lens5(
        sgs_matrix,
        gcg_matrix,
        adjacency,
        horizon_steps=[1, 2],
        alignment_threshold=0.5,
    )

    # Off-diagonal edge (0,1) is the only one above threshold and it matches the
    # GCG, so the AAS is 1.0 at every horizon (static adjacency -> flat curve).
    np.testing.assert_allclose(result.aas_per_horizon, np.array([1.0, 1.0]))


def test_lens5_fits_degradation_slope_over_minutes_not_horizon_index() -> None:
    """SGS/AAS rates must be fit against horizon in minutes, not the index.

    Horizons [6, 12, 42] steps are unequally spaced, so a fit over the index
    [0, 1, 2] weights the 30->60 min and 60->210 min gaps equally. The rate is
    defined per minute, so the slope must equal linregress(steps * min/step, y).
    """
    from scipy.stats import linregress

    horizon_steps = [6, 12, 42]
    minutes_per_step = 5.0
    # mean SGS chosen so the index-fit and minutes-fit slopes differ clearly.
    mean_sgs = np.array([0.5, 0.4, 0.0])
    sgs_matrix = np.tile(mean_sgs, (4, 1))  # (N=4, H=3), column means == mean_sgs
    gcg = np.zeros((4, 4), dtype=np.uint8)
    adjacency = np.zeros((4, 4))

    result = run_lens5(
        sgs_matrix,
        gcg,
        adjacency,
        horizon_steps=horizon_steps,
        minutes_per_step=minutes_per_step,
    )

    x_min = np.asarray(horizon_steps, dtype=float) * minutes_per_step
    expected = linregress(x_min, mean_sgs).slope
    index_slope = linregress(np.arange(3, dtype=float), mean_sgs).slope

    np.testing.assert_allclose(result.sgs_rate, expected, rtol=1e-9)
    assert not np.isclose(result.sgs_rate, index_slope), "slope must not be the horizon-index fit"


@pytest.mark.parametrize(
    ("raw_traffic", "max_lag", "significance", "match"),
    [
        (np.arange(12, dtype=float).reshape(6, 2), 0, 0.05, "max_lag"),
        (np.arange(12, dtype=float).reshape(6, 2), 1, 0.0, "significance"),
        (np.arange(12, dtype=float).reshape(6, 2), 1, 1.0, "significance"),
        (np.arange(6, dtype=float).reshape(6, 1), 1, 0.05, "at least 2 nodes"),
        (np.arange(12, dtype=float).reshape(6, 2), 3, 0.05, "too short"),
    ],
)
def test_lens2_rejects_invalid_granger_configuration(
    raw_traffic: np.ndarray,
    max_lag: int,
    significance: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_lens2(raw_traffic, max_lag=max_lag, significance=significance, n_jobs=1)


def test_lens2_raises_when_pair_failures_exceed_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    raw_traffic = rng.normal(size=(30, 3))

    def fail_pair(args: tuple[int, int, int]) -> tuple[int, int, float, float, int, bool]:
        i, j, _ = args
        return i, j, 1.0, 0.0, 0, False

    monkeypatch.setattr(lens2_granger, "_compute_pair", fail_pair)

    with pytest.raises(RuntimeError, match="Granger pair computation failed"):
        run_lens2(raw_traffic, max_lag=1, significance=0.05, n_jobs=1)


def test_lens2_computes_lagged_pearson_correlations_and_cache_round_trip(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_traffic = np.column_stack(
        [
            np.arange(20, dtype=np.float64),
            np.roll(np.arange(20, dtype=np.float64), 1),
            np.sin(np.arange(20, dtype=np.float64)),
        ]
    )

    def successful_pair(args: tuple[int, int, int]) -> tuple[int, int, float, float, int, bool]:
        i, j, _ = args
        return i, j, 1.0, 0.0, 0, True

    monkeypatch.setattr(lens2_granger, "_compute_pair", successful_pair)

    result = run_lens2(raw_traffic, max_lag=2, significance=0.05, n_jobs=1)
    save_granger_result(result, tmp_path, "SYNTH")
    loaded = load_granger_result(tmp_path, "SYNTH")

    assert result.pearson_correlations.shape == (3, 3, 2)
    assert np.isfinite(result.pearson_correlations).all()
    np.testing.assert_allclose(np.diagonal(result.pearson_correlations[:, :, 0]), np.zeros(3))
    np.testing.assert_allclose(loaded.pearson_correlations, result.pearson_correlations)


def test_lens3_returns_edge_classification_matrix_codes() -> None:
    adjacency = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ]
    )
    gcg_matrix = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = run_lens3(adjacency, gcg_matrix, threshold=0.5)

    expected = np.array(
        [
            [0, 1, 2],
            [3, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result.tp_fp_fn_matrix, expected)


def test_lens2_granger_does_not_print_to_stdout(capsys: pytest.CaptureFixture) -> None:
    rng = np.random.default_rng(42)
    raw_traffic = rng.normal(size=(30, 3)).astype(np.float64)
    run_lens2(raw_traffic, max_lag=1, significance=0.05, n_jobs=1)
    captured = capsys.readouterr()
    assert captured.out == "", f"Expected no stdout from run_lens2, got: {captured.out!r}"


def test_build_gcg_topk_keeps_k_strongest_incoming_edges() -> None:
    # fstats[i, j] = strength of i -> j; each target column keeps its top-k rows.
    fstats = np.array(
        [
            [0.0, 5.0, 9.0],
            [3.0, 0.0, 8.0],
            [1.0, 4.0, 0.0],
        ]
    )
    gcg = build_gcg_topk(fstats, top_k=1)
    expected = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(gcg, expected)
    assert gcg.sum(axis=0).tolist() == [1, 1, 1]  # exactly k incoming per node
    assert np.trace(gcg) == 0  # no self-loops


def test_build_gcg_topk_excludes_nonpositive_fstats_and_clamps_k() -> None:
    fstats = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # k exceeds available positive-F sources; only positive-F edges survive.
    gcg = build_gcg_topk(fstats, top_k=10)
    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(gcg, expected)


def test_load_granger_result_rederives_gcg_from_fstats(tmp_path) -> None:
    from analysis.lens2_granger import GrangerResult

    n = 6
    rng = np.random.default_rng(0)
    fstats = rng.random((n, n))
    stale = GrangerResult(
        gcg_matrix=np.ones((n, n), dtype=np.uint8),  # deliberately wrong — must be ignored
        pvalues=np.zeros((n, n)),
        fstats=fstats,
        optimal_lags=np.zeros((n, n), dtype=np.int32),
        pearson_correlations=np.zeros((n, n, 1)),
        bonferroni_threshold=1e-6,
    )
    save_granger_result(stale, tmp_path, "SYN")
    loaded = load_granger_result(tmp_path, "SYN", top_k=2)

    np.testing.assert_array_equal(loaded.gcg_matrix, build_gcg_topk(fstats, 2))
    assert loaded.gcg_matrix.sum(axis=0).max() <= 2  # at most k incoming per node


def test_lens4_returns_reproducible_communities_for_same_seed() -> None:
    adjacency = np.array(
        [
            [0.0, 1.0, 0.1, 0.0],
            [1.0, 0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    coords = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "latitude": [34.0, 34.01, 35.0, 35.01],
            "longitude": [-118.0, -118.01, -119.0, -119.01],
        }
    )

    first = run_lens4(adjacency, coords, num_runs=3, random_seed=7)
    second = run_lens4(adjacency, coords, num_runs=3, random_seed=7)

    np.testing.assert_array_equal(first.community_assignments, second.community_assignments)
    assert first.modularity == pytest.approx(second.modularity)
