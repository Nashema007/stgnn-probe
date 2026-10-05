"""Lens 2 — Causal Grounding Test.

Builds the ground-truth dependency graph using pairwise Granger causality
tests on the raw traffic time series.  Runs once per dataset and is cached.

Input shape
-----------
raw_traffic : (T, N)  un-normalized sensor readings
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level state for worker initialisation (must be top-level for pickle)
# ---------------------------------------------------------------------------

_WORKER_DATA: np.ndarray | None = None
_MAX_PAIR_FAILURE_RATE = 0.05
_MIN_OBSERVATIONS_PER_LAG = 3

PairResult = tuple[int, int, float, float, int, bool]


def _init_worker(data: np.ndarray) -> None:
    global _WORKER_DATA
    # Each multiprocessing worker also has its own BLAS thread pool
    # (OpenMP/OpenBLAS/MKL), which by default tries to use every core on the
    # machine. With n_jobs worker *processes* each also spawning a full set
    # of BLAS *threads*, the process count multiplies with the per-process
    # thread count (e.g. 16 workers x ~8 BLAS threads each on a 120-core
    # box), oversubscribing CPU far past the core count — symptoms are high
    # %sys (futex/scheduler contention) and near-zero speedup vs n_jobs=1.
    # threadpool_limits reconfigures the already-loaded BLAS libraries at
    # runtime, so this works regardless of what env vars were set before
    # the parent process started.
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)
    _WORKER_DATA = data


def _compute_pair(args: tuple[int, int, int]) -> PairResult:
    """Test whether node i Granger-causes node j.

    Returns (i, j, p_value, f_stat, lag, success).

    A single fixed lag ``p = max_lag`` is used for every ordered pair (no AIC
    lag selection). Every restricted/unrestricted pair of autoregressions then
    has the same specification, so the F-statistics share numerator and (with
    equal-length series) denominator degrees of freedom and are directly
    comparable across pairs for ranking. This matches the fixed historical
    input window used in the forecasting experiments. ``lag`` is 0 when the test
    fails or converges poorly.
    """
    i, j, max_lag = args
    assert _WORKER_DATA is not None
    ts_j = _WORKER_DATA[:, j].astype(float)
    ts_i = _WORKER_DATA[:, i].astype(float)

    # Skip degenerate time series
    if ts_j.std() < 1e-10 or ts_i.std() < 1e-10:
        return i, j, 1.0, 0.0, 0, True

    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.column_stack([ts_j, ts_i])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = grangercausalitytests(data, maxlag=max_lag)
        # Fixed lag = max_lag for every pair: identical test specification, so
        # raw F-statistics are comparable across pairs (no AIC lag selection).
        lag = max_lag
        p_val = float(results[lag][0]["ssr_ftest"][1])
        f_stat = float(results[lag][0]["ssr_ftest"][0])
        return i, j, p_val, f_stat, lag, True
    except Exception:
        return i, j, 1.0, 0.0, 0, False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GrangerResult:
    """Outputs from Lens 2."""

    gcg_matrix: np.ndarray  # (N, N) binary — GCG[i,j]=1 if i Granger-causes j
    pvalues: np.ndarray  # (N, N) raw p-values
    fstats: np.ndarray  # (N, N) F-statistics
    optimal_lags: np.ndarray  # (N, N) int — lag used per pair (fixed at max_lag); 0 = failed
    pearson_correlations: np.ndarray  # (N, N, L) lagged Pearson correlations
    bonferroni_threshold: float


def compute_lagged_pearson_correlations(raw_traffic: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute lagged Pearson correlations: source i at t-lag vs target j at t."""
    raw = np.asarray(raw_traffic, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"raw_traffic must have shape (T, N); got {raw.shape}.")
    _, num_nodes = raw.shape
    correlations = np.zeros((num_nodes, num_nodes, max_lag), dtype=np.float64)

    for lag in range(1, max_lag + 1):
        source = raw[:-lag]
        target = raw[lag:]
        source_std = source.std(axis=0)
        target_std = target.std(axis=0)
        for i in range(num_nodes):
            if source_std[i] < 1e-10:
                continue
            centered_source = source[:, i] - source[:, i].mean()
            for j in range(num_nodes):
                if i == j or target_std[j] < 1e-10:
                    continue
                centered_target = target[:, j] - target[:, j].mean()
                denom = np.linalg.norm(centered_source) * np.linalg.norm(centered_target)
                correlations[i, j, lag - 1] = (
                    float(np.dot(centered_source, centered_target) / denom) if denom > 0 else 0.0
                )

    return correlations


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def build_gcg_topk(fstats: np.ndarray, top_k: int) -> np.ndarray:
    """Binary GCG keeping each target node's top-k strongest incoming edges.

    For every target node ``j``, the ``k`` source nodes ``i`` with the largest
    F-statistic ``fstats[i, j]`` are marked as Granger-causing ``j``. This
    sparsifies the graph by *effect size* rather than p-value significance —
    the latter is non-discriminative in the large-sample regime (median
    p ~ 1e-48 here), so no significance threshold can control density. Density
    is at most ``k / (N - 1)`` per node; nodes with fewer than ``k`` valid
    (positive-F) sources get fewer edges. Self-loops are excluded.

    Parameters
    ----------
    fstats:
        (N, N) F-statistics; ``fstats[i, j]`` is the strength of ``i`` causing
        ``j``. Failed/non-causal pairs are 0.
    top_k:
        Number of incoming edges to keep per node. Clamped to ``[0, N - 1]``.
    """
    f = np.array(fstats, dtype=np.float64, copy=True)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError(f"fstats must be square (N, N); got {f.shape}.")
    n = f.shape[0]
    np.fill_diagonal(f, -np.inf)  # never select a self-loop
    k = max(0, min(top_k, n - 1))
    gcg = np.zeros((n, n), dtype=np.uint8)
    if k == 0:
        return gcg
    # Per column j: indices of the k rows with the largest fstats[:, j].
    top_idx = np.argpartition(-f, kth=k - 1, axis=0)[:k, :]  # (k, N)
    cols = np.broadcast_to(np.arange(n), (k, n))
    gcg[top_idx, cols] = 1
    # Drop edges backed by a non-positive F-stat (failed pairs / padding when a
    # node has fewer than k valid sources).
    gcg[f <= 0.0] = 0
    return gcg


def run_lens2(
    raw_traffic: np.ndarray,
    max_lag: int = 42,
    significance: float = 0.05,
    n_jobs: int = -1,
    top_k: int = 10,
) -> GrangerResult:
    """Compute the Granger Causality Graph (GCG) for all node pairs.

    Parameters
    ----------
    raw_traffic:
        Raw (un-normalized) sensor readings, shape (T, N).
    max_lag:
        Maximum lag to consider.  Set equal to the longest prediction horizon.
    significance:
        Nominal significance level. Retained for the reported Bonferroni
        threshold and the p-value diagnostic histogram; the GCG itself is built
        from ``top_k`` (see ``build_gcg_topk``), not this threshold.
    n_jobs:
        Number of worker processes.  -1 uses all available CPU cores.
    top_k:
        Number of strongest incoming causal edges kept per node in the GCG.
    """
    raw = np.asarray(raw_traffic, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"raw_traffic must have shape (T, N); got {raw.shape}.")
    T, N = raw.shape
    if N < 2:
        raise ValueError("raw_traffic must contain at least 2 nodes.")
    if max_lag < 1:
        raise ValueError("max_lag must be a positive integer.")
    if not 0 < significance < 1:
        raise ValueError("significance must be between 0 and 1.")
    min_observations = _MIN_OBSERVATIONS_PER_LAG * max_lag + 1
    if min_observations >= T:
        raise ValueError(
            f"raw_traffic is too short for the requested max_lag; got T={T}, max_lag={max_lag}."
        )

    n_workers = (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)
    bonferroni_threshold = significance / max(N * (N - 1), 1)

    pairs = [(i, j, max_lag) for i in range(N) for j in range(N) if i != j]

    pvalues = np.ones((N, N), dtype=np.float64)
    fstats = np.zeros((N, N), dtype=np.float64)
    optimal_lags = np.zeros((N, N), dtype=np.int32)
    pearson_correlations = compute_lagged_pearson_correlations(raw, max_lag)
    failures = 0

    if n_workers == 1:
        _init_worker(raw)
        pair_results = map(_compute_pair, pairs)
        for i, j, p_val, f_stat, opt_lag, success in tqdm(
            pair_results,
            total=len(pairs),
            desc="Granger pairs",
        ):
            pvalues[i, j] = p_val
            fstats[i, j] = f_stat
            optimal_lags[i, j] = opt_lag
            failures += int(not success)
    else:
        with Pool(n_workers, initializer=_init_worker, initargs=(raw,)) as pool:
            for i, j, p_val, f_stat, opt_lag, success in tqdm(
                pool.imap(_compute_pair, pairs, chunksize=64),
                total=len(pairs),
                desc="Granger pairs",
            ):
                pvalues[i, j] = p_val
                fstats[i, j] = f_stat
                optimal_lags[i, j] = opt_lag
                failures += int(not success)

    failure_rate = failures / len(pairs)
    if failure_rate > _MAX_PAIR_FAILURE_RATE:
        raise RuntimeError(
            "Granger pair computation failed for "
            f"{failures}/{len(pairs)} pairs ({failure_rate:.1%})."
        )
    if failures:
        warnings.warn(
            "Granger pair computation failed for "
            f"{failures}/{len(pairs)} pairs ({failure_rate:.1%}); "
            "failed pairs were treated as non-causal.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Build the GCG by effect size (top-k F-stat per node), not p-value
    # significance — the latter is saturated at this sample size (see
    # build_gcg_topk). bonferroni_threshold is still reported for the p-value
    # histogram diagnostic.
    gcg_matrix = build_gcg_topk(fstats, top_k)
    np.fill_diagonal(pvalues, 1.0)
    np.fill_diagonal(optimal_lags, 0)

    return GrangerResult(
        gcg_matrix=gcg_matrix,
        pvalues=pvalues,
        fstats=fstats,
        optimal_lags=optimal_lags,
        pearson_correlations=pearson_correlations,
        bonferroni_threshold=bonferroni_threshold,
    )


# ---------------------------------------------------------------------------
# Horizon-restricted GCG (used by Lens 5)
# ---------------------------------------------------------------------------


def gcg_at_horizon(result: GrangerResult, max_horizon_steps: int) -> np.ndarray:
    """Return a binary GCG restricted to pairs whose optimal lag <= max_horizon_steps.

    Parameters
    ----------
    result:
        Output from run_lens2.
    max_horizon_steps:
        Only include causal pairs whose AIC-optimal lag is within this many steps.

    Returns
    -------
    np.ndarray of shape (N, N), dtype uint8.
    """
    within_horizon = (result.optimal_lags > 0) & (result.optimal_lags <= max_horizon_steps)
    return (result.gcg_matrix & within_horizon).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_gcg_heatmap(result: GrangerResult, title: str = "Granger Causality Graph"):
    """Interactive heatmap of the binary GCG matrix."""
    import plotly.graph_objects as go

    density = float(result.gcg_matrix.mean())
    fig = go.Figure(
        go.Heatmap(
            z=result.gcg_matrix.tolist(),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Granger-causes"),
            hovertemplate="Source i=%{y}<br>Target j=%{x}<br>Causes=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title}<br><sup>Edge density: {density:.3f}</sup>",
        xaxis_title="Target node j",
        yaxis_title="Source node i",
        yaxis_autorange="reversed",
        width=620,
        height=600,
    )
    return fig


def plot_pvalue_histogram(result: GrangerResult):
    """Histogram of off-diagonal raw Granger p-values."""
    import altair as alt
    import pandas as pd

    N = result.pvalues.shape[0]
    mask = ~np.eye(N, dtype=bool)
    df = pd.DataFrame({"pvalue": result.pvalues[mask].ravel().tolist()})

    hist = (
        alt.Chart(df)
        .mark_bar(color="steelblue", opacity=0.85)
        .encode(
            x=alt.X("pvalue:Q", bin=alt.Bin(maxbins=50), title="p-value"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("pvalue:Q", bin=True, title="p-value bin"), "count()"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [result.bonferroni_threshold]}))
        .mark_rule(color="red", strokeDash=[4, 2])
        .encode(x=alt.X("x:Q"))
    )
    return (hist + rule).properties(
        title=(
            "Distribution of Granger p-values "
            f"(Bonferroni threshold: {result.bonferroni_threshold:.2e})"
        ),
        width=500,
        height=300,
    )


def plot_gcg_network(result: GrangerResult):
    """Interactive network visualisation of the Granger causality graph."""
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.from_numpy_array(result.gcg_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=0)
    n_nodes = len(G.nodes())

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#aaa"),
        hoverinfo="none",
        showlegend=False,
    )
    node_trace = go.Scatter(
        x=[pos[n][0] for n in range(n_nodes)],
        y=[pos[n][1] for n in range(n_nodes)],
        mode="markers",
        marker=dict(color="steelblue", size=8, line=dict(width=0.5, color="white")),
        text=[f"Node {n}<br>Out-degree: {G.out_degree(n)}" for n in range(n_nodes)],
        hoverinfo="text",
        showlegend=False,
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Granger causality network",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=650,
        height=600,
    )
    return fig
