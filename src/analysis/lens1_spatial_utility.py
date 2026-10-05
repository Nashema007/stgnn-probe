"""Lens 1 — Spatial Utility Test.

Compares per-node MAE of a spatial model against the TCN temporal baseline
to compute the Spatial Gain Score (SGS): positive means spatial context
helped, negative means it hurt.

Input shapes
------------
Predictions are ``(W, H, N, R)`` and ground truth is ``(W, H, N)`` (see
``lens0_performance.standardise_predictions``/``standardise_ground_truth``).
Per-node MAE is averaged over the window and run axes before computing SGS,
so ``mae_tcn``/``mae_model``/``sgs_matrix`` are always ``(N, H)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lens0_performance import standardise_ground_truth, standardise_predictions


@dataclass
class SpatialUtilityResult:
    """Results from Lens 1."""

    sgs_matrix: np.ndarray  # (N, H)  absolute SGS (MAE_tcn - MAE_model), in target units
    sgs_mean: np.ndarray  # (N,)    mean absolute SGS per node across horizons
    sgs_rel_matrix: np.ndarray  # (N, H)  relative SGS = (MAE_tcn - MAE_model) / MAE_tcn
    sgs_rel_mean: np.ndarray  # (N,)    mean relative SGS per node across horizons
    mae_tcn: np.ndarray  # (N, H)
    mae_model: np.ndarray  # (N, H)
    node_labels: np.ndarray  # (N,)    dtype str: beneficial/harmful/neutral (from relative SGS)
    mean_sgs: float  # scalar  overall mean absolute SGS
    mean_sgs_rel: float  # scalar  overall mean relative SGS (scale-free; use for cross-dataset)
    pct_beneficial: float
    pct_harmful: float
    pct_neutral: float
    threshold: float  # absolute-SGS reference threshold (units of the target)
    rel_threshold: float  # relative-SGS threshold used for node labels (fraction)


def _node_horizon_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-node, per-horizon MAE averaged over windows and runs.

    Parameters
    ----------
    pred:
        Standardised predictions, shape (W, H, N, R).
    truth:
        Standardised ground truth, shape (W, H, N).

    Returns
    -------
    np.ndarray
        Shape (N, H) — the average of the per-window error.
    """
    H, N = pred.shape[1], pred.shape[2]
    out = np.zeros((N, H), dtype=np.float64)
    for h_idx in range(H):
        yt = truth[:, h_idx, :][:, :, np.newaxis]  # (W, N, 1)
        yp = pred[:, h_idx, :, :]  # (W, N, R)
        out[:, h_idx] = np.abs(yt - yp).mean(axis=(0, 2))
    return out


def run_lens1(
    tcn_predictions: np.ndarray,
    model_predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.1,
    rel_threshold: float = 0.01,
) -> SpatialUtilityResult:
    """Compute Spatial Gain Scores.

    SGS is reported both absolutely (``MAE_tcn - MAE_model``, in target units)
    and relatively (``(MAE_tcn - MAE_model) / MAE_tcn``, scale-free). Cross-
    dataset comparison should use the relative form: the absolute SGS scales
    with each dataset's error magnitude, so a smaller absolute SGS on a
    lower-error dataset (e.g. PEMS-BAY) is partly a units effect, not weaker
    spatial utility. Node labels use the relative SGS for the same reason.

    Parameters
    ----------
    tcn_predictions:
        Temporal-only baseline predictions, shape (W, H, N, R).
    model_predictions:
        STGNN model predictions, shape (W, H, N, R).
    ground_truth:
        Ground truth values, shape (W, H, N).
    threshold:
        Absolute-SGS reference threshold (kept for reporting; not used for
        labels).
    rel_threshold:
        Relative SGS magnitude below which a node is labelled neutral
        (default 0.01 = 1% MAE improvement over TCN).
    """
    tcn = standardise_predictions(np.asarray(tcn_predictions, dtype=np.float64))
    model = standardise_predictions(np.asarray(model_predictions, dtype=np.float64))
    truth = standardise_ground_truth(np.asarray(ground_truth, dtype=np.float64))

    if tcn.shape[1:3] != model.shape[1:3]:
        raise ValueError(
            "tcn and model predictions must have the same shape (H, N); "
            f"got {tcn.shape[1:3]} and {model.shape[1:3]}."
        )
    H, N = tcn.shape[1], tcn.shape[2]
    if truth.shape[1:] != (H, N):
        raise ValueError(
            f"ground_truth shape must match (H, N) = {(H, N)} of predictions; "
            f"got {truth.shape[1:]}."
        )
    for name, arr in (("tcn", tcn), ("model", model)):
        if truth.shape[0] != arr.shape[0]:
            raise ValueError(
                f"{name}_predictions have {arr.shape[0]} windows but "
                f"ground_truth has {truth.shape[0]}; they must match."
            )

    # Average absolute error over windows and runs: (N, H)
    mae_tcn = _node_horizon_mae(tcn, truth)
    mae_model = _node_horizon_mae(model, truth)

    # Positive = spatial helped, negative = spatial hurt
    sgs_matrix = mae_tcn - mae_model  # (N, H)  absolute (target units)
    sgs_mean = sgs_matrix.mean(axis=1)  # (N,)

    # Relative SGS: fraction of TCN error removed. Scale-free, so comparable
    # across datasets with different error magnitudes. The per-node/horizon
    # ratio is kept for node labels; the headline scalar uses a *ratio of means*
    # (total error removed / total TCN error), which — unlike a mean of per-node
    # ratios — is not distorted by low-MAE nodes with large ratios.
    sgs_rel_matrix = sgs_matrix / np.maximum(mae_tcn, 1e-8)  # (N, H)
    sgs_rel_mean = sgs_rel_matrix.mean(axis=1)  # (N,)

    # Node labels use the relative SGS with a relative threshold — an absolute
    # threshold would flag fewer nodes on lower-error datasets purely by scale.
    labels = np.where(
        sgs_rel_mean > rel_threshold,
        "beneficial",
        np.where(sgs_rel_mean < -rel_threshold, "harmful", "neutral"),
    )

    n = len(labels)
    return SpatialUtilityResult(
        sgs_matrix=sgs_matrix,
        sgs_mean=sgs_mean,
        sgs_rel_matrix=sgs_rel_matrix,
        sgs_rel_mean=sgs_rel_mean,
        mae_tcn=mae_tcn,
        mae_model=mae_model,
        node_labels=labels,
        mean_sgs=float(sgs_mean.mean()),
        mean_sgs_rel=float(sgs_matrix.sum() / max(mae_tcn.sum(), 1e-8)),
        pct_beneficial=float((labels == "beneficial").sum() / n),
        pct_harmful=float((labels == "harmful").sum() / n),
        pct_neutral=float((labels == "neutral").sum() / n),
        threshold=threshold,
        rel_threshold=rel_threshold,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_sgs_distribution(result: SpatialUtilityResult):
    """Horizontal bar chart of mean SGS values sorted across nodes."""
    import altair as alt
    import pandas as pd

    sorted_sgs = np.sort(result.sgs_mean)[::-1]
    t = result.threshold
    categories = [
        "beneficial" if v > t else ("harmful" if v < -t else "neutral") for v in sorted_sgs
    ]
    df = pd.DataFrame(
        {
            "node": [str(i) for i in range(len(sorted_sgs))],
            "sgs": sorted_sgs.tolist(),
            "category": categories,
        }
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sgs:Q", title="Mean SGS"),
            y=alt.Y("node:N", title="Node (sorted by SGS)", sort=None),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=["beneficial", "harmful", "neutral"],
                    range=["#2e9e44", "#d62728", "steelblue"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("node:Q", title="Node"),
                alt.Tooltip("sgs:Q", title="Mean SGS", format=".4f"),
            ],
        )
    )
    rule_pos = (
        alt.Chart(pd.DataFrame({"x": [result.threshold]}))
        .mark_rule(color="#2e9e44", strokeDash=[4, 2])
        .encode(x="x:Q")
    )
    rule_neg = (
        alt.Chart(pd.DataFrame({"x": [-result.threshold]}))
        .mark_rule(color="#d62728", strokeDash=[4, 2])
        .encode(x="x:Q")
    )
    rule_zero = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeWidth=0.5)
        .encode(x="x:Q")
    )
    return (bars + rule_pos + rule_neg + rule_zero).properties(
        title="Spatial Gain Score distribution across nodes",
        width=600,
        height=max(240, len(sorted_sgs) * 12),
    )


def plot_spatial_gain_map(result: SpatialUtilityResult, coords: pd.DataFrame):
    """Scatter plot of nodes coloured by mean SGS (green = beneficial, red = harmful).

    Parameters
    ----------
    coords:
        DataFrame with columns [node_id, latitude, longitude].
    """
    import plotly.express as px

    df = coords.copy()
    df = df.assign(sgs=result.sgs_mean)
    abs_max = max(abs(float(result.sgs_mean.min())), abs(float(result.sgs_mean.max())), 1e-6)

    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="sgs",
        color_continuous_scale="RdYlGn",
        range_color=[-abs_max, abs_max],
        hover_data={"node_id": True, "sgs": ":.4f", "latitude": ":.5f", "longitude": ":.5f"},
        labels={"sgs": "Mean SGS"},
        title="Spatial Gain Score map (green = beneficial, red = harmful)",
    )
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.9,
            line=dict(width=0.8, color="#263238"),
        )
    )
    fig.update_geos(
        fitbounds="locations",
        visible=True,
        showland=True,
        landcolor="#e8edf2",
        showocean=True,
        oceancolor="#dce7ef",
        showlakes=True,
        lakecolor="#dce7ef",
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="#aeb8c2",
        bgcolor="#f4f6f8",
    )
    fig.update_layout(
        width=800,
        height=600,
        paper_bgcolor="#f4f6f8",
        plot_bgcolor="#f4f6f8",
    )
    return fig


def plot_sgs_horizon(result: SpatialUtilityResult, horizon_labels: list[str] | None = None):
    """Two-panel line chart of mean SGS and % beneficial nodes across horizons."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mean_per_h = result.sgs_matrix.mean(axis=0)
    pct_ben = (result.sgs_matrix > 0).mean(axis=0) * 100
    labels = horizon_labels or [str(h) for h in range(len(mean_per_h))]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mean SGS per horizon", "% beneficial nodes per horizon"),
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=mean_per_h.tolist(),
            mode="lines+markers",
            name="Mean SGS",
            line=dict(color="steelblue"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=pct_ben.tolist(),
            mode="lines+markers",
            name="% beneficial",
            line=dict(color="#2e9e44"),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Horizon", row=1, col=1)
    fig.update_xaxes(title_text="Horizon", row=1, col=2)
    fig.update_yaxes(title_text="Mean SGS", row=1, col=1)
    fig.update_yaxes(title_text="% nodes beneficial", range=[0, 100], row=1, col=2)
    fig.update_layout(width=900, height=400, showlegend=False)
    return fig
