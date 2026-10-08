"""Lens 5 — Horizon Degradation Test.

Not used in the paper: the per-horizon results are reported directly from Lenses 1 and 3.

Synthesises Lens 1 and Lens 3 to show how spatial utility and structural
alignment change as the prediction horizon grows.

The headline finding: models whose learned graph *maintains* its alignment with
Granger structure as the horizon grows also maintain their spatial utility.

The per-horizon AAS aligns each horizon's learned graph against the (fixed) GCG.
An earlier design restricted the GCG per horizon to Granger pairs with optimal
lag <= horizon, but empirically the AIC-optimal lags cluster at long values, so
short-horizon GCGs were empty and the AAS collapsed to 0 — the horizon signal
now comes from the model's own per-horizon adjacencies instead.

Input shapes
------------
sgs_matrix   : (N, H)  from Lens 1
gcg_matrix   : (N, N)  binary, from Lens 2
adjacency    : (N, N)  from model (fallback when per-horizon adjacencies absent)
horizon_steps: list[int], e.g. [6, 12, 18, 24, 30, 36, 42]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress, spearmanr

from .lens3_alignment import _binarise_adjacency, _prf


@dataclass
class DegradationResult:
    """Results from Lens 5."""

    horizon_steps: np.ndarray  # (H,) the horizon values
    mean_sgs_per_horizon: np.ndarray  # (H,) mean SGS across nodes per horizon
    pct_beneficial_per_horizon: np.ndarray  # (H,) fraction of nodes with SGS > 0
    aas_per_horizon: np.ndarray  # (H,) F1 of the per-horizon learned graph vs the GCG
    sgs_rate: float  # slope of linear fit to mean_sgs vs horizon (per minute)
    aas_rate: float  # slope of linear fit to aas vs horizon (per minute)
    sgs_r2: float  # R² of SGS linear fit
    aas_r2: float  # R² of AAS linear fit


def run_lens5(
    sgs_matrix: np.ndarray,
    gcg_matrix: np.ndarray,
    adjacency: np.ndarray,
    horizon_steps: list[int],
    alignment_threshold: float = 0.1,
    adjacencies: list[np.ndarray] | None = None,
    top_k: int | None = None,
    minutes_per_step: float = 5.0,
) -> DegradationResult:
    """Compute horizon degradation curves for SGS and AAS.

    Parameters
    ----------
    sgs_matrix:
        Per-node SGS at each horizon from Lens 1, shape (N, H).
    gcg_matrix:
        Binary Granger causality graph from Lens 2, shape (N, N). Fixed across
        horizons; the per-horizon signal comes from the learned graph.
    adjacency:
        Global learned adjacency matrix, shape (N, N). Used as fallback when
        ``adjacencies`` is None (then the AAS is flat across horizons).
    horizon_steps:
        Horizon values in time-steps, e.g. [6, 12, 18, 24, 30, 36, 42].
        Must match the H dimension of sgs_matrix.
    alignment_threshold:
        Edge-weight cutoff for binarising adjacency when computing AAS.
    adjacencies:
        Optional per-horizon learned adjacency matrices, one per entry in
        ``horizon_steps``. When provided, ``adjacencies[k]`` is aligned against
        the GCG at horizon k; this is what makes the AAS curve vary with horizon.
    minutes_per_step:
        Wall-clock minutes per time-step (5.0 for METR-LA / PEMS-BAY). The
        degradation slopes are fit against horizon in *minutes*
        (``horizon_steps * minutes_per_step``), not against the horizon index,
        so unequally spaced horizons (e.g. 30/60/210 min) contribute in
        proportion to real elapsed time. This matches the SGS-rate definition,
        which is expressed per minute.
    """
    sgs = np.asarray(sgs_matrix, dtype=np.float64)
    gcg = np.asarray(gcg_matrix, dtype=np.uint8)
    adj = np.asarray(adjacency, dtype=np.float64)
    h_steps = np.asarray(horizon_steps, dtype=np.int32)

    H = len(h_steps)
    if sgs.shape[1] != H:
        raise ValueError(
            f"sgs_matrix has {sgs.shape[1]} horizons but horizon_steps has {H} entries."
        )
    if adj.shape != gcg.shape:
        raise ValueError("adjacency and gcg_matrix must have the same shape.")
    if adjacencies is not None:
        if len(adjacencies) != H:
            raise ValueError(
                f"adjacencies must have one entry per horizon ({H}); got {len(adjacencies)}."
            )
        for k, a in enumerate(adjacencies):
            if np.asarray(a).shape != gcg.shape:
                raise ValueError(
                    f"adjacencies[{k}] shape {np.asarray(a).shape} does not match "
                    f"gcg_matrix shape {gcg.shape}."
                )

    mean_sgs = sgs.mean(axis=0)  # (H,)
    pct_beneficial = (sgs > 0).mean(axis=0)  # (H,)

    # AAS per horizon: align each horizon's learned graph against the fixed GCG.
    # Matched-sparsity (top_k) binarisation of the adjacency, consistent with
    # Lens 3 — an absolute threshold zeroes out near-uniform adjacencies.
    off_diagonal = ~np.eye(gcg.shape[0], dtype=bool)
    gcg_bin = gcg.astype(bool) & off_diagonal
    aas = np.zeros(H)
    for k in range(H):
        adj_h = np.asarray(adjacencies[k], dtype=np.float64) if adjacencies is not None else adj
        a_bin = _binarise_adjacency(adj_h, alignment_threshold, top_k, off_diagonal)
        _, _, f1, *_ = _prf(a_bin, gcg_bin)
        aas[k] = f1

    # Linear fits over horizon in *minutes* (not index): unequally spaced
    # horizons must contribute in proportion to real elapsed time, and the
    # SGS-rate is defined per minute.
    x = h_steps.astype(np.float64) * minutes_per_step
    sgs_fit = linregress(x, mean_sgs)
    aas_fit = linregress(x, aas)

    return DegradationResult(
        horizon_steps=h_steps,
        mean_sgs_per_horizon=mean_sgs,
        pct_beneficial_per_horizon=pct_beneficial,
        aas_per_horizon=aas,
        sgs_rate=float(sgs_fit.slope),
        aas_rate=float(aas_fit.slope),
        sgs_r2=float(sgs_fit.rvalue**2),
        aas_r2=float(aas_fit.rvalue**2),
    )


def compute_cross_model_correlation(
    degradation_results: dict[str, DegradationResult],
) -> tuple[float, float]:
    """Compute Spearman correlation between SGS rates and AAS rates across models.

    Returns
    -------
    (correlation, p_value)
        The headline finding: does structural alignment quality predict
        spatial utility degradation speed?
    """
    models = list(degradation_results.keys())
    if len(models) < 3:
        raise ValueError("Need at least 3 models to compute a meaningful correlation.")

    sgs_rates = np.array([degradation_results[m].sgs_rate for m in models])
    aas_rates = np.array([degradation_results[m].aas_rate for m in models])
    result = spearmanr(aas_rates, sgs_rates)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_sgs_degradation(
    results: dict[str, DegradationResult],
    horizon_labels: list[str] | None = None,
):
    """Line chart of mean SGS per horizon, one trace per model."""
    import pandas as pd
    import plotly.express as px

    first = next(iter(results.values()))
    labels = horizon_labels or [str(h) for h in first.horizon_steps]

    rows: list[dict[str, object]] = []
    for name, res in results.items():
        rows.extend(
            {
                "model": name,
                "horizon": labels[k],
                "mean_sgs": float(value),
            }
            for k, value in enumerate(res.mean_sgs_per_horizon)
        )
    fig = px.line(
        pd.DataFrame(rows),
        x="horizon",
        y="mean_sgs",
        color="model",
        markers=True,
        title="SGS degradation across horizons",
        labels={"horizon": "Horizon", "mean_sgs": "Mean SGS", "model": "Model"},
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.update_layout(width=700, height=420)
    return fig


def plot_aas_degradation(
    results: dict[str, DegradationResult],
    horizon_labels: list[str] | None = None,
):
    """Line chart of horizon-specific AAS per model."""
    import pandas as pd
    import plotly.express as px

    first = next(iter(results.values()))
    labels = horizon_labels or [str(h) for h in first.horizon_steps]
    all_aas = [v for res in results.values() for v in res.aas_per_horizon.tolist()]
    ymax = max(0.1, max(all_aas) if all_aas else 0.1) * 1.05

    rows: list[dict[str, object]] = []
    for name, res in results.items():
        rows.extend(
            {
                "model": name,
                "horizon": labels[k],
                "aas": float(value),
            }
            for k, value in enumerate(res.aas_per_horizon)
        )
    fig = px.line(
        pd.DataFrame(rows),
        x="horizon",
        y="aas",
        color="model",
        markers=True,
        title="Structural alignment degradation across horizons",
        labels={"horizon": "Horizon", "aas": "AAS (F1)", "model": "Model"},
    )
    fig.update_layout(
        yaxis=dict(range=[0, ymax]),
        width=700,
        height=420,
    )
    return fig


def plot_degradation_rate_correlation(
    results: dict[str, DegradationResult],
):
    """Scatter of AAS rate vs SGS rate across models (headline finding)."""
    import altair as alt
    import pandas as pd

    rows = [
        {"model": m, "aas_rate": r.aas_rate, "sgs_rate": r.sgs_rate} for m, r in results.items()
    ]
    df = pd.DataFrame(rows)

    try:
        corr, pval = compute_cross_model_correlation(results)
        title = f"AAS rate vs SGS rate — Spearman ρ = {corr:.2f} (p = {pval:.3f})"
    except ValueError:
        title = "AAS rate vs SGS rate"

    points = (
        alt.Chart(df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("aas_rate:Q", title="AAS degradation rate (slope)"),
            y=alt.Y("sgs_rate:Q", title="SGS degradation rate (slope)"),
            tooltip=[
                "model:N",
                alt.Tooltip("aas_rate:Q", format=".4f"),
                alt.Tooltip("sgs_rate:Q", format=".4f"),
            ],
        )
    )
    text_layer = (
        alt.Chart(df)
        .mark_text(align="left", dx=7, dy=-4, fontSize=11)
        .encode(x="aas_rate:Q", y="sgs_rate:Q", text="model:N")
    )
    rule_h = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="lightgray").encode(y="y:Q")
    rule_v = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="lightgray").encode(x="x:Q")
    return (rule_h + rule_v + points + text_layer).properties(title=title, width=450, height=380)


def plot_combined_dashboard(
    results: dict[str, DegradationResult],
    horizon_labels: list[str] | None = None,
):
    """4-panel interactive dashboard: SGS, % beneficial, AAS, and rate scatter."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    first = next(iter(results.values()))
    labels = horizon_labels or [str(h) for h in first.horizon_steps]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "SGS per horizon",
            "Beneficial nodes per horizon (%)",
            "Structural alignment per horizon",
            "Degradation rate correlation",
        ),
    )
    seen: set[str] = set()
    for name, res in results.items():
        show = name not in seen
        seen.add(name)
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=res.mean_sgs_per_horizon.tolist(),
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=show,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=(res.pct_beneficial_per_horizon * 100).tolist(),
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=res.aas_per_horizon.tolist(),
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=1)

    models = list(results.keys())
    aas_rates = [results[m].aas_rate for m in models]
    fig.add_trace(
        go.Scatter(
            x=aas_rates,
            y=[results[m].sgs_rate for m in models],
            mode="markers+text",
            text=models,
            textposition="top right",
            textfont=dict(size=9),
            marker=dict(size=8),
            cliponaxis=False,  # let right-edge labels draw past the axis
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=0, line=dict(color="lightgray", width=0.5), row=2, col=2)
    fig.add_vline(x=0, line=dict(color="lightgray", width=0.5), row=2, col=2)
    # Pad the x-range so "top right" labels on the rightmost points aren't clipped.
    if aas_rates:
        xmin, xmax = min(aas_rates), max(aas_rates)
        pad = ((xmax - xmin) or 1.0) * 0.35
        fig.update_xaxes(range=[xmin - pad, xmax + pad], row=2, col=2)

    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_xaxes(title_text="Horizon", row=1, col=1)
    fig.update_xaxes(title_text="Horizon", row=1, col=2)
    fig.update_xaxes(title_text="Horizon", row=2, col=1)
    fig.update_xaxes(title_text="AAS rate", row=2, col=2)
    fig.update_yaxes(title_text="Mean SGS", row=1, col=1)
    fig.update_yaxes(title_text="% beneficial nodes", row=1, col=2)
    fig.update_yaxes(title_text="AAS (F1)", row=2, col=1)
    fig.update_yaxes(title_text="SGS rate", row=2, col=2)
    fig.update_layout(width=1000, height=750)
    return fig
