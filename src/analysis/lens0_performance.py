"""Lens 0 — Forecasting Performance Benchmark.

Evaluates each model's raw prediction accuracy before any spatial
interpretation is performed.  It computes MAE, RMSE, and MAPE at four
levels of granularity:

  1. Per-run / per-horizon (metrics_by_run)
  2. Mean ± std across runs (metrics_by_horizon)
  3. Per-node (metrics_by_node)  — consumed later by Lens 1
  4. Horizon-group summaries (short / boundary / long range)

It also computes baseline improvements over ARIMA and TCN, ranks models by
horizon and horizon group, and fits a linear degradation slope for each model.

Internal shape standard
-----------------------
Predictions are **(W, H, N, R)**:
  W — paired test windows (every model is evaluated on the same canonical
      test windows as ``ground_truth.npy``, see
      ``scripts/run_training.py``'s shared evaluation pipeline)
  H — prediction horizons (ordered, matching ``horizons`` list)
  N — graph nodes / sensors
  R — runs / seeds / folds

Ground truth is **(W, H, N)** — paired 1:1 with every model's window axis;
``ground_truth.shape[0]`` must equal every prediction array's ``shape[0]``.

Metrics are computed as ``mean(|pred - true|)`` (etc.) over the flattened
``(W, N)`` (or ``(W, N, R)``) pairs, i.e. the *average of the per-window
error*.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAPE_ZERO_THRESHOLD: float = 1e-4
"""Ground-truth magnitudes at or below this are treated as missing/zero-coded
sensor readings and masked out of MAPE — matching the null-value convention in
``src/evaluation/metrics.py`` (``|y| > 1e-4``) and tsl's ``MaskedMAPE``. Clamping
the denominator instead (the previous behaviour) turned PEMS-BAY's zero-coded
gaps into ~1/eps blow-ups (MAPE in the tens of millions of percent)."""

_DEFAULT_HORIZON_GROUPS: dict[str, list[int]] = {
    "short_range": [30],
    "boundary": [60],
    "long_range": [210],
}

_MODEL_REGISTRY_DATA: list[dict[str, str]] = [
    {
        "model": "arima",
        "display_name": "ARIMA",
        "model_group": "statistical_baseline",
        "temporal_module": "autoregressive statistical model",
        "spatial_module": "none",
        "attention_or_adaptive_component": "none",
    },
    {
        "model": "tcn",
        "display_name": "TCN-PerNode",
        "model_group": "temporal_baseline",
        "temporal_module": "independent dilated temporal convolution per sensor",
        "spatial_module": "none",
        "attention_or_adaptive_component": "none",
    },
    {
        "model": "gwn",
        "display_name": "Graph WaveNet",
        "model_group": "graph_wavenet_based",
        "temporal_module": "gated temporal convolution",
        "spatial_module": "diffusion graph convolution",
        "attention_or_adaptive_component": "adaptive adjacency",
    },
    {
        "model": "gwn_v2",
        "display_name": "Graph WaveNet V2",
        "model_group": "graph_wavenet_based",
        "temporal_module": "gated temporal convolution",
        "spatial_module": "diffusion graph convolution",
        "attention_or_adaptive_component": "improved adaptive Graph WaveNet variant",
    },
    {
        "model": "stawnet",
        "display_name": "STAWnet",
        "model_group": "attention_adaptive_stgnn",
        "temporal_module": "gated temporal convolution",
        "spatial_module": "graph or adjacency-based spatial modelling",
        "attention_or_adaptive_component": "spatial-temporal attention",
    },
    {
        "model": "dssa_tcn",
        "display_name": "DSSA-TCN",
        "model_group": "attention_adaptive_stgnn",
        "temporal_module": "causal or dilated temporal convolution",
        "spatial_module": "diffusion graph convolution",
        "attention_or_adaptive_component": "dynamic sparse spatial attention",
    },
    {
        "model": "staeformer",
        "display_name": "STAEformer",
        "model_group": "attention_adaptive_stgnn",
        "temporal_module": "transformer",
        "spatial_module": "spatio-temporal adaptive embedding",
        "attention_or_adaptive_component": "transformer attention and adaptive embedding",
    },
    {
        "model": "d2stgnn",
        "display_name": "D2STGNN",
        "model_group": "decoupled_stgnn",
        "temporal_module": "decoupled diffusion/inherent signal branches",
        "spatial_module": "dynamic and static graph convolution",
        "attention_or_adaptive_component": "estimation gate and adaptive static graph",
    },
    {
        "model": "bigst",
        "display_name": "BigST",
        "model_group": "linear_complexity_stgnn",
        "temporal_module": "input embedding over the input window",
        "spatial_module": "linearised (random-feature) spatial attention",
        "attention_or_adaptive_component": "kernelised linear attention over node embeddings",
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Lens0Result:
    """All outputs produced by Lens 0.

    DataFrames use the column schemas documented in ``run_lens0``.
    ``degradation_rates`` is a list of dicts serialisable to JSON.
    """

    metrics_by_run: pd.DataFrame
    metrics_by_horizon: pd.DataFrame
    metrics_by_node: pd.DataFrame
    horizon_group_summary: pd.DataFrame
    baseline_improvements: pd.DataFrame
    model_rankings_by_horizon: pd.DataFrame
    model_rankings_by_group: pd.DataFrame
    degradation_rates: list[dict[str, Any]]
    model_registry: pd.DataFrame
    _summary: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a compact dict of headline scalar metrics."""
        return self._summary


# ---------------------------------------------------------------------------
# Shape standardisation
# ---------------------------------------------------------------------------


def standardise_predictions(arr: np.ndarray) -> np.ndarray:
    """Validate a prediction array against the repo standard shape **(W, H, N, R)**.

    Parameters
    ----------
    arr:
        Prediction array, shape ``(W, H, N, R)``.

    Returns
    -------
    np.ndarray
        The same array, dtype float64.

    Raises
    ------
    ValueError
        When ``arr`` is not 4-D.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 4:
        raise ValueError(f"predictions must have shape (W, H, N, R); got ndim={a.ndim}.")
    return a


def standardise_ground_truth(arr: np.ndarray) -> np.ndarray:
    """Validate a ground truth array against the repo standard shape **(W, H, N)**.

    Parameters
    ----------
    arr:
        Ground truth array, shape ``(W, H, N)``.

    Returns
    -------
    np.ndarray
        The same array, dtype float64.

    Raises
    ------
    ValueError
        When ``arr`` is not 3-D.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"ground_truth must have shape (W, H, N); got ndim={a.ndim}.")
    return a


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error over all elements."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error over all elements."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, masking near-zero ground truth.

    Entries with ``|y_true| <= _MAPE_ZERO_THRESHOLD`` are excluded rather than
    divided through a clamped denominator: those are missing/zero-coded sensor
    readings (pervasive in PEMS-BAY), and clamping would inflate MAPE by ~1/eps.
    Returns ``nan`` when no valid entries remain.
    """
    valid = np.abs(y_true) > _MAPE_ZERO_THRESHOLD
    if not valid.any():
        return float("nan")
    ape = np.abs(y_true[valid] - y_pred[valid]) / np.abs(y_true[valid])
    return float(np.mean(ape) * 100.0)


def _reduce_axes_except_node(ndim: int, node_axis: int = 1) -> tuple[int, ...]:
    """Every axis except the node axis — averaged over windows and/or runs."""
    return tuple(ax for ax in range(ndim) if ax != node_axis)


def _node_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-node MAE; shape (N,). Inputs broadcast to e.g. (W, N) or (W, N, R)."""
    err = np.abs(y_true - y_pred)
    return err.mean(axis=_reduce_axes_except_node(err.ndim))


def _node_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-node RMSE; shape (N,)."""
    err2 = (y_true - y_pred) ** 2
    return np.sqrt(err2.mean(axis=_reduce_axes_except_node(err2.ndim)))


def _node_mape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-node MAPE; shape (N,). Near-zero ground truth is masked per entry.

    Nodes whose windows are all masked (no valid ground truth) yield ``nan``.
    """
    valid = np.abs(y_true) > _MAPE_ZERO_THRESHOLD
    abs_true = np.where(valid, np.abs(y_true), 1.0)
    ape = np.where(valid, np.abs(y_true - y_pred) / abs_true, 0.0)
    axes = _reduce_axes_except_node(ape.ndim)
    counts = np.broadcast_to(valid, ape.shape).sum(axis=axes)
    summed = ape.sum(axis=axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
    return out * 100.0


# ---------------------------------------------------------------------------
# Degradation fit
# ---------------------------------------------------------------------------


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = intercept + slope * x via least squares (numpy only).

    Returns
    -------
    slope, intercept, r2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _build_model_registry() -> pd.DataFrame:
    """Return the static model metadata table."""
    return pd.DataFrame(_MODEL_REGISTRY_DATA)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: Any, path: Path) -> None:
    """Save a Plotly figure to HTML and optionally to PNG.

    PNG export requires kaleido.  If kaleido is not available the PNG step
    is skipped and a warning is logged — the run is not aborted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(path), scale=2)
    except Exception as exc:
        logger.warning(
            "PNG export skipped for %s (%s). Install kaleido for static images.",
            path.name,
            exc,
        )


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def _fig_metric_by_horizon(
    df: pd.DataFrame,
    metric: str,
    dataset: str,
) -> Any:
    import plotly.express as px

    col_mean = f"{metric}_mean"
    col_std = f"{metric}_std"
    title = f"{metric.upper()} by horizon — {dataset}"
    fig = px.line(
        df,
        x="horizon_minutes",
        y=col_mean,
        color="model",
        error_y=col_std,
        markers=True,
        title=title,
        labels={"horizon_minutes": "Horizon (min)", col_mean: metric.upper()},
    )
    fig.update_layout(width=900, height=500)
    return fig


def _fig_short_boundary_long(
    df: pd.DataFrame,
    dataset: str,
) -> Any:
    import plotly.express as px

    fig = px.bar(
        df,
        x="model",
        y="mae_mean",
        color="horizon_group",
        barmode="group",
        title=f"MAE by horizon group — {dataset}",
        labels={"mae_mean": "MAE", "horizon_group": "Group"},
    )
    fig.update_layout(width=900, height=500)
    return fig


def _fig_improvement(
    df: pd.DataFrame,
    baseline_model: str,
    dataset: str,
    *,
    exclude_models: tuple[str, ...] = (),
    y_range: tuple[float, float] | None = None,
    title_suffix: str = "",
) -> Any:
    import plotly.express as px

    sub = (
        df[df["baseline_model"] == baseline_model].copy()
        if "baseline_model" in df.columns
        else pd.DataFrame()
    )
    if exclude_models and not sub.empty:
        sub = sub[~sub["model"].isin(exclude_models)].copy()
    if sub.empty:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.update_layout(title=f"No improvement data for baseline {baseline_model} — {dataset}")
        return fig
    from .model_display import display_name, family_rank

    sub = sub.copy()
    sub["Model"] = sub["model"].map(display_name)
    legend_order = [display_name(m) for m in sorted(sub["model"].unique(), key=family_rank)]
    fig = px.line(
        sub,
        x="horizon_minutes",
        y="improvement_pct",
        color="Model",
        markers=True,
        category_orders={"Model": legend_order},
        title=f"% Improvement over {baseline_model.upper()} — {dataset}{title_suffix}",
        labels={"horizon_minutes": "Horizon (min)", "improvement_pct": "% improvement"},
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.8, dash="dash"))
    # Larger fonts for print legibility; legend follows the
    # taxonomy order rather than trace-appearance order.
    fig.update_layout(
        width=900,
        height=500,
        font=dict(size=16),
        title_font_size=18,
        legend=dict(title_text="Model", font=dict(size=15)),
    )
    fig.update_xaxes(title_font_size=17, tickfont_size=15)
    fig.update_yaxes(title_font_size=17, tickfont_size=15)
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def _fig_rank_heatmap(
    df: pd.DataFrame,
    dataset: str,
) -> Any:
    import plotly.graph_objects as go

    pivot = df.pivot(index="model", columns="horizon_minutes", values="rank")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn_r",
            text=pivot.values,
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Rank"),
        )
    )
    fig.update_layout(
        title=f"Model rank by horizon (lower MAE = rank 1) — {dataset}",
        xaxis_title="Horizon (min)",
        yaxis_title="Model",
        width=900,
        height=400,
    )
    return fig


def _fig_degradation_bar(
    rates: list[dict[str, Any]],
    dataset: str,
) -> Any:
    import plotly.express as px

    sub = [r for r in rates if r["dataset"] == dataset and r["metric"] == "mae"]
    if not sub:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.update_layout(title=f"No degradation data — {dataset}")
        return fig
    df_r = pd.DataFrame(sub)
    fig = px.bar(
        df_r.sort_values("slope", ascending=False),
        x="model",
        y="slope",
        color="model_group",
        title=f"MAE degradation slope (higher = degrades faster) — {dataset}",
        labels={"slope": "MAE / min slope"},
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.8, dash="dash"))
    fig.update_layout(width=900, height=500)
    return fig


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_lens0(  # noqa: PLR0912, PLR0915
    dataset: str,
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    horizons: list[int],
    model_groups: dict[str, list[str]],
    baselines: dict[str, str],
    output_dir: str | Path,
    primary_metric: str = "mae",
    horizon_groups: dict[str, list[int]] | None = None,
    ranking_lower_is_better: bool = True,
    save_figures: bool = True,
) -> Lens0Result:
    """Run Lens 0 and write all outputs to ``output_dir``.

    Parameters
    ----------
    dataset:
        Dataset identifier string (e.g. ``"METR-LA"``).
    ground_truth:
        Ground truth array.  Shape ``(W, H, N)`` (repo default).
    predictions:
        Mapping from model name to prediction array, shape ``(W, H, N, R)``.
    horizons:
        List of horizon steps in minutes, length H.
    model_groups:
        Mapping from group name to list of model names in that group.
    baselines:
        Mapping from role (e.g. ``"statistical"``, ``"temporal"``) to model name.
    output_dir:
        Root directory under which outputs are written.  A sub-directory
        ``lens0_performance/{dataset}/`` is created automatically.
    ranking_lower_is_better:
        Whether a lower ``primary_metric`` value ranks better (``True`` for
        error metrics like MAE/RMSE/MAPE). Drives both ``model_rankings_by_horizon``
        and ``model_rankings_by_group``, and ``summary()["best_by_horizon"]``.
    primary_metric:
        Metric used for ranking (``"mae"``).
    horizon_groups:
        Override the default short / boundary / long-range groupings.
    save_figures:
        Write HTML/PNG figures in addition to the CSV/JSON outputs. PNG
        export needs a working Chrome/kaleido install and can be slow or
        hang if one isn't available — disable for fast/unit-test runs.

    Returns
    -------
    Lens0Result
    """
    out = Path(output_dir) / "lens0_performance" / dataset
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    hgroups = horizon_groups or _DEFAULT_HORIZON_GROUPS

    # Invert model_groups: model → group
    model_to_group: dict[str, str] = {}
    for group, members in model_groups.items():
        for m in members:
            model_to_group[m] = group

    # -----------------------------------------------------------------------
    # Step 1 — validate all predictions are (W, H, N, R)
    # -----------------------------------------------------------------------
    std_preds: dict[str, np.ndarray] = {
        model_name: standardise_predictions(arr) for model_name, arr in predictions.items()
    }

    if not std_preds:
        raise ValueError("predictions dict is empty.")

    # -----------------------------------------------------------------------
    # Step 2 — validate ground truth is (W, H, N)
    # -----------------------------------------------------------------------
    gt = standardise_ground_truth(ground_truth)  # (W, H, N)
    H = gt.shape[1]

    if len(horizons) != H:
        raise ValueError(f"horizons list has {len(horizons)} entries but ground truth has H={H}.")

    # -----------------------------------------------------------------------
    # Step 3 — metrics_by_run  (one row per model × run × horizon)
    #
    # Computed as mean(|pred - true|) etc. over the flattened (window, node)
    # pairs — the average of the per-window error, not the error of the
    # per-window average.
    # -----------------------------------------------------------------------
    run_rows: list[dict[str, Any]] = []
    for model_name, pred in std_preds.items():
        group = model_to_group.get(model_name, "unknown")
        if gt.shape[0] != pred.shape[0]:
            raise ValueError(
                f"{model_name} predictions have {pred.shape[0]} windows but "
                f"ground_truth has {gt.shape[0]}; they must match."
            )
        if gt.shape[1] != pred.shape[1]:
            raise ValueError(
                f"{model_name} predictions have H={pred.shape[1]} horizons but "
                f"ground_truth has H={gt.shape[1]}; they must match (a stale "
                "predictions file saved under a different cfg.horizons set would "
                "otherwise be silently truncated/misaligned instead of raising)."
            )
        for r in range(pred.shape[-1]):
            for h_idx, h_min in enumerate(horizons):
                yt = gt[:, h_idx, :]  # (W, N)
                yp = pred[:, h_idx, :, r]  # (W, N)
                run_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "model_group": group,
                        "run": r,
                        "horizon_minutes": h_min,
                        "mae": _mae(yt, yp),
                        "rmse": _rmse(yt, yp),
                        "mape": _mape(yt, yp),
                    }
                )

    metrics_by_run = pd.DataFrame(run_rows)

    # -----------------------------------------------------------------------
    # Step 4 — metrics_by_horizon  (mean ± std across runs)
    # -----------------------------------------------------------------------
    agg = (
        metrics_by_run.groupby(["dataset", "model", "model_group", "horizon_minutes"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
        )
        .reset_index()
    )
    # std is NaN when R == 1; replace with 0 to keep downstream code simple
    for col in ("mae_std", "rmse_std", "mape_std"):
        agg[col] = agg[col].fillna(0.0)
    metrics_by_horizon = agg

    # -----------------------------------------------------------------------
    # Step 5 — metrics_by_node  (per node, averaged over windows and runs)
    # -----------------------------------------------------------------------
    node_rows: list[dict[str, Any]] = []
    for model_name, pred in std_preds.items():
        group = model_to_group.get(model_name, "unknown")
        for h_idx, h_min in enumerate(horizons):
            yt = gt[:, h_idx, :][:, :, np.newaxis]  # (W, N, 1)
            yp = pred[:, h_idx, :, :]  # (W, N, R)
            n_mae = _node_mae(yt, yp)  # (N,)
            n_rmse = _node_rmse(yt, yp)
            n_mape = _node_mape(yt, yp)
            for n_idx in range(n_mae.shape[0]):
                node_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "model_group": group,
                        "node_id": n_idx,
                        "horizon_minutes": h_min,
                        "mae": float(n_mae[n_idx]),
                        "rmse": float(n_rmse[n_idx]),
                        "mape": float(n_mape[n_idx]),
                    }
                )

    metrics_by_node = pd.DataFrame(node_rows)

    # -----------------------------------------------------------------------
    # Step 6 — horizon_group_summary
    # -----------------------------------------------------------------------
    h_min_to_group: dict[int, str] = {}
    for grp_name, grp_horizons in hgroups.items():
        for hm in grp_horizons:
            h_min_to_group[hm] = grp_name

    metrics_by_horizon["horizon_group"] = metrics_by_horizon["horizon_minutes"].map(h_min_to_group)
    group_summary = (
        metrics_by_horizon.dropna(subset=["horizon_group"])
        .groupby(["dataset", "model", "model_group", "horizon_group"])
        .agg(
            mae_mean=("mae_mean", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            mape_mean=("mape_mean", "mean"),
        )
        .reset_index()
    )
    horizon_group_summary = group_summary
    # Remove helper column from metrics_by_horizon
    metrics_by_horizon = metrics_by_horizon.drop(columns=["horizon_group"])

    # -----------------------------------------------------------------------
    # Step 7 — baseline_improvements
    # -----------------------------------------------------------------------
    improvement_rows: list[dict[str, Any]] = []
    for _role, bl_name in baselines.items():
        if bl_name not in std_preds:
            logger.warning(
                "Baseline model %r not found in predictions; skipping improvement calculation.",
                bl_name,
            )
            continue
        bl_horizon = metrics_by_horizon[metrics_by_horizon["model"] == bl_name][
            ["horizon_minutes", "mae_mean"]
        ].set_index("horizon_minutes")["mae_mean"]

        for model_name in std_preds:
            if model_name == bl_name:
                continue
            group = model_to_group.get(model_name, "unknown")
            model_horizon = metrics_by_horizon[metrics_by_horizon["model"] == model_name][
                ["horizon_minutes", "mae_mean"]
            ].set_index("horizon_minutes")["mae_mean"]
            for h_min in horizons:
                if h_min not in bl_horizon.index or h_min not in model_horizon.index:
                    continue
                bl_mae = float(bl_horizon[h_min])
                m_mae = float(model_horizon[h_min])
                if bl_mae == 0.0:
                    continue
                improvement_pct = (bl_mae - m_mae) / bl_mae * 100.0
                improvement_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "model_group": group,
                        "baseline_model": bl_name,
                        "horizon_minutes": h_min,
                        "mae": m_mae,
                        "baseline_mae": bl_mae,
                        "improvement_pct": improvement_pct,
                    }
                )

    baseline_improvements = pd.DataFrame(improvement_rows)

    # -----------------------------------------------------------------------
    # Step 8 — model rankings
    # -----------------------------------------------------------------------
    rank_col = f"{primary_metric}_mean"

    rankings_by_horizon = (
        metrics_by_horizon[["dataset", "model", "model_group", "horizon_minutes", rank_col]]
        .copy()
        .rename(columns={rank_col: primary_metric})
    )
    rankings_by_horizon["rank"] = (
        rankings_by_horizon.groupby("horizon_minutes")[primary_metric]
        .rank(method="min", ascending=ranking_lower_is_better)
        .astype(int)
    )
    model_rankings_by_horizon = rankings_by_horizon.sort_values(
        ["horizon_minutes", "rank"]
    ).reset_index(drop=True)

    group_rank_col = f"{primary_metric}_mean"
    rankings_by_group = (
        horizon_group_summary[["dataset", "model", "model_group", "horizon_group", group_rank_col]]
        .copy()
        .rename(columns={group_rank_col: primary_metric})
    )
    rankings_by_group["rank"] = (
        rankings_by_group.groupby("horizon_group")[primary_metric]
        .rank(method="min", ascending=ranking_lower_is_better)
        .astype(int)
    )
    model_rankings_by_group = rankings_by_group.sort_values(["horizon_group", "rank"]).reset_index(
        drop=True
    )

    # -----------------------------------------------------------------------
    # Step 9 — degradation rates (linear regression of MAE over horizons)
    # -----------------------------------------------------------------------
    degradation_rates: list[dict[str, Any]] = []
    x = np.array(horizons, dtype=np.float64)
    for model_name in std_preds:
        group = model_to_group.get(model_name, "unknown")
        model_agg = metrics_by_horizon[metrics_by_horizon["model"] == model_name].sort_values(
            "horizon_minutes"
        )
        for metric_col, metric_label in [
            ("mae_mean", "mae"),
            ("rmse_mean", "rmse"),
            ("mape_mean", "mape"),
        ]:
            y = model_agg[metric_col].to_numpy()
            if len(y) < 2:
                continue
            slope, intercept, r2 = _linear_regression(x, y)
            degradation_rates.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "model_group": group,
                    "metric": metric_label,
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    f"first_horizon_{metric_label}": float(y[0]),
                    f"last_horizon_{metric_label}": float(y[-1]),
                    "absolute_degradation": float(y[-1] - y[0]),
                    "percentage_degradation": float((y[-1] - y[0]) / y[0] * 100.0)
                    if y[0] != 0.0
                    else None,
                }
            )

    # -----------------------------------------------------------------------
    # Step 10 — model registry
    # -----------------------------------------------------------------------
    model_registry = _build_model_registry()

    # -----------------------------------------------------------------------
    # Step 11 — summary
    # -----------------------------------------------------------------------
    best_by_horizon: dict[str, Any] = {}
    for h_min in horizons:
        h_rows = metrics_by_horizon[metrics_by_horizon["horizon_minutes"] == h_min]
        best_idx = (
            h_rows[rank_col].idxmin() if ranking_lower_is_better else h_rows[rank_col].idxmax()
        )
        best = h_rows.loc[best_idx]
        best_by_horizon[str(h_min)] = {
            "model": str(best["model"]),
            "model_group": str(best["model_group"]),
            primary_metric: float(best[rank_col]),
        }

    # Seed budget of the stochastic models. Taking the first model is wrong when
    # it is a deterministic baseline (e.g. ARIMA is a single fit, last-axis 1),
    # which understates num_runs; the max over models reflects the real budget.
    num_runs = max(pred.shape[-1] for pred in std_preds.values())
    summary_dict: dict[str, Any] = {
        "dataset": dataset,
        "primary_metric": primary_metric,
        "num_models": len(std_preds),
        "num_horizons": len(horizons),
        "num_runs": num_runs,
        "best_by_horizon": best_by_horizon,
    }

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    metrics_by_run.to_csv(out / "metrics_by_run.csv", index=False)
    metrics_by_horizon.to_csv(out / "metrics_by_horizon.csv", index=False)
    metrics_by_node.to_csv(out / "metrics_by_node.csv", index=False)
    horizon_group_summary.to_csv(out / "horizon_group_summary.csv", index=False)
    baseline_improvements.to_csv(out / "baseline_improvements.csv", index=False)
    model_rankings_by_horizon.to_csv(out / "model_rankings_by_horizon.csv", index=False)
    model_rankings_by_group.to_csv(out / "model_rankings_by_group.csv", index=False)
    model_registry.to_csv(out / "model_registry.csv", index=False)

    # Save degradation_rates.json
    with open(out / "degradation_rates.json", "w") as f:
        json.dump(_json_safe(degradation_rates), f, indent=2)

    # Save summary.json
    with open(out / "summary.json", "w") as f:
        json.dump(_json_safe(summary_dict), f, indent=2)

    # -----------------------------------------------------------------------
    # Step 12 — figures
    # -----------------------------------------------------------------------
    if not save_figures:
        return Lens0Result(
            metrics_by_run=metrics_by_run,
            metrics_by_horizon=metrics_by_horizon,
            metrics_by_node=metrics_by_node,
            horizon_group_summary=horizon_group_summary,
            baseline_improvements=baseline_improvements,
            model_rankings_by_horizon=model_rankings_by_horizon,
            model_rankings_by_group=model_rankings_by_group,
            degradation_rates=degradation_rates,
            model_registry=model_registry,
            _summary=summary_dict,
        )

    try:
        for metric in ("mae", "rmse", "mape"):
            fig = _fig_metric_by_horizon(metrics_by_horizon, metric, dataset)
            _save_figure(fig, fig_dir / f"{metric}_by_horizon.png")

        fig_sbl = _fig_short_boundary_long(horizon_group_summary, dataset)
        _save_figure(fig_sbl, fig_dir / "short_boundary_long_mae.png")

        for _role, bl_name in baselines.items():
            fig_imp = _fig_improvement(baseline_improvements, bl_name, dataset)
            _save_figure(fig_imp, fig_dir / f"improvement_over_{bl_name}.png")

        # Spatial-utility variant: drop ARIMA (its large negative improvement
        # compresses the spatial models into an indistinguishable band) and pin a
        # shared y-axis so the METR-LA vs PEMS-BAY spatial-gain gap is directly
        # comparable across datasets. Kept alongside the full figure above.
        fig_imp_spatial = _fig_improvement(
            baseline_improvements,
            "tcn",
            dataset,
            exclude_models=("arima",),
            y_range=(-33.0, 18.0),
            title_suffix=" (spatial models)",
        )
        _save_figure(fig_imp_spatial, fig_dir / "improvement_over_tcn_no_arima.png")

        fig_rank = _fig_rank_heatmap(model_rankings_by_horizon, dataset)
        _save_figure(fig_rank, fig_dir / "model_rank_heatmap.png")

        fig_deg = _fig_degradation_bar(degradation_rates, dataset)
        _save_figure(fig_deg, fig_dir / "degradation_rate_bar.png")

    except ImportError as exc:
        logger.warning("Figure generation skipped — plotly not available: %s", exc)

    return Lens0Result(
        metrics_by_run=metrics_by_run,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_node=metrics_by_node,
        horizon_group_summary=horizon_group_summary,
        baseline_improvements=baseline_improvements,
        model_rankings_by_horizon=model_rankings_by_horizon,
        model_rankings_by_group=model_rankings_by_group,
        degradation_rates=degradation_rates,
        model_registry=model_registry,
        _summary=summary_dict,
    )
