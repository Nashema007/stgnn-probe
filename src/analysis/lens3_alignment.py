"""Lens 3 — Structural Alignment Test.

Quantifies how well a model's learned adjacency matrix matches the Granger
ground-truth graph from Lens 2.  The headline metric is the Adjacency
Alignment Score (AAS = F1 of precision/recall against the GCG).

Input shapes
------------
adjacency  : (N, N)  row-normalised learned weights in [0, 1]
gcg_matrix : (N, N)  binary ground-truth from Lens 2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    """Results from Lens 3."""

    precision: float
    recall: float
    f1: float  # Adjacency Alignment Score (AAS)
    tp: int
    fp: int
    fn: int
    threshold: float
    weighted_precision: float
    tp_fp_fn_matrix: np.ndarray  # (N, N), codes: 0=TN/diagonal, 1=TP, 2=FP, 3=FN
    # Threshold sweep: sorted arrays of (threshold, precision, recall, f1)
    sweep_thresholds: np.ndarray  # (S,)
    sweep_precision: np.ndarray  # (S,)
    sweep_recall: np.ndarray  # (S,)
    sweep_f1: np.ndarray  # (S,)


def _prf(a_bin: np.ndarray, gcg_bin: np.ndarray) -> tuple[float, float, float, int, int, int]:
    """Precision, recall, F1, TP, FP, FN from two off-diagonal boolean graphs."""
    tp = int((a_bin & gcg_bin).sum())
    fp = int((a_bin & ~gcg_bin).sum())
    fn = int((~a_bin & gcg_bin).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def _binarise_adjacency(
    adjacency: np.ndarray,
    threshold: float,
    top_k: int | None,
    off_diagonal: np.ndarray,
) -> np.ndarray:
    """Binarise the learned adjacency, off-diagonal only.

    ``top_k`` (matched-sparsity mode) keeps each node's k strongest incoming
    edges by weight — the correct comparison against a top-k GCG, since STGNN
    adjacencies are typically row-normalised to near-uniform weights that an
    absolute ``threshold`` would zero out entirely. Falls back to an absolute
    weight cutoff when ``top_k`` is None.
    """
    if top_k is not None:
        from .lens2_granger import build_gcg_topk

        return build_gcg_topk(adjacency, top_k).astype(bool) & off_diagonal
    return (adjacency > threshold) & off_diagonal


def _edge_classification_matrix(a_bin: np.ndarray, gcg_bin: np.ndarray, shape) -> np.ndarray:
    """Return edge classification codes: 0=TN/diagonal, 1=TP, 2=FP, 3=FN."""
    matrix = np.zeros(shape, dtype=np.uint8)
    matrix[a_bin & gcg_bin] = 1
    matrix[a_bin & ~gcg_bin] = 2
    matrix[~a_bin & gcg_bin] = 3
    return matrix


def run_lens3(
    adjacency: np.ndarray,
    gcg_matrix: np.ndarray,
    threshold: float = 0.1,
    sweep_min: float = 0.0,
    sweep_max: float = 1.0,
    sweep_steps: int = 21,
    top_k: int | None = None,
) -> AlignmentResult:
    """Compute structural alignment between learned adjacency and the GCG.

    Parameters
    ----------
    adjacency:
        Learned adjacency matrix, shape (N, N), values in [0, 1].
    gcg_matrix:
        Binary ground-truth Granger graph from Lens 2, shape (N, N).
    threshold:
        Edge-weight cutoff for binarising the adjacency (headline AAS) when
        ``top_k`` is None.
    sweep_min, sweep_max, sweep_steps:
        Range and resolution for the sweep plot (threshold sweep when ``top_k``
        is None, otherwise a k sweep from 1 to ``2 * top_k``).
    top_k:
        Matched-sparsity mode. When set, the adjacency is binarised by keeping
        each node's ``top_k`` strongest incoming edges — the same construction
        as the GCG — so the AAS compares two graphs of equal density instead of
        thresholding a near-uniform adjacency into emptiness. ``threshold`` is
        ignored; the returned ``threshold`` field carries ``top_k``.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    gcg = np.asarray(gcg_matrix, dtype=np.uint8)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be a square 2-D matrix; got {adj.shape}.")
    if gcg.shape != adj.shape:
        raise ValueError("adjacency and gcg_matrix must have the same shape.")

    n = adj.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)
    gcg_bin = gcg.astype(bool) & off_diagonal

    a_bin = _binarise_adjacency(adj, threshold, top_k, off_diagonal)
    precision, recall, f1, tp, fp, fn = _prf(a_bin, gcg_bin)
    tp_fp_fn_matrix = _edge_classification_matrix(a_bin, gcg_bin, adj.shape)
    eff_threshold = float(top_k) if top_k is not None else threshold

    # Weighted precision: how much off-diagonal weight sits on real GCG edges
    adj_off_diagonal = np.where(off_diagonal, adj, 0.0)
    total_weight = adj_off_diagonal.sum()
    w_precision = float((adj_off_diagonal * gcg).sum() / total_weight) if total_weight > 0 else 0.0

    # Sweep: over k (matched-sparsity mode) or over the weight threshold.
    if top_k is not None:
        k_max = int(min(max(2 * top_k, top_k), n - 1))
        sweep_x = np.unique(np.linspace(1, k_max, sweep_steps).astype(int)).astype(np.float64)
    else:
        sweep_x = np.linspace(sweep_min, sweep_max, sweep_steps)
    s_prec = np.zeros(sweep_x.size)
    s_rec = np.zeros(sweep_x.size)
    s_f1 = np.zeros(sweep_x.size)
    for i, x in enumerate(sweep_x):
        sweep_bin = _binarise_adjacency(
            adj,
            float(x),
            int(x) if top_k is not None else None,
            off_diagonal,
        )
        s_prec[i], s_rec[i], s_f1[i], *_ = _prf(sweep_bin, gcg_bin)

    return AlignmentResult(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        threshold=eff_threshold,
        weighted_precision=w_precision,
        tp_fp_fn_matrix=tp_fp_fn_matrix,
        sweep_thresholds=sweep_x,
        sweep_precision=s_prec,
        sweep_recall=s_rec,
        sweep_f1=s_f1,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_threshold_sweep(result: AlignmentResult, model_name: str = ""):
    """Line chart of AAS (F1), precision, and recall vs edge-weight threshold."""
    import altair as alt
    import pandas as pd

    # A k-sweep (matched-sparsity mode) uses integer edge counts > 1; the
    # threshold sweep uses weights in [0, 1]. Label the axis accordingly.
    is_k_sweep = bool(result.sweep_thresholds.max() > 1.5)
    x_title = "Edges kept per node (k)" if is_k_sweep else "Edge-weight threshold"
    df = pd.DataFrame(
        {
            "threshold": result.sweep_thresholds.tolist(),
            "F1 (AAS)": result.sweep_f1.tolist(),
            "Precision": result.sweep_precision.tolist(),
            "Recall": result.sweep_recall.tolist(),
        }
    )
    lines = (
        alt.Chart(df)
        .transform_fold(
            ["F1 (AAS)", "Precision", "Recall"],
            as_=["metric", "score"],
        )
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title=x_title),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["F1 (AAS)", "Precision", "Recall"],
                    range=["steelblue", "#2ca02c", "#ff7f0e"],
                ),
            ),
            strokeWidth=alt.condition(
                alt.datum.metric == "F1 (AAS)", alt.value(2.5), alt.value(1.5)
            ),
            tooltip=[
                "metric:N",
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("score:Q", format=".3f"),
            ],
        )
    )
    vline = (
        alt.Chart(pd.DataFrame({"x": [result.threshold]}))
        .mark_rule(color="black", strokeDash=[4, 2], opacity=0.6)
        .encode(x="x:Q")
    )
    sweep_kind = "sparsity" if is_k_sweep else "threshold"
    title = f"AAS {sweep_kind} sweep" + (f" — {model_name}" if model_name else "")
    return (lines + vline).properties(title=title, width=550, height=300)


def plot_alignment_comparison(results: dict[str, AlignmentResult]):
    """Bar chart of the consensus-graph AAS across models.

    Each bar is ``F1`` of the model's seed-mean (consensus) adjacency — a
    diagnostic of the representative graph, ``F1(mean G)``. It is deliberately
    *not* the mean-of-evaluations AAS (``mean F1(G)``, the paper's primary
    metric with its seed spread); the two differ because F1 is nonlinear, so
    this plot is labelled as the consensus graph to keep them from being read
    interchangeably.

    One bar per model. Under matched-density (top-k) comparison the two graphs
    have identical edge counts, so FP == FN and therefore precision == recall ==
    F1 by construction — only the F1 (AAS) carries information, so plotting all
    three is redundant. Bars are sorted descending and labelled with the score.
    """
    import altair as alt
    import pandas as pd

    from .model_display import display_name, family_rank

    df = pd.DataFrame(
        [
            {"model": display_name(model), "AAS": r.f1, "_order": family_rank(model)}
            for model, r in results.items()
        ]
    )
    order = df.sort_values("_order")["model"].tolist()
    y_max = max(0.1, float(df["AAS"].max()) * 1.15) if len(df) else 0.1
    # Bars follow the manuscript taxonomy order (not descending by value), and
    # fonts are enlarged for print legibility.
    base = alt.Chart(df).encode(
        x=alt.X(
            "model:N",
            title=None,
            sort=order,
            axis=alt.Axis(labelAngle=-30, labelFontSize=14, tickSize=4),
        ),
        y=alt.Y(
            "AAS:Q",
            title="AAS",
            scale=alt.Scale(domain=[0, y_max]),
            axis=alt.Axis(labelFontSize=13, titleFontSize=15),
        ),
    )
    bars = base.mark_bar(opacity=0.85, color="steelblue").encode(
        tooltip=["model:N", alt.Tooltip("AAS:Q", format=".3f")],
    )
    labels = base.mark_text(dy=-6, fontSize=13).encode(text=alt.Text("AAS:Q", format=".3f"))
    return (
        (bars + labels)
        .properties(
            title=alt.TitleParams(
                "Consensus-graph AAS across models (F1 of the seed-mean adjacency)",
                fontSize=15,
            ),
            width=max(300, len(results) * 90),
            height=350,
        )
        .configure_view(strokeWidth=0)
    )
