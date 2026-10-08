"""PyGWalker exploration utility for STGNN-Probe results.

Not used in the paper: an optional notebook utility.

Call ``explore(result)`` in a notebook cell to open one interactive
drag-and-drop explorer over node-level, horizon-level, and alignment metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .probe import ProbeResult


def _load_pygwalker():
    try:
        import pygwalker as pyg
    except ImportError as e:
        raise ImportError(
            "PyGWalker exploration requires pygwalker. Install with: pip install pygwalker"
        ) from e
    return pyg


def _node_frame(result: ProbeResult) -> pd.DataFrame:
    l1, l4 = result.lens1, result.lens4
    num_nodes = len(l1.sgs_mean)
    df = pd.DataFrame(
        {
            "table": "nodes",
            "node_id": np.arange(num_nodes, dtype=np.int32),
            "sgs_mean": l1.sgs_mean,
            "node_label": l1.node_labels,
            "community_id": l4.community_assignments,
            "degree_centrality": l4.degree_centrality,
            "betweenness_centrality": l4.betweenness_centrality,
            "eigenvector_centrality": l4.eigenvector_centrality,
            "closeness_centrality": l4.closeness_centrality,
        }
    )
    degree_comparison = result.degree_comparison.set_index("node_id")
    degree_columns = [
        "learned_out_degree",
        "learned_in_degree",
        "gcg_out_degree",
        "gcg_in_degree",
        "out_degree_delta",
        "in_degree_delta",
    ]
    for col in degree_columns:
        df[col] = degree_comparison[col].to_numpy()
    return df


def _horizon_frame(result: ProbeResult, horizon_labels: list[str] | None = None) -> pd.DataFrame:
    l5 = result.lens5
    num_horizons = len(l5.horizon_steps)
    labels = horizon_labels or [str(int(h)) for h in l5.horizon_steps]
    return pd.DataFrame(
        {
            "table": "horizons",
            "horizon_step": l5.horizon_steps.tolist(),
            "horizon_label": labels[:num_horizons],
            "mean_sgs": l5.mean_sgs_per_horizon.tolist(),
            "pct_beneficial": (l5.pct_beneficial_per_horizon * 100).tolist(),
            "aas": l5.aas_per_horizon.tolist(),
        }
    )


def _alignment_frame(result: ProbeResult) -> pd.DataFrame:
    l3 = result.lens3
    return pd.DataFrame(
        {
            "table": "alignment",
            "threshold": l3.sweep_thresholds.tolist(),
            "precision": l3.sweep_precision.tolist(),
            "recall": l3.sweep_recall.tolist(),
            "f1": l3.sweep_f1.tolist(),
        }
    )


def explore(result: ProbeResult):
    """Open one PyGWalker explorer over node, horizon, and alignment metrics."""
    pyg = _load_pygwalker()
    df = pd.concat(
        [
            _node_frame(result),
            _horizon_frame(result),
            _alignment_frame(result),
        ],
        ignore_index=True,
        sort=False,
    )
    return pyg.walk(df)


def explore_nodes(result: ProbeResult, dataset_name: str = ""):
    """Open a PyGWalker explorer over node-level probe metrics."""
    pyg = _load_pygwalker()
    df = _node_frame(result)
    if dataset_name:
        df = df.assign(dataset_name=dataset_name)
    return pyg.walk(df)


def explore_horizons(result: ProbeResult, horizon_labels: list[str] | None = None):
    """Open a PyGWalker explorer over per-horizon SGS and AAS metrics."""
    pyg = _load_pygwalker()
    return pyg.walk(_horizon_frame(result, horizon_labels))


def explore_alignment(result: ProbeResult):
    """Open a PyGWalker explorer over the Lens 3 threshold sweep."""
    pyg = _load_pygwalker()
    return pyg.walk(_alignment_frame(result))
