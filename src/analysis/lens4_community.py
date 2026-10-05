"""Lens 4 — Community Coherence Test.

Detects communities in the learned adjacency matrix (Louvain) and validates
whether they align with real geographic road structure (Haversine GCS).

Input shapes
------------
adjacency : (N, N)  row-normalised learned weights
coords    : DataFrame with columns [node_id, latitude, longitude]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------


def _pairwise_haversine_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorised pairwise Haversine distances in kilometres."""
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CommunityResult:
    """Results from Lens 4."""

    community_assignments: np.ndarray  # (N,) int — community id per node
    modularity: float
    num_communities: int
    # Modularity significance vs a degree-preserving edge-swap null (nan when
    # not computed). z = (Q_obs - null_mean) / null_std; null_mean is the
    # random-structure floor for this graph's own degree sequence.
    modularity_null_mean: float = float("nan")
    modularity_null_std: float = float("nan")
    modularity_zscore: float = float("nan")
    gcs_per_community: dict[int, float] = field(default_factory=dict)
    gcs_overall: float = 0.0
    degree_centrality: np.ndarray = field(default_factory=lambda: np.array([]))
    betweenness_centrality: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvector_centrality: np.ndarray = field(default_factory=lambda: np.array([]))
    closeness_centrality: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Modularity null
# ---------------------------------------------------------------------------


def _modularity_null(
    G, resolution: float, permutations: int, random_seed: int | None
) -> tuple[float, float, float]:
    """Modularity of degree-preserving edge-swapped copies of ``G``.

    Returns ``(null_mean, null_std, zscore)`` where ``zscore = (Q_obs -
    null_mean) / null_std``. Randomising while preserving each node's degree
    gives the chance modularity for this graph's own degree sequence, so the
    excess over the null is comparable across models of different density.
    Returns NaNs when the graph is too small/sparse to swap.
    """
    import community as community_louvain
    import networkx as nx

    n_edges = G.number_of_edges()
    if permutations < 1 or n_edges < 2:
        return float("nan"), float("nan"), float("nan")

    q_obs = community_louvain.modularity(
        community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=0),
        G,
        weight="weight",
    )
    null_vals: list[float] = []
    for r in range(permutations):
        rand_g = G.copy()
        try:
            nx.double_edge_swap(
                rand_g, nswap=5 * n_edges, max_tries=50 * n_edges, seed=(random_seed or 0) + r
            )
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue
        seed_r = None if random_seed is None else random_seed + 1000 + r
        part = community_louvain.best_partition(
            rand_g, weight="weight", resolution=resolution, random_state=seed_r
        )
        null_vals.append(community_louvain.modularity(part, rand_g, weight="weight"))

    if not null_vals:
        return float("nan"), float("nan"), float("nan")
    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    zscore = (q_obs - null_mean) / null_std if null_std > 0 else float("nan")
    return null_mean, null_std, zscore


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def run_lens4(
    adjacency: np.ndarray,
    coords: pd.DataFrame,
    num_runs: int = 10,
    resolution: float = 1.0,
    random_seed: int | None = 0,
    top_k: int | None = None,
    null_permutations: int = 20,
) -> CommunityResult:
    """Detect communities and compute geographic coherence.

    Parameters
    ----------
    adjacency:
        Learned adjacency matrix, shape (N, N), non-negative edge weights.
    coords:
        DataFrame with columns ``node_id``, ``latitude``, ``longitude``.
        Row order must correspond to node indices 0..N-1.
    num_runs:
        Number of Louvain runs; best modularity partition is returned.
    resolution:
        Louvain resolution parameter — higher values yield more communities.
    random_seed:
        Base random seed for Louvain runs.  None leaves Louvain stochastic.
    top_k:
        When set, community detection and graph centralities run on the
        density-matched graph (each node's ``top_k`` strongest edges, symmetrised
        and unweighted) rather than the raw weighted adjacency. This makes
        modularity comparable across models — otherwise a model with a peaked
        weight distribution scores higher modularity purely from edge density,
        not community structure. Weighted degree centrality still uses the raw
        adjacency.
    """
    try:
        import community as community_louvain
    except ImportError as e:
        raise ImportError(
            "Lens 4 requires python-louvain. Install it with: pip install python-louvain"
        ) from e

    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("Lens 4 requires networkx. Install it with: pip install networkx") from e

    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square 2-D; got {adj.shape}.")
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")
    N = adj.shape[0]
    if len(coords) != N:
        raise ValueError(f"coords has {len(coords)} rows but adjacency has {N} nodes.")
    required_columns = {"node_id", "latitude", "longitude"}
    missing_columns = required_columns - set(coords.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"coords is missing required columns: {missing}.")

    lats = coords["latitude"].to_numpy(dtype=np.float64)
    lons = coords["longitude"].to_numpy(dtype=np.float64)
    dist_matrix = _pairwise_haversine_km(lats, lons)  # (N, N)

    # Build the undirected graph for community detection / centralities. With
    # top_k, use the density-matched top-k graph (symmetrised, unweighted) so
    # modularity reflects structure, not each model's edge-weight scale.
    if top_k is not None:
        from .lens2_granger import build_gcg_topk

        directed = build_gcg_topk(adj, top_k)
        graph_matrix = ((directed | directed.T) > 0).astype(np.float64)
    else:
        graph_matrix = adj
    G = nx.from_numpy_array(graph_matrix)

    # Run Louvain multiple times; take partition with best modularity
    best_partition: dict[int, int] | None = None
    best_modularity = -1.0

    for run_idx in range(num_runs):
        run_seed = None if random_seed is None else random_seed + run_idx
        partition = community_louvain.best_partition(
            G,
            weight="weight",
            resolution=resolution,
            random_state=run_seed,
        )
        mod = community_louvain.modularity(partition, G, weight="weight")
        if mod > best_modularity:
            best_modularity = mod
            best_partition = partition

    assert best_partition is not None
    assignments = np.array([best_partition[i] for i in range(N)], dtype=np.int32)
    community_ids = np.unique(assignments)

    # Modularity significance: compare the observed modularity to a null of
    # degree-preserving edge-swapped graphs. This gives each model its own
    # random-structure floor (a high-degree graph has a higher chance
    # modularity), so the effect size Q_obs - null_mean is comparable across
    # models regardless of density/degree.
    null_mean, null_std, zscore = _modularity_null(G, resolution, null_permutations, random_seed)

    # Geographic Coherence Score per community
    gcs_per_community: dict[int, float] = {}
    for cid in community_ids:
        intra_mask = assignments == cid
        inter_mask = ~intra_mask
        n_intra = intra_mask.sum()
        n_inter = inter_mask.sum()

        if n_intra < 2 or n_inter == 0:
            # Singleton community or no inter-community nodes — skip
            gcs_per_community[int(cid)] = float("nan")
            continue

        intra_dists = dist_matrix[np.ix_(intra_mask, intra_mask)]
        # Exclude diagonal (self-distance = 0)
        intra_idx = np.triu_indices(n_intra, k=1)
        intra_mean = intra_dists[intra_idx].mean() if len(intra_idx[0]) > 0 else 0.0

        inter_dists = dist_matrix[np.ix_(intra_mask, inter_mask)]
        inter_mean = inter_dists.mean()

        gcs_per_community[int(cid)] = (
            float(inter_mean / intra_mean) if intra_mean > 0 else float("nan")
        )

    # Overall GCS: weighted mean by community size, ignoring NaN communities
    sizes = np.array([int((assignments == cid).sum()) for cid in community_ids], dtype=np.float64)
    gcs_vals = np.array([gcs_per_community[int(cid)] for cid in community_ids])
    valid = ~np.isnan(gcs_vals)
    gcs_overall = float(
        np.average(gcs_vals[valid], weights=sizes[valid]) if valid.any() else float("nan")
    )

    # Centrality measures
    degree_centrality = adj.sum(axis=1)

    try:
        bet_c = nx.betweenness_centrality(G, weight="weight", normalized=True, k=min(50, N))
        betweenness_centrality = np.array([bet_c[i] for i in range(N)])
    except Exception:
        betweenness_centrality = np.full(N, float("nan"))

    try:
        eig_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000, tol=1e-6)
        eigenvector_centrality = np.array([eig_c[i] for i in range(N)])
    except Exception:
        eigenvector_centrality = np.full(N, float("nan"))

    try:
        clo_c = nx.closeness_centrality(G)
        closeness_centrality = np.array([clo_c[i] for i in range(N)])
    except Exception:
        closeness_centrality = np.full(N, float("nan"))

    return CommunityResult(
        community_assignments=assignments,
        modularity=float(best_modularity),
        num_communities=int(len(community_ids)),
        modularity_null_mean=null_mean,
        modularity_null_std=null_std,
        modularity_zscore=zscore,
        gcs_per_community=gcs_per_community,
        gcs_overall=gcs_overall,
        degree_centrality=degree_centrality,
        betweenness_centrality=betweenness_centrality,
        eigenvector_centrality=eigenvector_centrality,
        closeness_centrality=closeness_centrality,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _style_geo_map(fig):
    """Apply a light basemap and marker outlines for static PNG contrast."""
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
        width=750,
        height=600,
        paper_bgcolor="#f4f6f8",
        plot_bgcolor="#f4f6f8",
    )
    return fig


def plot_community_map(result: CommunityResult, coords: pd.DataFrame, title: str = ""):
    """Scatter plot of nodes coloured by community membership."""
    import plotly.express as px

    df = coords.copy()
    df = df.assign(community=result.community_assignments.astype(str))
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="community",
        hover_data={"node_id": True, "community": True, "latitude": ":.5f", "longitude": ":.5f"},
        title=(
            title
            or f"Community map ({result.num_communities} communities, Q={result.modularity:.3f})"
        ),
        labels={"community": "Community"},
    )
    return _style_geo_map(fig)


def plot_adjacency_heatmap(adjacency: np.ndarray, result: CommunityResult, title: str = ""):
    """Adjacency matrix reordered by community, with community boundary lines."""
    import plotly.graph_objects as go

    order = np.argsort(result.community_assignments)
    adj_reordered = adjacency[np.ix_(order, order)]
    boundaries = (np.where(np.diff(result.community_assignments[order]))[0] + 1).tolist()
    n = len(order)

    shapes = []
    for b in boundaries:
        shapes.append(
            dict(
                type="line",
                x0=b - 0.5,
                x1=b - 0.5,
                y0=-0.5,
                y1=n - 0.5,
                line=dict(color="cyan", width=0.8),
            )
        )
        shapes.append(
            dict(
                type="line",
                y0=b - 0.5,
                y1=b - 0.5,
                x0=-0.5,
                x1=n - 0.5,
                line=dict(color="cyan", width=0.8),
            )
        )
    fig = go.Figure(
        go.Heatmap(
            z=adj_reordered.tolist(),
            colorscale="Hot_r",
            showscale=True,
            colorbar=dict(title="Edge weight"),
            hovertemplate="Row=%{y}<br>Col=%{x}<br>Weight=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or "Adjacency matrix (community-ordered)",
        xaxis_title="Node (reordered by community)",
        yaxis_title="Node (reordered by community)",
        yaxis_autorange="reversed",
        shapes=shapes,
        width=600,
        height=580,
    )
    return fig


def plot_centrality_map(
    result: CommunityResult,
    coords: pd.DataFrame,
    centrality: str = "degree",
):
    """Geographic scatter coloured by a chosen centrality measure."""
    import plotly.express as px

    centrality_map = {
        "degree": result.degree_centrality,
        "betweenness": result.betweenness_centrality,
        "eigenvector": result.eigenvector_centrality,
        "closeness": result.closeness_centrality,
    }
    if centrality not in centrality_map:
        raise ValueError(f"centrality must be one of {list(centrality_map)}.")

    df = coords.copy()
    df = df.assign(centrality_value=centrality_map[centrality])
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="centrality_value",
        color_continuous_scale="YlOrRd",
        hover_data={
            "node_id": True,
            "centrality_value": ":.4f",
            "latitude": ":.5f",
            "longitude": ":.5f",
        },
        labels={"centrality_value": f"{centrality.capitalize()} centrality"},
        title=f"{centrality.capitalize()} centrality map",
    )
    return _style_geo_map(fig)


def plot_centrality_by_community(
    result: CommunityResult,
    centrality: str = "degree",
):
    """Box plot of centrality values grouped by detected community."""
    import altair as alt
    import pandas as pd

    centrality_map = {
        "degree": result.degree_centrality,
        "betweenness": result.betweenness_centrality,
        "eigenvector": result.eigenvector_centrality,
        "closeness": result.closeness_centrality,
    }
    if centrality not in centrality_map:
        raise ValueError(f"centrality must be one of {list(centrality_map)}.")

    df = pd.DataFrame(
        {
            "community": result.community_assignments.astype(str),
            "centrality": centrality_map[centrality].tolist(),
        }
    )
    return (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("community:N", title="Community"),
            y=alt.Y("centrality:Q", title=f"{centrality.capitalize()} centrality"),
            color=alt.Color("community:N", legend=None),
            tooltip=["community:N", alt.Tooltip("centrality:Q", format=".4f")],
        )
        .properties(
            title=f"{centrality.capitalize()} centrality by community",
            width=max(300, result.num_communities * 60),
            height=300,
        )
    )
