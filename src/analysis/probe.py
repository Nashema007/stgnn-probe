"""ProbeRunner — orchestrates all five lenses for one model/dataset pair.

Usage
-----
runner = ProbeRunner(output_dir="outputs/")

# Run the dataset-level Granger computation once (cached automatically)
granger = runner.run_granger("METR-LA", raw_traffic)

# Run all lenses for each model
result = runner.run_model(
    model_name="gwn",
    dataset_name="METR-LA",
    model_predictions=...,   # (W, H, N, R)
    tcn_predictions=...,     # (W, H, N, R)
    ground_truth=...,        # (W, H, N)
    adjacency=...,           # (N, N)
    coords=...,              # DataFrame[node_id, latitude, longitude]
    horizon_steps=[6, 12, 18, 24, 30, 36, 42],
)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig, ProbeConfig
from .cross_model import compute_cross_model_correlations
from .io import (
    adjacency_path,
    comparative_dir,
    dataset_output_dir,
    granger_cache_exists,
    load_granger_result,
    model_horizon_adjacency_path,
    model_horizon_adjacency_seeds_path,
    model_output_dir,
    model_predictions_path,
    save_granger_result,
    save_json,
    save_npy,
    temporal_baseline_predictions_path,
)
from .lens0_performance import Lens0Result
from .lens0_performance import run_lens0 as _run_lens0_standalone
from .lens1_spatial_utility import SpatialUtilityResult, run_lens1
from .lens2_granger import GrangerResult, run_lens2
from .lens3_alignment import AlignmentResult, run_lens3
from .lens4_community import CommunityResult, run_lens4
from .lens5_degradation import DegradationResult, run_lens5


def _seed_mean(values: list[float]) -> float:
    """Mean over the finite seeds; nan only when every seed is non-finite.

    A single non-finite seed (e.g. an undefined modularity z-score when the
    null model is degenerate) must not wipe the whole estimate.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def _seed_sd(values: list[float]) -> float:
    """n-1 sample sd over the finite seeds; 0.0 when fewer than two are finite."""
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return 0.0
    return float(finite.std(ddof=1))


def _pooled_within_group_sd(groups: list[list[float]]) -> float:
    """Pooled *within-group* sample sd — the seed-stability spread.

    Each group is one horizon's seed evaluations. The pooled variance weights
    each horizon's within-seed variance by its degrees of freedom,
    ``sqrt( Σ_h (n_h-1) s_h² / Σ_h (n_h-1) )``, so horizon-to-horizon drift in
    the mean (which is what Lens 5 studies, not seed noise) never enters. Only
    finite values count; 0.0 when no horizon has ≥2 finite seeds.
    """
    num = 0.0
    den = 0.0
    for group in groups:
        arr = np.asarray(group, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size >= 2:
            num += (finite.size - 1) * float(finite.var(ddof=1))
            den += finite.size - 1
    return float(np.sqrt(num / den)) if den > 0 else 0.0


def _save_figure(fig, path: Path) -> None:
    """Save a Plotly or Altair figure to both PNG and HTML.

    PNG export requires kaleido (Plotly) or vl-convert-python (Altair).
    """
    html_path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "write_image"):
        fig.write_image(str(path), scale=2)
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    else:
        fig.save(str(path), scale_factor=2)
        fig.save(str(html_path))


def _group_by_dataset(
    results: dict[str, ProbeResult],
) -> dict[str, dict[str, ProbeResult]]:
    """Group ``{dataset}:{model}``-keyed results by dataset name, order-preserving."""
    grouped: dict[str, dict[str, ProbeResult]] = {}
    for name, result in results.items():
        dataset_name = name.split(":", 1)[0]
        grouped.setdefault(dataset_name, {})[name] = result
    return grouped


def normalize_adjacency(adjacency: np.ndarray, expected_nodes: int | None = None) -> np.ndarray:
    """Validate and row-normalize a finite non-negative adjacency matrix."""
    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be a square 2-D matrix; got {adj.shape}.")
    if expected_nodes is not None and adj.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            f"adjacency must have shape ({expected_nodes}, {expected_nodes}); got {adj.shape}."
        )
    if not np.isfinite(adj).all():
        raise ValueError("adjacency must contain only finite values.")
    if (adj < 0).any():
        raise ValueError("adjacency must be non-negative before row normalization.")

    row_sums = adj.sum(axis=1, keepdims=True)
    return np.divide(adj, row_sums, out=np.zeros_like(adj), where=row_sums > 0)


def compute_degree_comparison(adjacency: np.ndarray, gcg_matrix: np.ndarray) -> pd.DataFrame:
    """Compare learned weighted degree against binary Granger graph degree.

    Diagonal self-loops are excluded from both matrices. Ratios are zero when
    the corresponding GCG degree is zero so the output stays finite.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    gcg = np.asarray(gcg_matrix, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be a square 2-D matrix; got {adj.shape}.")
    if gcg.shape != adj.shape:
        raise ValueError("adjacency and gcg_matrix must have the same shape.")

    off_diagonal = ~np.eye(adj.shape[0], dtype=bool)
    learned = np.where(off_diagonal, adj, 0.0)
    gcg_binary = np.where(off_diagonal, gcg > 0, 0.0).astype(np.float64)

    learned_out = learned.sum(axis=1)
    learned_in = learned.sum(axis=0)
    gcg_out = gcg_binary.sum(axis=1)
    gcg_in = gcg_binary.sum(axis=0)
    out_delta = learned_out - gcg_out
    in_delta = learned_in - gcg_in
    out_ratio = np.divide(learned_out, gcg_out, out=np.zeros_like(learned_out), where=gcg_out > 0)
    in_ratio = np.divide(learned_in, gcg_in, out=np.zeros_like(learned_in), where=gcg_in > 0)

    return pd.DataFrame(
        {
            "node_id": np.arange(adj.shape[0], dtype=np.int32),
            "learned_out_degree": learned_out,
            "learned_in_degree": learned_in,
            "gcg_out_degree": gcg_out,
            "gcg_in_degree": gcg_in,
            "out_degree_delta": out_delta,
            "in_degree_delta": in_delta,
            "out_degree_ratio": out_ratio,
            "in_degree_ratio": in_ratio,
        }
    )


def plot_degree_comparison(degree_comparison: pd.DataFrame):
    """Two-panel Altair scatter comparing learned and GCG node degrees."""
    import altair as alt

    def _panel(x_col: str, y_col: str, title: str, max_val: float) -> alt.LayerChart:
        scatter = (
            alt.Chart(degree_comparison)
            .mark_point(color="steelblue", opacity=0.8, size=30)
            .encode(
                x=alt.X(f"{x_col}:Q", title="GCG degree"),
                y=alt.Y(f"{y_col}:Q", title="Learned weighted degree"),
                tooltip=[
                    alt.Tooltip(f"{x_col}:Q", format=".2f"),
                    alt.Tooltip(f"{y_col}:Q", format=".2f"),
                ],
            )
        )
        diag_df = pd.DataFrame({"_x": [0.0, max_val], "_y": [0.0, max_val]})
        diagonal = (
            alt.Chart(diag_df)
            .mark_line(color="black", strokeDash=[4, 3], strokeWidth=0.8)
            .encode(x="_x:Q", y="_y:Q")
        )
        return (scatter + diagonal).properties(title=title, width=300, height=300)

    out_max = max(
        float(degree_comparison["gcg_out_degree"].max()),
        float(degree_comparison["learned_out_degree"].max()),
        1.0,
    )
    in_max = max(
        float(degree_comparison["gcg_in_degree"].max()),
        float(degree_comparison["learned_in_degree"].max()),
        1.0,
    )
    return _panel("gcg_out_degree", "learned_out_degree", "Outgoing degree", out_max) | _panel(
        "gcg_in_degree", "learned_in_degree", "Incoming degree", in_max
    )


@dataclass
class StructureSeedSummary:
    """Structural measures computed per horizon×seed run, then aggregated.

    Training seeds are allocated independently for each horizon, so each measure
    is scored on every actual horizon×seed adjacency and only then aggregated;
    array position is never treated as a common seed across horizons.

    Three spreads are reported and must not be conflated:

    * ``*_seed_sd_pooled`` — pooled *within-horizon* seed sd. The seed-stability
      spread, the only one that answers "how sensitive is the measure to the
      random seed?". Use this (or the per-horizon seed sds) for stability claims.
    * ``per_horizon_*_seed_sd`` — the primary reporting form: one seed sd per
      horizon, alongside ``per_horizon_*_mean``.
    * ``*_horizon_mean_sd`` — sd of the per-horizon means, i.e. how much the
      measure moves *across* horizons (the Lens-5 degradation phenomenon), not
      seed noise.

    ``*_evaluation_grand_mean`` is the mean over all horizon×seed evaluations —
    the expected value for one individual model evaluation, and the primary
    point estimate. It is distinct from ``ProbeResult.lens3.f1`` /
    ``lens4.modularity``, which score the seed-mean *consensus* graph
    (``F1(mean G) ≠ mean F1(G)``) and are diagnostics of the representative
    graph, not average evaluation results.

    All sds are n-1 sample sds and are 0.0 where fewer than two finite values
    exist. With R=3 they are noisy — use them to bound ranking claims.
    """

    horizons: list[int]
    per_horizon_num_seeds: list[int]
    num_seed_evaluations: int
    # AAS (F1) over horizon×seed evaluations
    aas_evaluation_grand_mean: float
    aas_seed_sd_pooled: float
    aas_horizon_mean_sd: float
    aas_total_sd: float  # horizon+seed variation combined — NOT a seed-stability claim
    per_horizon_aas_mean: list[float]
    per_horizon_aas_seed_sd: list[float]
    # Modularity Q
    modularity_evaluation_grand_mean: float
    modularity_seed_sd_pooled: float
    modularity_horizon_mean_sd: float
    modularity_total_sd: float
    per_horizon_modularity_mean: list[float]
    per_horizon_modularity_seed_sd: list[float]
    # Modularity z-score (excess-Q)
    modularity_z_evaluation_grand_mean: float
    modularity_z_seed_sd_pooled: float
    modularity_z_horizon_mean_sd: float
    modularity_z_total_sd: float
    per_horizon_modularity_z_mean: list[float]
    per_horizon_modularity_z_seed_sd: list[float]

    def to_report_dict(self) -> dict[str, Any]:
        """Nested, explicitly-named JSON with the seed vs horizon spreads kept apart."""

        def _per_horizon(means: list[float], seed_sds: list[float]) -> dict[str, dict[str, float]]:
            return {
                str(h): {"mean": m, "seed_sd": s}
                for h, m, s in zip(self.horizons, means, seed_sds, strict=True)
            }

        return {
            "horizons": self.horizons,
            "per_horizon_num_seeds": self.per_horizon_num_seeds,
            "num_seed_evaluations": self.num_seed_evaluations,
            "aas": {
                "evaluation_grand_mean": self.aas_evaluation_grand_mean,
                "seed_sd_pooled": self.aas_seed_sd_pooled,
                "horizon_mean_sd": self.aas_horizon_mean_sd,
                "total_sd": self.aas_total_sd,
                "per_horizon": _per_horizon(
                    self.per_horizon_aas_mean, self.per_horizon_aas_seed_sd
                ),
            },
            "modularity": {
                "evaluation_grand_mean": self.modularity_evaluation_grand_mean,
                "seed_sd_pooled": self.modularity_seed_sd_pooled,
                "horizon_mean_sd": self.modularity_horizon_mean_sd,
                "total_sd": self.modularity_total_sd,
                "per_horizon": _per_horizon(
                    self.per_horizon_modularity_mean, self.per_horizon_modularity_seed_sd
                ),
            },
            "modularity_zscore": {
                "evaluation_grand_mean": self.modularity_z_evaluation_grand_mean,
                "seed_sd_pooled": self.modularity_z_seed_sd_pooled,
                "horizon_mean_sd": self.modularity_z_horizon_mean_sd,
                "total_sd": self.modularity_z_total_sd,
                "per_horizon": _per_horizon(
                    self.per_horizon_modularity_z_mean,
                    self.per_horizon_modularity_z_seed_sd,
                ),
            },
        }


@dataclass
class ProbeResult:
    """Full per-model analysis result."""

    model_name: str
    dataset_name: str
    lens1: SpatialUtilityResult
    lens3: AlignmentResult
    lens4: CommunityResult
    lens5: DegradationResult
    degree_comparison: pd.DataFrame
    lens3_per_horizon: list[AlignmentResult] | None = None
    structure_seed_summary: StructureSeedSummary | None = None

    def summary(self) -> dict[str, float]:
        """Key scalar metrics for quick inspection."""
        # `*_consensus_graph` = measure of the seed-mean representative graph
        # (F1(mean G), a diagnostic of the consensus graph). Distinct from the
        # `*_evaluation_grand_mean` fields below, which are mean(F1(G)) over
        # individual evaluations. The two are not interchangeable, so neither is
        # named with a bare `aas`/`modularity` that plotting code could grab by
        # accident.
        s = {
            "mean_sgs": self.lens1.mean_sgs,
            "mean_sgs_rel": self.lens1.mean_sgs_rel,
            "pct_beneficial": self.lens1.pct_beneficial,
            "pct_harmful": self.lens1.pct_harmful,
            "aas_consensus_graph": self.lens3.f1,
            "aas_consensus_precision": self.lens3.precision,
            "aas_consensus_recall": self.lens3.recall,
            "community_modularity_consensus_graph": self.lens4.modularity,
            "community_modularity_null": self.lens4.modularity_null_mean,
            "community_modularity_z_consensus_graph": self.lens4.modularity_zscore,
            "community_gcs": self.lens4.gcs_overall,
            "sgs_rate": self.lens5.sgs_rate,
            "aas_rate": self.lens5.aas_rate,
        }
        # Mean-of-evaluations statistics: the primary paper metrics. Seed spread
        # (`*_seed_sd_pooled`, within-horizon) is kept apart from horizon spread
        # (`*_horizon_mean_sd`) so degradation is never reported as seed noise.
        if self.structure_seed_summary is not None:
            ss = self.structure_seed_summary
            s.update(
                {
                    "num_seed_evaluations": float(ss.num_seed_evaluations),
                    "aas_evaluation_grand_mean": ss.aas_evaluation_grand_mean,
                    "aas_seed_sd_pooled": ss.aas_seed_sd_pooled,
                    "aas_horizon_mean_sd": ss.aas_horizon_mean_sd,
                    "community_modularity_evaluation_grand_mean": (
                        ss.modularity_evaluation_grand_mean
                    ),
                    "community_modularity_seed_sd_pooled": ss.modularity_seed_sd_pooled,
                    "community_modularity_horizon_mean_sd": ss.modularity_horizon_mean_sd,
                }
            )
        return s

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'─' * 55}")
        print(f"  STGNN-Probe — {self.model_name} on {self.dataset_name}")
        print(f"{'─' * 55}")
        print(f"  Lens 1  Mean SGS:        {s['mean_sgs']:+.4f}")
        print(f"          % beneficial:    {s['pct_beneficial'] * 100:.1f}%")
        print(f"          % harmful:       {s['pct_harmful'] * 100:.1f}%")
        print(f"  Lens 3  AAS (consensus): {s['aas_consensus_graph']:.4f}")
        print(f"          Precision:       {s['aas_consensus_precision']:.4f}")
        print(f"          Recall:          {s['aas_consensus_recall']:.4f}")
        ss = self.structure_seed_summary
        if ss is not None:
            print(
                f"          AAS grand mean:  {ss.aas_evaluation_grand_mean:.4f} "
                f"(seed sd {ss.aas_seed_sd_pooled:.4f}, horizon sd {ss.aas_horizon_mean_sd:.4f})"
            )
        print(f"  Lens 4  Communities:     {self.lens4.num_communities}")
        print(f"          Modularity Q:    {s['community_modularity_consensus_graph']:.4f}")
        _l4 = self.lens4
        _q_null = f"{_l4.modularity_null_mean:.3f} / {_l4.modularity_zscore:+.0f}"
        print(f"          Q null/z:        {_q_null}")
        print(f"          GCS overall:     {s['community_gcs']:.4f}")
        print(f"  Lens 5  SGS slope/min:   {s['sgs_rate']:+.5f}")
        print(f"          AAS slope/min:   {s['aas_rate']:+.5f}")
        print(f"{'─' * 55}")


class ProbeRunner:
    """Runs all five lenses and manages the output directory and Granger cache.

    Parameters
    ----------
    output_dir:
        Root directory for all outputs.  Created on first use.
    config:
        Probe configuration.  Defaults are used when None.
    """

    def __init__(self, output_dir: str | Path, config: ProbeConfig | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or ProbeConfig()
        self._granger_cache: dict[str, GrangerResult] = {}

    # ------------------------------------------------------------------
    # Lens 2 — cached per dataset
    # ------------------------------------------------------------------

    def run_granger(
        self,
        dataset_name: str,
        raw_traffic: np.ndarray,
        force_recompute: bool = False,
        save_figures: bool = True,
    ) -> GrangerResult:
        """Run Lens 2 for a dataset, loading from cache when available.

        Parameters
        ----------
        dataset_name:
            Identifier used for the cache file name.
        raw_traffic:
            Shape (T, N).
        force_recompute:
            Ignore any existing cache file and recompute.
        save_figures:
            Write dataset-level Granger PNG/HTML figures in addition to arrays.
        """
        if not force_recompute and dataset_name in self._granger_cache:
            result = self._granger_cache[dataset_name]
            self._save_granger_outputs(dataset_name, result, save_figures=save_figures)
            return result

        cfg = self.config.granger

        if not force_recompute and granger_cache_exists(self.output_dir, dataset_name):
            result = load_granger_result(self.output_dir, dataset_name, top_k=cfg.gcg_top_k)
            self._granger_cache[dataset_name] = result
            self._save_granger_outputs(dataset_name, result, save_figures=save_figures)
            return result

        result = run_lens2(
            raw_traffic,
            max_lag=cfg.max_lag,
            significance=cfg.significance,
            n_jobs=cfg.n_jobs,
            top_k=cfg.gcg_top_k,
        )
        save_granger_result(result, self.output_dir, dataset_name)
        self._save_granger_outputs(dataset_name, result, save_figures=save_figures)
        self._granger_cache[dataset_name] = result
        return result

    def _save_granger_outputs(
        self,
        dataset_name: str,
        result: GrangerResult,
        *,
        save_figures: bool,
    ) -> None:
        out = dataset_output_dir(self.output_dir, dataset_name) / "granger"
        out.mkdir(parents=True, exist_ok=True)
        save_npy(result.gcg_matrix, out / "gcg_matrix.npy")
        save_npy(result.pvalues, out / "granger_pvalues.npy")
        save_npy(result.fstats, out / "granger_fstats.npy")
        save_npy(result.pearson_correlations, out / "pearson_correlations.npy")

        if not save_figures:
            return

        from .lens2_granger import plot_gcg_heatmap, plot_gcg_network, plot_pvalue_histogram

        for fig, name in [
            (plot_gcg_heatmap(result), "gcg_heatmap.png"),
            (plot_gcg_network(result), "gcg_network.png"),
            (plot_pvalue_histogram(result), "pvalue_histogram.png"),
        ]:
            _save_figure(fig, out / name)

    def _dataset_config(self, dataset_name: str) -> DatasetConfig:
        for dataset in self.config.datasets:
            if dataset.name == dataset_name:
                return dataset
        raise ValueError(f"Unknown dataset {dataset_name!r}.")

    @staticmethod
    def _required_path(value: str | None, field_name: str) -> Path:
        if value is None:
            raise ValueError(f"Dataset config is missing {field_name}.")
        return Path(value)

    def run_from_config(
        self,
        dataset_name: str,
        model_name: str,
        *,
        force_granger: bool = False,
        save_figures: bool = True,
        tcn_baseline: str = "tcn",
    ) -> ProbeResult:
        """Load probe inputs from config and run all five lenses."""
        dataset = self._dataset_config(dataset_name)
        raw_traffic = np.load(self._required_path(dataset.raw_data, "raw_data"))
        ground_truth = np.load(self._required_path(dataset.ground_truth, "ground_truth"))
        coords = pd.read_csv(self._required_path(dataset.coordinates, "coordinates"))
        model_predictions = np.load(model_predictions_path(dataset, model_name))
        tcn_predictions = np.load(temporal_baseline_predictions_path(dataset, tcn_baseline))
        adjacency = normalize_adjacency(
            np.load(adjacency_path(dataset, model_name)),
            expected_nodes=dataset.num_nodes,
        )

        # Try to load per-horizon adjacency files (one per configured horizon).
        # Falls back to the single adjacency file when per-horizon files are absent.
        horizon_steps = dataset.horizons or list(range(1, ground_truth.shape[1] + 1))
        h_adjs: list[np.ndarray] = []
        for h in horizon_steps:
            p = model_horizon_adjacency_path(dataset, model_name, h)
            if p.exists():
                h_adjs.append(normalize_adjacency(np.load(p), expected_nodes=dataset.num_nodes))
        if len(h_adjs) == len(horizon_steps):
            adjacencies: list[np.ndarray] | None = h_adjs
            adjacency = np.mean(np.stack(h_adjs, axis=0), axis=0).astype(np.float64)
        else:
            adjacencies = None

        granger_result = self.run_granger(
            dataset_name,
            raw_traffic,
            force_recompute=force_granger,
            save_figures=save_figures,
        )
        self._save_dataset_gcg_communities(dataset_name, granger_result, coords)

        result = self.run_model(
            model_name=model_name,
            dataset_name=dataset_name,
            model_predictions=model_predictions,
            tcn_predictions=tcn_predictions,
            ground_truth=ground_truth,
            adjacency=adjacency,
            granger_result=granger_result,
            coords=coords,
            horizon_steps=horizon_steps,
            save_outputs=False,
            adjacencies=adjacencies,
        )
        # Mean-of-R structure: score every retained seed's adjacency and report
        # mean±sd, matching the mean-of-R accuracy table. Falls back to the
        # single seed-mean result when no per-seed stacks were retained.
        result.structure_seed_summary = self._compute_structure_seed_summary(
            dataset, model_name, granger_result, coords, horizon_steps
        )
        self._save_result(result)
        if save_figures:
            self.save_figures(
                result,
                coords,
                adjacency=adjacency,
                horizon_labels=[str(h) for h in dataset.horizon_minutes],
            )
        return result

    def run_all_from_config(
        self,
        *,
        force_granger: bool = False,
        save_figures: bool = True,
        run_performance: bool = True,
    ) -> dict[str, ProbeResult]:
        """Run every configured spatial model for every configured dataset.

        Also runs Lens 0 (forecasting performance benchmark) for every
        configured dataset unless ``run_performance=False``, so a single
        ``--all`` invocation covers both the structural lenses and the
        performance benchmark by default.
        """
        results: dict[str, ProbeResult] = {}
        for dataset in self.config.datasets:
            for model_name in self.config.models.spatial_models:
                key = f"{dataset.name}:{model_name}"
                results[key] = self.run_from_config(
                    dataset.name,
                    model_name,
                    force_granger=force_granger,
                    save_figures=save_figures,
                )
        if results and save_figures:
            self.save_comparative_figures(results)
        elif results:
            for dataset_name, ds_results in _group_by_dataset(results).items():
                comp_dir = comparative_dir(self.output_dir, dataset_name)
                comp_dir.mkdir(parents=True, exist_ok=True)
                summaries = {name: r.summary() for name, r in ds_results.items()}
                save_json(summaries, comp_dir / "comparative_summary.json")
                save_json(
                    compute_cross_model_correlations(summaries),
                    comp_dir / "cross_model_correlations.json",
                )
        if run_performance:
            self.run_all_performance_from_config(save_figures=save_figures)
        return results

    def _save_dataset_gcg_communities(
        self,
        dataset_name: str,
        granger_result: GrangerResult,
        coords: pd.DataFrame,
    ) -> None:
        community = run_lens4(
            granger_result.gcg_matrix.astype(np.float64),
            coords,
            num_runs=self.config.community.num_runs,
            resolution=self.config.community.resolution,
            random_seed=self.config.community.random_seed,
            null_permutations=0,  # GCG-community diagnostic; skip the null for speed
        )
        out = dataset_output_dir(self.output_dir, dataset_name) / "gcg_communities"
        self._save_community_outputs(community, out)

    # ------------------------------------------------------------------
    # Per-model run
    # ------------------------------------------------------------------

    def run_model(
        self,
        model_name: str,
        dataset_name: str,
        model_predictions: np.ndarray,
        tcn_predictions: np.ndarray,
        ground_truth: np.ndarray,
        adjacency: np.ndarray,
        granger_result: GrangerResult,
        coords: pd.DataFrame,
        horizon_steps: list[int] | None = None,
        save_outputs: bool = True,
        adjacencies: list[np.ndarray] | None = None,
    ) -> ProbeResult:
        """Run Lens 1, 3, 4, and 5 for one model.

        Parameters
        ----------
        model_predictions:
            Shape (W, H, N, R).
        tcn_predictions:
            Temporal-only baseline, shape (W, H, N, R).
        ground_truth:
            Shape (W, H, N).
        adjacency:
            Learned graph, shape (N, N), values in [0, 1].
        granger_result:
            Output from run_granger() for this dataset.
        coords:
            DataFrame with columns ``node_id``, ``latitude``, ``longitude``.
        horizon_steps:
            Horizon values in time-steps matching the H axis of predictions.
            Defaults to ``[1, 2, ..., H]`` when None.
        save_outputs:
            Write result arrays and summary JSON to the output directory.
        """
        H = ground_truth.shape[1]
        if horizon_steps is None:
            horizon_steps = list(range(1, H + 1))

        cfg = self.config

        # Lens 1 — Spatial Utility
        l1 = run_lens1(
            tcn_predictions,
            model_predictions,
            ground_truth,
            threshold=cfg.sgs_threshold,
            rel_threshold=cfg.sgs_rel_threshold,
        )

        # Lens 3 — Structural Alignment (global adjacency). Match the adjacency's
        # sparsity to the GCG's (top-k per node) so the AAS compares equal-density
        # graphs rather than thresholding a near-uniform adjacency to nothing.
        l3 = run_lens3(
            adjacency,
            granger_result.gcg_matrix,
            threshold=cfg.alignment.threshold,
            sweep_min=cfg.alignment.sweep_min,
            sweep_max=cfg.alignment.sweep_max,
            sweep_steps=cfg.alignment.sweep_steps,
            top_k=cfg.granger.gcg_top_k,
        )

        # Lens 3 per-horizon: compute one AlignmentResult per horizon adjacency
        l3_per_horizon: list[AlignmentResult] | None = None
        if adjacencies is not None:
            l3_per_horizon = [
                run_lens3(
                    adj_h,
                    granger_result.gcg_matrix,
                    threshold=cfg.alignment.threshold,
                    sweep_min=cfg.alignment.sweep_min,
                    sweep_max=cfg.alignment.sweep_max,
                    sweep_steps=cfg.alignment.sweep_steps,
                    top_k=cfg.granger.gcg_top_k,
                )
                for adj_h in adjacencies
            ]

        # Lens 4 — Community Coherence. Match the graph density to the AAS (top-k
        # per node) so modularity is comparable across models instead of being
        # driven by each model's raw edge-weight distribution.
        l4 = run_lens4(
            adjacency,
            coords,
            num_runs=cfg.community.num_runs,
            resolution=cfg.community.resolution,
            random_seed=cfg.community.random_seed,
            top_k=cfg.granger.gcg_top_k,
            null_permutations=cfg.community.null_permutations,
        )
        degree_comparison = compute_degree_comparison(adjacency, granger_result.gcg_matrix)

        # Lens 5 — Horizon Degradation
        l5 = run_lens5(
            l1.sgs_matrix,
            granger_result.gcg_matrix,
            adjacency,
            horizon_steps,
            alignment_threshold=cfg.alignment.threshold,
            adjacencies=adjacencies,
            top_k=cfg.granger.gcg_top_k,
        )

        result = ProbeResult(
            model_name=model_name,
            dataset_name=dataset_name,
            lens1=l1,
            lens3=l3,
            lens4=l4,
            lens5=l5,
            degree_comparison=degree_comparison,
            lens3_per_horizon=l3_per_horizon,
        )

        if save_outputs:
            self._save_result(result)

        return result

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------

    def _lens3_f1(self, adjacency: np.ndarray, gcg_matrix: np.ndarray) -> float:
        """Headline AAS (F1) for one already-normalized adjacency."""
        cfg = self.config
        return run_lens3(
            adjacency,
            gcg_matrix,
            threshold=cfg.alignment.threshold,
            sweep_min=cfg.alignment.sweep_min,
            sweep_max=cfg.alignment.sweep_max,
            sweep_steps=cfg.alignment.sweep_steps,
            top_k=cfg.granger.gcg_top_k,
        ).f1

    def _compute_structure_seed_summary(
        self,
        dataset: DatasetConfig,
        model_name: str,
        granger_result: GrangerResult,
        coords: pd.DataFrame,
        horizon_steps: list[int],
    ) -> StructureSeedSummary | None:
        """Compute measures for each retained horizon×seed adjacency.

        Reads the ``(R, N, N)`` per-horizon stacks written by training. Runs at
        different horizons use different seed draws, so global statistics pool
        scored horizon×seed observations rather than averaging matrices at the
        same array index. Returns ``None`` when the stacks are absent (older
        runs, or graph-free baselines).
        """
        cfg = self.config
        n = dataset.num_nodes
        gcg = granger_result.gcg_matrix

        horizons: list[int] = []
        per_h_counts: list[int] = []
        aas_groups: list[list[float]] = []  # one list of seed evals per horizon
        mod_groups: list[list[float]] = []
        modz_groups: list[list[float]] = []
        for h in horizon_steps:
            hpath = model_horizon_adjacency_seeds_path(dataset, model_name, h)
            if not hpath.exists():
                continue
            stack_h = np.load(hpath)
            if stack_h.ndim != 3 or stack_h.shape[0] == 0:
                continue
            aas_h: list[float] = []
            mod_h: list[float] = []
            modz_h: list[float] = []
            for seed_adj in stack_h:
                adj = normalize_adjacency(seed_adj, expected_nodes=n)
                aas_h.append(self._lens3_f1(adj, gcg))
                l4 = run_lens4(
                    adj,
                    coords,
                    num_runs=cfg.community.num_runs,
                    resolution=cfg.community.resolution,
                    random_seed=cfg.community.random_seed,
                    top_k=cfg.granger.gcg_top_k,
                    null_permutations=cfg.community.null_permutations,
                )
                mod_h.append(float(l4.modularity))
                modz_h.append(float(l4.modularity_zscore))
            horizons.append(int(h))
            per_h_counts.append(len(aas_h))
            aas_groups.append(aas_h)
            mod_groups.append(mod_h)
            modz_groups.append(modz_h)

        if not horizons:
            return None

        def _flat(groups: list[list[float]]) -> list[float]:
            return [x for group in groups for x in group]

        def _horizon_means(groups: list[list[float]]) -> list[float]:
            return [_seed_mean(group) for group in groups]

        aas_h_means = _horizon_means(aas_groups)
        mod_h_means = _horizon_means(mod_groups)
        modz_h_means = _horizon_means(modz_groups)

        return StructureSeedSummary(
            horizons=horizons,
            per_horizon_num_seeds=per_h_counts,
            num_seed_evaluations=len(_flat(aas_groups)),
            aas_evaluation_grand_mean=_seed_mean(_flat(aas_groups)),
            aas_seed_sd_pooled=_pooled_within_group_sd(aas_groups),
            aas_horizon_mean_sd=_seed_sd(aas_h_means),
            aas_total_sd=_seed_sd(_flat(aas_groups)),
            per_horizon_aas_mean=aas_h_means,
            per_horizon_aas_seed_sd=[_seed_sd(group) for group in aas_groups],
            modularity_evaluation_grand_mean=_seed_mean(_flat(mod_groups)),
            modularity_seed_sd_pooled=_pooled_within_group_sd(mod_groups),
            modularity_horizon_mean_sd=_seed_sd(mod_h_means),
            modularity_total_sd=_seed_sd(_flat(mod_groups)),
            per_horizon_modularity_mean=mod_h_means,
            per_horizon_modularity_seed_sd=[_seed_sd(group) for group in mod_groups],
            modularity_z_evaluation_grand_mean=_seed_mean(_flat(modz_groups)),
            modularity_z_seed_sd_pooled=_pooled_within_group_sd(modz_groups),
            modularity_z_horizon_mean_sd=_seed_sd(modz_h_means),
            modularity_z_total_sd=_seed_sd(_flat(modz_groups)),
            per_horizon_modularity_z_mean=modz_h_means,
            per_horizon_modularity_z_seed_sd=[_seed_sd(group) for group in modz_groups],
        )

    def _save_result(self, result: ProbeResult) -> None:
        out = model_output_dir(self.output_dir, result.model_name, result.dataset_name)
        out.mkdir(parents=True, exist_ok=True)
        if result.structure_seed_summary is not None:
            save_json(
                result.structure_seed_summary.to_report_dict(),
                out / "structure_seed_summary.json",
            )

        l1, l3, l4, l5 = result.lens1, result.lens3, result.lens4, result.lens5

        # Lens 1 arrays
        save_npy(l1.sgs_matrix, out / "lens1_sgs_matrix.npy")
        save_npy(l1.sgs_mean, out / "lens1_sgs_mean.npy")
        save_npy(l1.sgs_matrix, out / "sgs_matrix.npy")
        save_npy(l1.sgs_mean, out / "sgs_mean.npy")
        pd.DataFrame(
            {
                "node_id": np.arange(len(l1.sgs_mean), dtype=np.int32),
                "label": l1.node_labels,
                "sgs_mean": l1.sgs_mean,
            }
        ).to_csv(out / "node_classification.csv", index=False)

        # Lens 3 arrays
        save_npy(l3.sweep_thresholds, out / "lens3_sweep_thresholds.npy")
        save_npy(l3.sweep_f1, out / "lens3_sweep_f1.npy")
        save_npy(l3.sweep_precision, out / "lens3_sweep_precision.npy")
        save_npy(l3.sweep_recall, out / "lens3_sweep_recall.npy")
        save_npy(l3.tp_fp_fn_matrix, out / "tp_fp_fn_matrix.npy")
        save_json(
            {
                "precision": l3.precision,
                "recall": l3.recall,
                "f1": l3.f1,
                "threshold": l3.threshold,
                "weighted_precision": l3.weighted_precision,
                "tp": l3.tp,
                "fp": l3.fp,
                "fn": l3.fn,
            },
            out / "alignment_scores.json",
        )

        # Lens 3 per-horizon arrays (one JSON per horizon)
        if result.lens3_per_horizon is not None:
            ph_dir = out / "lens3_per_horizon"
            ph_dir.mkdir(parents=True, exist_ok=True)
            ph_records = []
            for h_idx, l3h in enumerate(result.lens3_per_horizon):
                ph_records.append(
                    {
                        "horizon_index": h_idx,
                        "precision": l3h.precision,
                        "recall": l3h.recall,
                        "f1": l3h.f1,
                        "threshold": l3h.threshold,
                        "tp": l3h.tp,
                        "fp": l3h.fp,
                        "fn": l3h.fn,
                    }
                )
            save_json({"per_horizon": ph_records}, ph_dir / "alignment_per_horizon.json")

        # Lens 4 arrays
        save_npy(l4.community_assignments, out / "lens4_community_assignments.npy")
        save_npy(l4.degree_centrality, out / "lens4_degree_centrality.npy")
        save_npy(l4.betweenness_centrality, out / "lens4_betweenness_centrality.npy")
        save_npy(l4.closeness_centrality, out / "lens4_closeness_centrality.npy")
        save_npy(l4.eigenvector_centrality, out / "lens4_eigenvector_centrality.npy")
        pd.DataFrame(
            {
                "node_id": np.arange(len(l4.degree_centrality), dtype=np.int32),
                "degree": l4.degree_centrality,
                "betweenness": l4.betweenness_centrality,
                "closeness": l4.closeness_centrality,
                "eigenvector": l4.eigenvector_centrality,
            }
        ).to_csv(out / "centrality_metrics.csv", index=False)
        self._save_community_outputs(l4, out)

        result.degree_comparison.to_csv(out / "degree_comparison.csv", index=False)
        save_json(
            {
                "nodes": result.degree_comparison.to_dict(orient="records"),
                "summary": {
                    "mean_learned_out_degree": result.degree_comparison[
                        "learned_out_degree"
                    ].mean(),
                    "mean_learned_in_degree": result.degree_comparison["learned_in_degree"].mean(),
                    "mean_gcg_out_degree": result.degree_comparison["gcg_out_degree"].mean(),
                    "mean_gcg_in_degree": result.degree_comparison["gcg_in_degree"].mean(),
                    "mean_out_degree_delta": result.degree_comparison["out_degree_delta"].mean(),
                    "mean_in_degree_delta": result.degree_comparison["in_degree_delta"].mean(),
                },
            },
            out / "degree_comparison.json",
        )

        # Lens 5 arrays
        save_npy(l5.mean_sgs_per_horizon, out / "lens5_mean_sgs_per_horizon.npy")
        save_npy(l5.aas_per_horizon, out / "lens5_aas_per_horizon.npy")
        save_json(
            {
                "sgs_rate": l5.sgs_rate,
                "aas_rate": l5.aas_rate,
                "sgs_r2": l5.sgs_r2,
                "aas_r2": l5.aas_r2,
                "horizon_steps": l5.horizon_steps.tolist(),
            },
            out / "degradation_rates.json",
        )

        # Summary JSON
        save_json(result.summary(), out / "summary.json")

        # Full metrics JSON
        full: dict[str, Any] = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "lens1": {
                "mean_sgs": l1.mean_sgs,
                "pct_beneficial": l1.pct_beneficial,
                "pct_harmful": l1.pct_harmful,
                "pct_neutral": l1.pct_neutral,
                "threshold": l1.threshold,
            },
            "lens3_consensus_graph": {
                "precision": l3.precision,
                "recall": l3.recall,
                "aas_consensus_graph": l3.f1,
                "tp": l3.tp,
                "fp": l3.fp,
                "fn": l3.fn,
                "threshold": l3.threshold,
                "weighted_precision": l3.weighted_precision,
            },
            "lens4_consensus_graph": {
                "num_communities": l4.num_communities,
                "modularity": l4.modularity,
                "gcs_overall": l4.gcs_overall,
                "gcs_per_community": {str(k): v for k, v in l4.gcs_per_community.items()},
            },
            "lens5": {
                "sgs_rate": l5.sgs_rate,
                "aas_rate": l5.aas_rate,
                "sgs_r2": l5.sgs_r2,
                "aas_r2": l5.aas_r2,
                "horizon_steps": l5.horizon_steps.tolist(),
            },
        }
        if result.structure_seed_summary is not None:
            full["structure_evaluations"] = result.structure_seed_summary.to_report_dict()
        save_json(full, out / "full_metrics.json")

    def _save_community_outputs(self, result: CommunityResult, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "node_id": np.arange(len(result.community_assignments), dtype=np.int32),
                "community_id": result.community_assignments,
            }
        ).to_csv(out / "community_assignments.csv", index=False)
        save_json(
            {
                "modularity": result.modularity,
                "modularity_null_mean": result.modularity_null_mean,
                "modularity_null_std": result.modularity_null_std,
                "modularity_zscore": result.modularity_zscore,
            },
            out / "modularity_score.json",
        )
        save_json(result.gcs_per_community, out / "gcs_per_community.json")
        save_json({"gcs_overall": result.gcs_overall}, out / "gcs_overall.json")

    def save_figures(
        self,
        result: ProbeResult,
        coords: pd.DataFrame,
        adjacency: np.ndarray | None = None,
        horizon_labels: list[str] | None = None,
    ) -> None:
        """Generate and save all per-model figures to the output directory."""
        from .lens1_spatial_utility import (
            plot_sgs_distribution,
            plot_sgs_horizon,
            plot_spatial_gain_map,
        )
        from .lens3_alignment import plot_threshold_sweep
        from .lens4_community import (
            plot_adjacency_heatmap,
            plot_centrality_by_community,
            plot_centrality_map,
            plot_community_map,
        )
        from .lens5_degradation import (
            plot_aas_degradation,
            plot_combined_dashboard,
            plot_degradation_rate_correlation,
            plot_sgs_degradation,
        )

        out = model_output_dir(self.output_dir, result.model_name, result.dataset_name)
        out.mkdir(parents=True, exist_ok=True)

        _save_figure(plot_sgs_distribution(result.lens1), out / "sgs_distribution.png")
        _save_figure(plot_spatial_gain_map(result.lens1, coords), out / "spatial_gain_map.png")
        _save_figure(plot_sgs_horizon(result.lens1, horizon_labels), out / "lens1_sgs_horizon.png")
        _save_figure(
            plot_threshold_sweep(result.lens3, result.model_name),
            out / "threshold_sweep.png",
        )
        _save_figure(plot_community_map(result.lens4, coords), out / "community_map.png")
        _save_figure(
            plot_centrality_map(result.lens4, coords, "degree"),
            out / "centrality_heatmap.png",
        )
        _save_figure(
            plot_centrality_by_community(result.lens4, "degree"),
            out / "centrality_by_community.png",
        )
        _save_figure(
            plot_degree_comparison(result.degree_comparison),
            out / "degree_comparison.png",
        )
        if adjacency is not None:
            _save_figure(
                plot_adjacency_heatmap(adjacency, result.lens4),
                out / "adjacency_heatmap.png",
            )

        lens5_map = {result.model_name: result.lens5}
        _save_figure(plot_sgs_degradation(lens5_map, horizon_labels), out / "sgs_degradation.png")
        _save_figure(plot_aas_degradation(lens5_map, horizon_labels), out / "aas_degradation.png")
        _save_figure(plot_degradation_rate_correlation(lens5_map), out / "correlation_plot.png")
        _save_figure(
            plot_combined_dashboard(lens5_map, horizon_labels),
            out / "combined_dashboard.png",
        )

    def save_comparative_figures(
        self,
        results: dict[str, ProbeResult],
        horizon_labels: list[str] | None = None,
    ) -> None:
        """Generate and save cross-model comparison figures."""
        from .lens3_alignment import plot_alignment_comparison
        from .lens5_degradation import (
            plot_aas_degradation,
            plot_combined_dashboard,
            plot_degradation_rate_correlation,
            plot_sgs_degradation,
        )

        # Write comparative outputs per dataset so multiple datasets coexist
        # (comparative/metr_la, comparative/pems_bay) instead of overwriting a
        # single shared directory on each run.
        for dataset_name, ds_results in _group_by_dataset(results).items():
            comp_dir = comparative_dir(self.output_dir, dataset_name)
            comp_dir.mkdir(parents=True, exist_ok=True)

            l3_map = {name: r.lens3 for name, r in ds_results.items()}
            l5_map = {name: r.lens5 for name, r in ds_results.items()}

            _save_figure(
                plot_alignment_comparison(l3_map),
                comp_dir / "lens3_alignment_comparison.png",
            )
            _save_figure(
                plot_sgs_degradation(l5_map, horizon_labels),
                comp_dir / "lens5_sgs_degradation.png",
            )
            _save_figure(
                plot_aas_degradation(l5_map, horizon_labels),
                comp_dir / "lens5_aas_degradation.png",
            )
            _save_figure(
                plot_combined_dashboard(l5_map, horizon_labels),
                comp_dir / "lens5_combined_dashboard.png",
            )
            if len(ds_results) >= 3:
                _save_figure(
                    plot_degradation_rate_correlation(l5_map),
                    comp_dir / "lens5_degradation_rate_correlation.png",
                )

            summaries = {name: r.summary() for name, r in ds_results.items()}
            save_json(summaries, comp_dir / "comparative_summary.json")
            save_json(
                compute_cross_model_correlations(summaries),
                comp_dir / "cross_model_correlations.json",
            )

    # ------------------------------------------------------------------
    # Lens 0 — Forecasting Performance Benchmark
    # ------------------------------------------------------------------

    def run_lens0(
        self,
        dataset_name: str,
        ground_truth: np.ndarray,
        predictions: dict[str, np.ndarray],
        horizons: list[int],
        model_groups: dict[str, list[str]],
        baselines: dict[str, str],
        save_figures: bool = True,
    ) -> Lens0Result:
        """Thin wrapper around the standalone ``run_lens0`` function.

        Uses ``self.output_dir`` as the output root.
        """
        return _run_lens0_standalone(
            dataset=dataset_name,
            ground_truth=ground_truth,
            predictions=predictions,
            horizons=horizons,
            model_groups=model_groups,
            baselines=baselines,
            output_dir=self.output_dir,
            primary_metric=self.config.performance.primary_metric,
            horizon_groups=self.config.performance.horizon_groups,
            ranking_lower_is_better=self.config.performance.ranking_lower_is_better,
            save_figures=save_figures,
        )

    def run_performance(self, dataset_name: str, save_figures: bool = True) -> Lens0Result:
        """Load predictions from disk and run Lens 0 for one dataset.

        All model names are taken from ``config.performance.model_groups``.
        Predictions are loaded from ``dataset.predictions_dir``.  Models
        whose prediction file does not exist are skipped with a warning.

        Repository-generated artifacts are always ``(W, H, N, R)`` regardless
        of model.
        """
        dataset = self._dataset_config(dataset_name)
        predictions_dir = self._required_path(dataset.predictions_dir, "predictions_dir")
        gt_path = self._required_path(dataset.ground_truth, "ground_truth")
        ground_truth = np.load(gt_path)

        perf_cfg = self.config.performance
        all_model_names: list[str] = []
        for members in perf_cfg.model_groups.values():
            all_model_names.extend(members)

        predictions: dict[str, np.ndarray] = {}
        for model_name in all_model_names:
            pred_path = predictions_dir / f"{model_name}_predictions.npy"
            if not pred_path.exists():
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "Prediction file not found for model %r; skipping.", model_name
                )
                continue
            predictions[model_name] = np.load(pred_path)

        return _run_lens0_standalone(
            dataset=dataset_name,
            ground_truth=ground_truth,
            predictions=predictions,
            horizons=dataset.horizon_minutes,
            model_groups=perf_cfg.model_groups,
            baselines=perf_cfg.baselines,
            output_dir=self.output_dir,
            primary_metric=perf_cfg.primary_metric,
            horizon_groups=perf_cfg.horizon_groups,
            ranking_lower_is_better=perf_cfg.ranking_lower_is_better,
            save_figures=save_figures,
        )

    def run_all_performance_from_config(self, save_figures: bool = True) -> dict[str, Lens0Result]:
        """Run ``run_performance`` for every dataset in the config.

        Returns
        -------
        dict[str, Lens0Result]
            Mapping from dataset name to its Lens0Result.
        """
        return {
            ds.name: self.run_performance(ds.name, save_figures=save_figures)
            for ds in self.config.datasets
        }
