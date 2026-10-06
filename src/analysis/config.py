"""Configuration dataclasses and YAML loader for STGNN-Probe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GrangerConfig:
    # Fixed lag p = max_lag for every pair (no lag selection); 12 matches the
    # 12-step model input window used in the paper.
    max_lag: int = 12
    significance: float = 0.05
    n_jobs: int = -1
    # The GCG keeps each target node's top-k strongest incoming causal edges by
    # F-statistic (effect size), giving a fixed per-node density of k/(N-1).
    # This replaces p-value significance thresholding, which is
    # non-discriminative here: with T~1e4 samples the Granger F-test flags almost
    # every pair as significant (median p ~ 1e-48), yielding an ~86%-dense GCG.
    gcg_top_k: int = 10


@dataclass
class CommunityConfig:
    algorithm: str = "louvain"
    num_runs: int = 10
    resolution: float = 1.0
    random_seed: int = 0
    # Degree-preserving edge-swap permutations for the modularity null (0 skips).
    null_permutations: int = 20


@dataclass
class AlignmentConfig:
    threshold: float = 0.1
    sweep_min: float = 0.0
    sweep_max: float = 1.0
    sweep_steps: int = 21


@dataclass
class DatasetConfig:
    name: str
    num_nodes: int
    raw_data: str | None = None
    coordinates: str | None = None
    ground_truth: str | None = None
    predictions_dir: str | None = None
    adjacency_dir: str | None = None
    horizons: list[int] = field(default_factory=lambda: [6, 12, 42])
    horizon_minutes: list[int] = field(default_factory=lambda: [30, 60, 210])


@dataclass
class ModelsConfig:
    temporal_baselines: list[str] = field(default_factory=lambda: ["tcn"])
    spatial_models: list[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Configuration for Lens 0 — Forecasting Performance Benchmark.

    MAE, RMSE, and MAPE are always computed and reported together for every
    model (there is no partial-metrics mode), so there is no ``metrics``
    selection field here. ``primary_metric`` drives ranking, and
    ``ranking_lower_is_better`` drives ranking direction.
    """

    primary_metric: str = "mae"
    horizon_groups: dict[str, list[int]] = field(
        default_factory=lambda: {
            "short_range": [30],
            "boundary": [60],
            "long_range": [210],
        }
    )
    baselines: dict[str, str] = field(
        default_factory=lambda: {"statistical": "arima", "temporal": "tcn"}
    )
    model_groups: dict[str, list[str]] = field(
        default_factory=lambda: {
            "statistical_baseline": ["arima"],
            "temporal_baseline": ["tcn"],
            "graph_wavenet_based": ["gwn", "gwn_v2"],
            "attention_adaptive_stgnn": ["stawnet", "dssa_tcn", "staeformer"],
        }
    )
    ranking_lower_is_better: bool = True


@dataclass
class ProbeConfig:
    granger: GrangerConfig = field(default_factory=GrangerConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    datasets: list[DatasetConfig] = field(default_factory=list)
    sgs_threshold: float = 0.1
    # Relative-SGS threshold for node labels (fraction of TCN MAE); scale-free.
    sgs_rel_threshold: float = 0.01
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


def load_config(path: str | Path) -> ProbeConfig:
    """Load a ProbeConfig from a YAML file."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    granger = GrangerConfig(**raw.get("granger", {}))
    community = CommunityConfig(**raw.get("community", {}))
    alignment = AlignmentConfig(**raw.get("alignment", {}))
    models = ModelsConfig(**raw.get("models", {}))
    performance = PerformanceConfig(**raw.get("performance", {}))

    datasets = [DatasetConfig(**ds) for ds in raw.get("datasets", [])]

    return ProbeConfig(
        granger=granger,
        community=community,
        alignment=alignment,
        models=models,
        datasets=datasets,
        sgs_threshold=float(raw.get("sgs_threshold", 0.1)),
        sgs_rel_threshold=float(raw.get("sgs_rel_threshold", 0.01)),
        performance=performance,
    )
