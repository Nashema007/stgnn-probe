"""Save/load utilities for STGNN-Probe results and Granger cache."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .config import DatasetConfig
    from .lens2_granger import GrangerResult


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def model_output_dir(output_dir: Path, model_name: str, dataset_name: str) -> Path:
    return output_dir / "per_model" / f"{model_name}_{dataset_name}"


def granger_cache_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / "granger_cache" / f"{dataset_name}.npz"


def dataset_output_dir(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / "dataset" / dataset_name


def dataset_slug(dataset_name: str) -> str:
    """Filesystem-friendly dataset identifier, e.g. 'METR-LA' -> 'metr_la'."""
    return dataset_name.lower().replace("-", "_")


def comparative_dir(output_dir: Path, dataset_name: str) -> Path:
    """Per-dataset comparative output dir, e.g. outputs/comparative/metr_la."""
    return output_dir / "comparative" / dataset_slug(dataset_name)


def _require_dir(value: str | None, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"Dataset config is missing {field_name}.")
    return Path(value)


def model_predictions_path(dataset: DatasetConfig, model_name: str) -> Path:
    return (
        _require_dir(dataset.predictions_dir, "predictions_dir") / f"{model_name}_predictions.npy"
    )


def temporal_baseline_predictions_path(dataset: DatasetConfig, baseline_name: str = "tcn") -> Path:
    return (
        _require_dir(dataset.predictions_dir, "predictions_dir")
        / f"{baseline_name}_predictions.npy"
    )


def adjacency_path(dataset: DatasetConfig, model_name: str) -> Path:
    return _require_dir(dataset.adjacency_dir, "adjacency_dir") / f"{model_name}_adjacency.npy"


def model_horizon_adjacency_path(dataset: DatasetConfig, model_name: str, horizon: int) -> Path:
    """Path for per-horizon learned adjacency: {adjacency_dir}/{model}_adjacency_h{h}.npy."""
    return (
        _require_dir(dataset.adjacency_dir, "adjacency_dir")
        / f"{model_name}_adjacency_h{horizon}.npy"
    )


def model_horizon_adjacency_seeds_path(
    dataset: DatasetConfig, model_name: str, horizon: int
) -> Path:
    """Path for the per-horizon per-seed stack: {model}_adjacency_h{h}_seeds.npy (R, N, N)."""
    return (
        _require_dir(dataset.adjacency_dir, "adjacency_dir")
        / f"{model_name}_adjacency_h{horizon}_seeds.npy"
    )


# ---------------------------------------------------------------------------
# Generic numpy / JSON save-load
# ---------------------------------------------------------------------------


def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_json(data: Mapping[Any, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(data), f, indent=2, allow_nan=False)


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# ---------------------------------------------------------------------------
# GrangerResult cache
# ---------------------------------------------------------------------------


def save_granger_result(result: GrangerResult, output_dir: Path, dataset_name: str) -> None:
    path = granger_cache_path(output_dir, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gcg_matrix=result.gcg_matrix,
        pvalues=result.pvalues,
        fstats=result.fstats,
        optimal_lags=result.optimal_lags,
        pearson_correlations=result.pearson_correlations,
        bonferroni_threshold=np.array([result.bonferroni_threshold]),
    )


def load_granger_result(output_dir: Path, dataset_name: str, top_k: int = 10) -> GrangerResult:
    from .lens2_granger import GrangerResult, build_gcg_topk

    path = granger_cache_path(output_dir, dataset_name)
    data = np.load(path)
    fstats = data["fstats"]
    # Re-derive the GCG from the cached F-statistics so a changed top_k takes
    # effect without recomputing the (expensive) pairwise Granger tests. The
    # cached gcg_matrix is ignored.
    gcg_matrix = build_gcg_topk(fstats, top_k)
    return GrangerResult(
        gcg_matrix=gcg_matrix,
        pvalues=data["pvalues"],
        fstats=fstats,
        optimal_lags=data["optimal_lags"],
        pearson_correlations=data["pearson_correlations"],
        bonferroni_threshold=float(data["bonferroni_threshold"][0]),
    )


def granger_cache_exists(output_dir: Path, dataset_name: str) -> bool:
    return granger_cache_path(output_dir, dataset_name).exists()
