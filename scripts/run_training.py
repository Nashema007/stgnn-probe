"""Training orchestrator for STGNN-Probe experiments.

Trains each configured model at 7 horizons × 3 random seeds (21 jobs per
model), aggregates per-window predictions into (W, H, N, R) arrays paired
1:1 against a (W, H, N) ground truth, extracts the learned adjacency, and
immediately submits STGNN-Probe analysis in a background thread so the GPU
can start on the next model without waiting.

Usage
-----
# All configured models
python scripts/run_training.py --config scripts/configs/metr_la_training.yaml

# Single model
python scripts/run_training.py --config scripts/configs/metr_la_training.yaml --model gwn

# Just the TCN temporal baseline
python scripts/run_training.py --config scripts/configs/metr_la_training.yaml --model tcn
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


SUPPORTED_SPATIAL_MODELS = {
    "gwn",
    "gwn_v2",
    "stawnet",
    "staeformer",
    "dssa_tcn",
    "d2stgnn",
    "bigst",
}
TEMPORAL_MODEL_ORDER = ("arima", "tcn")
SUPPORTED_TEMPORAL_MODELS = set(TEMPORAL_MODEL_ORDER)
SUPPORTED_MODELS = SUPPORTED_SPATIAL_MODELS | SUPPORTED_TEMPORAL_MODELS

_DEFAULT_SPATIAL_MODELS = ("gwn", "gwn_v2", "stawnet", "staeformer", "dssa_tcn")


def _default_run_order(spatial_models: list[str]) -> list[str]:
    """Return temporal baselines followed by configured spatial models."""
    return [*TEMPORAL_MODEL_ORDER, *spatial_models]


@dataclass
class TrainingExperimentConfig:
    dataset_name: str
    num_nodes: int
    horizons: list[int] = field(default_factory=lambda: [6, 12, 18, 24, 30, 36, 42])
    num_runs: int = 3
    output_dir: str = "outputs/"
    ground_truth_path: str = "data/probe_inputs/ground_truth.npy"
    predictions_dir: str = "data/probe_inputs/predictions/"
    adjacency_dir: str = "data/probe_inputs/adjacency/"
    probe_config_path: str | None = None
    spatial_models: list[str] = field(default_factory=lambda: list(_DEFAULT_SPATIAL_MODELS))
    # Shared trainer fields (not model-specific)
    in_len: int = 12
    in_dim: int = 2
    batch_size: int = 64
    device: str = "cpu"
    epochs: int = 100
    patience: int = 15
    checkpoint_dir: str = "checkpoints/"
    null_val: float = 0.0
    # Temporal split ratios, passed straight to tsl's TemporalSplitter: test is
    # the last `test_len` fraction of the series; val is `val_len` of whatever
    # remains *before* test; train is the rest. So the effective split is
    # (1 - test_len)*(1 - val_len) / (1 - test_len)*val_len / test_len.
    # val_len=0.125, test_len=0.2 is the canonical 70/10/20 default; the
    # production configs also set it explicitly. Both are in the run fingerprint
    # (and test_len in the ground-truth fingerprint), so changing either
    # invalidates stale cached runs instead of replaying them.
    val_len: float = 0.125
    test_len: float = 0.2
    # Directory containing per-model <name>_base.yaml files
    model_configs_dir: str = "scripts/configs/models"
    # Optional inline overrides — applied on top of the per-model YAML (useful for sweeps)
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # --- W&B (optional; off by default) ---
    use_wandb: bool = False
    wandb_entity: str | None = None  # if empty, falls back to WANDB_ENTITY from env/.env
    wandb_project: str = "stgnn-framework"
    wandb_mode: str = "online"  # "offline" for tests / no-network runs


def load_experiment_config(path: str | Path) -> TrainingExperimentConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    model_overrides = raw.pop("model_overrides", {})
    return TrainingExperimentConfig(**raw, model_overrides=model_overrides)


def _normalize_model_filter(model_filter: str | list[str] | None) -> list[str] | None:
    """Accept a single model name, a comma-separated string, a list, or
    ``None`` (run everything configured) — everywhere else in this module
    deals only with ``list[str] | None``.

    Order is preserved (so ``--model gwn,tcn`` runs gwn before tcn) and
    duplicates are dropped, in case the same name is passed twice.
    """
    if model_filter is None:
        return None
    if isinstance(model_filter, str):
        names = [m.strip() for m in model_filter.split(",") if m.strip()]
    else:
        names = [m.strip() for m in model_filter if m.strip()]
    return list(dict.fromkeys(names)) or None


def _validate_model_names(cfg: TrainingExperimentConfig, model_filter: list[str] | None) -> None:
    configured_models = set(cfg.spatial_models) | SUPPORTED_TEMPORAL_MODELS
    unknown_configured = set(cfg.spatial_models) - SUPPORTED_SPATIAL_MODELS
    if unknown_configured:
        raise ValueError(f"Unknown model(s) in spatial_models: {sorted(unknown_configured)}.")
    unknown_overrides = set(cfg.model_overrides) - SUPPORTED_MODELS
    if unknown_overrides:
        raise ValueError(f"Unknown model(s) in model_overrides: {sorted(unknown_overrides)}.")
    if model_filter is not None:
        unknown_filter = [m for m in model_filter if m not in configured_models]
        if unknown_filter:
            raise ValueError(f"Unknown model(s) {unknown_filter!r} for this experiment config.")


def _validate_paths(cfg: TrainingExperimentConfig, spatial_to_run: list[str]) -> None:
    for field_name in ("predictions_dir", "adjacency_dir"):
        path = Path(getattr(cfg, field_name))
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"{field_name} is not a directory: {path}")

    ground_truth_parent = Path(cfg.ground_truth_path).parent
    if ground_truth_parent.exists() and not ground_truth_parent.is_dir():
        raise NotADirectoryError(
            f"ground_truth_path parent is not a directory: {ground_truth_parent}"
        )

    if spatial_to_run:
        if cfg.probe_config_path is None:
            raise ValueError("probe_config_path is required when running spatial models.")
        probe_config_path = Path(cfg.probe_config_path)
        if not probe_config_path.exists():
            raise FileNotFoundError(f"probe_config_path does not exist: {probe_config_path}")


def validate_experiment_config(
    cfg: TrainingExperimentConfig,
    model_filter: str | list[str] | None = None,
) -> None:
    """Validate cheap config failures before expensive training starts.

    ``model_filter`` accepts a single model name, a comma-separated string
    (e.g. ``"gwn,tcn"``), or a list — see ``_normalize_model_filter``.
    """
    model_filter = _normalize_model_filter(model_filter)
    _validate_model_names(cfg, model_filter)
    if cfg.num_runs < 1:
        raise ValueError("num_runs must be at least 1.")

    models_to_run = model_filter if model_filter else _default_run_order(cfg.spatial_models)
    spatial_to_run = [m for m in models_to_run if m in SUPPORTED_SPATIAL_MODELS]

    if (
        any(m in models_to_run for m in ("dssa_tcn", "staeformer", "d2stgnn", "bigst"))
        and cfg.in_dim < 3
    ):
        raise ValueError(
            "dssa_tcn, staeformer, d2stgnn, and bigst require in_dim >= 3 "
            "for value, time-of-day, and day-of-week channels."
        )

    _validate_paths(cfg, spatial_to_run)
    _validate_seeds_log(cfg, models_to_run)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------


def _load_dataloaders(
    cfg: TrainingExperimentConfig,
    out_len: int,
    scaler: Any = None,
) -> tuple[dict, Any, tuple[Tensor, Tensor | None]]:
    """Return (dataloaders, scaler, tsl_graph) for every model, sourced from tsl.

    ``tsl_graph`` is the ``(edge_index, edge_weight)`` pair computed from the
    tsl dataset's own connectivity — the single source of adjacency for every
    model in this framework, spatial or not.

    ``scaler``:
        Pass the canonical scaler (fit once at ``max(cfg.horizons)`` in
        ``run()``) so every per-horizon training pipeline normalises with the
        exact same statistics as the shared evaluation loader every model is
        evaluated on — see ``data.tsl_pipeline.build_tsl_pipeline``'s
        ``scaler`` parameter docstring for why a per-horizon scaler would
        otherwise silently mismatch.
    """
    from data.tsl_pipeline import build_tsl_pipeline

    pipeline = build_tsl_pipeline(
        cfg.dataset_name,
        in_len=cfg.in_len,
        out_len=out_len,
        batch_size=cfg.batch_size,
        scaler=scaler,
        val_len=cfg.val_len,
        test_len=cfg.test_len,
    )
    return pipeline.dataloaders, pipeline.scaler, (pipeline.edge_index, pipeline.edge_weight)


# ---------------------------------------------------------------------------
# Per-model config loader
# ---------------------------------------------------------------------------


def _resolve_model_overrides(
    model_name: str,
    cfg: TrainingExperimentConfig,
) -> dict[str, Any]:
    """Return a flat param dict for model_name by merging the per-model YAML with
    any inline overrides from the experiment config.

    Precedence (low → high):
      1. architecture section of <model_name>_base.yaml
      2. datasets.<dataset_name>.training section of the same file
      3. cfg.model_overrides[model_name] (inline experiment or sweep overrides)
    """
    yaml_path = Path(cfg.model_configs_dir) / f"{model_name}_base.yaml"
    params: dict[str, Any] = {}

    if yaml_path.exists():
        with open(yaml_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        params.update(raw.get("architecture", {}))
        ds_training = raw.get("datasets", {}).get(cfg.dataset_name, {}).get("training", {})
        params.update(ds_training)

    params.update(cfg.model_overrides.get(model_name, {}))
    return params


def _dataset_artifact_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace("-", "_")


def _default_bigst_long_term_feat_path(cfg: TrainingExperimentConfig) -> Path:
    return (
        ROOT
        / "data"
        / "probe_inputs"
        / _dataset_artifact_slug(cfg.dataset_name)
        / "bigst_long_term_features.npy"
    )


def _preprocess_bigst_long_term_features(**kwargs: Any) -> Path:
    import importlib

    try:
        module = importlib.import_module("scripts.preprocess_bigst_features")
    except ModuleNotFoundError:
        module = importlib.import_module("preprocess_bigst_features")

    return module.preprocess_bigst_features(**kwargs)


def _ensure_bigst_long_term_features(
    cfg: TrainingExperimentConfig,
    overrides: dict[str, Any],
    feat_path: Path,
    device: torch.device,
) -> None:
    if feat_path.exists():
        return

    hid_dim = int(overrides.get("hid_dim", 32))
    long_feat_dim = int(overrides.get("long_feat_dim", hid_dim))
    print(
        f"  [bigst] long-term feature file missing; preprocessing {cfg.dataset_name} to {feat_path}"
    )
    _preprocess_bigst_long_term_features(
        dataset=cfg.dataset_name,
        epochs=int(overrides.get("long_term_epochs", 3)),
        input_length=int(overrides.get("long_term_input_length", 288)),
        output_length=int(overrides.get("long_term_output_length", 12)),
        batch_size=int(overrides.get("long_term_batch_size", min(cfg.batch_size, 8))),
        nhid=long_feat_dim,
        dropout=float(overrides.get("long_term_dropout", overrides.get("dropout", 0.3))),
        learning_rate=float(overrides.get("long_term_lr", overrides.get("lr", 0.001))),
        weight_decay=float(
            overrides.get("long_term_weight_decay", overrides.get("weight_decay", 0.0001))
        ),
        grad_clip=float(overrides.get("long_term_grad_clip", 5.0)),
        device=str(device),
        output=feat_path,
    )


# ---------------------------------------------------------------------------
# Cache fingerprinting
#
# Bump this whenever the data source or model implementation for any model
# changes in a way that makes previously cached predictions/adjacency
# incomparable (e.g. the tsl pipeline migration: GWN/TCN moved from local
# .npz files to tsl's auto-downloaded datasets and re-implemented models).
# Cached runs whose fingerprint doesn't match the current one are retrained
# instead of silently reused.
# ---------------------------------------------------------------------------

_CACHE_SCHEMA_VERSION = "tsl-pipeline-v3"
# v2: every per-horizon training pipeline now normalises with the canonical
# scaler (fit once at max(cfg.horizons)) instead of its own out_len-specific
# scaler — runs cached under v1 were trained on different normalisation
# statistics than the canonical eval loader they're evaluated against, so
# they must not be reused.
# v3: fingerprints hash the raw override dict, not the values models actually
# consumed, so a key-name bug in _build_model_and_adapter (dssa_tcn read
# "order" instead of "gcn_order", silently using gcn_order=1 instead of 2;
# gwn_v2/stawnet never threaded kernel_size/blocks/layers/etc through at all)
# left old caches indistinguishable from caches produced by the fix. Bumping
# here forces every model to retrain once under the corrected code.


def _resolved_trainer_fields(model_name: str, cfg: TrainingExperimentConfig) -> dict[str, Any]:
    """Return the effective TrainerConfig-relevant fields for model_name.

    Mirrors exactly what ``_train_one_run`` resolves into ``TrainerConfig``
    (including its hardcoded defaults for fields not in ``cfg`` itself), so
    the fingerprint changes whenever a setting that actually affects
    training would change — not just the per-model YAML/override dict.
    """
    overrides = _resolve_model_overrides(model_name, cfg)
    return {
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "patience": overrides.get("patience", cfg.patience),
        "null_val": cfg.null_val,
        "epochs": overrides.get("epochs", cfg.epochs),
        "lr": overrides.get("lr", 1e-3),
        "weight_decay": overrides.get("weight_decay", 1e-4),
        "grad_clip": overrides.get("grad_clip", 5.0),
        "scheduler": overrides.get("scheduler", "plateau"),
        "scheduler_kwargs": overrides.get("scheduler_kwargs", {}),
    }


def _compute_run_fingerprint(
    model_name: str,
    cfg: TrainingExperimentConfig,
    h: int,
) -> str:
    """Hash everything that determines a run's data/model identity.

    Two runs with the same fingerprint were produced by the same data
    source, preprocessing, resolved architecture config, and resolved
    trainer settings; a mismatch means the cached prediction/adjacency
    files are not comparable and must not be reused.
    """
    import hashlib

    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset_name": cfg.dataset_name,
        "model_name": model_name,
        "horizon": h,
        # Predictions are collected via the shared eval pipeline built at
        # max(cfg.horizons) (see run()), so its test_count — and therefore
        # this cached array's window dimension — depends on the full
        # horizons set, not just `h`. Must be hashed or a horizons change
        # can replay a stale cache whose window count no longer matches.
        "max_horizon": max(cfg.horizons),
        "in_len": cfg.in_len,
        "in_dim": cfg.in_dim,
        # Both split ratios change which rows train/val cover, so a cached
        # prediction from a different split must not be replayed. (test_len
        # also moves the eval windows; val_len only moves the train/val
        # boundary and the early-stopping set — but both still alter the
        # trained weights, hence the predictions.)
        "val_len": cfg.val_len,
        "test_len": cfg.test_len,
        "resolved_params": _resolve_model_overrides(model_name, cfg),
        "trainer": _resolved_trainer_fields(model_name, cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compute_ground_truth_fingerprint(cfg: TrainingExperimentConfig) -> str:
    """Hash everything that determines the ground truth array's identity.

    ``horizons`` is hashed in its configured order (not sorted) because
    ``_build_ground_truth`` builds the ``H`` axis in ``cfg.horizons``'s
    original order — reordering the same set of horizons changes which
    column each one occupies, so it must invalidate this cache too.
    """
    import hashlib

    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset_name": cfg.dataset_name,
        "horizons": cfg.horizons,
        "in_len": cfg.in_len,
        # test_len fixes where the test windows start; val_len does not touch
        # them, so the ground truth only needs to invalidate on test_len.
        "test_len": cfg.test_len,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Model + adapter factory
# ---------------------------------------------------------------------------


def _build_model_and_adapter(
    model_name: str,
    cfg: TrainingExperimentConfig,
    out_len: int,
    tsl_graph: tuple[Tensor, Tensor | None],
) -> tuple[torch.nn.Module, Any]:
    """Return (adapter, trainer_class) for the given model name.

    ``tsl_graph`` (``edge_index``, ``edge_weight``) is the single source of
    adjacency for every model — derived from the tsl dataset's connectivity in
    ``_load_dataloaders``. Models that need a dense ``(N, N)`` matrix (for
    ``build_supports()`` or Chebyshev polynomials) get one converted from it
    here; GWN consumes the sparse pair directly.
    """
    from torch_geometric.utils import to_dense_adj

    from data.adjacency import build_supports
    from models import DSSATCN, GWNv2, STAEformer, STAWnet, make_bigst, make_d2stgnn
    from training.adapters import (
        BigSTAdapter,
        D2STGNNAdapter,
        DSSATCNAdapter,
        GWNAdapter,
        GWNv2Adapter,
        STAEformerAdapter,
        STAWnetAdapter,
        TCNAdapter,
    )
    from training.device import resolve_device
    from training.trainer import GraphTrainer

    device = resolve_device(cfg.device)
    N = cfg.num_nodes
    in_dim = cfg.in_dim
    overrides = _resolve_model_overrides(model_name, cfg)

    if model_name == "tcn":
        from tsl.nn.models.temporal import TCNModel

        num_inputs = overrides.get("num_inputs", 1)
        model = TCNModel(
            input_size=num_inputs,
            output_size=1,
            horizon=out_len,
            hidden_size=overrides.get("hidden_size", 32),
            ff_size=overrides.get("ff_size", 32),
            kernel_size=overrides.get("kernel_size", 2),
            n_layers=overrides.get("n_layers", 4),
            n_convs_layer=overrides.get("n_convs_layer", 2),
            dilation=overrides.get("dilation", 2),
            gated=overrides.get("gated", False),
            resnet=overrides.get("resnet", True),
            norm=overrides.get("norm", "batch"),
            dropout=overrides.get("dropout", 0.2),
        )
        return TCNAdapter(model, num_inputs=num_inputs).to(device), GraphTrainer

    if model_name == "gwn":
        from tsl.nn.models.stgn import GraphWaveNetModel

        edge_index, edge_weight = tsl_graph
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device) if edge_weight is not None else None
        model = GraphWaveNetModel(
            input_size=in_dim,
            output_size=1,
            horizon=out_len,
            n_nodes=N,
            hidden_size=overrides.get("hidden_size", 32),
            ff_size=overrides.get("ff_size", 256),
            n_layers=overrides.get("n_layers", 8),
            temporal_kernel_size=overrides.get("temporal_kernel_size", 2),
            spatial_kernel_size=overrides.get("spatial_kernel_size", 2),
            learned_adjacency=overrides.get("learned_adjacency", True),
            emb_size=overrides.get("emb_size", 10),
            dilation=overrides.get("dilation", 2),
            dilation_mod=overrides.get("dilation_mod", 2),
            norm=overrides.get("norm", "batch"),
            dropout=overrides.get("dropout", 0.3),
        )
        return GWNAdapter(model, edge_index, edge_weight).to(device), GraphTrainer

    # --- Graph models needing a dense (N, N) matrix (doubletransition/laplacian/Chebyshev) ---
    edge_index, edge_weight = tsl_graph
    adj_np = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=N)[0].cpu().numpy()
    adj_type = overrides.get("adj_type", "doubletransition")
    supports_np = build_supports(adj_np, transition_type=adj_type)
    supports = [torch.tensor(s, dtype=torch.float32, device=device) for s in supports_np]

    if model_name == "gwn_v2":
        model = GWNv2(
            device=device,
            num_nodes=N,
            dropout=overrides.get("dropout", 0.3),
            supports=supports,
            do_graph_conv=overrides.get("do_graph_conv", True),
            addaptadj=overrides.get("addaptadj", True),
            in_dim=in_dim,
            out_dim=out_len,
            residual_channels=overrides.get("residual_channels", 32),
            dilation_channels=overrides.get("dilation_channels", 32),
            skip_channels=overrides.get("skip_channels", 256),
            end_channels=overrides.get("end_channels", 512),
            kernel_size=overrides.get("kernel_size", 2),
            blocks=overrides.get("blocks", 4),
            layers=overrides.get("layers", 2),
            cat_feat_gc=overrides.get("cat_feat_gc", False),
            apt_size=overrides.get("apt_size", 10),
        )
        return GWNv2Adapter(model), GraphTrainer

    if model_name == "stawnet":
        model = STAWnet(
            device=device,
            num_nodes=N,
            dropout=overrides.get("dropout", 0.3),
            supports=supports,
            graph_attention=overrides.get("graph_attention", True),
            adaptive_adjacency_matrix=overrides.get("adaptive_adjacency_matrix", True),
            use_node_embedding_only=overrides.get("use_node_embedding_only", False),
            use_node_embedding=overrides.get("use_node_embedding", False),
            in_dim=in_dim,
            out_dim=out_len,
            residual_channels=overrides.get("residual_channels", 32),
            dilation_channels=overrides.get("dilation_channels", 32),
            skip_channels=overrides.get("skip_channels", 256),
            end_channels=overrides.get("end_channels", 512),
            kernel_size=overrides.get("kernel_size", 2),
            blocks=overrides.get("blocks", 4),
            layers=overrides.get("layers", 2),
            emb_length=overrides.get("emb_length", 16),
        )
        return STAWnetAdapter(model), GraphTrainer

    if model_name == "staeformer":
        model = STAEformer(
            num_nodes=N,
            in_steps=cfg.in_len,
            out_steps=out_len,
            steps_per_day=overrides.get("steps_per_day", 288),
            input_dim=overrides.get("input_dim", cfg.in_dim),
            output_dim=1,
            input_embedding_dim=overrides.get("input_embedding_dim", 24),
            tod_embedding_dim=overrides.get("tod_embedding_dim", 24),
            dow_embedding_dim=overrides.get("dow_embedding_dim", 24),
            spatial_embedding_dim=overrides.get("spatial_embedding_dim", 0),
            adaptive_embedding_dim=overrides.get("adaptive_embedding_dim", 80),
            feed_forward_dim=overrides.get("feed_forward_dim", 256),
            num_heads=overrides.get("num_heads", 4),
            num_layers=overrides.get("num_layers", 3),
            dropout=overrides.get("dropout", 0.1),
            use_mixed_proj=overrides.get("use_mixed_proj", True),
        )
        return STAEformerAdapter(model), GraphTrainer

    if model_name == "dssa_tcn":
        # DSSA-TCN requires in_dim >= 3 (value + ToD + DoW)
        if in_dim < 3:
            raise ValueError(
                "dssa_tcn requires in_dim >= 3 for value, time-of-day, and day-of-week."
            )
        gcn_order = overrides.get("gcn_order", 2)
        model = DSSATCN(
            input_dim=in_dim,
            out_dim=out_len,
            num_nodes=N,
            adaptive_embedding_dim=overrides.get("adaptive_embedding_dim", 12),
            residual_channels=overrides.get("residual_channels", 32),
            dilation_channels=overrides.get("dilation_channels", 32),
            skip_channels=overrides.get("skip_channels", 256),
            end_channels=overrides.get("end_channels", 512),
            # blocks/layers set the temporal receptive field (RF = 1 + blocks*(2^layers - 1)).
            # Defaults 4/2 -> RF 13 (covers the in_len=12 main arm); the in_len=42
            # sensitivity arm overrides these to 6/3 -> RF 43 so the model spans the
            # full 42-step window. Without this passthrough the override is silently
            # dropped and DSSA-TCN trains blind to ~29 of the 42 input steps.
            blocks=overrides.get("blocks", 4),
            layers=overrides.get("layers", 2),
            adjs=supports,
            gcn_order=gcn_order,
        )
        return DSSATCNAdapter(model), GraphTrainer

    if model_name == "d2stgnn":
        # D2STGNN requires in_dim >= 3 (value + ToD + DoW); num_feat is every
        # channel besides the trailing ToD/DoW pair it expects appended.
        if in_dim < 3:
            raise ValueError(
                "d2stgnn requires in_dim >= 3 for value, time-of-day, and day-of-week."
            )
        model = make_d2stgnn(
            device,
            num_nodes=N,
            adj_mx=supports_np,
            num_feat=in_dim - 2,
            num_hidden=overrides.get("num_hidden", 32),
            node_hidden=overrides.get("node_hidden", 10),
            time_emb_dim=overrides.get("time_emb_dim", 10),
            dropout=overrides.get("dropout", 0.1),
            seq_length=out_len,
            in_seq_length=cfg.in_len,
            k_t=overrides.get("k_t", 3),
            k_s=overrides.get("k_s", 2),
            gap=overrides.get("gap", 3),
        )
        return D2STGNNAdapter(model).to(device), GraphTrainer

    if model_name == "bigst":
        # BigST requires in_dim >= 3 (value + ToD + DoW).
        if in_dim < 3:
            raise ValueError("bigst requires in_dim >= 3 for value, time-of-day, and day-of-week.")
        long_term_feat = None
        if overrides.get("use_long", False):
            feat_path = Path(
                overrides.get(
                    "long_term_feat_path",
                    _default_bigst_long_term_feat_path(cfg),
                )
            )
            _ensure_bigst_long_term_features(cfg, overrides, feat_path, device)
            long_term_feat = torch.tensor(np.load(feat_path), dtype=torch.float32, device=device)
            long_feat_dim = int(overrides.get("long_feat_dim", overrides.get("hid_dim", 32)))
            if long_term_feat.shape != (N, long_feat_dim):
                raise ValueError(
                    f"bigst long-term feature array at {feat_path} must have shape "
                    f"{(N, long_feat_dim)}, got {tuple(long_term_feat.shape)}."
                )
        model = make_bigst(
            num_nodes=N,
            in_dim=in_dim,
            hid_dim=overrides.get("hid_dim", 32),
            node_dim=overrides.get("node_dim", 32),
            time_dim=overrides.get("time_dim", 32),
            num_layers=overrides.get("num_layers", 3),
            random_feature_dim=overrides.get("random_feature_dim", 64),
            input_length=cfg.in_len,
            output_length=out_len,
            tau=overrides.get("tau", 1.0),
            dropout=overrides.get("dropout", 0.3),
            use_residual=overrides.get("use_residual", True),
            use_bn=overrides.get("use_bn", True),
            use_spatial=False,  # spatial_loss isn't wired into the trainer's loss yet
            use_long=overrides.get("use_long", False),
            long_feat_dim=overrides.get("long_feat_dim", overrides.get("hid_dim", 32)),
            supports=supports,
        ).to(device)
        return BigSTAdapter(model, long_term_feat=long_term_feat).to(device), GraphTrainer

    raise ValueError(f"Unknown model name {model_name!r}.")


# ---------------------------------------------------------------------------
# Seed log (persist to disk for crash recovery)
# ---------------------------------------------------------------------------


def _load_seeds_log(output_dir: Path) -> dict[str, dict[str, list[int]]]:
    path = output_dir / "seeds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_seeds_log(seeds_log: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "seeds.json", "w") as f:
        json.dump(seeds_log, f, indent=2)


def _validate_recorded_seeds(
    seeds_path: Path,
    model_name: str,
    horizon: int,
    existing: list[int],
    num_runs: int,
) -> None:
    """Reject a saved seed list that cannot be reconciled to ``num_runs`` entries.

    Short lists are extendable and therefore fine; duplicate or surplus seeds
    are not, because either would discard the identity of an independent run
    whose artifacts are already on disk.
    """
    if len(existing) != len(set(existing)):
        raise ValueError(
            f"{seeds_path} contains duplicate seeds for "
            f"{model_name} h={horizon}; refusing to overwrite an independent run."
        )
    if len(existing) > num_runs:
        raise ValueError(
            f"{seeds_path} already contains {len(existing)} runs for "
            f"{model_name} h={horizon}, but num_runs={num_runs}; increase num_runs to at least "
            f"{len(existing)} so existing independent-run identities are not discarded."
        )


def _validate_seeds_log(
    cfg: TrainingExperimentConfig,
    models_to_run: list[str],
) -> None:
    """Check every in-scope model's saved seeds before any training starts.

    ``_collect_predictions_and_adj`` re-checks the model it is about to train,
    but it runs per model and ARIMA is refit first — so a ``num_runs`` too small
    for the recorded seeds would otherwise surface only after a full ARIMA fit
    and any earlier models had already been trained.
    """
    output_dir = Path(cfg.output_dir)
    seeds_path = output_dir / "seeds.json"
    seeds_log = _load_seeds_log(output_dir)
    for model_name in models_to_run:
        if model_name == "arima":
            continue  # deterministic — never allocated a seed
        for horizon in cfg.horizons:
            existing = seeds_log.get(model_name, {}).get(str(horizon), [])
            _validate_recorded_seeds(seeds_path, model_name, horizon, existing, cfg.num_runs)


# ---------------------------------------------------------------------------
# Completed-run cache (skip already-finished horizon × seed jobs on resume)
# ---------------------------------------------------------------------------


def _load_completed_runs(output_dir: Path) -> dict:
    path = output_dir / "completed_runs.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_completed_runs(log: dict, output_dir: Path) -> None:
    with open(output_dir / "completed_runs.json", "w") as f:
        json.dump(log, f, indent=2)


def _run_artifact_dir(
    cfg: TrainingExperimentConfig,
    model_name: str,
    horizon: int,
    run_index: int,
    seed: int,
) -> Path:
    """Return the isolated disk directory for one independent seeded run."""
    return (
        Path(cfg.output_dir)
        / "runs"
        / model_name
        / f"h{horizon}"
        / f"run_{run_index + 1:02d}_seed_{seed}"
    )


def _save_run_manifest(
    run_dir: Path,
    *,
    model_name: str,
    cfg: TrainingExperimentConfig,
    horizon: int,
    run_index: int,
    seed: int,
    fingerprint: str,
    val_loss: float,
    predictions_path: Path,
    adjacency_path: Path | None,
    cached: bool,
) -> None:
    """Persist enough identity/config data to audit or reload one run alone."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": model_name,
        "dataset": cfg.dataset_name,
        "horizon": horizon,
        "run_index": run_index + 1,
        "seed": seed,
        "fingerprint": fingerprint,
        "cached": cached,
        "val_loss": val_loss,
        "predictions_path": str(predictions_path),
        "adjacency_path": str(adjacency_path) if adjacency_path is not None else None,
        "ground_truth_path": cfg.ground_truth_path,
        "checkpoint_dir": cfg.checkpoint_dir,
        "resolved_model_config": _resolve_model_overrides(model_name, cfg),
        "resolved_trainer_config": {
            **_resolved_trainer_fields(model_name, cfg),
            "seed": seed,
            "run_index": run_index + 1,
        },
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(payload, f, indent=2)


def _load_cached_adj(run_entry: dict) -> np.ndarray | None:
    adj_path = run_entry.get("adj_path")
    if adj_path and Path(adj_path).exists():
        return np.load(adj_path)
    return None


def _cached_adj_is_missing(run_entry: dict) -> bool:
    """True when the cache names an adjacency file that is no longer on disk.

    A null/absent ``adj_path`` is legitimate — TCN is graph-free and adjacency
    extraction is unavailable for some spatial models — so only a recorded path
    that has since been deleted counts as missing. Distinguishing the two keeps
    a vanished adjacency from being silently downgraded to "this model never had
    one", which would republish the model's global adjacency as a mean over
    fewer horizons than configured.
    """
    adj_path = run_entry.get("adj_path")
    if not adj_path:
        return False
    return not Path(adj_path).exists()


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------


def _try_extract_adj(
    adapter: Any,
    model_name: str,
    dataloaders: dict,
    device: torch.device,
) -> np.ndarray | None:
    """Extract adjacency from adapter, returning None on failure."""
    if not hasattr(adapter, "get_adjacency"):
        return None
    try:
        needs_loader = model_name in ("stawnet", "dssa_tcn", "staeformer")
        return adapter.get_adjacency(
            loader=dataloaders["test"] if needs_loader else None,
            device=device,
        )
    except Exception as exc:
        print(f"  [{model_name}] adjacency extraction failed: {exc}")
        return None


def _stack_node_axis(t: Tensor, node_dim: int) -> Tensor:
    """Move ``node_dim`` to the front and reinstate it as a size-1 axis.

    ``(B, C, N, T)`` with ``node_dim=2`` -> ``(N, B, C, 1, T)``: the same
    per-node ``(*, 1, *)`` shape ``_NodeSlicedLoader`` used to feed one node
    at a time, just stacked across every node along a new leading axis so
    all N independent per-node models can be run in a single vmapped call.
    """
    moved = t.movedim(node_dim, 0)
    return moved.unsqueeze(node_dim + 1)


def _inverse_transform_stacked(data: Tensor, scaler: Any) -> Tensor:
    """Denormalize a node-stacked tensor shaped ``(N, ...)`` using each node's
    own ``scaler`` statistics (dim 0 indexes the same node ordering the
    scaler was fit on) — the vectorized equivalent of calling
    ``scaler.inverse_transform(data, node_indices=[n])`` once per node.
    """
    mean = torch.as_tensor(scaler.mean, dtype=data.dtype, device=data.device)
    std = torch.as_tensor(scaler.std, dtype=data.dtype, device=data.device)
    shape = [data.shape[0]] + [1] * (data.dim() - 1)
    return data * std.view(*shape) + mean.view(*shape)


def _masked_mae_per_node(pred: Tensor, true: Tensor, null_val: float) -> Tensor:
    """Like ``evaluation.metrics.masked_mae`` but reduces over every dim
    except dim 0 (the node axis), returning ``(N,)`` instead of a scalar —
    each node's loss must stay separable so ``.sum().backward()`` still
    routes gradients only to that node's own stacked parameter slice.
    """
    import math

    if math.isnan(null_val):
        mask = ~torch.isnan(true)
    else:
        mask = true.abs() > 1e-4 if null_val == 0.0 else true != null_val
    mask = mask.float()
    err = (pred - true).abs() * mask
    dims = tuple(range(1, pred.dim()))
    return err.sum(dim=dims) / mask.sum(dim=dims).clamp(min=1e-8)


def _curriculum_task_level(epoch: int, warmup_epochs: int, out_len: int) -> int:
    if warmup_epochs <= 0:
        return out_len
    progress = min(epoch / warmup_epochs, 1.0)
    return max(1, int(round(progress * out_len)))


def _tcn_resume_ckpt_path(cfg: TrainingExperimentConfig, h: int, seed: int) -> Path:
    return Path(cfg.checkpoint_dir) / f"tcn_h{h}_s{seed}_ensemble_resume.pt"


def _save_tcn_resume_checkpoint(
    path: Path,
    epoch: int,
    state: Any,
    adam: Any,
    scheduler: Any,
    stopper: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "params": state.params,
        "buffers": state.buffers,
        "adam": adam.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stopper": stopper.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["rng_cuda"] = torch.cuda.get_rng_state_all()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.replace(path)


def _load_tcn_resume_checkpoint(
    path: Path,
    state: Any,
    adam: Any,
    scheduler: Any,
    stopper: Any,
) -> int:
    """Load a resume checkpoint if available; return the epoch to start from.

    A resume checkpoint is a best-effort optimisation (skip re-training
    epochs already done), not a correctness requirement — a fresh start
    from epoch 1 is always safe. A corrupt/incompatible file (e.g. written
    by an older code version, or truncated by a crash mid-write) must
    never abort the run; it's deleted and training starts over instead —
    mirrors ``GraphTrainer._load_resume_checkpoint``'s contract.
    """
    if not path.exists():
        return 1
    try:
        device = state.params[next(iter(state.params))].device
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Validate shapes against the freshly-built ensemble *before*
        # mutating anything — a checkpoint from a run with a different
        # num_nodes or architecture override (same horizon/seed, since the
        # path is keyed only on those) must be treated as incompatible here,
        # not allowed to silently corrupt state.params/buffers and surface
        # as a confusing shape-mismatch deep inside vmap/clip/where later.
        for key, current in (*state.params.items(), *state.buffers.items()):
            ckpt_dict = ckpt["params"] if key in ckpt["params"] else ckpt["buffers"]
            if key not in ckpt_dict:
                raise ValueError(f"checkpoint is missing key {key!r}")
            if ckpt_dict[key].shape != current.shape:
                raise ValueError(
                    f"checkpoint key {key!r} has shape {ckpt_dict[key].shape}, "
                    f"expected {current.shape} (num_nodes or architecture changed?)"
                )

        # Mutate the existing dicts in place rather than rebinding
        # state.params/state.buffers to new dict objects — ``adam`` holds a
        # reference to the *same* dict passed into its constructor, so
        # rebinding state.params here would leave adam.step() updating a
        # now-orphaned dict while ensemble_forward reads the loaded one.
        state.params.clear()
        state.params.update(ckpt["params"])
        state.buffers.clear()
        state.buffers.update(ckpt["buffers"])
        adam.load_state_dict(ckpt["adam"])
        scheduler.load_state_dict(ckpt["scheduler"])
        stopper.load_state_dict(ckpt["stopper"])
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch"])
        if torch.cuda.is_available() and "rng_cuda" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    except Exception as exc:
        print(
            f"[tcn] Resume checkpoint at {path} is corrupt/incompatible ({exc!r}) — "
            "deleting it and starting this run from epoch 1."
        )
        path.unlink(missing_ok=True)
        return 1
    start_epoch = ckpt["epoch"] + 1
    print(f"[tcn] Resuming ensemble training from epoch {start_epoch}.")
    return start_epoch


def _train_tcn_pernode(
    cfg: TrainingExperimentConfig,
    h: int,
    seed: int,
    r_idx: int,
    eval_test_loader: Any,
    canonical_scaler: Any,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Train ``cfg.num_nodes`` independent TCNs — one per node, no shared weights.

    Mirrors ARIMA's per-node design (every node gets its own independently
    fit model, no cross-node parameter sharing) but with a neural model
    instead of a statistical one.

    All N node-models train as one vectorized computation via
    ``torch.func.vmap`` (see ``training.vmap_ensemble``) instead of a
    Python loop over N sequential ``GraphTrainer.train()`` calls — each
    node's model/batch alone is too small to keep a GPU busy, so the loop
    left the GPU mostly idle between 207 tiny kernel launches per epoch
    while a single CPU process burned every core on per-node Python/tensor
    bookkeeping. The per-step update mechanism matches the loop's to
    float32 precision (forward, BatchNorm running stats, per-node gradient
    clipping, Adam, and per-node LR scheduling were each checked against
    PyTorch's own implementations — see ``training.vmap_ensemble``'s module
    docstring for what that comparison covers and doesn't) — just scheduled
    as one fused forward/backward per batch instead of N tiny ones.

    One W&B run is still made per (horizon, seed) rather than per node —
    207 nodes x 3 horizons would mean 621 tiny runs for METR-LA — logging a
    per-node table (val_loss, test MAE/MAPE/RMSE) so individual node
    performance is interrogable, plus the mean/max val_loss as the run
    summary.

    Per-node test MAE/MAPE/RMSE (at horizon ``h``, against the same
    canonical test windows every other model is scored against) are also
    written to ``{cfg.output_dir}/tcn_h{h}_s{seed}_node_metrics.json`` —
    ``run_preds``/``ground_truth.npy`` already let you recompute these
    yourself, but this saves doing that by hand.

    Returns
    -------
    run_preds : (W, N) per shared canonical test window, at horizon h
    val_loss  : mean validation loss across all N node-models
    run_adj   : always None — TCN is graph-free regardless of per-node fitting
    """
    from evaluation.metrics import masked_mae, masked_mape, masked_rmse
    from training.config import TrainerConfig
    from training.device import resolve_device
    from training.logger import make_logger
    from training.resource_monitor import ResourceSampler, gpu_memory_metrics
    from training.trainer import _seed_everything
    from training.vmap_ensemble import (
        PerNodeAdam,
        PerNodeEarlyStopping,
        PerNodeEpochScheduler,
        PerNodePlateau,
        build_ensemble,
        clip_grad_norm_per_node_,
        ensemble_forward,
    )

    R = cfg.num_runs
    N = cfg.num_nodes
    overrides = _resolve_model_overrides("tcn", cfg)
    device = resolve_device(cfg.device)

    scheduler_name = overrides.get("scheduler", "plateau")
    if scheduler_name not in ("plateau", "multistep", "exponential", "none"):
        raise ValueError(
            f"_train_tcn_pernode's vmap ensemble does not recognize scheduler="
            f"{scheduler_name!r}; expected 'plateau', 'multistep', 'exponential', or 'none'."
        )

    epochs = overrides.get("epochs", cfg.epochs)
    patience = overrides.get("patience", cfg.patience)
    lr = overrides.get("lr", 1e-3)
    weight_decay = overrides.get("weight_decay", 1e-4)
    grad_clip = overrides.get("grad_clip", 5.0)
    warmup_epochs = overrides.get("warmup_epochs", 0)
    raw_sched_kwargs = overrides.get("scheduler_kwargs", {})

    dataloaders, scaler, tsl_graph = _load_dataloaders(cfg, out_len=h, scaler=canonical_scaler)
    ckpt_name = f"tcn_h{h}_s{seed}"
    run_dir = _run_artifact_dir(cfg, "tcn", h, r_idx, seed)
    trainer_config = TrainerConfig(
        in_len=cfg.in_len,
        out_len=h,
        batch_size=cfg.batch_size,
        device=cfg.device,
        epochs=epochs,
        patience=patience,
        seed=seed,
        checkpoint_dir=cfg.checkpoint_dir,
        null_val=cfg.null_val,
        model_name=ckpt_name,
        dataset_name=cfg.dataset_name,
        horizon=h,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        scheduler=scheduler_name,
        scheduler_kwargs=raw_sched_kwargs,
        model_kwargs=overrides,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
        wandb_run_name=ckpt_name,
        wandb_mode=cfg.wandb_mode,
        metrics_dir=str(run_dir),
        run_index=r_idx + 1,
    )
    logger = make_logger(trainer_config, run_name=ckpt_name)
    logger.define_metric("node/*", step_metric="node_idx")
    run_sampler = ResourceSampler()
    epoch_sampler = ResourceSampler()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    run_t0 = time.time()

    try:
        _seed_everything(seed)

        def _model_factory() -> torch.nn.Module:
            adapter, _ = _build_model_and_adapter("tcn", cfg, out_len=h, tsl_graph=tsl_graph)
            return adapter

        state = build_ensemble(_model_factory, N, device)
        n_params = sum(value.numel() for value in state.params.values())
        adam = PerNodeAdam(state.params, N, lr=lr, weight_decay=weight_decay)
        if scheduler_name == "plateau":
            sched_kwargs = {**{"factor": 0.3, "patience": 10}, **raw_sched_kwargs}
            scheduler: PerNodePlateau | PerNodeEpochScheduler = PerNodePlateau(
                adam, N, **sched_kwargs
            )
        else:
            sched_kwargs = {
                **(
                    {"milestones": [50, 70, 100], "gamma": 0.1}
                    if scheduler_name == "multistep"
                    else {}
                ),
                **({"decay_rate": 0.97} if scheduler_name == "exponential" else {}),
                **raw_sched_kwargs,
            }
            scheduler = PerNodeEpochScheduler(adam, mode=scheduler_name, base_lr=lr, **sched_kwargs)
        stopper = PerNodeEarlyStopping(N, patience=patience, device=device)

        resume_path = _tcn_resume_ckpt_path(cfg, h, seed)
        start_epoch = _load_tcn_resume_checkpoint(resume_path, state, adam, scheduler, stopper)

        for epoch in range(start_epoch, epochs + 1):
            task_level = _curriculum_task_level(epoch, warmup_epochs, h)

            t0 = time.time()
            state.base.train()
            train_loss_sum = 0.0
            n_batches = 0
            for x, y, _y_full in dataloaders["train"]:
                x, y = x.to(device), y.to(device)
                x_stack = _stack_node_axis(x, node_dim=2)  # (N, B, C, 1, T)
                y_stack = _stack_node_axis(y, node_dim=1)  # (N, B, 1, H)

                adam.zero_grad()
                pred = ensemble_forward(state, x_stack, task_level=task_level)  # (N, B, 1, tl)
                y_tl = y_stack[..., :task_level]

                pred_real = _inverse_transform_stacked(pred, scaler)
                y_real = _inverse_transform_stacked(y_tl, scaler)

                loss_per_node = _masked_mae_per_node(pred_real, y_real, cfg.null_val)
                loss_per_node.sum().backward()
                grad_norms = clip_grad_norm_per_node_(state.params, grad_clip, N)
                nonfinite = (~torch.isfinite(grad_norms)).nonzero(as_tuple=True)[0]
                if nonfinite.numel() > 0:
                    print(
                        f"  [tcn] h={h:2d}  seed={seed}  WARNING: non-finite gradient norm on "
                        f"node(s) {nonfinite.tolist()} — zeroing their gradient this batch.",
                        flush=True,
                    )
                adam.step()

                train_loss_sum += loss_per_node.detach().mean().item()
                n_batches += 1
            train_loss = train_loss_sum / max(n_batches, 1)

            state.base.eval()
            val_pred_list = []
            val_true_list = []
            with torch.no_grad():
                for x, y, _y_full in dataloaders["val"]:
                    x, y = x.to(device), y.to(device)
                    x_stack = _stack_node_axis(x, node_dim=2)
                    y_stack = _stack_node_axis(y, node_dim=1)
                    pred = ensemble_forward(state, x_stack, task_level=h)
                    val_pred_list.append(_inverse_transform_stacked(pred, scaler))
                    val_true_list.append(_inverse_transform_stacked(y_stack, scaler))
            val_pred = torch.cat(val_pred_list, dim=1)  # (N, W_val, 1, H)
            val_true = torch.cat(val_true_list, dim=1)
            val_loss_per_node = _masked_mae_per_node(val_pred, val_true, cfg.null_val)

            if isinstance(scheduler, PerNodePlateau):
                scheduler.step(val_loss_per_node)
            else:
                scheduler.step(epoch)
            stopper.update(val_loss_per_node, state.params, state.buffers)

            elapsed = time.time() - t0
            mem_mb, cpu_pct = epoch_sampler.sample()
            gpu_metrics = gpu_memory_metrics(device)
            val_loss_mean = val_loss_per_node.mean().item()
            val_loss_max = val_loss_per_node.max().item()
            print(
                f"  [tcn] h={h:2d}  seed={seed}  run {r_idx + 1}/{R}  epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f}  val_loss_mean={val_loss_mean:.4f}  "
                f"val_loss_max={val_loss_max:.4f}  "
                f"nodes_done={int((stopper.num_bad_epochs >= patience).sum().item())}/{N}  "
                f"[{elapsed:.1f}s  mem={mem_mb:.0f}MB  cpu={cpu_pct:.0f}%]",
                flush=True,
            )
            logger.log_epoch("train", {"loss": train_loss}, epoch)
            logger.log_epoch(
                "val",
                {"loss_mean": val_loss_mean, "loss_max": val_loss_max},
                epoch,
            )
            logger.log_epoch(
                "system",
                {
                    "duration_s": elapsed,
                    "mem_rss_mb": mem_mb,
                    "cpu_percent": cpu_pct,
                    **gpu_metrics,
                },
                epoch,
            )

            if stopper.all_done:
                print(
                    f"  [tcn] h={h:2d}  seed={seed}  all {N} nodes early-stopped at epoch {epoch}."
                )
                resume_path.unlink(missing_ok=True)
                break

            _save_tcn_resume_checkpoint(resume_path, epoch, state, adam, scheduler, stopper)
        else:
            resume_path.unlink(missing_ok=True)

        assert stopper.best_params is not None and stopper.best_buffers is not None
        # Mutate in place rather than rebind — state.params/buffers are the
        # same dict objects adam (and any future code touching state after
        # this point) holds a reference to; rebinding would silently orphan
        # that reference, the exact footgun _load_tcn_resume_checkpoint's
        # in-place-mutation contract exists to avoid.
        state.params.clear()
        state.params.update(stopper.best_params)
        state.buffers.clear()
        state.buffers.update(stopper.best_buffers)
        val_losses = stopper.best_val_loss.cpu().numpy()
        mean_val_loss = float(val_losses.mean())

        best_ckpt_path = Path(cfg.checkpoint_dir) / f"tcn_h{h}_s{seed}_ensemble_best.pt"
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"params": state.params, "buffers": state.buffers, "val_loss": val_losses},
            best_ckpt_path,
        )
        print(f"  [tcn] h={h:2d}  seed={seed}  Saved best ensemble weights → {best_ckpt_path}")

        train_elapsed = time.time() - run_t0
        run_mem_mb, run_cpu_pct = run_sampler.sample()
        run_gpu_metrics = gpu_memory_metrics(device)
        _log_resource_usage(
            cfg.output_dir,
            {
                "model": "tcn",
                "dataset": cfg.dataset_name,
                "horizon": h,
                "seed": seed,
                "params": n_params,
                "peak_gpu_alloc_mb": round(run_gpu_metrics["gpu_peak_allocated_mb"], 1),
                "peak_gpu_reserved_mb": round(run_gpu_metrics["gpu_peak_reserved_mb"], 1),
                "train_time_s": round(train_elapsed, 1),
                "cpu_rss_mb": round(run_mem_mb, 1),
                "cpu_percent": round(run_cpu_pct, 1),
            },
        )

        state.base.eval()
        pred_chunks = []
        true_chunks = []
        with torch.no_grad():
            for x, _y, y_full in eval_test_loader:
                x = x.to(device)
                y_full = y_full.to(device)
                y_full_h = y_full[..., :h]
                y_h = y_full_h[:, 0]  # (B, N, h) — channel 0
                x_stack = _stack_node_axis(x, node_dim=2)
                pred = ensemble_forward(state, x_stack, task_level=h)  # (N, B, 1, h)
                pred_last = pred[..., -1:]
                true_last = _stack_node_axis(y_h, node_dim=1)[..., -1:]  # (N, B, 1, 1)
                pred_chunks.append(_inverse_transform_stacked(pred_last, scaler))
                true_chunks.append(_inverse_transform_stacked(true_last, scaler))
        pred_full = torch.cat(pred_chunks, dim=1).squeeze(2).squeeze(-1)  # (N, W)
        true_full = torch.cat(true_chunks, dim=1).squeeze(2).squeeze(-1)  # (N, W)

        run_preds = pred_full.cpu().numpy().T  # (W, N)
        run_trues = true_full.cpu().numpy().T  # (W, N)

        node_mae = np.zeros(N)
        node_mape = np.zeros(N)
        node_rmse = np.zeros(N)
        for n in range(N):
            pred_t = torch.from_numpy(run_preds[:, n]).float()[:, None, None]
            true_t = torch.from_numpy(run_trues[:, n]).float()[:, None, None]
            node_mae[n] = masked_mae(pred_t, true_t, cfg.null_val).item()
            node_mape[n] = masked_mape(pred_t, true_t, cfg.null_val).item()
            node_rmse[n] = masked_rmse(pred_t, true_t, cfg.null_val).item()
            logger.log_raw(
                {
                    "node_idx": n,
                    "node/val_loss": float(val_losses[n]),
                    "node/mae": node_mae[n],
                    "node/mape": node_mape[n],
                    "node/rmse": node_rmse[n],
                }
            )

        metrics_path = run_dir / "node_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "horizon": h,
                    "seed": seed,
                    "node_val_loss": val_losses.tolist(),
                    "node_mae": node_mae.tolist(),
                    "node_mape": node_mape.tolist(),
                    "node_rmse": node_rmse.tolist(),
                },
                f,
                indent=2,
            )
        print(f"  [tcn] h={h:2d}  seed={seed}  Saved per-node metrics → {metrics_path}")

        test_mae = float(node_mae.mean())
        test_mape = float(node_mape.mean())
        test_rmse = float(node_rmse.mean())
        logger.log_test_metrics(test_mae, test_mape, test_rmse, horizon=h)
        logger.log_table(
            "node_metrics_table",
            ["node", "val_loss", "mae", "mape", "rmse"],
            [
                [
                    n,
                    float(val_losses[n]),
                    float(node_mae[n]),
                    float(node_mape[n]),
                    float(node_rmse[n]),
                ]
                for n in range(N)
            ],
        )

        logger.log_summary(
            {
                "val_loss_mean": mean_val_loss,
                "val_loss_max": float(val_losses.max()),
                "mae_mean": test_mae,
                "mape_mean": test_mape,
                "rmse_mean": test_rmse,
                "test_mae": test_mae,
                "test_mape": test_mape,
                "test_rmse": test_rmse,
                "params": n_params,
                "train_time_s": train_elapsed,
                "mem_rss_mb": run_mem_mb,
                "cpu_percent": run_cpu_pct,
                **run_gpu_metrics,
            }
        )
    finally:
        logger.finish()

    return run_preds, mean_val_loss, None


_RESOURCE_CSV_HEADER = [
    "model",
    "dataset",
    "horizon",
    "seed",
    "params",
    "peak_gpu_alloc_mb",
    "peak_gpu_reserved_mb",
    "train_time_s",
    "cpu_rss_mb",
    "cpu_percent",
]


def _log_resource_usage(output_dir: str | Path, row: dict[str, Any]) -> None:
    """Append one (model, horizon, seed) row to ``<output_dir>/resource_usage.csv``.

    Written per training run so a per-model table is a later ``groupby`` —
    aggregate over seeds (mean±std) and/or horizons (max GPU, sum/mean time)
    however the paper needs. GPU columns are 0.0 when training on CPU/MPS.
    """
    path = Path(output_dir) / "resource_usage.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESOURCE_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _train_one_run(
    model_name: str,
    cfg: TrainingExperimentConfig,
    h: int,
    seed: int,
    r_idx: int,
    eval_test_loader: Any,
    canonical_scaler: Any,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Train a single (horizon, seed) run.

    Test predictions are collected via ``eval_test_loader`` — the canonical
    loader shared across every model/horizon/seed — rather than this run's
    own ``out_len=h`` test split, so every model is paired 1:1 against the
    same test windows used in ``ground_truth.npy``.

    ``canonical_scaler`` (fit once at ``max(cfg.horizons)``, see ``run()``)
    is used to train this run too — not a fresh scaler fit at ``out_len=h`` —
    so this model's training-time normalisation exactly matches the
    normalisation already baked into ``eval_test_loader``'s batches. Without
    this, the model would be trained on one set of statistics and evaluated
    on inputs normalised with another, silently corrupting predictions.

    ``model_name == "tcn"`` is dispatched to ``_train_tcn_pernode`` instead.

    Returns
    -------
    run_preds : (W, N) per shared canonical test window, at horizon h
    val_loss  : float
    run_adj   : (N, N) learned adjacency, or None
    """
    if model_name == "tcn":
        return _train_tcn_pernode(cfg, h, seed, r_idx, eval_test_loader, canonical_scaler)

    from training.config import TrainerConfig
    from training.logger import make_logger

    R = cfg.num_runs
    ckpt_name = f"{model_name}_h{h}_s{seed}"
    run_dir = _run_artifact_dir(cfg, model_name, h, r_idx, seed)
    overrides = _resolve_model_overrides(model_name, cfg)
    trainer_config = TrainerConfig(
        in_len=cfg.in_len,
        out_len=h,
        batch_size=cfg.batch_size,
        device=cfg.device,
        epochs=overrides.get("epochs", cfg.epochs),
        patience=overrides.get("patience", cfg.patience),
        seed=seed,
        checkpoint_dir=cfg.checkpoint_dir,
        null_val=cfg.null_val,
        model_name=ckpt_name,
        dataset_name=cfg.dataset_name,
        horizon=h,
        lr=overrides.get("lr", 1e-3),
        weight_decay=overrides.get("weight_decay", 1e-4),
        grad_clip=overrides.get("grad_clip", 5.0),
        scheduler=overrides.get("scheduler", "plateau"),
        scheduler_kwargs=overrides.get("scheduler_kwargs", {}),
        model_kwargs=overrides,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
        wandb_run_name=ckpt_name,
        wandb_mode=cfg.wandb_mode,
        metrics_dir=str(run_dir),
        run_index=r_idx + 1,
    )

    dataloaders, scaler, tsl_graph = _load_dataloaders(cfg, out_len=h, scaler=canonical_scaler)

    adapter, trainer_cls = _build_model_and_adapter(model_name, cfg, out_len=h, tsl_graph=tsl_graph)
    logger = make_logger(trainer_config, run_name=ckpt_name)
    trainer = trainer_cls(
        model=adapter,
        config=trainer_config,
        dataloaders=dataloaders,
        scaler=scaler,
        logger=logger,
    )

    from training.resource_monitor import ResourceSampler

    params_fn = getattr(adapter, "parameters", None)
    n_params = sum(p.numel() for p in params_fn()) if callable(params_fn) else 0
    use_cuda = torch.cuda.is_available() and trainer.device.type == "cuda"

    try:
        run_sampler = ResourceSampler()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(trainer.device)
        t0 = time.time()
        train_result = trainer.train()
        elapsed = time.time() - t0
        if use_cuda:
            peak_gpu_alloc_mb = torch.cuda.max_memory_allocated(trainer.device) / 1024**2
            peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(trainer.device) / 1024**2
        else:
            peak_gpu_alloc_mb = peak_gpu_reserved_mb = 0.0
        mem_mb, cpu_pct = run_sampler.sample()
        _log_resource_usage(
            cfg.output_dir,
            {
                "model": model_name,
                "dataset": cfg.dataset_name,
                "horizon": h,
                "seed": seed,
                "params": n_params,
                "peak_gpu_alloc_mb": round(peak_gpu_alloc_mb, 1),
                "peak_gpu_reserved_mb": round(peak_gpu_reserved_mb, 1),
                "train_time_s": round(elapsed, 1),
                "cpu_rss_mb": round(mem_mb, 1),
                "cpu_percent": round(cpu_pct, 1),
            },
        )
        val_loss = train_result.get("val_loss", float("inf"))
        print(
            f"  [{model_name}] h={h:2d}  seed={seed}  run {r_idx + 1}/{R}"
            f"  val_loss={val_loss:.4f}  trained in {elapsed:.0f}s"
            f"  {n_params:,} params  gpu_peak={peak_gpu_reserved_mb:.0f}MB"
            f"  mem={mem_mb:.0f}MB  cpu={cpu_pct:.0f}%"
        )

        run_preds, run_trues = trainer.predict_aligned(eval_test_loader, out_len=h)  # (W, N) each
        run_adj = _try_extract_adj(adapter, model_name, dataloaders, trainer.device)

        from evaluation.metrics import metric_global

        pred_t = torch.from_numpy(run_preds).float()[:, :, None]
        true_t = torch.from_numpy(run_trues).float()[:, :, None]
        test_mae, test_mape, test_rmse = metric_global(pred_t, true_t, cfg.null_val)
        print(
            f"  [{model_name}] h={h:2d}  seed={seed}  run {r_idx + 1}/{R}"
            f"  test_mae={test_mae:.4f}  test_rmse={test_rmse:.4f}  test_mape={test_mape:.4f}"
        )
        logger.log_test_metrics(test_mae, test_mape, test_rmse, horizon=h)
        logger.log_summary(
            {
                "val_loss": val_loss,
                "test_mae": test_mae,
                "test_mape": test_mape,
                "test_rmse": test_rmse,
                "params": n_params,
                "train_time_s": elapsed,
                "mem_rss_mb": mem_mb,
                "cpu_percent": cpu_pct,
                "gpu_peak_allocated_mb": peak_gpu_alloc_mb,
                "gpu_peak_reserved_mb": peak_gpu_reserved_mb,
            }
        )
    finally:
        logger.finish()
    return run_preds, val_loss, run_adj


def _resolve_run(
    model_name: str,
    cfg: TrainingExperimentConfig,
    h: int,
    seed: int,
    r_idx: int,
    ckpt_dir: Path,
    completed_runs: dict,
    eval_test_loader: Any,
    test_count: int,
    canonical_scaler: Any,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Return (preds, val_loss, adj) for one horizon×seed, using cache when available."""
    R = cfg.num_runs
    ckpt_name = f"{model_name}_h{h}_s{seed}"
    run_dir = _run_artifact_dir(cfg, model_name, h, r_idx, seed)
    isolated_preds_path = run_dir / "predictions.npy"
    legacy_preds_path = ckpt_dir / f"{ckpt_name}_preds.npy"
    run_entry = completed_runs.get(model_name, {}).get(str(h), {}).get(str(seed))
    fingerprint = _compute_run_fingerprint(model_name, cfg, h)
    cached_preds_path = (
        Path(run_entry["predictions_path"])
        if run_entry is not None and run_entry.get("predictions_path")
        else isolated_preds_path
        if isolated_preds_path.exists()
        else legacy_preds_path
    )

    if (
        run_entry is not None
        and run_entry.get("fingerprint") == fingerprint
        and cached_preds_path.exists()
    ):
        run_preds = np.load(cached_preds_path)
        if run_preds.shape != (test_count, cfg.num_nodes):
            # Belt-and-suspenders: a fingerprint collision or hand-edited
            # cache could still leave a window-count mismatch — never splice
            # a wrongly-shaped array into preds_out, retrain instead.
            print(
                f"  [{model_name}] h={h:2d}  seed={seed}  cached preds shape "
                f"{run_preds.shape} != expected {(test_count, cfg.num_nodes)} — retraining"
            )
        elif _cached_adj_is_missing(run_entry):
            # The cache promised an adjacency it can no longer deliver. Accepting
            # the run would drop this horizon from the model's global adjacency
            # without a word; retrain so the probe gets the graph it expects.
            print(
                f"  [{model_name}] h={h:2d}  seed={seed}  cached adjacency "
                f"{run_entry['adj_path']} is missing — retraining"
            )
        else:
            val_loss = run_entry["val_loss"]
            run_adj = _load_cached_adj(run_entry)
            run_dir.mkdir(parents=True, exist_ok=True)
            if cached_preds_path != isolated_preds_path:
                np.save(isolated_preds_path, run_preds.astype(np.float32))
            cached_isolated_adj_path: Path | None = None
            if run_adj is not None:
                cached_isolated_adj_path = run_dir / "adjacency.npy"
                np.save(cached_isolated_adj_path, run_adj.astype(np.float32))
            if not (run_dir / "manifest.json").exists():
                _save_run_manifest(
                    run_dir,
                    model_name=model_name,
                    cfg=cfg,
                    horizon=h,
                    run_index=r_idx,
                    seed=seed,
                    fingerprint=fingerprint,
                    val_loss=val_loss,
                    predictions_path=isolated_preds_path,
                    adjacency_path=cached_isolated_adj_path,
                    cached=True,
                )
            run_entry["predictions_path"] = str(isolated_preds_path)
            if cached_isolated_adj_path is not None:
                # Only ever repoint at an adjacency we just wrote. Overwriting
                # with None would erase the sole record of where this run's
                # adjacency lives, making the loss unrecoverable.
                run_entry["adj_path"] = str(cached_isolated_adj_path)
            run_entry["artifact_dir"] = str(run_dir)
            run_entry["run_index"] = r_idx + 1
            _save_completed_runs(completed_runs, Path(cfg.output_dir))
            print(
                f"  [{model_name}] h={h:2d}  seed={seed}  run {r_idx + 1}/{R}"
                f"  val_loss={val_loss:.4f}  [cached]"
            )
            return run_preds, val_loss, run_adj

    if run_entry is not None and run_entry.get("fingerprint") != fingerprint:
        print(
            f"  [{model_name}] h={h:2d}  seed={seed}  cached run is stale "
            "(data source, preprocessing, or architecture changed) — retraining"
        )

    run_preds, val_loss, run_adj = _train_one_run(
        model_name, cfg, h, seed, r_idx, eval_test_loader, canonical_scaler
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(isolated_preds_path, run_preds.astype(np.float32))
    isolated_adj_path: Path | None = None
    if run_adj is not None:
        isolated_adj_path = run_dir / "adjacency.npy"
        np.save(isolated_adj_path, run_adj.astype(np.float32))
    _save_run_manifest(
        run_dir,
        model_name=model_name,
        cfg=cfg,
        horizon=h,
        run_index=r_idx,
        seed=seed,
        fingerprint=fingerprint,
        val_loss=val_loss,
        predictions_path=isolated_preds_path,
        adjacency_path=isolated_adj_path,
        cached=False,
    )
    completed_runs.setdefault(model_name, {}).setdefault(str(h), {})[str(seed)] = {
        "val_loss": val_loss,
        "predictions_path": str(isolated_preds_path),
        "adj_path": str(isolated_adj_path) if isolated_adj_path is not None else None,
        "artifact_dir": str(run_dir),
        "run_index": r_idx + 1,
        "fingerprint": fingerprint,
    }
    _save_completed_runs(completed_runs, Path(cfg.output_dir))
    return run_preds, val_loss, run_adj


def _collect_predictions_and_adj(
    model_name: str,
    cfg: TrainingExperimentConfig,
    seeds_log: dict,
    completed_runs: dict,
    eval_test_loader: Any,
    test_count: int,
    canonical_scaler: Any,
) -> tuple[np.ndarray, dict[int, np.ndarray | None]]:
    """Train model at each horizon × seed, skipping any already-completed runs.

    Returns
    -------
    preds_out : (W, H, N, R)
        Per-window predictions at each horizon and seed, paired 1:1 with the
        same canonical test windows as ``ground_truth.npy`` (``W ==
        test_count``) — no averaging over windows.
    adj_per_horizon : dict[horizon -> (R, N, N) | None]
        Every seed's learned adjacency for each horizon, stacked along a
        leading seed axis (not just the best-val seed), so the probe can
        compute per-seed structural measures and report mean±sd to match the
        mean-of-R accuracy table. The leading axis holds only the seeds that
        actually produced an adjacency; it is < R only in the pathological
        case where some seeds fail extraction. None when no seed produced one
        (e.g. the graph-free TCN baselines).
    """
    N = cfg.num_nodes
    H = len(cfg.horizons)
    R = cfg.num_runs
    W = test_count
    preds_out = np.zeros((W, H, N, R), dtype=np.float32)
    adj_per_horizon: dict[int, np.ndarray | None] = {}

    output_dir = Path(cfg.output_dir)
    ckpt_dir = Path(cfg.checkpoint_dir)

    # Allocate all seeds before any training so a crashed run can be resumed
    # with the identical seeds. Short lists are extended to exactly R entries;
    # longer lists are rejected rather than truncated because removing their
    # seed identity would orphan already-completed independent-run artifacts.
    model_seeds = seeds_log.setdefault(model_name, {})
    for h in cfg.horizons:
        h_key = str(h)
        existing = model_seeds.get(h_key, [])
        _validate_recorded_seeds(output_dir / "seeds.json", model_name, h, existing, R)
        while len(existing) < R:
            candidate = random.randrange(1, 100_000)
            if candidate not in existing:
                existing.append(candidate)
        model_seeds[h_key] = existing
    _save_seeds_log(seeds_log, output_dir)

    for h_idx, h in enumerate(cfg.horizons):
        h_seeds = model_seeds[str(h)]
        assert len(h_seeds) == R, f"{model_name} h={h}: expected {R} seeds, got {len(h_seeds)}."
        seed_adjs: list[np.ndarray] = []

        for r_idx, seed in enumerate(h_seeds):
            run_preds, val_loss, run_adj = _resolve_run(
                model_name,
                cfg,
                h,
                seed,
                r_idx,
                ckpt_dir,
                completed_runs,
                eval_test_loader,
                W,
                canonical_scaler,
            )
            preds_out[:, h_idx, :, r_idx] = run_preds
            if run_adj is not None:
                seed_adjs.append(np.asarray(run_adj, dtype=np.float32))

        # Retain every seed's adjacency (stacked on a leading seed axis), not
        # just the best-val one, so downstream structure (AAS, modularity) can
        # be computed per seed and averaged — keeping the structure table
        # mean-of-R consistent with the accuracy table.
        adj_per_horizon[h] = np.stack(seed_adjs, axis=0) if seed_adjs else None

    return preds_out, adj_per_horizon


# ---------------------------------------------------------------------------
# ARIMA (deterministic — no seeds, no GPU, no adjacency)
# ---------------------------------------------------------------------------


def _collect_arima_predictions(
    cfg: TrainingExperimentConfig,
    test_start: int,
    test_count: int,
) -> np.ndarray:
    """Run ARIMA once per horizon (deterministic) and tile to (W, H, N, R).

    ARIMA is univariate and deterministic: all probe "runs" are identical.
    Aligned to ``test_start``/``test_count`` — the same canonical test window
    origin set every neural model is evaluated on — so its forecasts are
    paired 1:1 with ``ground_truth.npy``.
    """
    from data.tsl_pipeline import load_tsl_raw_array
    from training.arima_trainer import predict_arima

    data = load_tsl_raw_array(cfg.dataset_name)  # (T, N, 3) — raw unnormalised
    preds_1run = predict_arima(
        data,
        horizons=cfg.horizons,
        # predict_arima's test_start is the forecast origin, i.e. past the
        # input window — test_start here is the window's input start.
        test_start=test_start + cfg.in_len,
        test_count=test_count,
        verbose=True,
    )  # (W, H, N)
    # ARIMA is deterministic — tile the single run to (W, H, N, R)
    return np.repeat(preds_1run[:, :, :, None], cfg.num_runs, axis=-1).astype(np.float32)


def _log_arima_metrics(
    cfg: TrainingExperimentConfig,
    preds: np.ndarray,
    gt_path: Path,
    output_dir: Path,
    resource_metrics: dict[str, float],
) -> dict[str, Any]:
    """Compute ARIMA's one-shot test metrics and log them to W&B + disk.

    ARIMA has no training loop/epochs, so it never goes through
    ``training.logger.WandbLogger`` like the neural models do — this is its
    equivalent one-shot logging call, scored against the same
    ``ground_truth.npy`` every other model is paired against.
    """
    from evaluation.metrics import metric_per_horizon

    ground_truth = np.load(gt_path)  # (W, H, N)
    pred = torch.from_numpy(preds[..., 0]).permute(0, 2, 1).float()  # (W, N, H)
    true = torch.from_numpy(ground_truth).permute(0, 2, 1).float()  # (W, N, H)
    per_h = metric_per_horizon(pred, true, null_val=cfg.null_val)

    payload: dict[str, Any] = {
        "horizons": cfg.horizons,
        "mae_h": per_h["mae"].tolist(),
        "mape_h": per_h["mape"].tolist(),
        "rmse_h": per_h["rmse"].tolist(),
        "mae": float(per_h["mae"].mean()),
        "mape": float(per_h["mape"].mean()),
        "rmse": float(per_h["rmse"].mean()),
        "system": resource_metrics,
    }

    metrics_file = output_dir / "arima_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved ARIMA metrics → {metrics_file}")
    arima_run_dir = output_dir / "runs" / "arima" / "deterministic"
    arima_run_dir.mkdir(parents=True, exist_ok=True)
    with open(arima_run_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    if cfg.use_wandb:
        import wandb

        from training.env import resolve_wandb_entity

        init_kwargs: dict[str, Any] = dict(
            project=cfg.wandb_project,
            name="arima",
            config={"order": (3, 0, 1), "horizons": cfg.horizons, "dataset": cfg.dataset_name},
            mode=cfg.wandb_mode,
        )
        entity = resolve_wandb_entity(cfg.wandb_entity)
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs)
        for h_idx, horizon in enumerate(cfg.horizons):
            wandb.log(
                {
                    f"test/MAE_h{horizon}": payload["mae_h"][h_idx],
                    f"test/MAPE_h{horizon}": payload["mape_h"][h_idx],
                    f"test/RMSE_h{horizon}": payload["rmse_h"][h_idx],
                }
            )
        wandb.log(
            {
                "test/MAE": payload["mae"],
                "test/MAPE": payload["mape"],
                "test/RMSE": payload["rmse"],
                **{f"system/{key}": value for key, value in resource_metrics.items()},
            }
        )
        wandb.finish()

    return payload


# ---------------------------------------------------------------------------
# Stale-file cleanup
# ---------------------------------------------------------------------------


def _cleanup_old_adjacency_files(model_name: str, adjacency_dir: Path) -> None:
    """Remove {model}_adjacency_h*.npy files left by a previous run.

    Prevents stale per-horizon files from a different seed set or horizon list
    from persisting alongside freshly saved results.
    """
    for stale in adjacency_dir.glob(f"{model_name}_adjacency_h*.npy"):
        stale.unlink()
        print(f"  Removed stale adjacency: {stale.name}")


def _save_adjacency_files(
    model_name: str,
    adj_per_horizon: dict[int, np.ndarray | None],
    adjacency_dir: Path,
) -> None:
    """Write, for every horizon, both a full per-seed adjacency stack and a
    seed-mean representative, plus the global (over-horizon) counterparts.

    ``adj_per_horizon[h]`` is ``(R, N, N)`` — every seed's learned adjacency.
    Two artifacts are written per horizon so the mean-of-R structure work is
    decoupled from the existing single-adjacency probe:

    * ``{model}_adjacency_h{h}_seeds.npy`` — the ``(R, N, N)`` stack. This is
      the retained source the per-seed AAS/modularity mean±sd is computed from;
      it must exist after the multi-seed run or the structure table cannot be
      made seed-consistent without re-training.
    * ``{model}_adjacency_h{h}.npy`` — the seed-mean ``(N, N)`` graph the probe
      currently consumes and the figures plot (a representative graph, not the
      reported statistic).

    ``{model}_adjacency.npy`` is the mean of the per-horizon representatives.
    There is deliberately no global ``_seeds`` stack: runs are independently
    seeded per horizon, so array index ``r`` at two horizons does not identify
    one common training run. The probe reads the per-horizon stacks and scores
    each real horizon×seed adjacency independently.
    """
    _cleanup_old_adjacency_files(model_name, adjacency_dir)
    seed_means: list[np.ndarray] = []  # (N, N) per horizon
    for h, adj in adj_per_horizon.items():
        if adj is None:
            continue
        stack = np.asarray(adj, dtype=np.float32)
        if stack.ndim == 2:  # tolerate a lone (N, N) from a single-seed run
            stack = stack[None]
        seed_mean = stack.mean(axis=0).astype(np.float32)
        np.save(adjacency_dir / f"{model_name}_adjacency_h{h}_seeds.npy", stack)
        np.save(adjacency_dir / f"{model_name}_adjacency_h{h}.npy", seed_mean)
        seed_means.append(seed_mean)
        print(f"  Saved adjacency h={h:2d}  seeds{stack.shape} mean{seed_mean.shape}")
    if not seed_means:
        raise RuntimeError(f"No learned adjacency was extracted for spatial model {model_name}.")

    global_mean = np.mean(np.stack(seed_means, axis=0), axis=0).astype(np.float32)
    np.save(adjacency_dir / f"{model_name}_adjacency.npy", global_mean)
    stale_global_seeds = adjacency_dir / f"{model_name}_adjacency_seeds.npy"
    if stale_global_seeds.exists():
        stale_global_seeds.unlink()
        print(f"  Removed stale adjacency: {stale_global_seeds.name}")
    print(f"  Saved global adjacency mean{global_mean.shape}")


# ---------------------------------------------------------------------------
# Probe runner (called from background thread)
# ---------------------------------------------------------------------------


def _run_probe(probe_config_path: str, dataset_name: str, model_name: str) -> None:
    from analysis.config import load_config
    from analysis.probe import ProbeRunner

    try:
        cfg = load_config(probe_config_path)
        runner = ProbeRunner(
            output_dir=Path(cfg.datasets[0].raw_data).parent.parent / "probe_outputs",
            config=cfg,
        )
        result = runner.run_from_config(dataset_name, model_name, save_figures=True)
        result.print_summary()
    except Exception as exc:
        print(f"[probe] {model_name} failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: TrainingExperimentConfig, model_filter: str | list[str] | None = None) -> None:
    """Run training for the configured models.

    ``model_filter`` accepts a single model name, a comma-separated string
    (e.g. ``"gwn,tcn,arima"``), a list of names, or ``None`` to run every
    configured spatial model plus all temporal baselines. Models run in the
    order given.
    """
    model_filter = _normalize_model_filter(model_filter)
    validate_experiment_config(cfg, model_filter=model_filter)

    output_dir = Path(cfg.output_dir)
    predictions_dir = Path(cfg.predictions_dir)
    adjacency_dir = Path(cfg.adjacency_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    adjacency_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = model_filter if model_filter else _default_run_order(cfg.spatial_models)

    # Canonical evaluation pipeline shared by every model/horizon/seed: built
    # once at out_len=max(horizons), it defines the single test-window origin
    # set every model (and ARIMA) is paired against, so Lens 0 can compute
    # metrics from genuinely paired per-window samples instead of values
    # pre-averaged over windows first (see _build_ground_truth/predict_aligned).
    from data.tsl_pipeline import build_tsl_pipeline, compute_test_window_bounds, load_tsl_raw_array

    max_h = max(cfg.horizons)
    eval_pipeline = build_tsl_pipeline(
        cfg.dataset_name,
        in_len=cfg.in_len,
        out_len=max_h,
        batch_size=cfg.batch_size,
        val_len=cfg.val_len,
        test_len=cfg.test_len,
    )
    eval_test_loader = eval_pipeline.dataloaders["test"]
    n_steps = load_tsl_raw_array(cfg.dataset_name).shape[0]
    test_start, test_count = compute_test_window_bounds(
        n_steps, cfg.in_len, max_h, val_len=cfg.val_len, test_len=cfg.test_len
    )

    # Build ground truth reference once from the canonical test windows.
    # Guarded by a fingerprint so a stale ground truth from a previous data
    # source/horizon config is never silently reused.
    gt_path = Path(cfg.ground_truth_path)
    gt_fingerprint_path = gt_path.with_suffix(gt_path.suffix + ".fingerprint.json")
    gt_fingerprint = _compute_ground_truth_fingerprint(cfg)
    cached_gt_fingerprint = None
    if gt_fingerprint_path.exists():
        with open(gt_fingerprint_path) as f:
            cached_gt_fingerprint = json.load(f).get("fingerprint")

    if not gt_path.exists() or cached_gt_fingerprint != gt_fingerprint:
        if gt_path.exists():
            print("Ground truth cache is stale (dataset/horizon config changed) — rebuilding …")
        else:
            print("Building ground truth reference …")
        _build_ground_truth(cfg, gt_path, eval_test_loader, eval_pipeline.scaler)
        gt_fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_fingerprint_path, "w") as f:
            json.dump({"fingerprint": gt_fingerprint}, f)

    seeds_log = _load_seeds_log(output_dir)
    completed_runs = _load_completed_runs(output_dir)

    from training.resource_monitor import ResourceSampler

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_future = None

        for model_name in models_to_run:
            model_sampler = ResourceSampler()
            model_t0 = time.time()

            print(f"\n{'=' * 60}")
            print(f"  Model: {model_name}  ({cfg.dataset_name})")
            print(f"{'=' * 60}")

            # ── ARIMA: deterministic baseline, no training loop, no adjacency ──
            if model_name == "arima":
                print(f"  Running ARIMA (deterministic — 1 fit per horizon, tiled ×{cfg.num_runs})")
                preds = _collect_arima_predictions(cfg, test_start, test_count)
                preds_file = predictions_dir / "arima_predictions.npy"
                np.save(preds_file, preds)
                print(f"  Saved ARIMA predictions {preds.shape} → {preds_file}")
                model_mem, model_cpu = model_sampler.sample()
                model_elapsed = time.time() - model_t0
                arima_resources = {
                    "train_time_s": model_elapsed,
                    "mem_rss_mb": model_mem,
                    "cpu_percent": model_cpu,
                    "gpu_allocated_mb": 0.0,
                    "gpu_reserved_mb": 0.0,
                    "gpu_peak_allocated_mb": 0.0,
                    "gpu_peak_reserved_mb": 0.0,
                }
                arima_run_dir = output_dir / "runs" / "arima" / "deterministic"
                arima_run_dir.mkdir(parents=True, exist_ok=True)
                np.save(arima_run_dir / "predictions.npy", preds)
                _log_arima_metrics(cfg, preds, gt_path, output_dir, arima_resources)
                _log_resource_usage(
                    output_dir,
                    {
                        "model": "arima",
                        "dataset": cfg.dataset_name,
                        "horizon": "all",
                        "seed": "",
                        "params": 0,
                        "peak_gpu_alloc_mb": 0.0,
                        "peak_gpu_reserved_mb": 0.0,
                        "train_time_s": round(model_elapsed, 1),
                        "cpu_rss_mb": round(model_mem, 1),
                        "cpu_percent": round(model_cpu, 1),
                    },
                )
                print(
                    f"  [{model_name}] total {model_elapsed:.0f}s"
                    f"  mem={model_mem:.0f}MB  cpu={model_cpu:.0f}%"
                )
                continue  # no seeds log, no probe submission

            # ── Neural models (spatial + temporal baselines) ──
            print(
                f"  {len(cfg.horizons)} horizons × {cfg.num_runs} seeds = "
                f"{len(cfg.horizons) * cfg.num_runs} jobs"
            )

            preds, adj_per_horizon = _collect_predictions_and_adj(
                model_name,
                cfg,
                seeds_log,
                completed_runs,
                eval_test_loader,
                test_count,
                eval_pipeline.scaler,
            )

            preds_file = predictions_dir / f"{model_name}_predictions.npy"
            np.save(preds_file, preds)
            print(f"  Saved predictions  {preds.shape} → {preds_file}")

            is_spatial = model_name in SUPPORTED_SPATIAL_MODELS
            if is_spatial:
                _save_adjacency_files(model_name, adj_per_horizon, adjacency_dir)

            _save_seeds_log(seeds_log, output_dir)

            model_mem, model_cpu = model_sampler.sample()
            print(
                f"  [{model_name}] total {time.time() - model_t0:.0f}s"
                f"  mem={model_mem:.0f}MB  cpu={model_cpu:.0f}%"
            )

            # Pipeline: wait for previous probe, then start this one
            if pending_future is not None:
                pending_future.result()

            if is_spatial and cfg.probe_config_path:
                pending_future = pool.submit(
                    _run_probe, cfg.probe_config_path, cfg.dataset_name, model_name
                )

        if pending_future is not None:
            pending_future.result()

    print(f"\nAll runs complete. Seeds logged to {output_dir / 'seeds.json'}")


def _build_ground_truth(
    cfg: TrainingExperimentConfig,
    gt_path: Path,
    eval_test_loader: Any,
    scaler: Any,
) -> None:
    """Build and save (W, H, N) ground truth from the canonical test windows.

    Every model's predictions are paired 1:1 against this same array (no
    averaging over windows), so Lens 0 can compute metrics from genuinely
    paired per-window samples instead of values pre-averaged over windows.
    """
    horizon_indices = [h - 1 for h in cfg.horizons]
    all_true = []
    for _x, y, _y_full in eval_test_loader:
        y_real = scaler.inverse_transform(y)  # (B, N, max_h)
        if hasattr(y_real, "numpy"):
            y_real = y_real.numpy()
        all_true.append(np.asarray(y_real)[:, :, horizon_indices])  # (B, N, H)

    trues = np.concatenate(all_true, axis=0)  # (W, N, H)
    ground_truth = trues.transpose(0, 2, 1)  # (W, H, N)

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gt_path, ground_truth.astype(np.float32))
    print(f"  Saved ground truth {ground_truth.shape} → {gt_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train STGNN models and prepare STGNN-Probe inputs."
    )
    parser.add_argument("--config", required=True, help="Path to training experiment YAML config.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Run specific model(s) by name, comma-separated (e.g. 'gwn' or "
            "'gwn,tcn,arima'). Runs in the order given. Omit to run all "
            "configured models."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # CPU runs are a single process, but torch's intra-op thread pool
    # (OpenBLAS/MKL) defaults to spawning threads across every core on the
    # box. These per-node TCN models are tiny — letting them fan out to
    # 47+ cores buys no speedup and just adds scheduler/futex contention
    # (same symptom as the BLAS oversubscription fixed for Granger workers
    # in lens2_granger.py, but single-process here rather than per-worker).
    torch.set_num_threads(8)
    args = build_parser().parse_args(argv)
    cfg = load_experiment_config(args.config)
    run(cfg, model_filter=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
