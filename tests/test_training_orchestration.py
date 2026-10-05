from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import Tensor

from data.scaler import StandardScaler
from scripts import run_sweep, run_training
from scripts.run_training import TrainingExperimentConfig, run, validate_experiment_config


def _base_config(
    tmp_path: Path, *, spatial_models: list[str] | None = None
) -> TrainingExperimentConfig:
    probe_config_path = tmp_path / "probe.yaml"
    probe_config_path.write_text("datasets: []\n")
    return TrainingExperimentConfig(
        dataset_name="METR-LA",
        num_nodes=3,
        horizons=[1, 2],
        num_runs=1,
        output_dir=str(tmp_path / "outputs"),
        ground_truth_path=str(tmp_path / "probe_inputs" / "ground_truth.npy"),
        predictions_dir=str(tmp_path / "probe_inputs" / "predictions"),
        adjacency_dir=str(tmp_path / "probe_inputs" / "adjacency"),
        probe_config_path=str(probe_config_path),
        spatial_models=spatial_models or ["gwn"],
        in_len=4,
        in_dim=3,
        batch_size=2,
        epochs=1,
        patience=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


_SENTINEL_SCALER = object()
"""A unique object standing in for ``eval_pipeline.scaler`` — its identity,
not its (nonexistent) behaviour, is what `test_run_passes_eval_pipeline_scaler_into_collect`
checks: that `run()` threads the exact same scaler instance through, not a
fresh/different one."""


class _FakeTslPipeline:
    """Stand-in for TslPipeline — only `.dataloaders["test"]`/`.scaler` are
    ever touched, and both are faked out downstream in this test."""

    def __init__(self) -> None:
        self.dataloaders: dict[str, Any] = {"test": []}
        self.scaler: Any = _SENTINEL_SCALER


def test_run_writes_global_and_per_horizon_adjacency_and_submits_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config(tmp_path)
    probe_calls: list[tuple[str, str, str]] = []
    collect_scalers: list[Any] = []

    def fake_collect(
        model_name: str,
        cfg: TrainingExperimentConfig,
        seeds_log: dict[str, Any],
        completed_runs: dict[str, Any],
        eval_test_loader: Any,
        test_count: int,
        canonical_scaler: Any,
    ) -> tuple[np.ndarray, dict[int, np.ndarray | None]]:
        collect_scalers.append(canonical_scaler)
        predictions = np.ones(
            (test_count, len(cfg.horizons), cfg.num_nodes, cfg.num_runs), dtype=np.float32
        )
        if model_name == "tcn":
            return predictions, {}
        # Per-horizon adjacency is now every seed stacked on a leading axis:
        # (R, N, N). Vary values across the seed axis so a mean-vs-stack bug
        # (e.g. saving only one seed) would change the saved shapes.
        R, N = cfg.num_runs, cfg.num_nodes
        eye_stack = np.stack([np.eye(N, dtype=np.float32) * (r + 1) for r in range(R)], axis=0)
        adjacencies: dict[int, np.ndarray | None] = {
            1: eye_stack,
            2: np.full((R, N, N), 2.0, dtype=np.float32),
        }
        return predictions, adjacencies

    def fake_run_probe(probe_config_path: str, dataset_name: str, model_name: str) -> None:
        probe_calls.append((probe_config_path, dataset_name, model_name))

    def fake_build_ground_truth(
        cfg: TrainingExperimentConfig,
        gt_path: Path,
        eval_test_loader: Any,
        scaler: Any,
    ) -> None:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gt_path, np.ones((1, len(cfg.horizons), cfg.num_nodes), dtype=np.float32))

    def fake_arima_predictions(
        cfg: TrainingExperimentConfig, test_start: int, test_count: int
    ) -> np.ndarray:
        return np.ones(
            (test_count, len(cfg.horizons), cfg.num_nodes, cfg.num_runs), dtype=np.float32
        )

    def fake_build_tsl_pipeline(
        dataset_name: str,
        in_len: int,
        out_len: int,
        batch_size: int,
        val_len: float = 0.1,
        test_len: float = 0.2,
    ) -> _FakeTslPipeline:
        return _FakeTslPipeline()

    def fake_compute_test_window_bounds(
        n_steps: int, in_len: int, out_len: int, val_len: float = 0.1, test_len: float = 0.2
    ) -> tuple[int, int]:
        return 0, 1

    def fake_load_tsl_raw_array(dataset_name: str) -> np.ndarray:
        return np.zeros((100, 3, 3), dtype=np.float32)

    monkeypatch.setattr(run_training, "_collect_predictions_and_adj", fake_collect)
    monkeypatch.setattr(run_training, "_run_probe", fake_run_probe)
    monkeypatch.setattr(run_training, "_build_ground_truth", fake_build_ground_truth)
    monkeypatch.setattr(run_training, "_collect_arima_predictions", fake_arima_predictions)

    # run() does `from data.tsl_pipeline import ...` as a local import, and the
    # real module imports `tsl.data` at module level, which transitively
    # requires torch_sparse/torch_scatter (unavailable on macOS — see
    # Dockerfile). Inject a fake module into sys.modules *before* it's ever
    # imported, instead of monkeypatch.setattr("data.tsl_pipeline.X", ...)
    # (which itself imports the real module to patch its attribute), so this
    # orchestration test — which never needs the real tsl integration — runs
    # on any platform.
    fake_tsl_pipeline = types.ModuleType("data.tsl_pipeline")
    fake_tsl_pipeline.build_tsl_pipeline = fake_build_tsl_pipeline  # type: ignore[attr-defined]
    fake_tsl_pipeline.compute_test_window_bounds = fake_compute_test_window_bounds  # type: ignore[attr-defined]
    fake_tsl_pipeline.load_tsl_raw_array = fake_load_tsl_raw_array  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "data.tsl_pipeline", fake_tsl_pipeline)

    run(cfg, model_filter=None)

    predictions_dir = Path(cfg.predictions_dir)
    adjacency_dir = Path(cfg.adjacency_dir)
    assert (Path(cfg.ground_truth_path)).exists()
    assert np.load(predictions_dir / "tcn_predictions.npy").shape == (1, 2, 3, 1)
    assert np.load(predictions_dir / "gwn_predictions.npy").shape == (1, 2, 3, 1)
    assert np.load(adjacency_dir / "gwn_adjacency_h1.npy").shape == (3, 3)
    assert np.load(adjacency_dir / "gwn_adjacency_h2.npy").shape == (3, 3)
    np.testing.assert_allclose(
        np.load(adjacency_dir / "gwn_adjacency.npy"),
        (np.eye(3, dtype=np.float32) + np.full((3, 3), 2.0, dtype=np.float32)) / 2,
    )
    global_adj = np.load(adjacency_dir / "gwn_adjacency.npy")
    assert global_adj.dtype == np.float32, "Global adjacency must be float32"
    assert global_adj.any(), "Global adjacency must not be all-zero"
    assert np.isfinite(global_adj).all(), "Global adjacency must be finite"
    # Every seed's adjacency is retained as an (R, N, N) stack alongside the
    # (N, N) seed-mean representative; the representative is the seed-axis mean.
    R = cfg.num_runs
    assert np.load(adjacency_dir / "gwn_adjacency_h1_seeds.npy").shape == (R, 3, 3)
    assert np.load(adjacency_dir / "gwn_adjacency_h2_seeds.npy").shape == (R, 3, 3)
    np.testing.assert_allclose(
        np.load(adjacency_dir / "gwn_adjacency_h1.npy"),
        np.load(adjacency_dir / "gwn_adjacency_h1_seeds.npy").mean(axis=0),
    )
    assert not (adjacency_dir / "gwn_adjacency_seeds.npy").exists()
    assert np.load(predictions_dir / "arima_predictions.npy").shape == (1, 2, 3, 1)
    assert probe_calls == [(str(tmp_path / "probe.yaml"), "METR-LA", "gwn")]

    # run() must thread eval_pipeline's own scaler instance into every
    # _collect_predictions_and_adj call (one per spatial/temporal neural model) — not a
    # fresh/different one — so every model trains on the exact same
    # normalisation statistics it's later evaluated against.
    assert collect_scalers == [_SENTINEL_SCALER, _SENTINEL_SCALER]


def test_save_adjacency_keeps_partial_horizon_seed_stacks_separate(tmp_path: Path) -> None:
    adjacency_dir = tmp_path / "adjacency"
    adjacency_dir.mkdir()
    stale_global_stack = adjacency_dir / "gwn_adjacency_seeds.npy"
    np.save(stale_global_stack, np.zeros((1, 3, 3), dtype=np.float32))

    h1 = np.stack([np.eye(3, dtype=np.float32), np.ones((3, 3), dtype=np.float32)])
    h2 = np.stack([np.full((3, 3), 2.0, dtype=np.float32) for _ in range(3)])
    run_training._save_adjacency_files("gwn", {1: h1, 2: h2}, adjacency_dir)

    assert np.load(adjacency_dir / "gwn_adjacency_h1_seeds.npy").shape == (2, 3, 3)
    assert np.load(adjacency_dir / "gwn_adjacency_h2_seeds.npy").shape == (3, 3, 3)
    assert not stale_global_stack.exists()


def test_sweep_pipeline_uses_experiment_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_config(tmp_path)
    cfg.val_len = 0.125
    cfg.test_len = 0.2
    captured: dict[str, Any] = {}

    def fake_build_tsl_pipeline(*args: Any, **kwargs: Any) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    fake_tsl_pipeline = types.ModuleType("data.tsl_pipeline")
    fake_tsl_pipeline.build_tsl_pipeline = fake_build_tsl_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "data.tsl_pipeline", fake_tsl_pipeline)

    run_sweep._build_sweep_pipeline(cfg, horizon=12)

    assert captured["args"] == (cfg.dataset_name,)
    assert captured["kwargs"]["val_len"] == 0.125
    assert captured["kwargs"]["test_len"] == 0.2


def test_validation_rejects_missing_probe_config_for_spatial_models(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.probe_config_path = str(tmp_path / "missing_probe.yaml")

    with pytest.raises(FileNotFoundError, match="probe_config_path"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize("model_name", ["not_a_model", "astgcn", "tcn_shared"])
def test_validation_rejects_unknown_model_before_training(tmp_path: Path, model_name: str) -> None:
    cfg = _base_config(tmp_path)

    with pytest.raises(ValueError, match="Unknown model"):
        validate_experiment_config(cfg, model_filter=model_name)


def test_validation_accepts_comma_separated_model_filter(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path, spatial_models=["gwn"])

    validate_experiment_config(cfg, model_filter="gwn,tcn,arima")  # must not raise


def test_validation_rejects_unknown_model_in_comma_separated_filter(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path, spatial_models=["gwn"])

    with pytest.raises(ValueError, match=r"Unknown model.*not_a_model"):
        validate_experiment_config(cfg, model_filter="gwn,not_a_model")


def test_normalize_model_filter_preserves_order_and_dedupes() -> None:
    assert run_training._normalize_model_filter("gwn, tcn, gwn") == [
        "gwn",
        "tcn",
    ]
    assert run_training._normalize_model_filter(["gwn", "tcn"]) == ["gwn", "tcn"]
    assert run_training._normalize_model_filter(None) is None
    assert run_training._normalize_model_filter("") is None


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "scripts" / "configs"
_SHIPPED_TRAINING_CONFIGS = [
    "metr_la_training.yaml",
    "pems_bay_training.yaml",
    "metr_la_smoke_new_models.yaml",
]


@pytest.mark.parametrize("config_name", _SHIPPED_TRAINING_CONFIGS)
def test_probe_config_benchmarks_every_model_its_training_config_trains(config_name: str) -> None:
    """A shipped training config and its companion probe config must agree.

    Running a training config with no ``--model`` filter trains every temporal
    baseline plus every configured spatial model. Lens 0 discovers models from
    ``performance.model_groups`` (see ``ProbeRunner.run_performance``) and merely
    logs a warning for a missing prediction file — so a model trained but left
    out of that mapping costs a full training run and is then silently dropped
    from the comparison.
    """
    cfg = run_training.load_experiment_config(_CONFIG_DIR / config_name)
    trained = [*run_training.TEMPORAL_MODEL_ORDER, *cfg.spatial_models]

    assert cfg.probe_config_path is not None, f"{config_name} has no probe_config_path"
    with open(cfg.probe_config_path) as f:
        probe_raw = yaml.safe_load(f)
    grouped = {
        model for members in probe_raw["performance"]["model_groups"].values() for model in members
    }

    missing = [model for model in trained if model not in grouped]
    assert not missing, (
        f"{config_name} trains {missing} but {Path(cfg.probe_config_path).name} omits them "
        "from performance.model_groups, so Lens 0 would skip their predictions"
    )


@pytest.mark.parametrize("config_name", _SHIPPED_TRAINING_CONFIGS)
def test_shipped_training_configs_cap_epochs_for_every_neural_model(config_name: str) -> None:
    """Each model's own base YAML sets epochs, overriding the config's top-level value.

    ``_train_one_run``/``_train_tcn_pernode`` read ``overrides.get("epochs",
    cfg.epochs)``, and every ``<model>_base.yaml`` sets ``epochs`` in its
    ``datasets.<dataset>.training`` section — so ``cfg.epochs`` is only a
    fallback that never applies in practice. This pins the resolved value so a
    newly added model cannot silently train far longer than the config intends
    (the smoke config in particular is meaningless at 100 epochs).
    """
    cfg = run_training.load_experiment_config(_CONFIG_DIR / config_name)
    neural = [m for m in run_training.TEMPORAL_MODEL_ORDER if m != "arima"]

    resolved = {
        model: run_training._resolve_model_overrides(model, cfg).get("epochs", cfg.epochs)
        for model in [*neural, *cfg.spatial_models]
    }

    if "smoke" in config_name:
        assert all(epochs <= 3 for epochs in resolved.values()), (
            f"{config_name} is a low-cost smoke config but resolves {resolved}"
        )
    else:
        assert all(epochs > 3 for epochs in resolved.values()), (
            f"{config_name} is a production config but resolves {resolved}"
        )


def test_validation_rejects_unknown_model_overrides_before_training(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.model_overrides["not_a_model"] = {}

    with pytest.raises(ValueError, match="model_overrides"):
        validate_experiment_config(cfg)


def test_validation_rejects_dssa_tcn_without_time_feature_channels(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path, spatial_models=["dssa_tcn"])
    cfg.in_dim = 2

    with pytest.raises(
        ValueError, match="dssa_tcn, staeformer, d2stgnn, and bigst require in_dim >= 3"
    ):
        validate_experiment_config(cfg)


def test_ensure_bigst_long_term_features_generates_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_config(tmp_path, spatial_models=["bigst"])
    feat_path = tmp_path / "bigst_long_term_features.npy"
    calls: list[dict[str, Any]] = []

    def fake_preprocess(**kwargs: Any) -> Path:
        calls.append(kwargs)
        np.save(kwargs["output"], np.zeros((cfg.num_nodes, kwargs["nhid"]), dtype=np.float32))
        return Path(kwargs["output"])

    monkeypatch.setattr(run_training, "_preprocess_bigst_long_term_features", fake_preprocess)

    run_training._ensure_bigst_long_term_features(
        cfg,
        {"hid_dim": 16, "long_term_epochs": 2, "long_term_batch_size": 4},
        feat_path,
        torch.device("cpu"),
    )

    assert feat_path.exists()
    assert calls == [
        {
            "dataset": "METR-LA",
            "epochs": 2,
            "input_length": 288,
            "output_length": 12,
            "batch_size": 4,
            "nhid": 16,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip": 5.0,
            "device": "cpu",
            "output": feat_path,
        }
    ]


def test_ensure_bigst_long_term_features_reuses_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_config(tmp_path, spatial_models=["bigst"])
    feat_path = tmp_path / "bigst_long_term_features.npy"
    np.save(feat_path, np.zeros((cfg.num_nodes, 32), dtype=np.float32))

    def fail_preprocess(**_kwargs: Any) -> Path:
        raise AssertionError("preprocessing should not run for an existing feature file")

    monkeypatch.setattr(run_training, "_preprocess_bigst_long_term_features", fail_preprocess)

    run_training._ensure_bigst_long_term_features(
        cfg,
        {"hid_dim": 32},
        feat_path,
        torch.device("cpu"),
    )


def test_train_one_run_threads_wandb_config_into_logger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cfg.use_wandb``/``wandb_*`` must reach the per-run ``TrainerConfig`` and
    a real logger — not the hardcoded ``NoOpLogger`` this used to be wired to."""
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.use_wandb = True
    cfg.wandb_entity = "team-x"
    cfg.wandb_project = "proj-x"
    cfg.wandb_mode = "offline"
    model_name = "gwn"

    class _FakeTrainer:
        def __init__(self, model: Any, config: Any, dataloaders: Any, scaler: Any, logger: Any):
            self.device = torch.device("cpu")
            self.logger = logger

        def train(self) -> dict[str, float]:
            return {"val_loss": 1.0}

        def predict_aligned(self, loader: Any, out_len: int) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    class _FakeLogger:
        def __init__(self) -> None:
            self.finished = False

        def watch(self, model: Any) -> None:
            pass

        def log_test_metrics(self, mae: float, mape: float, rmse: float, horizon: int) -> None:
            pass

        def log_summary(self, metrics: dict[str, Any]) -> None:
            pass

        def finish(self) -> None:
            self.finished = True

    fake_logger = _FakeLogger()
    logger_calls: list[tuple[Any, str]] = []

    def fake_make_logger(trainer_config: Any, run_name: str = "") -> Any:
        logger_calls.append((trainer_config, run_name))
        return fake_logger

    def fake_load_dataloaders(
        cfg: Any, out_len: int, scaler: Any = None
    ) -> tuple[dict, Any, tuple[None, None]]:
        return {}, None, (None, None)

    monkeypatch.setattr(run_training, "_load_dataloaders", fake_load_dataloaders)
    monkeypatch.setattr(
        run_training,
        "_build_model_and_adapter",
        lambda model_name, cfg, out_len, tsl_graph: (object(), _FakeTrainer),
    )
    monkeypatch.setattr(run_training, "_try_extract_adj", lambda *a, **k: None)
    monkeypatch.setattr("training.logger.make_logger", fake_make_logger)

    run_training._train_one_run(
        model_name, cfg, h=1, seed=1, r_idx=0, eval_test_loader=None, canonical_scaler=None
    )

    assert len(logger_calls) == 1
    trainer_config, run_name = logger_calls[0]
    assert run_name == f"{model_name}_h1_s1"
    assert trainer_config.use_wandb is True
    assert trainer_config.wandb_entity == "team-x"
    assert trainer_config.wandb_project == "proj-x"
    assert trainer_config.wandb_mode == "offline"
    assert fake_logger.finished is True


def test_train_one_run_threads_patience_and_grad_clip_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """patience and grad_clip must resolve per-model the same way lr/epochs/
    scheduler already do — both were previously hardcoded to cfg.patience /
    TrainerConfig's bare default (5.0) with no override path at all, which
    silently applied this framework's generic defaults to every model
    regardless of what its original paper/codebase actually used (e.g.
    STAWnet's lack of early stopping, GWN-v2's clip=3)."""
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.patience = 15
    cfg.model_overrides["gwn"] = {"patience": 100, "grad_clip": 3.0}

    captured_configs: list[Any] = []

    class _FakeTrainer:
        def __init__(self, model: Any, config: Any, dataloaders: Any, scaler: Any, logger: Any):
            captured_configs.append(config)
            self.device = torch.device("cpu")

        def train(self) -> dict[str, float]:
            return {"val_loss": 1.0}

        def predict_aligned(self, loader: Any, out_len: int) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    def fake_load_dataloaders(
        cfg: Any, out_len: int, scaler: Any = None
    ) -> tuple[dict, Any, tuple[None, None]]:
        return {}, None, (None, None)

    monkeypatch.setattr(run_training, "_load_dataloaders", fake_load_dataloaders)
    monkeypatch.setattr(
        run_training,
        "_build_model_and_adapter",
        lambda model_name, cfg, out_len, tsl_graph: (object(), _FakeTrainer),
    )
    monkeypatch.setattr(run_training, "_try_extract_adj", lambda *a, **k: None)

    run_training._train_one_run(
        "gwn", cfg, h=1, seed=1, r_idx=0, eval_test_loader=None, canonical_scaler=None
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].patience == 100
    assert captured_configs[0].grad_clip == 3.0


def test_train_one_run_finishes_logger_even_when_training_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-training exception must not leave the W&B run open."""
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.use_wandb = True

    class _FakeTrainer:
        def __init__(self, model: Any, config: Any, dataloaders: Any, scaler: Any, logger: Any):
            self.device = torch.device("cpu")
            self.logger = logger

        def train(self) -> dict[str, float]:
            raise RuntimeError("boom")

        def predict_aligned(self, loader: Any, out_len: int) -> tuple[np.ndarray, np.ndarray]:
            raise AssertionError("should not be reached if train() raises")

    class _FakeLogger:
        def __init__(self) -> None:
            self.finished = False

        def watch(self, model: Any) -> None:
            pass

        def finish(self) -> None:
            self.finished = True

    fake_logger = _FakeLogger()

    def fake_load_dataloaders(
        cfg: Any, out_len: int, scaler: Any = None
    ) -> tuple[dict, Any, tuple[None, None]]:
        return {}, None, (None, None)

    monkeypatch.setattr(run_training, "_load_dataloaders", fake_load_dataloaders)
    monkeypatch.setattr(
        run_training,
        "_build_model_and_adapter",
        lambda model_name, cfg, out_len, tsl_graph: (object(), _FakeTrainer),
    )
    monkeypatch.setattr(run_training, "_try_extract_adj", lambda *a, **k: None)
    monkeypatch.setattr("training.logger.make_logger", lambda *a, **k: fake_logger)

    with pytest.raises(RuntimeError, match="boom"):
        run_training._train_one_run(
            "gwn", cfg, h=1, seed=1, r_idx=0, eval_test_loader=None, canonical_scaler=None
        )

    assert fake_logger.finished is True


def _fake_resolve_run(test_count: int, num_nodes: int):
    """Build a fake `_resolve_run` that returns a distinct constant per seed.

    Each run's prediction array is filled with its own seed value, so a test
    can assert no run column is left at the `np.zeros` default — the exact
    symptom of the bug a too-short seed list causes.
    """

    def _resolve_run(
        model_name: str,
        cfg: TrainingExperimentConfig,
        h: int,
        seed: int,
        r_idx: int,
        ckpt_dir: Path,
        completed_runs: dict[str, Any],
        eval_test_loader: Any,
        expected_test_count: int,
        canonical_scaler: Any,
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        run_preds = np.full((test_count, num_nodes), float(seed), dtype=np.float32)
        return run_preds, 1.0, None

    return _resolve_run


def test_collect_predictions_pads_short_seed_list_to_num_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A horizon's seed list saved under a smaller num_runs gets padded, not left short."""
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.num_runs = 3
    test_count = 2
    monkeypatch.setattr(run_training, "_resolve_run", _fake_resolve_run(test_count, cfg.num_nodes))

    # Horizon "1" was previously trained with num_runs=1; horizon "2" is new.
    seeds_log: dict[str, Any] = {"gwn": {"1": [111]}}

    preds_out, _ = run_training._collect_predictions_and_adj(
        "gwn",
        cfg,
        seeds_log,
        completed_runs={},
        eval_test_loader=None,
        test_count=test_count,
        canonical_scaler=None,
    )

    assert preds_out.shape == (test_count, len(cfg.horizons), cfg.num_nodes, cfg.num_runs)
    # No run column may be the np.zeros default — every column was actually trained.
    assert not np.any(np.all(preds_out == 0.0, axis=(0, 1, 2)))
    assert len(seeds_log["gwn"]["1"]) == cfg.num_runs
    assert seeds_log["gwn"]["1"][0] == 111  # the original seed is preserved, not regenerated


def test_collect_predictions_refuses_to_discard_existing_runs(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.num_runs = 2
    seeds_log: dict[str, Any] = {"gwn": {"1": [111, 222, 333, 444, 555], "2": [666, 777]}}

    with pytest.raises(ValueError, match="not discarded"):
        run_training._collect_predictions_and_adj(
            "gwn",
            cfg,
            seeds_log,
            completed_runs={},
            eval_test_loader=None,
            test_count=2,
            canonical_scaler=None,
        )


def test_collect_predictions_rejects_duplicate_seed_identity(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.num_runs = 2
    seeds_log: dict[str, Any] = {"gwn": {"1": [111, 111]}}

    with pytest.raises(ValueError, match="duplicate seeds"):
        run_training._collect_predictions_and_adj(
            "gwn",
            cfg,
            seeds_log,
            completed_runs={},
            eval_test_loader=None,
            test_count=2,
            canonical_scaler=None,
        )


def test_validate_config_rejects_too_small_num_runs_before_any_training(tmp_path: Path) -> None:
    """A num_runs below the recorded seed count fails up front, not mid-sweep.

    Without this, the per-model check inside _collect_predictions_and_adj only
    fires once that model's turn arrives — after ARIMA has been fully refit and
    every earlier model trained.
    """
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.num_runs = 2
    run_training._save_seeds_log(
        {"gwn": {"1": [111, 222, 333], "2": [444, 555]}}, Path(cfg.output_dir)
    )

    with pytest.raises(ValueError, match="not discarded"):
        validate_experiment_config(cfg)


def test_validate_config_accepts_seed_log_shorter_than_num_runs(tmp_path: Path) -> None:
    """Extending an experiment (num_runs 1 -> 3) is the supported path."""
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.num_runs = 3
    run_training._save_seeds_log({"gwn": {"1": [111], "2": [222]}}, Path(cfg.output_dir))

    validate_experiment_config(cfg)  # must not raise


def _synthetic_batches(
    num_batches: int, batch_size: int, in_dim: int, num_nodes: int, in_len: int, horizon: int
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Plain list of (x, y, y_full) tuples — GraphTrainer only ever iterates
    its dataloaders, so a real DataLoader/Dataset isn't needed to exercise it."""
    batches = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, in_dim, num_nodes, in_len)
        y = torch.randn(batch_size, num_nodes, horizon)
        y_full = torch.randn(batch_size, in_dim, num_nodes, horizon)
        batches.append((x, y, y_full))
    return batches


def test_train_tcn_pernode_fits_one_independent_model_per_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_train_tcn_pernode`` trains real per-node TCNs (real TCNModel +
    GraphTrainer, not mocked) against synthetic dataloaders, end to end.

    Unlike the other orchestration tests, this doesn't monkeypatch
    ``_collect_predictions_and_adj``/``_train_one_run`` — only
    ``_load_dataloaders`` is faked (to avoid downloading the real METR-LA
    dataset); everything below that (model construction, training,
    node-sliced prediction) runs for real.
    """
    cfg = _base_config(tmp_path)
    cfg.epochs = 1
    cfg.patience = 1
    cfg.model_overrides["tcn"] = {"epochs": 1, "patience": 1}
    h = cfg.horizons[-1]
    N, C, in_len, B = cfg.num_nodes, cfg.in_dim, cfg.in_len, cfg.batch_size

    train_batches = _synthetic_batches(2, B, C, N, in_len, h)
    val_batches = _synthetic_batches(1, B, C, N, in_len, h)
    eval_batches = _synthetic_batches(1, B, C, N, in_len, h)  # W == B windows

    scaler = StandardScaler().fit(np.random.randn(50, N, C).astype(np.float32))

    def fake_load_dataloaders(
        cfg: TrainingExperimentConfig, out_len: int, scaler: Any = None
    ) -> tuple[dict[str, Any], Any, tuple[Any, Any]]:
        return {"train": train_batches, "val": val_batches}, scaler, (None, None)

    monkeypatch.setattr(run_training, "_load_dataloaders", fake_load_dataloaders)

    run_preds, val_loss, run_adj = run_training._train_tcn_pernode(
        cfg, h=h, seed=123, r_idx=0, eval_test_loader=eval_batches, canonical_scaler=scaler
    )

    assert run_preds.shape == (B, N)  # W == B, one eval batch of size B
    assert np.isfinite(run_preds).all()
    assert run_adj is None  # TCN is graph-free regardless of per-node fitting
    assert np.isfinite(val_loss)

    metrics_path = (
        Path(cfg.output_dir) / "runs" / "tcn" / f"h{h}" / "run_01_seed_123" / "node_metrics.json"
    )
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["horizon"] == h
    assert metrics["seed"] == 123
    for key in ("node_val_loss", "node_mae", "node_mape", "node_rmse"):
        assert len(metrics[key]) == N
        assert all(np.isfinite(v) for v in metrics[key])


def test_train_tcn_pernode_logs_per_node_wandb_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``use_wandb=True``, every node's val_loss/MAE/MAPE/RMSE must reach
    a single per-(horizon, seed) W&B run as both per-node log points and a
    summary table — not just the val_loss-only logging this used to have."""
    import wandb

    cfg = _base_config(tmp_path)
    cfg.epochs = 1
    cfg.patience = 1
    cfg.use_wandb = True
    cfg.model_overrides["tcn"] = {"epochs": 1, "patience": 1}
    h = cfg.horizons[-1]
    N, C, in_len, B = cfg.num_nodes, cfg.in_dim, cfg.in_len, cfg.batch_size

    train_batches = _synthetic_batches(2, B, C, N, in_len, h)
    val_batches = _synthetic_batches(1, B, C, N, in_len, h)
    eval_batches = _synthetic_batches(1, B, C, N, in_len, h)
    scaler = StandardScaler().fit(np.random.randn(50, N, C).astype(np.float32))

    def fake_load_dataloaders(
        cfg: TrainingExperimentConfig, out_len: int, scaler: Any = None
    ) -> tuple[dict[str, Any], Any, tuple[Any, Any]]:
        return {"train": train_batches, "val": val_batches}, scaler, (None, None)

    monkeypatch.setattr(run_training, "_load_dataloaders", fake_load_dataloaders)

    logged: list[dict[str, Any]] = []
    summary_dict: dict[str, Any] = {}

    class _FakeTable:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns
            self.rows: list[tuple[Any, ...]] = []

        def add_data(self, *args: Any) -> None:
            self.rows.append(args)

    class _FakeRun:
        summary = summary_dict

        def log(self, data: dict[str, Any], step: int | None = None) -> None:
            logged.append(data)

        def finish(self) -> None:
            pass

    monkeypatch.setattr(wandb, "init", lambda **kwargs: _FakeRun())
    monkeypatch.setattr(wandb, "Table", _FakeTable)

    run_training._train_tcn_pernode(
        cfg, h=h, seed=99, r_idx=0, eval_test_loader=eval_batches, canonical_scaler=scaler
    )

    node_logs = [d for d in logged if "node/val_loss" in d]
    assert len(node_logs) == N
    assert all({"node/mae", "node/mape", "node/rmse"} <= d.keys() for d in node_logs)

    table_logs = [d["node_metrics_table"] for d in logged if "node_metrics_table" in d]
    assert len(table_logs) == 1
    assert len(table_logs[0].rows) == N

    assert summary_dict.keys() >= {"val_loss_mean", "val_loss_max", "mae_mean", "rmse_mean"}
    assert summary_dict.keys() >= {
        "train_time_s",
        "mem_rss_mb",
        "cpu_percent",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
    }
    system_logs = [d for d in logged if "system/mem_rss_mb" in d]
    assert len(system_logs) == 1
    assert system_logs[0].keys() >= {
        "system/cpu_percent",
        "system/gpu_allocated_mb",
        "system/gpu_reserved_mb",
        "system/gpu_peak_allocated_mb",
        "system/gpu_peak_reserved_mb",
    }
    run_dir = Path(cfg.output_dir) / "runs" / "tcn" / f"h{h}" / "run_01_seed_99"
    disk_events = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        event.get("split") == "system" and "gpu_peak_allocated_mb" in event["metrics"]
        for event in disk_events
    )
    disk_summary = json.loads((run_dir / "summary.json").read_text())
    assert disk_summary.keys() >= {
        "test_mae",
        "mem_rss_mb",
        "cpu_percent",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
    }


def test_train_one_run_dispatches_tcn_to_pernode_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_train_one_run("tcn", ...)`` must go through ``_train_tcn_pernode``,
    not the shared-weight path every other model uses."""
    cfg = _base_config(tmp_path)
    calls: list[tuple] = []

    def fake_train_tcn_pernode(
        cfg: TrainingExperimentConfig,
        h: int,
        seed: int,
        r_idx: int,
        eval_test_loader: Any,
        canonical_scaler: Any,
    ) -> tuple[np.ndarray, float, None]:
        calls.append((h, seed, r_idx))
        return np.zeros((2, cfg.num_nodes), dtype=np.float32), 0.5, None

    monkeypatch.setattr(run_training, "_train_tcn_pernode", fake_train_tcn_pernode)

    run_preds, val_loss, run_adj = run_training._train_one_run(
        "tcn", cfg, h=1, seed=7, r_idx=0, eval_test_loader=[], canonical_scaler=None
    )

    assert calls == [(1, 7, 0)]
    assert run_preds.shape == (2, cfg.num_nodes)
    assert val_loss == 0.5
    assert run_adj is None


def test_resolve_run_isolates_each_seed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.horizons = [1]
    cfg.num_runs = 2
    completed_runs: dict[str, Any] = {}

    monkeypatch.setattr(run_training, "_compute_run_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(
        run_training,
        "_train_one_run",
        lambda model_name, cfg, h, seed, r_idx, eval_test_loader, canonical_scaler: (
            np.full((2, cfg.num_nodes), seed, dtype=np.float32),
            float(seed),
            np.eye(cfg.num_nodes, dtype=np.float32),
        ),
    )

    for run_index, seed in enumerate((11, 22)):
        run_training._resolve_run(
            "gwn",
            cfg,
            h=1,
            seed=seed,
            r_idx=run_index,
            ckpt_dir=Path(cfg.checkpoint_dir),
            completed_runs=completed_runs,
            eval_test_loader=None,
            test_count=2,
            canonical_scaler=None,
        )

    first_dir = Path(cfg.output_dir) / "runs" / "gwn" / "h1" / "run_01_seed_11"
    second_dir = Path(cfg.output_dir) / "runs" / "gwn" / "h1" / "run_02_seed_22"
    assert first_dir != second_dir
    for run_dir, seed in ((first_dir, 11), (second_dir, 22)):
        np.testing.assert_allclose(np.load(run_dir / "predictions.npy"), seed)
        assert (run_dir / "adjacency.npy").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["seed"] == seed
        assert manifest["run_index"] in (1, 2)
        assert manifest["cached"] is False


def test_resolve_run_materializes_existing_legacy_run_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.horizons = [1]
    seed = 111
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True)
    legacy_preds = ckpt_dir / f"gwn_h1_s{seed}_preds.npy"
    legacy_adj = ckpt_dir / f"gwn_h1_s{seed}_adj.npy"
    np.save(legacy_preds, np.ones((2, cfg.num_nodes), dtype=np.float32))
    np.save(legacy_adj, np.eye(cfg.num_nodes, dtype=np.float32))
    completed_runs: dict[str, Any] = {
        "gwn": {
            "1": {
                str(seed): {
                    "val_loss": 0.5,
                    "adj_path": str(legacy_adj),
                    "fingerprint": "fingerprint",
                }
            }
        }
    }

    monkeypatch.setattr(run_training, "_compute_run_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(
        run_training,
        "_train_one_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached first run must not retrain")
        ),
    )

    run_training._resolve_run(
        "gwn",
        cfg,
        h=1,
        seed=seed,
        r_idx=0,
        ckpt_dir=ckpt_dir,
        completed_runs=completed_runs,
        eval_test_loader=None,
        test_count=2,
        canonical_scaler=None,
    )

    run_dir = Path(cfg.output_dir) / "runs" / "gwn" / "h1" / f"run_01_seed_{seed}"
    assert (run_dir / "predictions.npy").exists()
    assert (run_dir / "adjacency.npy").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["cached"] is True
    assert completed_runs["gwn"]["1"][str(seed)]["artifact_dir"] == str(run_dir)


def test_resolve_run_retrains_and_keeps_pointer_when_cached_adjacency_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache promising an adjacency it can no longer deliver must not be accepted.

    Accepting it would return adj=None, and since _save_adjacency_files rewrites
    the model's global adjacency as the mean over whichever horizons survived,
    the probe would silently receive a different graph. Nulling the recorded
    adj_path on the way past would also make the loss unrecoverable.
    """
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.horizons = [1]
    seed = 111
    run_dir = Path(cfg.output_dir) / "runs" / "gwn" / "h1" / f"run_01_seed_{seed}"
    run_dir.mkdir(parents=True)
    np.save(run_dir / "predictions.npy", np.ones((2, cfg.num_nodes), dtype=np.float32))
    deleted_adj = Path(cfg.checkpoint_dir) / f"gwn_h1_s{seed}_adj.npy"  # never written
    completed_runs: dict[str, Any] = {
        "gwn": {
            "1": {
                str(seed): {
                    "val_loss": 0.5,
                    "predictions_path": str(run_dir / "predictions.npy"),
                    "adj_path": str(deleted_adj),
                    "fingerprint": "fingerprint",
                }
            }
        }
    }

    monkeypatch.setattr(run_training, "_compute_run_fingerprint", lambda *args: "fingerprint")
    retrained: list[int] = []

    def _fake_train(model_name, cfg, h, seed, r_idx, eval_test_loader, canonical_scaler):
        retrained.append(seed)
        return (
            np.full((2, cfg.num_nodes), 7.0, dtype=np.float32),
            0.25,
            np.eye(cfg.num_nodes, dtype=np.float32),
        )

    monkeypatch.setattr(run_training, "_train_one_run", _fake_train)

    _preds, _val_loss, run_adj = run_training._resolve_run(
        "gwn",
        cfg,
        h=1,
        seed=seed,
        r_idx=0,
        ckpt_dir=Path(cfg.checkpoint_dir),
        completed_runs=completed_runs,
        eval_test_loader=None,
        test_count=2,
        canonical_scaler=None,
    )

    assert retrained == [seed]  # the incomplete cache was rejected, not accepted
    assert run_adj is not None  # this horizon still contributes a learned graph
    entry = completed_runs["gwn"]["1"][str(seed)]
    assert entry["adj_path"] == str(run_dir / "adjacency.npy")
    assert (run_dir / "adjacency.npy").exists()


def test_resolve_run_keeps_null_adj_path_for_graph_free_cached_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legitimately null adj_path (TCN is graph-free) still resolves from cache."""
    cfg = _base_config(tmp_path, spatial_models=["gwn"])
    cfg.horizons = [1]
    seed = 222
    run_dir = Path(cfg.output_dir) / "runs" / "tcn" / "h1" / f"run_01_seed_{seed}"
    run_dir.mkdir(parents=True)
    np.save(run_dir / "predictions.npy", np.ones((2, cfg.num_nodes), dtype=np.float32))
    completed_runs: dict[str, Any] = {
        "tcn": {
            "1": {
                str(seed): {
                    "val_loss": 0.5,
                    "predictions_path": str(run_dir / "predictions.npy"),
                    "adj_path": None,
                    "fingerprint": "fingerprint",
                }
            }
        }
    }

    monkeypatch.setattr(run_training, "_compute_run_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(
        run_training,
        "_train_one_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a graph-free cached run must not retrain")
        ),
    )

    _preds, _val_loss, run_adj = run_training._resolve_run(
        "tcn",
        cfg,
        h=1,
        seed=seed,
        r_idx=0,
        ckpt_dir=Path(cfg.checkpoint_dir),
        completed_runs=completed_runs,
        eval_test_loader=None,
        test_count=2,
        canonical_scaler=None,
    )

    assert run_adj is None
    assert completed_runs["tcn"]["1"][str(seed)]["adj_path"] is None
