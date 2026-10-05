from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.dataset import load_dataset
from training.adapters import GWNAdapter
from training.config import TrainerConfig
from training.logger import DiskLogger, NoOpLogger, WandbLogger, make_logger
from training.trainer import GraphTrainer


class _FakeGWN(nn.Module):
    def __init__(self, horizon: int, nodes: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.nodes = nodes
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None) -> torch.Tensor:
        return x.new_ones(x.size(0), self.horizon, self.nodes, 1) * self.bias


def test_graph_trainer_smoke_cpu_only(tmp_path) -> None:
    horizon = 3
    nodes = 3
    rng = np.random.default_rng(0)
    data = rng.random((90, nodes, 2)).astype(np.float32)
    path = tmp_path / "data.npz"
    np.savez(path, data=data)
    dataloaders, scaler = load_dataset(
        str(path),
        in_len=4,
        out_len=horizon,
        batch_size=2,
        num_workers=0,
    )
    config = TrainerConfig(
        model_name="smoke-gwn",
        device="cpu",
        epochs=1,
        use_wandb=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        out_len=horizon,
    )
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    trainer = GraphTrainer(
        GWNAdapter(_FakeGWN(horizon=horizon, nodes=nodes), no_edges),
        config,
        dataloaders,
        scaler,
        NoOpLogger(),
    )

    result = trainer.train()
    metrics = trainer.evaluate("test")

    assert torch.device(config.device).type == "cpu"
    assert np.isfinite(result["val_loss"])
    assert np.isfinite(metrics["mae"])
    assert metrics["mae_h"].shape == (horizon,)


def _build_smoke_trainer(tmp_path, **config_kwargs: Any) -> GraphTrainer:
    horizon, nodes = 3, 3
    rng = np.random.default_rng(0)
    data = rng.random((90, nodes, 2)).astype(np.float32)
    path = tmp_path / "data.npz"
    np.savez(path, data=data)
    dataloaders, scaler = load_dataset(str(path), in_len=4, out_len=horizon, batch_size=2)
    config = TrainerConfig(
        model_name="smoke-gwn",
        device="cpu",
        epochs=1,
        use_wandb=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        out_len=horizon,
        **config_kwargs,
    )
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    return GraphTrainer(
        GWNAdapter(_FakeGWN(horizon=horizon, nodes=nodes), no_edges),
        config,
        dataloaders,
        scaler,
        NoOpLogger(),
    )


def test_scheduler_none_keeps_lr_fixed(tmp_path) -> None:
    """scheduler='none' must never decay lr — the regime ported models like
    STAWnet were tuned against in their original codebase, where applying
    this framework's default ReduceLROnPlateau measurably hurt results."""
    trainer = _build_smoke_trainer(tmp_path, scheduler="none", lr=0.001)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    for _ in range(20):
        trainer.scheduler.step()

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)


def test_scheduler_rejects_unknown_value(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        _build_smoke_trainer(tmp_path, scheduler="not-a-real-scheduler")


def test_scheduler_exponential_decays_every_step_unconditionally(tmp_path) -> None:
    """scheduler='exponential' must decay lr *= decay_rate every .step() call,
    regardless of validation loss — matching GWN-v2's original
    LambdaLR(lr_lambda=lambda epoch: lr_decay_rate**epoch), not a plateau-
    triggered decay."""
    decay_rate = 0.9
    trainer = _build_smoke_trainer(
        tmp_path, scheduler="exponential", scheduler_kwargs={"decay_rate": decay_rate}, lr=0.001
    )
    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    for _ in range(3):
        trainer.scheduler.step()

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * decay_rate**3)


def test_scheduler_exponential_defaults_decay_rate_to_0_97(tmp_path) -> None:
    trainer = _build_smoke_trainer(tmp_path, scheduler="exponential", lr=0.001)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    trainer.scheduler.step()

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.97)


def test_wandb_disabled_does_not_import_wandb(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)

    logger = make_logger(TrainerConfig(use_wandb=False))

    assert isinstance(logger, NoOpLogger)
    assert "wandb" not in sys.modules


def test_disk_logger_persists_epoch_test_and_summary_metrics(tmp_path) -> None:
    metrics_dir = tmp_path / "run"
    logger = make_logger(TrainerConfig(use_wandb=False, metrics_dir=str(metrics_dir)))

    logger.log_epoch(
        "system",
        {
            "duration_s": 1.2,
            "mem_rss_mb": 128.0,
            "cpu_percent": 75.0,
            "gpu_peak_allocated_mb": 64.0,
        },
        epoch=1,
    )
    logger.log_test_metrics(1.0, 2.0, 3.0, horizon=12)
    logger.log_summary({"val_loss": 0.5, "seed": 42})
    logger.finish()

    assert isinstance(logger, DiskLogger)
    events = [json.loads(line) for line in (metrics_dir / "metrics.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["run_started", "epoch", "test", "summary"]
    assert events[1]["metrics"]["mem_rss_mb"] == 128.0
    assert json.loads((metrics_dir / "summary.json").read_text()) == {
        "attempt": 1,
        "val_loss": 0.5,
        "seed": 42,
    }


def test_disk_logger_separates_attempts_reusing_one_run_directory(tmp_path) -> None:
    """A retrain or crash-resume reopens the run dir; the two attempts must stay apart.

    metrics.jsonl is append-only by design (a mid-run resume continues the same
    logical run), so each event carries an attempt number instead — otherwise two
    epoch-0..N curves read as one. summary.json must not inherit keys from the
    attempt that was discarded.
    """
    metrics_dir = tmp_path / "run_01_seed_111"

    first = make_logger(TrainerConfig(use_wandb=False, metrics_dir=str(metrics_dir)))
    for epoch in range(3):
        first.log_epoch("train", {"loss": 10.0 - epoch}, epoch)
    first.log_summary({"val_loss": 7.0, "test_mae": 7.5, "params": 100})
    first.finish()

    # Same directory, e.g. after the run fingerprint changed and it retrained.
    second = make_logger(TrainerConfig(use_wandb=False, metrics_dir=str(metrics_dir)))
    for epoch in range(2):
        second.log_epoch("train", {"loss": 3.0 - epoch}, epoch)
    second.log_summary({"val_loss": 2.0, "test_mae": 2.1})
    second.finish()

    events = [json.loads(line) for line in (metrics_dir / "metrics.jsonl").read_text().splitlines()]
    epochs = [(e["attempt"], e["step"]) for e in events if e["type"] == "epoch"]
    assert epochs == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    assert [e["attempt"] for e in events if e["type"] == "run_started"] == [1, 2]

    # params=100 came from attempt 1 only — it must not survive into attempt 2's summary.
    assert json.loads((metrics_dir / "summary.json").read_text()) == {
        "attempt": 2,
        "val_loss": 2.0,
        "test_mae": 2.1,
    }


def test_disk_logger_merges_repeated_summary_calls_within_one_attempt(tmp_path) -> None:
    """Merging is still the behaviour inside a single attempt."""
    metrics_dir = tmp_path / "run"
    logger = make_logger(TrainerConfig(use_wandb=False, metrics_dir=str(metrics_dir)))

    logger.log_summary({"val_loss": 0.5})
    logger.log_summary({"test_mae": 1.5})
    logger.finish()

    assert json.loads((metrics_dir / "summary.json").read_text()) == {
        "attempt": 1,
        "val_loss": 0.5,
        "test_mae": 1.5,
    }


def test_resume_checkpoint_round_trips_under_default_torch_load(tmp_path) -> None:
    """Resume checkpoints embed RNG state (numpy ndarrays via
    np.random.get_state()), which torch>=2.6's weights_only=True default
    can't unpickle (numpy.core.multiarray._reconstruct isn't an allowed
    global). _load_resume_checkpoint must pass weights_only=False since
    these are checkpoints the framework itself wrote, not untrusted files."""
    trainer = _build_smoke_trainer(tmp_path)
    trainer._save_resume_checkpoint(epoch=1)

    start_epoch = trainer._load_resume_checkpoint()

    assert start_epoch == 2


def test_corrupt_resume_checkpoint_falls_back_to_epoch_1(tmp_path) -> None:
    """A resume checkpoint is a best-effort speedup, not a correctness
    requirement — e.g. one written by an incompatible torch/numpy version,
    or truncated by a crash mid-write, must not abort the whole run. It
    should be discarded and training should fall back to epoch 1."""
    trainer = _build_smoke_trainer(tmp_path)
    trainer._save_resume_checkpoint(epoch=1)
    with open(trainer._resume_ckpt_path, "wb") as f:
        f.write(b"not a valid checkpoint")

    start_epoch = trainer._load_resume_checkpoint()

    assert start_epoch == 1
    assert not os.path.exists(trainer._resume_ckpt_path)


def test_wandb_logger_rejects_missing_dataset_name_and_horizon(monkeypatch, tmp_path) -> None:
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: None,
        log=lambda *args, **kwargs: None,
        watch=lambda *args, **kwargs: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.env._REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="dataset_name"):
        WandbLogger(
            TrainerConfig(use_wandb=True, model_name="missing_fields", wandb_mode="offline")
        )


def test_wandb_offline_mode_is_passed_to_wandb(monkeypatch, tmp_path) -> None:
    calls: dict[str, Any] = {}

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("init", kwargs),
        log=lambda *args, **kwargs: None,
        watch=lambda *args, **kwargs: None,
        finish=lambda: calls.setdefault("finish", True),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.chdir(tmp_path)
    # Isolate from this repo's own (gitignored, developer-local) .env file —
    # without this, get_env_or_dotenv()'s repo-root fallback would still find
    # the real WANDB_ENTITY on any machine that has one checked out.
    monkeypatch.setattr("training.env._REPO_ROOT", tmp_path)

    logger = make_logger(
        TrainerConfig(
            use_wandb=True,
            model_name="offline",
            dataset_name="METR-LA",
            horizon=12,
            wandb_mode="offline",
        )
    )
    logger.finish()

    assert isinstance(logger, WandbLogger)
    assert "entity" not in calls["init"]
    assert calls["init"]["mode"] == "offline"
    assert calls["finish"] is True


def test_wandb_entity_defaults_to_environment(monkeypatch, tmp_path) -> None:
    calls: dict[str, Any] = {}

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("init", kwargs),
        log=lambda *args, **kwargs: None,
        watch=lambda *args, **kwargs: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_ENTITY", "team-from-env")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.env._REPO_ROOT", tmp_path)

    logger = make_logger(
        TrainerConfig(
            use_wandb=True,
            model_name="env",
            dataset_name="METR-LA",
            horizon=12,
            wandb_mode="offline",
        )
    )

    assert isinstance(logger, WandbLogger)
    assert calls["init"]["entity"] == "team-from-env"


def test_wandb_entity_defaults_to_dotenv(monkeypatch, tmp_path) -> None:
    calls: dict[str, Any] = {}

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("init", kwargs),
        log=lambda *args, **kwargs: None,
        watch=lambda *args, **kwargs: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.env._REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("WANDB_ENTITY=team-from-dotenv\n")

    logger = make_logger(
        TrainerConfig(
            use_wandb=True,
            model_name="dotenv",
            dataset_name="METR-LA",
            horizon=12,
            wandb_mode="offline",
        )
    )

    assert isinstance(logger, WandbLogger)
    assert calls["init"]["entity"] == "team-from-dotenv"
