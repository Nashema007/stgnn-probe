"""Experiment logger wrappers for W&B and no-op fallback."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch.nn as nn

from .config import TrainerConfig
from .env import resolve_wandb_entity


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DiskLogger:
    """Append run metrics to JSONL and keep a mergeable summary on disk.

    A run directory outlives any single training attempt: a crash-resume, or a
    retrain after the run fingerprint changed, reopens the same directory. Every
    event therefore carries a 1-based ``attempt`` number (counted from the
    ``run_started`` markers already in ``metrics.jsonl``) so a reader can split
    the file back into per-attempt series instead of reading two concatenated
    epoch-0..N curves as one. ``summary.json`` is merged only within an attempt
    — the first ``log_summary`` of a new attempt replaces the file rather than
    inheriting keys from the attempt that was discarded.
    """

    def __init__(self, metrics_dir: str | Path) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.metrics_dir / "metrics.jsonl"
        self._summary_path = self.metrics_dir / "summary.json"
        self._attempt = self._count_previous_attempts() + 1
        self._summary_started = False
        self._append({"type": "run_started"})

    def _count_previous_attempts(self) -> int:
        """Count ``run_started`` markers already written to this run directory."""
        if not self._events_path.exists():
            return 0
        attempts = 0
        with open(self._events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A partially flushed final line from a killed process must
                    # not stop us from numbering this attempt.
                    continue
                if isinstance(event, dict) and event.get("type") == "run_started":
                    attempts += 1
        return attempts

    def _append(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "attempt": self._attempt,
            **event,
        }
        with open(self._events_path, "a") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")

    def log_epoch(self, split: str, metrics: dict[str, float], epoch: int) -> None:
        self._append({"type": "epoch", "split": split, "step": epoch, "metrics": metrics})

    def log_test_horizons(
        self,
        mae_h: np.ndarray,
        mape_h: np.ndarray,
        rmse_h: np.ndarray,
    ) -> None:
        for h, (mae, mape, rmse) in enumerate(zip(mae_h, mape_h, rmse_h, strict=False)):
            self._append(
                {
                    "type": "test_horizon",
                    "horizon": h + 1,
                    "metrics": {"MAE": mae, "MAPE": mape, "RMSE": rmse},
                }
            )

    def log_test_metrics(self, mae: float, mape: float, rmse: float, horizon: int) -> None:
        self._append(
            {
                "type": "test",
                "horizon": horizon,
                "metrics": {"MAE": mae, "MAPE": mape, "RMSE": rmse},
            }
        )

    def log_raw(self, metrics: dict[str, Any], step: int | None = None) -> None:
        event: dict[str, Any] = {"type": "metrics", "metrics": metrics}
        if step is not None:
            event["step"] = step
        self._append(event)

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        event: dict[str, Any] = {"type": "metric_definition", "name": name}
        if step_metric is not None:
            event["step_metric"] = step_metric
        self._append(event)

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        self._append({"type": "table", "key": key, "columns": columns, "rows": rows})

    def log_summary(self, metrics: dict[str, Any]) -> None:
        summary: dict[str, Any] = {}
        if self._summary_started and self._summary_path.exists():
            # Merge within this attempt only — a summary left by a previous
            # attempt describes a run that was discarded, so inheriting its
            # keys would mix two runs' numbers in one file.
            with open(self._summary_path) as f:
                summary = json.load(f)
        summary.update({"attempt": self._attempt, **metrics})
        self._summary_started = True
        tmp_path = self._summary_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        tmp_path.replace(self._summary_path)
        self._append({"type": "summary", "metrics": metrics})

    def watch(self, model: nn.Module) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger:
    """Write every event to W&B and, when configured, the per-run disk log."""

    def __init__(self, config: TrainerConfig, run_name: str = "") -> None:
        import wandb  # lazy import so wandb is optional at import time

        missing = [
            field
            for field, value in (
                ("dataset_name", config.dataset_name),
                ("horizon", config.horizon),
                ("model_name", config.model_name),
            )
            if not value or value == "unknown"
        ]
        if missing:
            raise ValueError(
                f"TrainerConfig is missing required W&B fields {missing!r} — "
                "set them before enabling use_wandb so every run is logged "
                "with a real dataset/horizon/model_name instead of a blank value."
            )

        run_name = run_name or config.wandb_run_name or config.model_name
        init_kwargs: dict[str, Any] = dict(
            project=config.wandb_project,
            name=run_name or None,
            config=vars(config) if hasattr(config, "__dict__") else {},
            mode=cast(Any, config.wandb_mode),
        )
        entity = resolve_wandb_entity(config.wandb_entity)
        if entity:
            init_kwargs["entity"] = entity
        self._disk = DiskLogger(config.metrics_dir) if config.metrics_dir else None
        self._run = wandb.init(**init_kwargs)
        self._wandb = wandb
        self.define_metric("epoch")
        for namespace in ("train/*", "val/*", "system/*"):
            self.define_metric(namespace, step_metric="epoch")

    def _log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        log = getattr(self._run, "log", None)
        if log is None:
            log = self._wandb.log
        if step is None:
            log(metrics)
        else:
            log(metrics, step=step)

    def log_epoch(self, split: str, metrics: dict[str, float], epoch: int) -> None:
        self._log({"epoch": epoch, **{f"{split}/{k}": v for k, v in metrics.items()}})
        if self._disk is not None:
            self._disk.log_epoch(split, metrics, epoch)

    def log_test_horizons(
        self,
        mae_h: np.ndarray,
        mape_h: np.ndarray,
        rmse_h: np.ndarray,
    ) -> None:
        for h, (mae, mape, rmse) in enumerate(zip(mae_h, mape_h, rmse_h, strict=False)):
            self._log(
                {
                    f"test/MAE_h{h + 1}": mae,
                    f"test/MAPE_h{h + 1}": mape,
                    f"test/RMSE_h{h + 1}": rmse,
                }
            )
        if self._disk is not None:
            self._disk.log_test_horizons(mae_h, mape_h, rmse_h)

    def log_test_metrics(self, mae: float, mape: float, rmse: float, horizon: int) -> None:
        """Log scalar test-set metrics (canonical test windows, final horizon step).

        Each call corresponds to one (model, horizon, seed) run, so both a
        flat key (for this run's own charts) and an ``_h{horizon}`` key
        (so runs at different horizons stay distinguishable when compared
        across a W&B project) are logged.
        """
        self._log(
            {
                "test/MAE": mae,
                "test/MAPE": mape,
                "test/RMSE": rmse,
                f"test/MAE_h{horizon}": mae,
                f"test/MAPE_h{horizon}": mape,
                f"test/RMSE_h{horizon}": rmse,
            }
        )
        if self._disk is not None:
            self._disk.log_test_metrics(mae, mape, rmse, horizon)

    def log_raw(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._log(metrics, step=step)
        if self._disk is not None:
            self._disk.log_raw(metrics, step=step)

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        define_metric = getattr(self._run, "define_metric", None)
        if define_metric is not None:
            if step_metric is None:
                define_metric(name)
            else:
                define_metric(name, step_metric=step_metric)
        if self._disk is not None:
            self._disk.define_metric(name, step_metric=step_metric)

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        wandb_columns: list[str | int] = list(columns)
        table = self._wandb.Table(columns=wandb_columns)
        for row in rows:
            table.add_data(*row)
        self._log({key: table})
        if self._disk is not None:
            self._disk.log_table(key, columns, rows)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        summary = getattr(self._run, "summary", None)
        if summary is not None:
            for key, value in metrics.items():
                summary[key] = value
        self._log({f"summary/{key}": value for key, value in metrics.items()})
        if self._disk is not None:
            self._disk.log_summary(metrics)

    def watch(self, model: nn.Module) -> None:
        self._wandb.watch(model, log="gradients", log_freq=100)

    def finish(self) -> None:
        finish = getattr(self._run, "finish", None)
        if finish is not None:
            finish()
        else:
            self._wandb.finish()
        if self._disk is not None:
            self._disk.finish()


class NoOpLogger:
    """Drop-in replacement when ``use_wandb=False``."""

    def log_epoch(self, split: str, metrics: dict[str, float], epoch: int) -> None:
        pass

    def log_test_horizons(
        self,
        mae_h: np.ndarray,
        mape_h: np.ndarray,
        rmse_h: np.ndarray,
    ) -> None:
        pass

    def log_test_metrics(self, mae: float, mape: float, rmse: float, horizon: int) -> None:
        pass

    def log_raw(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        pass

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        pass

    def log_summary(self, metrics: dict[str, Any]) -> None:
        pass

    def watch(self, model: nn.Module) -> None:
        pass

    def finish(self) -> None:
        pass


def make_logger(config: TrainerConfig, run_name: str = "") -> WandbLogger | DiskLogger | NoOpLogger:
    """Return a dual W&B/disk logger, disk-only logger, or no-op logger."""
    if config.use_wandb:
        return WandbLogger(config, run_name=run_name)
    if config.metrics_dir:
        return DiskLogger(config.metrics_dir)
    return NoOpLogger()
