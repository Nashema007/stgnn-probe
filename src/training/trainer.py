"""Shared graph-model trainer for STGNN benchmarking experiments."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from data.scaler import StandardScaler
from evaluation.metrics import masked_mae, metric_global, metric_per_horizon

from .config import TrainerConfig
from .device import resolve_device
from .logger import DiskLogger, NoOpLogger, WandbLogger
from .resource_monitor import ResourceSampler, gpu_memory_metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphTrainer:
    """Train and evaluate any graph-based STGNN wrapped in a GraphModelAdapter.

    The trainer assumes the dataloader yields 3-tuples:
        x      : (B, C, N, T)
        y      : (B, N, H)  — channel-0 target for loss/metrics
        y_full : (B, C, N, H) — all channels (unused by most adapters)

    Parameters
    ----------
    model:
        A ``GraphModelAdapter`` instance (already wraps the underlying model).
    config:
        Trainer configuration.
    dataloaders:
        Dict with keys ``"train"``, ``"val"``, ``"test"``.
    scaler:
        Fitted ``StandardScaler`` — used to inverse-transform predictions
        before computing metrics.
    logger:
        ``WandbLogger`` or ``NoOpLogger``.
    node_indices:
        Original node indices (into the ``N`` axis ``scaler`` was fit on)
        that ``dataloaders``' node axis corresponds to. Required whenever
        the dataloaders have already been sliced down to a subset of
        nodes (e.g. one node at a time, as TCN's per-node training does)
        — otherwise inverse-transforming would broadcast node 0's
        mean/std against every node's predictions. Defaults to ``None``,
        meaning the dataloaders carry every node in ``scaler``'s original
        order (the common case for all spatial models).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        dataloaders: dict[str, DataLoader],
        scaler: StandardScaler,
        logger: WandbLogger | DiskLogger | NoOpLogger,
        node_indices: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.scaler = scaler
        self.logger = logger
        self.node_indices = node_indices

        self.device = resolve_device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler: (
            torch.optim.lr_scheduler.MultiStepLR
            | torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.LambdaLR
        )
        if config.scheduler == "multistep":
            kw = {**{"milestones": [50, 70, 100], "gamma": 0.1}, **config.scheduler_kwargs}
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **kw)
        elif config.scheduler == "plateau":
            kw = {**{"mode": "min", "factor": 0.3, "patience": 10}, **config.scheduler_kwargs}
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kw)
        elif config.scheduler == "none":
            # No-op: keeps lr fixed for the whole run. Some ported
            # architectures (e.g. STAWnet) were tuned in their original
            # paper/codebase against a constant learning rate for the full
            # training run — decaying lr is a regime the model was never
            # validated against, and silently applying one elsewhere's
            # default can measurably hurt those models' results.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _epoch: 1.0
            )
        elif config.scheduler == "exponential":
            # Smooth per-epoch decay (lr *= decay_rate every epoch),
            # unconditional on validation plateaus — e.g. GWN-v2's original
            # codebase (sshleifer/Graph-WaveNet) uses
            # LambdaLR(lr_lambda=lambda epoch: decay_rate**epoch), not
            # ReduceLROnPlateau.
            decay_rate = config.scheduler_kwargs.get("decay_rate", 0.97)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: decay_rate**epoch
            )
        else:
            raise ValueError(
                f"Unknown scheduler {config.scheduler!r}; expected 'plateau', 'multistep', "
                "'exponential', or 'none'."
            )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self._ckpt_path = os.path.join(config.checkpoint_dir, f"{config.model_name}_best.pt")
        self._resume_ckpt_path = os.path.join(
            config.checkpoint_dir, f"{config.model_name}_resume.pt"
        )

        self._batches_seen = 0
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _inverse_transform(self, data: Tensor | np.ndarray) -> Tensor | np.ndarray:
        return self.scaler.inverse_transform(data, node_indices=self.node_indices)

    def train(self) -> dict[str, float]:
        """Run the full train + val loop with early stopping.

        Returns the best validation metrics.
        """
        _seed_everything(self.config.seed)
        self.logger.watch(self.model)
        epoch_sampler = ResourceSampler()

        start_epoch = self._load_resume_checkpoint()

        for epoch in range(start_epoch, self.config.epochs + 1):
            task_level = self._curriculum_task_level(epoch)

            t0 = time.time()
            train_loss = self._train_epoch(epoch, task_level)
            val_loss, val_mae, val_mape, val_rmse = self._val_epoch(epoch)
            elapsed = time.time() - t0
            mem_mb, cpu_pct = epoch_sampler.sample()
            gpu_metrics = gpu_memory_metrics(self.device)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            self.logger.log_epoch("train", {"loss": train_loss}, epoch)
            self.logger.log_epoch(
                "val",
                {"loss": val_loss, "MAE": val_mae, "MAPE": val_mape, "RMSE": val_rmse},
                epoch,
            )
            self.logger.log_epoch(
                "system",
                {
                    "duration_s": elapsed,
                    "mem_rss_mb": mem_mb,
                    "cpu_percent": cpu_pct,
                    **gpu_metrics,
                },
                epoch,
            )

            print(
                f"[{config_name(self.config)}] epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f}  val_mae={val_mae:.4f}  "
                f"val_rmse={val_rmse:.4f}  [{elapsed:.1f}s  "
                f"mem={mem_mb:.0f}MB  cpu={cpu_pct:.0f}%]",
                flush=True,
            )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}.", flush=True)
                    self._delete_resume_checkpoint()
                    break

            self._save_resume_checkpoint(epoch)
        else:
            self._delete_resume_checkpoint()

        self._load_best_checkpoint()
        return {"val_loss": self._best_val_loss}

    def predict(self, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        """Return per-window, per-horizon predictions and ground truth.

        Returns
        -------
        preds : (W, N, H) — inverse-transformed, one row per test window
        trues : (W, N, H) — same
        """
        self.model.eval()
        all_pred: list[Tensor] = []
        all_true: list[Tensor] = []

        with torch.no_grad():
            for batch in self.dataloaders[split]:
                x, y, y_full = _to_device(batch, self.device)
                pred = self.model(x, y_full=y_full, task_level=self.config.out_len)
                all_pred.append(_as_tensor(self._inverse_transform(pred), self.device))
                all_true.append(_as_tensor(self._inverse_transform(y), self.device))

        preds = torch.cat(all_pred, dim=0).cpu().numpy()  # (W, N, H)
        trues = torch.cat(all_true, dim=0).cpu().numpy()
        return preds, trues

    def predict_aligned(self, loader: Any, out_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Predict on an externally supplied loader at a shared test-window origin set.

        Unlike ``predict()`` (which iterates this trainer's own test split and
        averages away the window axis), this evaluates the model on windows
        supplied by ``loader`` — typically a canonical loader shared across
        every model/horizon so Lens 0 can compute metrics from genuinely
        paired per-window samples instead of values pre-averaged over windows.

        ``loader`` yields ``(x, y, y_full)`` batches built at some horizon
        ``H >= out_len``; only the first ``out_len`` target steps are used,
        so the same loader can be reused across models trained at different
        horizons as long as ``out_len`` for each call is `<=` the loader's own
        horizon.

        Returns
        -------
        preds : (W, N) — inverse-transformed prediction at the final
            (``out_len``-th) horizon step, one row per window in ``loader``.
        trues : (W, N) — paired ground truth at the same windows/horizon.
        """
        self.model.eval()
        all_pred: list[Tensor] = []
        all_true: list[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                x, _y, y_full = _to_device(batch, self.device)
                y_full_h = y_full[..., :out_len]
                y_h = y_full_h[:, 0]
                pred = self.model(x, y_full=y_full_h, task_level=out_len)
                pred_last = pred[:, :, -1:]
                true_last = y_h[:, :, -1:]
                all_pred.append(_as_tensor(self._inverse_transform(pred_last), self.device))
                all_true.append(_as_tensor(self._inverse_transform(true_last), self.device))

        preds = torch.cat(all_pred, dim=0).squeeze(-1).cpu().numpy()  # (W, N)
        trues = torch.cat(all_true, dim=0).squeeze(-1).cpu().numpy()
        return preds, trues

    def evaluate(self, split: str = "test") -> dict:
        """Evaluate on *split* and return global + per-horizon metrics.

        Returns a dict with keys:
            ``mae``, ``mape``, ``rmse`` — global scalars
            ``mae_h``, ``mape_h``, ``rmse_h`` — per-horizon numpy arrays
        """
        self.model.eval()
        loader = self.dataloaders[split]

        all_pred: list[Tensor] = []
        all_true: list[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                x, y, y_full = _to_device(batch, self.device)
                pred = self.model(x, y_full=y_full, task_level=self.config.out_len)
                pred_real = self._inverse_transform(pred)
                y_real = self._inverse_transform(y)
                all_pred.append(
                    pred_real if isinstance(pred_real, Tensor) else torch.tensor(pred_real)
                )
                all_true.append(y_real if isinstance(y_real, Tensor) else torch.tensor(y_real))

        preds = torch.cat(all_pred, dim=0)  # (N_total, N_nodes, H)
        trues = torch.cat(all_true, dim=0)

        mae, mape, rmse = metric_global(preds, trues, self.config.null_val)
        per_h = metric_per_horizon(preds, trues, self.config.null_val)

        self.logger.log_test_horizons(per_h["mae"], per_h["mape"], per_h["rmse"])

        return {
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "mae_h": per_h["mae"],
            "mape_h": per_h["mape"],
            "rmse_h": per_h["rmse"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _curriculum_task_level(self, epoch: int) -> int:
        """Compute active forecast horizon for curriculum warmup."""
        if self.config.warmup_epochs <= 0:
            return self.config.out_len
        progress = min(epoch / self.config.warmup_epochs, 1.0)
        return max(1, int(round(progress * self.config.out_len)))

    def _train_epoch(self, epoch: int, task_level: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.dataloaders["train"]:
            x, y, y_full = _to_device(batch, self.device)
            self._batches_seen += 1

            self.optimizer.zero_grad()
            pred = self.model(
                x,
                y_full=y_full,
                batches_seen=self._batches_seen,
                task_level=task_level,
            )  # (B, N, task_level)

            # Align y to task_level for curriculum warmup
            y_tl = y[:, :, :task_level]

            pred_real = self._inverse_transform(pred)
            y_real = self._inverse_transform(y_tl)

            if not isinstance(pred_real, Tensor):
                pred_real = torch.tensor(pred_real, device=self.device)
            if not isinstance(y_real, Tensor):
                y_real = torch.tensor(y_real, device=self.device)

            loss = masked_mae(pred_real, y_real, self.config.null_val)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        self.model.eval()
        preds_list: list[Tensor] = []
        trues_list: list[Tensor] = []

        with torch.no_grad():
            for batch in self.dataloaders["val"]:
                x, y, y_full = _to_device(batch, self.device)
                pred = self.model(x, y_full=y_full, task_level=self.config.out_len)
                pred_real = self._inverse_transform(pred)
                y_real = self._inverse_transform(y)
                preds_list.append(_as_tensor(pred_real, self.device))
                trues_list.append(_as_tensor(y_real, self.device))

        preds = torch.cat(preds_list, dim=0)
        trues = torch.cat(trues_list, dim=0)
        mae, mape, rmse = metric_global(preds, trues, self.config.null_val)
        val_loss = masked_mae(preds, trues, self.config.null_val).item()
        return val_loss, mae, mape, rmse

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        torch.save(
            {"epoch": epoch, "val_loss": val_loss, "state_dict": self.model.state_dict()},
            self._ckpt_path,
        )

    def _load_best_checkpoint(self) -> None:
        if os.path.exists(self._ckpt_path):
            ckpt = torch.load(self._ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"])

    def _save_resume_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "batches_seen": self._batches_seen,
            "best_val_loss": self._best_val_loss,
            "patience_counter": self._patience_counter,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            ckpt["rng_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(ckpt, self._resume_ckpt_path)

    def _load_resume_checkpoint(self) -> int:
        """Load resume state if available; return the epoch to start from.

        A resume checkpoint is a best-effort optimisation (skip re-training
        epochs already done), not a correctness requirement — a fresh start
        from epoch 1 is always safe. So a corrupt/incompatible file (e.g.
        written by an older torch/numpy version, or truncated by a crash
        mid-write) must never abort the run; it's deleted and training
        starts over instead.
        """
        if not os.path.exists(self._resume_ckpt_path):
            return 1
        try:
            ckpt = torch.load(self._resume_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self._batches_seen = ckpt["batches_seen"]
            self._best_val_loss = ckpt["best_val_loss"]
            self._patience_counter = ckpt["patience_counter"]
            random.setstate(ckpt["rng_python"])
            np.random.set_state(ckpt["rng_numpy"])
            torch.set_rng_state(ckpt["rng_torch"])
            if torch.cuda.is_available() and "rng_cuda" in ckpt:
                torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception as exc:
            print(
                f"[{config_name(self.config)}] Resume checkpoint at "
                f"{self._resume_ckpt_path} is corrupt/incompatible ({exc!r}) — "
                "deleting it and starting this run from epoch 1."
            )
            os.remove(self._resume_ckpt_path)
            return 1
        start_epoch = ckpt["epoch"] + 1
        print(f"[{config_name(self.config)}] Resuming from epoch {start_epoch}.")
        return start_epoch

    def _delete_resume_checkpoint(self) -> None:
        if os.path.exists(self._resume_ckpt_path):
            os.remove(self._resume_ckpt_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def config_name(config: TrainerConfig) -> str:
    return config.model_name or "model"


def _to_device(
    batch: tuple[Tensor, Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    x, y, y_full = batch
    return x.to(device), y.to(device), y_full.to(device)


def _as_tensor(data: Tensor | np.ndarray, device: torch.device) -> Tensor:
    if isinstance(data, Tensor):
        return data.to(device)
    return torch.from_numpy(data).to(device)
