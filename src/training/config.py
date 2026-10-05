"""Trainer configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    """Unified configuration for graph-model and univariate trainers."""

    # --- Data ---
    in_len: int = 12
    out_len: int = 12
    batch_size: int = 64
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    num_workers: int = 0
    dataset_name: str = ""  # e.g. "METR-LA" — required when use_wandb=True
    horizon: int = 0  # forecast horizon for this run (== out_len); required when use_wandb=True

    # --- Optimisation ---
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 15  # early stopping patience (val loss)
    seed: int = 42

    warmup_epochs: int = 0  # ramp task_level from 1→out_len over warmup_epochs epochs; 0 = disabled

    # --- W&B ---
    use_wandb: bool = False
    wandb_entity: str | None = None  # if empty, falls back to WANDB_ENTITY from env/.env
    wandb_project: str = "stgnn-framework"
    wandb_run_name: str = ""  # auto-generated if empty
    wandb_mode: str = "online"  # "offline" for tests / no-network runs
    metrics_dir: str | None = None  # optional per-run JSONL + summary output directory
    run_index: int = 0  # 1-based independent-run index; 0 when not applicable

    # --- Misc ---
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    null_val: float = 0.0  # mask value for metric computation

    # --- Model name (for logging) ---
    model_name: str = "unknown"

    # --- Adjacency transition type ---
    adj_type: str = "doubletransition"  # "doubletransition" | "laplacian" | "single"

    # --- LR scheduler ---
    scheduler: str = "plateau"  # "plateau" | "multistep"
    scheduler_kwargs: dict = field(default_factory=dict)

    # --- Extra model kwargs passed through to factory functions ---
    model_kwargs: dict = field(default_factory=dict)
