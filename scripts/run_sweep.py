"""W&B hyperparameter sweep entry point for STGNN models.

Creates a W&B sweep from a sweep config YAML, then launches a sweep agent that
trains one (model, horizon, seed) job per call, reading hyperparameter values
from wandb.config.

Usage
-----
# Create sweep and run 30 agents:
python scripts/run_sweep.py \\
    --config scripts/configs/metr_la_training.yaml \\
    --sweep  scripts/configs/sweeps/gwn_sweep.yaml \\
    --model  gwn \\
    --horizon 12 \\
    --count  30

# Resume an existing sweep (skip sweep creation):
python scripts/run_sweep.py \\
    --config scripts/configs/metr_la_training.yaml \\
    --sweep  scripts/configs/sweeps/gwn_sweep.yaml \\
    --model  gwn --horizon 12 \\
    --sweep-id <existing-sweep-id> \\
    --count  10
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _build_sweep_pipeline(cfg: Any, horizon: int) -> Any:
    """Build the sweep loader with the experiment's exact temporal split."""
    from data.tsl_pipeline import build_tsl_pipeline

    return build_tsl_pipeline(
        cfg.dataset_name,
        in_len=cfg.in_len,
        out_len=horizon,
        batch_size=cfg.batch_size,
        val_len=cfg.val_len,
        test_len=cfg.test_len,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a W&B hyperparameter sweep for an STGNN model.")
    p.add_argument("--config", required=True, help="Path to training experiment YAML config.")
    p.add_argument("--sweep", required=True, help="Path to W&B sweep config YAML.")
    p.add_argument("--model", required=True, help="Model name (gwn, staeformer, etc.).")
    p.add_argument("--horizon", type=int, default=12, help="Forecast horizon to optimise for.")
    p.add_argument("--count", type=int, default=20, help="Number of sweep runs to execute.")
    p.add_argument(
        "--sweep-id",
        default=None,
        help="Resume an existing W&B sweep instead of creating a new one.",
    )
    p.add_argument(
        "--project",
        default=None,
        help="W&B project name. Overrides wandb_project in TrainerConfig.",
    )
    p.add_argument(
        "--entity",
        default=None,
        help=(
            "W&B entity/team where runs and sweeps are logged. "
            "Defaults to WANDB_ENTITY from env/.env."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    import wandb

    from training.config import TrainerConfig
    from training.env import resolve_wandb_entity
    from training.logger import WandbLogger

    args = build_parser().parse_args(argv)

    # Load experiment + sweep configs
    from run_training import (
        _build_model_and_adapter,
        load_experiment_config,
        validate_experiment_config,
    )

    cfg = load_experiment_config(args.config)
    validate_experiment_config(cfg, model_filter=args.model)

    with open(args.sweep) as f:
        sweep_cfg: dict = yaml.safe_load(f)

    project = args.project or "stgnn-framework"
    entity = resolve_wandb_entity(args.entity)
    wandb_target_kwargs = {"project": project}
    if entity:
        wandb_target_kwargs["entity"] = entity

    # Create or reuse sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming sweep {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_cfg, **wandb_target_kwargs)
        print(f"Created sweep {sweep_id}")

    model_name = args.model
    horizon = args.horizon

    def _sweep_train() -> None:
        run = wandb.init(**wandb_target_kwargs)
        wc = dict(wandb.config)

        # Merge sweep hyperparams into model_overrides for this run
        exp_cfg = copy.deepcopy(cfg)
        base_overrides = dict(exp_cfg.model_overrides.get(model_name, {}))
        base_overrides.update(wc)
        exp_cfg.model_overrides[model_name] = base_overrides

        seed = random.randint(1, 99_999)
        ckpt_name = f"{model_name}_sweep_{run.id}_h{horizon}_s{seed}"
        overrides = exp_cfg.model_overrides.get(model_name, {})

        trainer_config = TrainerConfig(
            in_len=cfg.in_len,
            out_len=horizon,
            batch_size=cfg.batch_size,
            device=cfg.device,
            epochs=overrides.get("epochs", cfg.epochs),
            patience=cfg.patience,
            seed=seed,
            checkpoint_dir=cfg.checkpoint_dir,
            null_val=cfg.null_val,
            model_name=ckpt_name,
            dataset_name=cfg.dataset_name,
            horizon=horizon,
            lr=overrides.get("lr", 1e-3),
            weight_decay=overrides.get("weight_decay", 1e-4),
            scheduler=overrides.get("scheduler", "plateau"),
            scheduler_kwargs=overrides.get("scheduler_kwargs", {}),
            use_wandb=True,
            wandb_entity=entity,
            wandb_project=project,
            wandb_run_name=ckpt_name,
        )

        pipeline = _build_sweep_pipeline(cfg, horizon)
        dataloaders, scaler = pipeline.dataloaders, pipeline.scaler
        tsl_graph = (pipeline.edge_index, pipeline.edge_weight)

        adapter, trainer_cls = _build_model_and_adapter(
            model_name, exp_cfg, out_len=horizon, tsl_graph=tsl_graph
        )
        logger = WandbLogger(trainer_config, run_name=ckpt_name)

        trainer = trainer_cls(
            model=adapter,
            config=trainer_config,
            dataloaders=dataloaders,
            scaler=scaler,
            logger=logger,
        )

        t0 = time.time()
        result = trainer.train()
        elapsed = time.time() - t0
        print(
            f"[sweep] {model_name} h={horizon} seed={seed} "
            f"val_loss={result.get('val_loss', float('inf')):.4f} "
            f"elapsed={elapsed:.0f}s"
        )
        logger.finish()

    wandb.agent(sweep_id, function=_sweep_train, count=args.count, **wandb_target_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
