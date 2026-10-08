"""Walk-forward validation for STGNN models.

Not used in the paper, which uses one chronological 70/10/20 split.

Trains a model independently on each expanding-window fold and reports per-fold
and aggregated MAE / MAPE / RMSE. Uses the dataset's pre-defined time boundaries
(see DEFAULT_WALK_FORWARD_BOUNDARIES in src/data/dataset.py).

Usage
-----
python scripts/run_walk_forward.py \\
    --config  scripts/configs/metr_la_training.yaml \\
    --model   gwn \\
    --dataset METR-LA

python scripts/run_walk_forward.py \\
    --config  scripts/configs/pems_bay_training.yaml \\
    --model   staeformer \\
    --dataset PEMS-BAY \\
    --horizon 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Walk-forward validation: train a model on each expanding-window fold."
    )
    p.add_argument("--config", required=True, help="Path to training experiment YAML config.")
    p.add_argument("--model", required=True, help="Model name (gwn, staeformer, etc.).")
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name for built-in walk-forward boundaries (METR-LA or PEMS-BAY).",
    )
    p.add_argument("--horizon", type=int, default=12, help="Forecast horizon.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for all folds.")
    return p


def main(argv: list[str] | None = None) -> int:
    from run_training import (
        _build_model_and_adapter,
        _resolve_model_overrides,
        load_experiment_config,
    )

    from data.tsl_pipeline import build_tsl_walk_forward_pipeline
    from training.config import TrainerConfig
    from training.logger import NoOpLogger

    args = build_parser().parse_args(argv)
    cfg = load_experiment_config(args.config)
    overrides = _resolve_model_overrides(args.model, cfg)
    horizon = args.horizon

    print(f"\nWalk-forward validation: {args.model} on {args.dataset} (h={horizon})")
    print(f"Loading folds from tsl.datasets ({args.dataset}) ...")

    folds, edge_index, edge_weight = build_tsl_walk_forward_pipeline(
        args.dataset,
        in_len=cfg.in_len,
        out_len=horizon,
        batch_size=cfg.batch_size,
    )
    tsl_graph = (edge_index, edge_weight)
    print(f"Found {len(folds)} folds.\n")

    results: list[dict] = []

    for fold in folds:
        a, b, c = fold.boundaries
        print(f"{'─' * 56}")
        print(f"Fold {fold.index}  train[:{a}]  val[{a}:{b}]  test[{b}:{c}]")

        ckpt_name = f"{args.model}_wf_fold{fold.index}_h{horizon}"
        trainer_config = TrainerConfig(
            in_len=cfg.in_len,
            out_len=horizon,
            batch_size=cfg.batch_size,
            device=cfg.device,
            epochs=overrides.get("epochs", cfg.epochs),
            patience=overrides.get("patience", cfg.patience),
            seed=args.seed,
            checkpoint_dir=str(Path(cfg.checkpoint_dir) / "walk_forward"),
            null_val=cfg.null_val,
            model_name=ckpt_name,
            dataset_name=args.dataset,
            horizon=horizon,
            lr=overrides.get("lr", 1e-3),
            weight_decay=overrides.get("weight_decay", 1e-4),
            grad_clip=overrides.get("grad_clip", 5.0),
            scheduler=overrides.get("scheduler", "plateau"),
            scheduler_kwargs=overrides.get("scheduler_kwargs", {}),
        )

        adapter, trainer_cls = _build_model_and_adapter(
            args.model, cfg, out_len=horizon, tsl_graph=tsl_graph
        )
        trainer = trainer_cls(
            model=adapter,
            config=trainer_config,
            dataloaders=fold.dataloaders,
            scaler=fold.scaler,
            logger=NoOpLogger(),
        )

        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

        metrics = trainer.evaluate(split="test")
        results.append(metrics)

        print(
            f"  MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.4f}  "
            f"RMSE={metrics['rmse']:.4f}  [{elapsed:.0f}s]"
        )

    # Aggregated summary
    print(f"\n{'═' * 56}")
    print(f"Walk-forward summary: {args.model} on {args.dataset}  h={horizon}")
    print(f"{'─' * 56}")
    for name, arr in [
        ("MAE ", np.array([r["mae"] for r in results])),
        ("MAPE", np.array([r["mape"] for r in results])),
        ("RMSE", np.array([r["rmse"] for r in results])),
    ]:
        mn, sd, lo, hi = arr.mean(), arr.std(), arr.min(), arr.max()
        print(f"  {name} : {mn:.4f} ± {sd:.4f}  (min {lo:.4f}  max {hi:.4f})")
    print(f"{'═' * 56}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
