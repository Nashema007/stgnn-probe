"""Re-extract learned adjacency from existing checkpoints (no retraining).

Consistency fix: DSSA-TCN and STAEformer previously extracted their
learned-representation proxy from a single evaluation batch, while STAWnet
averaged over the whole test set. The adapters now all average over the full
test set (see src/training/adapters.py). This script regenerates the
saved adjacency artifacts under that corrected extraction *from the existing
trained weights*, so Table 2's predictions/MAE are untouched — only the
structural inputs (AAS / Q_exc / Fig 1 / Table 3) change.

It loads each ``{model}_h{h}_s{seed}_best.pt`` checkpoint, rebuilds the adapter
exactly as training did, runs the (now full-test) get_adjacency over the test
loader, and rewrites ``{model}_adjacency_h{h}_seeds.npy`` / ``_h{h}.npy`` /
``_adjacency.npy`` via the same _save_adjacency_files helper the trainer uses.

After running, re-run the probe to refresh the structural outputs:
    python run_probe.py --all --skip-performance

Usage
-----
    python scripts/reextract_adjacency.py --config scripts/configs/metr_la_training.yaml
    python scripts/reextract_adjacency.py --config scripts/configs/pems_bay_training.yaml
    # limit models / write elsewhere for a dry run:
    python scripts/reextract_adjacency.py --config <cfg> --models staeformer,dssa_tcn \
        --adjacency-dir /tmp/adj_check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_training import (  # noqa: E402
    _build_model_and_adapter,
    _load_dataloaders,
    _load_seeds_log,
    _save_adjacency_files,
    load_experiment_config,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-extract learned adjacency from checkpoints.")
    p.add_argument("--config", required=True, help="Path to the training experiment YAML.")
    p.add_argument(
        "--models",
        default="staeformer,dssa_tcn",
        help="Comma-separated models to re-extract (default: the two the fix targets).",
    )
    p.add_argument(
        "--adjacency-dir",
        default=None,
        help="Output directory for the *_adjacency*.npy files (default: cfg.adjacency_dir).",
    )
    p.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated subset of horizons to re-extract (default: all in the config).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from data.tsl_pipeline import build_tsl_pipeline
    from training.device import resolve_device

    args = build_parser().parse_args(argv)
    cfg = load_experiment_config(args.config)
    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    adjacency_dir = Path(args.adjacency_dir) if args.adjacency_dir else Path(cfg.adjacency_dir)
    adjacency_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    horizons = (
        [int(h) for h in args.horizons.split(",") if h.strip()]
        if args.horizons
        else list(cfg.horizons)
    )
    seeds_log = _load_seeds_log(output_dir)

    # Canonical scaler: fit once at the largest horizon, exactly as run_training does,
    # so per-horizon loaders normalise test inputs identically to the training run.
    max_h = max(cfg.horizons)
    canonical_scaler = build_tsl_pipeline(
        cfg.dataset_name,
        in_len=cfg.in_len,
        out_len=max_h,
        batch_size=cfg.batch_size,
        val_len=cfg.val_len,
        test_len=cfg.test_len,
    ).scaler

    for model_name in models:
        if model_name not in seeds_log:
            print(f"[skip] {model_name}: no seeds recorded in {output_dir / 'seeds.json'}")
            continue
        print(f"\n=== {model_name} ({cfg.dataset_name}) ===")
        adj_per_horizon: dict[int, np.ndarray] = {}
        for h in horizons:
            seeds = seeds_log[model_name].get(str(h), [])
            if not seeds:
                print(f"  h={h}: no seeds; skipping")
                continue
            dataloaders, _scaler, tsl_graph = _load_dataloaders(
                cfg, out_len=h, scaler=canonical_scaler
            )
            per_seed: list[np.ndarray] = []
            for seed in seeds:
                ckpt_path = Path(cfg.checkpoint_dir) / f"{model_name}_h{h}_s{seed}_best.pt"
                if not ckpt_path.exists():
                    print(f"  h={h} seed={seed}: MISSING checkpoint {ckpt_path} — skipping")
                    continue
                adapter, _ = _build_model_and_adapter(
                    model_name, cfg, out_len=h, tsl_graph=tsl_graph
                )
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                # strict=True on purpose: a checkpoint whose architecture no longer
                # matches the current code (e.g. a stale pre-bugfix DSSA-TCN weight
                # file) must fail loudly here, never silently load partial/garbage
                # weights and produce a plausible-but-wrong adjacency.
                try:
                    adapter.load_state_dict(ckpt["state_dict"], strict=True)
                except RuntimeError as exc:
                    print(
                        f"  h={h} seed={seed}: SKIP — checkpoint incompatible with current "
                        f"{model_name} architecture ({str(exc).splitlines()[0]}). "
                        f"Re-extract where the matching (paper) checkpoints live."
                    )
                    continue
                adapter.to(device).eval()
                adj = adapter.get_adjacency(dataloaders["test"], device)
                if adj is None:
                    print(f"  h={h} seed={seed}: get_adjacency returned None — skipping")
                    continue
                per_seed.append(np.asarray(adj, dtype=np.float32))
                print(f"  h={h} seed={seed}: extracted adj {adj.shape}")
            if per_seed:
                adj_per_horizon[h] = np.stack(per_seed, axis=0)  # (R, N, N)
        if adj_per_horizon:
            _save_adjacency_files(model_name, adj_per_horizon, adjacency_dir)
        else:
            print(f"  [warn] {model_name}: nothing extracted — no files written")

    print(f"\nDone. Adjacency written under {adjacency_dir}")
    print(
        "Next: python run_probe.py --all --skip-performance (refresh AAS / Q_exc / Fig 1 / Table 3)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
