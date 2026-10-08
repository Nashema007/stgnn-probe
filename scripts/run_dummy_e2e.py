"""Deterministic CPU-only dummy workflow for full STGNN-Probe wiring.

Not used in the paper: a synthetic smoke test of the pipeline wiring.

This script is intentionally separate from the production training configs. It
trains tiny synthetic model runs for every supported model, exports standardized
probe inputs, and runs STGNN-Probe to produce dummy reports and figures.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse
from tsl.nn.models.stgn import GraphWaveNetModel
from tsl.nn.models.temporal import TCNModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.config import load_config  # noqa: E402
from analysis.probe import ProbeRunner  # noqa: E402
from data.dataset import SlidingWindowDataset  # noqa: E402
from data.scaler import StandardScaler  # noqa: E402
from models import DSSATCN, GWNv2, STAEformer, STAWnet  # noqa: E402
from training.adapters import (  # noqa: E402
    DSSATCNAdapter,
    GWNAdapter,
    GWNv2Adapter,
    STAEformerAdapter,
    STAWnetAdapter,
    TCNAdapter,
)
from training.arima_trainer import predict_arima  # noqa: E402
from training.config import TrainerConfig  # noqa: E402
from training.logger import NoOpLogger  # noqa: E402
from training.trainer import GraphTrainer  # noqa: E402

HORIZONS = [6, 12, 18, 24, 30, 36, 42]
SPATIAL_MODELS = ("gwn", "gwn_v2", "stawnet", "staeformer", "dssa_tcn")
ALL_MODELS = (*SPATIAL_MODELS, "tcn", "arima")
NUM_NODES = 5
NUM_CHANNELS = 3
IN_LEN = 12
MAX_HORIZON = max(HORIZONS)

MODEL_SEEDS = {
    "gwn": 102,
    "gwn_v2": 103,
    "stawnet": 104,
    "staeformer": 105,
    "dssa_tcn": 106,
    "tcn": 107,
    "arima": 109,
}


@dataclass(frozen=True)
class DummyE2EResult:
    output_dir: Path
    probe_inputs_dir: Path
    probe_config_path: Path
    models: tuple[str, ...]
    horizons: list[int]
    probe_results: dict[str, Any]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_traffic(num_timesteps: int = 220) -> np.ndarray:
    """Return deterministic traffic data shaped (T, N, C)."""
    rng = np.random.default_rng(42)
    t = np.arange(num_timesteps, dtype=np.float32)
    data = np.zeros((num_timesteps, NUM_NODES, NUM_CHANNELS), dtype=np.float32)

    for node in range(NUM_NODES):
        phase = node * 0.35
        base = 20.0 + node * 2.0
        signal = base + 3.0 * np.sin(t / 9.0 + phase) + 1.5 * np.cos(t / 17.0 - phase)
        trend = 0.015 * t
        noise = rng.normal(0.0, 0.05, size=num_timesteps).astype(np.float32)
        data[:, node, 0] = signal + trend + noise

    data[:, :, 1] = ((t % 288) / 288.0)[:, None]
    data[:, :, 2] = (((t // 24) % 7) / 7.0)[:, None]
    return data


def ring_adjacency(num_nodes: int = NUM_NODES) -> np.ndarray:
    adj = np.eye(num_nodes, dtype=np.float32)
    for idx in range(num_nodes):
        adj[idx, (idx + 1) % num_nodes] = 1.0
        adj[idx, (idx - 1) % num_nodes] = 1.0
    return adj


def make_coordinates(num_nodes: int = NUM_NODES) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": np.arange(num_nodes, dtype=np.int32),
            "latitude": 34.0 + np.arange(num_nodes) * 0.01,
            "longitude": -118.0 - np.arange(num_nodes) * 0.01,
        }
    )


def make_dataloaders(
    data: np.ndarray,
    out_len: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, DataLoader], StandardScaler]:
    split_len = IN_LEN + out_len + 6
    train = data[:split_len]
    val = data[split_len : 2 * split_len]
    test = data[2 * split_len : 3 * split_len]

    scaler = StandardScaler().fit(train)
    train_ds = SlidingWindowDataset(scaler.transform(train), IN_LEN, out_len)
    val_ds = SlidingWindowDataset(scaler.transform(val), IN_LEN, out_len)
    test_ds = SlidingWindowDataset(scaler.transform(test), IN_LEN, out_len)

    generator = torch.Generator().manual_seed(seed)
    return (
        {
            "train": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                generator=generator,
            ),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        },
        scaler,
    )


def _supports(device: torch.device) -> list[torch.Tensor]:
    adj = torch.tensor(ring_adjacency(), dtype=torch.float32, device=device)
    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    transition = adj / row_sum
    return [transition, transition.t()]


def build_model_and_trainer(
    model_name: str,
    device: torch.device,
    out_len: int,
) -> tuple[torch.nn.Module, type[GraphTrainer]]:
    """Build tiny model + matching trainer class for the dummy e2e workflow."""
    seed_everything(MODEL_SEEDS[model_name])
    supports = _supports(device)

    if model_name == "tcn":
        # The temporal baseline sees all three canonical channels, matching
        # the production tcn_base.yaml configuration.
        return (
            TCNAdapter(
                TCNModel(
                    input_size=NUM_CHANNELS,
                    output_size=1,
                    horizon=out_len,
                    hidden_size=4,
                    ff_size=4,
                    n_layers=2,
                    norm="batch",
                ),
                num_inputs=NUM_CHANNELS,
            ),
            GraphTrainer,
        )

    if model_name == "gwn":
        edge_index, edge_weight = dense_to_sparse(
            torch.tensor(ring_adjacency(), dtype=torch.float32, device=device)
        )
        return (
            GWNAdapter(
                GraphWaveNetModel(
                    input_size=NUM_CHANNELS,
                    output_size=1,
                    horizon=out_len,
                    n_nodes=NUM_NODES,
                    hidden_size=4,
                    ff_size=8,
                    n_layers=2,
                    emb_size=2,
                ),
                edge_index,
                edge_weight,
            ),
            GraphTrainer,
        )

    if model_name == "gwn_v2":
        return (
            GWNv2Adapter(
                GWNv2(
                    device,
                    NUM_NODES,
                    dropout=0.0,
                    supports=supports,
                    in_dim=NUM_CHANNELS,
                    out_dim=out_len,
                    residual_channels=2,
                    dilation_channels=2,
                    skip_channels=4,
                    end_channels=8,
                    blocks=4,
                    layers=2,
                    apt_size=2,
                )
            ),
            GraphTrainer,
        )

    if model_name == "stawnet":
        return (
            STAWnetAdapter(
                STAWnet(
                    device,
                    NUM_NODES,
                    dropout=0.0,
                    in_dim=NUM_CHANNELS,
                    out_dim=out_len,
                    residual_channels=2,
                    dilation_channels=2,
                    skip_channels=4,
                    end_channels=8,
                    blocks=4,
                    layers=2,
                    emb_length=2,
                )
            ),
            GraphTrainer,
        )

    if model_name == "staeformer":
        return (
            STAEformerAdapter(
                STAEformer(
                    num_nodes=NUM_NODES,
                    in_steps=IN_LEN,
                    out_steps=out_len,
                    steps_per_day=288,
                    input_dim=1,
                    output_dim=1,
                    input_embedding_dim=4,
                    tod_embedding_dim=4,
                    dow_embedding_dim=4,
                    spatial_embedding_dim=0,
                    adaptive_embedding_dim=4,
                    feed_forward_dim=8,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                )
            ),
            GraphTrainer,
        )

    if model_name == "dssa_tcn":
        return (
            DSSATCNAdapter(
                DSSATCN(
                    input_dim=NUM_CHANNELS,
                    out_dim=out_len,
                    num_nodes=NUM_NODES,
                    residual_channels=2,
                    dilation_channels=2,
                    skip_channels=4,
                    end_channels=8,
                    blocks=1,
                    layers=1,
                    input_embedding_dim=2,
                    tod_embedding_dim=2,
                    dow_embedding_dim=2,
                    adaptive_embedding_dim=2,
                    feed_forward_dim=8,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    adjs=supports,
                    gcn_order=1,
                    use_topk=False,
                )
            ),
            GraphTrainer,
        )

    raise ValueError(f"Unknown model {model_name!r}.")


def train_neural_model(
    model_name: str,
    data: np.ndarray,
    output_dir: Path,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    seed = MODEL_SEEDS[model_name]
    seed_everything(seed)
    dataloaders, scaler = make_dataloaders(data, MAX_HORIZON, batch_size=2, seed=seed)
    model, trainer_cls = build_model_and_trainer(model_name, torch.device("cpu"), MAX_HORIZON)

    config = TrainerConfig(
        in_len=IN_LEN,
        out_len=MAX_HORIZON,
        batch_size=2,
        epochs=epochs,
        patience=max(epochs, 1),
        seed=seed,
        device="cpu",
        lr=1e-2,
        weight_decay=0.0,
        use_wandb=False,
        checkpoint_dir=str(output_dir / "checkpoints"),
        model_name=f"dummy_{model_name}",
        dataset_name="dummy_e2e",
        horizon=MAX_HORIZON,
        null_val=0.0,
    )
    trainer = trainer_cls(model, config, dataloaders, scaler, NoOpLogger())
    trainer.train()
    preds_full, truth_full = trainer.predict("test")  # (W, N, MAX_HORIZON) each

    adjacency = None
    if model_name in SPATIAL_MODELS and hasattr(model, "get_adjacency"):
        needs_loader = model_name in {"stawnet", "dssa_tcn", "staeformer"}
        # cast: nn.Module.__getattr__ is typed Tensor | Module, so mypy can't
        # know get_adjacency is a real method here — guarded by hasattr above.
        adjacency = cast(Any, model).get_adjacency(
            loader=dataloaders["test"] if needs_loader else None,
            device=torch.device("cpu"),
        )
    return preds_full, truth_full, adjacency


def select_horizons(values: np.ndarray) -> np.ndarray:
    """Slice the configured HORIZONS steps out of the trailing H axis."""
    return values[..., [h - 1 for h in HORIZONS]].astype(np.float32)


def write_probe_config(
    output_dir: Path,
    probe_inputs_dir: Path,
    save_path: Path,
    spatial_models: tuple[str, ...],
) -> None:
    config = {
        "datasets": [
            {
                "name": "DUMMY",
                "num_nodes": NUM_NODES,
                "raw_data": str(probe_inputs_dir / "raw.npy"),
                "coordinates": str(probe_inputs_dir / "coords.csv"),
                "ground_truth": str(probe_inputs_dir / "ground_truth.npy"),
                "predictions_dir": str(probe_inputs_dir / "predictions"),
                "adjacency_dir": str(probe_inputs_dir / "adjacency"),
                "horizons": HORIZONS,
                "horizon_minutes": [h * 5 for h in HORIZONS],
            }
        ],
        "models": {
            "temporal_baselines": ["tcn", "arima"],
            "spatial_models": list(spatial_models),
        },
        "granger": {"max_lag": 2, "significance": 0.05, "n_jobs": 1},
        "community": {"num_runs": 2, "resolution": 1.0, "random_seed": 0},
        "alignment": {"threshold": 0.1, "sweep_steps": 5},
        "sgs_threshold": 0.1,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_dummy_e2e(
    output_dir: str | Path = "outputs/e2e_dummy",
    epochs: int = 5,
    save_figures: bool = True,
    models: tuple[str, ...] = ALL_MODELS,
) -> DummyE2EResult:
    if epochs < 1 or epochs > 5:
        raise ValueError("epochs must be between 1 and 5 for the dummy e2e workflow.")
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown dummy e2e models: {sorted(unknown)}.")

    output_root = Path(output_dir)
    probe_inputs_dir = output_root / "probe_inputs"
    predictions_dir = probe_inputs_dir / "predictions"
    adjacency_dir = probe_inputs_dir / "adjacency"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    adjacency_dir.mkdir(parents=True, exist_ok=True)

    data = make_synthetic_traffic()
    np.save(probe_inputs_dir / "raw.npy", data[:, :, 0])
    make_coordinates().to_csv(probe_inputs_dir / "coords.csv", index=False)

    ground_truth: np.ndarray | None = None
    spatial_models = tuple(model for model in models if model in SPATIAL_MODELS)

    for model_name in models:
        print(f"[dummy-e2e] running {model_name}")
        if model_name == "arima":
            # Align to the same test windows make_dataloaders()/SlidingWindowDataset
            # produce for the neural models below: test = data[2*split_len : 3*split_len],
            # windowed with (IN_LEN, MAX_HORIZON) -> split_len - IN_LEN - MAX_HORIZON + 1 windows.
            split_len = IN_LEN + MAX_HORIZON + 6
            arima_test_start = 2 * split_len + IN_LEN
            arima_test_count = split_len - IN_LEN - MAX_HORIZON + 1
            preds_per_window = predict_arima(
                data,
                horizons=HORIZONS,
                test_start=arima_test_start,
                test_count=arima_test_count,
                order=(1, 0, 0),
            )  # (W, H, N) — already the repo-default per-window layout
            np.save(predictions_dir / "arima_predictions.npy", preds_per_window[:, :, :, None])
            continue

        preds_full, truth_full, adjacency = train_neural_model(
            model_name,
            data,
            output_root,
            epochs,
        )  # (W, N, MAX_HORIZON) each
        preds_whn = select_horizons(preds_full)  # (W, N, H)
        preds_whnr = np.transpose(preds_whn, (0, 2, 1))[:, :, :, None]  # (W, H, N, 1)
        np.save(predictions_dir / f"{model_name}_predictions.npy", preds_whnr)

        if ground_truth is None:
            truth_whn = select_horizons(truth_full)  # (W, N, H)
            ground_truth = np.transpose(truth_whn, (0, 2, 1))  # (W, H, N)
            np.save(probe_inputs_dir / "ground_truth.npy", ground_truth)

        if model_name in SPATIAL_MODELS:
            if adjacency is None:
                adjacency = ring_adjacency()
            np.save(adjacency_dir / f"{model_name}_adjacency.npy", adjacency.astype(np.float32))

    if ground_truth is None:
        _, truth_full, _ = train_neural_model("tcn", data, output_root, epochs)
        truth_whn = select_horizons(truth_full)  # (W, N, H)
        ground_truth = np.transpose(truth_whn, (0, 2, 1))  # (W, H, N)
        np.save(probe_inputs_dir / "ground_truth.npy", ground_truth)

    probe_results: dict[str, Any] = {}
    probe_config_path = output_root / "dummy_probe_config.yaml"
    write_probe_config(output_root, probe_inputs_dir, probe_config_path, spatial_models)
    if spatial_models:
        config = load_config(probe_config_path)
        runner = ProbeRunner(output_dir=output_root / "probe_outputs", config=config)
        probe_results = runner.run_all_from_config(save_figures=save_figures)

    return DummyE2EResult(
        output_dir=output_root,
        probe_inputs_dir=probe_inputs_dir,
        probe_config_path=probe_config_path,
        models=tuple(models),
        horizons=HORIZONS,
        probe_results=probe_results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic dummy e2e workflow.")
    parser.add_argument("--output-dir", default="outputs/e2e_dummy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_dummy_e2e(
        output_dir=args.output_dir,
        epochs=args.epochs,
        save_figures=not args.no_figures,
    )
    print(f"Dummy e2e complete: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
