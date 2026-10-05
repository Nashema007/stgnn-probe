"""Offline pretraining of BigST's long-term feature branch.

BigST (VLDB 2024) optionally augments its main per-window forecast with a
"long-term" feature embedding extracted by a separately pretrained
transformer (``new_models/BigST/preprocess/model.py::linear_transformer``,
ported to ``src/models/bigst_longterm.py::LinearTransformer``). The original
repo trained that branch on a held-out long-term corpus (one week at 5-minute
resolution, ``input_length=2016``) completely separate from the live training
data.

This framework only has the tsl-loaded METR-LA/PEMS-BAY datasets available —
there is no separate long-term data source — so, per project decision, this
script substitutes a daily window (``input_length=288`` by default, one day
at 5-minute resolution) pretrained for a handful of epochs directly on the
existing train split, reusing ``src/data/tsl_pipeline.py::build_tsl_pipeline``
for data loading rather than writing a new loader. This is a deliberate
simplification: the original week-long corpus is unavailable here, and a few
epochs on a daily window is a reasonable "smoke" substitute, not a faithful
reproduction of the paper's pretraining recipe.

After training, this script runs one more forward pass over the train split
in eval mode and averages the extracted ``feat`` (B, N, nhid) over the batch
dimension to produce a single static ``(N, nhid)`` long-term feature array,
saved via ``np.save``. This array becomes the static long-term feature
embedding fed to every window at main-model training/eval time (see
``BigSTAdapter`` in ``src/training/adapters.py``) — a simplification given
we're substituting daily-window pretraining for the original week-long
corpus; there is no per-window variation in the resulting feature.

Usage
-----
python scripts/preprocess_bigst_features.py --dataset METR-LA --epochs 3
python scripts/preprocess_bigst_features.py --dataset PEMS-BAY --epochs 3 \\
    --output data/probe_inputs/pems_bay/bigst_long_term_features.npy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.tsl_pipeline import build_tsl_pipeline  # noqa: E402
from evaluation.metrics import masked_mae  # noqa: E402
from models.bigst_longterm import LinearTransformer  # noqa: E402


def _dataset_slug(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="METR-LA", choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--input_length",
        type=int,
        default=288,
        help="Long-term input window length (default 288 = one day at 5-min "
        "resolution; the original paper used 2016 = one week, but this "
        "framework's datasets are shorter than that long-term corpus).",
    )
    parser.add_argument("--output_length", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to data/probe_inputs/<dataset>/bigst_long_term_features.npy",
    )
    return parser.parse_args()


def preprocess_bigst_features(
    *,
    dataset: str,
    epochs: int = 3,
    input_length: int = 288,
    output_length: int = 12,
    batch_size: int = 8,
    nhid: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    grad_clip: float = 5.0,
    device: str | torch.device = "cpu",
    output: str | Path | None = None,
) -> Path:
    """Train BigST's long-term branch and save a static ``(N, nhid)`` feature file."""
    device = torch.device(device)

    if input_length % 12 != 0:
        raise ValueError(
            f"--input_length must be a multiple of 12 (LinearTransformer's "
            f"context_conv uses kernel/stride 12); got {input_length}."
        )

    default_output_path = (
        ROOT / "data" / "probe_inputs" / _dataset_slug(dataset) / "bigst_long_term_features.npy"
    )
    output_path = Path(output) if output is not None else default_output_path

    print(f"Loading tsl pipeline for {dataset} (in_len={input_length}, out_len={output_length})...")
    pipeline = build_tsl_pipeline(
        dataset,
        in_len=input_length,
        out_len=output_length,
        batch_size=batch_size,
    )
    train_loader = pipeline.dataloaders["train"]

    # Infer num_nodes/in_dim from one batch. Canonical layout is (B, C, N, T).
    sample_x, _, _ = next(iter(train_loader))
    num_nodes = sample_x.shape[2]
    in_dim = sample_x.shape[1]
    print(f"num_nodes={num_nodes}, in_dim={in_dim}")

    model = LinearTransformer(
        input_length=input_length,
        output_length=output_length,
        in_dim=in_dim,
        num_nodes=num_nodes,
        nhid=nhid,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_losses = []
        for x, y, _y_full in train_loader:
            x = x.to(device)  # (B, C, N, T)
            y = y.to(device)  # (B, N, H)

            # LinearTransformer.forward expects (B, N, T, D) — node-major,
            # same convention as the main BigST model.
            x_lt = x.permute(0, 2, 3, 1)  # (B, N, T, C)

            optimizer.zero_grad()
            pred, _feat = model(x_lt)  # pred: (B, N, output_length)
            loss = masked_mae(pred, y, 0.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs} | train MAE: {mean_loss:.4f} | {elapsed:.1f}s")

    print("Extracting long-term features over the train split (eval mode)...")
    model.eval()
    feat_sums = None
    feat_count = 0
    with torch.no_grad():
        for x, _y, _y_full in train_loader:
            x = x.to(device)
            x_lt = x.permute(0, 2, 3, 1)  # (B, N, T, C)
            _pred, feat = model(x_lt)  # feat: (B, N, nhid)
            batch_sum = feat.sum(dim=0)  # (N, nhid)
            feat_sums = batch_sum if feat_sums is None else feat_sums + batch_sum
            feat_count += feat.shape[0]

    if feat_count == 0 or feat_sums is None:
        raise RuntimeError("Train loader produced no batches; cannot extract long-term features.")

    feat_mean = (feat_sums / feat_count).cpu().numpy().astype(np.float32)  # (N, nhid)
    print(f"Long-term feature array shape: {feat_mean.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feat_mean)
    print(f"Saved long-term features to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    preprocess_bigst_features(
        dataset=args.dataset,
        epochs=args.epochs,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        nhid=args.nhid,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        output=args.output,
    )


if __name__ == "__main__":
    main()
