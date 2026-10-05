"""Convert a raw HDF traffic file into pre-windowed train/val/test .npz splits.

Produces ``train.npz``, ``val.npz``, and ``test.npz`` in the output directory,
each containing ``x: (B, in_len, N, C)`` and ``y: (B, out_len, N, C)`` arrays.
This is the standard Graph WaveNet data layout for fixed-split evaluation.

Usage
-----
python scripts/prepare_standard_splits.py \\
    --raw-file src/data/raw-data/metr-la.h5 \\
    --output-dir data/METR-LA/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a raw HDF traffic dataframe to pre-windowed train/val/test splits."
    )
    p.add_argument("--raw-file", required=True, help="Path to the raw HDF traffic dataframe.")
    p.add_argument(
        "--output-dir", required=True, help="Directory to write train/val/test .npz files into."
    )
    p.add_argument(
        "--seq-length-x", type=int, default=12, help="Input sequence length (default: 12)."
    )
    p.add_argument(
        "--seq-length-y", type=int, default=12, help="Output sequence length (default: 12)."
    )
    p.add_argument("--y-start", type=int, default=1, help="First y offset step (default: 1).")
    p.add_argument("--dow", action="store_true", help="Include day-of-week channel.")
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples for training (default: 0.7).",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for testing (default: 0.2).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from data.generation import save_graph_wavenet_splits_from_hdf

    args = build_parser().parse_args(argv)
    shapes = save_graph_wavenet_splits_from_hdf(
        traffic_df_filename=args.raw_file,
        output_dir=args.output_dir,
        in_len=args.seq_length_x,
        out_len=args.seq_length_y,
        y_start=args.y_start,
        add_time_in_day=True,
        add_day_in_week=args.dow,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
    )
    for split, shape in shapes.items():
        print(f"{split:5s}  x shape: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
