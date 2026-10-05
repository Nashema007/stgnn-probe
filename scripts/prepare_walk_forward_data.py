"""Convert a raw HDF traffic file into a raw feature .npz for walk-forward validation.

Produces a single ``<dataset>.npz`` containing a ``data: (T, N, C)`` array with
channels ``[speed, tod, dow]``. Windowing into folds happens later at training
time via ``load_walk_forward_datasets()`` in ``src/data/dataset.py``.

Usage
-----
python scripts/prepare_walk_forward_data.py \\
    --dataset METR-LA \\
    --raw-file src/data/raw-data/metr-la.h5 \\
    --output-dir data/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a raw HDF traffic dataframe to a walk-forward feature .npz."
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name used as the output filename stem (e.g. METR-LA → metr-la.npz).",
    )
    p.add_argument("--raw-file", required=True, help="Path to the raw HDF traffic dataframe.")
    p.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write the .npz file into (default: data/).",
    )
    p.add_argument("--dow", action="store_true", help="Include day-of-week channel.")
    return p


def main(argv: list[str] | None = None) -> int:
    from data.generation import save_walk_forward_features_from_hdf

    args = build_parser().parse_args(argv)
    out_path = save_walk_forward_features_from_hdf(
        traffic_df_filename=args.raw_file,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )
    data = np.load(out_path)["data"]
    print(f"Written: {out_path}")
    print(f"Shape:   {data.shape}  (T, N, C) — channels [speed, tod, dow/7]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
