"""Export repo-local raw traffic data into STGNN-Probe input files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare standardized STGNN-Probe raw inputs.")
    parser.add_argument("--dataset", required=True, help="Dataset name: METR-LA or PEMS-BAY.")
    parser.add_argument(
        "--source",
        choices=["tsl", "legacy"],
        default="tsl",
        help=(
            "'tsl' (default) sources raw_data/distance_edges/road_adjacency from tsl's own "
            "auto-downloaded dataset cache (--tsl-cache-dir) — no manual download needed, and "
            "guarantees the same adjacency every spatial model trains on. 'legacy' sources from "
            "manually-downloaded DCRNN raw files (--raw-dir) instead."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default="src/data/raw-data",
        help="[--source legacy] Directory containing local raw METR-LA / PEMS-BAY files.",
    )
    parser.add_argument(
        "--tsl-cache-dir",
        default="data/tsl_cache",
        help="[--source tsl] Directory tsl downloads/caches its dataset into.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/probe_inputs",
        help="Directory where standardized STGNN-Probe inputs will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.source == "tsl":
        from data.tsl_pipeline import export_probe_dataset_inputs_from_tsl

        paths = export_probe_dataset_inputs_from_tsl(
            args.dataset, args.output_dir, tsl_cache_dir=args.tsl_cache_dir
        )
    else:
        from data.raw_sources import export_probe_dataset_inputs

        paths = export_probe_dataset_inputs(args.dataset, args.output_dir, raw_dir=args.raw_dir)

    print(f"Prepared STGNN-Probe raw inputs for {args.dataset} (source={args.source}):")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
