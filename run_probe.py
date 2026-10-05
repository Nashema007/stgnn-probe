"""Command-line entry point for STGNN-Probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run STGNN-Probe analyses.")
    parser.add_argument("--config", default="config.yaml", help="Path to STGNN-Probe YAML config.")
    parser.add_argument("--dataset", help="Dataset name to run.")
    parser.add_argument("--model", help="Spatial model name to run.")
    parser.add_argument("--all", action="store_true", help="Run all configured datasets/models.")
    parser.add_argument(
        "--performance",
        action="store_true",
        help=(
            "Run Lens 0 (forecasting performance benchmark) for all configured "
            "datasets. Can be combined with --all, or run on its own to "
            "(re)generate performance results without redoing the structural "
            "lenses."
        ),
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="With --all, skip Lens 0 so only the structural lenses are (re)run.",
    )
    parser.add_argument("--force-granger", action="store_true", help="Recompute Granger cache.")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figure generation.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for reports and caches.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from analysis.config import load_config
    from analysis.probe import ProbeRunner

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    runner = ProbeRunner(output_dir=Path(args.output_dir), config=config)
    save_figures = not args.no_figures

    if args.skip_performance and not args.all:
        raise SystemExit("--skip-performance only applies together with --all.")

    if args.all:
        results = runner.run_all_from_config(
            force_granger=args.force_granger,
            save_figures=save_figures,
            run_performance=not args.skip_performance,
        )
        print(f"STGNN-Probe complete for {len(results)} model/dataset runs")
        return 0

    if args.performance:
        perf_results = runner.run_all_performance_from_config(save_figures=save_figures)
        print(f"STGNN-Probe performance benchmark complete for {len(perf_results)} dataset(s)")
        return 0

    if not args.dataset or not args.model:
        raise SystemExit(
            "--dataset and --model are required unless --all or --performance is used."
        )

    result = runner.run_from_config(
        args.dataset,
        args.model,
        force_granger=args.force_granger,
        save_figures=save_figures,
    )
    result.print_summary()
    print(f"STGNN-Probe complete for {args.model} on {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
