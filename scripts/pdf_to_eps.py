#!/usr/bin/env python
"""Convert vector PDF figures to TRUE-VECTOR EPS with Ghostscript.

Unlike ``png_to_eps.py`` (which wraps a raster bitmap in an EPS container), this
converts an existing *vector* PDF to a *vector* EPS via Ghostscript's
``eps2write`` device — small files, clean lines, no watermark. Use it on the
``.pdf`` files already in ``paper_figures/`` after installing Ghostscript:

    brew install ghostscript            # macOS  (apt-get install ghostscript on Linux)
    python scripts/pdf_to_eps.py paper_figures

It does not re-render the figures, so it needs no Chrome/kaleido — only gs.

Usage
-----
    python scripts/pdf_to_eps.py paper_figures            # every *.pdf -> *.eps
    python scripts/pdf_to_eps.py fig1.pdf fig2.pdf        # specific files
    python scripts/pdf_to_eps.py paper_figures --out eps  # into a separate dir
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_ghostscript() -> str | None:
    for name in ("gs", "gsc", "gswin64c", "gswin32c"):
        found = shutil.which(name)
        if found:
            return found
    for path in ("/opt/homebrew/bin/gs", "/usr/local/bin/gs"):
        if Path(path).is_file():
            return path
    return None


def collect_pdfs(inputs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            pdfs.extend(sorted(item.glob("*.pdf")))
        elif item.suffix.lower() == ".pdf":
            pdfs.append(item)
        else:
            print(f"  skip (not a .pdf): {item}", file=sys.stderr)
    return pdfs


def convert(gs: str, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            gs,
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=eps2write",
            f"-sOutputFile={dst}",
            str(src),
        ],
        check=True,
        capture_output=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files and/or directories.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output dir (default: beside each PDF)."
    )
    args = parser.parse_args(argv)

    gs = find_ghostscript()
    if not gs:
        print(
            "Ghostscript not found. Install it and retry:\n"
            "  macOS : brew install ghostscript\n"
            "  Linux : sudo apt-get install ghostscript",
            file=sys.stderr,
        )
        return 2

    pdfs = collect_pdfs(args.inputs)
    if not pdfs:
        print("No PDF files found.", file=sys.stderr)
        return 1

    print(f"Converting {len(pdfs)} PDF(s) to vector EPS via {gs} ...")
    failed = 0
    for src in pdfs:
        dst = (args.out / src.name if args.out else src).with_suffix(".eps")
        try:
            convert(gs, src, dst)
            print(f"  {src.name} -> {dst}  [{dst.stat().st_size / 1024:.0f} KB]")
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode(errors="replace")[:200] if e.stderr else ""
            print(f"  FAILED {src.name}: {msg}", file=sys.stderr)
            failed += 1

    print("Done." if not failed else f"Done with {failed} failure(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
