#!/usr/bin/env python
"""Convert PNG figures to EPS for LaTeX/Overleaf inclusion.

EPS is a vector container, but a PNG is raster, so the bitmap is embedded
as-is (no vectorisation happens or is possible from a PNG). The physical size
of the embedded image is set from its DPI so ``\\includegraphics`` scales it
sensibly; pass ``--dpi`` to override when the PNG carries no DPI metadata.

Requires only Pillow (no Ghostscript — that is needed for *reading* EPS, not
writing). RGBA images are flattened onto a white background because EPS has no
alpha channel.

Usage
-----
    # Convert every PNG in paper_figures/ to an .eps next to it
    python scripts/png_to_eps.py paper_figures

    # Write the EPS files to a separate directory at a fixed DPI
    python scripts/png_to_eps.py paper_figures --out paper_figures_eps --dpi 300

    # Convert specific files
    python scripts/png_to_eps.py fig1.png fig2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _collect_pngs(inputs: list[Path]) -> list[Path]:
    """Expand directories to their ``*.png`` contents; keep files as given."""
    pngs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            pngs.extend(sorted(item.glob("*.png")))
        elif item.suffix.lower() == ".png":
            pngs.append(item)
        else:
            print(f"  skip (not a .png): {item}", file=sys.stderr)
    return pngs


def convert_one(src: Path, dst: Path, dpi: int | None) -> tuple[int, int]:
    """Convert a single PNG to EPS. Returns the (width, height) in pixels."""
    with Image.open(src) as img:
        # EPS has no alpha; composite onto white so transparent areas render
        # white rather than black.
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            flat = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            flat = img.convert("RGB")

        # Pillow derives the EPS bounding box (in points) from the DPI, so a
        # correct DPI keeps the figure a reasonable physical size in the PDF.
        effective_dpi = dpi or img.info.get("dpi", (100, 100))[0] or 100
        dst.parent.mkdir(parents=True, exist_ok=True)
        flat.save(dst, format="EPS", dpi=(effective_dpi, effective_dpi))
        return flat.size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="PNG files and/or directories containing PNGs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the EPS files (default: alongside each PNG).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Force this DPI instead of reading it from the PNG (default: PNG's own, or 100).",
    )
    args = parser.parse_args(argv)

    pngs = _collect_pngs(args.inputs)
    if not pngs:
        print("No PNG files found.", file=sys.stderr)
        return 1

    print(f"Converting {len(pngs)} PNG(s) to EPS...")
    for src in pngs:
        dst = (args.out / src.name if args.out else src).with_suffix(".eps")
        w, h = convert_one(src, dst, args.dpi)
        size_mb = dst.stat().st_size / 1e6
        print(f"  {src.name}  ({w}x{h})  ->  {dst}  [{size_mb:.1f} MB]")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
