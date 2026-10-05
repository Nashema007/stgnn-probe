#!/usr/bin/env python
"""Rebuild the curated paper-figure bundle from source in multiple formats.

The eight curated figures are regenerated from their original chart specs
(embedded in the ``.html`` files the probe writes), not by re-wrapping the PNGs,
so PNG and PDF come out as native, crisp, vector-quality exports.

Two chart libraries are used by the probe and both are handled here:
  * Plotly   -> exported with kaleido       (lens0 SGS, adjacency heatmaps, lens5)
  * Vega-Lite -> exported with vl-convert    (lens3 alignment bars)

Format support in this environment
----------------------------------
  * png : native, high-DPI (``--scale``)          -- both libraries
  * pdf : native TRUE VECTOR (recommended for LaTeX/Overleaf) -- both libraries
  * svg : native TRUE VECTOR                        -- both libraries
  * eps : TRUE VECTOR when Ghostscript (``gs``) is installed -- we render the
          figure to a vector PDF and convert it with ``gs -sDEVICE=eps2write``.
          If ``gs`` is not found we fall back to wrapping the high-DPI PNG in an
          EPS container (raster, large) and say so. Install with
          ``brew install ghostscript`` (macOS) / ``apt-get install ghostscript``.

Usage
-----
    python scripts/regenerate_figures.py                        # png,pdf,eps -> paper_figures/
    python scripts/regenerate_figures.py --formats png,pdf,svg  # add svg, skip eps
    python scripts/regenerate_figures.py --scale 4 --out build/figs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (source .html, output basename) — mirrors paper_figures/README.md.
FIGURES: list[tuple[str, str]] = [
    (
        "outputs/lens0_performance/METR-LA/figures/improvement_over_tcn_no_arima.html",
        "fig1_sgs_over_tcn_metr_la",
    ),
    (
        "outputs/lens0_performance/PEMS-BAY/figures/improvement_over_tcn_no_arima.html",
        "fig1_sgs_over_tcn_pems_bay",
    ),
    ("outputs/comparative/metr_la/lens3_alignment_comparison.html", "fig2_aas_alignment_metr_la"),
    ("outputs/comparative/pems_bay/lens3_alignment_comparison.html", "fig2_aas_alignment_pems_bay"),
    ("outputs/per_model/bigst_METR-LA/adjacency_heatmap.html", "fig3a_adjacency_bigst_structured"),
    ("outputs/per_model/d2stgnn_METR-LA/adjacency_heatmap.html", "fig3b_adjacency_d2stgnn_diffuse"),
    (
        "outputs/comparative/metr_la/lens5_combined_dashboard.html",
        "fig4_degradation_dashboard_metr_la",
    ),
    (
        "outputs/comparative/pems_bay/lens5_combined_dashboard.html",
        "fig4_degradation_dashboard_pems_bay",
    ),
]

VECTOR_HINT = "pdf"  # what we point users to when they ask for EPS


# --------------------------------------------------------------------------- #
# Loading chart specs back out of the probe's HTML                            #
# --------------------------------------------------------------------------- #
def _detect_kind(html: str) -> str:
    if "Plotly.newPlot(" in html:
        return "plotly"
    if "vega-lite" in html or "var spec =" in html:
        return "vega"
    raise ValueError("Unrecognised chart HTML (neither Plotly nor Vega-Lite).")


def _load_plotly(html: str):
    """Reconstruct a Plotly figure from its ``Plotly.newPlot(div, data, layout)``."""
    import plotly.io as pio

    dec = json.JSONDecoder()
    start = html.index("Plotly.newPlot(")
    data_at = html.index("[", start)  # data array
    data, end = dec.raw_decode(html, data_at)
    layout_at = html.index("{", end)  # layout object
    layout, _ = dec.raw_decode(html, layout_at)
    return pio.from_json(json.dumps({"data": data, "layout": layout}))


def _load_vega(html: str) -> str:
    """Return the Vega-Lite spec (as a JSON string) embedded as ``var spec = {...}``."""
    start = html.index("var spec =")
    spec_at = html.index("{", start)
    spec, _ = json.JSONDecoder().raw_decode(html, spec_at)
    return json.dumps(spec)


def _enlarge_plotly_fonts(fig) -> None:
    """Scale up label fonts of a Plotly figure for print legibility.

    The probe emits interactive charts at ~10pt on a 900x500 canvas. Once the
    figure is shrunk to ~0.4 text-width in the paper, those labels become
    unreadable, so the fonts are set large enough to survive the downscale. The
    redundant in-figure title is dropped (the LaTeX caption already names the
    figure), which also frees vertical space for the plot.

    The canvas is enlarged in step with the fonts: at 36pt on the original
    500px-tall canvas the rotated y-axis title is taller than the plot area and
    bleeds off the top, and a pinned ``r`` margin clips the longest legend entry
    (STAEformer). Growing width/height and letting ``autoexpand`` size the
    margins keeps both inside the canvas. Because the export is vector, only the
    font-to-canvas ratio matters, so this changes legibility, not scale.

    Marker and line sizes are also enlarged (see below), for the same
    downscale reason. Applied only to the figures that need it (see main);
    other figures are unchanged.
    """
    fig.update_layout(
        title_text="",  # redundant with the LaTeX caption; removing it declutters and frees space
        width=1400,  # was 900x500 -- too small to hold the enlarged labels
        height=760,
        font=dict(size=30),  # base: legend entries, hover
        legend=dict(font=dict(size=30), title_font=dict(size=30)),
        # autoexpand grows these to fit the legend and the axis titles; the
        # previous tight t/r pin is what clipped them.
        margin=dict(l=20, r=20, t=30, b=20, autoexpand=True),
    )
    # Each panel is placed at ~0.49\textwidth, roughly a 0.12x downscale of this
    # 1400px canvas measured in points, so tick labels at size 30 rendered at
    # only ~3.6pt in the paper. Ticks 78 / titles 90 render at roughly 9.5pt /
    # 11pt -- near the LNCS body size -- while the 1400x760 canvas with
    # autoexpand margins still leaves the plot area dominant. Raising the
    # font-to-canvas ratio (not the export --scale, which cancels out for a
    # vector PDF placed at a fixed width) is what enlarges the print size.
    # title_standoff pushes the rotated y-title clear of the (now large) tick
    # numbers; without it the "(%)"/"rel" of the title overrun the "12"/"8"
    # ticks. autoexpand then grows the left margin to hold both.
    # Tick font 68: at dtick=3 the y-axis carries ~6 labels, which fit the
    # compact panel at this size without collision. Axis titles are 78 (reduced
    # from 90 on review so they no longer dominate the small panels; ticks,
    # legend and subfigure labels are left as-is). The figure has zero page
    # slack, so the canvas is kept compact rather than grown to hold larger ticks.
    fig.update_xaxes(title_font=dict(size=78), tickfont=dict(size=68), title_standoff=30)
    fig.update_yaxes(title_font=dict(size=78), tickfont=dict(size=68), title_standoff=50)
    # Series lines and point markers: the probe emits ~2px lines and ~6px
    # markers, which downscale to hairlines and specks at column width. Thicken
    # the lines and enlarge the markers so the seven series and their three
    # horizon points read clearly against the enlarged axes.
    fig.update_traces(line=dict(width=9), marker=dict(size=17), selector=dict(type="scatter"))
    for ann in fig.layout.annotations or ():
        if ann.font is not None and ann.font.size is not None:
            ann.font.size = max(ann.font.size, 30)
        else:
            ann.update(font=dict(size=30))
    fig.update_traces(textfont_size=30, selector=dict(type="bar"))


# Per-dataset y-axis range for the fig1 panels, in % (see _refit_fig1_axes).
FIG1_Y_RANGE: dict[str, tuple[float, float]] = {
    "fig1_sgs_over_tcn_metr_la": (-2.0, 16.0),  # data spans -1.2 .. +14.6
    "fig1_sgs_over_tcn_pems_bay": (-4.0, 12.0),  # data spans -2.6 .. +9.7
}


def _apply_fig1_display_names(fig) -> None:
    """Rename fig1's legend entries to table names and sort them by family.

    The locally cached probe HTML predates the probe's display-name change, so
    its traces still carry raw config slugs (``gwn_v2``, ``dssa_tcn``). Applying
    the probe's own mapping here keeps the paper figure's legend identical to
    the manuscript tables regardless of which vintage of the HTML is on disk.
    Both steps are no-ops on an already-renamed spec: unknown keys pass through
    display_name unchanged, and an equal family_rank leaves the (already
    taxonomy-ordered) sequence untouched because the sort is stable.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from analysis.model_display import display_name, family_rank

    fig.data = tuple(sorted(fig.data, key=lambda tr: family_rank(tr.name or "")))
    for tr in fig.data:
        tr.name = display_name(tr.name or "")
        tr.legendgroup = tr.name
    # The entries are self-evidently models; the title only costs panel width.
    fig.update_layout(legend_title_text="")


def _refit_fig1_axes(fig, base: str) -> None:
    """Retighten and relabel the axes of the baseline-relative-gain figure.

    Range: the probe pins y to (-33, 18) so that both datasets share a scale
    *with* ARIMA-excluded but TCN(shared) and ASTGCN still present -- those two
    dive to -48% and -34%, and set the floor. Both are dropped from this figure
    (see main), so the remaining seven models span only -2.6% to +14.6% and the
    old pin wastes about two-thirds of the panel.

    Each panel is now fitted to its own dataset (FIG1_Y_RANGE) rather than to a
    shared scale, so each uses its full height. The panels are consequently
    *not* height-comparable: METR-LA's gains are roughly twice PEMS-BAY's, and
    the differing axes hide that, so the caption says the ranges differ and the
    ratio claim in the text is carried by the numbers rather than the picture.
    Ranges stay fixed pins rather than autoranges so a re-run cannot quietly
    change the scale; the guard below fails loudly if the data outgrows one.

    Label: the plotted quantity is the aggregate relative SGS at horizon h
    (manuscript eq:sgs-agg), so the axis is named with that symbol rather than
    the probe's generic "% improvement" (which covers any baseline, ARIMA
    included). Referred to by label, not number, so it survives renumbering.
    """
    lo, hi = FIG1_Y_RANGE[base]
    ys = []
    for tr in fig.data:
        y = tr.y
        if isinstance(y, dict):  # plotly's base64 typed-array encoding
            import base64

            import numpy as np

            y = np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
        ys.extend(float(v) for v in y)
    if ys and not (lo <= min(ys) and max(ys) <= hi):
        raise ValueError(
            f"{base} data spans [{min(ys):.2f}, {max(ys):.2f}], outside the pinned "
            f"y-range {(lo, hi)}; update FIG1_Y_RANGE[{base!r}]."
        )
    fig.update_yaxes(
        range=[lo, hi],
        dtick=3,
        # SGS^agg_rel(h). <sup>/<sub> are honoured by the kaleido export.
        title_text="SGS<sup>agg</sup><sub>rel</sub> (%)",
    )
    # Only three horizons are evaluated; label them rather than letting plotly
    # pick round numbers (50/100/150/200) that no data point sits on.
    fig.update_xaxes(tickvals=[30, 60, 210], title_text="Horizon (min)")


def _relabel_fig2_axis(spec_json: str) -> str:
    """Title fig2's value axis "AAS", matching the manuscript and Table tab:probe.

    Older cached charts label the axis "AAS (F1)". AAS *is* the F1 of the two
    top-k edge sets, so the parenthetical adds nothing for a reader of the paper
    and invites confusion with a separate consensus-graph diagnostic the probe
    also emits. Mirrors the same rename in ``lens3_alignment.py`` so the export
    is correct whether or not the cached HTML predates it.
    """
    spec = json.loads(spec_json)

    def walk(node) -> None:
        if isinstance(node, dict):
            enc = node.get("encoding")
            if isinstance(enc, dict):
                channel = enc.get("y")
                if isinstance(channel, dict) and channel.get("title") == "AAS (F1)":
                    channel["title"] = "AAS"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return json.dumps(spec)


def _drop_vega_models(spec_json: str, names: set[str]) -> str:
    """Remove rows for the given model names from every inline Vega-Lite dataset.

    Model names are matched case-insensitively against the ``model`` field, so
    a dropped architecture (e.g. astgcn) is excluded from the bar chart without
    re-running the probe. Returns the modified spec as a JSON string.
    """
    spec = json.loads(spec_json)
    wanted = {n.lower() for n in names}
    datasets = spec.get("datasets")
    if isinstance(datasets, dict):
        for key, rows in datasets.items():
            if isinstance(rows, list):
                datasets[key] = [
                    r
                    for r in rows
                    if not (isinstance(r, dict) and str(r.get("model", "")).lower() in wanted)
                ]
    return json.dumps(spec)


def _enlarge_vega_fonts(spec_json: str) -> str:
    """Scale up label fonts of a Vega-Lite spec for print legibility.

    Vega counterpart of _enlarge_plotly_fonts. The AAS bar chart sets its fonts
    per encoding/layer (so a global ``config`` would not override them); we walk
    the spec and set axis label/title fonts and any text-mark (bar-value) fonts
    directly, and drop the redundant in-figure title. Returns the modified spec
    as a JSON string (the vega path passes specs around as strings).
    """
    spec = json.loads(spec_json)
    spec.pop("title", None)  # redundant with the LaTeX caption; declutters and frees space

    def walk(node) -> None:
        if isinstance(node, dict):
            enc = node.get("encoding")
            if isinstance(enc, dict):
                for ch in ("x", "y"):
                    channel = enc.get(ch)
                    if isinstance(channel, dict) and isinstance(channel.get("axis"), dict):
                        channel["axis"]["labelFontSize"] = 32
                        channel["axis"]["titleFontSize"] = 38
            mark = node.get("mark")
            if isinstance(mark, dict) and mark.get("type") == "text":
                mark["fontSize"] = 28
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return json.dumps(spec)


# --------------------------------------------------------------------------- #
# Per-format writers                                                          #
# --------------------------------------------------------------------------- #
def _write_native(kind: str, fig_or_spec, fmt: str, dst: Path, scale: int) -> None:
    """Write png/pdf/svg natively from the source library."""
    if kind == "plotly":
        fig_or_spec.write_image(str(dst), format=fmt, scale=scale)
    else:  # vega
        import vl_convert as vlc

        if fmt == "png":
            dst.write_bytes(vlc.vegalite_to_png(fig_or_spec, scale=scale))
        elif fmt == "pdf":
            dst.write_bytes(vlc.vegalite_to_pdf(fig_or_spec))
        elif fmt == "svg":
            dst.write_text(vlc.vegalite_to_svg(fig_or_spec))
        else:
            raise ValueError(fmt)


def _find_ghostscript() -> str | None:
    """Locate the Ghostscript binary (PATH first, then common Homebrew paths)."""
    import shutil

    for name in ("gs", "gsc", "gswin64c", "gswin32c"):
        found = shutil.which(name)
        if found:
            return found
    for path in ("/opt/homebrew/bin/gs", "/usr/local/bin/gs"):
        if Path(path).is_file():
            return path
    return None


def _eps_from_pdf_ghostscript(gs: str, pdf_path: Path, dst: Path) -> None:
    """TRUE-VECTOR EPS: convert a vector PDF to EPS via Ghostscript's eps2write."""
    import subprocess

    subprocess.run(
        [
            gs,
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=eps2write",
            f"-sOutputFile={dst}",
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
    )


def _write_eps_from_png(png_path: Path, dst: Path) -> None:
    """Fallback EPS: wrap an existing raster PNG in an EPS container (not vector)."""
    from PIL import Image

    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            flat = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            flat = img.convert("RGB")
        dpi = img.info.get("dpi", (150, 150))[0] or 150
        flat.save(dst, format="EPS", dpi=(dpi, dpi))


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("paper_figures"), help="Output directory.")
    parser.add_argument(
        "--formats",
        default="png,pdf,eps",
        help="Comma-separated: png,pdf,svg,eps (default: png,pdf,eps).",
    )
    parser.add_argument(
        "--scale", type=int, default=3, help="Raster upscaling for PNG (default: 3)."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root for source paths.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated output-name prefixes to rebuild (e.g. 'fig1,fig2'). "
        "Default: all. Use this to refresh just the figures the manuscript "
        "includes, without writing the unused ones alongside them.",
    )
    args = parser.parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - {"png", "pdf", "svg", "eps"}
    if unknown:
        print(f"Unknown format(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    gs = _find_ghostscript() if "eps" in formats else None
    if "eps" in formats:
        if gs:
            print(f"EPS: true vector via Ghostscript ({gs}).\n")
        else:
            print(
                "NOTE: Ghostscript not found -> EPS falls back to a raster wrapper.\n"
                f"      Install gs (brew install ghostscript) & re-run, or use '{VECTOR_HINT}'.\n"
            )

    only = tuple(p.strip() for p in args.only.split(",") if p.strip())
    selected = [(s, b) for s, b in FIGURES if not only or b.startswith(only)]
    if only and not selected:
        print(f"--only {args.only!r} matched no figures.", file=sys.stderr)
        return 2

    ok, failed = 0, 0
    for rel_src, base in selected:
        src = args.root / rel_src
        if not src.exists():
            print(f"  MISSING source: {src}", file=sys.stderr)
            failed += 1
            continue
        html = src.read_text()
        kind = _detect_kind(html)
        loaded = _load_plotly(html) if kind == "plotly" else _load_vega(html)
        # Enlarge labels for print on the spatial-gain-over-TCN figure.
        if kind == "plotly" and base.startswith("fig1_sgs_over_tcn"):
            # tcn_shared is a baseline control, not an STGNN, and astgcn has been
            # dropped from the study; neither belongs in a figure about STGNN
            # baseline-relative gain, so drop those traces.
            _dropped = {"tcn_shared", "astgcn"}
            loaded.data = tuple(tr for tr in loaded.data if (tr.name or "").lower() not in _dropped)
            _enlarge_plotly_fonts(loaded)
            # Must follow the drop: the range is fitted to what remains.
            _refit_fig1_axes(loaded, base)
            _apply_fig1_display_names(loaded)  # after the drop, which matches on raw slugs
        if kind == "vega" and base.startswith("fig2_aas_alignment"):
            # astgcn has been dropped from the study; remove its dataset row.
            loaded = _drop_vega_models(loaded, {"astgcn(r)", "astgcn"})
            loaded = _relabel_fig2_axis(loaded)
            loaded = _enlarge_vega_fonts(loaded)

        # EPS needs a vector PDF (for gs) or the PNG (for the raster fallback).
        want_eps = "eps" in formats
        pdf_path = args.out / f"{base}.pdf"
        png_path = args.out / f"{base}.png"
        need_pdf = "pdf" in formats or (want_eps and gs is not None)
        need_png = "png" in formats or (want_eps and gs is None)
        produced: list[str] = []
        try:
            if need_png:
                _write_native(kind, loaded, "png", png_path, args.scale)
                produced.append("png")
            if need_pdf:
                _write_native(kind, loaded, "pdf", pdf_path, args.scale)
                produced.append("pdf")
            if "svg" in formats:
                _write_native(kind, loaded, "svg", args.out / f"{base}.svg", args.scale)
                produced.append("svg")
            if want_eps:
                if gs is not None:
                    _eps_from_pdf_ghostscript(gs, pdf_path, args.out / f"{base}.eps")
                    produced.append("eps(vector)")
                else:
                    _write_eps_from_png(png_path, args.out / f"{base}.eps")
                    produced.append("eps(raster)")
            # Drop any scratch files rendered only as an EPS input.
            if "pdf" not in formats and pdf_path.exists():
                pdf_path.unlink()
                produced = [p for p in produced if p != "pdf"]
            if "png" not in formats and png_path.exists():
                png_path.unlink()
                produced = [p for p in produced if p != "png"]
            print(f"  [{kind:6}] {base}: {', '.join(produced)}")
            ok += 1
        except Exception as e:  # noqa: BLE001 — report and continue the batch
            print(f"  FAILED {base}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {ok} ok, {failed} failed -> {args.out}/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
