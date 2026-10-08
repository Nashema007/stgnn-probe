#!/usr/bin/env python
"""Build the manuscript figure set in ``figures/`` with print-layout fixes.

``regenerate_figures.py`` rebuilds the full eight-figure bundle from the probe's
own cached chart HTML. This script produces only the two figures the manuscript
includes, and applies two layout changes for the two-column print layout:

  fig1 (baseline-relative gain vs horizon)
      "improve visibility for the two column/subfig layout. You can use a single
      legend, y-axis, ... instead of repeating?"

      The per-panel legend was consuming roughly a third of each panel's width,
      which at 0.4\\textwidth left very little plot. The legend is dropped from
      both panels and emitted once as a standalone horizontal strip
      (``fig1_sgs_legend``) that the manuscript places beneath them. A shared
      *y-axis* is deliberately NOT used: METR-LA's gains are about twice
      PEMS-BAY's, and forcing one scale would flatten the smaller panel. The
      caption says the ranges differ, and the ratio claim is carried by the
      numbers rather than by curve height.

  fig2 (AAS vs the Granger reference)
      "add the expected overlap value as a horizontal dashed line in each panel"

      Each panel now draws k/(N-1) as a dashed rule, so a bar can be read
      against chance directly. Panel spacing is handled on the LaTeX side.

      AAS is computed here rather than read from the probe, because the probe
      scores exactly-uniform exported matrices by argpartition tie order. See
      _compute_aas: those evaluations are assigned chance, k/(N-1), instead.
      Every matrix is first put in the canonical source -> target orientation
      (``analysis.orientation``); DSSA-TCN and D2STGNN are transposed.

Data provenance
---------------
fig1 comes from the probe's cached lens0 chart HTML, exactly as
``regenerate_figures.py`` reads it; the SGS values there reproduce
``mean_sgs_rel`` in ``outputs/per_model/*/summary.json``.

fig2 does NOT come from the cached lens3 HTML. That HTML is doubly superseded:
it holds *consensus-graph* AAS at ``p_max=42`` (e.g. STAWnet 0.545), whereas
the manuscript reports the *horizon x seed grand mean* at fixed ``p=12``
(STAWnet 0.429). Rebuilding fig2 from that HTML
would contradict Table 3.

The published AAS comes from the completed fixed-lag ``p=12`` probe run, read
from ``outputs_fixed12/per_model/<model>_<dataset>/structure_seed_summary.json``
under key ``aas.evaluation_grand_mean``. That tree ships in
``outputs_fixed12_brinjal_20260816_183843.tar.gz``, the first fixed-p=12 bundle
covering all 7 models on both datasets. Superseded sources that must NOT be
used here:

  * ``outputs/``                -- the ``p_max=42`` run (STAWnet 0.4188).
  * ``outputs_pmax12/``         -- AIC bounded at ``p_max=12``; agrees with
                                   fixed-p=12 on 11 of 14 cells at 3 dp but
                                   differs on PEMS-BAY STAWnet, STAEformer and
                                   BigST.
  * earlier ``outputs_fixed12`` bundles -- incomplete on PEMS-BAY.

``--aas-source`` accepts either the ``.tar.gz`` bundle or an already-extracted
directory, and defaults to whichever is present. All 14 loaded values are
asserted against ``AAS_P12_EXPECTED`` (the values the manuscript reports) so the
figure can never silently drift from the manuscript; a mismatch is a hard
error naming the model.

Usage
-----
    python scripts/finalize_paper_figures.py                  # -> figures/
    python scripts/finalize_paper_figures.py --out build/figs
    python scripts/finalize_paper_figures.py --aas-source outputs_fixed12
    python scripts/finalize_paper_figures.py --aas-source outputs --no-manuscript-check   # a re-run
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regenerate_figures import (  # noqa: E402
    _apply_fig1_display_names,
    _detect_kind,
    _enlarge_plotly_fonts,
    _enlarge_vega_fonts,
    _eps_from_pdf_ghostscript,
    _find_ghostscript,
    _load_plotly,
    _refit_fig1_axes,
    _write_native,
)

from analysis.orientation import to_incoming_convention  # noqa: E402

FIG1_SOURCES: list[tuple[str, str]] = [
    (
        "outputs/lens0_performance/METR-LA/figures/improvement_over_tcn_no_arima.html",
        "fig1_sgs_over_tcn_metr_la",
    ),
    (
        "outputs/lens0_performance/PEMS-BAY/figures/improvement_over_tcn_no_arima.html",
        "fig1_sgs_over_tcn_pems_bay",
    ),
]

# Not STGNNs / dropped from the study; mirrors regenerate_figures.py.
FIG1_DROP = {"tcn_shared", "astgcn"}

# Taxonomy order, matching Table 1 and Table 3, with the probe's config slugs.
MODELS = ["GWN", "GWN v2", "STAWnet", "DSSA-TCN", "STAEformer", "D2STGNN", "BigST"]
MODEL_SLUGS = ["gwn", "gwn_v2", "stawnet", "dssa_tcn", "staeformer", "d2stgnn", "bigst"]

# The AAS values as they appear in the manuscript, to 3 dp. AAS is not tabulated
# in tab:probe, so these surface only in fig2's bars and the
# Section 5.4 range sentence. Not the data source -- a tripwire: values are
# loaded from the fixed-p=12 run (see "Data provenance") and checked against
# these, so a figure/manuscript divergence fails loudly instead of shipping.
AAS_P12_EXPECTED: dict[str, list[float]] = {
    "METR-LA": [0.045, 0.065, 0.429, 0.050, 0.373, 0.053, 0.489],
    "PEMS-BAY": [0.024, 0.041, 0.158, 0.025, 0.193, 0.031, 0.211],
}

# Where the completed fixed-p=12 probe outputs live, in preference order.
AAS_SOURCE_CANDIDATES = [
    "outputs_fixed12",
    "outputs_fixed12_brinjal_20260816_183843.tar.gz",
]
AAS_TREE = "outputs_fixed12"  # the completed fixed-p=12 run; other trees are superseded
AAS_KEY = ("aas", "evaluation_grand_mean")

# Sensor counts, for the Eq. (aas-null) random-selection expectation k/(N-1).
NUM_NODES = {"METR-LA": 207, "PEMS-BAY": 325}
TOP_K = 10

FIG2_BASENAME = {"METR-LA": "fig2_aas_alignment_metr_la", "PEMS-BAY": "fig2_aas_alignment_pems_bay"}


def _strip_legend(fig) -> None:
    """Hide the per-panel legend and reclaim the width it occupied."""
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20, autoexpand=True))


def _build_legend_strip(fig):
    """Return a legend-only figure carrying fig1's series, laid out horizontally.

    The traces are copied so the strip inherits the panels' exact colours and
    marker styles, then emptied of data so only the legend renders. Both fig1
    panels carry the same seven series in the same order, so one strip serves
    both.
    """
    strip = copy.deepcopy(fig)
    for tr in strip.data:
        tr.x, tr.y = [None], [None]
        tr.showlegend = True
    # The panels carry a zero-parity rule as a layout shape; it would otherwise
    # be redrawn across the legend strip.
    strip.layout.shapes = ()
    strip.layout.annotations = ()
    strip.update_layout(
        showlegend=True,
        # Sized so the seven entries nearly fill the canvas: the strip is drawn
        # at \textwidth, so surplus canvas would shrink the text. Widen this if
        # a model is added and the row wraps.
        width=1750,
        height=140,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=0.5,
            yanchor="middle",
            font=dict(size=44),
            title_text="",
            tracegroupgap=0,
            itemwidth=30,
        ),
        margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    strip.update_xaxes(visible=False, showgrid=False, zeroline=False)
    strip.update_yaxes(visible=False, showgrid=False, zeroline=False)
    return strip


def _resolve_aas_source(root: Path, explicit: Path | None) -> Path:
    """Locate the fixed-p=12 Granger cache: an extracted tree or the results bundle."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"--aas-source not found: {explicit}")
        return explicit
    for name in AAS_SOURCE_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No completed fixed-p=12 source found. Expected one of "
        f"{AAS_SOURCE_CANDIDATES} under {root}, or pass --aas-source. Do not "
        "substitute outputs/ (p_max=42), outputs_pmax12/ (AIC-bounded) or an "
        "earlier outputs_fixed12 bundle (incomplete on PEMS-BAY)."
    )


def _load_granger_fstats(source: Path) -> dict[str, np.ndarray]:
    """Read the fixed-p=12 Granger F-statistics for both datasets."""
    import numpy as np

    rel = {ds: f"granger_cache/{ds}.npz" for ds in NUM_NODES}
    out: dict[str, np.ndarray] = {}
    if source.is_dir():
        for ds, path in rel.items():
            f = source / path
            if not f.is_file():
                raise FileNotFoundError(f"missing {f}")
            out[ds] = np.load(f)["fstats"]
    else:
        wanted = {f"{AAS_TREE}/{p}": ds for ds, p in rel.items()}
        with tarfile.open(source) as tar:
            for member in tar:
                name = wanted.get(member.name.lstrip("./"))
                if name is None:
                    continue
                handle = tar.extractfile(member)
                if handle is not None:
                    out[name] = np.load(io.BytesIO(handle.read()))["fstats"]
        missing = set(rel) - set(out)
        if missing:
            raise FileNotFoundError(f"{source} is missing {sorted(missing)} under {AAS_TREE}/")
    return out


def _topk_incoming(weights, k: int):
    """Keep each target node's k strongest incoming edges.

    Mirrors ``analysis.lens2_granger.build_gcg_topk``: self-loops excluded, and
    edges backed by a non-positive weight dropped (which affects only the
    Granger reference's zeroed diagonal, never a softmax-valued model matrix).
    """
    import numpy as np

    f = np.array(weights, dtype=np.float64, copy=True)
    n = f.shape[0]
    np.fill_diagonal(f, -np.inf)
    k = max(0, min(k, n - 1))
    g = np.zeros((n, n), dtype=bool)
    idx = np.argpartition(-f, kth=k - 1, axis=0)[:k, :]
    g[idx, np.broadcast_to(np.arange(n), (k, n))] = True
    g[f <= 0.0] = False
    return g


def _compute_aas(source: Path, root: Path) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Compute AAS per model, scoring exactly-uniform representations at chance.

    An exactly uniform exported matrix (every entry 1/N) carries no ranking
    among candidate sources, so its top-k set would be decided by
    ``argpartition``'s tie order -- arbitrary, and implementation-dependent.
    Such an evaluation is instead assigned the expected AAS under uniform
    random tie resolution, k/(N-1). Degenerate evaluations are neither dropped
    (which would leave models averaged over different numbers of horizons, and
    would flatter a model for having learned nothing) nor scored zero
    (uniformity means "no preference", not "confirmed no overlap").

    Returns the per-dataset AAS in ``MODELS`` order, and the count of uniform
    representations encountered.
    """
    import numpy as np

    fstats = _load_granger_fstats(source)
    dirs = {"METR-LA": "metr_la", "PEMS-BAY": "pems_bay"}
    out: dict[str, list[float]] = {}
    uniform: dict[str, int] = {}

    for ds, n in NUM_NODES.items():
        reference = _topk_incoming(fstats[ds], TOP_K)
        n_ref = int(reference.sum())
        chance = TOP_K / (n - 1)
        row, n_uniform = [], 0
        for slug in MODEL_SLUGS:
            scores = []
            for horizon in (6, 12, 42):
                adjacency_dir = root / f"data/probe_inputs/{dirs[ds]}/adjacency"
                path = adjacency_dir / f"{slug}_adjacency_h{horizon}_seeds.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"missing adjacency {path}")
                stack = np.load(path)
                if stack.ndim == 2:
                    stack = stack[None]
                for seed in range(stack.shape[0]):
                    matrix = to_incoming_convention(stack[seed], slug)
                    if len(np.unique(matrix)) == 1:  # exactly uniform -> no ranking
                        scores.append(chance)
                        n_uniform += 1
                        continue
                    learned = _topk_incoming(matrix, TOP_K)
                    shared = int((learned & reference).sum())
                    precision = shared / max(int(learned.sum()), 1)
                    recall = shared / max(n_ref, 1)
                    scores.append(
                        0.0
                        if precision + recall == 0
                        else 2 * precision * recall / (precision + recall)
                    )
            row.append(float(np.mean(scores)))
        out[ds] = row
        uniform[ds] = n_uniform
    return out, uniform


def _check_against_table3(loaded: dict[str, list[float]]) -> None:
    """Fail loudly if the computed AAS no longer rounds to the manuscript values."""
    bad = [
        f"{ds}/{name}: computed {v:.4f} -> {round(v, 3)}, manuscript says {e:.3f}"
        for ds, expected in AAS_P12_EXPECTED.items()
        for name, v, e in zip(MODELS, loaded[ds], expected, strict=True)
        if round(v, 3) != e
    ]
    if bad:
        raise ValueError(
            "Computed AAS disagrees with the manuscript:\n  "
            + "\n  ".join(bad)
            + "\nEither the wrong source was read, or the manuscript needs updating."
        )


def _build_fig2_spec(dataset: str, values: list[float]) -> str:
    """Build the AAS bar chart for one dataset, with the random-overlap rule."""
    expected = TOP_K / (NUM_NODES[dataset] - 1)
    # Headroom for the value labels above each bar; 1.28 reproduces the domain
    # of the previously approved figure (0.489 -> 0.627 on METR-LA).
    domain = [0, max(values) * 1.28]
    rows = [{"model": m, "AAS": v} for m, v in zip(MODELS, values, strict=True)]

    x_enc = {
        "field": "model",
        "type": "nominal",
        "title": None,
        "sort": MODELS,
        "axis": {"labelAngle": -30, "labelFontSize": 14, "tickSize": 4},
    }
    y_enc = {
        "field": "AAS",
        "type": "quantitative",
        "title": "AAS",
        "scale": {"domain": domain},
        "axis": {"labelFontSize": 13, "titleFontSize": 15},
    }

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.4.1.json",
        "config": {"view": {"continuousWidth": 300, "continuousHeight": 300, "strokeWidth": 0}},
        "data": {"values": rows},
        "width": 720,
        "height": 350,
        "layer": [
            {
                "mark": {"type": "bar", "color": "steelblue", "opacity": 0.85},
                "encoding": {"x": x_enc, "y": y_enc},
            },
            {
                # Eq. (aas-null): expected overlap under independent random
                # edge selection. HM asked for this as a dashed line so each
                # bar can be read against chance. Drawn before the value
                # labels so a label on a near-chance bar stays legible where
                # the two coincide.
                "mark": {
                    "type": "rule",
                    "strokeDash": [10, 7],
                    "color": "#333333",
                    "strokeWidth": 3,
                },
                "encoding": {
                    "y": {
                        "datum": expected,
                        "type": "quantitative",
                        "scale": {"domain": domain},
                    }
                },
            },
            {
                "mark": {"type": "text", "dy": -6, "fontSize": 13},
                "encoding": {
                    "x": x_enc,
                    "y": y_enc,
                    "text": {"field": "AAS", "type": "quantitative", "format": ".3f"},
                },
            },
        ],
    }
    return _enlarge_vega_fonts(json.dumps(spec))


def _emit(kind, obj, base: str, out: Path, formats: list[str], scale: int, gs: str | None) -> None:
    """Write one figure in every requested format, routing EPS through a vector PDF."""
    pdf_path = out / f"{base}.pdf"
    png_path = out / f"{base}.png"
    want_eps = "eps" in formats
    produced: list[str] = []

    if "png" in formats:
        _write_native(kind, obj, "png", png_path, scale)
        produced.append("png")
    if "pdf" in formats or (want_eps and gs is not None):
        _write_native(kind, obj, "pdf", pdf_path, scale)
        produced.append("pdf")
    if "svg" in formats:
        _write_native(kind, obj, "svg", out / f"{base}.svg", scale)
        produced.append("svg")
    if want_eps:
        if gs is None:
            raise RuntimeError("Ghostscript not found; EPS requires it for a true-vector export.")
        _eps_from_pdf_ghostscript(gs, pdf_path, out / f"{base}.eps")
        produced.append("eps")
    if "pdf" not in formats and pdf_path.exists():
        pdf_path.unlink()
        produced = [p for p in produced if p != "pdf"]

    print(f"  {base}: {', '.join(produced)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--formats", default="pdf,eps")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--aas-source",
        type=Path,
        default=None,
        help="Directory holding per_model/, or a .tar.gz bundle holding "
        f"{AAS_TREE}/per_model/. Defaults to the first of "
        f"{AAS_SOURCE_CANDIDATES} that exists under --root.",
    )
    parser.add_argument(
        "--no-manuscript-check",
        action="store_true",
        help="Skip asserting the computed AAS against the published values; needed for a "
        "re-run, whose seeds differ from the paper's.",
    )
    args = parser.parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    gs = _find_ghostscript() if "eps" in formats else None
    if "eps" in formats and gs is None:
        print("Ghostscript not found; install it or drop 'eps' from --formats.", file=sys.stderr)
        return 2

    print(f"fig1: panels without legends + one shared legend strip -> {args.out}/")
    legend_written = False
    for rel_src, base in FIG1_SOURCES:
        src = args.root / rel_src
        if not src.exists():
            print(f"  MISSING source: {src}", file=sys.stderr)
            return 1
        html = src.read_text()
        if _detect_kind(html) != "plotly":
            print(f"  {src} is not a Plotly chart", file=sys.stderr)
            return 1
        fig = _load_plotly(html)
        fig.data = tuple(tr for tr in fig.data if (tr.name or "").lower() not in FIG1_DROP)
        _enlarge_plotly_fonts(fig)
        _refit_fig1_axes(fig, base)
        _apply_fig1_display_names(fig)
        if not legend_written:
            _emit(
                "plotly",
                _build_legend_strip(fig),
                "fig1_sgs_legend",
                args.out,
                formats,
                args.scale,
                gs,
            )
            legend_written = True
        _strip_legend(fig)
        _emit("plotly", fig, base, args.out, formats, args.scale, gs)

    aas_source = _resolve_aas_source(args.root, args.aas_source)
    aas, uniform = _compute_aas(aas_source, args.root)
    if args.no_manuscript_check:
        print(f"fig2: AAS computed against {aas_source} (manuscript check skipped)")
        for ds, values in aas.items():
            pairs = zip(MODELS, values, strict=True)
            print(f"  {ds}: " + ", ".join(f"{m} {v:.3f}" for m, v in pairs))
    else:
        _check_against_table3(aas)
        print(f"fig2: AAS computed against {aas_source} ({AAS_TREE}); all 14 match the manuscript")
    total_uniform = sum(uniform.values())
    print(
        f"  exactly-uniform representations scored at chance k/(N-1): "
        f"{total_uniform} of 126 "
        f"({uniform['METR-LA']} METR-LA, {uniform['PEMS-BAY']} PEMS-BAY)"
    )
    for dataset, base in FIG2_BASENAME.items():
        expected = TOP_K / (NUM_NODES[dataset] - 1)
        print(f"  {dataset}: expected random overlap k/(N-1) = {expected:.3f}")
        _emit(
            "vega", _build_fig2_spec(dataset, aas[dataset]), base, args.out, formats, args.scale, gs
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
