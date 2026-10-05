"""Display names and family ordering for models in paper figures.

Figure axes and legends should use the same human-readable names and the same
taxonomy (family) ordering as the manuscript tables, rather than raw config
slugs sorted by value. Keys may be bare slugs (``"stawnet"``) or dataset-
prefixed (``"PEMS-BAY:stawnet"``); both are handled.
"""

from __future__ import annotations

# Human-readable names, matching the manuscript tables.
MODEL_DISPLAY: dict[str, str] = {
    "gwn": "GWN",
    "gwn_v2": "GWN v2",
    "stawnet": "STAWnet",
    "dssa_tcn": "DSSA-TCN",
    "staeformer": "STAEformer",
    "d2stgnn": "D2STGNN",
    "bigst": "BigST",
    "tcn": "TCN",
    "arima": "ARIMA",
}

# Taxonomy (family) order used for legends and bar order, matching the tables:
# graph-wavenet family, then attention/adaptive, then decoupled/linear, baselines last.
FAMILY_ORDER: tuple[str, ...] = (
    "gwn",
    "gwn_v2",
    "stawnet",
    "dssa_tcn",
    "staeformer",
    "d2stgnn",
    "bigst",
    "tcn",
    "arima",
)


def _slug(key: str) -> str:
    """Bare model slug, dropping any ``dataset:`` prefix."""
    return key.split(":")[-1]


def display_name(key: str) -> str:
    """Human-readable model name; unknown slugs pass through unchanged."""
    slug = _slug(key)
    return MODEL_DISPLAY.get(slug, slug)


def family_rank(key: str) -> int:
    """Sort index in taxonomy order; unknown slugs sort to the end."""
    slug = _slug(key)
    return FAMILY_ORDER.index(slug) if slug in FAMILY_ORDER else len(FAMILY_ORDER)


def display_order(keys: list[str]) -> list[str]:
    """Display names for ``keys`` sorted into taxonomy order."""
    return [display_name(k) for k in sorted(keys, key=family_rank)]
