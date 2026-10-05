"""Cross-model rank correlations among probe diagnostics.

The probe reports, for each model, three scalar diagnostics: relative
spatial gain (SGS), alignment to the Granger reference (AAS) and excess
community modularity. This module correlates those diagnostics *across
models* — one observation per model, computed separately per dataset —
using Spearman's rank correlation.

The quantities used are exactly those displayed in the probe summary
table, so a reader can reconstruct the reported correlations from the
published table:

* SGS  = ``mean_sgs_rel`` (grand-mean relative gain);
* AAS  = ``aas_evaluation_grand_mean`` (grand mean over the horizon x seed
  evaluations, *not* the consensus-graph value);
* excess modularity = ``community_modularity_evaluation_grand_mean`` minus
  ``community_modularity_null`` (grand-mean observed minus grand-mean null).

For the small model counts here (n ~ 8) the p-value is an exact two-sided
permutation test; for larger n it falls back to the t approximation.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations
from math import sqrt

# Keys read from each model's ``ProbeResult.summary()`` dict.
_SGS_KEY = "mean_sgs_rel"
_AAS_KEY = "aas_evaluation_grand_mean"
_MOD_KEY = "community_modularity_evaluation_grand_mean"
_NULL_KEY = "community_modularity_null"

# Correlations to report, as (name, left diagnostic, right diagnostic).
_PAIRS = (
    ("sgs_rel__aas", "sgs_rel", "aas"),
    ("sgs_rel__excess_modularity", "sgs_rel", "excess_modularity"),
    ("aas__excess_modularity", "aas", "excess_modularity"),
)

# Exact permutation p-values are enumerated up to this many models
# (10! = 3.6M is the practical ceiling); larger n uses the t approximation.
_EXACT_MAX_N = 10


def excess_modularity(summary: Mapping[str, float]) -> float:
    """Grand-mean excess modularity: observed minus degree-preserving null."""
    return float(summary[_MOD_KEY]) - float(summary[_NULL_KEY])


def _average_ranks(values: list[float]) -> list[float]:
    """Ranks with ties broken by the mean rank (1-based)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    den = sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_average_ranks(x), _average_ranks(y))


def _permutation_p(x: list[float], y: list[float], rho: float) -> float:
    """Exact two-sided p: fraction of label permutations at least as extreme."""
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    target = abs(rho) - 1e-12
    at_least = 0
    total = 0
    for perm in permutations(rx):
        total += 1
        if abs(_pearson(list(perm), ry)) >= target:
            at_least += 1
    return at_least / total


def _t_approx_p(rho: float, n: int) -> float:
    """Two-sided p from the Student-t approximation (larger samples)."""
    if abs(rho) >= 1.0:
        return 0.0
    from scipy.stats import t  # local import; large-n path only

    stat = rho * sqrt((n - 2) / (1 - rho * rho))
    return float(2 * t.sf(abs(stat), df=n - 2))


def compute_cross_model_correlations(
    summaries: Mapping[str, Mapping[str, float]],
) -> dict:
    """Spearman correlations among SGS, AAS and excess modularity across models.

    Parameters
    ----------
    summaries:
        Mapping of model name to that model's ``ProbeResult.summary()`` dict.
        Models missing any required grand-mean key are skipped (they cannot be
        placed on the table-consistent scale) and listed under ``skipped``.

    Returns
    -------
    dict
        ``{"models", "skipped", "n", "diagnostics", "correlations"}`` where
        ``correlations`` maps each pair name to ``{"rho", "p_value", "method"}``.
    """
    required = (_SGS_KEY, _AAS_KEY, _MOD_KEY, _NULL_KEY)
    models: list[str] = []
    skipped: list[str] = []
    for name in sorted(summaries):
        if all(k in summaries[name] and summaries[name][k] is not None for k in required):
            models.append(name)
        else:
            skipped.append(name)

    diagnostics = {
        "sgs_rel": [float(summaries[m][_SGS_KEY]) for m in models],
        "aas": [float(summaries[m][_AAS_KEY]) for m in models],
        "excess_modularity": [excess_modularity(summaries[m]) for m in models],
    }

    correlations: dict[str, dict] = {}
    n = len(models)
    exact = 3 <= n <= _EXACT_MAX_N
    for pair_name, left, right in _PAIRS:
        if n < 3:
            correlations[pair_name] = {
                "rho": float("nan"),
                "p_value": float("nan"),
                "method": "insufficient_n",
            }
            continue
        x, y = diagnostics[left], diagnostics[right]
        rho = _spearman(x, y)
        if exact:
            p = _permutation_p(x, y, rho)
            method = "permutation_exact"
        else:
            p = _t_approx_p(rho, n)
            method = "t_approx"
        correlations[pair_name] = {"rho": rho, "p_value": p, "method": method}

    return {
        "models": models,
        "skipped": skipped,
        "n": n,
        "diagnostics": diagnostics,
        "correlations": correlations,
    }
