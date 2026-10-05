from __future__ import annotations

import math

import pytest

from analysis.cross_model import (
    compute_cross_model_correlations,
    excess_modularity,
)


def _summary(sgs: float, aas: float, mod: float, null: float) -> dict[str, float]:
    return {
        "mean_sgs_rel": sgs,
        "aas_evaluation_grand_mean": aas,
        "community_modularity_evaluation_grand_mean": mod,
        "community_modularity_null": null,
    }


def test_excess_modularity_is_observed_minus_null() -> None:
    assert excess_modularity(_summary(0.1, 0.2, 0.6, 0.25)) == pytest.approx(0.35)


def test_perfect_monotone_alignment_gives_rho_one() -> None:
    # AAS and excess modularity increase together across models -> rho = +1.
    summaries = {
        f"m{i}": _summary(sgs=0.0, aas=0.1 * i, mod=0.1 * i + 0.3, null=0.2)
        for i in range(5)
    }
    out = compute_cross_model_correlations(summaries)
    assert out["n"] == 5
    assert out["skipped"] == []
    corr = out["correlations"]["aas__excess_modularity"]
    assert corr["rho"] == pytest.approx(1.0)
    assert corr["p_value"] < 0.05
    assert corr["method"] == "permutation_exact"


def test_reversed_order_gives_rho_minus_one() -> None:
    summaries = {
        f"m{i}": _summary(sgs=0.1 * i, aas=-0.1 * i, mod=0.3, null=0.2)
        for i in range(4)
    }
    corr = compute_cross_model_correlations(summaries)["correlations"]["sgs_rel__aas"]
    assert corr["rho"] == pytest.approx(-1.0)


def test_models_missing_grand_mean_keys_are_skipped() -> None:
    summaries = {
        "good1": _summary(0.1, 0.2, 0.5, 0.2),
        "good2": _summary(0.2, 0.3, 0.6, 0.2),
        "good3": _summary(0.3, 0.4, 0.7, 0.2),
        "no_seed_summary": {"mean_sgs_rel": 0.4},  # consensus-only fallback model
    }
    out = compute_cross_model_correlations(summaries)
    assert out["skipped"] == ["no_seed_summary"]
    assert set(out["models"]) == {"good1", "good2", "good3"}
    assert out["n"] == 3


def test_insufficient_models_reports_nan_not_error() -> None:
    out = compute_cross_model_correlations({"a": _summary(0.1, 0.2, 0.5, 0.2)})
    corr = out["correlations"]["aas__excess_modularity"]
    assert math.isnan(corr["rho"])
    assert corr["method"] == "insufficient_n"
