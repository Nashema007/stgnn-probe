from __future__ import annotations

from analysis.model_display import (
    FAMILY_ORDER,
    display_name,
    display_order,
    family_rank,
)


def test_display_name_handles_bare_and_prefixed_keys() -> None:
    assert display_name("stawnet") == "STAWnet"
    assert display_name("PEMS-BAY:stawnet") == "STAWnet"
    assert display_name("gwn_v2") == "GWN v2"


def test_unknown_slug_passes_through() -> None:
    assert display_name("mystery") == "mystery"
    assert family_rank("mystery") == len(FAMILY_ORDER)


def test_family_rank_follows_taxonomy_order() -> None:
    assert family_rank("gwn") < family_rank("stawnet") < family_rank("bigst")
    assert family_rank("METR-LA:gwn") < family_rank("METR-LA:dssa_tcn")


def test_display_order_sorts_into_taxonomy() -> None:
    keys = ["PEMS-BAY:bigst", "PEMS-BAY:gwn", "PEMS-BAY:stawnet"]
    assert display_order(keys) == ["GWN", "STAWnet", "BigST"]
