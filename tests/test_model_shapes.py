from __future__ import annotations

import pytest
from model_test_utils import (
    B,
    H,
    N,
    assert_finite_float_output,
    graph_model_specs,
)


@pytest.mark.parametrize(("name", "build_adapter", "make_x"), graph_model_specs())
def test_graph_model_forward_outputs_are_finite(name, build_adapter, make_x) -> None:
    adapter = build_adapter()
    adapter.eval()
    x = make_x()

    out = adapter(x)

    assert_finite_float_output(out, (B, N, H))
