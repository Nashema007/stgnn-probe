from __future__ import annotations

import pytest
import torch
from model_test_utils import graph_model_specs, seed_all, torch_model_specs


@pytest.mark.parametrize(("name", "build_model", "forward_model"), torch_model_specs())
def test_same_seed_produces_same_initial_output(name, build_model, forward_model) -> None:
    seed_all(23)
    first = build_model()
    first.eval()
    with torch.no_grad():
        out_a = forward_model(first)

    seed_all(23)
    second = build_model()
    second.eval()
    with torch.no_grad():
        out_b = forward_model(second)

    assert torch.allclose(out_a, out_b, atol=1e-6)


@pytest.mark.parametrize(("name", "build_adapter", "make_x"), graph_model_specs())
def test_eval_mode_is_stable_for_identical_inputs(name, build_adapter, make_x) -> None:
    adapter = build_adapter()
    adapter.eval()
    x = make_x()

    with torch.no_grad():
        out_a = adapter(x)
        out_b = adapter(x)

    assert torch.allclose(out_a, out_b, atol=1e-6)
