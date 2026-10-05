from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from model_test_utils import (
    assert_has_nonzero_finite_gradient,
    assert_parameter_changed,
    canonical_target,
    clone_trainable_parameters,
    torch_model_specs,
)


@pytest.mark.parametrize(("name", "build_model", "forward_model"), torch_model_specs())
def test_pytorch_model_one_optimizer_step_has_finite_gradients(
    name, build_model, forward_model
) -> None:
    model = build_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = canonical_target()
    before = clone_trainable_parameters(model)

    pred = forward_model(model)
    loss = F.mse_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()

    assert torch.isfinite(loss)
    assert_has_nonzero_finite_gradient(model)
    optimizer.step()
    assert_parameter_changed(model, before)


@pytest.mark.parametrize(("name", "build_model", "forward_model"), torch_model_specs())
def test_pytorch_model_tiny_overfit_loss_decreases(name, build_model, forward_model) -> None:
    model = build_model()
    model.train()
    lr = 2e-2 if name == "staeformer" else 5e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = canonical_target()
    losses: list[float] = []

    for _ in range(6):
        optimizer.zero_grad()
        loss = F.mse_loss(forward_model(model), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]
