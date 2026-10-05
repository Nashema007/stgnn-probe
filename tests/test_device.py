from __future__ import annotations

import pytest
import torch

from training.device import default_device, resolve_device


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_default_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert default_device().type == "cuda"


def test_default_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(mps_backend, "is_available", lambda: False)

    assert default_device().type == "cpu"


def test_resolve_device_fails_clearly_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("quantum")
