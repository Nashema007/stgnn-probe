"""Runtime device selection for CPU, CUDA, and MPS training."""

from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve a requested training device to a validated ``torch.device``.

    ``auto`` chooses CUDA first, then MPS, then CPU. Explicit ``cuda`` or
    ``mps`` requests fail clearly when that backend is unavailable.
    """
    if isinstance(requested, torch.device):
        device = requested
    else:
        value = requested.strip().lower()
        if value in {"", "auto"}:
            return default_device()
        try:
            device = torch.device(value)
        except RuntimeError as exc:
            raise ValueError("device must be one of 'auto', 'cpu', 'cuda', or 'mps'.") from exc

    if device.type == "cpu":
        return device
    if device.type == "cuda":
        if torch.cuda.is_available():
            return device
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return device
        raise RuntimeError("MPS was requested but is not available on this machine.")
    raise ValueError("device must be one of 'auto', 'cpu', 'cuda', or 'mps'.")


def default_device() -> torch.device:
    """Return the best available runtime device for this machine."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
