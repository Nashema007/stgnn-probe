"""Process resource usage sampling for training logs.

``cpu_percent`` is measured relative to one CPU core, so it can exceed 100%
on multi-core machines while a process is using several threads — same
convention as ``top``/``htop``.
"""

from __future__ import annotations

import psutil
import torch


class ResourceSampler:
    """Tracks memory + CPU usage since this sampler was created.

    Each instance owns its own ``psutil.Process`` handle, so nesting two
    samplers (e.g. one per training run, one per epoch inside that run) does
    not corrupt either one's "since last call" CPU-time baseline — that
    state lives on the ``Process`` object, not globally per-PID.
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)  # prime: first call is always 0.0

    def sample(self) -> tuple[float, float]:
        """Return ``(rss_mb, cpu_percent)`` since this sampler was created or last sampled."""
        rss_mb = self._process.memory_info().rss / (1024**2)
        cpu_percent = self._process.cpu_percent(interval=None)
        return rss_mb, cpu_percent


def gpu_memory_metrics(device: torch.device) -> dict[str, float]:
    """Return current and cumulative peak CUDA memory, or zeros off CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_peak_allocated_mb": 0.0,
            "gpu_peak_reserved_mb": 0.0,
        }
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
