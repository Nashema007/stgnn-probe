"""Per-node StandardScaler for traffic time-series data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


class StandardScaler:
    """Per-node z-score normalization fitted on channel 0.

    Expects data in shape *(T, N, C)*. Mean and std are computed per-node
    over the time axis using channel 0 (the traffic speed/flow channel).
    The same statistics are applied to all channels at inference time.

    Example::

        scaler = StandardScaler()
        scaler.fit(train_data)          # train_data: (T_train, N, C)
        x_norm = scaler.transform(x)    # x: (T, N, C) or (B, N, H) tensor
        x_orig = scaler.inverse_transform(x_norm)
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None  # (N,)
        self.std: np.ndarray | None = None  # (N,)

    def fit(self, data: np.ndarray) -> StandardScaler:
        """Fit mean/std from channel 0 of *data* shaped *(T, N, C)*."""
        data = np.asarray(data, dtype=np.float32)
        self.mean = data[:, :, 0].mean(axis=0)  # (N,)
        self.std = data[:, :, 0].std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet. Call fit() first.")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize *data* shaped *(T, N, C)*. Returns same shape."""
        self._check_fitted()
        assert self.mean is not None
        assert self.std is not None
        data = np.asarray(data, dtype=np.float32).copy()
        data[:, :, 0] = (data[:, :, 0] - self.mean) / self.std
        return data

    def inverse_transform(
        self,
        data: Tensor | np.ndarray,
        node_indices: Sequence[int] | np.ndarray | None = None,
    ) -> Tensor | np.ndarray:
        """Denormalize predictions shaped *(B, N, H)* or *(T, N, C)*.

        Accepts both numpy arrays and torch tensors. Returns the same type.

        ``node_indices`` must be passed whenever *data*'s node axis has
        already been sliced down to a subset of the ``N`` nodes this
        scaler was fit on (e.g. one node at a time) — it selects the
        matching ``mean``/``std`` entries so denormalization uses each
        node's own statistics instead of broadcasting node 0's against
        every slice. Defaults to all ``N`` nodes in order, matching the
        unsliced case.
        """
        self._check_fitted()
        assert self.mean is not None
        assert self.std is not None
        if node_indices is None:
            mean_np = self.mean
            std_np = self.std
        else:
            idx = np.asarray(node_indices)
            mean_np = self.mean[idx]
            std_np = self.std[idx]

        if isinstance(data, Tensor):
            mean = torch.as_tensor(mean_np, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(std_np, dtype=data.dtype, device=data.device)

            if data.ndim == 3:
                # (B, N, H) or (T, N, C) — node axis is dim 1
                return data * std[None, :, None] + mean[None, :, None]
            if data.ndim == 4:
                # (B, C, N, T) — node axis is dim 2
                return data * std[None, None, :, None] + mean[None, None, :, None]
            raise ValueError(f"Unsupported data ndim {data.ndim}; expected 3 or 4.")

        arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 3:
            # (B, N, H) or (T, N, C) — node axis is dim 1
            arr = arr * std_np[np.newaxis, :, np.newaxis] + mean_np[np.newaxis, :, np.newaxis]
        elif arr.ndim == 4:
            # (B, C, N, T) — node axis is dim 2
            arr = (
                arr * std_np[np.newaxis, np.newaxis, :, np.newaxis]
                + mean_np[np.newaxis, np.newaxis, :, np.newaxis]
            )
        else:
            raise ValueError(f"Unsupported data ndim {arr.ndim}; expected 3 or 4.")

        return arr
