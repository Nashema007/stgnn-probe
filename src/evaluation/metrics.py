"""Traffic forecasting evaluation metrics.

All functions accept predictions and ground-truth tensors with shape
*(B, N, H)* where B = batch size, N = number of nodes, H = horizon.

Null values (default: 0.0) are masked out before computing metrics —
the standard convention for METR-LA and PEMS-BAY benchmarks. The masked
reductions themselves are delegated to ``tsl.metrics.torch`` so the loss
computed during training matches the library's own forecasting metrics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE

# ---------------------------------------------------------------------------
# Masking helper
# ---------------------------------------------------------------------------


def _mask(labels: Tensor, null_val: float) -> Tensor:
    if np.isnan(null_val):
        return ~torch.isnan(labels)
    return labels.abs() > 1e-4 if null_val == 0.0 else labels != null_val


# ---------------------------------------------------------------------------
# Primitive masked losses  (return scalar tensors)
# ---------------------------------------------------------------------------


def masked_mae(pred: Tensor, true: Tensor, null_val: float = 0.0) -> Tensor:
    """Mean Absolute Error, masking null values."""
    mask = _mask(true, null_val)
    # Each call constructs a fresh metric module (stateless usage — no
    # accumulation across calls), so it must be moved to pred's device:
    # torchmetrics' internal state starts on CPU and now raises on a
    # device mismatch instead of silently working, which surfaces as soon
    # as pred/true are on GPU.
    return MaskedMAE(mask_nans=True).to(pred.device)(pred, true, mask)


def masked_mape(pred: Tensor, true: Tensor, null_val: float = 0.0, eps: float = 1e-8) -> Tensor:
    """Mean Absolute Percentage Error, masking null values.

    ``eps`` is accepted for backward compatibility but unused — tsl's
    ``MaskedMAPE`` masks out the infinities produced by division-by-zero
    instead of clamping the denominator.
    """
    del eps
    mask = _mask(true, null_val)
    return MaskedMAPE(mask_nans=True).to(pred.device)(pred, true, mask)


def masked_rmse(pred: Tensor, true: Tensor, null_val: float = 0.0) -> Tensor:
    """Root Mean Squared Error, masking null values."""
    mask = _mask(true, null_val)
    mse = MaskedMSE(mask_nans=True).to(pred.device)(pred, true, mask)
    return torch.sqrt(mse)


# ---------------------------------------------------------------------------
# Aggregated convenience wrappers
# ---------------------------------------------------------------------------


def metric_global(
    pred: Tensor,
    true: Tensor,
    null_val: float = 0.0,
) -> tuple[float, float, float]:
    """Return *(MAE, MAPE, RMSE)* scalars averaged over all B, N, H."""
    mae = masked_mae(pred, true, null_val).item()
    mape = masked_mape(pred, true, null_val).item()
    rmse = masked_rmse(pred, true, null_val).item()
    return mae, mape, rmse


def metric_per_horizon(
    pred: Tensor,
    true: Tensor,
    null_val: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return per-horizon metrics, shape *(H,)* for each metric.

    Useful for horizon-degradation plots.
    """
    H = pred.shape[-1]
    mae_h, mape_h, rmse_h = np.zeros(H), np.zeros(H), np.zeros(H)
    for h in range(H):
        p, t = pred[..., h], true[..., h]
        mae_h[h] = masked_mae(p, t, null_val).item()
        mape_h[h] = masked_mape(p, t, null_val).item()
        rmse_h[h] = masked_rmse(p, t, null_val).item()
    return {"mae": mae_h, "mape": mape_h, "rmse": rmse_h}


def metric_per_node(
    pred: Tensor,
    true: Tensor,
    null_val: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return per-node metrics, shape *(N,)* for each metric.

    Useful for node-level analysis and adjacency influence validation.
    """
    N = pred.shape[1]
    mae_n, mape_n, rmse_n = np.zeros(N), np.zeros(N), np.zeros(N)
    for n in range(N):
        p, t = pred[:, n, :], true[:, n, :]
        mae_n[n] = masked_mae(p, t, null_val).item()
        mape_n[n] = masked_mape(p, t, null_val).item()
        rmse_n[n] = masked_rmse(p, t, null_val).item()
    return {"mae": mae_n, "mape": mape_n, "rmse": rmse_n}


# Back-compat alias
metric = metric_global
