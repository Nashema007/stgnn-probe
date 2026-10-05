"""ARIMA trainer: rolling multi-step-ahead forecast over all nodes."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, cast

import numpy as np

from evaluation.metrics import masked_mae as _mae
from evaluation.metrics import masked_mape as _mape
from evaluation.metrics import masked_rmse as _rmse
from models.arima import RollingARIMA

from .env import resolve_wandb_entity


def _fit_predict_one_node(
    n_idx: int,
    train_series: np.ndarray,
    test_series: np.ndarray,
    order: tuple[int, int, int],
    max_h: int,
) -> tuple[int, dict[str, list[np.ndarray]] | None]:
    """Fit + rolling-forecast a single node. Module-level so it can be pickled
    for ``ProcessPoolExecutor`` — the per-node ARIMA fits are independent of
    each other (and of dataset/node count), so they're run concurrently
    across processes rather than in a Python loop.
    """
    try:
        model = RollingARIMA(order=order)
        model.fit(train_series)
        result = model.rolling_forecast(test_series, horizon=max_h, rolling_step=1)
        return n_idx, result
    except Exception:
        return n_idx, None


def predict_arima(
    data: np.ndarray,
    horizons: list[int],
    test_start: int,
    test_count: int,
    order: tuple[int, int, int] = (3, 0, 1),
    verbose: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Return per-window forecasts aligned to the shared tsl test windows.

    Shape ``(test_count, len(horizons), N)`` — paired 1:1 with the same test
    windows every neural model is evaluated on (see
    ``data.tsl_pipeline.compute_test_window_bounds``), so Lens 0 can compute
    metrics by averaging per-window error rather than averaging values over
    windows first.

    ``test_start`` must already be shifted past the input window, i.e. it is
    the raw-row index of the first test window's *forecast origin*
    (``window_input_start + in_len``), not the input-window start itself —
    callers building it from ``compute_test_window_bounds`` (which returns
    the input-window start) must add ``in_len`` before calling this function.
    A forecast made at origin ``test_start + w`` for horizon ``h`` then
    targets row ``test_start + w + h - 1``, which is exactly the row the
    matching neural-model window ``w`` predicts for the same horizon.

    A forecast is computed at every one of the ``test_count`` origins (no
    ``rolling_step`` subsampling or forward-filling): each origin only costs
    a Kalman-filter pass over the already-fit parameters
    (``RollingARIMA.rolling_forecast`` never re-optimises), so this is not
    the expensive per-origin refit the previous subsampling guarded against.
    Forward-filling skipped origins previously paired a forecast made at one
    origin against ground truth from a different, later origin — an invalid
    pairing — which this avoids entirely.

    History strictly before ``test_start`` (i.e. tsl's train+val region plus
    every test window's own input rows) is used to fit each node's ARIMA
    model. ARIMA has no early-stopping/hyperparameter-tuning step, so unlike
    the neural models it has no use for a held-out validation split.

    Parameters
    ----------
    data:
        Raw (un-normalised) array of shape *(T, N, C)*. Channel 0 is used.
    horizons:
        Forecast horizons in time-steps, e.g. [6, 12, 18, 24, 30, 36, 42].
    test_start:
        Raw-row index of the first test window's forecast origin (i.e. the
        input-window start plus ``in_len`` — see above).
    test_count:
        Number of test windows (``W``) — must match ``ground_truth.npy``.
    verbose:
        Print per-node progress every 10 nodes.
    n_jobs:
        Number of worker processes for the per-node fits, which are
        independent across nodes. Defaults to ``os.cpu_count()``. Pass ``1``
        to run sequentially in-process (e.g. under a debugger).
    """
    T, N, _ = data.shape
    channel = data[:, :, 0]  # (T, N)
    max_h = max(horizons)

    if test_start + test_count + max_h - 1 > T:
        raise ValueError(
            "test_start/test_count/max(horizons) exceed available data length "
            f"({test_start=}, {test_count=}, {max_h=}, T={T})."
        )

    train_ch = channel[:test_start]
    test_ch = channel[test_start : test_start + test_count + max_h - 1]

    mean = train_ch.mean(axis=0)
    std = np.where(train_ch.std(axis=0) == 0, 1.0, train_ch.std(axis=0))
    train_norm = (train_ch - mean) / std
    test_norm = (test_ch - mean) / std

    from tqdm import tqdm

    preds_out = np.zeros((test_count, len(horizons), N), dtype=np.float32)
    skipped = 0

    workers = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, N))

    bar = tqdm(
        total=N,
        desc="  [arima] fitting nodes",
        unit="node",
        disable=not verbose,
        dynamic_ncols=True,
    )

    if workers == 1:
        for n_idx in range(N):
            _, result = _fit_predict_one_node(
                n_idx, train_norm[:, n_idx], test_norm[:, n_idx], order, max_h
            )
            if result is None:
                skipped += 1
                bar.set_postfix(skipped=skipped)
            else:
                for h_idx, h in enumerate(horizons):
                    fc_real = result["forecasts_by_h"][h - 1] * std[n_idx] + mean[n_idx]
                    preds_out[:, h_idx, n_idx] = fc_real
            bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fit_predict_one_node,
                    n_idx,
                    train_norm[:, n_idx],
                    test_norm[:, n_idx],
                    order,
                    max_h,
                ): n_idx
                for n_idx in range(N)
            }
            for future in as_completed(futures):
                n_idx, result = future.result()
                if result is None:
                    skipped += 1
                    bar.set_postfix(skipped=skipped)
                else:
                    for h_idx, h in enumerate(horizons):
                        fc_real = result["forecasts_by_h"][h - 1] * std[n_idx] + mean[n_idx]
                        preds_out[:, h_idx, n_idx] = fc_real
                bar.update(1)

    bar.close()
    return preds_out


def run_arima(
    data: np.ndarray,
    order: tuple[int, int, int] = (3, 0, 1),
    horizon: int = 12,
    rolling_step: int = 10,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    use_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_project: str = "stgnn-framework",
    wandb_run_name: str = "arima",
    wandb_mode: str = "online",
    null_val: float = 0.0,
) -> dict:
    """Run rolling ARIMA on all nodes and return aggregated metrics.

    Parameters
    ----------
    data:
        Raw (un-normalised) array of shape *(T, N, C)*. Channel 0 is used.
    order:
        ARIMA (p, d, q) order.
    horizon:
        Forecast horizon (number of steps ahead).
    rolling_step:
        Stride between consecutive rolling-forecast origins.
    val_ratio / test_ratio:
        Walk-forward split fractions.  ARIMA trains on the train split and
        evaluates on the test split.
    use_wandb:
        Whether to log results to Weights & Biases.

    Returns
    -------
    dict with keys:
        ``mae_h``, ``mape_h``, ``rmse_h`` — per-horizon arrays (H,)
        ``mae``, ``mape``, ``rmse``       — global scalar averages
        ``skipped``                       — number of nodes that failed to fit
    """
    T, N, _ = data.shape
    channel = data[:, :, 0]  # (T, N) — use traffic feature channel only

    n_test = int(T * test_ratio)
    n_val = int(T * val_ratio)
    n_train = T - n_val - n_test

    train_ch = channel[:n_train]
    test_ch = channel[n_train + n_val :]

    # Per-node normalization (fit on train)
    mean = train_ch.mean(axis=0)  # (N,)
    std = train_ch.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    train_norm = (train_ch - mean) / std
    test_norm = (test_ch - mean) / std

    forecasts_by_h: list[list[float]] = [[] for _ in range(horizon)]
    targets_by_h: list[list[float]] = [[] for _ in range(horizon)]
    skipped = 0

    for n_idx in range(N):
        try:
            model = RollingARIMA(order=order)
            model.fit(train_norm[:, n_idx])
            result = model.rolling_forecast(
                test_norm[:, n_idx], horizon=horizon, rolling_step=rolling_step
            )
            for h in range(horizon):
                forecast_real = result["forecasts_by_h"][h] * std[n_idx] + mean[n_idx]
                target_real = result["targets_by_h"][h] * std[n_idx] + mean[n_idx]
                forecasts_by_h[h].extend(forecast_real.tolist())
                targets_by_h[h].extend(target_real.tolist())
        except Exception:
            skipped += 1

    # Forecasts and targets were denormalized per node before aggregation.
    # Compute metrics per horizon
    import torch

    mae_h = np.zeros(horizon)
    mape_h = np.zeros(horizon)
    rmse_h = np.zeros(horizon)

    for h in range(horizon):
        if len(forecasts_by_h[h]) == 0:
            continue
        pred_t = torch.tensor(forecasts_by_h[h], dtype=torch.float32)
        true_t = torch.tensor(targets_by_h[h], dtype=torch.float32)
        # Treat as (B,) tensors — reshape to (B, 1, 1) for metric functions
        pred_t = pred_t.unsqueeze(-1).unsqueeze(-1)
        true_t = true_t.unsqueeze(-1).unsqueeze(-1)
        mae_h[h] = _mae(pred_t, true_t, null_val).item()
        mape_h[h] = _mape(pred_t, true_t, null_val).item()
        rmse_h[h] = _rmse(pred_t, true_t, null_val).item()

    mae = float(mae_h.mean())
    mape = float(mape_h.mean())
    rmse = float(rmse_h.mean())

    if use_wandb:
        import wandb

        init_kwargs: dict[str, Any] = dict(
            project=wandb_project,
            name=wandb_run_name,
            mode=cast(Any, wandb_mode),
        )
        entity = resolve_wandb_entity(wandb_entity)
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs)
        for h in range(horizon):
            wandb.log(
                {
                    f"test/MAE_h{h + 1}": mae_h[h],
                    f"test/MAPE_h{h + 1}": mape_h[h],
                    f"test/RMSE_h{h + 1}": rmse_h[h],
                }
            )
        wandb.log({"test/MAE": mae, "test/MAPE": mape, "test/RMSE": rmse})
        wandb.finish()

    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "mae_h": mae_h,
        "mape_h": mape_h,
        "rmse_h": rmse_h,
        "skipped": skipped,
    }
