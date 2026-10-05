from __future__ import annotations

import numpy as np

from models.arima import RollingARIMA
from training.arima_trainer import run_arima


def test_rolling_arima_fit_and_forecast_outputs_are_finite() -> None:
    series = np.sin(np.linspace(0, 4 * np.pi, 48)) + np.linspace(0, 1, 48)
    model = RollingARIMA(order=(1, 0, 0)).fit(series[:32])

    result = model.rolling_forecast(series[32:], horizon=3, rolling_step=4)

    assert len(result["forecasts_by_h"]) == 3
    assert result["forecasts_by_h"][0].shape == result["targets_by_h"][0].shape
    assert result["forecasts_by_h"][0].ndim == 1
    for forecasts in result["forecasts_by_h"]:
        assert np.isfinite(forecasts).all()


def test_arima_trainer_returns_finite_metrics_without_wandb() -> None:
    t = np.linspace(0, 4 * np.pi, 80)
    node_a = np.sin(t) + np.linspace(0, 1, 80)
    node_b = np.cos(t) + np.linspace(1, 2, 80)
    data = np.stack([node_a, node_b], axis=1).astype(np.float32)
    data = np.stack([data, data], axis=-1)

    result = run_arima(data, order=(1, 0, 0), horizon=3, rolling_step=5, use_wandb=False)

    assert result["mae_h"].shape == (3,)
    assert result["skipped"] == 0
    assert np.isfinite(result["mae"])
    assert np.isfinite(result["mape"])
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["mae_h"]).all()
