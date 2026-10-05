from __future__ import annotations

from typing import Any

import numpy as np


class RollingARIMA:
    """Rolling-window ARIMA forecaster for a single time series.

    Fits once on training data to obtain parameter estimates, then uses
    statsmodels' `filter()` to advance through test-set history at each
    rolling step — avoiding full re-fitting at every window.

    Usage::

        model = RollingARIMA(order=(3, 0, 1))
        model.fit(train_series)
        result = model.rolling_forecast(test_series, horizon=12, rolling_step=10)
        # result["forecasts_by_h"][h_idx] -> np.ndarray of forecasts at step h
        # result["targets_by_h"][h_idx]   -> np.ndarray of ground-truth targets
    """

    def __init__(self, order: tuple[int, int, int] = (3, 0, 1)):
        self.order = order
        self._result: Any | None = None
        self._train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: np.ndarray) -> RollingARIMA:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # lazy import

        train = np.asarray(train, dtype=float)
        p, d, q = self.order
        if len(train) <= max(p, q) + d + 1:
            raise ValueError("Not enough training observations for the configured ARIMA order")

        self._result = _ARIMA(train, order=self.order).fit()
        self._train = train
        return self

    # ------------------------------------------------------------------
    # Rolling forecast
    # ------------------------------------------------------------------

    def rolling_forecast(
        self,
        test: np.ndarray,
        horizon: int,
        rolling_step: int = 10,
    ) -> dict[str, list[np.ndarray]]:
        """Perform rolling multi-step-ahead forecast over a test set.

        Parameters
        ----------
        test:
            Test-set observations (1-D array).
        horizon:
            Number of steps ahead to forecast at each window position.
        rolling_step:
            Stride between consecutive forecast origins.

        Returns
        -------
        dict with keys:
            ``forecasts_by_h`` — list of length ``horizon``, each element is a
            1-D array of point forecasts at that step ahead.
            ``targets_by_h``   — matching ground-truth arrays.

        Notes
        -----
        Advances the Kalman filter state via ``MLEResults.extend()`` rather
        than reconstructing an ``ARIMA`` model over the full train+test
        history at every origin. ``extend()`` filters only the newly
        revealed chunk of ``test`` (using the final filtered state from the
        previous origin as its initialization), so per-origin cost is
        O(rolling_step) instead of O(len(history)) — the previous
        from-scratch reconstruction made the whole rolling pass quadratic in
        the number of origins.
        """
        if self._result is None or self._train is None:
            raise RuntimeError("Call fit() before rolling_forecast()")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if rolling_step <= 0:
            raise ValueError("rolling_step must be > 0")

        test = np.asarray(test, dtype=float)
        if len(test) < horizon:
            raise ValueError("test series is shorter than the forecast horizon")

        forecasts_by_h: list[list[float]] = [[] for _ in range(horizon)]
        targets_by_h: list[list[float]] = [[] for _ in range(horizon)]

        result = self._result
        revealed = 0  # number of leading `test` points already folded into `result`

        for t in range(0, len(test) - horizon + 1, rolling_step):
            if t > revealed:
                result = result.extend(test[revealed:t])
                revealed = t
            forecast_t = result.forecast(steps=horizon)
            for h_idx in range(horizon):
                forecasts_by_h[h_idx].append(float(forecast_t[h_idx]))
                targets_by_h[h_idx].append(float(test[t + h_idx]))

        return {
            "forecasts_by_h": [np.array(f) for f in forecasts_by_h],
            "targets_by_h": [np.array(t) for t in targets_by_h],
        }
