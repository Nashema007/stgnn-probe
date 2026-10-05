from __future__ import annotations

import json
import math

import numpy as np

from analysis.io import save_json


def test_save_json_replaces_non_finite_values_with_null(tmp_path) -> None:
    path = tmp_path / "metrics.json"

    save_json(
        {
            "nan_value": float("nan"),
            "inf_value": np.float64(math.inf),
            "nested": {
                "neg_inf": -math.inf,
                "array": np.array([1.0, np.nan, np.inf]),
                "int_value": np.int64(4),
            },
        },
        path,
    )

    text = path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text

    parsed = json.loads(text)
    assert parsed == {
        "nan_value": None,
        "inf_value": None,
        "nested": {
            "neg_inf": None,
            "array": [1.0, None, None],
            "int_value": 4,
        },
    }
