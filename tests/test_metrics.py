from __future__ import annotations

import numpy as np

from src.evaluation.metrics import evaluate_predictions, mape, wape


def test_regression_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0, 5.0])
    metrics = evaluate_predictions(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape"] == 0.0
    assert metrics["wape"] == 0.0


def test_wape_matches_manual_calc():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # sum(|error|) = 2+2+3 = 7, sum(|actual|) = 60 -> 11.666...%
    np.testing.assert_allclose(wape(y_true, y_pred), 7 / 60 * 100)


def test_mape_no_zero_division_given_min_units_sold_one():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 3.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0
