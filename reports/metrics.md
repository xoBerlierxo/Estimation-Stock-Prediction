# Model Evaluation Report

Generated automatically by `src/training/train.py`. Every number below comes from an actual run of the pipeline against `data/raw/train.csv`; none of it is hand-entered.

## Model comparison (held-out test set, last 8 weeks)

| model             |   cv_score_log_rmse |    rmse |     mae |     r2 |    mape |    wape |
|:------------------|--------------------:|--------:|--------:|-------:|--------:|--------:|
| xgboost           |              0.4251 | 25.4422 | 12.9024 | 0.8006 | 36.0514 | 24.9073 |
| lightgbm          |              0.4224 | 25.8082 | 13.0287 | 0.7948 | 36.0941 | 25.1511 |
| random_forest     |              0.4325 | 26.3473 | 13.1803 | 0.7862 | 36.0321 | 25.4438 |
| linear_regression |              0.4723 | 40.7506 | 17.2123 | 0.4885 | 44.497  | 33.2273 |

## Selected model: `xgboost`

`xgboost` was selected by lowest RMSE on a held-out test set of the last 8 weeks, never touched during hyperparameter search. Hyperparameters were tuned with `RandomizedSearchCV` (15 iterations) using a 5-fold expanding-window (rolling-origin) time-series CV scheme on the log1p-transformed target, so no future week ever leaks into a training fold.

## Metric notes

- **RMSE / MAE / R²**: standard regression metrics on the original `units_sold` scale (predictions from the log1p-trained model are inverse-transformed with `expm1` before scoring).
- **WAPE** (weighted absolute percentage error) is the primary percentage-style metric for this dataset.
- **MAPE** is reported for completeness but is unstable here: `units_sold` has a minimum of 1, so small absolute errors on low-volume rows produce triple-digit percentage errors.
