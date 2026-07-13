# Project Report: Rebuilding SKU Demand Forecasting for Production Quality

## Why this rebuild happened

This repository started as a SmartBridge internship deliverable: a
proof-of-concept notebook, a pickled Random Forest, and a bare Flask form.
It demonstrated the idea but wouldn't survive review from an ML engineer or
hiring manager -- most importantly, its core feature-engineering step had a
real correctness bug (below), and there was no test suite, no reproducible
metrics, no input validation, and no packaging discipline. This report
documents what changed, why, and what the rebuilt pipeline actually measures
-- every number here is reproducible by running `python -m src.training.train
--config configs/train.yaml` against the committed `data/raw/train.csv`.

## The bug that was fixed

The original `Training/sales demand.ipynb` (preserved at
`docs/legacy/original_notebook.ipynb`) built its lag features like this:

```python
prep.dataset['key'] = prep.df['week'].astype(str) + '_' + prep.df['store_id'].astype(str)
prep.dataset = prep.df.drop(['record_ID', 'week', 'store_id', 'sku_id', ...], axis=1)
prep.dataset = prep.df.groupby('key').sum()
...
prep.df['day_1'] = prep.df['units_sold'].shift(-1)
```

Two compounding problems:

1. **`sku_id` was dropped from the grouping key.** Despite the project
   being about *SKU-level* forecasting, `key = week_store_id` sums
   `units_sold` across every SKU in a store for that week -- the model
   never actually saw per-SKU history.
2. **`groupby('key').sum()` re-sorts rows alphabetically by that string
   key**, not chronologically. The subsequent `.shift()` calls therefore
   produced lag features of a scrambled, cross-SKU-aggregated series --
   not real lags of anything coherent.

The rebuilt `src/features/build_features.py` sorts explicitly by
`[store_id, sku_id, week]` and computes every lag/rolling feature with
`groupby(['store_id', 'sku_id'])`, so each lag is a genuine prior
observation of that exact series. `tests/test_features.py::test_no_leakage_after_shuffled_input`
feeds the pipeline randomly-shuffled input rows and asserts the output is
identical to pre-sorted input -- directly reproducing and guarding against
the original failure mode.

A second, smaller bug was found during the rebuild: one row in the real
dataset has a missing `total_price`, which silently propagated `NaN` into
the `discount`/`price_diff` features and crashed `LinearRegression` and
`RandomForestRegressor` (neither accepts `NaN` inputs). Fixed by imputing
`total_price` with `base_price` for that row in `src/data/load.py`, with a
regression test (`test_build_feature_matrix_no_nans_in_feature_columns`)
guarding it.

## Engineering improvements

- **Repository structure**: reorganized into `src/{data,features,models,training,evaluation,visualization,utils}`,
  `app/`, `tests/`, `configs/`, `reports/`, `models/`, `docs/legacy/` --
  each module has one responsibility, versus one 812-line notebook holding
  every step.
- **Reproducibility**: every pipeline parameter lives in `configs/train.yaml`;
  every trained artifact's `models/metadata.json` snapshots the resolved
  config, git commit, training date, and library versions.
- **Serving**: replaced a raw-4-unlabeled-floats Flask form with a FastAPI
  service (`app/`) that takes realistic inputs (store, SKU, week, planned
  price/promotion) and reuses the exact training-time feature functions --
  eliminating train/serve skew by construction rather than by discipline.
- **Testing**: 41 tests (unit, integration, data-validation, model-persistence,
  API) -- see `TESTS.md`. Zero tests existed before.
- **Tooling**: `pyproject.toml`, `black`/`isort`/`flake8`/`mypy`, pre-commit
  hooks, a GitHub Actions CI workflow, and a Dockerfile -- verified locally
  (`docker build` + `docker run` + live `curl` checks against the container).
- **Cruft removal**: deleted `.DS_Store` files, empty placeholder scripts,
  a duplicated `requirements.txt`, and the 36MB pickled model that had no
  accompanying metadata; archived (not deleted) the original notebook/PDF
  under `docs/legacy/` for the historical record.

## ML improvements

- **Correct, leakage-safe feature engineering** (the core fix, above):
  per-series lags (1/2/3/4/8/12 weeks), rolling mean/std (4/8-week windows,
  computed on the lag-1 series so a row's own value never leaks into its
  own rolling stat), price features (`discount`, `price_diff`), calendar
  features, and one-hot encoded store/SKU identity.
- **Missing-value handling**: `total_price` imputed with `base_price`
  rather than left as `NaN` (see above).
- **Target transform**: trained on `log1p(units_sold)`, inverse-transformed
  with `expm1` before scoring -- justified by the heavy right skew in
  `units_sold` (mean 51.7, max 2876). Deliberately did *not* clip or remove
  high-`units_sold` rows: those are genuine promotional spikes correlated
  with `is_featured_sku`/`is_display_sku`, and are exactly what the model
  should learn, not noise to discard.
- **Time-series-aware validation**: an 8-week holdout test set, touched
  exactly once, plus a custom rolling-origin CV (`src/training/split.py`)
  that runs `TimeSeriesSplit` over the *unique week index* rather than row
  order -- guaranteeing every fold boundary falls on a whole week across
  all 1,155 series, which naively applying `TimeSeriesSplit` to row order
  would not guarantee.
- **Model comparison**: Linear Regression (baseline), Random Forest,
  XGBoost, and LightGBM, each tuned with `RandomizedSearchCV` over the
  leakage-safe CV folds, selected by test RMSE.
- **Evaluation depth**: RMSE/MAE/R² plus WAPE as the primary
  percentage-style metric (MAPE is reported but flagged as unstable here,
  since `units_sold` has a minimum of 1 and a few units of error on a
  low-volume row reads as a triple-digit percentage). Feature importance,
  residual, actual-vs-predicted, and model-comparison plots are generated
  by `src/visualization/plots.py`, not hand-built in a notebook.
- **A genuine data quality finding**: `src/data/validate.py::validate_panel_completeness`,
  written as a general-purpose guard, surfaced a real 8-day gap in the
  source calendar (2012-02-27 -> 2012-03-06) that the original notebook
  never checked for.

## Resulting performance

Held-out test set: the last 8 weeks of the panel (2013-05-21 through
2013-07-09), never touched during model selection or hyperparameter
search.

| model | RMSE | MAE | R² | MAPE | WAPE |
|---|---|---|---|---|---|
| **xgboost (selected)** | **25.44** | **12.90** | **0.801** | 36.1% | **24.9%** |
| lightgbm | 25.81 | 13.03 | 0.795 | 36.1% | 25.2% |
| random_forest | 26.35 | 13.18 | 0.786 | 36.0% | 25.4% |
| linear_regression (baseline) | 40.75 | 17.21 | 0.489 | 44.5% | 33.2% |

XGBoost was selected (RandomizedSearchCV, 15 iterations, 5-fold
rolling-origin CV; best params: `n_estimators=500, max_depth=7,
learning_rate=0.05, subsample=0.85, colsample_bytree=0.7`). Relative to the
linear baseline: **37.6% lower RMSE**, **25.0% lower MAE**, and R² roughly
doubled (0.489 -> 0.801). Full comparison table and figures:
[`reports/metrics.md`](reports/metrics.md), [`reports/figures/`](reports/figures/).

Two engineering tradeoffs affected these exact numbers and are disclosed
here rather than left implicit: the Random Forest hyperparameter grid was
capped (`max_depth<=20`, `n_estimators<=250`) after an uncapped grid proved
too slow to search exhaustively in this environment, and `n_iter_search`
was set to 15 (not a larger number) for the same reason. Both are
documented in `configs/train.yaml` and `src/models/registry.py`; neither
affected the *winner* (XGBoost, which trains fast enough that its grid was
unconstrained).

## Suggested resume bullet points

- Rebuilt a retail demand-forecasting pipeline from a proof-of-concept
  notebook into a tested, containerized ML service; **fixed a
  feature-engineering bug that had silently discarded per-SKU granularity
  and computed lag features on a shuffled series**, replacing it with a
  leakage-safe, per-series time-series feature pipeline covered by
  regression tests.
- Compared Linear Regression, Random Forest, XGBoost, and LightGBM with
  `RandomizedSearchCV` over a custom rolling-origin (expanding-window)
  cross-validation scheme; **reduced test-set RMSE by 37.6% and
  approximately doubled R² (0.49 -> 0.80)** versus the linear baseline.
- Designed and shipped a FastAPI inference service that reconstructs
  lag/rolling features from a bundled reference history at request time,
  sharing one feature-engineering code path with training to eliminate
  train/serve skew; verified end-to-end locally and in a built Docker
  image.
- Authored a 41-test suite (unit, integration, data-validation,
  model-persistence, API) plus a GitHub Actions CI pipeline (lint,
  type-check, test) and pre-commit hooks, taking the project from zero
  tests to CI-enforced quality gates.
