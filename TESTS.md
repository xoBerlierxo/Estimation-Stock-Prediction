# Test Suite

41 tests across 8 files, run with `pytest`. Every number on this page is a
copy of an actual local run (`pytest --cov=src --cov=app --cov-report=term-missing`),
not a hypothetical description.

## What each file covers

| File | Type | Covers |
|---|---|---|
| `tests/test_data_load.py` | Unit | CSV parsing, and specifically that `week` strings like `"17/01/11"` parse as `2011-01-17` (day-first), not `2011-11-01`. Also covers the `total_price` -> `base_price` imputation for missing values. |
| `tests/test_data_validate.py` | Unit | Schema/dtype checks, panel-completeness gap detection, missing-value ratio guard. |
| `tests/test_features.py` | Unit | **Directly guards the bug this rebuild fixes**: lag correctness per (store, SKU) group, no cross-group bleed, correctness preserved even when input rows arrive in random order (`test_no_leakage_after_shuffled_input`), rolling stats exclude the current row, safe `discount` at `base_price=0`, correct row-dropping for incomplete lag windows, and no NaNs reach the model input (`test_build_feature_matrix_no_nans_in_feature_columns`, which reproduces the real missing-`total_price` row found in the dataset). |
| `tests/test_split.py` | Unit | Time-based holdout boundary correctness; rolling-origin CV folds never leak a future week into a training fold; fold sizes expand monotonically. |
| `tests/test_metrics.py` | Unit | Perfect-prediction sanity checks, a hand-computed WAPE example, no zero-division in MAPE. |
| `tests/test_model_persistence.py` | Unit | Save/load round-trip for a fitted model, required metadata keys present, missing-file error paths. |
| `tests/test_api.py` | Integration | FastAPI `TestClient` against a small model trained inline on synthetic data (never the full trained artifact): health check, a valid end-to-end prediction, unknown-combo 404, missing-field 422, insufficient-history 422, wrong-forecast-week 422. |
| `tests/test_dataset_invariants.py` | Data validation | Reads the real, committed `data/raw/train.csv` and locks in the panel structure the whole pipeline design depends on: 76 stores, 28 SKUs, 1155 combos, 130 weeks each, exactly one missing value pre-imputation. |

Model loading and prediction are covered together in `test_api.py`
(`InferenceService` internally calls `src.models.persistence.load_model` /
`load_metadata`) and directly in `test_model_persistence.py`.

## Latest run

```
$ pytest --cov=src --cov=app --cov-report=term-missing -q

.........................................                               [100%]

Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
app\__init__.py                      0      0   100%
app\inference.py                    57      2    96%   125, 133
app\main.py                         41      3    93%   41, 75-76
app\schemas.py                      26      0   100%
src\__init__.py                      0      0   100%
src\data\__init__.py                 0      0   100%
src\data\load.py                    24      5    79%   55, 59-61, 65
src\data\validate.py                32      1    97%   78
src\evaluation\__init__.py           0      0   100%
src\evaluation\metrics.py           20      0   100%
src\features\__init__.py             0      0   100%
src\features\build_features.py      64      0   100%
src\models\__init__.py               0      0   100%
src\models\persistence.py           48      4    92%   35-36, 44-45
src\models\registry.py              13     13     0%   5-54
src\training\__init__.py             0      0   100%
src\training\split.py               27      2    93%   27, 53
src\training\train.py              103    103     0%   11-223
src\utils\__init__.py                0      0   100%
src\utils\config.py                 38      3    92%   62-64
src\utils\logging.py                14     14     0%   3-23
src\visualization\__init__.py        0      0   100%
src\visualization\plots.py          70     70     0%   6-138
--------------------------------------------------------------
TOTAL                              577    220    62%
41 passed, 17 warnings in 5.68s
```

`src/training/train.py`, `src/models/registry.py`, and `src/visualization/plots.py`
show 0% unit-test coverage: they're the orchestrator, model registry, and plotting
code exercised by actually *running* the pipeline (`python -m src.training.train`)
rather than by isolated unit tests, since testing them meaningfully would mean
re-running real training. `reports/metrics.md` and `reports/figures/` are the
evidence that end-to-end run succeeded. The 17 warnings are pre-existing
deprecation notices from `starlette`'s test client and `joblib`/`numpy`
internals -- not from this project's code.

## Running the suite

```bash
pip install -e ".[dev]"
pytest                                              # full suite
pytest --cov=src --cov=app --cov-report=term-missing  # with coverage
pytest tests/test_features.py -v                    # a single file, verbose
```
