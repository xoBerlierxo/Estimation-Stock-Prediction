# Legacy artifacts

This folder preserves the original SmartBridge internship deliverable
(`original_notebook.ipynb`, `original_documentation.pdf`) exactly as it was
before the rebuild documented in the repository root's `PROJECT_REPORT.md`.
It's kept for the record, not used by the current pipeline.

## The bug that motivated the rebuild

`original_notebook.ipynb` (cells 6-9) builds its "lag" features like this:

```python
prep.dataset['key'] = prep.df['week'].astype(str) + '_' + prep.df['store_id'].astype(str)
prep.dataset = prep.df.drop(['record_ID', 'week', 'store_id', 'sku_id', ...], axis=1)
prep.dataset = prep.df.groupby('key').sum()
...
prep.df['day_1'] = prep.df['units_sold'].shift(-1)
```

Two problems compound here:

1. **`sku_id` is dropped from the key.** Despite the project being about
   *SKU-level* demand forecasting, `key = week_store_id` groups and sums
   `units_sold` across every SKU in a store for that week. The model never
   actually sees per-SKU history.
2. **`groupby('key').sum()` re-sorts rows alphabetically by the string key**
   (e.g. `"17/01/11_8091"` sorts before `"24/01/11_8091"` only by luck of
   string comparison, and breaks entirely once week/store combinations
   don't sort the same way lexicographically as chronologically). The
   subsequent `.shift(-1)` calls are therefore lag features of a
   scrambled, cross-SKU-aggregated series -- not real lags of anything
   coherent.

The rebuilt pipeline (`src/features/build_features.py`) fixes this by
explicitly sorting on `[store_id, sku_id, week]` and computing every
lag/rolling feature with `groupby(['store_id', 'sku_id'])`, so each lag is a
genuine prior observation of that exact (store, SKU) series. This is
covered directly by `tests/test_features.py::test_no_leakage_after_shuffled_input`,
which feeds the feature pipeline randomly-shuffled rows and asserts the
output is identical to feeding it pre-sorted rows.
