"""Loading and persisting the SKU-level demand dataset.

`week` is stored as `dd/mm/yy` strings (e.g. `"17/01/11"` -> 2011-01-17).
Parsing with the wrong day/month order is a real footgun on this dataset,
so `dayfirst=True` is explicit rather than relied upon implicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DTYPES = {
    "record_ID": "int64",
    "store_id": "int64",
    "sku_id": "int64",
    "total_price": "float64",
    "base_price": "float64",
    "is_featured_sku": "int8",
    "is_display_sku": "int8",
}


def _read_raw_csv(path: str | Path, date_col: str = "week") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%y", dayfirst=True)
    for col, dtype in RAW_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def load_raw_train(path: str | Path, date_col: str = "week") -> pd.DataFrame:
    """Load the labeled training panel (includes `units_sold`).

    `total_price` has a small number of missing values in this dataset (1 in
    150,150 rows as of the committed snapshot). They are imputed with
    `base_price` -- i.e. assuming no promotional discount was recorded for
    that row -- rather than left as NaN, since several downstream models
    (linear regression, random forest) cannot fit on missing values.
    """
    df = _read_raw_csv(path, date_col=date_col)
    df["units_sold"] = df["units_sold"].astype("float64")
    df["total_price"] = df["total_price"].fillna(df["base_price"])
    return df


def load_raw_test(path: str | Path, date_col: str = "week") -> pd.DataFrame:
    """Load the unlabeled Kaggle holdout (no `units_sold`, no ground truth).

    Kept for provenance only -- it is never used to evaluate this project's
    model since there is no way to score predictions against it.
    """
    return _read_raw_csv(path, date_col=date_col)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_processed(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
