"""Feature engineering for per-(store, SKU) weekly demand series.

This module replaces the original notebook's broken lag-feature step, which
built a `week_store` key (dropping `sku_id`), grouped with `.sum()` (which
re-sorts rows alphabetically by that string key), and only then took
`.shift()` -- producing lags of a scrambled, cross-SKU-aggregated series.

Here, every lag/rolling feature is computed on a frame explicitly sorted by
`group_cols + [date_col]`, `groupby(group_cols)` per real (store_id, sku_id)
series, so each lag is a genuine prior observation of that exact series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import FeatureConfig

NON_FEATURE_COLUMNS = {"record_ID", "units_sold", "units_sold_log"}


def add_calendar_features(df: pd.DataFrame, date_col: str = "week") -> pd.DataFrame:
    """Add calendar features derived purely from the date itself (no leakage)."""
    df = df.copy()
    dt = df[date_col].dt
    df["month"] = dt.month.astype("int16")
    df["weekofyear"] = dt.isocalendar().week.astype("int16")
    df["quarter"] = dt.quarter.astype("int16")
    df["year"] = dt.year.astype("int16")

    unique_weeks = np.sort(df[date_col].unique())
    week_rank = {week: idx for idx, week in enumerate(unique_weeks)}
    df["weeks_since_start"] = df[date_col].map(week_rank).astype("int16")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add discount / price-gap features, safe against a zero base_price."""
    df = df.copy()
    safe_base = df["base_price"].replace(0, np.nan)
    df["discount"] = ((df["base_price"] - df["total_price"]) / safe_base).fillna(0.0)
    df["price_diff"] = df["base_price"] - df["total_price"]
    return df


def build_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add `{target_col}_lag_{n}` columns for each n in `lags`.

    Requires `df` to already be sorted by `group_cols + [date_col]`.
    """
    df = df.copy()
    grouped_target = df.groupby(group_cols)[target_col]
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = grouped_target.shift(lag)
    return df


def build_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    windows: list[int],
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling mean/std of `target_col` per group, computed on the
    lag-1 series so a row's own value never leaks into its own rolling stat.

    Requires `df` to already be sorted by `group_cols + [date_col]`.
    """
    stats = stats or ["mean", "std"]
    df = df.copy()
    shifted = df.groupby(group_cols)[target_col].shift(1)
    grouped_shifted = shifted.groupby([df[c] for c in group_cols])

    n_group_levels = len(group_cols)
    for window in windows:
        rolled = grouped_shifted.rolling(window=window, min_periods=window)
        if "mean" in stats:
            result = rolled.mean()
            df[f"{target_col}_roll_mean_{window}"] = result.reset_index(
                level=list(range(n_group_levels)), drop=True
            )
        if "std" in stats:
            result = rolled.std()
            df[f"{target_col}_roll_std_{window}"] = result.reset_index(
                level=list(range(n_group_levels)), drop=True
            )
    return df


def encode_categoricals(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("store_id", "sku_id"),
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode `cols`, replacing the raw columns with dummy columns.

    If `categories` is given (the full set of category values seen during
    training), each column is cast to a `Categorical` with that fixed
    category set first, so a caller with only a single category present
    (e.g. one store/SKU at inference time) still produces every dummy
    column the trained model expects (as all-zero, aside from its own).
    """
    df = df.copy()
    for col in cols:
        if categories is not None and col in categories:
            df[col] = pd.Categorical(df[col], categories=categories[col])
    dummies = pd.get_dummies(df[list(cols)], columns=list(cols), prefix=list(cols))
    df = pd.concat([df.drop(columns=list(cols)), dummies], axis=1)
    return df, list(dummies.columns)


def build_feature_matrix(
    df: pd.DataFrame,
    config: FeatureConfig,
    group_cols: list[str],
    date_col: str,
    target_col: str,
    category_values: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Orchestrate the full feature pipeline and return (feature_df, feature_columns).

    `feature_columns` is the exact, ordered list of model input columns --
    this list is persisted in `models/metadata.json` and reused verbatim by
    `app/inference.py` so training and serving never diverge.
    """
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    df = add_calendar_features(df, date_col=date_col)
    df = add_price_features(df)
    df = build_lag_features(df, group_cols, target_col, config.lags)
    df = build_rolling_features(df, group_cols, target_col, config.rolling_windows, config.rolling_stats)

    lag_roll_cols = [c for c in df.columns if f"{target_col}_lag_" in c or f"{target_col}_roll_" in c]
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)

    if config.log_transform_target and target_col in df.columns:
        df[f"{target_col}_log"] = np.log1p(df[target_col])

    df, dummy_cols = encode_categoricals(df, cols=tuple(group_cols), categories=category_values)

    feature_columns = [
        c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != date_col and c not in group_cols
    ]
    # feature_columns already excludes group_cols because encode_categoricals
    # replaced them with dummy_cols, which are included via the list comprehension above.
    return df, feature_columns
