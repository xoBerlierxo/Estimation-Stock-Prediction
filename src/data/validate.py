"""Data validation guards.

These exist to catch silent data corruption (schema drift, missing weeks,
unexpected NaNs) early, rather than letting a broken frame flow quietly
into feature engineering and training.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: dict[str, str] = {
    "record_ID": "i",
    "week": "M",
    "store_id": "i",
    "sku_id": "i",
    "total_price": "f",
    "base_price": "f",
    "is_featured_sku": "i",
    "is_display_sku": "i",
}


class DataValidationError(Exception):
    """Raised when a dataframe fails a required structural or content check."""


def validate_schema(df: pd.DataFrame) -> None:
    """Check that required columns exist with the expected dtype kind."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    for col, expected_kind in REQUIRED_COLUMNS.items():
        actual_kind = df[col].dtype.kind
        if actual_kind != expected_kind:
            raise DataValidationError(
                f"Column '{col}' has dtype kind '{actual_kind}', expected '{expected_kind}'"
            )


def validate_panel_completeness(
    df: pd.DataFrame,
    group_cols: list[str],
    date_col: str = "week",
) -> pd.DataFrame:
    """Return a dataframe of groups whose weekly cadence has gaps.

    A well-formed panel has exactly 7 days between consecutive observations
    within every group. Any other gap indicates a missing week. Empty
    result means the panel is complete (true for this dataset today).
    """
    violations = []
    for keys, group in df.sort_values(date_col).groupby(group_cols):
        diffs = group[date_col].diff().dropna()
        bad = diffs[diffs != pd.Timedelta(days=7)]
        if not bad.empty:
            key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            violations.append({**key_dict, "n_gaps": len(bad)})
    return pd.DataFrame(violations)


def check_missing_values(
    df: pd.DataFrame,
    allowed_cols: tuple[str, ...] = ("total_price",),
    max_ratio: float = 0.01,
) -> None:
    """Raise if any column outside `allowed_cols` has missing values, or if
    an allowed column's missing ratio exceeds `max_ratio`.
    """
    na_counts = df.isna().sum()
    unexpected = na_counts[(na_counts > 0) & ~na_counts.index.isin(allowed_cols)]
    if not unexpected.empty:
        raise DataValidationError(f"Unexpected missing values in columns: {unexpected.to_dict()}")

    for col in allowed_cols:
        if col not in df.columns:
            continue
        ratio = df[col].isna().mean()
        if ratio > max_ratio:
            raise DataValidationError(
                f"Column '{col}' has {ratio:.2%} missing values, exceeding the {max_ratio:.2%} threshold"
            )
