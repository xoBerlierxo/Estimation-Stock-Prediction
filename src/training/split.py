"""Time-based splitting for a balanced weekly panel.

Every split boundary is computed from the actual unique weeks present in the
data (never a hardcoded date), and always falls on a whole-week boundary so
no fold ever contains a partial week split across train/validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def get_unique_sorted_weeks(df: pd.DataFrame, date_col: str = "week") -> np.ndarray:
    return np.sort(df[date_col].unique())


def time_based_holdout_split(
    df: pd.DataFrame,
    date_col: str = "week",
    n_test_weeks: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train_val, test) using the last `n_test_weeks` weeks as test."""
    weeks = get_unique_sorted_weeks(df, date_col)
    if n_test_weeks >= len(weeks):
        raise ValueError(f"n_test_weeks={n_test_weeks} >= number of unique weeks ({len(weeks)})")
    cutoff = weeks[-n_test_weeks]
    train_val = df[df[date_col] < cutoff].reset_index(drop=True)
    test = df[df[date_col] >= cutoff].reset_index(drop=True)
    return train_val, test


def rolling_origin_splits(
    df: pd.DataFrame,
    date_col: str = "week",
    n_splits: int = 5,
    val_weeks: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds that never split a week across train/val.

    Runs `TimeSeriesSplit` over the *unique week index*, not row order, then
    maps each fold back to row positions. This guarantees every fold's
    boundary falls on a whole week (all series together) -- applying
    `TimeSeriesSplit` directly to row order would slice a single week's
    ~1155 rows across a fold boundary.

    Returns a list of (train_row_positions, val_row_positions) arrays,
    suitable for passing to `sklearn`'s `cv=` argument via index position.
    """
    weeks = get_unique_sorted_weeks(df, date_col)
    if val_weeks * n_splits >= len(weeks):
        raise ValueError(
            f"val_weeks*n_splits={val_weeks * n_splits} >= number of unique weeks ({len(weeks)})"
        )

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=val_weeks)
    week_positions = np.arange(len(weeks))

    week_to_row_positions: dict[np.datetime64, np.ndarray] = {
        week: np.flatnonzero((df[date_col] == week).to_numpy()) for week in weeks
    }

    folds = []
    for train_week_idx, val_week_idx in tscv.split(week_positions):
        train_rows = np.concatenate([week_to_row_positions[weeks[i]] for i in train_week_idx])
        val_rows = np.concatenate([week_to_row_positions[weeks[i]] for i in val_week_idx])
        folds.append((np.sort(train_rows), np.sort(val_rows)))
    return folds
