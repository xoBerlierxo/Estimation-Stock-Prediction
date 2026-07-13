from __future__ import annotations

import numpy as np
import pytest

from src.data.validate import (
    DataValidationError,
    check_missing_values,
    validate_panel_completeness,
    validate_schema,
)


def test_validate_schema_passes_on_valid_df(sample_raw_df):
    validate_schema(sample_raw_df)  # should not raise


def test_validate_schema_raises_on_missing_column(sample_raw_df):
    df = sample_raw_df.drop(columns=["total_price"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_raises_on_wrong_dtype(sample_raw_df):
    df = sample_raw_df.copy()
    df["store_id"] = df["store_id"].astype(str)
    with pytest.raises(DataValidationError, match="dtype kind"):
        validate_schema(df)


def test_panel_completeness_detects_no_gap_in_well_formed_panel(sample_raw_df):
    violations = validate_panel_completeness(sample_raw_df, ["store_id", "sku_id"])
    assert violations.empty


def test_panel_completeness_detects_gap(sample_raw_df):
    df = sample_raw_df.drop(sample_raw_df.index[5])  # remove one week from group 1
    violations = validate_panel_completeness(df, ["store_id", "sku_id"])
    assert not violations.empty
    assert violations.iloc[0]["store_id"] == 101


def test_missing_value_ratio_within_threshold(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "total_price"] = np.nan
    check_missing_values(df, max_ratio=0.05)  # 1/40 rows == 2.5%, within a 5% threshold


def test_missing_value_ratio_exceeds_threshold_raises(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[: len(df) // 2, "total_price"] = np.nan
    with pytest.raises(DataValidationError, match="exceeding"):
        check_missing_values(df, max_ratio=0.01)


def test_unexpected_missing_column_raises(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "units_sold"] = np.nan
    with pytest.raises(DataValidationError, match="Unexpected missing values"):
        check_missing_values(df)
