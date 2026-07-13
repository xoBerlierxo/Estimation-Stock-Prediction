from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import (
    add_price_features,
    build_feature_matrix,
    build_lag_features,
    build_rolling_features,
    encode_categoricals,
)
from src.utils.config import FeatureConfig

GROUP_COLS = ["store_id", "sku_id"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(GROUP_COLS + ["week"]).reset_index(drop=True)


def test_build_lag_features_respects_group_boundaries(sample_raw_df):
    df = build_lag_features(_sorted(sample_raw_df), GROUP_COLS, "units_sold", [1])
    first_rows = df.groupby(GROUP_COLS).head(1)
    assert first_rows["units_sold_lag_1"].isna().all()


def test_build_lag_features_values_correct(sample_raw_df):
    df = build_lag_features(_sorted(sample_raw_df), GROUP_COLS, "units_sold", [1, 2])
    group = df[(df.store_id == 101) & (df.sku_id == 1)].reset_index(drop=True)
    # base_units=10, units_sold at week i is 10+i
    assert group.loc[5, "units_sold_lag_1"] == group.loc[4, "units_sold"]
    assert group.loc[5, "units_sold_lag_2"] == group.loc[3, "units_sold"]


def test_no_leakage_after_shuffled_input(shuffled_sample_raw_df):
    """Reproduces the original bug directly: `build_feature_matrix` must sort
    internally so that even randomly-ordered input rows (as the original
    groupby(key).sum() implicitly re-sorted them alphabetically) produce lag
    features that reflect true chronological order per group.
    """
    config = FeatureConfig(lags=[1], rolling_windows=[2], rolling_stats=["mean"])
    feat_df, _ = build_feature_matrix(shuffled_sample_raw_df, config, GROUP_COLS, "week", "units_sold")
    group = feat_df[(feat_df.store_id_102) & (feat_df.sku_id_2)].sort_values("week").reset_index(drop=True)
    # base_units=50, units_sold at week i is 50+i
    for i in range(1, len(group)):
        assert group.loc[i, "units_sold_lag_1"] == group.loc[i - 1, "units_sold"]


def test_rolling_features_exclude_current_row(sample_raw_df):
    df = _sorted(sample_raw_df).copy()
    outlier_idx = df[(df.store_id == 101) & (df.sku_id == 1)].index[10]
    df.loc[outlier_idx, "units_sold"] = 99999.0

    result = build_rolling_features(df, GROUP_COLS, "units_sold", [4], ["mean"])
    outlier_row_roll_mean = result.loc[outlier_idx, "units_sold_roll_mean_4"]
    assert outlier_row_roll_mean < 1000  # unaffected by its own extreme value


def test_discount_feature_handles_zero_base_price(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "base_price"] = 0.0
    result = add_price_features(df)
    assert np.isfinite(result.loc[0, "discount"])
    assert result.loc[0, "discount"] == 0.0


def test_feature_matrix_drops_incomplete_lag_rows(sample_raw_df):
    config = FeatureConfig(lags=[1, 2, 3, 4], rolling_windows=[4], rolling_stats=["mean"])
    feat_df, _ = build_feature_matrix(sample_raw_df, config, GROUP_COLS, "week", "units_sold")
    # 2 groups x 20 weeks, drop first max(lag)=4 rows per group -> 2 * 16 = 32 rows
    assert len(feat_df) == 32
    assert feat_df.filter(like="_lag_").isna().sum().sum() == 0
    assert feat_df.filter(like="_roll_").isna().sum().sum() == 0


def test_encode_categoricals_produces_all_known_dummy_columns_for_partial_input(sample_raw_df):
    """Simulates inference-time encoding where only one store/sku is present:
    dummy columns for every category seen at training time must still appear.
    """
    category_values = {
        "store_id": sorted(sample_raw_df["store_id"].unique().tolist()),
        "sku_id": sorted(sample_raw_df["sku_id"].unique().tolist()),
    }
    single_row_df = sample_raw_df.iloc[[0]]
    encoded, dummy_cols = encode_categoricals(
        single_row_df, cols=("store_id", "sku_id"), categories=category_values
    )
    assert len(dummy_cols) == len(category_values["store_id"]) + len(category_values["sku_id"])
    assert set(dummy_cols).issubset(encoded.columns)


def test_build_feature_matrix_feature_columns_exclude_identifiers(sample_raw_df):
    config = FeatureConfig(lags=[1, 2], rolling_windows=[2], rolling_stats=["mean"])
    _, feature_columns = build_feature_matrix(sample_raw_df, config, GROUP_COLS, "week", "units_sold")
    for excluded in ("record_ID", "week", "units_sold", "store_id", "sku_id"):
        assert excluded not in feature_columns


def test_build_feature_matrix_no_nans_in_feature_columns(sample_raw_df):
    """Guards the exact bug found in the real dataset: a missing total_price
    must not silently propagate NaNs into discount/price_diff and break
    models that can't fit on missing values (LinearRegression, RandomForest).
    """
    df = sample_raw_df.copy()
    df.loc[0, "total_price"] = np.nan
    df["total_price"] = df["total_price"].fillna(df["base_price"])

    config = FeatureConfig(lags=[1, 2], rolling_windows=[2], rolling_stats=["mean"])
    feat_df, feature_columns = build_feature_matrix(df, config, GROUP_COLS, "week", "units_sold")
    assert feat_df[feature_columns].isna().sum().sum() == 0
