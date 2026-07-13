from __future__ import annotations

from src.training.split import rolling_origin_splits, time_based_holdout_split


def test_time_based_holdout_split_boundary(sample_raw_df):
    n_groups = sample_raw_df.groupby(["store_id", "sku_id"]).ngroups
    train_val, test = time_based_holdout_split(sample_raw_df, n_test_weeks=5)

    assert train_val["week"].max() < test["week"].min()
    assert len(test) == 5 * n_groups


def test_rolling_origin_splits_no_future_leakage(sample_raw_df):
    folds = rolling_origin_splits(sample_raw_df, n_splits=3, val_weeks=2)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_weeks = sample_raw_df.iloc[train_idx]["week"]
        val_weeks = sample_raw_df.iloc[val_idx]["week"]
        assert train_weeks.max() < val_weeks.min()


def test_rolling_origin_splits_expanding_window(sample_raw_df):
    folds = rolling_origin_splits(sample_raw_df, n_splits=3, val_weeks=2)
    train_sizes = [len(train_idx) for train_idx, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert all(s2 > s1 for s1, s2 in zip(train_sizes, train_sizes[1:]))


def test_rolling_origin_splits_val_size_constant(sample_raw_df):
    n_groups = sample_raw_df.groupby(["store_id", "sku_id"]).ngroups
    folds = rolling_origin_splits(sample_raw_df, n_splits=3, val_weeks=2)
    for _, val_idx in folds:
        assert len(val_idx) == 2 * n_groups
