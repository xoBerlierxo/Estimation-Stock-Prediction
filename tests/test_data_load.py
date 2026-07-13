from __future__ import annotations

import pandas as pd

from src.data.load import load_raw_train


def test_load_raw_train_parses_week_as_datetime(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "record_ID,week,store_id,sku_id,total_price,base_price,is_featured_sku,is_display_sku,units_sold\n"
        "1,17/01/11,8091,216418,99.0,111.0,0,0,20\n"
    )
    df = load_raw_train(csv_path)
    assert pd.api.types.is_datetime64_any_dtype(df["week"])


def test_load_raw_train_dayfirst_parsing(tmp_path):
    """`17/01/11` must parse as 2011-01-17, not 2011-11-01 -- day comes first."""
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "record_ID,week,store_id,sku_id,total_price,base_price,is_featured_sku,is_display_sku,units_sold\n"
        "1,17/01/11,8091,216418,99.0,111.0,0,0,20\n"
    )
    df = load_raw_train(csv_path)
    parsed = df.loc[0, "week"]
    assert (parsed.year, parsed.month, parsed.day) == (2011, 1, 17)


def test_load_raw_train_imputes_missing_total_price_with_base_price(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "record_ID,week,store_id,sku_id,total_price,base_price,is_featured_sku,is_display_sku,units_sold\n"
        "1,17/01/11,8091,216418,,111.5,0,0,20\n"
    )
    df = load_raw_train(csv_path)
    assert df.loc[0, "total_price"] == 111.5
    assert df["total_price"].isna().sum() == 0


def test_load_raw_train_dtypes(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "record_ID,week,store_id,sku_id,total_price,base_price,is_featured_sku,is_display_sku,units_sold\n"
        "1,17/01/11,8091,216418,99.0,111.0,1,0,20\n"
    )
    df = load_raw_train(csv_path)
    assert df["store_id"].dtype.kind == "i"
    assert df["sku_id"].dtype.kind == "i"
    assert df["is_featured_sku"].dtype.kind == "i"
    assert df["units_sold"].dtype.kind == "f"
