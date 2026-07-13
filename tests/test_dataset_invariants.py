"""Integration test against the real, committed dataset.

Locks in the panel structure this whole pipeline design depends on -- if
this ever fails, the raw dataset has changed shape and the feature/split
design in this repo needs to be revisited.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.load import load_raw_train

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/raw/train.csv not present")
def test_real_dataset_panel_structure():
    df = load_raw_train(DATA_PATH)

    assert df["store_id"].nunique() == 76
    assert df["sku_id"].nunique() == 28
    assert df.groupby(["store_id", "sku_id"]).ngroups == 1155

    weeks_per_group = df.groupby(["store_id", "sku_id"]).size()
    assert (weeks_per_group == 130).all()

    # load_raw_train imputes the one known missing total_price with base_price
    assert df.isna().sum().sum() == 0


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/raw/train.csv not present")
def test_real_dataset_has_exactly_one_missing_value_before_imputation():
    raw = pd.read_csv(DATA_PATH)
    assert raw.isna().sum().sum() == 1
    assert raw["total_price"].isna().sum() == 1
