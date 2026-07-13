from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Two (store, sku) series x 20 weeks each, with known, hand-computable values.

    Deterministic and independent of the real CSV so unit tests run fast and
    don't depend on dataset contents.
    """
    weeks = pd.date_range("2020-01-06", periods=20, freq="7D")
    rows = []
    record_id = 1
    for store_id, sku_id, base_units in [(101, 1, 10), (102, 2, 50)]:
        for i, week in enumerate(weeks):
            rows.append(
                {
                    "record_ID": record_id,
                    "week": week,
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "total_price": 100.0 - i,
                    "base_price": 100.0,
                    "is_featured_sku": int(i % 5 == 0),
                    "is_display_sku": int(i % 7 == 0),
                    "units_sold": float(base_units + i),
                }
            )
            record_id += 1
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def shuffled_sample_raw_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    return sample_raw_df.sample(frac=1.0, random_state=0).reset_index(drop=True)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
