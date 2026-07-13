from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

import app.main as app_main
from src.features.build_features import build_feature_matrix
from src.models.persistence import save_model
from src.utils.config import FeatureConfig

GROUP_COLS = ["store_id", "sku_id"]
FEATURE_CONFIG = FeatureConfig(
    lags=[1, 2, 3, 4, 8, 12], rolling_windows=[4, 8], rolling_stats=["mean", "std"]
)
CATEGORY_VALUES = {"store_id": [101, 102], "sku_id": [1, 2]}


@pytest.fixture
def api_client(tmp_path, sample_raw_df):
    feat_df, feature_columns = build_feature_matrix(
        sample_raw_df, FEATURE_CONFIG, GROUP_COLS, "week", "units_sold", category_values=CATEGORY_VALUES
    )
    model = LinearRegression().fit(feat_df[feature_columns], feat_df["units_sold_log"])

    model_dir = tmp_path / "model"
    save_model(
        model=model,
        feature_columns=feature_columns,
        category_values=CATEGORY_VALUES,
        model_type="linear_regression",
        metrics={"rmse": 0.0},
        config_snapshot={"features": FEATURE_CONFIG.model_dump()},
        out_dir=model_dir,
    )

    history = sample_raw_df.sort_values(GROUP_COLS + ["week"]).groupby(GROUP_COLS).tail(12)
    # a combo with only 5 weeks of history, to exercise the insufficient-history path
    short_history = (
        sample_raw_df[(sample_raw_df.store_id == 101) & (sample_raw_df.sku_id == 1)].tail(5).copy()
    )
    short_history["store_id"] = 103
    short_history["sku_id"] = 3
    history = pd.concat([history, short_history], ignore_index=True)

    history_path = tmp_path / "reference_history.parquet"
    history.to_parquet(history_path, index=False)

    app_main.MODEL_PATH = model_dir / "model.joblib"
    app_main.METADATA_PATH = model_dir / "metadata.json"
    app_main.REFERENCE_HISTORY_PATH = history_path

    with TestClient(app_main.app) as client:
        yield client


def _next_predictable_week(sample_raw_df, store_id, sku_id) -> str:
    group = sample_raw_df[(sample_raw_df.store_id == store_id) & (sample_raw_df.sku_id == sku_id)]
    last_week = group["week"].max()
    return (last_week + pd.Timedelta(days=7)).date().isoformat()


def test_health_endpoint_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stores_skus_lists_known_combos(api_client):
    response = api_client.get("/stores-skus")
    assert response.status_code == 200
    combos = {(c["store_id"], c["sku_id"]) for c in response.json()}
    assert (101, 1) in combos
    assert (102, 2) in combos


def test_predict_valid_request_returns_200_and_nonnegative_prediction(api_client, sample_raw_df):
    week = _next_predictable_week(sample_raw_df, 101, 1)
    response = api_client.post(
        "/predict",
        json={
            "store_id": 101,
            "sku_id": 1,
            "week": week,
            "total_price": 90.0,
            "base_price": 100.0,
            "is_featured_sku": 1,
            "is_display_sku": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_units_sold"] >= 0


def test_predict_unknown_store_sku_returns_404(api_client, sample_raw_df):
    week = _next_predictable_week(sample_raw_df, 101, 1)
    response = api_client.post(
        "/predict",
        json={
            "store_id": 9999,
            "sku_id": 9999,
            "week": week,
            "total_price": 90.0,
            "base_price": 100.0,
            "is_featured_sku": 0,
            "is_display_sku": 0,
        },
    )
    assert response.status_code == 404


def test_predict_missing_field_returns_422(api_client, sample_raw_df):
    week = _next_predictable_week(sample_raw_df, 101, 1)
    response = api_client.post(
        "/predict",
        json={
            "store_id": 101,
            "sku_id": 1,
            "week": week,
            "total_price": 90.0,
            # base_price missing
            "is_featured_sku": 0,
            "is_display_sku": 0,
        },
    )
    assert response.status_code == 422


def test_predict_insufficient_history_returns_422(api_client, sample_raw_df):
    week = _next_predictable_week(sample_raw_df, 101, 1)
    response = api_client.post(
        "/predict",
        json={
            "store_id": 103,
            "sku_id": 3,
            "week": week,
            "total_price": 90.0,
            "base_price": 100.0,
            "is_featured_sku": 0,
            "is_display_sku": 0,
        },
    )
    assert response.status_code == 422


def test_predict_wrong_forecast_week_returns_422(api_client, sample_raw_df):
    response = api_client.post(
        "/predict",
        json={
            "store_id": 101,
            "sku_id": 1,
            "week": "2099-01-01",
            "total_price": 90.0,
            "base_price": 100.0,
            "is_featured_sku": 0,
            "is_display_sku": 0,
        },
    )
    assert response.status_code == 422
