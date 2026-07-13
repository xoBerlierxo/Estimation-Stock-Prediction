from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.models.persistence import load_metadata, load_model, save_model


def test_save_and_load_model_roundtrip(tmp_path):
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = (2 * X).ravel()
    model = LinearRegression().fit(X, y)

    save_model(
        model=model,
        feature_columns=["x"],
        category_values={"store_id": [1, 2]},
        model_type="linear_regression",
        metrics={"rmse": 0.0},
        config_snapshot={"foo": "bar"},
        out_dir=tmp_path,
    )

    loaded = load_model(tmp_path / "model.joblib")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_metadata_contains_required_keys(tmp_path):
    model = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])
    save_model(
        model=model,
        feature_columns=["x"],
        category_values={"store_id": [1, 2]},
        model_type="linear_regression",
        metrics={"rmse": 0.1},
        config_snapshot={"foo": "bar"},
        out_dir=tmp_path,
    )
    metadata = load_metadata(tmp_path / "metadata.json")
    assert metadata.feature_columns == ["x"]
    assert metadata.model_type == "linear_regression"
    assert metadata.metrics == {"rmse": 0.1}
    assert "scikit-learn" in metadata.library_versions


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


def test_load_metadata_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "does_not_exist.json")
