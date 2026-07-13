"""Inference service: reconstructs the exact training-time feature pipeline
for a single prediction request.

Deliberately reuses `src.features.build_features.build_feature_matrix` (the
same function `src/training/train.py` calls) rather than re-implementing
feature logic here -- this is the single mechanism that prevents train/serve
skew. The only "custom" logic here is turning one API request into a
one-row-longer history frame that `build_feature_matrix` can process
identically to training data.

v1 scope, documented deliberately rather than silently assumed:
- Only supports forecasting the week immediately following the latest known
  history for a given (store_id, sku_id) -- no multi-week-ahead recursive
  forecasting, since that would compound model error in an undocumented way.
- No cold-start support: a (store_id, sku_id) combo not seen during training
  (i.e. not present in the bundled reference history) cannot be scored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import build_feature_matrix
from src.models.persistence import ModelMetadata, load_metadata, load_model
from src.utils.config import FeatureConfig

GROUP_COLS = ["store_id", "sku_id"]
DATE_COL = "week"
TARGET_COL = "units_sold"
MIN_HISTORY_WEEKS = 12


class UnknownComboError(Exception):
    """Raised when the requested (store_id, sku_id) isn't in the reference history."""


class InsufficientHistoryError(Exception):
    """Raised when fewer than MIN_HISTORY_WEEKS of prior history are available."""


class UnsupportedForecastWeekError(Exception):
    """Raised when the requested week isn't the immediate next week after history."""


class FeatureDriftError(Exception):
    """Raised when feature engineering doesn't produce the expected model inputs.

    Indicates a bug or a model/metadata/reference-history version mismatch,
    not a client error.
    """


class InferenceService:
    def __init__(self, model_path: str | Path, metadata_path: str | Path, reference_history_path: str | Path):
        self.model = load_model(model_path)
        self.metadata: ModelMetadata = load_metadata(metadata_path)
        self.feature_config = FeatureConfig.model_validate(self.metadata.config_snapshot["features"])

        history = pd.read_parquet(reference_history_path)
        history[DATE_COL] = pd.to_datetime(history[DATE_COL])
        self.history = history.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)

    def list_combos(self) -> pd.DataFrame:
        summary = self.history.groupby(GROUP_COLS)[DATE_COL].max().reset_index()
        summary["next_predictable_week"] = summary[DATE_COL] + pd.Timedelta(days=7)
        return summary.rename(columns={DATE_COL: "latest_known_week"})

    def _combo_history(self, store_id: int, sku_id: int) -> pd.DataFrame:
        mask = (self.history["store_id"] == store_id) & (self.history["sku_id"] == sku_id)
        combo_history = self.history[mask]
        if combo_history.empty:
            raise UnknownComboError(f"No history found for store_id={store_id}, sku_id={sku_id}.")
        return combo_history

    def predict(self, request) -> float:
        combo_history = self._combo_history(request.store_id, request.sku_id)

        if len(combo_history) < MIN_HISTORY_WEEKS:
            raise InsufficientHistoryError(
                f"Only {len(combo_history)} weeks of history available for "
                f"store_id={request.store_id}, sku_id={request.sku_id}; "
                f"at least {MIN_HISTORY_WEEKS} are required."
            )

        last_known_week = combo_history[DATE_COL].max()
        expected_week = last_known_week + pd.Timedelta(days=7)
        request_week = pd.Timestamp(request.week)
        if request_week != expected_week:
            raise UnsupportedForecastWeekError(
                f"This service only forecasts the week immediately following the latest known "
                f"history for a store/SKU. For store_id={request.store_id}, sku_id={request.sku_id}, "
                f"that week is {expected_week.date()}, but {request_week.date()} was requested."
            )

        synthetic_row = pd.DataFrame(
            [
                {
                    "record_ID": -1,
                    DATE_COL: request_week,
                    "store_id": request.store_id,
                    "sku_id": request.sku_id,
                    "total_price": request.total_price,
                    "base_price": request.base_price,
                    "is_featured_sku": request.is_featured_sku,
                    "is_display_sku": request.is_display_sku,
                    TARGET_COL: np.nan,
                }
            ]
        )
        combined = pd.concat([combo_history, synthetic_row], ignore_index=True)

        feature_df, _ = build_feature_matrix(
            combined,
            self.feature_config,
            GROUP_COLS,
            DATE_COL,
            TARGET_COL,
            category_values=self.metadata.category_values,
        )

        if feature_df.empty:
            raise FeatureDriftError(
                "Feature engineering produced no usable row for this request -- likely a "
                "mismatch between the bundled reference history and the trained model."
            )

        row = feature_df.iloc[[-1]].reindex(columns=self.metadata.feature_columns, fill_value=0)
        engineered_cols = [c for c in self.metadata.feature_columns if "_lag_" in c or "_roll_" in c]
        if row[engineered_cols].isna().any().any():
            raise FeatureDriftError("Engineered lag/rolling features are missing for this request.")

        prediction_label = self.model.predict(row)[0]
        prediction = (
            np.expm1(prediction_label) if self.feature_config.log_transform_target else prediction_label
        )
        return float(max(0.0, round(prediction, 2)))

    @property
    def model_version(self) -> str:
        return self.metadata.git_commit or self.metadata.training_date
