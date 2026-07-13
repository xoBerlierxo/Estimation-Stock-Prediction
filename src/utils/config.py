"""Typed configuration for the training pipeline, loaded from a YAML file.

Every tunable parameter (paths, feature windows, split sizes, model search
space, output locations) lives here so a training run is fully reproducible
from `configs/train.yaml` plus the code at a given git commit.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_train_path: Path
    raw_test_path: Path
    processed_dir: Path
    date_col: str = "week"
    group_cols: list[str] = Field(default_factory=lambda: ["store_id", "sku_id"])
    target_col: str = "units_sold"


class FeatureConfig(BaseModel):
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 8, 12])
    rolling_windows: list[int] = Field(default_factory=lambda: [4, 8])
    rolling_stats: list[str] = Field(default_factory=lambda: ["mean", "std"])
    log_transform_target: bool = True


class SplitConfig(BaseModel):
    n_test_weeks: int = 8
    cv_n_splits: int = 5
    cv_val_weeks: int = 8


class ModelConfig(BaseModel):
    candidates: list[str] = Field(
        default_factory=lambda: ["linear_regression", "random_forest", "xgboost", "lightgbm"]
    )
    random_state: int = 42
    n_iter_search: int = 25
    scoring: str = "neg_root_mean_squared_error"


class OutputConfig(BaseModel):
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


class TrainConfig(BaseModel):
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a `TrainConfig` from a YAML file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig.model_validate(raw)
