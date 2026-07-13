"""Model factories and hyperparameter search spaces for the candidates compared
in this project. Kept in one place so `train.py` stays a thin orchestrator.
"""

from __future__ import annotations

from typing import Callable

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

MODEL_REGISTRY: dict[str, Callable[[int], BaseEstimator]] = {
    "linear_regression": lambda random_state: LinearRegression(),
    "random_forest": lambda random_state: RandomForestRegressor(random_state=random_state, n_jobs=-1),
    "xgboost": lambda random_state: XGBRegressor(
        random_state=random_state, n_jobs=-1, objective="reg:squarederror"
    ),
    "lightgbm": lambda random_state: LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1),
}

PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "linear_regression": {},
    "random_forest": {
        # max_depth is deliberately capped (no `None`/unlimited option): with 125
        # mostly-sparse one-hot columns, unbounded trees are dramatically slower
        # to fit for negligible accuracy gain over a depth-capped forest.
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [8, 12, 16, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4, 8],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300, 500],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
    },
}


def get_model(name: str, random_state: int) -> BaseEstimator:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](random_state)
