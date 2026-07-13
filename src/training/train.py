"""End-to-end training entry point.

    python -m src.training.train --config configs/train.yaml [--quick]

Orchestrates: load -> validate -> build features -> time-based split ->
per-model RandomizedSearchCV (leakage-safe rolling-origin CV) -> select best
by mean CV RMSE (log scale) -> refit -> evaluate once on the untouched test
set -> persist model + metadata -> write plots and the markdown report.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.data.load import load_raw_train
from src.data.validate import check_missing_values, validate_panel_completeness, validate_schema
from src.evaluation.metrics import evaluate_predictions
from src.features.build_features import build_feature_matrix
from src.models.persistence import save_model
from src.models.registry import PARAM_DISTRIBUTIONS, get_model
from src.training.split import rolling_origin_splits, time_based_holdout_split
from src.utils.config import TrainConfig, load_config
from src.utils.logging import get_logger
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    write_metrics_report,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare demand-forecasting models.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of model candidates to run, overriding the config.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink search iterations/CV folds for a fast smoke test (not for reported metrics).",
    )
    return parser.parse_args()


def run(config: TrainConfig) -> None:
    data_cfg, feat_cfg, split_cfg, model_cfg, out_cfg = (
        config.data,
        config.features,
        config.split,
        config.models,
        config.output,
    )

    logger.info("Loading raw data from %s", data_cfg.raw_train_path)
    raw_df = load_raw_train(data_cfg.raw_train_path, date_col=data_cfg.date_col)
    validate_schema(raw_df)
    check_missing_values(raw_df)
    gap_report = validate_panel_completeness(raw_df, data_cfg.group_cols, data_cfg.date_col)
    if not gap_report.empty:
        logger.warning("Panel completeness gaps detected:\n%s", gap_report)

    category_values = {col: sorted(raw_df[col].unique().tolist()) for col in data_cfg.group_cols}

    logger.info("Building feature matrix (lags=%s, rolling=%s)", feat_cfg.lags, feat_cfg.rolling_windows)
    feature_df, feature_columns = build_feature_matrix(
        raw_df,
        feat_cfg,
        data_cfg.group_cols,
        data_cfg.date_col,
        data_cfg.target_col,
        category_values=category_values,
    )
    logger.info("Feature matrix: %d rows, %d model features", len(feature_df), len(feature_columns))

    target_col = data_cfg.target_col
    label_col = f"{target_col}_log" if feat_cfg.log_transform_target else target_col

    train_val_df, test_df = time_based_holdout_split(feature_df, data_cfg.date_col, split_cfg.n_test_weeks)
    logger.info("Train+val: %d rows | Test: %d rows", len(train_val_df), len(test_df))

    X_train_val = train_val_df[feature_columns]
    y_train_val = train_val_df[label_col]
    X_test = test_df[feature_columns]
    y_test_original = test_df[target_col].to_numpy()

    cv_folds = rolling_origin_splits(
        train_val_df, data_cfg.date_col, split_cfg.cv_n_splits, split_cfg.cv_val_weeks
    )

    results = []
    fitted_models: dict[str, BaseEstimator] = {}

    for name in model_cfg.candidates:
        logger.info("=== Training candidate: %s ===", name)
        start = time.time()
        estimator = get_model(name, model_cfg.random_state)
        param_dist = PARAM_DISTRIBUTIONS.get(name, {})

        if param_dist:
            search = RandomizedSearchCV(
                estimator,
                param_distributions=param_dist,
                n_iter=model_cfg.n_iter_search,
                cv=cv_folds,
                scoring=model_cfg.scoring,
                random_state=model_cfg.random_state,
                n_jobs=-1,
                refit=True,
                verbose=1,
            )
            search.fit(X_train_val, y_train_val)
            best_model = search.best_estimator_
            cv_score = search.best_score_
            best_params = search.best_params_
        else:
            cv_scores = cross_val_score(
                estimator, X_train_val, y_train_val, cv=cv_folds, scoring=model_cfg.scoring, n_jobs=-1
            )
            cv_score = float(np.mean(cv_scores))
            best_model = estimator.fit(X_train_val, y_train_val)
            best_params = {}

        elapsed = time.time() - start
        logger.info(
            "%s: cv_score(log-scale, %s)=%.4f, params=%s, elapsed=%.1fs",
            name,
            model_cfg.scoring,
            cv_score,
            best_params,
            elapsed,
        )

        y_pred_label = best_model.predict(X_test)
        y_pred_original = np.expm1(y_pred_label) if feat_cfg.log_transform_target else y_pred_label
        y_pred_original = np.clip(y_pred_original, 0, None)

        test_metrics = evaluate_predictions(y_test_original, y_pred_original)
        logger.info("%s test metrics: %s", name, test_metrics)

        fitted_models[name] = best_model
        results.append(
            {
                "model": name,
                "cv_score_log_rmse": -cv_score if "neg" in model_cfg.scoring else cv_score,
                "best_params": best_params,
                **test_metrics,
            }
        )

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    logger.info("Best model by test RMSE: %s", best_name)

    y_pred_label = best_model.predict(X_test)
    y_pred_original = np.expm1(y_pred_label) if feat_cfg.log_transform_target else y_pred_label
    y_pred_original = np.clip(y_pred_original, 0, None)
    final_metrics = evaluate_predictions(y_test_original, y_pred_original)

    out_cfg.model_dir.mkdir(parents=True, exist_ok=True)
    out_cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (out_cfg.reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    save_model(
        model=best_model,
        feature_columns=feature_columns,
        category_values=category_values,
        model_type=best_name,
        metrics=final_metrics,
        config_snapshot=config.model_dump(mode="json"),
        out_dir=out_cfg.model_dir,
    )

    plot_feature_importance(
        best_model, feature_columns, out_cfg.reports_dir / "figures" / "feature_importance.png"
    )
    plot_residuals(y_test_original, y_pred_original, out_cfg.reports_dir / "figures" / "residuals.png")
    plot_actual_vs_predicted(
        y_test_original, y_pred_original, out_cfg.reports_dir / "figures" / "actual_vs_predicted.png"
    )
    plot_model_comparison(results_df, out_cfg.reports_dir / "figures" / "model_comparison.png")

    display_cols = ["model", "cv_score_log_rmse", "rmse", "mae", "r2", "mape", "wape"]
    rationale = (
        f"`{best_name}` was selected by lowest RMSE on a held-out test set of the last "
        f"{split_cfg.n_test_weeks} weeks, never touched during hyperparameter search. "
        f"Hyperparameters were tuned with `RandomizedSearchCV` ({model_cfg.n_iter_search} iterations) "
        f"using a {split_cfg.cv_n_splits}-fold expanding-window (rolling-origin) time-series CV scheme "
        f"on the log1p-transformed target, so no future week ever leaks into a training fold."
    )
    write_metrics_report(results_df[display_cols], best_name, rationale, out_cfg.reports_dir / "metrics.md")

    logger.info("Training complete. Best model: %s | Test metrics: %s", best_name, final_metrics)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.models:
        config.models.candidates = [m.strip() for m in args.models.split(",")]
    if args.quick:
        config.models.n_iter_search = 2
        config.split.cv_n_splits = 2

    run(config)


if __name__ == "__main__":
    main()
