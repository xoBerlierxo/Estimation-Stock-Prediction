"""Model serialization and the metadata contract that keeps training and
serving in sync.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel
from sklearn.base import BaseEstimator


class ModelMetadata(BaseModel):
    feature_columns: list[str]
    category_values: dict[str, list]
    training_date: str
    git_commit: str | None
    model_type: str
    metrics: dict[str, Any]
    config_snapshot: dict[str, Any]
    library_versions: dict[str, str]


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return None


def _library_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("scikit-learn", "xgboost", "lightgbm", "pandas", "numpy"):
        try:
            versions[pkg] = pkg_version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def save_model(
    model: BaseEstimator,
    feature_columns: list[str],
    category_values: dict[str, list],
    model_type: str,
    metrics: dict[str, Any],
    config_snapshot: dict[str, Any],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")

    metadata = ModelMetadata(
        feature_columns=feature_columns,
        category_values=category_values,
        training_date=datetime.now(timezone.utc).isoformat(),
        git_commit=_current_git_commit(),
        model_type=model_type,
        metrics=metrics,
        config_snapshot=config_snapshot,
        library_versions=_library_versions(),
    )
    (out_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def load_model(path: str | Path) -> BaseEstimator:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_metadata(path: str | Path) -> ModelMetadata:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return ModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))
