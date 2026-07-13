"""FastAPI inference service for SKU-level weekly demand forecasting.

    uvicorn app.main:app --reload

Auto-generated docs at /docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.inference import (
    FeatureDriftError,
    InferenceService,
    InsufficientHistoryError,
    UnknownComboError,
    UnsupportedForecastWeekError,
)
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse, StoreSkuCombo

MODEL_PATH = Path("models/model.joblib")
METADATA_PATH = Path("models/metadata.json")
REFERENCE_HISTORY_PATH = Path("app/data/reference_history.parquet")

_service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = InferenceService(MODEL_PATH, METADATA_PATH, REFERENCE_HISTORY_PATH)
    yield
    _service = None


def get_service() -> InferenceService:
    if _service is None:
        raise RuntimeError("InferenceService not initialized.")
    return _service


app = FastAPI(
    title="SKU Demand Forecasting API",
    description="Predicts next-week units sold for a given store/SKU given planned price/promotion inputs.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    return HealthResponse(status="ok", model_version=service.model_version)


@app.get("/stores-skus", response_model=list[StoreSkuCombo])
def stores_skus() -> list[StoreSkuCombo]:
    service = get_service()
    combos = service.list_combos()
    return [StoreSkuCombo(**row) for row in combos.to_dict(orient="records")]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    service = get_service()
    try:
        predicted_units_sold = service.predict(request)
    except UnknownComboError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InsufficientHistoryError, UnsupportedForecastWeekError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FeatureDriftError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        store_id=request.store_id,
        sku_id=request.sku_id,
        week=request.week,
        predicted_units_sold=predicted_units_sold,
        model_version=service.model_version,
    )
