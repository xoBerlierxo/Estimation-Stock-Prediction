"""Request/response contracts for the inference API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """What a merchandising/pricing planner knows ahead of a week: which
    store/SKU, the week being planned for, and that week's price/promo plan.
    """

    store_id: int
    sku_id: int
    week: date = Field(
        description=(
            "The week being forecast (must be the week immediately after the "
            "latest known history for this store/SKU)."
        )
    )
    total_price: float = Field(gt=0, description="Planned selling price for the week.")
    base_price: float = Field(gt=0, description="List/undiscounted price for the week.")
    is_featured_sku: Literal[0, 1] = Field(
        description="Whether the SKU is in the featured promotion for the week."
    )
    is_display_sku: Literal[0, 1] = Field(
        description="Whether the SKU gets a display placement for the week."
    )


class PredictionResponse(BaseModel):
    store_id: int
    sku_id: int
    week: date
    predicted_units_sold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


class StoreSkuCombo(BaseModel):
    store_id: int
    sku_id: int
    latest_known_week: date
    next_predictable_week: date
