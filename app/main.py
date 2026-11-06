"""FastAPI routes for e-commerce return predictions."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_one

app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description=(
        "Calculates delivery and return-history features and estimates the "
        "probability that an e-commerce order will be returned."
    ),
    version="1.0.0",
)


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Product_Category: str = Field(min_length=1, max_length=80)
    Product_Price_INR: float = Field(ge=0)
    Discount_Percent: float = Field(ge=0, le=100)
    Expected_Delivery_Days: float = Field(ge=0)
    Actual_Delivery_Days: float = Field(ge=0)
    Customer_Rating_PreOrder: float = Field(ge=0, le=5)
    Previous_Returns: int = Field(ge=0)
    Previous_Orders: int = Field(ge=0)
    Payment_Method: str = Field(min_length=1, max_length=80)
    Damaged_Package_Flag: Literal["Yes", "No"]
    Package_Size: Literal["Small", "Medium", "Large"]
    Promotion_Active: Literal["Yes", "No"]


class PredictionResponse(BaseModel):
    predicted_return_status: Literal["Returned", "Not Returned"]
    prediction_probability: float = Field(
        ge=0,
        le=1,
        description="Estimated probability that the product will be returned.",
    )
    model_name: str


@lru_cache(maxsize=1)
def _model() -> dict:
    return load_model()


@app.get("/health")
def health() -> dict[str, str | bool]:
    try:
        _model()
    except (OSError, ValueError, ImportError) as exc:
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": str(exc),
        }
    return {"status": "healthy", "model_loaded": True}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    response_description="Predicted status and probability of a return.",
)
def predict(product: ProductInput) -> PredictionResponse:
    try:
        result = predict_one(_model(), product.model_dump())
    except (OSError, ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model is unavailable: {exc}",
        ) from exc
    return PredictionResponse(**result)
