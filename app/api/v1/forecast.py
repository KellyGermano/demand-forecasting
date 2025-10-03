from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.settings import settings
from src.forecasting.inference import predict_demand

router = APIRouter()


class ForecastResponse(BaseModel):
    predictions: List[float] = Field(..., description="List of predicted demand values")
    horizon: int = Field(..., description="Number of days forecasted")
    model_id: str = Field(..., description="ID of the model used")
    model_type: str = Field(..., description="Type of ML model")
    metrics: dict = Field(..., description="Model performance metrics")


class ErrorResponse(BaseModel):
    detail: str


@router.get(
    "/forecast",
    response_model=ForecastResponse,
    responses={
        503: {"model": ErrorResponse, "description": "No trained model available"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
    },
    summary="Get demand forecast",
    description="Returns demand predictions for the specified number of future days",
)
async def get_forecast(
    horizon: int = Query(
        default=settings.default_forecast_horizon,
        ge=1,
        le=settings.max_forecast_horizon,
        description=f"Number of days to forecast (1-{settings.max_forecast_horizon})",
    )
):
    try:
        result = predict_demand(
            historical_data_path=f"{settings.data_path}/raw/demand_data.csv",
            horizon=horizon,
            registry_path=settings.registry_path,
        )

        return ForecastResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model available. Please train a model first. Error: {str(e)}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Required file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get(
    "/models/current",
    summary="Get current model info",
    description="Returns information about the currently deployed model",
)
async def get_current_model():
    try:
        from src.forecasting.train import ModelRegistry

        registry = ModelRegistry(settings.registry_path)
        model_info = registry.get_current_model()

        if model_info is None:
            raise HTTPException(
                status_code=404, detail="No model currently deployed"
            )

        return {
            "model_id": registry.registry["current_model"],
            "info": model_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get(
    "/metrics",
    summary="Get model metrics",
    description="Returns performance metrics of the current model",
)
async def get_metrics():
    try:
        from src.forecasting.train import ModelRegistry

        registry = ModelRegistry(settings.registry_path)
        model_info = registry.get_current_model()

        if model_info is None:
            raise HTTPException(
                status_code=404, detail="No model currently deployed"
            )

        return {
            "model_id": registry.registry["current_model"],
            "metrics": model_info.get("metrics", {}),
            "model_type": model_info.get("model_type"),
            "created_at": model_info.get("created_at"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")