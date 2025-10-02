from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from src.forecasting.train import train_demand_forecasting_model

router = APIRouter()


class TrainingRequest(BaseModel):
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size (0.1-0.5)")
    n_estimators: int = Field(default=100, ge=10, le=500, description="Number of trees")
    max_depth: int = Field(default=20, ge=5, le=50, description="Maximum tree depth")
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class TrainingResponse(BaseModel):
    status: str
    message: str
    model_id: str
    metrics: dict


class TrainingStartedResponse(BaseModel):
    status: str
    message: str


def train_model_task(
    test_size: float,
    n_estimators: int,
    max_depth: int,
    random_state: int
):
    try:
        train_demand_forecasting_model(
            data_path=None,
            test_size=test_size,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")


@router.post(
    "/models/train",
    response_model=TrainingResponse,
    summary="Train a new model",
    description="Trains a new demand forecasting model with the specified parameters",
)
async def train_model(request: TrainingRequest = TrainingRequest()):
    try:
        print(f"Starting training with parameters: {request.model_dump()}")

        model, metrics, model_id = train_demand_forecasting_model(
            data_path=None,
            test_size=request.test_size,
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            random_state=request.random_state,
        )

        return TrainingResponse(
            status="success",
            message="Model trained successfully",
            model_id=model_id,
            metrics=metrics,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.post(
    "/models/train-async",
    response_model=TrainingStartedResponse,
    summary="Train a new model (async)",
    description="Starts training a new model in the background",
)
async def train_model_async(
    background_tasks: BackgroundTasks,
    request: TrainingRequest = TrainingRequest()
):
    background_tasks.add_task(
        train_model_task,
        request.test_size,
        request.n_estimators,
        request.max_depth,
        request.random_state,
    )

    return TrainingStartedResponse(
        status="started",
        message="Model training started in background. Check /api/v1/metrics for updates.",
    )