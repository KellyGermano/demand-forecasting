from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.settings import settings
from app.api.v1 import forecast, training

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Machine Learning API for demand forecasting with time series analysis",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "message": "Demand Forecasting API is running",
        "version": settings.app_version,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    from src.forecasting.train import ModelRegistry

    try:
        registry = ModelRegistry(settings.registry_path)
        current_model = registry.get_current_model()
        model_available = current_model is not None
    except Exception:
        model_available = False

    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model_available": model_available,
    }


app.include_router(
    forecast.router,
    prefix=settings.api_v1_prefix,
    tags=["Forecasting"],
)

app.include_router(
    training.router,
    prefix=settings.api_v1_prefix,
    tags=["Training"],
)


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Check /docs for available endpoints."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)