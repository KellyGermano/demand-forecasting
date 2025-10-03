
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Demand Forecasting API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"

    api_key: str = "dev-api-key-change-in-production"
    debug: bool = False

    model_path: str = "models/production"
    data_path: str = "data"
    registry_path: str = "models/registry.json"

    max_forecast_horizon: int = 90
    default_forecast_horizon: int = 7

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()