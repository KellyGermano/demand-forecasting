import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.forecasting.models import DemandForecaster
from src.forecasting.features import create_all_features, get_feature_columns
from src.forecasting.train import ModelRegistry


class DemandPredictor:
    def __init__(self, registry_path: str | Path = "models/registry.json"):
        self.registry = ModelRegistry(registry_path)
        self.model: Optional[DemandForecaster] = None
        self.model_info: Optional[dict] = None
        self._load_current_model()

    def _load_current_model(self) -> None:
        model_info = self.registry.get_current_model()
        if model_info is None:
            raise ValueError(
                "No trained model found in registry. Please train a model first."
            )

        model_path = Path(model_info["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = DemandForecaster.load(model_path)
        self.model_info = model_info
        print(f"Loaded model: {self.registry.registry['current_model']}")

    def predict_future(
        self, historical_data: pd.DataFrame, horizon: int = 7
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")

        predictions = []
        current_data = historical_data.copy()

        for step in range(horizon):
            df_with_features = create_all_features(current_data)
            df_with_features = df_with_features.dropna()

            if len(df_with_features) == 0:
                raise ValueError("Not enough historical data to create features")

            last_row = df_with_features.iloc[[-1]]
            X = last_row[get_feature_columns()]

            pred = self.model.predict(X)[0]
            predictions.append(pred)

            next_date = current_data["date"].max() + pd.Timedelta(days=1)
            new_row = pd.DataFrame({"date": [next_date], "demand": [pred]})
            current_data = pd.concat([current_data, new_row], ignore_index=True)

        return np.array(predictions)

    def get_model_info(self) -> dict:
        if self.model_info is None:
            raise ValueError("Model not loaded")
        return self.model_info

    def get_model_metrics(self) -> dict:
        if self.model_info is None:
            raise ValueError("Model not loaded")
        return self.model_info.get("metrics", {})


def predict_demand(
    historical_data_path: str | Path,
    horizon: int = 7,
    registry_path: str | Path = "models/registry.json",
) -> dict:
    from src.forecasting.data import load_data

    historical_data = load_data(historical_data_path)

    predictor = DemandPredictor(registry_path)

    predictions = predictor.predict_future(historical_data, horizon=horizon)

    model_info = predictor.get_model_info()

    return {
        "predictions": predictions.tolist(),
        "horizon": horizon,
        "model_id": model_info.get("created_at"),
        "model_type": model_info.get("model_type"),
        "metrics": model_info.get("metrics", {}),
    }


if __name__ == "__main__":
    print("Testing inference system...\n")

    result = predict_demand(
        historical_data_path="data/raw/demand_data.csv", horizon=14
    )

    print(f"Forecast horizon: {result['horizon']} days")
    print(f"Model type: {result['model_type']}")
    print(f"\nModel metrics:")
    for metric, value in result["metrics"].items():
        print(f"  {metric}: {value:.2f}")

    print(f"\nPredictions for next {result['horizon']} days:")
    for i, pred in enumerate(result["predictions"], 1):
        print(f"  Day {i}: {pred:.2f}")

    print(f"\nMean prediction: {np.mean(result['predictions']):.2f}")
    print(f"Std prediction: {np.std(result['predictions']):.2f}")