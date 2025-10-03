import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.forecasting.data import generate_synthetic_demand, load_data, save_data
from src.forecasting.evaluate import (
    calculate_metrics,
    plot_predictions,
    plot_residuals,
    print_metrics,
)
from src.forecasting.features import create_all_features, get_feature_columns
from src.forecasting.models import DemandForecaster


class ModelRegistry:
    def __init__(self, registry_path: str | Path = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {"current_model": None, "models": {}}
            self._save_registry()

    def _save_registry(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_id: str,
        model_path: str,
        metrics: dict,
        model_type: str = "RandomForest",
    ) -> None:
        self.registry["models"][model_id] = {
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "model_type": model_type,
            "path": model_path,
        }
        self.registry["current_model"] = model_id
        self._save_registry()
        print(f"Model {model_id} registered successfully")

    def get_current_model(self) -> Optional[dict]:
        if self.registry["current_model"] is None:
            return None
        return self.registry["models"].get(self.registry["current_model"])

    def get_all_models(self) -> dict:
        return self.registry["models"]


def train_demand_forecasting_model(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 20,
    random_state: int = 42,
) -> tuple[DemandForecaster, dict, str]:
    print("\n" + "=" * 60)
    print("DEMAND FORECASTING MODEL TRAINING")
    print("=" * 60)

    if data_path is None:
        print("\nGenerating synthetic data...")
        df = generate_synthetic_demand(periods=730, random_seed=random_state)
        save_data(df, "data/raw/demand_data.csv")
    else:
        print(f"\nLoading data from {data_path}...")
        df = load_data(data_path)

    print(f"Loaded {len(df)} records")

    print("\nCreating features...")
    df = create_all_features(df)
    df = df.dropna()
    print(f"After feature engineering and dropna: {len(df)} records")

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nTrain set: {len(train_df)} records")
    print(f"Test set:  {len(test_df)} records")

    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols]
    y_train = train_df["demand"]
    X_test = test_df[feature_cols]
    y_test = test_df["demand"]

    print("\nTraining Random Forest model...")
    model = DemandForecaster(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Training completed!")

    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test.values, y_pred)
    print_metrics(metrics)

    print("\nTop 5 important features:")
    feature_importance = model.get_feature_importance()
    print(feature_importance.head().to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"model-{timestamp}"
    model_path = f"models/production/{model_id}.joblib"

    print(f"\nSaving model to {model_path}...")
    model.save(model_path)

    print("\nGenerating evaluation plots...")
    plot_predictions(
        y_test.values,
        y_pred,
        dates=test_df["date"],
        save_path="reports/figures/predictions.png",
    )
    plot_residuals(
        y_test.values, y_pred, save_path="reports/figures/residuals.png"
    )

    print("\nRegistering model...")
    registry = ModelRegistry()
    registry.register_model(
        model_id=model_id, model_path=model_path, metrics=metrics, model_type="RandomForest"
    )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")

    return model, metrics, model_id


if __name__ == "__main__":
    train_demand_forecasting_model()