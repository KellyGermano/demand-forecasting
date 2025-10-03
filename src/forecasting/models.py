from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class DemandForecaster:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DemandForecaster":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            raise ValueError("Model not trained yet")

        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, got {list(X.columns)}"
            )

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Model not trained yet")

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str | Path) -> "DemandForecaster":
        return joblib.load(filepath)


if __name__ == "__main__":
    from src.forecasting.data import generate_synthetic_demand
    from src.forecasting.features import create_all_features, get_feature_columns

    df = generate_synthetic_demand(periods=100)
    df = create_all_features(df)
    df = df.dropna()

    X = df[get_feature_columns()]
    y = df["demand"]

    model = DemandForecaster()
    model.fit(X, y)

    predictions = model.predict(X[:5])
    print("Sample predictions:", predictions)

    print("\nTop 5 features:")
    print(model.get_feature_importance().head())