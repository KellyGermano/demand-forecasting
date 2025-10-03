from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class DemandForecaster:
    """
    A wrapper for a RandomForestRegressor model to forecast demand.

    This class encapsulates the model training, prediction, and serialization
    (saving/loading), ensuring a consistent interface and feature handling.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initializes the DemandForecaster with a RandomForestRegressor model.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            random_state (int): Controls both the randomness of the bootstrapping of the samples used
                                when building trees and the sampling of the features to consider
                                when looking for the best split at each node.
            n_jobs (int): The number of jobs to run in parallel. -1 means using all processors.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # This type hint is the key to fixing the mypy error.
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DemandForecaster":
        """
        Trains the forecasting model.

        Args:
            X (pd.DataFrame): The input features for training.
            y (pd.Series): The target variable (demand).

        Returns:
            DemandForecaster: The instance of the class itself.
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            X (pd.DataFrame): The input features for prediction.

        Returns:
            np.ndarray: The predicted values.

        Raises:
            ValueError: If the model has not been trained yet or if feature names mismatch.
        """
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet. Call .fit() first.")

        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, got {list(X.columns)}"
            )

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Gets the feature importances from the trained model.

        Returns:
            pd.DataFrame: A DataFrame with features and their importance scores, sorted descending.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.feature_names is None or not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model has not been trained yet. Call .fit() first.")

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save(self, filepath: str | Path) -> None:
        """
        Saves the entire model instance to a file.

        Args:
            filepath (str | Path): The path to save the model file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str | Path) -> "DemandForecaster":
        """
        Loads a model instance from a file.

        Args:
            filepath (str | Path): The path to the model file.

        Returns:
            DemandForecaster: The loaded model instance.
        """
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    # This block is for demonstration and testing purposes.
    # It will only run when the script is executed directly.
    print("Running a demonstration of the DemandForecaster class...")

    # Create synthetic data for the example
    def generate_synthetic_demand(periods: int, start_date: str = "2023-01-01"):
        dates = pd.date_range(start=start_date, periods=periods, freq="D")
        base_demand = 100 + 10 * np.sin(np.arange(periods) * 2 * np.pi / 365.25)
        noise = np.random.normal(0, 5, periods)
        demand = base_demand + noise
        return pd.DataFrame({"date": dates, "demand": demand}).set_index("date")

    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Creates time series features from a datetime index."""
        df_copy = df.copy()

        # Esta línea asegura que el índice sea de tipo fecha y soluciona el error
        df_copy.index = pd.to_datetime(df_copy.index)

        df_copy["dayofweek"] = df_copy.index.dayofweek
        df_copy["month"] = df_copy.index.month
        df_copy["year"] = df_copy.index.year
        df_copy["dayofyear"] = df_copy.index.dayofyear
        return df_copy

    # 1. Generate and prepare data
    df_data = generate_synthetic_demand(periods=500)
    df_features = create_time_features(df_data)
    df_final = df_features.dropna()

    feature_cols = ["dayofweek", "month", "year", "dayofyear"]
    X_train = df_final[feature_cols]
    y_train = df_final["demand"]

    # 2. Train the model
    model_instance = DemandForecaster()
    model_instance.fit(X_train, y_train)
    print("\nModel has been trained.")

    # 3. Make predictions
    sample_predictions = model_instance.predict(X_train.head())
    print(f"\nSample predictions for the first 5 data points: \n{sample_predictions}")

    # 4. Get feature importance
    feature_importances = model_instance.get_feature_importance()
    print("\nTop 5 features by importance:")
    print(feature_importances.head())

    # 5. Save and load the model
    model_path = Path("models/demand_forecaster.joblib")
    model_instance.save(model_path)
    loaded_model = DemandForecaster.load(model_path)
    print(f"\nModel re-loaded successfully from {model_path}")

    # Verify loaded model works
    loaded_model_predictions = loaded_model.predict(X_train.head())
    assert np.allclose(sample_predictions, loaded_model_predictions)
    print("\nVerified that the loaded model produces the same predictions.")
