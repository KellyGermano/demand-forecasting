import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.forecasting.models import DemandForecaster
from src.forecasting.features import get_feature_columns


def test_model_initialization():
    model = DemandForecaster(n_estimators=50, max_depth=10, random_state=42)

    assert model.model.n_estimators == 50
    assert model.model.max_depth == 10
    assert model.model.random_state == 42
    assert model.feature_names is None


def test_model_fit_predict(sample_data_with_features):
    df = sample_data_with_features
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["demand"]

    model = DemandForecaster(n_estimators=10, random_state=42)
    model.fit(X, y)

    assert model.feature_names == feature_cols

    predictions = model.predict(X[:5])
    assert len(predictions) == 5
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_model_predict_without_training(sample_data_with_features):
    df = sample_data_with_features
    X = df[get_feature_columns()]

    model = DemandForecaster()

    with pytest.raises(ValueError, match="Model not trained yet"):
        model.predict(X)


def test_model_feature_mismatch(sample_data_with_features):
    df = sample_data_with_features
    feature_cols = get_feature_columns()
    X_train = df[feature_cols]
    y_train = df["demand"]

    model = DemandForecaster(random_state=42)
    model.fit(X_train, y_train)

    X_wrong = df[feature_cols[:5]]

    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(X_wrong)


def test_model_get_feature_importance(sample_data_with_features):
    df = sample_data_with_features
    X = df[get_feature_columns()]
    y = df["demand"]

    model = DemandForecaster(n_estimators=10, random_state=42)
    model.fit(X, y)

    importance_df = model.get_feature_importance()

    assert isinstance(importance_df, pd.DataFrame)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(importance_df) == len(get_feature_columns())
    assert importance_df["importance"].sum() > 0


def test_model_save_load(sample_data_with_features, temp_dir):
    df = sample_data_with_features
    X = df[get_feature_columns()]
    y = df["demand"]

    model = DemandForecaster(n_estimators=10, random_state=42)
    model.fit(X, y)

    original_predictions = model.predict(X[:5])

    model_path = temp_dir / "test_model.joblib"
    model.save(model_path)

    assert model_path.exists()

    loaded_model = DemandForecaster.load(model_path)
    loaded_predictions = loaded_model.predict(X[:5])

    np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)