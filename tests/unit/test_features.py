import pytest
import pandas as pd
import numpy as np

from src.forecasting.features import (
    create_lag_features,
    create_rolling_features,
    create_temporal_features,
    create_all_features,
    get_feature_columns,
)


def test_create_lag_features(sample_data):
    df = create_lag_features(sample_data, lags=[1, 7])

    assert "lag_1" in df.columns
    assert "lag_7" in df.columns

    assert df["lag_1"].iloc[1] == df["demand"].iloc[0]
    assert df["lag_7"].iloc[7] == df["demand"].iloc[0]

    assert df["lag_1"].iloc[0] != df["lag_1"].iloc[0]


def test_create_rolling_features(sample_data):
    df = create_rolling_features(sample_data, windows=[7])

    assert "rolling_mean_7" in df.columns
    assert "rolling_std_7" in df.columns

    assert df["rolling_mean_7"].iloc[:8].isna().sum() >= 7


def test_create_temporal_features(sample_data):
    df = create_temporal_features(sample_data)

    assert "day_of_week" in df.columns
    assert "month" in df.columns
    assert "is_weekend" in df.columns

    assert df["day_of_week"].min() >= 0
    assert df["day_of_week"].max() <= 6

    assert df["month"].min() >= 1
    assert df["month"].max() <= 12

    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_create_all_features(sample_data):
    df = create_all_features(sample_data)

    expected_columns = get_feature_columns()
    for col in expected_columns:
        assert col in df.columns, f"Missing feature: {col}"

    assert "date" in df.columns
    assert "demand" in df.columns


def test_get_feature_columns():
    feature_cols = get_feature_columns()

    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 0

    assert "lag_1" in feature_cols
    assert "rolling_mean_7" in feature_cols
    assert "day_of_week" in feature_cols
    assert "is_weekend" in feature_cols


def test_features_no_data_leakage(sample_data):
    df = create_all_features(sample_data)

    for lag in [1, 7, 30]:
        col = f"lag_{lag}"
        if col in df.columns:
            assert df[col].iloc[lag] == df["demand"].iloc[0]