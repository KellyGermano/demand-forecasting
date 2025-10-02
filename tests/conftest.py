import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.forecasting.data import generate_synthetic_demand
from src.forecasting.features import create_all_features


@pytest.fixture
def sample_data():
    np.random.seed(42)
    df = generate_synthetic_demand(periods=100, random_seed=42)
    return df


@pytest.fixture
def sample_data_with_features():
    np.random.seed(42)
    df = generate_synthetic_demand(periods=100, random_seed=42)
    df = create_all_features(df)
    df = df.dropna()
    return df


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_train_test_split(sample_data_with_features):
    df = sample_data_with_features
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df