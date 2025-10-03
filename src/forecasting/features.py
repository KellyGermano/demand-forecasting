import pandas as pd


def create_lag_features(
    df: pd.DataFrame, target_col: str = "demand", lags: list[int] = [1, 7, 30]
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame, target_col: str = "demand", windows: list[int] = [7, 30]
) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window).std()
        )
    return df


def create_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_temporal_features(df)
    return df


def get_feature_columns() -> list[str]:
    return [
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_30",
        "rolling_std_30",
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "is_weekend",
    ]


if __name__ == "__main__":
    from src.forecasting.data import generate_synthetic_demand

    df = generate_synthetic_demand(periods=100)
    df_features = create_all_features(df)

    print("Original shape:", df.shape)
    print("With features shape:", df_features.shape)
    print("\nFeature columns:")
    for col in get_feature_columns():
        print(f"  - {col}")
    print(f"\nNaN count after feature creation:\n{df_features.isna().sum()}")