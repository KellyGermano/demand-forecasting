import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_synthetic_demand(
    start_date: str = "2023-01-01",
    periods: int = 730,
    base_demand: float = 100.0,
    trend_slope: float = 0.05,
    weekly_amplitude: float = 15.0,
    yearly_amplitude: float = 20.0,
    noise_level: float = 5.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)

    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    trend = base_demand + trend_slope * np.arange(periods)

    weekly_seasonality = weekly_amplitude * np.sin(
        2 * np.pi * np.arange(periods) / 7
    )

    yearly_seasonality = yearly_amplitude * np.sin(
        2 * np.pi * np.arange(periods) / 365.25
    )

    noise = np.random.normal(0, noise_level, periods)

    demand = trend + weekly_seasonality + yearly_seasonality + noise
    demand = np.maximum(demand, 0)

    df = pd.DataFrame({"date": dates, "demand": demand})

    return df


def save_data(df: pd.DataFrame, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


if __name__ == "__main__":
    df = generate_synthetic_demand()
    save_data(df, "data/raw/demand_data.csv")
    print(f"Generated {len(df)} days of synthetic demand data")
    print(f"\nSample:\n{df.head()}")
    print(f"\nStats:\n{df['demand'].describe()}")