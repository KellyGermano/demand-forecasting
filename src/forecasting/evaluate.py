from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    smape = (
        np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    )

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "smape": float(smape)}


def print_metrics(metrics: dict[str, float]) -> None:
    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"MAE (Mean Absolute Error):      {metrics['mae']:.2f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.2f}")
    print(f"MAPE (Mean Abs % Error):        {metrics['mape']:.2f}%")
    print(f"sMAPE (Symmetric MAPE):         {metrics['smape']:.2f}%")
    print("=" * 50 + "\n")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series = None,
    save_path: str | Path = None,
) -> None:
    plt.figure(figsize=(12, 6))

    if dates is not None:
        plt.plot(dates, y_true, label="Actual", color="blue", linewidth=2)
        plt.plot(dates, y_pred, label="Predicted", color="red", linewidth=2, alpha=0.7)
        plt.xlabel("Date")
    else:
        plt.plot(y_true, label="Actual", color="blue", linewidth=2)
        plt.plot(y_pred, label="Predicted", color="red", linewidth=2, alpha=0.7)
        plt.xlabel("Time Step")

    plt.ylabel("Demand")
    plt.title("Actual vs Predicted Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path = None
) -> None:
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, edgecolor="black")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Residuals plot saved to {save_path}")

    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.rand(100) * 100
    y_pred = y_true + np.random.randn(100) * 5

    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)

    plot_predictions(y_true, y_pred, save_path="reports/figures/test_predictions.png")
    plot_residuals(y_true, y_pred, save_path="reports/figures/test_residuals.png")