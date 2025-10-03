from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculates regression metrics from true and predicted values."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero for MAPE calculation
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mask = y_true_arr != 0

    if np.any(mask):
        mape = float(
            np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
        )
    else:
        mape = 0.0

    smape = float(
        np.mean(2 * np.abs(y_pred_arr - y_true_arr) / (np.abs(y_true_arr) + np.abs(y_pred_arr)))
        * 100
    )

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "smape": float(smape)}


def print_metrics(metrics: dict[str, float]) -> None:
    """Prints the evaluation metrics in a formatted way."""
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
    dates: Optional[pd.Series | pd.Index] = None,
    save_path: str | Path | None = None,
) -> None:
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(12, 6))

    if dates is not None and not dates.empty:
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
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path | None = None
) -> None:
    """Plots residual analysis graphs."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual Plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")
    axes[0].grid(True, alpha=0.3)

    # Residual Distribution
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
    y_true_data = np.random.rand(100) * 100
    y_true_data[y_true_data < 1] = 1
    y_pred_data = y_true_data + np.random.randn(100) * 5

    # Calculate and print metrics
    model_metrics = calculate_metrics(y_true_data, y_pred_data)
    print_metrics(model_metrics)

    plot_predictions(y_true_data, y_pred_data, save_path="reports/figures/test_predictions.png")
    plot_residuals(y_true_data, y_pred_data, save_path="reports/figures/test_residuals.png")
