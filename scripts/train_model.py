import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecasting.train import train_demand_forecasting_model


def main():
    print("Starting model training...")
    try:
        model, metrics, model_id = train_demand_forecasting_model()
        print("\n✓ Training successful!")
        print(f"✓ Model ID: {model_id}")
        print(f"✓ MAE: {metrics['mae']:.2f}")
        print(f"✓ MAPE: {metrics['mape']:.2f}%")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())