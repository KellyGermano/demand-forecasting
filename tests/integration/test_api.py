from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "model_available" in data


def test_forecast_without_model():
    registry_path = Path("models/registry.json")
    if not registry_path.exists() or not Path("models/production").exists():
        response = client.get("/api/v1/forecast?horizon=7")
        assert response.status_code in [503, 500]


def test_forecast_with_valid_horizon():
    response = client.get("/api/v1/forecast?horizon=7")

    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "horizon" in data
        assert "model_id" in data
        assert "metrics" in data

        assert len(data["predictions"]) == 7
        assert data["horizon"] == 7
        assert all(isinstance(p, (int, float)) for p in data["predictions"])


def test_forecast_with_invalid_horizon():
    response = client.get("/api/v1/forecast?horizon=0")
    assert response.status_code == 422

    response = client.get("/api/v1/forecast?horizon=1000")
    assert response.status_code == 422


def test_forecast_with_default_horizon():
    response = client.get("/api/v1/forecast")

    if response.status_code == 200:
        data = response.json()
        assert data["horizon"] == 7


def test_train_model_endpoint():
    response = client.post(
        "/api/v1/models/train",
        json={
            "test_size": 0.2,
            "n_estimators": 10,
            "max_depth": 10,
            "random_state": 42
        }
    )

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "model_id" in data
        assert "metrics" in data
        assert "mae" in data["metrics"]
        assert "mape" in data["metrics"]


def test_get_current_model():
    response = client.get("/api/v1/models/current")

    if response.status_code == 200:
        data = response.json()
        assert "model_id" in data
        assert "info" in data
    elif response.status_code == 404:
        data = response.json()
        assert "detail" in data


def test_get_metrics():
    response = client.get("/api/v1/metrics")

    if response.status_code == 200:
        data = response.json()
        assert "metrics" in data
        assert "model_id" in data
        assert "model_type" in data
    elif response.status_code == 404:
        data = response.json()
        assert "detail" in data


def test_openapi_docs():
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data


def test_not_found_endpoint():
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404