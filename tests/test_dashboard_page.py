import numpy as np

import web_app.app as app_module
from src.model_registry import ModelRegistry
from src.ops_store import OpsStore


class ConstantPredictModel:
    def predict(self, df):
        return np.zeros(len(df), dtype=int)


def _setup_client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "all_models", {})
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "ops_store", OpsStore(str(tmp_path / "ops_test.db")))
    monkeypatch.setattr(app_module, "model_registry", ModelRegistry(str(tmp_path / "model_registry.json")))
    return app_module.app.test_client()


def test_dashboard_renders_without_model(monkeypatch, tmp_path):
    client = _setup_client(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "detection_service", None)

    response = client.get("/dashboard")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Analytics Dashboard" in page
    assert "No trained model found" in page
    assert "Recent Alerts" in page


def test_dashboard_renders_model_analytics(monkeypatch, tmp_path):
    client = _setup_client(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "model", ConstantPredictModel())
    monkeypatch.setattr(app_module, "detection_service", None)

    response = client.get("/dashboard")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Prediction distribution chart" in page
    assert "Accuracy" in page
    assert "Model Registry" in page
