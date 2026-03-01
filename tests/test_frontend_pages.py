import json

import web_app.app as app_module
from src.model_registry import ModelRegistry


def test_home_page_renders_runtime_strip():
    client = app_module.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "View Full Ops Dashboard" in body
    assert "Model" in body
    assert "Firewall" in body


def test_models_page_renders_registry_without_model_results(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(str(results_dir / "model_registry.json"))
    registry.register("rf_registry", "models/rf.pkl", 0.92, 0.91, 3.5)

    monkeypatch.setattr(app_module, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(app_module, "model_registry", registry)

    client = app_module.app.test_client()
    response = client.get("/models")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Recent Registry Entries" in body
    assert "rf_registry" in body


def test_models_page_handles_partial_model_results(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_results_file = results_dir / "model_results_20260101_010101.json"
    model_results_file.write_text(
        json.dumps(
            [
                {
                    "name": "rf_partial",
                    "accuracy": 0.93,
                    "f1_score": 0.91,
                    "training_time": 4.2,
                }
            ]
        ),
        encoding="utf-8",
    )

    registry = ModelRegistry(str(results_dir / "model_registry.json"))
    monkeypatch.setattr(app_module, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(app_module, "model_registry", registry)

    client = app_module.app.test_client()
    response = client.get("/models")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Rf Partial" in body
    assert "93.00%" in body
