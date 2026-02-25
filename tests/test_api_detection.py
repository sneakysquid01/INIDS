import web_app.app as app_module
from src.ops_store import OpsStore
import src.auth_service as auth_module


class FakeModel:
    def __init__(self, pred: int, proba: list[float]):
        self.pred = pred
        self.proba = proba

    def predict(self, _df):
        return [self.pred]

    def predict_proba(self, _df):
        return [self.proba]


def _setup_app(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "model", FakeModel(pred=1, proba=[0.05, 0.95]))
    monkeypatch.setattr(app_module, "detection_service", None)
    monkeypatch.setattr(app_module, "all_models", {})
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "ops_store", OpsStore(str(tmp_path / "ops_test.db")))
    return app_module.app.test_client()


def test_api_health_and_predict(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    response = client.post(
        "/api/predict",
        json={"features": {"duration": 1, "src_bytes": 10, "dst_bytes": 5}},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "Attack"
    assert payload["alert"] is not None


def test_api_alerts_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    client.post("/api/predict", json={"features": {"duration": 1}})

    alerts = client.get("/api/alerts?limit=10")
    assert alerts.status_code == 200
    data = alerts.get_json()
    assert data["count"] >= 1


def test_api_policy_actions_and_audit(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    set_policy = client.post(
        "/api/policy",
        json={"mode": "auto_block", "block_ttl_seconds": 120, "confidence_block_threshold": 80},
    )
    assert set_policy.status_code == 200
    assert set_policy.get_json()["mode"] == "auto_block"

    predict = client.post(
        "/api/predict",
        json={"features": {"duration": 1}, "source": "2.2.2.2"},
    )
    assert predict.status_code == 200
    assert predict.get_json()["prevention_action"] is not None

    actions = client.get("/api/actions?limit=10")
    assert actions.status_code == 200
    assert actions.get_json()["count"] >= 1

    audit = client.get("/api/audit?limit=10")
    assert audit.status_code == 200
    assert audit.get_json()["count"] >= 1


def test_api_requires_key_when_auth_enabled(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        auth_module._auth_service,
        "principals",
        {"admin-token": auth_module.Principal(role="admin", token="admin-token")},
    )

    unauthorized = client.get("/api/audit")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/audit", headers={"X-API-Key": "admin-token"})
    assert authorized.status_code == 200

    monkeypatch.setattr(auth_module._auth_service, "principals", {})


def test_api_metrics_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    client.post("/api/predict", json={"features": {"duration": 1}})
    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    body = metrics.get_data(as_text=True)
    assert "inids_requests_total" in body
    assert "inids_predictions_total" in body
