import web_app.app as app_module
from src.ops_store import OpsStore

ANALYST_KEY = "test-analyst-key"
ADMIN_KEY = "test-admin-key"
ANALYST_HEADERS = {"X-API-Key": ANALYST_KEY}
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}


class FakeModel:
    def __init__(self, pred: int, proba: list[float]):
        self.pred = pred
        self.proba = proba

    def predict(self, _df):
        return [self.pred]

    def predict_proba(self, _df):
        return [self.proba]


def _setup_app(monkeypatch, tmp_path):
    monkeypatch.setenv("INIDS_ANALYST_API_KEY", ANALYST_KEY)
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", ADMIN_KEY)
    store = OpsStore(str(tmp_path / "ops_test.db"))
    monkeypatch.setattr(app_module, "model", FakeModel(pred=1, proba=[0.05, 0.95]))
    monkeypatch.setattr(app_module, "detection_service", None)
    monkeypatch.setattr(app_module, "all_models", {})
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "ops_store", store)
    app_module.app.ops_store = store
    return app_module.app.test_client()


def test_api_health_and_predict(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    response = client.post(
        "/api/predict",
        json={"features": {"duration": 1, "src_bytes": 10, "dst_bytes": 5}},
        headers=ANALYST_HEADERS,
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "Attack"
    assert payload["alert"] is not None


def test_api_alerts_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    client.post("/api/predict", json={"features": {"duration": 1}}, headers=ANALYST_HEADERS)

    alerts = client.get("/api/alerts?limit=10", headers=ANALYST_HEADERS)
    assert alerts.status_code == 200
    data = alerts.get_json()
    assert data["count"] >= 1


def test_api_policy_actions_and_audit(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    set_policy = client.post(
        "/api/policy",
        json={"mode": "auto_block", "block_ttl_seconds": 120, "confidence_block_threshold": 80},
        headers=ADMIN_HEADERS,
    )
    assert set_policy.status_code == 200
    assert set_policy.get_json()["mode"] == "auto_block"

    predict = client.post(
        "/api/predict",
        json={"features": {"duration": 1}, "source": "2.2.2.2"},
        headers=ANALYST_HEADERS,
    )
    assert predict.status_code == 200
    assert predict.get_json()["prevention_action"] is not None

    actions = client.get("/api/actions?limit=10", headers=ANALYST_HEADERS)
    assert actions.status_code == 200
    assert actions.get_json()["count"] >= 1

    audit = client.get("/api/audit?limit=10", headers=ADMIN_HEADERS)
    assert audit.status_code == 200
    assert audit.get_json()["count"] >= 1


def test_api_requires_key_when_auth_enabled(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    unauthorized = client.get("/api/audit")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/audit", headers=ADMIN_HEADERS)
    assert authorized.status_code == 200


def test_api_metrics_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    client.post("/api/predict", json={"features": {"duration": 1}}, headers=ANALYST_HEADERS)
    metrics = client.get("/api/metrics", headers=ANALYST_HEADERS)
    assert metrics.status_code == 200
    body = metrics.get_data(as_text=True)
    assert "inids_requests_total" in body
    assert "inids_predictions_total" in body


def test_predict_allowlist_bypasses_enforcement_only(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    app_module.prevention_service.set_policy(mode="auto_block", dry_run=False, confidence_block_threshold=80)
    app_module.allowlist.add("3.3.3.3", reason="trusted")
    try:
        predict = client.post(
            "/api/predict",
            json={"features": {"duration": 1}, "source": "3.3.3.3"},
            headers=ANALYST_HEADERS,
        )
        assert predict.status_code == 200
        payload = predict.get_json()
        assert payload["prediction"] == "Attack"
        assert payload["prevention_action"] is None

        actions = client.get("/api/actions?limit=20", headers=ANALYST_HEADERS)
        assert actions.status_code == 200
        assert all(str(a.get("target")) != "3.3.3.3" for a in actions.get_json()["actions"])
    finally:
        app_module.allowlist.remove("3.3.3.3")
        app_module.prevention_service.set_policy(mode="monitor", dry_run=True)
