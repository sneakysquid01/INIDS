import web_app.app as app_module
from src.ops_store import OpsStore
import src.auth_service as auth_module
from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from src.model_registry import ModelRegistry


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
    monkeypatch.setattr(app_module, "model_registry", ModelRegistry(str(tmp_path / "model_registry.json")))
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


def test_api_ingest_and_process(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    ingest = client.post(
        "/api/ingest",
        json={"source": "replay", "rows": [{"duration": 1, "src_bytes": 10, "dst_bytes": 5}]},
    )
    assert ingest.status_code == 200
    assert ingest.get_json()["queued"] == 1

    processed = client.post("/api/ingest/process", json={"max_items": 10})
    assert processed.status_code == 200
    payload = processed.get_json()
    assert payload["processed"] >= 1
    assert isinstance(payload["results"], list)


def test_api_actions_cleanup(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        auth_module._auth_service,
        "principals",
        {"admin-token": auth_module.Principal(role="admin", token="admin-token")},
    )

    client.post(
        "/api/policy",
        headers={"X-API-Key": "admin-token"},
        json={"mode": "auto_block", "block_ttl_seconds": 1, "confidence_block_threshold": 80},
    )
    client.post("/api/predict", json={"features": {"duration": 1}, "source": "3.3.3.3"})

    cleanup = client.post(
        "/api/actions/cleanup",
        headers={"X-API-Key": "admin-token"},
        json={"now": "2999-01-01T00:00:00+00:00"},
    )
    assert cleanup.status_code == 200
    assert cleanup.get_json()["removed"] >= 1

    monkeypatch.setattr(auth_module._auth_service, "principals", {})


def test_api_explain_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    explain = client.post(
        "/api/explain",
        json={"features": {"duration": 500, "src_bytes": 100}, "top_k": 4},
    )
    assert explain.status_code == 200
    payload = explain.get_json()
    assert payload["top_k"] == 4
    assert len(payload["explanation"]) == 4


def test_api_rate_limit(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "rate_limiter", InMemoryRateLimiter(RateLimitConfig(requests=1, window_seconds=60)))

    first = client.post("/api/predict", json={"features": {"duration": 1}})
    assert first.status_code == 200

    second = client.post("/api/predict", json={"features": {"duration": 1}})
    assert second.status_code == 429
    assert second.get_json()["error"] == "rate_limited"


def test_api_ingest_log_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)

    zeek = client.post(
        "/api/ingest/log",
        json={
            "type": "zeek",
            "records": [{"proto": "tcp", "duration": 1.1, "orig_bytes": 10, "resp_bytes": 5}],
        },
    )
    assert zeek.status_code == 200
    assert zeek.get_json()["queued"] == 1

    suri = client.post(
        "/api/ingest/log",
        json={
            "type": "suricata",
            "records": [{"proto": "TCP", "flow": {"age": 3, "bytes_toserver": 8, "bytes_toclient": 4}}],
        },
    )
    assert suri.status_code == 200
    assert suri.get_json()["type"] == "suricata"


def test_api_model_registry_endpoint(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    app_module.model_registry.register("rf", "models/rf.pkl", 0.99, 0.98, 10.0)

    out = client.get("/api/models/registry?limit=5")
    assert out.status_code == 200
    payload = out.get_json()
    assert payload["count"] >= 1
    assert payload["models"][0]["name"] == "rf"


def test_api_explain_rejects_invalid_top_k(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    response = client.post(
        "/api/explain",
        json={"features": {"duration": 1}, "top_k": "invalid"},
    )
    assert response.status_code == 400
    assert "top_k" in response.get_json()["error"]


def test_api_ingest_process_rejects_invalid_max_items(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    response = client.post("/api/ingest/process", json={"max_items": "not-a-number"})
    assert response.status_code == 400
    assert "max_items" in response.get_json()["error"]


def test_api_not_found_returns_json(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_api_actions_cleanup_rejects_invalid_now(monkeypatch, tmp_path):
    client = _setup_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        auth_module._auth_service,
        "principals",
        {"admin-token": auth_module.Principal(role="admin", token="admin-token")},
    )
    response = client.post(
        "/api/actions/cleanup",
        headers={"X-API-Key": "admin-token"},
        json={"now": "not-a-timestamp"},
    )
    assert response.status_code == 400
    assert "now" in response.get_json()["error"]
    monkeypatch.setattr(auth_module._auth_service, "principals", {})
