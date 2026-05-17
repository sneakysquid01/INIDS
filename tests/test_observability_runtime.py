from __future__ import annotations

import logging


def test_json_formatter_output_contains_core_fields():
    from src.observability.json_logging import JSONFormatter

    formatter = JSONFormatter(service_name="inids-test")
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert '"service":"inids-test"' in output
    assert '"message":"hello"' in output
    assert '"level":"INFO"' in output


def test_siem_exporter_emit_and_flush_jsonl():
    from src.observability.siem_exporter import SiemExporter

    exporter = SiemExporter(max_buffer=10)
    exporter.emit({"event": "detection", "id": 1})
    exporter.emit({"event": "action", "id": 2})

    text = exporter.flush_jsonl(max_items=10)
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 2
    assert exporter.pending() == 0


def test_policy_store_update_history_and_rollback():
    from src.policy.policy_store import PolicyStore

    store = PolicyStore(initial_config={"mode": "monitor", "block_ttl_seconds": 300, "confidence_block_threshold": 85.0, "dry_run": True})
    v2 = store.update({"mode": "auto_block", "block_ttl_seconds": 120, "confidence_block_threshold": 80.0, "dry_run": False}, changed_by="tester", reason="tighten")
    assert v2.version == 2

    v3 = store.rollback(1, changed_by="tester")
    assert v3 is not None
    assert v3.version == 3
    assert v3.config["mode"] == "monitor"

    history = store.history(limit=10)
    assert len(history) == 3
    # Latest first: rollback entry should be first and initial entry last.
    assert history[0]["version"] == 3
    assert history[-1]["version"] == 1


_OBS_ADMIN_KEY = "obs-test-admin-key"
_OBS_ANALYST_KEY = "obs-test-analyst-key"
_ADMIN_H = {"X-API-Key": _OBS_ADMIN_KEY}
_ANALYST_H = {"X-API-Key": _OBS_ANALYST_KEY}


def _seed_store(monkeypatch, tmp_path, app_module):
    from src.ops_store import OpsStore
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", _OBS_ADMIN_KEY)
    monkeypatch.setenv("INIDS_ANALYST_API_KEY", _OBS_ANALYST_KEY)
    store = OpsStore(str(tmp_path / "ops_e.db"))
    monkeypatch.setattr(app_module, "ops_store", store)
    app_module.app.ops_store = store
    return store


def test_api_policy_history_and_rollback(monkeypatch, tmp_path):
    import web_app.app as app_module
    from src.policy.policy_store import PolicyStore

    _seed_store(monkeypatch, tmp_path, app_module)
    app_module.prevention_service.set_policy(mode="monitor", block_ttl_seconds=300, confidence_block_threshold=85.0, dry_run=True)
    monkeypatch.setattr(app_module, "policy_store", PolicyStore(initial_config=app_module.prevention_service.policy.to_dict()))

    client = app_module.app.test_client()

    resp1 = client.post(
        "/api/policy",
        json={
            "mode": "auto_block",
            "block_ttl_seconds": 120,
            "confidence_block_threshold": 80,
            "dry_run": False,
            "reason": "phase-e-test",
        },
        headers=_ADMIN_H,
    )
    assert resp1.status_code == 200

    hist = client.get("/api/policy/history?limit=10", headers=_ADMIN_H)
    assert hist.status_code == 200
    hdata = hist.get_json()
    assert hdata["count"] >= 2

    rollback = client.post("/api/policy/rollback", json={"to_version": 1}, headers=_ADMIN_H)
    assert rollback.status_code == 200
    rdata = rollback.get_json()
    assert rdata["policy"]["mode"] == "monitor"


def test_api_siem_flush(monkeypatch, tmp_path):
    import web_app.app as app_module
    from src.observability.siem_exporter import SiemExporter

    _seed_store(monkeypatch, tmp_path, app_module)
    monkeypatch.setattr(app_module, "siem_exporter", SiemExporter())
    app_module.siem_exporter.emit({"event": "detection", "id": "abc"})
    app_module.siem_exporter.emit({"event": "action", "id": "def"})

    client = app_module.app.test_client()
    resp = client.get("/api/siem/flush?limit=10", headers=_ADMIN_H)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 2
    assert "jsonl" in data


def test_api_detect_updates_engine_metrics(monkeypatch, tmp_path):
    import web_app.app as app_module
    from src.detection.engine_base import EngineResult

    _seed_store(monkeypatch, tmp_path, app_module)

    class Agg:
        verdict = "attack"

        @staticmethod
        def to_dict():
            return {"verdict": "attack", "engine_results": []}

    def fake_eval(_features):
        return [
            EngineResult(
                engine_id="signature",
                engine_type="signature",
                verdict="attack",
                confidence=95.0,
                severity="high",
                attack_type="rule_match",
            )
        ]

    monkeypatch.setattr(app_module.engine_registry, "evaluate_all", fake_eval)
    monkeypatch.setattr(app_module.engine_aggregator, "aggregate", lambda _r: Agg())

    client = app_module.app.test_client()
    resp = client.post("/api/detect", json={"source": "10.0.0.1", "features": {"src_bytes": 10, "dst_bytes": 2}}, headers=_ANALYST_H)
    assert resp.status_code == 200

    metrics = client.get("/api/metrics", headers=_ANALYST_H)
    body = metrics.get_data(as_text=True)
    assert "inids_engine_evaluations_total" in body
    assert "inids_engine_attacks_total" in body
    assert "inids_engine_signature_evaluations_total" in body
    assert "inids_engine_signature_attacks_total" in body
