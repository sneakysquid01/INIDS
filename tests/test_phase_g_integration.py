from __future__ import annotations

from datetime import datetime, timezone


def _make_detection_event(ip: str = "10.0.0.1"):
    from src.core.event_bus import DetectionEvent

    return DetectionEvent(
        source_ip=ip,
        prediction="Attack",
        confidence=95.0,
        features={"src_bytes": 10, "dst_bytes": 2},
        attack_type="test",
        profile="balanced",
        severity="high",
        suspicious=False,
        reason="unit",
    )


def _make_risk_event(ip: str = "10.0.0.1"):
    from src.core.event_bus import RiskScoreEvent

    det = _make_detection_event(ip)
    return RiskScoreEvent(detection=det, risk_score=88.0, components={"confidence": 88.0})


def _make_policy_event(ip: str = "10.0.0.1"):
    from src.core.event_bus import PolicyDecisionEvent

    risk = _make_risk_event(ip)
    return PolicyDecisionEvent(
        risk=risk,
        decision="BLOCK",
        reason="phase_g",
        ttl_seconds=120,
    )


def _make_action_event(ip: str = "10.0.0.1"):
    from src.core.event_bus import ActionEvent

    decision = _make_policy_event(ip)
    return ActionEvent(
        decision=decision,
        action="block",
        target=ip,
        reason="phase_g",
        dry_run=True,
        executed=False,
        status="DRY_RUN",
        adapter="mock",
        expires_at=datetime.now(timezone.utc).isoformat(),
    )


def test_event_bus_detection_emits_to_siem(monkeypatch):
    import web_app.app as app_module
    from src.observability.siem_exporter import SiemExporter

    exporter = SiemExporter()
    monkeypatch.setattr(app_module, "siem_exporter", exporter)

    app_module.event_bus.publish(_make_detection_event("10.1.1.1"))
    assert exporter.pending() >= 1


def test_event_bus_risk_emits_to_siem(monkeypatch):
    import web_app.app as app_module
    from src.observability.siem_exporter import SiemExporter

    exporter = SiemExporter()
    monkeypatch.setattr(app_module, "siem_exporter", exporter)

    app_module.event_bus.publish(_make_risk_event("10.2.2.2"))
    assert exporter.pending() >= 1


def test_event_bus_policy_emits_to_siem(monkeypatch):
    import web_app.app as app_module
    from src.observability.siem_exporter import SiemExporter

    exporter = SiemExporter()
    monkeypatch.setattr(app_module, "siem_exporter", exporter)

    app_module.event_bus.publish(_make_policy_event("10.3.3.3"))
    assert exporter.pending() >= 1


def test_event_bus_action_emits_to_siem(monkeypatch):
    import web_app.app as app_module
    from src.observability.siem_exporter import SiemExporter

    exporter = SiemExporter()
    monkeypatch.setattr(app_module, "siem_exporter", exporter)

    app_module.event_bus.publish(_make_action_event("10.4.4.4"))
    assert exporter.pending() >= 1


def test_detection_and_action_event_counters_increment(monkeypatch):
    import web_app.app as app_module

    before_detection = app_module.metrics_service.get("detection_events_total")
    before_action = app_module.metrics_service.get("action_events_total")

    app_module.event_bus.publish(_make_detection_event("10.5.5.5"))
    app_module.event_bus.publish(_make_action_event("10.5.5.5"))

    after_detection = app_module.metrics_service.get("detection_events_total")
    after_action = app_module.metrics_service.get("action_events_total")

    assert after_detection >= before_detection + 1
    assert after_action >= before_action + 1


def test_api_policy_rollback_version_not_found(monkeypatch, tmp_path):
    import web_app.app as app_module
    from src.ops_store import OpsStore
    from src.policy.policy_store import PolicyStore

    monkeypatch.setattr(app_module, "ops_store", OpsStore(str(tmp_path / "ops_g.db")))
    monkeypatch.setattr(
        app_module,
        "policy_store",
        PolicyStore(initial_config=app_module.prevention_service.policy.to_dict()),
    )

    client = app_module.app.test_client()
    resp = client.post("/api/policy/rollback", json={"to_version": 9999})
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "version_not_found"


def test_api_policy_history_limit(monkeypatch, tmp_path):
    import web_app.app as app_module
    from src.ops_store import OpsStore
    from src.policy.policy_store import PolicyStore

    monkeypatch.setattr(app_module, "ops_store", OpsStore(str(tmp_path / "ops_hist.db")))
    store = PolicyStore(initial_config=app_module.prevention_service.policy.to_dict())
    monkeypatch.setattr(app_module, "policy_store", store)

    # Create several versions
    for idx in range(4):
        store.update(
            {
                "mode": "auto_block" if idx % 2 else "monitor",
                "block_ttl_seconds": 120 + idx,
                "confidence_block_threshold": 80.0,
                "dry_run": True,
            },
            changed_by="phase_g",
            reason=f"u{idx}",
        )

    client = app_module.app.test_client()
    resp = client.get("/api/policy/history?limit=2")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 2
    assert len(data["history"]) == 2


def test_prometheus_dynamic_engine_metrics_lines_present(monkeypatch):
    import web_app.app as app_module

    # Simulate per-engine counter activity
    app_module.metrics_service.inc("engine_custom_evaluations_total")
    app_module.metrics_service.inc("engine_custom_attacks_total")

    metrics = app_module.metrics_service.as_prometheus()
    assert "inids_engine_custom_evaluations_total" in metrics
    assert "inids_engine_custom_attacks_total" in metrics
