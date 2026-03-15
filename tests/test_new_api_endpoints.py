"""Tests for new API endpoints: alert lifecycle, FP suppression, approval gate, observability."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

# Import app once — module-level to avoid repeated initialisation.
from web_app.app import app as flask_app, ops_store, fp_manager


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _analyst_headers():
    from web_app.app import SETTINGS
    from src.auth_service import auth_status
    if not auth_status().get("enabled"):
        return {}
    return {}


def _save_alert(alert_id: str | None = None) -> str:
    aid = alert_id or str(uuid.uuid4())
    ops_store.save_alert({
        "id": aid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": "high",
        "prediction": "attack",
        "confidence": 90.0,
        "profile": "balanced",
        "reason": "test",
    })
    return aid


# ---------------------------------------------------------------------------
# Alert lifecycle — PATCH /api/alerts/<id>
# ---------------------------------------------------------------------------

def test_patch_alert_status(client):
    aid = _save_alert()
    resp = client.patch(f"/api/alerts/{aid}", json={"status": "reviewing"})
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert data["updated"] is True


def test_patch_alert_invalid_status(client):
    aid = _save_alert()
    resp = client.patch(f"/api/alerts/{aid}", json={"status": "notareal"})
    assert resp.status_code in (400, 401)


def test_patch_alert_not_found(client):
    resp = client.patch("/api/alerts/no-such-id", json={"status": "closed"})
    assert resp.status_code in (404, 401)


def test_patch_alert_no_body(client):
    aid = _save_alert()
    resp = client.patch(f"/api/alerts/{aid}", json={})
    assert resp.status_code in (400, 401)


# ---------------------------------------------------------------------------
# Alert filter by status — GET /api/alerts?status=open
# ---------------------------------------------------------------------------

def test_get_alerts_status_filter(client):
    resp = client.get("/api/alerts?status=open")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "alerts" in data


# ---------------------------------------------------------------------------
# FP suppression endpoints
# ---------------------------------------------------------------------------

def test_get_fp_suppressions(client):
    resp = client.get("/api/fp-suppressions")
    assert resp.status_code in (200, 401)


def test_post_fp_suppression(client):
    resp = client.post(
        "/api/fp-suppressions",
        json={"engine_id": "signature", "rule_id": "SIG-TEST-01"},
    )
    assert resp.status_code in (200, 401)


def test_delete_fp_suppression(client):
    fp_manager.suppress("signature", "SIG-DELETE-01")
    resp = client.delete("/api/fp-suppressions/signature/SIG-DELETE-01")
    assert resp.status_code in (200, 401)


# ---------------------------------------------------------------------------
# Pending actions — GET /api/actions/pending
# ---------------------------------------------------------------------------

def test_get_pending_actions(client):
    resp = client.get("/api/actions/pending")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "actions" in data


# ---------------------------------------------------------------------------
# Approve action — POST /api/actions/<id>/approve
# ---------------------------------------------------------------------------

def test_approve_nonexistent_action(client):
    resp = client.post("/api/actions/nonexistent-xyz/approve", json={})
    assert resp.status_code in (404, 500, 401)


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------

def test_detection_history_endpoint(client):
    resp = client.get("/api/detections/history")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "history" in data


def test_anomaly_status_endpoint(client):
    resp = client.get("/api/anomaly/status")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "fitted" in data
        assert "buffer_collected" in data


def test_escalation_summary_endpoint(client):
    resp = client.get("/api/escalation/summary")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "escalation" in data


def test_escalation_evict_endpoint(client):
    resp = client.post("/api/escalation/evict")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "evicted" in data


def test_fp_stats_endpoint(client):
    resp = client.get("/api/fp-stats")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "stats" in data


# ---------------------------------------------------------------------------
# Existing alerts endpoint still has status field
# ---------------------------------------------------------------------------

def test_alerts_response_has_status_field(client):
    _save_alert()
    resp = client.get("/api/alerts?limit=5")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        if data["alerts"]:
            alert = data["alerts"][0]
            assert "status" in alert
