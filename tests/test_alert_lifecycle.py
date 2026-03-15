"""Tests for alert lifecycle management (status, assignee, close_reason)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.ops_store import OpsStore


@pytest.fixture()
def store(tmp_path):
    return OpsStore(str(tmp_path / "alerts_test.db"))


def _make_alert(store: OpsStore) -> str:
    alert_id = str(uuid.uuid4())
    store.save_alert({
        "id": alert_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": "high",
        "prediction": "attack",
        "confidence": 92.5,
        "profile": "balanced",
        "reason": "test",
    })
    return alert_id


def test_new_alerts_have_open_status(store):
    alert_id = _make_alert(store)
    alerts = store.list_alerts(limit=10)
    found = next((a for a in alerts if a["id"] == alert_id), None)
    assert found is not None
    # Default status is 'open' (or NULL treated as open)
    assert found.get("status", "open") in ("open", None, "")


def test_update_alert_status(store):
    alert_id = _make_alert(store)
    updated = store.update_alert(alert_id, status="reviewing")
    assert updated is True
    alerts = store.list_alerts(limit=50)
    found = next((a for a in alerts if a["id"] == alert_id), None)
    assert found["status"] == "reviewing"


def test_update_alert_assignee_and_close_reason(store):
    alert_id = _make_alert(store)
    store.update_alert(alert_id, assignee="alice", close_reason="known false positive")
    alerts = store.list_alerts(limit=50)
    found = next((a for a in alerts if a["id"] == alert_id), None)
    assert found["assignee"] == "alice"
    assert found["close_reason"] == "known false positive"


def test_update_alert_status_transitions(store):
    alert_id = _make_alert(store)
    for status in ("reviewing", "escalated", "closed"):
        result = store.update_alert(alert_id, status=status)
        assert result is True
    alerts = store.list_alerts(limit=50)
    found = next((a for a in alerts if a["id"] == alert_id), None)
    assert found["status"] == "closed"


def test_update_alert_invalid_status(store):
    alert_id = _make_alert(store)
    with pytest.raises(ValueError):
        store.update_alert(alert_id, status="hacked")


def test_filter_alerts_by_status(store):
    id_open = _make_alert(store)
    id_closed = _make_alert(store)
    store.update_alert(id_closed, status="closed")

    open_alerts = store.list_alerts(limit=50, status="open")
    closed_alerts = store.list_alerts(limit=50, status="closed")

    open_ids = {a["id"] for a in open_alerts}
    closed_ids = {a["id"] for a in closed_alerts}

    assert id_open in open_ids
    assert id_closed not in open_ids
    assert id_closed in closed_ids


def test_update_nonexistent_alert_returns_false(store):
    result = store.update_alert("nonexistent-id-xyz", status="closed")
    assert result is False


def test_update_alert_no_fields_returns_false(store):
    alert_id = _make_alert(store)
    result = store.update_alert(alert_id)
    assert result is False
