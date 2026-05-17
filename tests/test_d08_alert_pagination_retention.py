"""D-08: Alert pagination + retention security regression tests.

Verifies:
1. GET /api/alerts supports limit and offset query params.
2. OpsStore.list_alerts() accepts offset parameter.
3. OpsStore.delete_alerts_older_than() deletes only old rows.
4. _run_alert_retention() is a no-op when INIDS_ALERT_RETENTION_DAYS is 0 or unset.
5. POST /api/admin/alert-retention endpoint exists and is admin-only.
6. Retention respects the configured threshold — newer alerts survive.
"""
import os
import tempfile
import uuid
import inspect
from datetime import datetime, timezone, timedelta

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from src.ops_store import OpsStore
    return OpsStore(path), path


def _alert(*, alert_id=None, source_ip="1.2.3.4", attack_type="portscan",
           timestamp=None, severity="high"):
    if alert_id is None:
        alert_id = str(uuid.uuid4())
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "id": alert_id,
        "timestamp": timestamp,
        "severity": severity,
        "prediction": "Attack",
        "confidence": 95.0,
        "profile": "balanced",
        "reason": "model_prediction",
        "source_ip": source_ip,
        "attack_type": attack_type,
        "risk_score": 0.9,
    }


def _ts(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# ---------------------------------------------------------------------------
# D-08-1: OpsStore.list_alerts() offset parameter
# ---------------------------------------------------------------------------

class TestListAlertsOffset:
    def test_offset_zero_returns_all(self):
        store, path = _make_store()
        try:
            for i in range(5):
                store.save_alert(_alert(source_ip=f"10.0.0.{i}"))
            result = store.list_alerts(limit=10, offset=0)
            assert len(result) == 5
        finally:
            os.unlink(path)

    def test_offset_skips_rows(self):
        store, path = _make_store()
        try:
            # Insert with distinct timestamps so ordering is deterministic
            base = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
            for i in range(5):
                ts = (base - timedelta(minutes=i)).isoformat()
                store.save_alert(_alert(source_ip=f"10.0.0.{i}", timestamp=ts))
            page1 = store.list_alerts(limit=3, offset=0)
            page2 = store.list_alerts(limit=3, offset=3)
            assert len(page1) == 3
            assert len(page2) == 2
            ids1 = {a["id"] for a in page1}
            ids2 = {a["id"] for a in page2}
            assert ids1.isdisjoint(ids2), "Pages must not overlap"
        finally:
            os.unlink(path)

    def test_offset_beyond_end_returns_empty(self):
        store, path = _make_store()
        try:
            store.save_alert(_alert())
            result = store.list_alerts(limit=10, offset=100)
            assert result == []
        finally:
            os.unlink(path)

    def test_offset_default_is_zero(self):
        store, path = _make_store()
        try:
            store.save_alert(_alert())
            without_offset = store.list_alerts(limit=10)
            with_offset = store.list_alerts(limit=10, offset=0)
            assert len(without_offset) == len(with_offset)
        finally:
            os.unlink(path)

    def test_list_alerts_signature_has_offset(self):
        from src.ops_store import OpsStore
        sig = inspect.signature(OpsStore.list_alerts)
        assert "offset" in sig.parameters, "list_alerts() must have an 'offset' parameter"


# ---------------------------------------------------------------------------
# D-08-2: OpsStore.delete_alerts_older_than()
# ---------------------------------------------------------------------------

class TestDeleteAlertsOlderThan:
    def test_deletes_old_alerts(self):
        store, path = _make_store()
        try:
            old_ts = _ts(10)
            new_ts = _ts(1)
            store.save_alert(_alert(timestamp=old_ts))
            store.save_alert(_alert(timestamp=new_ts))

            cutoff = _ts(5)
            deleted = store.delete_alerts_older_than(cutoff)
            assert deleted == 1

            remaining = store.list_alerts(limit=100)
            assert len(remaining) == 1
            assert remaining[0]["timestamp"] == new_ts
        finally:
            os.unlink(path)

    def test_deletes_nothing_if_all_new(self):
        store, path = _make_store()
        try:
            store.save_alert(_alert(timestamp=_ts(1)))
            store.save_alert(_alert(timestamp=_ts(2)))
            cutoff = _ts(5)
            deleted = store.delete_alerts_older_than(cutoff)
            assert deleted == 0
            assert len(store.list_alerts(limit=100)) == 2
        finally:
            os.unlink(path)

    def test_deletes_all_if_all_old(self):
        store, path = _make_store()
        try:
            store.save_alert(_alert(timestamp=_ts(30)))
            store.save_alert(_alert(timestamp=_ts(60)))
            cutoff = _ts(5)
            deleted = store.delete_alerts_older_than(cutoff)
            assert deleted == 2
            assert store.list_alerts(limit=100) == []
        finally:
            os.unlink(path)

    def test_returns_int(self):
        store, path = _make_store()
        try:
            result = store.delete_alerts_older_than(_ts(5))
            assert isinstance(result, int)
        finally:
            os.unlink(path)

    def test_method_exists_on_ops_store(self):
        from src.ops_store import OpsStore
        assert callable(getattr(OpsStore, "delete_alerts_older_than", None))


# ---------------------------------------------------------------------------
# D-08-3: _run_alert_retention() no-op when disabled
# ---------------------------------------------------------------------------

class TestRunAlertRetentionNoOp:
    def test_no_op_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("INIDS_ALERT_RETENTION_DAYS", raising=False)
        import web_app.app as app_module
        result = app_module._run_alert_retention()
        assert result == 0

    def test_no_op_when_env_zero(self, monkeypatch):
        monkeypatch.setenv("INIDS_ALERT_RETENTION_DAYS", "0")
        import web_app.app as app_module
        result = app_module._run_alert_retention()
        assert result == 0

    def test_no_op_when_env_empty_string(self, monkeypatch):
        monkeypatch.setenv("INIDS_ALERT_RETENTION_DAYS", "")
        import web_app.app as app_module
        result = app_module._run_alert_retention()
        assert result == 0

    def test_no_op_when_env_negative(self, monkeypatch):
        monkeypatch.setenv("INIDS_ALERT_RETENTION_DAYS", "-5")
        import web_app.app as app_module
        result = app_module._run_alert_retention()
        assert result == 0

    def test_no_op_when_env_invalid(self, monkeypatch):
        monkeypatch.setenv("INIDS_ALERT_RETENTION_DAYS", "notanumber")
        import web_app.app as app_module
        result = app_module._run_alert_retention()
        assert result == 0

    def test_function_exists_in_app_module(self):
        import web_app.app as app_module
        assert callable(getattr(app_module, "_run_alert_retention", None))


# ---------------------------------------------------------------------------
# D-08-4: _run_alert_retention() deletes correctly when enabled
# ---------------------------------------------------------------------------

class TestRunAlertRetentionEnabled:
    def test_deletes_old_alerts_via_app_function(self, monkeypatch):
        monkeypatch.setenv("INIDS_ALERT_RETENTION_DAYS", "5")

        import web_app.app as app_module

        store, path = _make_store()
        original_ops_store = app_module.ops_store
        try:
            app_module.ops_store = store
            store.save_alert(_alert(timestamp=_ts(10)))
            store.save_alert(_alert(timestamp=_ts(1)))

            deleted = app_module._run_alert_retention()
            assert deleted == 1

            remaining = store.list_alerts(limit=100)
            assert len(remaining) == 1
        finally:
            app_module.ops_store = original_ops_store
            os.unlink(path)

    def test_returns_zero_when_nothing_old(self, monkeypatch):
        monkeypatch.setenv("INIDS_ALERT_RETENTION_DAYS", "5")

        import web_app.app as app_module

        store, path = _make_store()
        original_ops_store = app_module.ops_store
        try:
            app_module.ops_store = store
            store.save_alert(_alert(timestamp=_ts(1)))

            deleted = app_module._run_alert_retention()
            assert deleted == 0
        finally:
            app_module.ops_store = original_ops_store
            os.unlink(path)


# ---------------------------------------------------------------------------
# D-08-5: GET /api/alerts pagination response shape
# ---------------------------------------------------------------------------

class TestGetAlertsResponseShape:
    def test_response_includes_offset_and_limit_fields(self):
        import web_app.app as app_module
        client = app_module.app.test_client()

        with app_module.app.test_request_context():
            resp = client.get("/api/alerts?limit=10&offset=0")
            # 200 or auth-gated — check body if 200
            if resp.status_code == 200:
                data = resp.get_json()
                assert "offset" in data, "Response must include 'offset' field"
                assert "limit" in data, "Response must include 'limit' field"
                assert "alerts" in data, "Response must include 'alerts' field"
                assert "count" in data, "Response must include 'count' field"

    def test_offset_param_accepted_without_error(self):
        import web_app.app as app_module
        client = app_module.app.test_client()

        with app_module.app.test_request_context():
            resp = client.get("/api/alerts?limit=5&offset=10")
            assert resp.status_code in (200, 401, 403), (
                f"Unexpected status {resp.status_code} — endpoint must exist"
            )

    def test_source_code_has_offset_in_get_alerts(self):
        import web_app.app as app_module
        source = inspect.getsource(app_module)
        assert "offset" in source, "app.py must reference 'offset' for pagination"


# ---------------------------------------------------------------------------
# D-08-6: POST /api/admin/alert-retention endpoint
# ---------------------------------------------------------------------------

def _retention_endpoint_source():
    """Return the combined source where /api/admin/alert-retention may live (app or blueprint)."""
    import web_app.app as app_module
    import web_app.blueprints.auth as auth_bp_module
    return inspect.getsource(app_module) + inspect.getsource(auth_bp_module)


class TestAdminAlertRetentionEndpoint:
    def test_endpoint_exists_in_source(self):
        source = _retention_endpoint_source()
        assert "/api/admin/alert-retention" in source, (
            "POST /api/admin/alert-retention endpoint must be defined in app.py or auth blueprint"
        )

    def test_endpoint_requires_admin_role(self):
        source = _retention_endpoint_source()
        idx = source.find("/api/admin/alert-retention")
        snippet = source[max(0, idx - 20):idx + 300]
        assert "require_role" in snippet and "admin" in snippet, (
            "alert-retention endpoint must be gated by require_role('admin')"
        )

    def test_endpoint_returns_deleted_count(self):
        source = _retention_endpoint_source()
        idx = source.find("/api/admin/alert-retention")
        snippet = source[idx:idx + 400]
        assert '"deleted"' in snippet or "'deleted'" in snippet, (
            "alert-retention endpoint must return {'deleted': <count>}"
        )

    def test_endpoint_calls_run_alert_retention(self):
        source = _retention_endpoint_source()
        idx = source.find("/api/admin/alert-retention")
        snippet = source[idx:idx + 500]
        assert "_run_alert_retention" in snippet, (
            "alert-retention endpoint must call _run_alert_retention()"
        )

    def test_endpoint_accessible_via_test_client(self):
        import web_app.app as app_module
        client = app_module.app.test_client()
        with app_module.app.test_request_context():
            resp = client.post("/api/admin/alert-retention")
            assert resp.status_code in (200, 401, 403), (
                f"Unexpected status {resp.status_code} — endpoint must be registered"
            )


# ---------------------------------------------------------------------------
# D-08-7: _start_alert_retention_thread wired into _ensure_scheduler_started
# ---------------------------------------------------------------------------

class TestRetentionThreadWired:
    def test_start_alert_retention_thread_in_ensure_scheduler(self):
        import web_app.app as app_module
        src = inspect.getsource(app_module._ensure_scheduler_started)
        assert "_start_alert_retention_thread" in src, (
            "_start_alert_retention_thread() must be called from _ensure_scheduler_started()"
        )

    def test_start_alert_retention_thread_exists(self):
        import web_app.app as app_module
        assert callable(getattr(app_module, "_start_alert_retention_thread", None))
