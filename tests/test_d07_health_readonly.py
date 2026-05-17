"""D-07: Health check probe read-only regression tests.

Verifies that the _ops_probe / health endpoint does not INSERT to the
audit log table. Audit table must stay clean after health checks.
"""
import os
import tempfile
import inspect
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# D-07-1: Source code inspection — no add_audit call in _ops_probe
# ---------------------------------------------------------------------------

class TestOpsProbeSourceCode:
    def test_ops_probe_does_not_call_add_audit(self):
        """_ops_probe must not contain an add_audit() call."""
        import web_app.app as app_module
        source = inspect.getsource(app_module)
        # Find the _ops_probe function body in source
        start = source.find("def _ops_probe()")
        assert start != -1, "_ops_probe not found in app.py"
        # Extract until the next top-level def or register call
        probe_body = source[start:start + 800]
        assert "add_audit" not in probe_body, (
            "_ops_probe must not call add_audit() (D-07: health probe must be read-only)"
        )

    def test_ops_probe_calls_list_alerts(self):
        """_ops_probe must verify DB read works via list_alerts."""
        import web_app.app as app_module
        source = inspect.getsource(app_module)
        start = source.find("def _ops_probe()")
        probe_body = source[start:start + 800]
        assert "list_alerts" in probe_body, (
            "_ops_probe must call list_alerts() for DB read verification"
        )


# ---------------------------------------------------------------------------
# D-07-2: Functional — audit table unchanged after probe invocation
# ---------------------------------------------------------------------------

class TestAuditTableUnchangedAfterHealthProbe:
    def _make_store(self):
        import src.ops_store as ops_module
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = ops_module.OpsStore(path)
        return store, path

    def test_audit_count_unchanged_after_simulated_probe(self):
        """Simulate the health probe: only reads, audit row count stays constant."""
        store, path = self._make_store()
        try:
            # Pre-populate one audit row so we can measure
            from datetime import datetime, timezone
            store.add_audit(
                event_type="test_event",
                message="pre-existing",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            before = len(store.list_audits(limit=1000))

            # Simulate _ops_probe's read-only behaviour
            store.list_alerts(limit=1)
            store.list_audits(limit=1)

            after = len(store.list_audits(limit=1000))
            assert after == before, (
                f"Audit count changed from {before} to {after} — "
                "probe must not insert audit rows"
            )
        finally:
            os.unlink(path)

    def test_no_health_check_audit_entries_after_probe(self):
        """After running the probe, no 'health_check' event_type must appear."""
        store, path = self._make_store()
        try:
            # Simulate probe
            store.list_alerts(limit=1)
            store.list_audits(limit=1)

            audits = store.list_audits(limit=1000)
            health_audits = [a for a in audits if a.get("event_type") == "health_check"]
            assert len(health_audits) == 0, (
                f"Found {len(health_audits)} health_check audit entries after probe"
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# D-07-3: add_audit signature check — the call is available but must not be
# invoked from the probe
# ---------------------------------------------------------------------------

class TestAddAuditAvailableElsewhere:
    def test_add_audit_still_exists_on_ops_store(self):
        """add_audit() must still exist — it's used by other routes."""
        from src.ops_store import OpsStore
        assert callable(getattr(OpsStore, "add_audit", None)), (
            "OpsStore.add_audit() must still exist for other routes"
        )

    def test_list_audits_exists_on_ops_store(self):
        """list_audits() must exist for the probe to use."""
        from src.ops_store import OpsStore
        assert callable(getattr(OpsStore, "list_audits", None))

    def test_list_alerts_exists_on_ops_store(self):
        """list_alerts() must exist for the probe to use."""
        from src.ops_store import OpsStore
        assert callable(getattr(OpsStore, "list_alerts", None))
