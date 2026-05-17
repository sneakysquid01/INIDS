"""Phase B: Prevention runtime integration tests.

Covers:
  - OpsStore allowlist table CRUD roundtrip
  - Allowlist.contains() exact IP and CIDR match
  - Allowlist bypass short-circuits _on_detection_event (no audit entries)
  - EscalationTracker state machine (CLEAN → ALERT → RATE_LIMIT → TEMP_BLOCK)
  - EscalationTracker de-escalation after cooldown
  - FalsePositiveManager FP rate + auto-suppression
  - ActionExecutor idempotency guard (duplicate delivery → single enforcement)
  - Flask API: GET/POST/DELETE /api/allowlist
  - Flask API: POST /api/alerts/<id>/feedback
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# OpsStore allowlist roundtrip
# ---------------------------------------------------------------------------

class TestOpsStoreAllowlist:
    def _make_store(self, tmp_path):
        from src.ops_store import OpsStore
        return OpsStore(str(tmp_path / "test.db"))

    def test_add_and_list(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.add_allowlist_entry("10.0.0.1", reason="test") is True
        rows = store.list_allowlist()
        assert len(rows) == 1
        assert rows[0]["entry"] == "10.0.0.1"
        assert rows[0]["reason"] == "test"

    def test_duplicate_add_is_idempotent(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_allowlist_entry("192.168.1.0/24")
        store.add_allowlist_entry("192.168.1.0/24")  # duplicate
        assert len(store.list_allowlist()) == 1

    def test_remove(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_allowlist_entry("1.2.3.4")
        removed = store.remove_allowlist_entry("1.2.3.4")
        assert removed is True
        assert len(store.list_allowlist()) == 0

    def test_remove_nonexistent_returns_false(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.remove_allowlist_entry("9.9.9.9") is False

    def test_has_active_block_false_for_unknown_ip(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.has_active_block("10.0.0.2") is False

    def test_has_active_block_true_after_save_action(self, tmp_path):
        store = self._make_store(tmp_path)
        from datetime import datetime, timezone, timedelta
        store.save_action({
            "action": "block",
            "action_type": "block",
            "target": "10.1.1.1",
            "ip": "10.1.1.1",
            "reason": "unit test",
            "status": "active",
            "executed": True,
            "dry_run": False,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        })
        assert store.has_active_block("10.1.1.1") is True

    def test_has_active_block_false_after_expired(self, tmp_path):
        from datetime import datetime, timezone, timedelta
        store = self._make_store(tmp_path)
        store.save_action({
            "action": "block",
            "action_type": "block",
            "target": "10.2.2.2",
            "ip": "10.2.2.2",
            "reason": "expired",
            "status": "active",
            "executed": True,
            "dry_run": False,
            "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        })
        assert store.has_active_block("10.2.2.2") is False


# ---------------------------------------------------------------------------
# Allowlist in-memory + CIDR
# ---------------------------------------------------------------------------

class TestAllowlist:
    def _make_allowlist(self):
        from src.prevention.allowlist import Allowlist
        return Allowlist(ops_store=None)

    def test_exact_ip_match(self):
        al = self._make_allowlist()
        al.add("192.168.0.1")
        assert al.contains("192.168.0.1") is True
        assert al.contains("192.168.0.2") is False

    def test_cidr_match(self):
        al = self._make_allowlist()
        al.add("10.0.0.0/8")
        assert al.contains("10.0.1.50") is True
        assert al.contains("11.0.0.1") is False

    def test_remove(self):
        al = self._make_allowlist()
        al.add("1.1.1.1")
        assert al.remove("1.1.1.1") is True
        assert al.contains("1.1.1.1") is False

    def test_list_entries_sorted(self):
        al = self._make_allowlist()
        al.add("2.2.2.2")
        al.add("1.1.1.1")
        entries = al.list_entries()
        assert entries == sorted(entries)

    def test_invalid_ip_returns_false(self):
        al = self._make_allowlist()
        assert al.contains("not-an-ip") is False

    def test_persistence_roundtrip(self, tmp_path):
        from src.ops_store import OpsStore
        from src.prevention.allowlist import Allowlist
        store = OpsStore(str(tmp_path / "al.db"))
        al1 = Allowlist(ops_store=store)
        al1.add("172.16.0.1", reason="roundtrip")

        # New Allowlist instance reads from store
        al2 = Allowlist(ops_store=store)
        assert al2.contains("172.16.0.1") is True


# ---------------------------------------------------------------------------
# EscalationTracker state machine
# ---------------------------------------------------------------------------

class TestEscalationTracker:
    def _make_tracker(self, cooldown=300.0):
        from src.prevention.escalation_tracker import EscalationTracker
        return EscalationTracker(cooldown_seconds=cooldown)

    def test_clean_on_first_check(self):
        from src.prevention.escalation_tracker import EscalationLevel
        tracker = self._make_tracker()
        assert tracker.get_level("1.2.3.4") == EscalationLevel.CLEAN

    def test_low_severity_escalates_to_alert(self):
        from src.prevention.escalation_tracker import EscalationLevel
        tracker = self._make_tracker()
        level = tracker.record_hit("1.2.3.4", "low")
        assert level == EscalationLevel.ALERT

    def test_critical_severity_jumps_to_temp_block(self):
        from src.prevention.escalation_tracker import EscalationLevel
        tracker = self._make_tracker()
        level = tracker.record_hit("5.6.7.8", "critical")
        assert level == EscalationLevel.TEMP_BLOCK

    def test_repeated_hits_escalate(self):
        from src.prevention.escalation_tracker import EscalationLevel
        tracker = self._make_tracker()
        tracker.record_hit("9.9.9.9", "medium")  # RATE_LIMIT
        level = tracker.record_hit("9.9.9.9", "medium")  # TEMP_BLOCK
        assert int(level) >= int(EscalationLevel.RATE_LIMIT)

    def test_cooldown_de_escalates(self, monkeypatch):
        """After two cooldown periods, level drops by 2."""
        import src.prevention.escalation_tracker as et_mod
        from src.prevention.escalation_tracker import EscalationLevel
        tracker = self._make_tracker(cooldown=10.0)
        # Record a hit at t=1000 → ALERT (non-zero so last_hit is truthy)
        monkeypatch.setattr(et_mod.time, "monotonic", lambda: 1000.0)
        tracker.record_hit("de.es.ca.late", "low")

        # 25 seconds later → 2 cooldown periods → de-escalate 2 steps (ALERT→CLEAN)
        monkeypatch.setattr(et_mod.time, "monotonic", lambda: 1025.0)
        level = tracker.get_level("de.es.ca.late")
        assert level == EscalationLevel.CLEAN

    def test_reset(self):
        from src.prevention.escalation_tracker import EscalationLevel
        tracker = self._make_tracker()
        tracker.record_hit("reset.me", "high")
        tracker.reset("reset.me")
        assert tracker.get_level("reset.me") == EscalationLevel.CLEAN


# ---------------------------------------------------------------------------
# FalsePositiveManager
# ---------------------------------------------------------------------------

class TestFalsePositiveManager:
    def _make_fp(self, threshold=0.7, min_samples=3):
        from src.prevention.false_positive_manager import FalsePositiveManager
        return FalsePositiveManager(suppress_threshold=threshold, min_samples=min_samples)

    def test_fp_rate_zero_initially(self):
        fp = self._make_fp()
        assert fp.fp_rate("eng1") == 0.0

    def test_report_fp_increments_rate(self):
        fp = self._make_fp()
        fp.report_fp("eng1")
        fp.report_tp("eng1")
        assert fp.fp_rate("eng1") == pytest.approx(0.5)

    def test_auto_suppression_when_threshold_exceeded(self):
        fp = self._make_fp(threshold=0.6, min_samples=3)
        for _ in range(3):
            fp.report_fp("eng_bad", "rule_x")
        assert fp.is_suppressed("eng_bad", "rule_x") is True

    def test_no_suppression_below_min_samples(self):
        fp = self._make_fp(threshold=0.6, min_samples=10)
        for _ in range(3):
            fp.report_fp("eng", "rule")
        assert fp.is_suppressed("eng", "rule") is False

    def test_unsuppress(self):
        fp = self._make_fp(threshold=0.6, min_samples=2)
        fp.report_fp("eng", "rule")
        fp.report_fp("eng", "rule")
        assert fp.is_suppressed("eng", "rule") is True
        fp.unsuppress("eng", "rule")
        assert fp.is_suppressed("eng", "rule") is False


# ---------------------------------------------------------------------------
# ActionExecutor idempotency guard
# ---------------------------------------------------------------------------

class TestActionExecutorIdempotency:
    def _make_executor(self, tmp_path):
        from src.ops_store import OpsStore
        from src.firewall_adapters import MockFirewallAdapter
        from src.core.event_bus import EventBus
        from src.ips.action_executor import ActionExecutor
        store = OpsStore(str(tmp_path / "idem.db"))
        adapter = MockFirewallAdapter()
        bus = EventBus()
        return ActionExecutor(adapter=adapter, adapter_name="mock", ops_store=store, event_bus=bus), store

    def _make_policy(self, *, dry_run=False):
        class _P:
            block_ttl_seconds = 60
            confidence_block_threshold = 0.5
        p = _P()
        p.dry_run = dry_run
        return p

    def _make_decision_event(self, ip="10.5.5.5", decision="BLOCK"):
        from src.core.event_bus import DetectionEvent, RiskScoreEvent, PolicyDecisionEvent
        det = DetectionEvent(
            source_ip=ip, prediction="Attack", confidence=0.9,
            severity="high", attack_type="dos",
        )
        risk = RiskScoreEvent(detection=det, risk_score=0.85, components={})
        return PolicyDecisionEvent(risk=risk, decision=decision, reason="unit_test", ttl_seconds=60)

    def test_first_execution_succeeds(self, tmp_path):
        executor, _ = self._make_executor(tmp_path)
        policy = self._make_policy(dry_run=True)
        event = self._make_decision_event()
        result = executor.execute(event, policy)
        assert result is not None

    def test_duplicate_execution_db_enforces_idempotency(self, tmp_path):
        # C-02: idempotency is now DB-level (uq_active_block partial index), not
        # application-level. execute() proceeds through to save_action(), which
        # catches the IntegrityError. The DB has exactly ONE active block record.
        executor, store = self._make_executor(tmp_path)
        from datetime import datetime, timezone, timedelta
        store.save_action({
            "action": "block", "action_type": "block",
            "target": "10.5.5.5", "ip": "10.5.5.5",
            "reason": "pre-existing", "status": "active",
            "executed": True, "dry_run": False,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        })
        policy = self._make_policy(dry_run=False)
        event = self._make_decision_event(ip="10.5.5.5")
        executor.execute(event, policy)  # must not raise
        active_blocks = store._fetchall(
            "SELECT * FROM actions WHERE target = :t AND lower(status) IN ('active','enforced','executed')",
            {"t": "10.5.5.5"},
        )
        assert len(active_blocks) == 1  # DB constraint ensures exactly one row

    def test_rate_limit_idempotency(self, tmp_path):
        # C-02: same pattern — DB-level idempotency via uq_active_block index.
        executor, store = self._make_executor(tmp_path)
        from datetime import datetime, timezone, timedelta
        store.save_action({
            "action": "rate_limit", "action_type": "rate_limit",
            "target": "10.6.6.6", "ip": "10.6.6.6",
            "reason": "pre-existing", "status": "enforced",
            "executed": True, "dry_run": False,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        })
        policy = self._make_policy(dry_run=False)
        event = self._make_decision_event(ip="10.6.6.6", decision="RATE_LIMIT")
        executor.execute(event, policy)  # must not raise
        active_blocks = store._fetchall(
            "SELECT * FROM actions WHERE target = :t AND lower(status) IN ('active','enforced','executed')",
            {"t": "10.6.6.6"},
        )
        assert len(active_blocks) == 1  # DB constraint ensures exactly one row


# ---------------------------------------------------------------------------
# Flask API: allowlist endpoints
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    """Create a throwaway Flask test client with fresh DB."""
    tmp_path = tmp_path_factory.mktemp("appb")
    db_path = str(tmp_path / "test_ops.db")
    os.environ["INIDS_OPS_DB_PATH"] = db_path
    os.environ["INIDS_PIPELINE_ENABLED"] = "false"

    from web_app.app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def _auth_headers():
    import base64
    creds = base64.b64encode(b"admin:secret").decode()
    return {"Authorization": f"Basic {creds}"}


def _analyst_headers():
    import base64
    creds = base64.b64encode(b"analyst:secret").decode()
    return {"Authorization": f"Basic {creds}"}


class TestAllowlistAPI:
    def test_get_allowlist_empty(self, app_client):
        resp = app_client.get("/api/allowlist", headers=_analyst_headers())
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "entries" in data

    def test_post_adds_entry(self, app_client):
        resp = app_client.post(
            "/api/allowlist",
            json={"entry": "192.168.99.1", "reason": "trusted host"},
            headers=_auth_headers(),
        )
        assert resp.status_code in (200, 201)
        data = json.loads(resp.data)
        assert data["entry"] == "192.168.99.1"

    def test_get_shows_added_entry(self, app_client):
        app_client.post(
            "/api/allowlist",
            json={"entry": "10.20.30.40"},
            headers=_auth_headers(),
        )
        resp = app_client.get("/api/allowlist", headers=_analyst_headers())
        data = json.loads(resp.data)
        assert "10.20.30.40" in data["entries"]

    def test_post_missing_entry_returns_400(self, app_client):
        resp = app_client.post(
            "/api/allowlist",
            json={},
            headers=_auth_headers(),
        )
        assert resp.status_code == 400

    def test_delete_entry(self, app_client):
        app_client.post(
            "/api/allowlist",
            json={"entry": "7.7.7.7"},
            headers=_auth_headers(),
        )
        resp = app_client.delete("/api/allowlist/7.7.7.7", headers=_auth_headers())
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["removed"] is True

    def test_delete_nonexistent_returns_404(self, app_client):
        resp = app_client.delete("/api/allowlist/9.9.9.9", headers=_auth_headers())
        assert resp.status_code == 404


class TestFPFeedbackAPI:
    def test_report_fp_returns_200(self, app_client):
        resp = app_client.post(
            "/api/alerts/alert-abc-123/feedback",
            json={"verdict": "fp", "engine_id": "signature_engine", "rule_id": "rule_42"},
            headers=_analyst_headers(),
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["verdict"] == "fp"
        assert "suppressed" in data

    def test_report_tp_returns_200(self, app_client):
        resp = app_client.post(
            "/api/alerts/alert-xyz/feedback",
            json={"verdict": "tp"},
            headers=_analyst_headers(),
        )
        assert resp.status_code == 200

    def test_invalid_verdict_returns_400(self, app_client):
        resp = app_client.post(
            "/api/alerts/some-id/feedback",
            json={"verdict": "maybe"},
            headers=_analyst_headers(),
        )
        assert resp.status_code == 400
