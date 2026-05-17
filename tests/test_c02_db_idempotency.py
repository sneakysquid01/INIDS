"""C-02 security regression tests — DB-Level Idempotency for Prevention (Step 20).

Validation checkpoints per PLAN.md C-02 spec:
1. SQLite version ≥ 3.8.9 (partial index support)
2. Duplicate active block INSERT returns existing record, no exception raised to caller
3. uq_active_block index exists in sqlite_master

Additional coverage:
- Idempotency gate applies only to active/enforced/executed block/temp_block/rate_limit rows
- Non-active-block duplicates are NOT constrained (can insert same target with different status)
- save_action() return type: None on success, dict on duplicate
- execute() no longer calls has_active_block() (pre-check removed)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ops_store import OpsStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_store(tmp_path: Path) -> OpsStore:
    return OpsStore(str(tmp_path / "test.db"))


def _insert_block(store: OpsStore, target: str, status: str = "active", action_type: str = "block") -> dict | None:
    return store.save_action({
        "action": action_type,
        "action_type": action_type,
        "target": target,
        "ip": target,
        "reason": "test",
        "status": status,
        "dry_run": False,
        "executed": True,
    })


# ---------------------------------------------------------------------------
# Checkpoint 1: SQLite version
# ---------------------------------------------------------------------------

class TestSQLiteVersion:
    def test_sqlite_version_at_least_3_8_9(self):
        """Partial index support requires SQLite ≥ 3.8.9."""
        parts = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
        assert parts >= (3, 8, 9), (
            f"SQLite {sqlite3.sqlite_version} does not support partial indexes. "
            "Minimum required: 3.8.9"
        )


# ---------------------------------------------------------------------------
# Checkpoint 2 + 3: index exists, duplicate returns existing record
# ---------------------------------------------------------------------------

class TestUqActiveBlockIndex:
    def test_index_exists_in_sqlite_master(self, tmp_path):
        """Checkpoint 3: uq_active_block index must exist after migration v4."""
        _fresh_store(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='uq_active_block'"
        ).fetchone()
        conn.close()
        assert row is not None, "uq_active_block index not found in sqlite_master"

    def test_schema_version_is_4(self, tmp_path):
        store = _fresh_store(tmp_path)
        assert store.SCHEMA_VERSION >= 4

    def test_first_active_block_insert_returns_none(self, tmp_path):
        store = _fresh_store(tmp_path)
        result = _insert_block(store, "10.0.0.1")
        assert result is None

    def test_duplicate_active_block_returns_existing_record(self, tmp_path):
        """Checkpoint 2: second INSERT for same target/status returns existing record."""
        store = _fresh_store(tmp_path)
        _insert_block(store, "10.0.0.2")
        existing = _insert_block(store, "10.0.0.2")
        assert existing is not None
        assert existing["target"] == "10.0.0.2"

    def test_duplicate_does_not_raise_to_caller(self, tmp_path):
        """No IntegrityError must propagate from save_action()."""
        store = _fresh_store(tmp_path)
        _insert_block(store, "10.0.0.3")
        try:
            _insert_block(store, "10.0.0.3")
        except Exception as exc:
            pytest.fail(f"save_action() raised unexpectedly: {exc}")

    def test_duplicate_temp_block_returns_existing_record(self, tmp_path):
        store = _fresh_store(tmp_path)
        _insert_block(store, "10.0.0.4", action_type="temp_block")
        existing = _insert_block(store, "10.0.0.4", action_type="temp_block")
        assert existing is not None

    def test_duplicate_rate_limit_returns_existing_record(self, tmp_path):
        store = _fresh_store(tmp_path)
        _insert_block(store, "10.0.0.5", action_type="rate_limit")
        existing = _insert_block(store, "10.0.0.5", action_type="rate_limit")
        assert existing is not None


# ---------------------------------------------------------------------------
# Constraint scope: only active/enforced/executed block rows
# ---------------------------------------------------------------------------

class TestConstraintScope:
    def test_different_targets_can_both_be_active(self, tmp_path):
        store = _fresh_store(tmp_path)
        assert _insert_block(store, "10.1.0.1") is None
        assert _insert_block(store, "10.1.0.2") is None

    def test_non_block_action_type_not_constrained(self, tmp_path):
        """Non-block action types (e.g. 'alert') bypass the unique constraint."""
        store = _fresh_store(tmp_path)
        result1 = store.save_action({
            "action": "alert",
            "action_type": "alert",
            "target": "10.2.0.1",
            "ip": "10.2.0.1",
            "reason": "test",
            "status": "active",
        })
        result2 = store.save_action({
            "action": "alert",
            "action_type": "alert",
            "target": "10.2.0.1",
            "ip": "10.2.0.1",
            "reason": "test2",
            "status": "active",
        })
        assert result1 is None
        assert result2 is None

    def test_expired_status_not_constrained(self, tmp_path):
        """An 'expired' block can be re-inserted as new active block."""
        store = _fresh_store(tmp_path)
        store.save_action({
            "action": "block",
            "action_type": "block",
            "target": "10.3.0.1",
            "ip": "10.3.0.1",
            "reason": "old",
            "status": "expired",
        })
        result = _insert_block(store, "10.3.0.1")
        assert result is None

    def test_enforced_status_covered_by_constraint(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.save_action({
            "action": "block",
            "action_type": "block",
            "target": "10.4.0.1",
            "ip": "10.4.0.1",
            "reason": "r1",
            "status": "enforced",
        })
        existing = _insert_block(store, "10.4.0.1")
        assert existing is not None

    def test_executed_status_covered_by_constraint(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.save_action({
            "action": "block",
            "action_type": "block",
            "target": "10.5.0.1",
            "ip": "10.5.0.1",
            "reason": "r1",
            "status": "executed",
        })
        existing = _insert_block(store, "10.5.0.1")
        assert existing is not None


# ---------------------------------------------------------------------------
# execute() no longer calls has_active_block()
# ---------------------------------------------------------------------------

class TestExecuteNoPreCheck:
    """Verify the read-check-write pattern has been removed from execute()."""

    def _make_executor(self, store):
        from src.ips.action_executor import ActionExecutor
        adapter = MagicMock()
        adapter.block.return_value = (True, "blocked")
        adapter.unblock.return_value = (True, "unblocked")
        return ActionExecutor(
            adapter=adapter,
            adapter_name="mock",
            ops_store=store,
        )

    def _make_decision(self, target: str, decision: str = "BLOCK"):
        detection = MagicMock()
        detection.source = target
        risk = MagicMock()
        risk.detection = detection
        event = MagicMock()
        event.decision = decision
        event.risk = risk
        event.reason = "test"
        event.ttl_seconds = 60
        return event

    def test_execute_does_not_call_has_active_block(self, tmp_path):
        """has_active_block() must not be called in execute() (removed in C-02)."""
        store = _fresh_store(tmp_path)
        store.has_active_block = MagicMock(return_value=False)

        executor = self._make_executor(store)
        decision = self._make_decision("10.10.0.1")
        policy = MagicMock()
        policy.dry_run = True

        executor.execute(decision, policy)
        store.has_active_block.assert_not_called()

    def test_execute_second_call_for_same_target_does_not_raise(self, tmp_path):
        """Two execute() calls for same target must not raise (DB handles idempotency)."""
        store = _fresh_store(tmp_path)
        executor = self._make_executor(store)
        policy = MagicMock()
        policy.dry_run = False
        policy.block_ttl_seconds = 60

        decision = self._make_decision("10.10.0.2")

        executor.execute(decision, policy)
        try:
            executor.execute(decision, policy)
        except Exception as exc:
            pytest.fail(f"execute() raised on second call for same target: {exc}")

    def test_migration_v4_idempotent_on_fresh_db(self, tmp_path):
        """Creating a second OpsStore on the same DB must not fail (index already exists)."""
        _fresh_store(tmp_path)
        OpsStore(str(tmp_path / "test.db"))  # second init — must not raise
