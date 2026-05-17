"""Regression tests for B-01: version-gated schema migrations.

Validates:
- Fresh DB: all migrations apply, schema_version = SCHEMA_VERSION
- Existing DB: no migrations re-run, schema_version unchanged
- Migrations are idempotent (multiple OpsStore init calls are safe)
- _verify_schema_version raises on version mismatch
- _verify_schema_version re-raises all exceptions (not just RuntimeError)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.ops_store import OpsStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema_version(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return bool(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fresh database
# ---------------------------------------------------------------------------

class TestFreshDatabase:
    def test_fresh_db_reaches_current_schema_version(self, tmp_path):
        db = str(tmp_path / "fresh.db")
        store = OpsStore(db)
        assert _schema_version(db) == OpsStore.SCHEMA_VERSION

    def test_fresh_db_creates_all_tables(self, tmp_path):
        db = str(tmp_path / "fresh.db")
        OpsStore(db)
        for table in ("alerts", "actions", "audits", "fp_suppressions", "allowlist", "schema_version"):
            assert _table_exists(db, table), f"Table {table!r} missing after fresh init"

    def test_fresh_db_alerts_has_extended_columns(self, tmp_path):
        db = str(tmp_path / "fresh.db")
        OpsStore(db)
        conn = sqlite3.connect(db)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(alerts)").fetchall()}
        finally:
            conn.close()
        for col in ("status", "assignee", "close_reason", "source_ip", "attack_type", "risk_score"):
            assert col in cols, f"Column {col!r} missing from alerts after fresh init"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestMigrationIdempotency:
    def test_schema_migration_idempotent(self, tmp_path):
        """Initialising OpsStore twice must not re-run migrations or change the version."""
        db = str(tmp_path / "idem.db")
        OpsStore(db)
        v1 = _schema_version(db)

        # Second init — should be a no-op
        OpsStore(db)
        v2 = _schema_version(db)

        assert v1 == v2 == OpsStore.SCHEMA_VERSION

    def test_ten_inits_idempotent(self, tmp_path):
        db = str(tmp_path / "multi.db")
        for _ in range(10):
            OpsStore(db)
        assert _schema_version(db) == OpsStore.SCHEMA_VERSION

    def test_no_migrations_rerun_on_existing_db(self, tmp_path):
        """An existing DB (tables present, schema_version absent) is treated as v2."""
        db_path = tmp_path / "legacy.db"

        # Simulate a legacy DB: create tables manually, no schema_version table
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE alerts (id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, "
            "severity TEXT NOT NULL, prediction TEXT NOT NULL, confidence REAL NOT NULL, "
            "profile TEXT NOT NULL, reason TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "action TEXT NOT NULL, target TEXT NOT NULL, reason TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        # OpsStore should recognise this as an existing DB and skip migrations
        store = OpsStore(str(db_path))
        assert _schema_version(str(db_path)) == OpsStore.SCHEMA_VERSION

        # Critically, no duplicate schema_version rows
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        conn.close()
        assert count == 1, f"Expected 1 schema_version row, found {count}"


# ---------------------------------------------------------------------------
# _verify_schema_version strictness
# ---------------------------------------------------------------------------

class TestVerifySchemaVersion:
    def test_version_mismatch_raises_runtime_error(self, tmp_path):
        db = str(tmp_path / "mismatch.db")
        store = OpsStore(db)

        # Manually corrupt the version
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (999)")
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="schema version mismatch"):
            store._verify_schema_version()

    def test_verify_does_not_swallow_exceptions(self, tmp_path):
        """_verify_schema_version must not silently absorb errors."""
        db = str(tmp_path / "noschema.db")
        store = OpsStore(db)

        # Drop the schema_version table to force a DB error
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE schema_version")
        conn.commit()
        conn.close()

        # Should raise — not silently pass
        with pytest.raises(Exception):
            store._verify_schema_version()
