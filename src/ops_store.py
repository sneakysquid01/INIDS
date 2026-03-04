from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OpsStore:
    """Simple SQLite-backed operational store for alerts/actions/audit events."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    profile TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    expires_at TEXT,
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
                    created_at TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0,
                    dry_run INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    adapter TEXT NOT NULL DEFAULT 'mock'
=======
                    created_at TEXT NOT NULL
>>>>>>> theirs
=======
                    created_at TEXT NOT NULL
>>>>>>> theirs
=======
                    created_at TEXT NOT NULL
>>>>>>> theirs
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
            self._ensure_actions_columns(conn)

    @staticmethod
    def _ensure_actions_columns(conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(actions)").fetchall()
        }
        migrations: list[tuple[str, str]] = [
            ("executed", "ALTER TABLE actions ADD COLUMN executed INTEGER NOT NULL DEFAULT 0"),
            ("dry_run", "ALTER TABLE actions ADD COLUMN dry_run INTEGER NOT NULL DEFAULT 1"),
            ("status", "ALTER TABLE actions ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"),
            ("adapter", "ALTER TABLE actions ADD COLUMN adapter TEXT NOT NULL DEFAULT 'mock'"),
        ]
        for column, ddl in migrations:
            if column not in existing:
                conn.execute(ddl)
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def save_alert(self, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO alerts (id, timestamp, severity, prediction, confidence, profile, reason)
                VALUES (:id, :timestamp, :severity, :prediction, :confidence, :profile, :reason)
                """,
                payload,
            )

    def list_alerts(self, limit: int = 50, severity: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT id, timestamp, severity, prediction, confidence, profile, reason FROM alerts"
        params: list[Any] = []
        if severity:
            query += " WHERE lower(severity) = lower(?)"
            params.append(severity)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def save_action(self, payload: dict[str, Any]) -> None:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        row = {
            "action": payload.get("action"),
            "target": payload.get("target"),
            "reason": payload.get("reason"),
            "expires_at": payload.get("expires_at"),
            "created_at": payload.get("created_at"),
            "executed": 1 if bool(payload.get("executed", False)) else 0,
            "dry_run": 1 if bool(payload.get("dry_run", False)) else 0,
            "status": payload.get("status", "active"),
            "adapter": payload.get("adapter", "mock"),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO actions (action, target, reason, expires_at, created_at, executed, dry_run, status, adapter)
                VALUES (:action, :target, :reason, :expires_at, :created_at, :executed, :dry_run, :status, :adapter)
                """,
                row,
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO actions (action, target, reason, expires_at, created_at)
                VALUES (:action, :target, :reason, :expires_at, :created_at)
                """,
                payload,
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
            )

    def list_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
                SELECT action, target, reason, expires_at, created_at, executed, dry_run, status, adapter
=======
                SELECT action, target, reason, expires_at, created_at
>>>>>>> theirs
=======
                SELECT action, target, reason, expires_at, created_at
>>>>>>> theirs
=======
                SELECT action, target, reason, expires_at, created_at
>>>>>>> theirs
                FROM actions ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        payload = [dict(r) for r in rows]
        for row in payload:
            row["executed"] = bool(row.get("executed", 0))
            row["dry_run"] = bool(row.get("dry_run", 0))
        return payload

    @staticmethod
    def _parse_iso8601(value: str) -> datetime:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def list_expired_actions(self, now_iso: str | None = None) -> list[dict[str, Any]]:
        now_dt = self._parse_iso8601(now_iso) if now_iso else datetime.now(timezone.utc)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, action, target, reason, expires_at, created_at, executed, dry_run, status, adapter
                FROM actions
                WHERE expires_at IS NOT NULL
                """
            ).fetchall()
        expired: list[dict[str, Any]] = []
        for row in rows:
            expires_at = row["expires_at"]
            if not expires_at:
                continue
            try:
                expires_dt = self._parse_iso8601(expires_at)
            except ValueError:
                continue
            if expires_dt <= now_dt:
                payload = dict(row)
                payload["executed"] = bool(payload.get("executed", 0))
                payload["dry_run"] = bool(payload.get("dry_run", 0))
                expired.append(payload)
        return expired

    def cleanup_expired_actions(self, now_iso: str | None = None) -> int:
        expired = self.list_expired_actions(now_iso=now_iso)
        expired_ids = [int(row["id"]) for row in expired]
        if not expired_ids:
            return 0
        self.delete_actions(expired_ids)
        return len(expired_ids)

    def list_active_blocks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, action, target, reason, expires_at, created_at, executed, dry_run, status, adapter
                FROM actions
                WHERE action = 'block'
                  AND status IN ('active', 'temporary_block', 'blocked')
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        payload = [dict(r) for r in rows]
        for row in payload:
            row["executed"] = bool(row.get("executed", 0))
            row["dry_run"] = bool(row.get("dry_run", 0))
        return payload

    def delete_actions(self, action_ids: list[int]) -> int:
        if not action_ids:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM actions WHERE id = ?",
                [(int(rid),) for rid in action_ids],
            )
        return len(action_ids)

    def update_action_status(self, action_id: int, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE actions SET status = ? WHERE id = ?",
                (status, int(action_id)),
            )
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        return [dict(r) for r in rows]


    def cleanup_expired_actions(self, now_iso: str | None = None) -> int:
        now_value = now_iso or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM actions
                WHERE expires_at IS NOT NULL AND expires_at <= ?
                """,
                (now_value,),
            ).fetchall()
            expired_ids = [r["id"] for r in rows]
            if expired_ids:
                conn.executemany(
                    "DELETE FROM actions WHERE id = ?",
                    [(rid,) for rid in expired_ids],
                )
        return len(expired_ids)
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def add_audit(self, event_type: str, message: str, created_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audits (event_type, message, created_at)
                VALUES (?, ?, ?)
                """,
                (event_type, message, created_at),
            )

    def list_audits(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, message, created_at
                FROM audits ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
