from __future__ import annotations

import sqlite3
from contextlib import contextmanager
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
                    created_at TEXT NOT NULL
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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO actions (action, target, reason, expires_at, created_at)
                VALUES (:action, :target, :reason, :expires_at, :created_at)
                """,
                payload,
            )

    def list_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action, target, reason, expires_at, created_at
                FROM actions ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

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
