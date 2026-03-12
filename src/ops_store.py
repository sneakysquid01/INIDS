from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid

try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None


class OpsStore:
    """Operational persistence supporting SQLite (dev) and PostgreSQL (prod)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_url = self._normalize_db_url(db_path)
        self._is_postgres = self.db_url.startswith("postgresql://")
        self._engine = None

        if self._is_postgres:
            if create_engine is None or text is None:
                raise RuntimeError("PostgreSQL backend requires SQLAlchemy to be installed")
            self._engine = create_engine(self.db_url, future=True)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _normalize_db_url(raw_value: str) -> str:
        value = str(raw_value or "").strip()
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self._is_postgres:
            assert self._engine is not None
            with self._engine.begin() as conn:
                yield conn
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _execute(self, query: str, params: dict[str, Any] | None = None):
        with self._connect() as conn:
            if self._is_postgres:
                return conn.execute(text(query), params or {})
            return conn.execute(query, params or {})

    def _fetchall(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if self._is_postgres:
                rows = conn.execute(text(query), params or {}).mappings().all()
                return [dict(row) for row in rows]
            rows = conn.execute(query, params or {}).fetchall()
            return [dict(row) for row in rows]

    def _init_db(self) -> None:
        if self._is_postgres:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    profile TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id BIGSERIAL PRIMARY KEY,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    action_id TEXT,
                    ip TEXT,
                    action_type TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    executed_at TEXT,
                    adapter TEXT,
                    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
                    executed BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS audits (
                    id BIGSERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS allowlist (
                    id BIGSERIAL PRIMARY KEY,
                    entry TEXT UNIQUE NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
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
                        action_id TEXT,
                        ip TEXT,
                        action_type TEXT,
                        status TEXT NOT NULL DEFAULT 'active',
                        expires_at TEXT,
                        created_at TEXT NOT NULL,
                        executed_at TEXT,
                        adapter TEXT,
                        dry_run INTEGER NOT NULL DEFAULT 0,
                        executed INTEGER NOT NULL DEFAULT 0
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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS allowlist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entry TEXT UNIQUE NOT NULL,
                        reason TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )

        self._migrate_actions_table()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_bool_int(value: Any, default: int = 0) -> int:
        if value is None:
            return int(default)
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if int(value) != 0 else 0
        return 1 if str(value).strip().lower() in {"1", "true", "yes", "y"} else 0

    def _migrate_actions_table(self) -> None:
        migrations = {
            "action_id": "TEXT",
            "ip": "TEXT",
            "action_type": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "executed_at": "TEXT",
            "adapter": "TEXT",
            "dry_run": "BOOLEAN NOT NULL DEFAULT FALSE" if self._is_postgres else "INTEGER NOT NULL DEFAULT 0",
            "executed": "BOOLEAN NOT NULL DEFAULT FALSE" if self._is_postgres else "INTEGER NOT NULL DEFAULT 0",
        }

        if self._is_postgres:
            for col, col_type in migrations.items():
                self._execute(f"ALTER TABLE actions ADD COLUMN IF NOT EXISTS {col} {col_type}")
            self._execute("UPDATE actions SET action_id = COALESCE(action_id, '')")
            self._execute("UPDATE actions SET ip = COALESCE(ip, target)")
            self._execute("UPDATE actions SET action_type = COALESCE(action_type, action)")
            self._execute("UPDATE actions SET status = COALESCE(status, 'active')")
            self._execute("UPDATE actions SET dry_run = COALESCE(dry_run, FALSE)")
            self._execute("UPDATE actions SET executed = COALESCE(executed, FALSE)")
            return

        with self._connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(actions)").fetchall()}
            for col, col_type in migrations.items():
                if col not in columns:
                    conn.execute(f"ALTER TABLE actions ADD COLUMN {col} {col_type}")
            conn.execute("UPDATE actions SET action_id = COALESCE(action_id, '')")
            conn.execute("UPDATE actions SET ip = COALESCE(ip, target)")
            conn.execute("UPDATE actions SET action_type = COALESCE(action_type, action)")
            conn.execute("UPDATE actions SET status = COALESCE(status, 'active')")
            conn.execute("UPDATE actions SET dry_run = COALESCE(dry_run, 0)")
            conn.execute("UPDATE actions SET executed = COALESCE(executed, 0)")

    def save_alert(self, payload: dict[str, Any]) -> None:
        if self._is_postgres:
            self._execute(
                """
                INSERT INTO alerts (id, timestamp, severity, prediction, confidence, profile, reason)
                VALUES (:id, :timestamp, :severity, :prediction, :confidence, :profile, :reason)
                ON CONFLICT (id) DO NOTHING
                """,
                payload,
            )
            return
        self._execute(
            """
            INSERT OR IGNORE INTO alerts (id, timestamp, severity, prediction, confidence, profile, reason)
            VALUES (:id, :timestamp, :severity, :prediction, :confidence, :profile, :reason)
            """,
            payload,
        )

    def list_alerts(self, limit: int = 50, severity: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT id, timestamp, severity, prediction, confidence, profile, reason FROM alerts"
        params: dict[str, Any] = {"limit": int(limit)}
        if severity:
            query += " WHERE lower(severity) = lower(:severity)"
            params["severity"] = severity
        query += " ORDER BY timestamp DESC LIMIT :limit"
        return self._fetchall(query, params)

    def save_action(self, payload: dict[str, Any]) -> None:
        action = payload.get("action", payload.get("action_type", "block"))
        target = payload.get("target", payload.get("ip", "unknown"))
        now_iso = self._utc_now_iso()
        created_at = payload.get("created_at") or now_iso
        action_id = payload.get("action_id") or f"act_{uuid.uuid4().hex[:16]}"
        status = payload.get("status", "active")
        executed = self._to_bool_int(payload.get("executed"), default=0)
        dry_run = self._to_bool_int(payload.get("dry_run"), default=0)
        executed_at = payload.get("executed_at")
        if executed and not executed_at:
            executed_at = created_at

        insert_payload = {
            "action": action,
            "target": target,
            "reason": payload.get("reason", ""),
            "action_id": action_id,
            "ip": payload.get("ip", target),
            "action_type": payload.get("action_type", action),
            "status": status,
            "expires_at": payload.get("expires_at"),
            "created_at": created_at,
            "executed_at": executed_at,
            "adapter": payload.get("adapter"),
            "dry_run": bool(dry_run) if self._is_postgres else dry_run,
            "executed": bool(executed) if self._is_postgres else executed,
        }
        self._execute(
            """
            INSERT INTO actions (
                action,
                target,
                reason,
                action_id,
                ip,
                action_type,
                status,
                expires_at,
                created_at,
                executed_at,
                adapter,
                dry_run,
                executed
            )
            VALUES (
                :action,
                :target,
                :reason,
                :action_id,
                :ip,
                :action_type,
                :status,
                :expires_at,
                :created_at,
                :executed_at,
                :adapter,
                :dry_run,
                :executed
            )
            """,
            insert_payload,
        )

    def list_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._fetchall(
            """
            SELECT
                action,
                target,
                reason,
                expires_at,
                created_at,
                action_id,
                ip,
                action_type,
                status,
                executed_at,
                adapter,
                dry_run,
                executed
            FROM actions ORDER BY id DESC LIMIT :limit
            """,
            {"limit": int(limit)},
        )

    def list_active_blocks(self, limit: int = 5000) -> list[dict[str, Any]]:
        return self._fetchall(
            """
            SELECT
                id,
                action_id,
                target,
                ip,
                action,
                action_type,
                reason,
                status,
                adapter,
                created_at,
                expires_at,
                executed_at,
                dry_run,
                executed
            FROM actions
            WHERE lower(COALESCE(action_type, action, '')) IN ('block', 'temp_block', 'rate_limit')
              AND lower(COALESCE(status, '')) IN ('active', 'executed', 'enforced')
              AND (expires_at IS NULL OR expires_at = '' OR expires_at > :now_iso)
            ORDER BY id DESC
            LIMIT :limit
            """,
            {"now_iso": self._utc_now_iso(), "limit": int(limit)},
        )

    def list_expired_actions(self, now_iso: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        if not now_iso:
            now_iso = self._utc_now_iso()
        return self._fetchall(
            """
            SELECT
                id,
                action_id,
                target,
                ip,
                action,
                action_type,
                reason,
                status,
                adapter,
                created_at,
                expires_at,
                executed_at,
                dry_run,
                executed
            FROM actions
            WHERE expires_at IS NOT NULL
              AND expires_at != ''
              AND expires_at <= :now_iso
            ORDER BY id ASC
            LIMIT :limit
            """,
            {"now_iso": now_iso, "limit": int(limit)},
        )

    def update_action_status(
        self,
        action_id: int | str,
        status: str,
        executed_at: str | None = None,
    ) -> int:
        query = "UPDATE actions SET status = :status"
        params: dict[str, Any] = {"status": status}
        if executed_at is not None:
            query += ", executed_at = :executed_at"
            params["executed_at"] = executed_at
        if isinstance(action_id, int):
            query += " WHERE id = :lookup_id"
            params["lookup_id"] = action_id
        else:
            query += " WHERE action_id = :lookup_id"
            params["lookup_id"] = str(action_id)
        cursor = self._execute(query, params)
        return int(cursor.rowcount or 0)

    def delete_actions(self, action_ids: list[int | str]) -> int:
        if not action_ids:
            return 0
        removed = 0

        int_ids = [x for x in action_ids if isinstance(x, int)]
        str_ids = [str(x) for x in action_ids if not isinstance(x, int)]

        if int_ids:
            params = {f"id_{i}": v for i, v in enumerate(int_ids)}
            placeholders = ", ".join(f":id_{i}" for i in range(len(int_ids)))
            cursor = self._execute(f"DELETE FROM actions WHERE id IN ({placeholders})", params)
            removed += int(cursor.rowcount or 0)
        if str_ids:
            params = {f"aid_{i}": v for i, v in enumerate(str_ids)}
            placeholders = ", ".join(f":aid_{i}" for i in range(len(str_ids)))
            cursor = self._execute(f"DELETE FROM actions WHERE action_id IN ({placeholders})", params)
            removed += int(cursor.rowcount or 0)
        return removed

    def cleanup_expired_actions(self, now_iso: str | None = None) -> int:
        if not now_iso:
            now_iso = self._utc_now_iso()
        expired = self.list_expired_actions(now_iso=now_iso, limit=100000)
        ids = [int(row["id"]) for row in expired]
        return self.delete_actions(ids)

    def add_audit(self, event_type: str, message: str, created_at: str) -> None:
        self._execute(
            """
            INSERT INTO audits (event_type, message, created_at)
            VALUES (:event_type, :message, :created_at)
            """,
            {"event_type": event_type, "message": message, "created_at": created_at},
        )

    def list_audits(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._fetchall(
            """
            SELECT event_type, message, created_at
            FROM audits ORDER BY id DESC LIMIT :limit
            """,
            {"limit": int(limit)},
        )

    # ------------------------------------------------------------------
    # Allowlist
    # ------------------------------------------------------------------

    def list_allowlist(self) -> list[dict[str, Any]]:
        return self._fetchall(
            "SELECT id, entry, reason, created_at FROM allowlist ORDER BY created_at DESC"
        )

    def add_allowlist_entry(self, entry: str, *, reason: str = "") -> bool:
        payload = {"entry": entry, "reason": reason, "created_at": self._utc_now_iso()}
        try:
            if self._is_postgres:
                self._execute(
                    """
                    INSERT INTO allowlist (entry, reason, created_at)
                    VALUES (:entry, :reason, :created_at)
                    ON CONFLICT (entry) DO NOTHING
                    """,
                    payload,
                )
            else:
                self._execute(
                    """
                    INSERT OR IGNORE INTO allowlist (entry, reason, created_at)
                    VALUES (:entry, :reason, :created_at)
                    """,
                    payload,
                )
            return True
        except Exception:
            return False

    def remove_allowlist_entry(self, entry: str) -> bool:
        cursor = self._execute(
            "DELETE FROM allowlist WHERE entry = :entry",
            {"entry": entry},
        )
        return bool(getattr(cursor, "rowcount", 0))

    def has_active_block(self, ip: str) -> bool:
        """Return True if ``ip`` already has an active block/rate-limit record."""
        rows = self._fetchall(
            """
            SELECT 1 FROM actions
            WHERE lower(COALESCE(action_type, action, '')) IN ('block', 'temp_block', 'rate_limit')
              AND lower(COALESCE(status, '')) IN ('active', 'executed', 'enforced')
              AND (expires_at IS NULL OR expires_at = '' OR expires_at > :now_iso)
              AND target = :target
            LIMIT 1
            """,
            {"now_iso": self._utc_now_iso(), "target": ip},
        )
        return bool(rows)
