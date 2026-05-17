from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
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
    
    # Current schema version - increment when making breaking schema changes
    SCHEMA_VERSION = 4

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

    def _execute(self, query: str, params: Any = None):
        with self._connect() as conn:
            if self._is_postgres:
                return conn.execute(text(query), params or {})
            return conn.execute(query, params or {})

    def _fetchall(self, query: str, params: Any = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if self._is_postgres:
                rows = conn.execute(text(query), params or {}).mappings().all()
                return [dict(row) for row in rows]
            rows = conn.execute(query, params or {}).fetchall()
            return [dict(row) for row in rows]

    def _fetchone(self, query: str, params: Any = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            if self._is_postgres:
                row = conn.execute(text(query), params or {}).mappings().first()
                return dict(row) if row is not None else None
            row = conn.execute(query, params or {}).fetchone()
            return dict(row) if row is not None else None

    def _init_db(self) -> None:
        self._run_migrations()
        self._verify_schema_version()

    # ------------------------------------------------------------------
    # Version-gated migration framework (B-01)
    # ------------------------------------------------------------------

    def _get_schema_version(self) -> int:
        """Return the current schema version.

        Returns 0 for a fresh database (no tables yet).
        Returns SCHEMA_VERSION for an existing database that predates the
        schema_version table — those databases already have all migrations applied.
        Creates the schema_version table on first call if absent.
        """
        try:
            result = self._fetchone(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            return int(result["version"]) if result else 0
        except Exception:
            pass

        # schema_version table is absent — determine whether this is a fresh or existing DB.
        try:
            if self._is_postgres:
                existing = self._fetchone(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'alerts' AND table_schema = 'public'"
                )
            else:
                existing = self._fetchone(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='alerts'"
                )
        except Exception:
            existing = None

        # Existing DBs without schema_version already have all migrations applied.
        initial = self.SCHEMA_VERSION if existing else 0
        self._create_schema_version_table()
        if initial > 0:
            self._set_schema_version(initial)
        return initial

    def _create_schema_version_table(self) -> None:
        if self._is_postgres:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        else:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                )
                """
            )

    def _set_schema_version(self, version: int) -> None:
        if self._is_postgres:
            self._execute(
                "INSERT INTO schema_version (version) VALUES (:v) ON CONFLICT DO NOTHING",
                {"v": version},
            )
        else:
            self._execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (:v)",
                {"v": version},
            )

    def _run_migrations(self) -> None:
        """Apply pending migrations in order; each runs exactly once per database."""
        import logging
        logger = logging.getLogger(__name__)
        current = self._get_schema_version()
        migrations = [
            (1, self._migration_v1_create_tables),
            (2, self._migration_v2_add_columns),
            (3, self._migration_v3_auth_tables),
            (4, self._migration_v4_actions_idempotency_index),
        ]
        for version, fn in migrations:
            if version > current:
                logger.info("Applying schema migration v%d", version)
                fn()
                self._set_schema_version(version)

    def _migration_v1_create_tables(self) -> None:
        """Schema v1: create all base tables."""
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
                    reason TEXT NOT NULL,
                    source_ip TEXT NOT NULL DEFAULT '',
                    attack_type TEXT NOT NULL DEFAULT '',
                    risk_score DOUBLE PRECISION NOT NULL DEFAULT 0.0
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
                CREATE TABLE IF NOT EXISTS fp_suppressions (
                    id BIGSERIAL PRIMARY KEY,
                    engine_id TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    suppressed INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    UNIQUE (engine_id, rule_id)
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
                        reason TEXT NOT NULL,
                        source_ip TEXT NOT NULL DEFAULT '',
                        attack_type TEXT NOT NULL DEFAULT '',
                        risk_score REAL NOT NULL DEFAULT 0.0
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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fp_suppressions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        engine_id TEXT NOT NULL,
                        rule_id TEXT NOT NULL,
                        suppressed INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        UNIQUE (engine_id, rule_id)
                    )
                    """
                )

    def _migration_v2_add_columns(self) -> None:
        """Schema v2: add extended columns to actions and alerts; normalize existing rows."""
        action_cols = {
            "action_id": "TEXT",
            "ip": "TEXT",
            "action_type": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "executed_at": "TEXT",
            "adapter": "TEXT",
            "dry_run": "BOOLEAN NOT NULL DEFAULT FALSE" if self._is_postgres else "INTEGER NOT NULL DEFAULT 0",
            "executed": "BOOLEAN NOT NULL DEFAULT FALSE" if self._is_postgres else "INTEGER NOT NULL DEFAULT 0",
        }
        alert_cols = {
            "status": "TEXT NOT NULL DEFAULT 'open'",
            "assignee": "TEXT",
            "close_reason": "TEXT",
            "status_updated_at": "TEXT",
            "source_ip": "TEXT NOT NULL DEFAULT ''",
            "attack_type": "TEXT NOT NULL DEFAULT ''",
            "risk_score": "DOUBLE PRECISION NOT NULL DEFAULT 0.0" if self._is_postgres else "REAL NOT NULL DEFAULT 0.0",
        }
        if self._is_postgres:
            for col, col_type in action_cols.items():
                self._execute(f"ALTER TABLE actions ADD COLUMN IF NOT EXISTS {col} {col_type}")
            self._execute("UPDATE actions SET action_id = COALESCE(action_id, '')")
            self._execute("UPDATE actions SET ip = COALESCE(ip, target)")
            self._execute("UPDATE actions SET action_type = COALESCE(action_type, action)")
            self._execute("UPDATE actions SET status = COALESCE(status, 'active')")
            self._execute("UPDATE actions SET dry_run = COALESCE(dry_run, FALSE)")
            self._execute("UPDATE actions SET executed = COALESCE(executed, FALSE)")
            for col, col_type in alert_cols.items():
                self._execute(f"ALTER TABLE alerts ADD COLUMN IF NOT EXISTS {col} {col_type}")
            self._execute("UPDATE alerts SET source_ip = COALESCE(source_ip, '')")
            self._execute("UPDATE alerts SET attack_type = COALESCE(attack_type, '')")
            self._execute("UPDATE alerts SET risk_score = COALESCE(risk_score, 0.0)")
            return

        with self._connect() as conn:
            action_existing = {row["name"] for row in conn.execute("PRAGMA table_info(actions)").fetchall()}
            for col, col_type in action_cols.items():
                if col not in action_existing:
                    conn.execute(f"ALTER TABLE actions ADD COLUMN {col} {col_type}")
            conn.execute("UPDATE actions SET action_id = COALESCE(action_id, '')")
            conn.execute("UPDATE actions SET ip = COALESCE(ip, target)")
            conn.execute("UPDATE actions SET action_type = COALESCE(action_type, action)")
            conn.execute("UPDATE actions SET status = COALESCE(status, 'active')")
            conn.execute("UPDATE actions SET dry_run = COALESCE(dry_run, 0)")
            conn.execute("UPDATE actions SET executed = COALESCE(executed, 0)")

            alert_existing = {row["name"] for row in conn.execute("PRAGMA table_info(alerts)").fetchall()}
            for col, col_type in alert_cols.items():
                if col not in alert_existing:
                    conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} {col_type}")
            conn.execute("UPDATE alerts SET source_ip = COALESCE(source_ip, '')")
            conn.execute("UPDATE alerts SET attack_type = COALESCE(attack_type, '')")
            conn.execute("UPDATE alerts SET risk_score = COALESCE(risk_score, 0.0)")

    def _migration_v3_auth_tables(self) -> None:
        """Schema v3: add users, api_keys, revoked_tokens tables.

        G-AUTH-1: idx_revoked_jti index created here (every auth check queries this table).
        Pre-seeds service accounts from env-var API keys (idempotent — INSERT OR IGNORE).
        """
        if self._is_postgres:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    roles TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            self._execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)"
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    jti TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT NOT NULL
                )
                """
            )
            # G-AUTH-1: index prevents full table scan on every JWT validation
            self._execute(
                "CREATE INDEX IF NOT EXISTS idx_revoked_jti ON revoked_tokens(jti)"
            )
        else:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        roles TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        key_hash TEXT UNIQUE NOT NULL,
                        label TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS revoked_tokens (
                        jti TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        revoked_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_revoked_jti ON revoked_tokens(jti)"
                )

        self._seed_service_accounts()

    def _seed_service_accounts(self) -> None:
        """Populate users + api_keys from env-var API keys (idempotent).

        Called once as part of migration v3. Existing rows are left untouched
        (INSERT OR IGNORE / ON CONFLICT DO NOTHING).
        """
        import hashlib
        import os

        env_key_map = {
            "INIDS_ADMIN_API_KEY": "admin",
            "INIDS_ANALYST_API_KEY": "analyst",
            "INIDS_SENSOR_API_KEY": "sensor",
            "INIDS_VIEWER_API_KEY": "viewer",
        }
        now = self._utc_now_iso()
        for env_var, role in env_key_map.items():
            raw_key = os.environ.get(env_var, "").strip()
            if not raw_key:
                continue
            user_id = f"svc-{role}"
            username = f"svc_{role}"
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            key_id = f"key-{role}"
            if self._is_postgres:
                self._execute(
                    "INSERT INTO users (user_id, username, roles, created_at) "
                    "VALUES (:uid, :uname, :roles, :now) ON CONFLICT DO NOTHING",
                    {"uid": user_id, "uname": username, "roles": role, "now": now},
                )
                self._execute(
                    "INSERT INTO api_keys (key_id, user_id, key_hash, label, created_at) "
                    "VALUES (:kid, :uid, :khash, :label, :now) ON CONFLICT DO NOTHING",
                    {"kid": key_id, "uid": user_id, "khash": key_hash, "label": env_var, "now": now},
                )
            else:
                self._execute(
                    "INSERT OR IGNORE INTO users (user_id, username, roles, created_at) "
                    "VALUES (:uid, :uname, :roles, :now)",
                    {"uid": user_id, "uname": username, "roles": role, "now": now},
                )
                self._execute(
                    "INSERT OR IGNORE INTO api_keys (key_id, user_id, key_hash, label, created_at) "
                    "VALUES (:kid, :uid, :khash, :label, :now)",
                    {"kid": key_id, "uid": user_id, "khash": key_hash, "label": env_var, "now": now},
                )

    def _migration_v4_actions_idempotency_index(self) -> None:
        """Schema v4: DB-level idempotency gate for prevention actions (C-02 / Step 20).

        Creates a partial unique index so a duplicate active block for the same target
        raises IntegrityError at the DB layer, not via application-level pre-checks.

        Requires SQLite ≥ 3.8.9 for partial index support.
        Runtime verified: sqlite3.sqlite_version ≥ 3.8.9 (checked at deploy time).
        """
        sql = """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_block
            ON actions(target)
            WHERE lower(status) IN ('active', 'enforced', 'executed')
              AND lower(action_type) IN ('block', 'temp_block', 'rate_limit')
        """
        self._execute(sql)

    def _verify_schema_version(self) -> None:
        """Confirm schema_version matches SCHEMA_VERSION; re-raise on any failure."""
        import logging
        logger = logging.getLogger(__name__)
        result = self._fetchone(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        db_version = int(result["version"]) if result else None
        if db_version != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version mismatch: "
                f"expected {self.SCHEMA_VERSION}, found {db_version}. "
                "Run migrations or reset the database."
            )
        logger.info("Schema version verified: %d", db_version)

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

    def save_alert(self, payload: dict[str, Any]) -> None:
        insert_payload = {
            "id": payload["id"],
            "timestamp": payload["timestamp"],
            "severity": str(payload["severity"]),
            "prediction": str(payload["prediction"]),
            "confidence": float(payload["confidence"]),
            "profile": str(payload["profile"]),
            "reason": str(payload["reason"]),
            "source_ip": str(payload.get("source_ip") or payload.get("src_ip") or ""),
            "attack_type": str(payload.get("attack_type") or ""),
            "risk_score": float(payload.get("risk_score") or 0.0),
        }
        if self._is_postgres:
            self._execute(
                """
                INSERT INTO alerts (id, timestamp, severity, prediction, confidence, profile, reason, source_ip, attack_type, risk_score)
                VALUES (:id, :timestamp, :severity, :prediction, :confidence, :profile, :reason, :source_ip, :attack_type, :risk_score)
                ON CONFLICT (id) DO NOTHING
                """,
                insert_payload,
            )
            return
        self._execute(
            """
            INSERT OR IGNORE INTO alerts (id, timestamp, severity, prediction, confidence, profile, reason, source_ip, attack_type, risk_score)
            VALUES (:id, :timestamp, :severity, :prediction, :confidence, :profile, :reason, :source_ip, :attack_type, :risk_score)
            """,
            insert_payload,
        )

    def list_alerts(
        self,
        limit: int = 50,
        severity: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id, timestamp, severity, prediction, confidence, profile, reason, "
            "COALESCE(status, 'open') AS status, assignee, close_reason, status_updated_at, "
            "COALESCE(source_ip, '') AS source_ip, COALESCE(attack_type, '') AS attack_type, COALESCE(risk_score, 0.0) AS risk_score "
            "FROM alerts"
        )
        params: dict[str, Any] = {"limit": int(limit)}
        conditions: list[str] = []
        if severity:
            conditions.append("lower(severity) = lower(:severity)")
            params["severity"] = severity
        if status:
            conditions.append("lower(COALESCE(status, 'open')) = lower(:status)")
            params["status"] = status
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT :limit"
        return self._fetchall(query, params)

    def update_alert(
        self,
        alert_id: str,
        *,
        status: str | None = None,
        assignee: str | None = None,
        close_reason: str | None = None,
    ) -> bool:
        """Update lifecycle fields for an alert. Returns True if a row was updated."""
        updates: list[str] = []
        params: dict[str, Any] = {"alert_id": alert_id}
        valid_statuses = {"open", "reviewing", "closed", "escalated"}
        if status is not None:
            if status.lower() not in valid_statuses:
                raise ValueError(f"status must be one of {valid_statuses}")
            updates.append("status = :status")
            params["status"] = status.lower()
            updates.append("status_updated_at = :status_updated_at")
            params["status_updated_at"] = self._utc_now_iso()
        if assignee is not None:
            updates.append("assignee = :assignee")
            params["assignee"] = assignee
        if close_reason is not None:
            updates.append("close_reason = :close_reason")
            params["close_reason"] = close_reason
        if not updates:
            return False
        query = f"UPDATE alerts SET {', '.join(updates)} WHERE id = :alert_id"
        cursor = self._execute(query, params)
        return bool(getattr(cursor, "rowcount", 0))

    # ------------------------------------------------------------------
    # False-positive suppression
    # ------------------------------------------------------------------

    def save_fp_suppression(self, engine_id: str, rule_id: str) -> None:
        """Persist a suppression entry; idempotent."""
        now_iso = self._utc_now_iso()
        if self._is_postgres:
            self._execute(
                """
                INSERT INTO fp_suppressions (engine_id, rule_id, suppressed, created_at)
                VALUES (:engine_id, :rule_id, 1, :created_at)
                ON CONFLICT (engine_id, rule_id) DO UPDATE SET suppressed = 1
                """,
                {"engine_id": engine_id, "rule_id": rule_id, "created_at": now_iso},
            )
        else:
            self._execute(
                """
                INSERT OR REPLACE INTO fp_suppressions (engine_id, rule_id, suppressed, created_at)
                VALUES (:engine_id, :rule_id, 1, :created_at)
                """,
                {"engine_id": engine_id, "rule_id": rule_id, "created_at": now_iso},
            )

    def delete_fp_suppression(self, engine_id: str, rule_id: str) -> bool:
        cursor = self._execute(
            "DELETE FROM fp_suppressions WHERE engine_id = :engine_id AND rule_id = :rule_id",
            {"engine_id": engine_id, "rule_id": rule_id},
        )
        return bool(getattr(cursor, "rowcount", 0))

    def list_fp_suppressions(self) -> list[dict[str, Any]]:
        return self._fetchall(
            "SELECT engine_id, rule_id, suppressed, created_at FROM fp_suppressions WHERE suppressed = 1"
        )

    def save_action(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Persist a prevention action. Returns None on success.

        If a duplicate active block for the same target violates the uq_active_block
        partial index (migration v4), the IntegrityError is caught here and the
        existing record is returned instead of raising. Callers never see the error.
        This replaces the application-level has_active_block() read-check-write pattern.
        """
        import sqlite3 as _sqlite3

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
        try:
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
            return None
        except _sqlite3.IntegrityError:
            # uq_active_block partial unique index violation: duplicate active block.
            # Return the existing record so the caller can treat this as a no-op.
            return self._fetchone(
                """
                SELECT * FROM actions
                WHERE target = :target
                  AND lower(COALESCE(status, '')) IN ('active', 'enforced', 'executed')
                  AND lower(COALESCE(action_type, action, '')) IN ('block', 'temp_block', 'rate_limit')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                {"target": target},
            )
        except Exception as exc:
            # PostgreSQL IntegrityError (arrives as sqlalchemy.exc.IntegrityError)
            exc_name = type(exc).__name__
            exc_str = str(exc).lower()
            if "integrityerror" in exc_name.lower() or "unique" in exc_str:
                return self._fetchone(
                    """
                    SELECT * FROM actions
                    WHERE target = :target
                      AND lower(COALESCE(status, '')) IN ('active', 'enforced', 'executed')
                      AND lower(COALESCE(action_type, action, '')) IN ('block', 'temp_block', 'rate_limit')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    {"target": target},
                )
            raise

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

    # ------------------------------------------------------------------
    # Unified auth queries (C-01)
    # ------------------------------------------------------------------

    def get_user_by_key_hash(self, key_hash: str) -> dict[str, Any] | None:
        """Return user_id, username, roles for a given API key hash.

        Joins api_keys → users. Uses idx_api_keys_hash (migration v3).
        """
        return self._fetchone(
            """
            SELECT u.user_id, u.username, u.roles
            FROM api_keys ak
            JOIN users u ON u.user_id = ak.user_id
            WHERE ak.key_hash = :key_hash
            """,
            {"key_hash": key_hash},
        )

    def add_revoked_token(
        self, jti: str, user_id: str, expires_at: str, revoked_at: str
    ) -> None:
        """Record a revoked JWT jti. Idempotent (duplicate jti is ignored)."""
        if self._is_postgres:
            self._execute(
                """
                INSERT INTO revoked_tokens (jti, user_id, expires_at, revoked_at)
                VALUES (:jti, :uid, :expires, :revoked)
                ON CONFLICT (jti) DO NOTHING
                """,
                {"jti": jti, "uid": user_id, "expires": expires_at, "revoked": revoked_at},
            )
        else:
            self._execute(
                """
                INSERT OR IGNORE INTO revoked_tokens (jti, user_id, expires_at, revoked_at)
                VALUES (:jti, :uid, :expires, :revoked)
                """,
                {"jti": jti, "uid": user_id, "expires": expires_at, "revoked": revoked_at},
            )

    def is_token_revoked(self, jti: str) -> bool:
        """Return True if jti appears in the revoked_tokens table.

        Uses idx_revoked_jti (G-AUTH-1) — O(log n) per auth check.
        """
        row = self._fetchone(
            "SELECT 1 FROM revoked_tokens WHERE jti = :jti",
            {"jti": jti},
        )
        return row is not None

    def cleanup_revoked_tokens(self, before_iso: str | None = None) -> int:
        """Delete expired records from revoked_tokens (C-04 daily cleanup).

        Removes rows whose expires_at is before the cutoff timestamp. These
        tokens would already be expired and rejected by the auth check, so
        their revocation records are no longer needed.

        - before_iso: ISO-8601 cutoff (default: now). Rows with expires_at
          strictly before this value are deleted.

        Returns: count of deleted rows.
        """
        cutoff = before_iso or self._utc_now_iso()
        result = self._execute(
            "DELETE FROM revoked_tokens WHERE expires_at < :cutoff",
            {"cutoff": cutoff},
        )
        return int(getattr(result, "rowcount", 0) or 0)

    # ------------------------------------------------------------------
    # RBAC migration (C-03)
    # ------------------------------------------------------------------

    def migrate_rbac_users(self, rbac_db_path: str) -> dict[str, int]:
        """Migrate users from an inids_rbac.db SQLite file to OpsStore users table.

        C-03 (Step 17): reads users/roles via cross-database INSERT OR IGNORE.
        Idempotent — safe to run for bulk migration and again for the final
        delta sync pass.

        - rbac_db_path missing → no-op, returns zeroed stats.
        - Only migrates is_active = 1 users.
        - Role names from user_roles JOIN roles are comma-joined into users.roles.

        Returns: {"migrated": int, "skipped": int, "errors": int}
        """
        _log = logging.getLogger(__name__)
        stats: dict[str, int] = {"migrated": 0, "skipped": 0, "errors": 0}

        if not Path(rbac_db_path).exists():
            return stats

        # Read all active users and their role names from the RBAC SQLite DB.
        # Cross-DB: no shared transaction — migration cannot be atomic.
        rbac_rows: list[Any] = []
        try:
            rbac_conn = sqlite3.connect(rbac_db_path)
            rbac_conn.row_factory = sqlite3.Row
            try:
                rbac_rows = rbac_conn.execute(
                    """
                    SELECT u.id AS user_id, u.username, u.created_at,
                           GROUP_CONCAT(r.name, ',') AS roles
                    FROM users u
                    LEFT JOIN user_roles ur ON ur.user_id = u.id
                    LEFT JOIN roles r ON r.id = ur.role_id
                    WHERE u.is_active = 1
                    GROUP BY u.id
                    """
                ).fetchall()
            except Exception as exc:
                _log.error("migrate_rbac_users: RBAC query failed: %s", exc)
                stats["errors"] += 1
                return stats
            finally:
                rbac_conn.close()
        except Exception as exc:
            _log.error("migrate_rbac_users: cannot open %s: %s", rbac_db_path, exc)
            stats["errors"] += 1
            return stats

        now = self._utc_now_iso()
        if self._is_postgres:
            sql = (
                "INSERT INTO users (user_id, username, roles, created_at) "
                "VALUES (:uid, :uname, :roles, :now) ON CONFLICT DO NOTHING"
            )
        else:
            sql = (
                "INSERT OR IGNORE INTO users (user_id, username, roles, created_at) "
                "VALUES (:uid, :uname, :roles, :now)"
            )

        for row in rbac_rows:
            user_id = str(row["user_id"])
            username = str(row["username"])
            roles = str(row["roles"] or "")
            created_at = str(row["created_at"] or now)
            try:
                result = self._execute(
                    sql,
                    {"uid": user_id, "uname": username, "roles": roles, "now": created_at},
                )
                if getattr(result, "rowcount", 1) == 1:
                    stats["migrated"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:
                _log.error("migrate_rbac_users: failed for user %s: %s", user_id, exc)
                stats["errors"] += 1

        return stats
