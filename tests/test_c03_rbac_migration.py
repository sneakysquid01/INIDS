"""C-03 security regression tests — RBAC Migration into OpsStore (Step 17).

Validation checkpoints per PLAN.md C-03 spec:
1. migrate_rbac_users() with non-existent source → no-op, no error
2. Users in inids_rbac.db migrate to OpsStore users table
3. Role names are correctly comma-joined and stored in users.roles
4. Migration is idempotent (INSERT OR IGNORE semantics)
5. Final sync pass picks up new users created in the migration window
6. Inactive users (is_active=0) are not migrated
7. Pre-existing OpsStore users are not overwritten by migration
8. Migration stats accurately report migrated / skipped / errors
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ops_store import OpsStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_store(tmp_path: Path) -> OpsStore:
    return OpsStore(str(tmp_path / "ops.db"))


def _create_rbac_db(path: Path, users: list[dict] | None = None) -> None:
    """Create a minimal inids_rbac.db with the RBAC schema and optional user rows."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            full_name TEXT,
            is_active INTEGER DEFAULT 1,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS roles (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            is_builtin INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id TEXT REFERENCES users(id),
            role_id TEXT REFERENCES roles(id)
        );
        CREATE TABLE IF NOT EXISTS permissions (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            resource TEXT,
            action TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id TEXT REFERENCES roles(id),
            permission_id TEXT REFERENCES permissions(id)
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            action TEXT,
            resource TEXT,
            resource_id TEXT,
            allowed INTEGER,
            reason TEXT,
            timestamp TEXT
        );

        INSERT OR IGNORE INTO roles (id, name, is_builtin) VALUES ('admin', 'admin', 1);
        INSERT OR IGNORE INTO roles (id, name, is_builtin) VALUES ('analyst', 'analyst', 1);
        INSERT OR IGNORE INTO roles (id, name, is_builtin) VALUES ('viewer', 'viewer', 1);
        INSERT OR IGNORE INTO roles (id, name, is_builtin) VALUES ('sensor', 'sensor', 1);
    """)

    if users:
        now = datetime.now(timezone.utc).isoformat()
        for u in users:
            conn.execute(
                "INSERT OR IGNORE INTO users "
                "(id, username, email, is_active, is_admin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    u["id"],
                    u["username"],
                    u.get("email", ""),
                    1 if u.get("is_active", True) else 0,
                    1 if u.get("is_admin", False) else 0,
                    now,
                ),
            )
            for role in u.get("roles", []):
                conn.execute(
                    "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
                    (u["id"], role),
                )

    conn.commit()
    conn.close()


def _add_rbac_user(rbac_path: Path, user_id: str, username: str, roles: list[str]) -> None:
    """Insert a single user into an existing inids_rbac.db (simulates a write during migration window)."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(rbac_path))
    conn.execute(
        "INSERT OR IGNORE INTO users (id, username, is_active, created_at) VALUES (?, ?, 1, ?)",
        (user_id, username, now),
    )
    for role in roles:
        conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (user_id, role),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Checkpoint 1: Non-existent source → no-op
# ---------------------------------------------------------------------------

class TestMissingSource:
    def test_missing_db_returns_zeroed_stats(self, tmp_path):
        """Checkpoint 1: non-existent rbac_db_path returns zeroed stats."""
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(tmp_path / "does_not_exist.db"))
        assert stats == {"migrated": 0, "skipped": 0, "errors": 0}

    def test_missing_db_does_not_raise(self, tmp_path):
        store = _fresh_store(tmp_path)
        try:
            store.migrate_rbac_users(str(tmp_path / "does_not_exist.db"))
        except Exception as exc:
            pytest.fail(f"migrate_rbac_users() raised on missing file: {exc}")

    def test_empty_rbac_db_no_users_returns_zeroed_stats(self, tmp_path):
        """Empty RBAC database (schema only, no users) → no-op."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path)  # schema only, no users
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["migrated"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0


# ---------------------------------------------------------------------------
# Checkpoints 2 + 3: Users and roles migrate correctly
# ---------------------------------------------------------------------------

class TestUserMigration:
    def test_user_appears_in_ops_store_after_migration(self, tmp_path):
        """Checkpoint 2: migrated user exists in OpsStore users table."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "u-alice", "username": "alice", "roles": ["admin"]},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))

        assert stats["migrated"] == 1
        row = store._fetchone("SELECT * FROM users WHERE user_id = 'u-alice'")
        assert row is not None
        assert row["username"] == "alice"

    def test_single_role_stored_correctly(self, tmp_path):
        """Checkpoint 3: single RBAC role is stored in users.roles."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "u-bob", "username": "bob", "roles": ["analyst"]},
        ])
        store = _fresh_store(tmp_path)
        store.migrate_rbac_users(str(rbac_path))
        row = store._fetchone("SELECT roles FROM users WHERE user_id = 'u-bob'")
        assert "analyst" in (row["roles"] or "").split(",")

    def test_multiple_roles_comma_joined(self, tmp_path):
        """Checkpoint 3: multiple RBAC roles are comma-joined in users.roles."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "u-carol", "username": "carol", "roles": ["analyst", "viewer"]},
        ])
        store = _fresh_store(tmp_path)
        store.migrate_rbac_users(str(rbac_path))
        row = store._fetchone("SELECT roles FROM users WHERE user_id = 'u-carol'")
        role_set = set(r for r in (row["roles"] or "").split(",") if r)
        assert role_set == {"analyst", "viewer"}

    def test_admin_role_preserved(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "u-admin", "username": "dave_admin", "roles": ["admin"]},
        ])
        store = _fresh_store(tmp_path)
        store.migrate_rbac_users(str(rbac_path))
        row = store._fetchone("SELECT roles FROM users WHERE user_id = 'u-admin'")
        assert "admin" in (row["roles"] or "").split(",")

    def test_user_with_no_roles_migrated(self, tmp_path):
        """User with no roles assigned is still migrated (roles stored as empty string)."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "u-norole", "username": "norole_user", "roles": []},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["migrated"] == 1
        row = store._fetchone("SELECT * FROM users WHERE user_id = 'u-norole'")
        assert row is not None

    def test_multiple_users_all_migrated(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "m1", "username": "multi_user1", "roles": ["admin"]},
            {"id": "m2", "username": "multi_user2", "roles": ["analyst"]},
            {"id": "m3", "username": "multi_user3", "roles": ["viewer"]},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["migrated"] == 3
        assert stats["errors"] == 0


# ---------------------------------------------------------------------------
# Checkpoint 4: Idempotency (INSERT OR IGNORE semantics)
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_second_run_skips_already_migrated_users(self, tmp_path):
        """Checkpoint 4: running migration twice skips previously inserted users."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "idem1", "username": "idem_alice", "roles": ["analyst"]},
        ])
        store = _fresh_store(tmp_path)
        stats1 = store.migrate_rbac_users(str(rbac_path))
        stats2 = store.migrate_rbac_users(str(rbac_path))

        assert stats1["migrated"] == 1
        assert stats2["migrated"] == 0
        assert stats2["skipped"] == 1
        assert stats2["errors"] == 0

    def test_second_run_does_not_raise(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "idem2", "username": "idem_bob", "roles": ["viewer"]},
        ])
        store = _fresh_store(tmp_path)
        store.migrate_rbac_users(str(rbac_path))
        try:
            store.migrate_rbac_users(str(rbac_path))
        except Exception as exc:
            pytest.fail(f"Second migrate_rbac_users() raised unexpectedly: {exc}")

    def test_migration_idempotent_on_fresh_db(self, tmp_path):
        """Creating a second OpsStore on the same DB file must not fail after migration."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "idem3", "username": "idem_carol", "roles": ["admin"]},
        ])
        store = _fresh_store(tmp_path)
        store.migrate_rbac_users(str(rbac_path))
        OpsStore(str(tmp_path / "ops.db"))  # second OpsStore init — must not raise


# ---------------------------------------------------------------------------
# Checkpoint 5: Final sync pass picks up delta records
# ---------------------------------------------------------------------------

class TestDeltaSync:
    def test_sync_pass_captures_user_added_during_migration_window(self, tmp_path):
        """Checkpoint 5: user created in RBAC after bulk migration is caught by sync pass."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "d1", "username": "delta_initial", "roles": ["analyst"]},
        ])
        store = _fresh_store(tmp_path)

        # Bulk migration
        stats1 = store.migrate_rbac_users(str(rbac_path))
        assert stats1["migrated"] == 1

        # Simulate write during migration window — new user added to RBAC
        _add_rbac_user(rbac_path, "d2", "delta_window_user", ["viewer"])

        # Final sync pass should capture the delta
        stats2 = store.migrate_rbac_users(str(rbac_path))
        assert stats2["migrated"] == 1     # d2 is new
        assert stats2["skipped"] == 1     # d1 already present

        row = store._fetchone("SELECT * FROM users WHERE user_id = 'd2'")
        assert row is not None
        assert row["username"] == "delta_window_user"

    def test_sync_pass_no_delta_returns_all_skipped(self, tmp_path):
        """When no new users created, sync pass reports all skipped."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "nd1", "username": "nodelta_user", "roles": ["admin"]},
        ])
        store = _fresh_store(tmp_path)
        store.migrate_rbac_users(str(rbac_path))
        sync_stats = store.migrate_rbac_users(str(rbac_path))
        assert sync_stats["migrated"] == 0
        assert sync_stats["skipped"] == 1


# ---------------------------------------------------------------------------
# Checkpoint 6: Inactive users excluded
# ---------------------------------------------------------------------------

class TestInactiveUsers:
    def test_inactive_user_not_migrated(self, tmp_path):
        """Checkpoint 6: is_active=0 users are excluded from migration."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "act1", "username": "active_user", "roles": ["analyst"], "is_active": True},
            {"id": "ina1", "username": "inactive_user", "roles": ["analyst"], "is_active": False},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))

        assert stats["migrated"] == 1  # only active user

        active_row = store._fetchone("SELECT * FROM users WHERE user_id = 'act1'")
        assert active_row is not None

        inactive_row = store._fetchone("SELECT * FROM users WHERE user_id = 'ina1'")
        assert inactive_row is None

    def test_only_inactive_users_returns_zero_migrated(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "ina2", "username": "only_inactive", "roles": ["admin"], "is_active": False},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["migrated"] == 0
        assert stats["errors"] == 0


# ---------------------------------------------------------------------------
# Checkpoint 7: Pre-existing OpsStore users are not overwritten
# ---------------------------------------------------------------------------

class TestNoOverwrite:
    def test_existing_ops_store_user_roles_preserved(self, tmp_path):
        """Checkpoint 7: INSERT OR IGNORE does not overwrite an existing OpsStore user."""
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "pre1", "username": "preexisting", "roles": ["analyst"]},
        ])
        store = _fresh_store(tmp_path)

        # Pre-populate OpsStore with different roles for the same user_id
        store._execute(
            "INSERT OR IGNORE INTO users (user_id, username, roles, created_at) "
            "VALUES (:uid, :uname, :roles, :now)",
            {
                "uid": "pre1",
                "uname": "preexisting",
                "roles": "admin,analyst",
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )

        store.migrate_rbac_users(str(rbac_path))

        # Roles must not have been overwritten — "admin" must still be present
        row = store._fetchone("SELECT roles FROM users WHERE user_id = 'pre1'")
        assert row is not None
        role_set = set(r for r in (row["roles"] or "").split(",") if r)
        assert "admin" in role_set, "Pre-existing admin role must not be overwritten by migration"

    def test_existing_user_counted_as_skipped(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "pre2", "username": "pre_skipped", "roles": ["viewer"]},
        ])
        store = _fresh_store(tmp_path)
        store._execute(
            "INSERT OR IGNORE INTO users (user_id, username, roles, created_at) "
            "VALUES (:uid, :uname, :roles, :now)",
            {
                "uid": "pre2",
                "uname": "pre_skipped",
                "roles": "admin",
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["skipped"] == 1
        assert stats["migrated"] == 0


# ---------------------------------------------------------------------------
# Checkpoint 8: Migration stats accurately reported
# ---------------------------------------------------------------------------

class TestMigrationStats:
    def test_stats_keys_always_present(self, tmp_path):
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(tmp_path / "absent.db"))
        assert "migrated" in stats
        assert "skipped" in stats
        assert "errors" in stats

    def test_stats_migrated_count_matches_users(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "s1", "username": "stats_u1", "roles": ["admin"]},
            {"id": "s2", "username": "stats_u2", "roles": ["analyst"]},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["migrated"] == 2
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

    def test_stats_no_errors_on_clean_migration(self, tmp_path):
        rbac_path = tmp_path / "rbac.db"
        _create_rbac_db(rbac_path, users=[
            {"id": "e1", "username": "error_test_user", "roles": ["viewer"]},
        ])
        store = _fresh_store(tmp_path)
        stats = store.migrate_rbac_users(str(rbac_path))
        assert stats["errors"] == 0
