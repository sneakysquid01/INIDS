#!/usr/bin/env python3
"""C-03 migration script: migrate inids_rbac.db RBAC users to OpsStore users table.

Usage:
    python scripts/migrate_rbac.py [--rbac-db PATH] [--ops-db PATH] [--sync]

Options:
    --rbac-db PATH   Path to inids_rbac.db  (default: data/inids_rbac.db)
    --ops-db  PATH   Path to inids_ops.db   (default: data/inids_ops.db)
    --sync           Run a final sync pass after the bulk migration
                     to capture any RBAC writes made during the migration window.

Exit codes:
    0  Success (or no RBAC database found — this is normal if RBAC was never used)
    1  One or more users failed to migrate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ops_store import OpsStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="C-03: Migrate RBAC users from inids_rbac.db to OpsStore"
    )
    parser.add_argument(
        "--rbac-db",
        default="data/inids_rbac.db",
        help="Path to source inids_rbac.db (default: data/inids_rbac.db)",
    )
    parser.add_argument(
        "--ops-db",
        default="data/inids_ops.db",
        help="Path to target inids_ops.db (default: data/inids_ops.db)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run a second sync pass to capture delta records from the migration window",
    )
    args = parser.parse_args()

    rbac_db = str(args.rbac_db)
    ops_db = str(args.ops_db)

    if not Path(rbac_db).exists():
        print(f"[C-03] RBAC source database not found: {rbac_db}")
        print("[C-03] No RBAC data to migrate — normal if RBAC was never initialised.")
        return 0

    print(f"[C-03] Opening OpsStore: {ops_db}")
    store = OpsStore(ops_db)

    print(f"[C-03] Bulk migration from: {rbac_db}")
    stats = store.migrate_rbac_users(rbac_db)
    print(
        f"[C-03] Bulk migration: migrated={stats['migrated']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )

    if args.sync:
        print("[C-03] Running final sync pass (delta check)...")
        sync_stats = store.migrate_rbac_users(rbac_db)
        print(
            f"[C-03] Sync pass:        migrated={sync_stats['migrated']} "
            f"skipped={sync_stats['skipped']} errors={sync_stats['errors']}"
        )
        if sync_stats["migrated"] > 0:
            print(
                f"[C-03] WARNING: {sync_stats['migrated']} new user(s) found in delta sync — "
                "RBAC was modified during the migration window. Verify post-sync state."
            )
        else:
            print("[C-03] Delta sync clean — no new records created during migration window.")

    total_errors = stats["errors"] + (sync_stats["errors"] if args.sync else 0)
    if total_errors > 0:
        print(f"[C-03] FAILED: {total_errors} error(s) during migration. Check logs.")
        return 1

    print("[C-03] Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
