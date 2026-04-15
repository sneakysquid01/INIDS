"""Hierarchical incident aggregation - groups alerts into activities and incidents.

Inspired by WatchAD's three-level aggregation: Alert → Activity → Invasion
- Alert: Individual detection event
- Activity: Alerts grouped by same attack type + key target (same unique_id) within time window
- Incident: Activities grouped by source IP within time window

This provides operators with consolidated view of attack progression.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class IncidentAggregator:
    """Aggregates alerts into activities and incidents."""

    # Time windows for grouping
    ACTIVITY_WINDOW_SECONDS = 7 * 24 * 3600  # 7 days
    INCIDENT_WINDOW_SECONDS = 7 * 24 * 3600  # 7 days

    def __init__(self, ops_store) -> None:
        """Initialize aggregator with reference to OPS store."""
        self._ops_store = ops_store
        self._ensure_tables_exist()

    def _ensure_tables_exist(self) -> None:
        """Create activities and incidents tables if they don't exist."""
        # Create activities table
        if self._ops_store._is_postgres:
            self._ops_store._execute("""
                CREATE TABLE IF NOT EXISTS activities (
                    id TEXT PRIMARY KEY,
                    unique_id TEXT NOT NULL,
                    alert_code TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    repeat_count INTEGER NOT NULL DEFAULT 1,
                    incident_id TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            self._ops_store._execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    source_ip TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    activity_count INTEGER NOT NULL DEFAULT 0,
                    alert_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL
                )
            """)
            # Create indexes for efficient queries
            self._ops_store._execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_unique_id ON activities (unique_id)
            """)
            self._ops_store._execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_source_ip ON activities (source_ip)
            """)
            self._ops_store._execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_source_ip ON incidents (source_ip)
            """)
        else:
            with self._ops_store._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS activities (
                        id TEXT PRIMARY KEY,
                        unique_id TEXT NOT NULL,
                        alert_code TEXT NOT NULL,
                        source_ip TEXT NOT NULL,
                        attack_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        repeat_count INTEGER NOT NULL DEFAULT 1,
                        incident_id TEXT,
                        description TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id TEXT PRIMARY KEY,
                        source_ip TEXT NOT NULL,
                        description TEXT,
                        severity TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        activity_count INTEGER NOT NULL DEFAULT 0,
                        alert_count INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'open',
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_activities_unique_id ON activities (unique_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_activities_source_ip ON activities (source_ip)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_source_ip ON incidents (source_ip)
                """)

    def aggregate_alert(
        self,
        alert_id: str,
        alert_code: str,
        source_ip: str,
        attack_type: str,
        severity: str,
        timestamp: str,
        description: str = "",
    ) -> tuple[str, str]:
        """Aggregate alert into activity and incident.
        
        Returns: (activity_id, incident_id)
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Step 1: Create or update activity
        unique_id = self._compute_unique_id(alert_code, source_ip, attack_type)
        activity = self._get_or_create_activity(
            unique_id=unique_id,
            alert_code=alert_code,
            source_ip=source_ip,
            attack_type=attack_type,
            severity=severity,
            timestamp=timestamp,
            description=description,
        )
        activity_id = activity["id"]
        
        # Step 2: Create or update incident
        incident = self._get_or_create_incident(
            source_ip=source_ip,
            severity=severity,
            timestamp=timestamp,
        )
        incident_id = incident["id"]
        
        # Step 3: Link activity to incident
        self._link_activity_to_incident(activity_id, incident_id)
        
        return activity_id, incident_id

    def _compute_unique_id(self, alert_code: str, source_ip: str, attack_type: str) -> str:
        """Compute unique activity key from alert properties."""
        content = f"{alert_code}|{source_ip}|{attack_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_or_create_activity(
        self,
        unique_id: str,
        alert_code: str,
        source_ip: str,
        attack_type: str,
        severity: str,
        timestamp: str,
        description: str,
    ) -> dict[str, Any]:
        """Get existing activity or create new one."""
        now_iso = datetime.now(timezone.utc).isoformat()
        activity_window_ago = (
            datetime.now(timezone.utc) - timedelta(seconds=self.ACTIVITY_WINDOW_SECONDS)
        ).isoformat()
        
        # Check for existing activity within time window
        existing = self._ops_store._fetchall(
            """
            SELECT id, repeat_count, last_seen, severity FROM activities
            WHERE unique_id = :unique_id AND last_seen > :window_ago
            ORDER BY last_seen DESC LIMIT 1
            """,
            {"unique_id": unique_id, "window_ago": activity_window_ago},
        )
        
        if existing:
            # Update existing activity
            activity = existing[0]
            activity_id = activity["id"]
            new_repeat_count = activity["repeat_count"] + 1
            new_severity = self._escalate_severity(activity["severity"], severity)
            
            self._ops_store._execute(
                """
                UPDATE activities
                SET repeat_count = :repeat_count, last_seen = :now, severity = :severity
                WHERE id = :id
                """,
                {"id": activity_id, "repeat_count": new_repeat_count, "now": now_iso, "severity": new_severity},
            )
            return {
                "id": activity_id,
                "repeat_count": new_repeat_count,
                "severity": new_severity,
                "last_seen": now_iso,
            }
        else:
            # Create new activity
            activity_id = f"act_{hashlib.md5(f'{unique_id}|{now_iso}'.encode()).hexdigest()[:10]}"
            self._ops_store._execute(
                """
                INSERT INTO activities 
                (id, unique_id, alert_code, source_ip, attack_type, severity, first_seen, last_seen, repeat_count, description, created_at)
                VALUES (:id, :unique_id, :alert_code, :source_ip, :attack_type, :severity, :first_seen, :last_seen, 1, :description, :created_at)
                """,
                {
                    "id": activity_id,
                    "unique_id": unique_id,
                    "alert_code": alert_code,
                    "source_ip": source_ip,
                    "attack_type": attack_type,
                    "severity": severity,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "description": description,
                    "created_at": now_iso,
                },
            )
            return {
                "id": activity_id,
                "repeat_count": 1,
                "severity": severity,
                "last_seen": timestamp,
            }

    def _get_or_create_incident(
        self,
        source_ip: str,
        severity: str,
        timestamp: str,
    ) -> dict[str, Any]:
        """Get existing incident or create new one."""
        now_iso = datetime.now(timezone.utc).isoformat()
        incident_window_ago = (
            datetime.now(timezone.utc) - timedelta(seconds=self.INCIDENT_WINDOW_SECONDS)
        ).isoformat()
        
        # Check for existing incident within time window
        existing = self._ops_store._fetchall(
            """
            SELECT id, severity, last_seen, alert_count FROM incidents
            WHERE source_ip = :source_ip AND last_seen > :window_ago
            ORDER BY last_seen DESC LIMIT 1
            """,
            {"source_ip": source_ip, "window_ago": incident_window_ago},
        )
        
        if existing:
            # Update existing incident
            incident = existing[0]
            incident_id = incident["id"]
            new_severity = self._escalate_severity(incident["severity"], severity)
            new_alert_count = incident["alert_count"] + 1
            
            self._ops_store._execute(
                """
                UPDATE incidents
                SET last_seen = :now, severity = :severity, alert_count = :alert_count
                WHERE id = :id
                """,
                {"id": incident_id, "now": now_iso, "severity": new_severity, "alert_count": new_alert_count},
            )
            return {
                "id": incident_id,
                "severity": new_severity,
                "last_seen": now_iso,
                "alert_count": new_alert_count,
            }
        else:
            # Create new incident
            incident_id = f"inc_{hashlib.md5(f'{source_ip}|{now_iso}'.encode()).hexdigest()[:10]}"
            self._ops_store._execute(
                """
                INSERT INTO incidents 
                (id, source_ip, severity, first_seen, last_seen, activity_count, alert_count, status, created_at)
                VALUES (:id, :source_ip, :severity, :first_seen, :last_seen, 0, 1, 'open', :created_at)
                """,
                {
                    "id": incident_id,
                    "source_ip": source_ip,
                    "severity": severity,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "created_at": now_iso,
                },
            )
            return {
                "id": incident_id,
                "severity": severity,
                "last_seen": timestamp,
                "alert_count": 1,
            }

    def _link_activity_to_incident(self, activity_id: str, incident_id: str) -> None:
        """Link activity to incident and update counts."""
        # Check if activity is already linked to this incident
        existing = self._ops_store._fetchall(
            "SELECT id FROM activities WHERE id = :id AND incident_id = :incident_id",
            {"id": activity_id, "incident_id": incident_id},
        )
        
        if not existing:
            # Link activity to incident
            self._ops_store._execute(
                "UPDATE activities SET incident_id = :incident_id WHERE id = :id",
                {"id": activity_id, "incident_id": incident_id},
            )
            # Increment incident activity count
            self._ops_store._execute(
                "UPDATE incidents SET activity_count = activity_count + 1 WHERE id = :id",
                {"id": incident_id},
            )

    @staticmethod
    def _escalate_severity(current: str, new: str) -> str:
        """Escalate severity if new is worse than current."""
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        current_rank = severity_order.get(current, 0)
        new_rank = severity_order.get(new, 0)
        return new if new_rank > current_rank else current

    def get_activities(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent activities."""
        return self._ops_store._fetchall(
            """
            SELECT * FROM activities
            ORDER BY last_seen DESC LIMIT :limit
            """,
            {"limit": limit},
        )

    def get_incidents(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent incidents."""
        return self._ops_store._fetchall(
            """
            SELECT * FROM incidents
            ORDER BY last_seen DESC LIMIT :limit
            """,
            {"limit": limit},
        )

    def get_incident_details(self, incident_id: str) -> dict[str, Any] | None:
        """Get detailed incident info with all activities."""
        incident_rows = self._ops_store._fetchall(
            "SELECT * FROM incidents WHERE id = :id",
            {"id": incident_id},
        )
        if not incident_rows:
            return None
        
        incident = incident_rows[0]
        activities = self._ops_store._fetchall(
            "SELECT * FROM activities WHERE incident_id = :incident_id ORDER BY last_seen DESC",
            {"incident_id": incident_id},
        )
        
        return {
            "incident": incident,
            "activities": activities,
            "activity_count": len(activities),
        }
