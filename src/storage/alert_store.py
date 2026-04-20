"""
SQLite-based Alert Store for persistent storage
Replaces InMemoryAlertStore for INIDS 2.0
"""

import sqlite3
import logging
import threading
import json
from datetime import datetime, timezone, timedelta
from dataclasses import asdict, dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert data class"""
    id: str
    timestamp: str
    severity: str
    prediction: str
    confidence: float
    profile: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class SQLiteAlertStore:
    """
    SQLite-based alert store for persistent storage.
    Replaces InMemoryAlertStore for INIDS 2.0.
    
    Features:
    - Persistent storage across restarts
    - Fast indexing on timestamp and severity
    - Retention policy (default: 30 days)
    - Thread-safe operations
    """

    def __init__(self, db_path: str = "data/alerts.db", max_retention_days: int = 30):
        """
        Initialize SQLite alert store.
        
        Args:
            db_path: Path to SQLite database file
            max_retention_days: Retention policy (default: 30 days)
        """
        self.db_path = db_path
        self.max_retention_days = max_retention_days
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        profile TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for fast queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON alerts(timestamp DESC)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_severity 
                    ON alerts(severity)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_created_at 
                    ON alerts(created_at DESC)
                ''')
                
                conn.commit()
                logger.info(f"SQLite alert store initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize alert store: {e}")
            raise

    def add(self, alert: Optional[Alert]) -> None:
        """
        Add an alert to the store.
        
        Args:
            alert: Alert object to store
        """
        if alert is None:
            return

        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO alerts (id, timestamp, severity, prediction, confidence, profile, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.id,
                        alert.timestamp,
                        alert.severity,
                        alert.prediction,
                        alert.confidence,
                        alert.profile,
                        alert.reason
                    ))
                    conn.commit()
                
                # Clean up old alerts (retention policy)
                self._cleanup_old_alerts()
        except Exception as e:
            logger.error(f"Failed to add alert: {e}")

    def list_alerts(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
        offset: int = 0
    ) -> List[Alert]:
        """
        Retrieve alerts from the store.
        
        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity (optional)
            offset: Offset for pagination
            
        Returns:
            List of Alert objects
        """
        limit = max(1, min(limit, 10000))
        offset = max(0, offset)
        
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    if severity:
                        cursor.execute('''
                            SELECT id, timestamp, severity, prediction, confidence, profile, reason
                            FROM alerts
                            WHERE severity = ?
                            ORDER BY created_at DESC
                            LIMIT ? OFFSET ?
                        ''', (severity.strip().lower(), limit, offset))
                    else:
                        cursor.execute('''
                            SELECT id, timestamp, severity, prediction, confidence, profile, reason
                            FROM alerts
                            ORDER BY created_at DESC
                            LIMIT ? OFFSET ?
                        ''', (limit, offset))
                    
                    rows = cursor.fetchall()
                    alerts = [
                        Alert(
                            id=row[0],
                            timestamp=row[1],
                            severity=row[2],
                            prediction=row[3],
                            confidence=row[4],
                            profile=row[5],
                            reason=row[6]
                        )
                        for row in rows
                    ]
                    return alerts
        except Exception as e:
            logger.error(f"Failed to list alerts: {e}")
            return []

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Retrieve a specific alert by ID.
        
        Args:
            alert_id: ID of the alert to retrieve
            
        Returns:
            Alert object or None if not found
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT id, timestamp, severity, prediction, confidence, profile, reason
                        FROM alerts
                        WHERE id = ?
                    ''', (alert_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return Alert(
                            id=row[0],
                            timestamp=row[1],
                            severity=row[2],
                            prediction=row[3],
                            confidence=row[4],
                            profile=row[5],
                            reason=row[6]
                        )
                    return None
        except Exception as e:
            logger.error(f"Failed to get alert {alert_id}: {e}")
            return None

    def count_alerts(self, severity: Optional[str] = None) -> int:
        """
        Count alerts in the store.
        
        Args:
            severity: Filter by severity (optional)
            
        Returns:
            Total count of alerts
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    if severity:
                        cursor.execute('''
                            SELECT COUNT(*) FROM alerts WHERE severity = ?
                        ''', (severity.strip().lower(),))
                    else:
                        cursor.execute('SELECT COUNT(*) FROM alerts')
                    
                    result = cursor.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count alerts: {e}")
            return 0

    def _cleanup_old_alerts(self) -> None:
        """Remove alerts older than retention policy."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_retention_days)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM alerts
                    WHERE created_at < ?
                ''', (cutoff_str,))
                
                if cursor.rowcount > 0:
                    logger.debug(f"Cleaned up {cursor.rowcount} old alerts")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Total count
                    cursor.execute('SELECT COUNT(*) FROM alerts')
                    total = cursor.fetchone()[0]
                    
                    # Count by severity
                    cursor.execute('''
                        SELECT severity, COUNT(*) as count
                        FROM alerts
                        GROUP BY severity
                    ''')
                    severity_stats = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Latest alert
                    cursor.execute('''
                        SELECT timestamp FROM alerts
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''')
                    latest = cursor.fetchone()[0] if cursor.fetchone() else None
                    
                    return {
                        "total_alerts": total,
                        "by_severity": severity_stats,
                        "latest_alert": latest,
                        "db_path": self.db_path
                    }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Backward compatibility alias
class InMemoryAlertStore(SQLiteAlertStore):
    """
    Backward compatibility wrapper around SQLiteAlertStore.
    Uses SQLite instead of in-memory storage for persistence.
    """
    
    def __init__(self, max_items: int = 1000):
        """Initialize with backward compatibility for max_items."""
        # Use SQLite with a conservative retention policy
        super().__init__(db_path="data/alerts.db", max_retention_days=30)
        self.max_items = max_items
        logger.info(f"InMemoryAlertStore initialized (actually using SQLite for persistence)")
