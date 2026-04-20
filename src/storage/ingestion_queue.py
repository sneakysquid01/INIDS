"""
SQLite-based Ingestion Queue for persistent buffering
Replaces InMemoryIngestionQueue for INIDS 2.0
"""

import sqlite3
import logging
import threading
import json
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class IngestionRecord:
    """Ingestion record data class"""
    source: str
    payload: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class SQLiteIngestionQueue:
    """
    SQLite-based ingestion queue for persistent buffering.
    Replaces InMemoryIngestionQueue for INIDS 2.0.
    
    Features:
    - Persistent storage across restarts
    - FIFO ordering
    - Atomic enqueue/dequeue operations
    - Thread-safe operations
    - Size limits with automatic cleanup
    """

    def __init__(self, db_path: str = "data/ingestion.db", max_items: int = 100000):
        """
        Initialize SQLite ingestion queue.
        
        Args:
            db_path: Path to SQLite database file
            max_items: Maximum items to keep in queue
        """
        self.db_path = db_path
        self.max_items = max(1, int(max_items))
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create ingestion queue table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ingestion_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT 0
                    )
                ''')
                
                # Create index for efficient queue operations
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ingestion_queue_created_at 
                    ON ingestion_queue(created_at ASC)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ingestion_queue_processed 
                    ON ingestion_queue(processed)
                ''')
                
                conn.commit()
                logger.info(f"SQLite ingestion queue initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ingestion queue: {e}")
            raise

    def enqueue(self, record: IngestionRecord) -> None:
        """
        Add a record to the queue.
        
        Args:
            record: IngestionRecord to queue
        """
        try:
            with self._lock:
                payload_json = json.dumps(record.payload)
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO ingestion_queue (source, payload)
                        VALUES (?, ?)
                    ''', (record.source, payload_json))
                    conn.commit()
                
                # Enforce size limit
                self._enforce_size_limit()
        except Exception as e:
            logger.error(f"Failed to enqueue record: {e}")

    def dequeue(self) -> Optional[IngestionRecord]:
        """
        Remove and return the oldest record from the queue.
        
        Returns:
            IngestionRecord or None if queue is empty
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get oldest record
                    cursor.execute('''
                        SELECT id, source, payload
                        FROM ingestion_queue
                        WHERE processed = 0
                        ORDER BY created_at ASC
                        LIMIT 1
                    ''')
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    record_id, source, payload_json = row
                    
                    # Mark as processed
                    cursor.execute('''
                        UPDATE ingestion_queue
                        SET processed = 1
                        WHERE id = ?
                    ''', (record_id,))
                    conn.commit()
                    
                    # Parse payload
                    try:
                        payload = json.loads(payload_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode payload for record {record_id}")
                        payload = {}
                    
                    return IngestionRecord(source=source, payload=payload)
        except Exception as e:
            logger.error(f"Failed to dequeue record: {e}")
            return None

    def size(self) -> int:
        """
        Get current queue size (unprocessed records).
        
        Returns:
            Number of unprocessed records in queue
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM ingestion_queue WHERE processed = 0')
                    result = cursor.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0

    def _enforce_size_limit(self) -> None:
        """Remove oldest processed records if queue exceeds size limit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current total size
                cursor.execute('SELECT COUNT(*) FROM ingestion_queue')
                total = cursor.fetchone()[0]
                
                # If over limit, remove oldest processed records
                if total > self.max_items:
                    excess = total - self.max_items
                    cursor.execute('''
                        DELETE FROM ingestion_queue
                        WHERE id IN (
                            SELECT id FROM ingestion_queue
                            WHERE processed = 1
                            ORDER BY created_at ASC
                            LIMIT ?
                        )
                    ''', (excess,))
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.debug(f"Enforced size limit: removed {cursor.rowcount} old records")
        except Exception as e:
            logger.error(f"Failed to enforce size limit: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the queue.
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Unprocessed count
                    cursor.execute('SELECT COUNT(*) FROM ingestion_queue WHERE processed = 0')
                    unprocessed = cursor.fetchone()[0]
                    
                    # Processed count
                    cursor.execute('SELECT COUNT(*) FROM ingestion_queue WHERE processed = 1')
                    processed = cursor.fetchone()[0]
                    
                    # Count by source
                    cursor.execute('''
                        SELECT source, COUNT(*) as count
                        FROM ingestion_queue
                        WHERE processed = 0
                        GROUP BY source
                    ''')
                    by_source = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    return {
                        "unprocessed_records": unprocessed,
                        "processed_records": processed,
                        "total_records": unprocessed + processed,
                        "by_source": by_source,
                        "db_path": self.db_path
                    }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"error": str(e)}


# Backward compatibility alias  
class InMemoryIngestionQueue(SQLiteIngestionQueue):
    """
    Backward compatibility wrapper around SQLiteIngestionQueue.
    Uses SQLite instead of in-memory storage for persistence.
    """
    
    def __init__(self, max_items: int = 10000):
        """Initialize with backward compatibility for max_items."""
        # Use SQLite with configurable max items
        super().__init__(db_path="data/ingestion.db", max_items=max_items)
        logger.info(f"InMemoryIngestionQueue initialized (actually using SQLite for persistence)")
