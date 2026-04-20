"""
Dataset Collector for INIDS 2.0 ML Lifecycle
Collects detection features and feedback for model retraining
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
class TrainingRecord:
    """Training record for ML model"""
    detection_id: str
    features: Dict[str, Any]
    label: str  # "benign" or "malicious"
    confidence: float
    feedback_type: Optional[str] = None  # "tp" (true positive) or "fp" (false positive)


class DatasetCollector:
    """
    Collects detection features and analyst feedback for model retraining.
    Maintains persistent training dataset in SQLite.
    """

    def __init__(self, db_path: str = "data/training.db", retention_days: int = 30):
        """
        Initialize DatasetCollector.

        Args:
            db_path: Path to SQLite training database
            retention_days: How long to keep training data
        """
        self.db_path = db_path
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize training database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create training samples table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detection_id TEXT UNIQUE NOT NULL,
                        features TEXT NOT NULL,  -- JSON
                        label TEXT NOT NULL,  -- 'benign' or 'malicious'
                        confidence REAL NOT NULL,
                        feedback_type TEXT,  -- 'tp' or 'fp' if analyst-marked
                        marked_by_analyst BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        marked_at DATETIME
                    )
                ''')
                
                # Create indexes
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_training_created_at 
                    ON training_samples(created_at DESC)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_training_marked 
                    ON training_samples(marked_by_analyst)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_training_label 
                    ON training_samples(label)
                ''')
                
                conn.commit()
                logger.info(f"DatasetCollector initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize training database: {e}")
            raise

    def record_detection(
        self,
        detection_id: str,
        features: Dict[str, Any],
        label: str,
        confidence: float
    ) -> None:
        """
        Record a detection for potential training.

        Args:
            detection_id: Unique ID for this detection
            features: Feature dictionary from detection
            label: Ground truth label ('benign' or 'malicious')
            confidence: Model confidence score
        """
        try:
            with self._lock:
                features_json = json.dumps(features)
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR IGNORE INTO training_samples
                        (detection_id, features, label, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (detection_id, features_json, label, confidence))
                    conn.commit()
                    logger.debug(f"Recorded detection {detection_id} for training")
        except Exception as e:
            logger.error(f"Failed to record detection: {e}")

    def mark_feedback(
        self,
        detection_id: str,
        feedback_type: str
    ) -> None:
        """
        Mark analyst feedback on a detection.

        Args:
            detection_id: ID of the detection
            feedback_type: 'tp' (true positive) or 'fp' (false positive)
        """
        if feedback_type not in ('tp', 'fp'):
            logger.warning(f"Invalid feedback type: {feedback_type}")
            return

        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE training_samples
                        SET feedback_type = ?,
                            marked_by_analyst = 1,
                            marked_at = CURRENT_TIMESTAMP
                        WHERE detection_id = ?
                    ''', (feedback_type, detection_id))
                    conn.commit()
                    logger.debug(f"Marked feedback for {detection_id}: {feedback_type}")
        except Exception as e:
            logger.error(f"Failed to mark feedback: {e}")

    def get_labeled_data(self) -> List[Dict[str, Any]]:
        """
        Get all labeled training data for model training.

        Returns:
            List of training records (analyst-labeled only)
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT detection_id, features, feedback_type, confidence
                        FROM training_samples
                        WHERE marked_by_analyst = 1
                        ORDER BY marked_at DESC
                    ''')
                    
                    records = []
                    for row in cursor.fetchall():
                        detection_id, features_json, feedback_type, confidence = row
                        try:
                            features = json.loads(features_json)
                            records.append({
                                'detection_id': detection_id,
                                'features': features,
                                'feedback_type': feedback_type,  # tp or fp
                                'confidence': confidence
                            })
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode features for {detection_id}")
                    
                    return records
        except Exception as e:
            logger.error(f"Failed to get labeled data: {e}")
            return []

    def get_new_feedback(self, since_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get newly labeled feedback since timestamp.

        Args:
            since_timestamp: ISO timestamp to get feedback after (optional)

        Returns:
            List of newly labeled training records
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    if since_timestamp:
                        cursor.execute('''
                            SELECT detection_id, features, feedback_type, confidence, marked_at
                            FROM training_samples
                            WHERE marked_by_analyst = 1 AND marked_at > ?
                            ORDER BY marked_at DESC
                        ''', (since_timestamp,))
                    else:
                        cursor.execute('''
                            SELECT detection_id, features, feedback_type, confidence, marked_at
                            FROM training_samples
                            WHERE marked_by_analyst = 1
                            ORDER BY marked_at DESC
                        ''')
                    
                    records = []
                    for row in cursor.fetchall():
                        detection_id, features_json, feedback_type, confidence, marked_at = row
                        try:
                            features = json.loads(features_json)
                            records.append({
                                'detection_id': detection_id,
                                'features': features,
                                'feedback_type': feedback_type,
                                'confidence': confidence,
                                'marked_at': marked_at
                            })
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode features for {detection_id}")
                    
                    return records
        except Exception as e:
            logger.error(f"Failed to get new feedback: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected training data.

        Returns:
            Dictionary with training data statistics
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Total samples
                    cursor.execute('SELECT COUNT(*) FROM training_samples')
                    total = cursor.fetchone()[0]
                    
                    # Analyst-marked samples
                    cursor.execute('SELECT COUNT(*) FROM training_samples WHERE marked_by_analyst = 1')
                    marked = cursor.fetchone()[0]
                    
                    # Count by feedback type
                    cursor.execute('''
                        SELECT feedback_type, COUNT(*) as count
                        FROM training_samples
                        WHERE marked_by_analyst = 1
                        GROUP BY feedback_type
                    ''')
                    by_type = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Count by label
                    cursor.execute('''
                        SELECT label, COUNT(*) as count
                        FROM training_samples
                        GROUP BY label
                    ''')
                    by_label = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    return {
                        'total_samples': total,
                        'analyst_marked': marked,
                        'by_feedback_type': by_type,
                        'by_label': by_label,
                        'db_path': self.db_path
                    }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

    def export_for_training(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export labeled data for model training.

        Args:
            output_path: Optional path to save CSV export

        Returns:
            Dictionary with training data and metadata
        """
        try:
            labeled_data = self.get_labeled_data()
            
            # Convert to training format
            X = []
            y = []
            
            for record in labeled_data:
                features = record['features']
                label = 1 if record['feedback_type'] == 'tp' else 0  # 1=malicious, 0=benign
                
                X.append(features)
                y.append(label)
            
            result = {
                'X': X,
                'y': y,
                'sample_count': len(X),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if output_path:
                try:
                    import pandas as pd
                    df = pd.DataFrame(X)
                    df['label'] = y
                    df.to_csv(output_path, index=False)
                    logger.info(f"Exported training data to {output_path}")
                    result['csv_path'] = output_path
                except Exception as e:
                    logger.warning(f"Failed to export CSV: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return {'error': str(e)}
