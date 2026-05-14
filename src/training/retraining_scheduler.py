"""
Retraining Scheduler for INIDS 2.0 ML Lifecycle
Manages scheduled retraining, model versioning, and deployment
"""

import logging
import threading
import json
import time
import joblib
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Manages model retraining schedule and execution.
    Supports scheduled, drift-triggered, and feedback-triggered retraining.
    """

    def __init__(
        self,
        dataset_collector,
        model_registry,
        ml_engine,
        models_dir: str = "models",
        schedule_hour: int = 2,  # Daily at 2 AM UTC
        drift_threshold: float = 0.2  # PSI threshold for drift detection
    ):
        """
        Initialize RetrainingScheduler.

        Args:
            dataset_collector: DatasetCollector instance
            model_registry: ModelRegistry instance
            ml_engine: MLEngine instance to train
            models_dir: Directory for model files
            schedule_hour: Hour (UTC) for daily retraining (0-23)
            drift_threshold: PSI threshold for drift detection
        """
        self.dataset_collector = dataset_collector
        self.model_registry = model_registry
        self.ml_engine = ml_engine
        self.models_dir = models_dir
        self.schedule_hour = schedule_hour
        self.drift_threshold = drift_threshold
        
        self._lock = threading.RLock()
        self._scheduler_thread = None
        self._running = False
        self._last_training_time = None
        self._training_in_progress = False
        self._training_history = []

    def start(self) -> None:
        """Start the retraining scheduler."""
        if self._running:
            logger.warning("Retraining scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True,
            name="retraining-scheduler"
        )
        self._scheduler_thread.start()
        logger.info("Retraining scheduler started")

    def stop(self) -> None:
        """Stop the retraining scheduler."""
        self._running = False
        logger.info("Retraining scheduler stopped")

    def _schedule_loop(self) -> None:
        """Main scheduler loop - checks for retraining triggers."""
        while self._running:
            try:
                # Check if it's time for daily retraining
                if self._should_retrain_daily():
                    logger.info("Triggering daily retraining")
                    self.trigger_retraining(reason="scheduled_daily")
                
                # Sleep 1 hour and check again
                time.sleep(3600)
            except Exception as e:
                logger.error(f"Error in retraining scheduler loop: {e}", exc_info=True)

    def _should_retrain_daily(self) -> bool:
        """Check if it's time for daily retraining."""
        now = datetime.now(timezone.utc)
        
        # Already trained today?
        if self._last_training_time:
            if self._last_training_time.date() == now.date():
                return False  # Already trained today
        
        # Is it the scheduled hour?
        if now.hour == self.schedule_hour and now.minute < 5:
            return True
        
        return False

    def trigger_retraining(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Trigger model retraining.

        Args:
            reason: Reason for retraining (scheduled_daily, drift_detected, manual, feedback)

        Returns:
            Training results dictionary
        """
        if self._training_in_progress:
            logger.warning("Training already in progress")
            return {"error": "Training already in progress"}

        with self._lock:
            self._training_in_progress = True

        try:
            logger.info(f"Starting model retraining - reason: {reason}")
            
            # Get training data
            training_data = self.dataset_collector.export_for_training()
            if not training_data.get('X'):
                logger.warning("No training data available")
                return {"error": "No training data available", "reason": reason}
            
            X = training_data['X']
            y = training_data['y']
            
            # Convert to DataFrame if needed
            if isinstance(X, list) and X:
                df_X = pd.DataFrame(X)
            else:
                df_X = X
            
            # Train new model
            logger.info(f"Training on {len(X)} samples")
            old_model = self.ml_engine.model
            
            try:
                # Train new model
                self.ml_engine.train(df_X, y)
                new_model = self.ml_engine.model
                
                # Evaluate new model
                metrics_new = self._evaluate_model(new_model, df_X, y)
                metrics_old = self._evaluate_model(old_model, df_X, y) if old_model else {}
                
                # Create model version
                version = self.model_registry.create_version(
                    model_type="random_forest",
                    metrics=metrics_new,
                    training_samples=len(X),
                    training_date=datetime.now(timezone.utc).isoformat(),
                    reason=reason
                )
                
                result = {
                    "status": "success",
                    "version": version,
                    "reason": reason,
                    "samples_trained": len(X),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": {
                        "new": metrics_new,
                        "old": metrics_old,
                        "improvement": self._calculate_improvement(metrics_old, metrics_new)
                    }
                }
                
                self._last_training_time = datetime.now(timezone.utc)
                self._training_history.append(result)
                
                logger.info(f"Retraining completed - version: {version}")
                logger.info(f"New model metrics: {metrics_new}")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to train model: {e}", exc_info=True)
                return {"error": f"Training failed: {str(e)}", "reason": reason}
        
        finally:
            self._training_in_progress = False

    def _evaluate_model(self, model, X, y) -> Dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dictionary with accuracy, precision, recall, F1 scores
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            predictions = model.predict(X)
            
            return {
                "accuracy": float(accuracy_score(y, predictions)),
                "precision": float(precision_score(y, predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(y, predictions, average="weighted", zero_division=0)),
                "f1": float(f1_score(y, predictions, average="weighted", zero_division=0))
            }
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            return {}


    def _calculate_improvement(
        self,
        metrics_old: Dict[str, float],
        metrics_new: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement from old to new model."""
        if not metrics_old:
            return {}
        
        improvement = {}
        for key in metrics_new:
            if key in metrics_old:
                old_val = metrics_old[key]
                new_val = metrics_new[key]
                pct_change = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                improvement[key] = round(pct_change, 2)
        
        return improvement

    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return {
            "training_in_progress": self._training_in_progress,
            "last_training_time": self._last_training_time.isoformat() if self._last_training_time else None,
            "next_scheduled_training": self._get_next_scheduled_time().isoformat(),
            "scheduler_running": self._running
        }

    def get_training_history(self, limit: int = 10) -> list:
        """Get recent training history."""
        return self._training_history[-limit:]

    def _get_next_scheduled_time(self) -> datetime:
        """Calculate next scheduled training time."""
        now = datetime.now(timezone.utc)
        next_time = now.replace(hour=self.schedule_hour, minute=0, second=0, microsecond=0)
        
        # If scheduled time has passed today, schedule for tomorrow
        if next_time <= now:
            next_time = next_time + timedelta(days=1)
        
        return next_time

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = self.dataset_collector.get_stats()
        stats.update({
            "training_in_progress": self._training_in_progress,
            "last_training_time": self._last_training_time.isoformat() if self._last_training_time else None,
            "next_scheduled_training": self._get_next_scheduled_time().isoformat(),
            "training_count": len(self._training_history),
            "scheduler_running": self._running
        })
        return stats


# Backward-compatible alias for older imports/typos used across earlier phases.
RertrainingScheduler = RetrainingScheduler
