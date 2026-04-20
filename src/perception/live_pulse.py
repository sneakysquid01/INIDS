"""
Live System Pulse - Real-time animated metrics dashboard.

Provides a heartbeat-like visualization of system health showing
flows, alerts, model accuracy, and threat levels in real-time.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from collections import deque
from threading import RLock

logger = logging.getLogger(__name__)


class LiveSystemPulse:
    """
    Maintains and broadcasts real-time metrics in a time-series format.

    Tracks: flow rate, alert rate, model accuracy, threat level, etc.
    over a rolling window (last 1 hour, 1 min granularity).
    """

    def __init__(self, window_minutes: int = 60):
        """
        Initialize the Live System Pulse.

        Args:
            window_minutes: How many minutes of historical data to keep
        """
        self.lock = RLock()
        self.window_minutes = window_minutes

        # Time-series buckets (one entry per minute)
        self.flows_per_second = deque(maxlen=window_minutes)
        self.alerts_per_minute = deque(maxlen=window_minutes)
        self.blocked_ips = deque(maxlen=window_minutes)
        self.model_accuracy = deque(maxlen=window_minutes)
        self.system_health = deque(maxlen=window_minutes)  # 0-100
        self.threat_level = deque(maxlen=window_minutes)  # 0-100

        # Current values
        self.current_flows_per_second = 0
        self.current_alerts_per_minute = 0
        self.current_blocked_ips = 0
        self.current_model_accuracy = 95.0
        self.current_threat_level = 10

        logger.info(f"LiveSystemPulse initialized with {window_minutes}min window")

    def record_flow_count(self, flow_count: int) -> None:
        """Record number of flows processed in this second."""
        with self.lock:
            self.current_flows_per_second = flow_count
            self.flows_per_second.append(flow_count)

    def record_alert(self) -> None:
        """Record that an alert was triggered."""
        with self.lock:
            self.current_alerts_per_minute += 1

    def reset_minute(self) -> None:
        """Called at end of each minute to save current metrics and reset."""
        with self.lock:
            # Save current minute's data
            self.alerts_per_minute.append(self.current_alerts_per_minute)

            # Calculate health score (inverse of threat)
            health = max(0, 100 - self.current_threat_level)
            self.system_health.append(health)

            # Add threat level snapshot
            self.threat_level.append(self.current_threat_level)

            # Add model accuracy snapshot
            self.model_accuracy.append(self.current_model_accuracy)

            # Add blocked IPs snapshot
            self.blocked_ips.append(self.current_blocked_ips)

            # Reset counters
            self.current_alerts_per_minute = 0

    def update_model_accuracy(self, accuracy: float) -> None:
        """Update the current model accuracy percentage."""
        with self.lock:
            self.current_model_accuracy = max(0, min(100, accuracy))

    def update_threat_level(self, threat_level: float) -> None:
        """Update the current threat level (0-100)."""
        with self.lock:
            self.current_threat_level = max(0, min(100, threat_level))

    def update_blocked_ips(self, count: int) -> None:
        """Update the number of currently blocked IPs."""
        with self.lock:
            self.current_blocked_ips = count

    def get_pulse_status(self) -> Dict[str, Any]:
        """
        Get current pulse status - snapshot of current metrics.

        Returns data suitable for dashboard real-time display.
        """
        with self.lock:
            # Calculate rolling averages
            avg_flows = sum(self.flows_per_second) / len(self.flows_per_second) if self.flows_per_second else 0
            avg_alerts = sum(self.alerts_per_minute) / len(self.alerts_per_minute) if self.alerts_per_minute else 0
            avg_health = sum(self.system_health) / len(self.system_health) if self.system_health else 100
            avg_threat = sum(self.threat_level) / len(self.threat_level) if self.threat_level else 0

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current": {
                    "flows_per_second": self.current_flows_per_second,
                    "alerts_per_minute": self.current_alerts_per_minute,
                    "blocked_ips": self.current_blocked_ips,
                    "model_accuracy_percent": self.current_model_accuracy,
                    "threat_level_percent": self.current_threat_level,
                    "health_percent": max(0, 100 - self.current_threat_level)
                },
                "rolling_averages": {
                    "avg_flows_per_second": round(avg_flows, 2),
                    "avg_alerts_per_minute": round(avg_alerts, 2),
                    "avg_threat_level": round(avg_threat, 2),
                    "avg_health": round(avg_health, 2),
                },
                "status": self._determine_status(self.current_threat_level),
                "pulse_strength": self._calculate_pulse_strength(avg_alerts),
            }

    def get_time_series(self, metric: str) -> List[Dict[str, Any]]:
        """
        Get time-series data for a specific metric.

        Args:
            metric: One of "flows", "alerts", "accuracy", "threat", "health"

        Returns:
            List of [timestamp, value] pairs
        """
        with self.lock:
            now = datetime.now(timezone.utc)
            series = []

            if metric == "flows":
                source = self.flows_per_second
            elif metric == "alerts":
                source = self.alerts_per_minute
            elif metric == "accuracy":
                source = self.model_accuracy
            elif metric == "threat":
                source = self.threat_level
            elif metric == "health":
                source = self.system_health
            else:
                return []

            # Generate timestamps going backwards
            for i, value in enumerate(reversed(source)):
                timestamp = now - timedelta(minutes=len(source) - 1 - i)
                series.append({
                    "timestamp": timestamp.isoformat(),
                    "value": value
                })

            return series

    def _determine_status(self, threat_level: float) -> str:
        """Determine overall system status from threat level."""
        if threat_level < 20:
            return "SAFE"
        elif threat_level < 50:
            return "SUSPICIOUS"
        elif threat_level < 80:
            return "WARNING"
        else:
            return "CRITICAL"

    def _calculate_pulse_strength(self, alert_rate: float) -> float:
        """
        Calculate 'pulse strength' - how active the system is.

        Returns 0-1 score representing system activity level.
        """
        # Map alerts to pulse (0 alerts = 0, >10 alerts/min = 1.0)
        return min(1.0, alert_rate / 10.0)

    def get_alert_heatmap(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Generate alert heatmap data for hourly visualization.

        Returns list of hour buckets with alert counts.
        """
        with self.lock:
            # In production, this would query actual hourly aggregates
            # For now, use current data to simulate
            heatmap = []
            now = datetime.now(timezone.utc)

            for hour_offset in range(hours):
                hour_time = now - timedelta(hours=hours - hour_offset)
                # Simulate data based on threat level
                alert_count = int(self.current_alerts_per_minute * (0.5 + 0.5 * (hour_offset / hours)))

                heatmap.append({
                    "hour": hour_time.strftime("%H:%M"),
                    "hour_datetime": hour_time.isoformat(),
                    "alert_count": alert_count,
                    "intensity": min(1.0, alert_count / 10.0)  # 0-1 intensity
                })

            return heatmap

    def get_activity_sparkline(self, metric: str = "alerts", points: int = 20) -> List[float]:
        """
        Get sparkline data (simplified time series).

        Useful for small inline charts.
        """
        with self.lock:
            series = self.get_time_series(metric)

            if len(series) <= points:
                return [item["value"] for item in series]

            # Downsample to requested number of points
            step = len(series) // points
            return [series[i * step]["value"] for i in range(points)]
