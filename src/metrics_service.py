from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsService:
    """In-memory counters exposed in Prometheus text format."""

    def __init__(self):
        self._counters = defaultdict(int)
        self._lock = Lock()

    def inc(self, key: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        with self._lock:
            self._counters[key] += amount

    def get(self, key: str) -> int:
        with self._lock:
            return self._counters.get(key, 0)

    def as_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP inids_requests_total Total API requests processed by INIDS",
                "# TYPE inids_requests_total counter",
                f"inids_requests_total {self._counters.get('requests_total', 0)}",
                "# HELP inids_predictions_total Total prediction API calls",
                "# TYPE inids_predictions_total counter",
                f"inids_predictions_total {self._counters.get('predictions_total', 0)}",
                "# HELP inids_alerts_total Total alerts generated",
                "# TYPE inids_alerts_total counter",
                f"inids_alerts_total {self._counters.get('alerts_total', 0)}",
                "# HELP inids_prevention_actions_total Total prevention actions generated",
                "# TYPE inids_prevention_actions_total counter",
                f"inids_prevention_actions_total {self._counters.get('prevention_actions_total', 0)}",
                "# HELP inids_policy_updates_total Total policy updates",
                "# TYPE inids_policy_updates_total counter",
                f"inids_policy_updates_total {self._counters.get('policy_updates_total', 0)}",
                "# HELP inids_unauthorized_total Unauthorized API access attempts",
                "# TYPE inids_unauthorized_total counter",
                f"inids_unauthorized_total {self._counters.get('unauthorized_total', 0)}",
                "# HELP inids_ingested_total Total records accepted into ingestion queue",
                "# TYPE inids_ingested_total counter",
                f"inids_ingested_total {self._counters.get('ingested_total', 0)}",
                "# HELP inids_processed_ingestion_total Total ingestion records processed",
                "# TYPE inids_processed_ingestion_total counter",
                f"inids_processed_ingestion_total {self._counters.get('processed_ingestion_total', 0)}",
                "# HELP inids_expired_actions_cleaned_total Total expired actions removed by cleanup",
                "# TYPE inids_expired_actions_cleaned_total counter",
                f"inids_expired_actions_cleaned_total {self._counters.get('expired_actions_cleaned_total', 0)}",
                "# HELP inids_rate_limited_total Requests rejected by rate limiter",
                "# TYPE inids_rate_limited_total counter",
                f"inids_rate_limited_total {self._counters.get('rate_limited_total', 0)}",
            ]
        return "\n".join(lines) + "\n"
