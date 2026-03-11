from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class MetricsService:
    """In-memory counters, gauges, and histograms exposed in Prometheus text format."""

    def __init__(self):
        self._counters = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def inc(self, key: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        with self._lock:
            self._counters[key] += amount

    def get(self, key: str) -> int:
        with self._lock:
            return self._counters.get(key, 0)

    # ------------------------------------------------------------------
    # Gauges
    # ------------------------------------------------------------------

    def set_gauge(self, key: str, value: float) -> None:
        with self._lock:
            self._gauges[key] = value

    def get_gauge(self, key: str) -> float:
        with self._lock:
            return self._gauges.get(key, 0.0)

    # ------------------------------------------------------------------
    # Histograms (observation-based)
    # ------------------------------------------------------------------

    def observe(self, key: str, value: float) -> None:
        with self._lock:
            bucket = self._histograms[key]
            bucket.append(value)
            if len(bucket) > 10_000:
                self._histograms[key] = bucket[-5_000:]

    def observe_risk_score(self, score: float) -> None:
        self.observe("risk_score", score)
        with self._lock:
            self._counters["risk_score_count"] += 1
            self._counters["risk_score_sum"] += float(score)

    def observe_latency(self, key: str, start_time: float) -> None:
        """Record elapsed time since ``start_time`` (seconds via time.monotonic)."""
        elapsed = time.monotonic() - start_time
        self.observe(key, elapsed)

    # ------------------------------------------------------------------
    # Prometheus export
    # ------------------------------------------------------------------

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
                "# HELP inids_detection_events_total Total detection events published",
                "# TYPE inids_detection_events_total counter",
                f"inids_detection_events_total {self._counters.get('detection_events_total', 0)}",
                "# HELP inids_action_events_total Total action events published",
                "# TYPE inids_action_events_total counter",
                f"inids_action_events_total {self._counters.get('action_events_total', 0)}",
                "# HELP inids_risk_score_sum Sum of risk scores",
                "# TYPE inids_risk_score_sum counter",
                f"inids_risk_score_sum {self._counters.get('risk_score_sum', 0)}",
                "# HELP inids_risk_score_count Number of risk-score observations",
                "# TYPE inids_risk_score_count counter",
                f"inids_risk_score_count {self._counters.get('risk_score_count', 0)}",
                "# HELP inids_engine_evaluations_total Multi-engine evaluations",
                "# TYPE inids_engine_evaluations_total counter",
                f"inids_engine_evaluations_total {self._counters.get('engine_evaluations_total', 0)}",
                "# HELP inids_engine_attacks_total Multi-engine attack verdicts",
                "# TYPE inids_engine_attacks_total counter",
                f"inids_engine_attacks_total {self._counters.get('engine_attacks_total', 0)}",
            ]

            # Append gauge lines
            for gk, gv in sorted(self._gauges.items()):
                lines.append(f"# TYPE inids_{gk} gauge")
                lines.append(f"inids_{gk} {gv}")
        return "\n".join(lines) + "\n"
