"""
Phase F Part 5: Machine Learning & Anomaly Detection

Advanced ML-based anomaly detection:
- Statistical baseline learning per host
- Behavioral profiling (connection patterns)
- Ensemble classification
- Custom rule engine
- Anomaly scoring and alerting

Thread-safe with global singleton pattern.
"""

import time
import threading
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set, Callable
from logging import getLogger
from collections import OrderedDict, defaultdict
from statistics import mean, stdev, StatisticsError
import json


logger = getLogger(__name__)

# Anomaly scoring thresholds
ANOMALY_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.95,
}

# Statistical baseline windows
BASELINE_WINDOW = 3600  # 1 hour
LEARNING_PERIOD = 86400  # 24 hours
MIN_SAMPLES = 10


@dataclass
class HostProfile:
    """Behavioral profile for a host."""
    host_ip: str
    connection_count: int = 0
    avg_bytes_sent: float = 0.0
    avg_bytes_received: float = 0.0
    unique_destinations: Set[str] = field(default_factory=set)
    protocols_used: Set[str] = field(default_factory=set)
    peak_hour: int = -1
    last_seen: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    learning_complete: bool = False
    samples: List[float] = field(default_factory=list)

    def to_dict(self):
        return {
            "host_ip": self.host_ip,
            "connection_count": self.connection_count,
            "avg_bytes_sent": self.avg_bytes_sent,
            "avg_bytes_received": self.avg_bytes_received,
            "unique_destinations": list(self.unique_destinations),
            "protocols_used": list(self.protocols_used),
            "peak_hour": self.peak_hour,
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "learning_complete": self.learning_complete,
            "sample_count": len(self.samples),
        }


@dataclass
class AnomalyScore:
    """Anomaly detection result."""
    score: float  # 0-1.0
    severity: str  # low, medium, high, critical
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class MLDetectionResult:
    """Result of ML anomaly detection."""
    is_anomalous: bool = False
    anomaly_score: float = 0.0  # 0-1.0
    severity: str = "none"  # none, low, medium, high, critical
    reasons: List[str] = field(default_factory=list)
    model_confidence: float = 0.0
    ensemble_votes: Dict[str, bool] = field(default_factory=dict)
    host_profile_match: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class MLStats:
    """ML anomaly detection statistics."""
    total_analyzed: int = 0
    anomalies_detected: int = 0
    low_severity: int = 0
    medium_severity: int = 0
    high_severity: int = 0
    critical_severity: int = 0
    hosts_profiled: int = 0
    learning_complete: int = 0
    last_analysis_time: float = 0.0


class StatisticalBaseline:
    """Statistical baseline for anomaly detection."""

    def __init__(self, window_size: int = 3600, min_samples: int = 10):
        self.window_size = window_size
        self.min_samples = min_samples
        self.baselines = {}
        self.lock = threading.RLock()

    def add_sample(self, key: str, value: float):
        """Add sample to baseline."""
        with self.lock:
            if key not in self.baselines:
                self.baselines[key] = {
                    "samples": [],
                    "mean": 0.0,
                    "stdev": 0.0,
                    "last_update": time.time(),
                }

            baseline = self.baselines[key]
            baseline["samples"].append((time.time(), value))

            # Remove old samples outside window
            cutoff = time.time() - self.window_size
            baseline["samples"] = [
                (t, v) for t, v in baseline["samples"] if t > cutoff
            ]

            # Update statistics
            if len(baseline["samples"]) >= self.min_samples:
                values = [v for t, v in baseline["samples"]]
                baseline["mean"] = mean(values)
                try:
                    baseline["stdev"] = stdev(values)
                except StatisticsError:
                    baseline["stdev"] = 0.0
                baseline["last_update"] = time.time()

    def get_baseline(self, key: str) -> Tuple[float, float, bool]:
        """Get baseline (mean, stdev, is_ready)."""
        with self.lock:
            if key not in self.baselines:
                return 0.0, 0.0, False

            baseline = self.baselines[key]
            is_ready = len(baseline["samples"]) >= self.min_samples
            return baseline["mean"], baseline["stdev"], is_ready

    def is_anomalous(self, key: str, value: float, sigma: float = 3.0) -> bool:
        """Check if value is anomalous (statistical outlier)."""
        mean_val, stdev_val, is_ready = self.get_baseline(key)
        if not is_ready or stdev_val == 0:
            return False

        # Z-score calculation
        z_score = abs((value - mean_val) / stdev_val)
        return z_score > sigma


class BehavioralProfiler:
    """Profiles host behavior over time."""

    def __init__(self, learning_period: int = 86400):
        self.learning_period = learning_period
        self.profiles = {}
        self.lock = threading.RLock()

    def update_profile(self, host_ip: str, **kwargs):
        """Update or create host profile."""
        with self.lock:
            if host_ip not in self.profiles:
                self.profiles[host_ip] = HostProfile(host_ip=host_ip)

            profile = self.profiles[host_ip]

            # Update basic fields
            if "connection_count" in kwargs:
                profile.connection_count += kwargs["connection_count"]
            if "bytes_sent" in kwargs:
                profile.samples.append(kwargs["bytes_sent"])
                if len(profile.samples) > 0:
                    profile.avg_bytes_sent = mean(profile.samples)
            if "destination" in kwargs:
                profile.unique_destinations.add(kwargs["destination"])
            if "protocol" in kwargs:
                profile.protocols_used.add(kwargs["protocol"])

            profile.last_seen = time.time()

            # Check if learning period is complete
            age = profile.last_seen - profile.first_seen
            if not profile.learning_complete and age > self.learning_period:
                profile.learning_complete = True

            return profile

    def get_profile(self, host_ip: str) -> Optional[HostProfile]:
        """Get host profile."""
        with self.lock:
            profile = self.profiles.get(host_ip)
            if profile and not profile.learning_complete:
                profile.learning_complete = (time.time() - profile.first_seen) >= self.learning_period
            return profile

    def get_all_profiles(self) -> Dict[str, HostProfile]:
        """Get all profiles."""
        with self.lock:
            return dict(self.profiles)

    def compare_to_baseline(self, host_ip: str, test_metrics: Dict) -> float:
        """Compare current metrics to baseline. Returns 0-1.0 match score."""
        profile = self.get_profile(host_ip)
        if not profile:
            return 0.5  # Unknown profile

        similarity = 0.0
        weight = 0.0

        # Compare connection count
        if "connection_count" in test_metrics and profile.connection_count > 0:
            ratio = test_metrics["connection_count"] / (profile.connection_count + 1)
            if ratio < 5:  # Within 5x
                similarity += 0.25
            weight += 0.25

        # Compare protocols
        if "protocol" in test_metrics and profile.protocols_used:
            if test_metrics["protocol"] in profile.protocols_used:
                similarity += 0.25
            weight += 0.25

        # Compare destination count
        if "destination" in test_metrics and profile.unique_destinations:
            similarity += 0.25
            weight += 0.25

        # Compare learning status
        if profile.learning_complete:
            similarity += 0.25
            weight += 0.25

        return similarity / max(weight, 0.01)


class EnsembleClassifier:
    """Ensemble of multiple anomaly detectors."""

    def __init__(self):
        self.classifiers = {}
        self.weights = {}
        self.lock = threading.RLock()

    def add_classifier(
        self, name: str, classifier_func: Callable, weight: float = 1.0
    ):
        """Add classifier to ensemble."""
        with self.lock:
            self.classifiers[name] = classifier_func
            self.weights[name] = weight

    def classify(self, **kwargs) -> Tuple[bool, Dict[str, bool], float]:
        """Classify using ensemble voting. Returns (is_anomalous, votes, confidence)."""
        with self.lock:
            votes = {}
            confidence_sum = 0.0

            for name, classifier in self.classifiers.items():
                try:
                    is_anomalous = classifier(**kwargs)
                    votes[name] = is_anomalous
                    if is_anomalous:
                        confidence_sum += self.weights.get(name, 1.0)
                except Exception as e:
                    logger.debug(f"Classifier {name} error: {e}")
                    votes[name] = False

            # Calculate confidence
            total_weight = sum(self.weights.values())
            confidence = confidence_sum / max(total_weight, 0.01) if votes else 0.0
            is_anomalous = confidence > 0.0

            return is_anomalous, votes, confidence


class CustomRuleEngine:
    """Custom rule engine for anomaly detection."""

    def __init__(self):
        self.rules = []
        self.lock = threading.RLock()

    def add_rule(
        self,
        name: str,
        condition: Callable,
        severity: str = "medium",
    ):
        """Add custom rule."""
        with self.lock:
            self.rules.append({"name": name, "condition": condition, "severity": severity})

    def evaluate_rules(self, **context) -> List[Tuple[str, str]]:
        """Evaluate all rules. Returns list of (rule_name, severity)."""
        violations = []

        with self.lock:
            for rule in self.rules:
                try:
                    if rule["condition"](**context):
                        violations.append((rule["name"], rule["severity"]))
                except Exception as e:
                    logger.debug(f"Rule {rule['name']} error: {e}")

        return violations


class MLAnomalyDetector:
    """Main ML anomaly detection engine."""

    def __init__(self, learning_period: int = 86400, cache_size: int = 10000):
        self.baseline = StatisticalBaseline()
        self.profiler = BehavioralProfiler(learning_period)
        self.ensemble = EnsembleClassifier()
        self.rules = CustomRuleEngine()
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.stats = MLStats()
        self.lock = threading.RLock()

        # Setup default classifiers
        self._setup_default_classifiers()

    def _setup_default_classifiers(self):
        """Setup default classifiers for ensemble."""
        # Classifier 1: Connection rate anomaly
        def conn_rate_classifier(
            host_ip: str = "", connection_count: int = 0, **kwargs
        ) -> bool:
            key = f"{host_ip}_conn_rate"
            self.baseline.add_sample(key, connection_count)
            return self.baseline.is_anomalous(key, connection_count, sigma=2.5)

        self.ensemble.add_classifier("connection_rate", conn_rate_classifier, weight=1.5)

        # Classifier 2: Data volume anomaly
        def data_volume_classifier(
            host_ip: str = "", bytes_sent: float = 0, **kwargs
        ) -> bool:
            key = f"{host_ip}_data_volume"
            self.baseline.add_sample(key, bytes_sent)
            return self.baseline.is_anomalous(key, bytes_sent, sigma=2.0)

        self.ensemble.add_classifier("data_volume", data_volume_classifier, weight=1.0)

        # Classifier 3: Protocol anomaly
        def protocol_classifier(host_ip: str = "", protocol: str = "", **kwargs) -> bool:
            profile = self.profiler.get_profile(host_ip)
            if not profile or not profile.learning_complete:
                return False
            # Protocols not in baseline = anomaly
            return (
                profile.protocols_used
                and protocol not in profile.protocols_used
                and len(profile.protocols_used) > 0
            )

        self.ensemble.add_classifier("protocol_usage", protocol_classifier, weight=1.2)

    def analyze(
        self,
        host_ip: str,
        connection_count: int = 0,
        bytes_sent: float = 0.0,
        bytes_received: float = 0.0,
        destination: str = "",
        protocol: str = "",
    ) -> MLDetectionResult:
        """Analyze traffic for anomalies."""

        # Update profile
        self.profiler.update_profile(
            host_ip,
            connection_count=connection_count,
            bytes_sent=bytes_sent,
            destination=destination,
            protocol=protocol,
        )

        result = MLDetectionResult()

        with self.lock:
            # Ensemble classification
            is_anomalous, votes, confidence = self.ensemble.classify(
                host_ip=host_ip,
                connection_count=connection_count,
                bytes_sent=bytes_sent,
                destination=destination,
                protocol=protocol,
            )

            result.ensemble_votes = votes
            result.model_confidence = confidence

            # Custom rules
            violations = self.rules.evaluate_rules(
                host_ip=host_ip,
                connection_count=connection_count,
                bytes_sent=bytes_sent,
                destination=destination,
                protocol=protocol,
            )

            # Behavioral profile matching
            profile = self.profiler.get_profile(host_ip)
            if profile:
                result.host_profile_match = self.profiler.compare_to_baseline(
                    host_ip,
                    {
                        "connection_count": connection_count,
                        "destination": destination,
                        "protocol": protocol,
                    },
                )

            # Calculate anomaly score
            score_components = []

            # Ensemble score
            score_components.append(confidence * 0.9)

            # Rule violations
            if violations:
                max_severity = max(
                    (ANOMALY_THRESHOLDS.get(sev, 0.5) for _, sev in violations),
                    default=0.5,
                )
                score_components.append(max_severity * 0.3)
                result.reasons.extend([name for name, _ in violations])

            # Profile mismatch
            if profile and profile.learning_complete:
                mismatch = max(0, 1.0 - result.host_profile_match)
                score_components.append(mismatch * 0.3)
                if mismatch > 0.4:
                    result.reasons.append(f"Profile mismatch: {mismatch:.2f}")

            result.anomaly_score = min(sum(score_components), 1.0) if score_components else 0.0

            # Determine severity
            if result.anomaly_score >= ANOMALY_THRESHOLDS["critical"]:
                result.severity = "critical"
            elif result.anomaly_score >= ANOMALY_THRESHOLDS["high"]:
                result.severity = "high"
            elif result.anomaly_score >= ANOMALY_THRESHOLDS["medium"]:
                result.severity = "medium"
            elif result.anomaly_score >= ANOMALY_THRESHOLDS["low"]:
                result.severity = "low"
            else:
                result.severity = "none"

            result.is_anomalous = result.anomaly_score > 0.5

            # Update stats
            self.stats.total_analyzed += 1
            if result.is_anomalous:
                self.stats.anomalies_detected += 1
                if result.severity == "critical":
                    self.stats.critical_severity += 1
                elif result.severity == "high":
                    self.stats.high_severity += 1
                elif result.severity == "medium":
                    self.stats.medium_severity += 1
                elif result.severity == "low":
                    self.stats.low_severity += 1

            self.stats.hosts_profiled = len(self.profiler.profiles)
            self.stats.learning_complete = sum(
                1
                for p in self.profiler.profiles.values()
                if p.learning_complete
            )
            self.stats.last_analysis_time = time.time()

        return result

    def add_custom_rule(
        self, name: str, condition: Callable, severity: str = "medium"
    ):
        """Add custom detection rule."""
        self.rules.add_rule(name, condition, severity)

    def get_profile(self, host_ip: str) -> Optional[Dict]:
        """Get host profile as dict."""
        profile = self.profiler.get_profile(host_ip)
        return profile.to_dict() if profile else None

    def get_stats(self) -> MLStats:
        """Get analysis statistics."""
        with self.lock:
            return MLStats(
                total_analyzed=self.stats.total_analyzed,
                anomalies_detected=self.stats.anomalies_detected,
                low_severity=self.stats.low_severity,
                medium_severity=self.stats.medium_severity,
                high_severity=self.stats.high_severity,
                critical_severity=self.stats.critical_severity,
                hosts_profiled=self.stats.hosts_profiled,
                learning_complete=self.stats.learning_complete,
                last_analysis_time=self.stats.last_analysis_time,
            )


# Global singleton
_ml_detector = None
_ml_detector_lock = threading.RLock()


def get_ml_detector() -> MLAnomalyDetector:
    """Get global ML detector instance."""
    global _ml_detector
    if _ml_detector is None:
        with _ml_detector_lock:
            if _ml_detector is None:
                _ml_detector = MLAnomalyDetector()
    return _ml_detector


def init_ml_detector(learning_period: int = 86400) -> MLAnomalyDetector:
    """Initialize ML detector with custom settings."""
    global _ml_detector
    with _ml_detector_lock:
        _ml_detector = MLAnomalyDetector(learning_period=learning_period)
    return _ml_detector


def enrich_eve_event_with_ml(
    eve_event: Dict, ml_result: MLDetectionResult
) -> Dict:
    """Enrich EVE JSON event with ML analysis."""
    if "ml_analysis" not in eve_event:
        eve_event["ml_analysis"] = {}

    eve_event["ml_analysis"]["anomaly_score"] = ml_result.anomaly_score
    eve_event["ml_analysis"]["severity"] = ml_result.severity
    eve_event["ml_analysis"]["is_anomalous"] = ml_result.is_anomalous
    eve_event["ml_analysis"]["model_confidence"] = ml_result.model_confidence
    eve_event["ml_analysis"]["ensemble_votes"] = ml_result.ensemble_votes
    eve_event["ml_analysis"]["host_profile_match"] = ml_result.host_profile_match
    eve_event["ml_analysis"]["reasons"] = ml_result.reasons

    return eve_event
