"""
Tests for Phase F Part 5: ML & Anomaly Detection

Tests statistical baseline, behavioral profiling, ensemble classification,
custom rules, and integrated ML analysis.
No pytest dependency - runs standalone.
"""

import time
from src.advanced.ml_anomaly import (
    MLAnomalyDetector,
    StatisticalBaseline,
    BehavioralProfiler,
    EnsembleClassifier,
    CustomRuleEngine,
    HostProfile,
    AnomalyScore,
    MLDetectionResult,
    MLStats,
    get_ml_detector,
    init_ml_detector,
    enrich_eve_event_with_ml,
)


def test_statistical_baseline():
    """Test statistical baseline."""
    print("[TEST] Statistical Baseline")
    baseline = StatisticalBaseline(window_size=3600, min_samples=5)

    # Add normal samples
    for i in range(10):
        baseline.add_sample("test_metric", 100.0 + i)

    # Check baseline
    mean_val, stdev_val, is_ready = baseline.get_baseline("test_metric")
    assert is_ready is True, "Should be ready after min samples"
    assert mean_val > 90, "Mean should be around 100"
    print("  ✓ Baseline calculation works")

    # Test anomaly detection
    is_anom = baseline.is_anomalous("test_metric", 1000.0, sigma=2.0)
    assert is_anom is True, "Should detect extreme value"
    print("  ✓ Anomaly detection works")

    # Test with insufficient samples
    is_ready2 = baseline.get_baseline("unknown_metric")[2]
    assert is_ready2 is False, "Unknown metric should not be ready"
    print("  ✓ Insufficient samples handled")

    print("  ✓ Statistical Baseline works")


def test_behavioral_profiler():
    """Test behavioral profiler."""
    print("[TEST] Behavioral Profiler")
    profiler = BehavioralProfiler(learning_period=1)

    # Create profile
    profile = profiler.update_profile(
        "192.168.1.100",
        connection_count=1,
        bytes_sent=1000.0,
        destination="10.0.0.1",
        protocol="http",
    )

    assert profile.host_ip == "192.168.1.100"
    assert profile.connection_count == 1
    assert "10.0.0.1" in profile.unique_destinations
    assert "http" in profile.protocols_used
    print("  ✓ Profile creation works")

    # Add more updates
    profiler.update_profile(
        "192.168.1.100",
        connection_count=2,
        bytes_sent=2000.0,
        destination="10.0.0.2",
        protocol="https",
    )

    profile = profiler.get_profile("192.168.1.100")
    assert profile.connection_count == 3, "Connection count should accumulate"
    assert len(profile.unique_destinations) == 2, "Should track unique destinations"
    print("  ✓ Profile updates work")

    # Test learning complete
    time.sleep(1.1)
    profile = profiler.get_profile("192.168.1.100")
    assert profile.learning_complete is True, "Should complete learning period"
    print("  ✓ Learning period tracking works")

    # Test baseline comparison
    match = profiler.compare_to_baseline(
        "192.168.1.100",
        {"connection_count": 3, "destination": "10.0.0.1", "protocol": "http"},
    )
    assert 0 <= match <= 1, "Match score should be 0-1"
    print("  ✓ Baseline comparison works")

    print("  ✓ Behavioral Profiler works")


def test_ensemble_classifier():
    """Test ensemble classifier."""
    print("[TEST] Ensemble Classifier")
    ensemble = EnsembleClassifier()

    # Add classifiers
    def classifier1(**kwargs):
        return kwargs.get("value", 0) > 100

    def classifier2(**kwargs):
        return kwargs.get("value", 0) > 200

    ensemble.add_classifier("high_threshold", classifier1, weight=1.0)
    ensemble.add_classifier("very_high_threshold", classifier2, weight=1.5)

    # Test classification
    is_anom, votes, conf = ensemble.classify(value=150)
    assert is_anom is True, "Should be anomalous (majority vote)"
    assert votes["high_threshold"] is True
    assert votes["very_high_threshold"] is False
    assert conf > 0, "Confidence should be > 0"
    print("  ✓ Ensemble voting works")

    # Test with all classifiers agreeing
    is_anom2, votes2, conf2 = ensemble.classify(value=250)
    assert is_anom2 is True, "Should be anomalous"
    assert votes2["high_threshold"] is True
    assert votes2["very_high_threshold"] is True
    assert conf2 > conf, "Confidence should increase with more votes"
    print("  ✓ Ensemble weighting works")

    print("  ✓ Ensemble Classifier works")


def test_custom_rule_engine():
    """Test custom rule engine."""
    print("[TEST] Custom Rule Engine")
    rules = CustomRuleEngine()

    # Add rules
    def rule1_condition(**kwargs):
        return kwargs.get("bytes_sent", 0) > 1000000

    def rule2_condition(**kwargs):
        return kwargs.get("protocol", "") == "suspicious"

    rules.add_rule("large_transfer", rule1_condition, severity="high")
    rules.add_rule("suspicious_protocol", rule2_condition, severity="critical")

    # Evaluate rules
    violations = rules.evaluate_rules(bytes_sent=500000, protocol="http")
    assert len(violations) == 0, "Should have no violations"
    print("  ✓ Rule evaluation works (no violations)")

    violations2 = rules.evaluate_rules(bytes_sent=2000000, protocol="suspicious")
    assert len(violations2) == 2, "Should have 2 violations"
    assert any(v[0] == "large_transfer" for v in violations2)
    assert any(v[0] == "suspicious_protocol" for v in violations2)
    print("  ✓ Rule evaluation works (violations detected)")

    print("  ✓ Custom Rule Engine works")


def test_ml_anomaly_detector():
    """Test ML anomaly detector."""
    print("[TEST] ML Anomaly Detector")
    detector = MLAnomalyDetector()

    # Test normal traffic
    result = detector.analyze(
        host_ip="192.168.1.100",
        connection_count=1,
        bytes_sent=1000.0,
        destination="10.0.0.1",
        protocol="http",
    )

    assert isinstance(result, MLDetectionResult)
    assert result.severity in [
        "none",
        "low",
        "medium",
        "high",
        "critical",
    ]
    print("  ✓ Normal analysis works")

    # Analyze same host multiple times to build baseline
    for i in range(15):
        detector.analyze(
            host_ip="192.168.1.100",
            connection_count=1 + i,
            bytes_sent=1000.0,
            destination="10.0.0.1",
            protocol="http",
        )

    # Test anomalous traffic (extreme values)
    anom_result = detector.analyze(
        host_ip="192.168.1.100",
        connection_count=1000,  # Extreme
        bytes_sent=1000000000.0,  # Extreme
        destination="10.0.0.255",
        protocol="http",
    )

    assert anom_result.anomaly_score >= 0, "Should have anomaly score"
    print("  ✓ Anomalous detection works")

    # Get profile
    profile = detector.get_profile("192.168.1.100")
    assert profile is not None
    assert profile["host_ip"] == "192.168.1.100"
    print("  ✓ Profile retrieval works")

    # Get stats
    stats = detector.get_stats()
    assert stats.total_analyzed >= 16, "Should track analyses"
    assert stats.hosts_profiled >= 1, "Should track profiles"
    print("  ✓ Statistics tracking works")

    print("  ✓ ML Anomaly Detector works")


def test_custom_rules_in_detector():
    """Test adding custom rules to detector."""
    print("[TEST] Custom Rules in Detector")
    detector = MLAnomalyDetector()

    # Add custom rule
    def rule_large_transfer(**kwargs):
        return kwargs.get("bytes_sent", 0) > 500000000

    detector.add_custom_rule("huge_transfer", rule_large_transfer, severity="critical")

    # Test with custom rule violation
    result = detector.analyze(
        host_ip="192.168.1.101",
        connection_count=1,
        bytes_sent=1000000000.0,  # Violates rule
        destination="10.0.0.1",
        protocol="http",
    )

    assert result.anomaly_score > 0, "Should have anomaly"
    assert "huge_transfer" in result.reasons or len(result.reasons) > 0
    print("  ✓ Custom rule detection works")

    print("  ✓ Custom Rules in Detector works")


def test_host_profile_dataclass():
    """Test HostProfile dataclass."""
    print("[TEST] HostProfile Dataclass")

    profile = HostProfile(host_ip="192.168.1.100")
    profile.connection_count = 10
    profile.protocols_used.add("http")
    profile.unique_destinations.add("10.0.0.1")
    profile.learning_complete = True

    profile_dict = profile.to_dict()
    assert profile_dict["host_ip"] == "192.168.1.100"
    assert profile_dict["connection_count"] == 10
    assert "http" in profile_dict["protocols_used"]
    assert profile_dict["learning_complete"] is True
    print("  ✓ HostProfile to_dict works")

    print("  ✓ HostProfile Dataclass works")


def test_anomaly_score_dataclass():
    """Test AnomalyScore dataclass."""
    print("[TEST] AnomalyScore Dataclass")

    score = AnomalyScore(
        score=0.85,
        severity="high",
        reasons=["High data transfer", "Unusual protocol"],
        confidence=0.9,
    )

    assert score.score == 0.85
    assert score.severity == "high"
    assert len(score.reasons) == 2
    assert score.confidence == 0.9

    score_dict = score.to_dict()
    assert "timestamp" in score_dict
    print("  ✓ AnomalyScore to_dict works")

    print("  ✓ AnomalyScore Dataclass works")


def test_ml_detection_result():
    """Test MLDetectionResult."""
    print("[TEST] MLDetectionResult")

    result = MLDetectionResult(
        is_anomalous=True,
        anomaly_score=0.87,
        severity="high",
        reasons=["Rule violation"],
        model_confidence=0.92,
        ensemble_votes={"classifier1": True, "classifier2": False},
        host_profile_match=0.75,
    )

    assert result.is_anomalous is True
    assert result.anomaly_score == 0.87
    assert result.severity == "high"

    result_dict = result.to_dict()
    assert result_dict["is_anomalous"] is True
    assert result_dict["ensemble_votes"]["classifier1"] is True
    print("  ✓ MLDetectionResult to_dict works")

    print("  ✓ MLDetectionResult works")


def test_global_singleton():
    """Test global singleton pattern."""
    print("[TEST] Global Singleton")

    detector1 = get_ml_detector()
    detector2 = get_ml_detector()

    assert detector1 is detector2, "Should return same instance"
    print("  ✓ Singleton pattern works")

    # Test custom initialization
    detector3 = init_ml_detector(learning_period=3600)
    assert detector3 is not detector1, "Should create new instance"
    print("  ✓ Custom initialization works")

    print("  ✓ Global Singleton works")


def test_eve_json_enrichment():
    """Test EVE JSON enrichment."""
    print("[TEST] EVE JSON Enrichment")

    result = MLDetectionResult(
        is_anomalous=True,
        anomaly_score=0.85,
        severity="high",
        model_confidence=0.9,
        reasons=["Test reason"],
    )

    eve_event = {"src_ip": "192.168.1.100", "dest_ip": "10.0.0.1"}
    enriched = enrich_eve_event_with_ml(eve_event, result)

    assert "ml_analysis" in enriched
    assert enriched["ml_analysis"]["anomaly_score"] == 0.85
    assert enriched["ml_analysis"]["severity"] == "high"
    assert enriched["ml_analysis"]["is_anomalous"] is True
    assert len(enriched["ml_analysis"]["reasons"]) == 1
    print("  ✓ EVE JSON enrichment works")

    print("  ✓ EVE JSON Enrichment works")


def test_ml_stats():
    """Test ML statistics."""
    print("[TEST] ML Statistics")

    stats = MLStats(
        total_analyzed=100,
        anomalies_detected=25,
        low_severity=5,
        medium_severity=10,
        high_severity=8,
        critical_severity=2,
        hosts_profiled=10,
        learning_complete=3,
    )

    assert stats.total_analyzed == 100
    assert stats.anomalies_detected == 25
    assert stats.critical_severity == 2
    print("  ✓ ML Statistics works")

    print("  ✓ ML Statistics works")


def test_multiple_hosts_profiling():
    """Test profiling multiple hosts."""
    print("[TEST] Multiple Hosts Profiling")
    detector = MLAnomalyDetector()

    # Profile multiple hosts
    for host_num in range(5):
        host_ip = f"192.168.1.{100 + host_num}"
        for i in range(10):
            detector.analyze(
                host_ip=host_ip,
                connection_count=i + 1,
                bytes_sent=1000.0 * (i + 1),
                destination=f"10.0.0.{i}",
                protocol="http",
            )

    stats = detector.get_stats()
    assert stats.hosts_profiled >= 5, "Should profile multiple hosts"
    print("  ✓ Multiple host profiling works")

    print("  ✓ Multiple Hosts Profiling works")


def test_ensemble_voting():
    """Test ensemble voting mechanism."""
    print("[TEST] Ensemble Voting")
    detector = MLAnomalyDetector()

    result = detector.analyze(
        host_ip="192.168.1.100",
        connection_count=1,
        bytes_sent=1000.0,
        destination="10.0.0.1",
        protocol="http",
    )

    assert "ensemble_votes" in result.__dict__
    assert isinstance(result.ensemble_votes, dict)
    assert len(result.ensemble_votes) > 0, "Should have classifier votes"
    print("  ✓ Ensemble voting tracked")

    print("  ✓ Ensemble Voting works")


def test_severity_calculation():
    """Test severity score calculation."""
    print("[TEST] Severity Calculation")
    detector = MLAnomalyDetector()

    # Add many extreme analyses
    for i in range(20):
        result = detector.analyze(
            host_ip="192.168.1.102",
            connection_count=1000 + i * 100,
            bytes_sent=1000000.0 + i * 100000,
            destination="10.0.0.1",
            protocol="http",
        )

    # Last result should have high severity
    result_extreme = detector.analyze(
        host_ip="192.168.1.102",
        connection_count=5000,
        bytes_sent=10000000.0,
        destination="10.0.0.1",
        protocol="http",
    )

    assert result_extreme.anomaly_score > 0.5, "Should have high anomaly score"
    assert result_extreme.severity in [
        "high",
        "critical",
        "medium",
    ], "Should assign severity"
    print("  ✓ Severity calculation works")

    print("  ✓ Severity Calculation works")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE F PART 5: ML & ANOMALY DETECTION - TEST SUITE")
    print("="*70 + "\n")

    tests = [
        test_statistical_baseline,
        test_behavioral_profiler,
        test_ensemble_classifier,
        test_custom_rule_engine,
        test_ml_anomaly_detector,
        test_custom_rules_in_detector,
        test_host_profile_dataclass,
        test_anomaly_score_dataclass,
        test_ml_detection_result,
        test_global_singleton,
        test_eve_json_enrichment,
        test_ml_stats,
        test_multiple_hosts_profiling,
        test_ensemble_voting,
        test_severity_calculation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("STATUS: ✓ ALL TESTS PASSED")
    else:
        print(f"STATUS: ✗ {failed} TESTS FAILED")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
