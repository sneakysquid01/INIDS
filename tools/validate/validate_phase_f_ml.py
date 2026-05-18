"""
Phase F Validation Script - ML & Anomaly Detection

Validates Phase F Part 5 (ML anomaly detection) completeness.
No pytest dependency - runs standalone.
"""

import sys


def validate_ml_imports():
    """Validate all ML imports work."""
    print("[VALIDATION] Checking ML imports...")
    
    try:
        from src.advanced import ml_anomaly
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
        print("  ✓ All ML imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_statistical_baseline():
    """Validate statistical baseline."""
    print("[VALIDATION] Checking statistical baseline...")
    
    try:
        from src.advanced.ml_anomaly import StatisticalBaseline
        
        baseline = StatisticalBaseline(window_size=3600, min_samples=5)
        
        # Add samples
        for i in range(10):
            baseline.add_sample("metric", 100.0 + i)
        
        # Check
        mean_val, stdev_val, is_ready = baseline.get_baseline("metric")
        assert is_ready is True
        assert mean_val > 90
        
        # Check anomaly
        is_anom = baseline.is_anomalous("metric", 1000.0, sigma=2.0)
        assert is_anom is True
        
        print("  ✓ Statistical baseline works")
        return True
    except Exception as e:
        print(f"  ✗ Statistical baseline validation failed: {e}")
        return False


def validate_behavioral_profiler():
    """Validate behavioral profiler."""
    print("[VALIDATION] Checking behavioral profiler...")
    
    try:
        from src.advanced.ml_anomaly import BehavioralProfiler
        import time
        
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
        
        # Test learning
        time.sleep(1.1)
        profile = profiler.get_profile("192.168.1.100")
        assert profile.learning_complete is True
        
        print("  ✓ Behavioral profiler works")
        return True
    except Exception as e:
        print(f"  ✗ Behavioral profiler validation failed: {e}")
        return False


def validate_ensemble_classifier():
    """Validate ensemble classifier."""
    print("[VALIDATION] Checking ensemble classifier...")
    
    try:
        from src.advanced.ml_anomaly import EnsembleClassifier
        
        ensemble = EnsembleClassifier()
        
        def clf1(**kwargs):
            return kwargs.get("value", 0) > 100
        
        def clf2(**kwargs):
            return kwargs.get("value", 0) > 200
        
        ensemble.add_classifier("clf1", clf1, weight=1.0)
        ensemble.add_classifier("clf2", clf2, weight=1.5)
        
        # Test
        is_anom, votes, conf = ensemble.classify(value=150)
        assert is_anom is True
        assert votes["clf1"] is True
        assert votes["clf2"] is False
        
        print("  ✓ Ensemble classifier works")
        return True
    except Exception as e:
        print(f"  ✗ Ensemble classifier validation failed: {e}")
        return False


def validate_custom_rules():
    """Validate custom rule engine."""
    print("[VALIDATION] Checking custom rule engine...")
    
    try:
        from src.advanced.ml_anomaly import CustomRuleEngine
        
        rules = CustomRuleEngine()
        
        def rule1(**kwargs):
            return kwargs.get("bytes", 0) > 1000000
        
        rules.add_rule("large_transfer", rule1, severity="high")
        
        # Test
        violations = rules.evaluate_rules(bytes=500000)
        assert len(violations) == 0
        
        violations2 = rules.evaluate_rules(bytes=2000000)
        assert len(violations2) == 1
        assert violations2[0][0] == "large_transfer"
        
        print("  ✓ Custom rule engine works")
        return True
    except Exception as e:
        print(f"  ✗ Custom rule engine validation failed: {e}")
        return False


def validate_ml_detector():
    """Validate ML detector."""
    print("[VALIDATION] Checking ML detector...")
    
    try:
        from src.advanced.ml_anomaly import MLAnomalyDetector
        
        detector = MLAnomalyDetector()
        
        # Normal analysis
        result = detector.analyze(
            host_ip="192.168.1.100",
            connection_count=1,
            bytes_sent=1000.0,
            destination="10.0.0.1",
            protocol="http",
        )
        
        assert result.anomaly_score >= 0
        assert result.severity in ["none", "low", "medium", "high", "critical"]
        
        # Get profile
        profile = detector.get_profile("192.168.1.100")
        assert profile is not None
        
        # Get stats
        stats = detector.get_stats()
        assert stats.total_analyzed >= 1
        
        print("  ✓ ML detector works")
        return True
    except Exception as e:
        print(f"  ✗ ML detector validation failed: {e}")
        return False


def validate_custom_rules_in_detector():
    """Validate custom rules in detector."""
    print("[VALIDATION] Checking custom rules in detector...")
    
    try:
        from src.advanced.ml_anomaly import MLAnomalyDetector
        
        detector = MLAnomalyDetector()
        
        def rule(**kwargs):
            return kwargs.get("bytes_sent", 0) > 500000000
        
        detector.add_custom_rule("huge", rule, severity="critical")
        
        result = detector.analyze(
            host_ip="192.168.1.101",
            connection_count=1,
            bytes_sent=1000000000.0,
            destination="10.0.0.1",
            protocol="http",
        )
        
        assert result.anomaly_score > 0
        
        print("  ✓ Custom rules in detector work")
        return True
    except Exception as e:
        print(f"  ✗ Custom rules in detector validation failed: {e}")
        return False


def validate_host_profile():
    """Validate HostProfile dataclass."""
    print("[VALIDATION] Checking HostProfile dataclass...")
    
    try:
        from src.advanced.ml_anomaly import HostProfile
        
        profile = HostProfile(host_ip="192.168.1.100")
        profile.connection_count = 10
        profile.protocols_used.add("http")
        
        profile_dict = profile.to_dict()
        assert profile_dict["host_ip"] == "192.168.1.100"
        assert profile_dict["connection_count"] == 10
        
        print("  ✓ HostProfile dataclass works")
        return True
    except Exception as e:
        print(f"  ✗ HostProfile dataclass validation failed: {e}")
        return False


def validate_anomaly_score():
    """Validate AnomalyScore dataclass."""
    print("[VALIDATION] Checking AnomalyScore dataclass...")
    
    try:
        from src.advanced.ml_anomaly import AnomalyScore
        
        score = AnomalyScore(
            score=0.85,
            severity="high",
            confidence=0.9,
        )
        
        assert score.score == 0.85
        assert score.severity == "high"
        
        score_dict = score.to_dict()
        assert "timestamp" in score_dict
        
        print("  ✓ AnomalyScore dataclass works")
        return True
    except Exception as e:
        print(f"  ✗ AnomalyScore dataclass validation failed: {e}")
        return False


def validate_ml_detection_result():
    """Validate MLDetectionResult."""
    print("[VALIDATION] Checking MLDetectionResult...")
    
    try:
        from src.advanced.ml_anomaly import MLDetectionResult
        
        result = MLDetectionResult(
            is_anomalous=True,
            anomaly_score=0.87,
            severity="high",
        )
        
        assert result.is_anomalous is True
        assert result.anomaly_score == 0.87
        
        result_dict = result.to_dict()
        assert result_dict["is_anomalous"] is True
        
        print("  ✓ MLDetectionResult works")
        return True
    except Exception as e:
        print(f"  ✗ MLDetectionResult validation failed: {e}")
        return False


def validate_global_singleton():
    """Validate global singleton."""
    print("[VALIDATION] Checking global singleton...")
    
    try:
        from src.advanced.ml_anomaly import get_ml_detector, init_ml_detector
        
        d1 = get_ml_detector()
        d2 = get_ml_detector()
        
        assert d1 is d2
        
        d3 = init_ml_detector()
        assert d3 is not d1
        
        print("  ✓ Global singleton works")
        return True
    except Exception as e:
        print(f"  ✗ Global singleton validation failed: {e}")
        return False


def validate_eve_enrichment():
    """Validate EVE JSON enrichment."""
    print("[VALIDATION] Checking EVE enrichment...")
    
    try:
        from src.advanced.ml_anomaly import MLDetectionResult, enrich_eve_event_with_ml
        
        result = MLDetectionResult(
            is_anomalous=True,
            anomaly_score=0.85,
            severity="high",
        )
        
        eve = {"src_ip": "192.168.1.100"}
        enriched = enrich_eve_event_with_ml(eve, result)
        
        assert "ml_analysis" in enriched
        assert enriched["ml_analysis"]["severity"] == "high"
        
        print("  ✓ EVE enrichment works")
        return True
    except Exception as e:
        print(f"  ✗ EVE enrichment validation failed: {e}")
        return False


def validate_ml_stats():
    """Validate ML statistics."""
    print("[VALIDATION] Checking ML statistics...")
    
    try:
        from src.advanced.ml_anomaly import MLStats
        
        stats = MLStats(
            total_analyzed=100,
            anomalies_detected=25,
            critical_severity=2,
        )
        
        assert stats.total_analyzed == 100
        assert stats.anomalies_detected == 25
        
        print("  ✓ ML statistics work")
        return True
    except Exception as e:
        print(f"  ✗ ML statistics validation failed: {e}")
        return False


def validate_multiple_hosts():
    """Validate multiple host profiling."""
    print("[VALIDATION] Checking multiple host profiling...")
    
    try:
        from src.advanced.ml_anomaly import MLAnomalyDetector
        
        detector = MLAnomalyDetector()
        
        # Profile multiple hosts
        for i in range(3):
            host = f"192.168.1.{100 + i}"
            for j in range(5):
                detector.analyze(
                    host_ip=host,
                    connection_count=j + 1,
                    bytes_sent=1000.0,
                    destination="10.0.0.1",
                    protocol="http",
                )
        
        stats = detector.get_stats()
        assert stats.hosts_profiled >= 3
        
        print("  ✓ Multiple host profiling works")
        return True
    except Exception as e:
        print(f"  ✗ Multiple host profiling validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("PHASE F PART 5: ML & ANOMALY DETECTION - VALIDATION")
    print("="*60 + "\n")
    
    validations = [
        validate_ml_imports,
        validate_statistical_baseline,
        validate_behavioral_profiler,
        validate_ensemble_classifier,
        validate_custom_rules,
        validate_ml_detector,
        validate_custom_rules_in_detector,
        validate_host_profile,
        validate_anomaly_score,
        validate_ml_detection_result,
        validate_global_singleton,
        validate_eve_enrichment,
        validate_ml_stats,
        validate_multiple_hosts,
    ]
    
    passed = 0
    failed = 0
    
    for val_func in validations:
        try:
            if val_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Validation error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(validations)} validations passed")
    if failed == 0:
        print("STATUS: ✓ ALL VALIDATIONS PASSED")
    else:
        print(f"STATUS: ✗ {failed} VALIDATIONS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
