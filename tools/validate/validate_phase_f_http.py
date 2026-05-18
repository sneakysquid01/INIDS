"""
Phase F Validation Script - HTTP Pattern Detection

Validates Phase F Part 4 (HTTP pattern detection) completeness.
No pytest dependency - runs standalone.
"""

import sys


def validate_http_imports():
    """Validate all HTTP imports work."""
    print("[VALIDATION] Checking HTTP imports...")
    
    try:
        from src.advanced import http_patterns
        from src.advanced.http_patterns import (
            HTTPPatternAnalyzer,
            HTTPSignatureDetector,
            BotDetector,
            HTTPAnomalyDetector,
            EncodingDetector,
            HTTPSignatureMatch,
            HTTPAnomalyResult,
            HTTPAnalysisResult,
            HTTPStats,
            HTTPPatternCache,
            get_http_analyzer,
            init_http_analyzer,
            enrich_eve_event_with_http,
        )
        print("  ✓ All HTTP imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_signature_detector():
    """Validate HTTP signature detector."""
    print("[VALIDATION] Checking HTTP signature detector...")
    
    try:
        from src.advanced.http_patterns import HTTPSignatureDetector
        
        detector = HTTPSignatureDetector()
        
        # Test SQL injection
        payload = b"SELECT * FROM users WHERE 1=1"
        matches = detector.find_signatures(payload)
        assert len(matches) > 0, "Should detect SQL injection"
        assert matches[0].severity in ["critical", "high"]
        
        print("  ✓ Signature detector works")
        return True
    except Exception as e:
        print(f"  ✗ Signature detector validation failed: {e}")
        return False


def validate_bot_detector():
    """Validate bot detector."""
    print("[VALIDATION] Checking bot detector...")
    
    try:
        from src.advanced.http_patterns import BotDetector
        
        detector = BotDetector()
        
        # Test Nmap detection
        detected, bot_type, confidence = detector.detect_bot(
            user_agent="nmap/7.80"
        )
        assert detected is True, "Should detect Nmap"
        assert bot_type == "nmap"
        
        # Test normal UA
        detected2, bot_type2, conf2 = detector.detect_bot(
            user_agent="Mozilla/5.0"
        )
        assert confidence > conf2, "Bot should have higher confidence"
        
        print("  ✓ Bot detector works")
        return True
    except Exception as e:
        print(f"  ✗ Bot detector validation failed: {e}")
        return False


def validate_anomaly_detector():
    """Validate anomaly detector."""
    print("[VALIDATION] Checking anomaly detector...")
    
    try:
        from src.advanced.http_patterns import HTTPAnomalyDetector
        
        detector = HTTPAnomalyDetector()
        
        # Test missing UA
        anomalies = detector.detect_anomalies(
            method="GET",
            headers={},
            body=b"",
            url="/"
        )
        assert len(anomalies) > 0, "Should detect missing UA"
        
        # Test normal
        anomalies2 = detector.detect_anomalies(
            method="GET",
            headers={"User-Agent": "Mozilla/5.0"},
            body=b"",
            url="/"
        )
        
        print("  ✓ Anomaly detector works")
        return True
    except Exception as e:
        print(f"  ✗ Anomaly detector validation failed: {e}")
        return False


def validate_encoding_detector():
    """Validate encoding detector."""
    print("[VALIDATION] Checking encoding detector...")
    
    try:
        from src.advanced.http_patterns import EncodingDetector
        
        detector = EncodingDetector()
        
        # Test base64
        encodings = detector.detect_encodings("aGVsbG8gd29ybGQ=")
        assert "base64" in encodings, "Should detect base64"
        
        # Test hex
        encodings = detector.detect_encodings("48656c6c6f")
        assert "hex" in encodings, "Should detect hex"
        
        print("  ✓ Encoding detector works")
        return True
    except Exception as e:
        print(f"  ✗ Encoding detector validation failed: {e}")
        return False


def validate_pattern_analyzer():
    """Validate pattern analyzer."""
    print("[VALIDATION] Checking pattern analyzer...")
    
    try:
        from src.advanced.http_patterns import HTTPPatternAnalyzer
        
        analyzer = HTTPPatternAnalyzer()
        
        # Test malicious request
        result = analyzer.analyze(
            method="POST",
            url="/login?id=1 UNION SELECT",
            headers={"User-Agent": "Mozilla/5.0"},
            body=b"SELECT * FROM users"
        )
        
        assert result.risk_score > 0.4, "Should have risk"
        assert len(result.signatures_found) > 0, "Should find signatures"
        
        # Test normal request
        result2 = analyzer.analyze(
            method="GET",
            url="/home",
            headers={"User-Agent": "Mozilla/5.0"},
            body=b""
        )
        
        assert result2.risk_score < result.risk_score
        
        print("  ✓ Pattern analyzer works")
        return True
    except Exception as e:
        print(f"  ✗ Pattern analyzer validation failed: {e}")
        return False


def validate_pattern_cache():
    """Validate pattern cache."""
    print("[VALIDATION] Checking pattern cache...")
    
    try:
        from src.advanced.http_patterns import HTTPPatternCache, HTTPAnalysisResult
        import time
        
        cache = HTTPPatternCache(max_size=100, ttl_seconds=3600)
        
        result = HTTPAnalysisResult(risk_score=0.8)
        cache.put("key1", result)
        
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.risk_score == 0.8
        
        # Test short TTL
        cache_short = HTTPPatternCache(max_size=100, ttl_seconds=1)
        cache_short.put("key2", result)
        time.sleep(1.1)
        expired = cache_short.get("key2")
        assert expired is None, "Should expire"
        
        print("  ✓ Pattern cache works")
        return True
    except Exception as e:
        print(f"  ✗ Pattern cache validation failed: {e}")
        return False


def validate_custom_signatures():
    """Validate custom signature adding."""
    print("[VALIDATION] Checking custom signatures...")
    
    try:
        from src.advanced.http_patterns import HTTPPatternAnalyzer
        
        analyzer = HTTPPatternAnalyzer()
        analyzer.add_custom_signature(
            name="test_sig",
            patterns=[b"malicious"],
            severity="critical"
        )
        
        result = analyzer.analyze(
            method="POST",
            url="/",
            headers={"User-Agent": "test"},
            body=b"This is malicious"
        )
        
        has_custom = any(s.signature_name == "test_sig" for s in result.signatures_found)
        assert has_custom, "Should detect custom signature"
        
        print("  ✓ Custom signatures work")
        return True
    except Exception as e:
        print(f"  ✗ Custom signatures validation failed: {e}")
        return False


def validate_risk_scoring():
    """Validate risk score calculation."""
    print("[VALIDATION] Checking risk scoring...")
    
    try:
        from src.advanced.http_patterns import HTTPPatternAnalyzer
        
        analyzer = HTTPPatternAnalyzer()
        
        # Critical payload
        result_high = analyzer.analyze(
            method="POST",
            url="/",
            headers={"User-Agent": "Mozilla/5.0"},
            body=b"<script>alert('xss')</script> UNION SELECT"
        )
        
        # Normal
        result_low = analyzer.analyze(
            method="GET",
            url="/index",
            headers={"User-Agent": "Mozilla/5.0"},
            body=b"Hello"
        )
        
        assert result_high.risk_score > result_low.risk_score
        assert 0.0 <= result_high.risk_score <= 1.0
        assert 0.0 <= result_low.risk_score <= 1.0
        
        print("  ✓ Risk scoring works")
        return True
    except Exception as e:
        print(f"  ✗ Risk scoring validation failed: {e}")
        return False


def validate_global_singleton():
    """Validate global singleton."""
    print("[VALIDATION] Checking global singleton...")
    
    try:
        from src.advanced.http_patterns import get_http_analyzer, init_http_analyzer
        
        a1 = get_http_analyzer()
        a2 = get_http_analyzer()
        
        assert a1 is a2, "Should be same instance"
        
        a3 = init_http_analyzer()
        assert a3 is not a1, "Should be different after init"
        
        print("  ✓ Global singleton works")
        return True
    except Exception as e:
        print(f"  ✗ Global singleton validation failed: {e}")
        return False


def validate_eve_enrichment():
    """Validate EVE JSON enrichment."""
    print("[VALIDATION] Checking EVE enrichment...")
    
    try:
        from src.advanced.http_patterns import HTTPAnalysisResult, enrich_eve_event_with_http
        
        result = HTTPAnalysisResult(risk_score=0.85)
        eve = {"event_type": "http"}
        
        enriched = enrich_eve_event_with_http(eve, result)
        
        assert "http" in enriched
        assert "analysis" in enriched["http"]
        assert "threat_level" in enriched["http"]
        assert enriched["http"]["threat_level"] == "high"
        
        print("  ✓ EVE enrichment works")
        return True
    except Exception as e:
        print(f"  ✗ EVE enrichment validation failed: {e}")
        return False


def validate_http_stats():
    """Validate HTTP statistics."""
    print("[VALIDATION] Checking HTTP stats...")
    
    try:
        from src.advanced.http_patterns import HTTPStats, HTTPPatternAnalyzer
        
        stats = HTTPStats(
            total_analyzed=50,
            signatures_found=10,
            anomalies_detected=5,
            bots_detected=2,
            high_risk=3,
            critical_risk=1,
        )
        
        assert stats.total_analyzed == 50
        assert stats.critical_risk == 1
        
        # Test analyzer stats
        analyzer = HTTPPatternAnalyzer()
        analyzer.analyze(
            method="GET",
            url="/test",
            headers={"User-Agent": "test"},
            body=b"test"
        )
        
        a_stats = analyzer.get_stats()
        assert a_stats.total_analyzed >= 1
        
        print("  ✓ HTTP stats work")
        return True
    except Exception as e:
        print(f"  ✗ HTTP stats validation failed: {e}")
        return False


def validate_dataclasses():
    """Validate dataclass structures."""
    print("[VALIDATION] Checking dataclasses...")
    
    try:
        from src.advanced.http_patterns import (
            HTTPSignatureMatch,
            HTTPAnomalyResult,
            HTTPAnalysisResult,
        )
        
        # Test HTTPSignatureMatch
        match = HTTPSignatureMatch(
            signature_name="test",
            pattern_type="regex",
            severity="high",
            matched_pattern="test",
            position=0,
        )
        match_dict = match.to_dict()
        assert "timestamp" in match_dict
        
        # Test HTTPAnomalyResult
        anomaly = HTTPAnomalyResult(
            anomaly_type="test",
            description="test",
            severity="medium",
            confidence=0.8,
        )
        anomaly_dict = anomaly.to_dict()
        assert "timestamp" in anomaly_dict
        
        # Test HTTPAnalysisResult
        result = HTTPAnalysisResult(
            bot_detected=True,
            bot_type="test",
            risk_score=0.9,
        )
        result_dict = result.to_dict()
        assert "bot_type" in result_dict
        
        print("  ✓ Dataclasses work")
        return True
    except Exception as e:
        print(f"  ✗ Dataclasses validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("PHASE F PART 4: HTTP PATTERN DETECTION - VALIDATION")
    print("="*60 + "\n")
    
    validations = [
        validate_http_imports,
        validate_signature_detector,
        validate_bot_detector,
        validate_anomaly_detector,
        validate_encoding_detector,
        validate_pattern_analyzer,
        validate_pattern_cache,
        validate_custom_signatures,
        validate_risk_scoring,
        validate_global_singleton,
        validate_eve_enrichment,
        validate_http_stats,
        validate_dataclasses,
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
