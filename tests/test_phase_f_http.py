"""
Tests for Phase F Part 4: HTTP Pattern Detection

Tests HTTP signature detection, anomaly detection, bot detection,
encoding detection, and integrated analysis.
No pytest dependency - runs standalone.
"""

import time
from src.advanced.http_patterns import (
    HTTPSignatureDetector,
    BotDetector,
    HTTPAnomalyDetector,
    EncodingDetector,
    HTTPPatternAnalyzer,
    HTTPSignatureMatch,
    HTTPAnomalyResult,
    HTTPAnalysisResult,
    HTTPStats,
    HTTPPatternCache,
    get_http_analyzer,
    init_http_analyzer,
    enrich_eve_event_with_http,
)


def test_http_signature_detector():
    """Test HTTP signature detector."""
    print("[TEST] HTTP Signature Detector")
    detector = HTTPSignatureDetector()

    # Test SQL injection
    payload_sql = b"SELECT * FROM users WHERE id=1 OR '1'='1'"
    matches = detector.find_signatures(payload_sql)
    assert len(matches) > 0, "Should detect SQL injection"
    assert matches[0].severity == "critical"
    print("  ✓ SQL injection detected")

    # Test XSS
    payload_xss = b"<script>alert('xss')</script>"
    matches = detector.find_signatures(payload_xss)
    assert len(matches) > 0, "Should detect XSS"
    assert matches[0].severity == "critical"
    print("  ✓ XSS detected")

    # Test path traversal
    payload_pt = b"GET ../../etc/passwd HTTP/1.1"
    matches = detector.find_signatures(payload_pt)
    assert len(matches) > 0, "Should detect path traversal"
    print("  ✓ Path traversal detected")

    # Test command injection
    payload_ci = b"; cat /etc/passwd"
    matches = detector.find_signatures(payload_ci)
    assert len(matches) > 0, "Should detect command injection"
    print("  ✓ Command injection detected")

    print("  ✓ HTTP Signature Detector works")


def test_bot_detector():
    """Test bot detector."""
    print("[TEST] Bot Detector")
    detector = BotDetector()

    # Test Nmap detection
    headers = {"User-Agent": "nmap/7.80"}
    detected, bot_type, confidence = detector.detect_bot(
        user_agent="nmap/7.80", headers=headers, url=""
    )
    assert detected is True, "Should detect Nmap"
    assert bot_type == "nmap"
    assert confidence > 0.5
    print("  ✓ Nmap detected")

    # Test SQLMap detection
    detected, bot_type, confidence = detector.detect_bot(
        user_agent="sqlmap/1.4.7", headers={}, url=""
    )
    assert detected is True, "Should detect SQLMap"
    assert bot_type == "sqlmap"
    print("  ✓ SQLMap detected")

    # Test crawler detection
    detected, bot_type, confidence = detector.detect_bot(
        user_agent="Mozilla/5.0 (compatible; Googlebot/2.1)", headers={}, url=""
    )
    assert detected is True, "Should detect crawler"
    print("  ✓ Web crawler detected")

    # Test normal user-agent
    detected, bot_type, confidence = detector.detect_bot(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)", headers={}, url=""
    )
    assert detected is False or confidence < 0.5, "Should not detect normal UA as bot"
    print("  ✓ Normal user-agent not flagged")

    print("  ✓ Bot Detector works")


def test_http_anomaly_detector():
    """Test HTTP anomaly detector."""
    print("[TEST] HTTP Anomaly Detector")
    detector = HTTPAnomalyDetector()

    # Test missing User-Agent
    anomalies = detector.detect_anomalies(
        method="GET", headers={}, body=b"", url="/"
    )
    assert len(anomalies) > 0, "Should detect missing User-Agent"
    print("  ✓ Missing User-Agent detected")

    # Test unusual method
    anomalies = detector.detect_anomalies(
        method="INVALID", headers={"User-Agent": "test"}, body=b"", url="/"
    )
    assert len(anomalies) > 0, "Should detect unusual method"
    found_method = any(a.anomaly_type == "unusual_method" for a in anomalies)
    assert found_method, "Should have unusual_method anomaly"
    print("  ✓ Unusual method detected")

    # Test excessive headers
    many_headers = {f"Header-{i}": f"value-{i}" for i in range(60)}
    anomalies = detector.detect_anomalies(
        method="GET", headers=many_headers, body=b"", url="/"
    )
    found_excessive = any(a.anomaly_type == "excessive_headers" for a in anomalies)
    assert found_excessive, "Should detect excessive headers"
    print("  ✓ Excessive headers detected")

    # Test HTTP smuggling (double-chunked)
    anomalies = detector.detect_anomalies(
        method="POST",
        headers={"Transfer-Encoding": "chunked, chunked", "User-Agent": "test"},
        body=b"",
        url="/",
    )
    found_smuggling = any(a.anomaly_type == "double_encoding" for a in anomalies)
    assert found_smuggling, "Should detect HTTP smuggling"
    print("  ✓ HTTP smuggling detected")

    # Test normal request
    anomalies = detector.detect_anomalies(
        method="GET",
        headers={"User-Agent": "Mozilla/5.0", "Host": "example.com"},
        body=b"",
        url="/",
    )
    # May have some low-severity anomalies but not critical
    critical = [a for a in anomalies if a.severity == "critical"]
    assert len(critical) == 0, "Normal request should not have critical anomalies"
    print("  ✓ Normal request passes")

    print("  ✓ HTTP Anomaly Detector works")


def test_encoding_detector():
    """Test encoding detector."""
    print("[TEST] Encoding Detector")
    detector = EncodingDetector()

    # Test base64
    encodings = detector.detect_encodings("aGVsbG8gd29ybGQ=")
    assert "base64" in encodings, "Should detect base64"
    print("  ✓ Base64 detected")

    # Test hex
    encodings = detector.detect_encodings("48656c6c6f")
    assert "hex" in encodings, "Should detect hex"
    print("  ✓ Hex detected")

    # Test URL encoding
    encodings = detector.detect_encodings("hello%20world%21")
    assert "url_encoded" in encodings, "Should detect URL encoding"
    print("  ✓ URL encoding detected")

    # Test Unicode encoding
    encodings = detector.detect_encodings("\\u0041\\u0042\\u0043")
    assert "unicode_encoded" in encodings, "Should detect Unicode encoding"
    print("  ✓ Unicode encoding detected")

    print("  ✓ Encoding Detector works")


def test_http_pattern_analyzer():
    """Test HTTP pattern analyzer."""
    print("[TEST] HTTP Pattern Analyzer")
    analyzer = HTTPPatternAnalyzer()

    # Test complete analysis
    result = analyzer.analyze(
        method="GET",
        url="/admin?id=1 UNION SELECT * FROM users--",
        headers={"User-Agent": "Mozilla/5.0"},
        body=b"SELECT * FROM users",
    )

    assert isinstance(result, HTTPAnalysisResult)
    assert len(result.signatures_found) > 0, "Should find signatures"
    assert result.risk_score > 0.5, "Should have high risk score"
    print("  ✓ Malicious request analyzed")

    # Test normal request
    result_normal = analyzer.analyze(
        method="GET",
        url="/index.html",
        headers={"User-Agent": "Mozilla/5.0", "Host": "example.com"},
        body=b"",
    )

    assert isinstance(result_normal, HTTPAnalysisResult)
    assert result_normal.risk_score < 0.5, "Normal request should have low risk"
    print("  ✓ Normal request analyzed")

    # Test bot detection in analyzer
    result_bot = analyzer.analyze(
        method="GET",
        url="/robots.txt",
        headers={"User-Agent": "nmap/7.80"},
        body=b"",
    )

    assert result_bot.bot_detected is True, "Should detect bot"
    print("  ✓ Bot detected in analyzer")

    # Test stats updates
    stats = analyzer.get_stats()
    assert stats.total_analyzed >= 3, "Stats should track analyses"
    assert stats.signatures_found > 0, "Stats should track signatures"
    print("  ✓ Statistics tracked")

    print("  ✓ HTTP Pattern Analyzer works")


def test_http_pattern_cache():
    """Test HTTP pattern cache."""
    print("[TEST] HTTP Pattern Cache")
    cache = HTTPPatternCache(max_size=100, ttl_seconds=3600)

    # Create result
    result = HTTPAnalysisResult(risk_score=0.8)

    # Store and retrieve
    cache.put("test_key", result)
    retrieved = cache.get("test_key")
    assert retrieved is not None, "Cache should store and retrieve"
    assert retrieved.risk_score == 0.8
    print("  ✓ Cache stores/retrieves")

    # Test TTL
    cache_short = HTTPPatternCache(max_size=100, ttl_seconds=1)
    cache_short.put("short_key", result)
    retrieved_short = cache_short.get("short_key")
    assert retrieved_short is not None
    time.sleep(1.1)
    retrieved_expired = cache_short.get("short_key")
    assert retrieved_expired is None, "Cache should expire entries"
    print("  ✓ Cache TTL works")

    # Test LRU eviction
    cache_small = HTTPPatternCache(max_size=2, ttl_seconds=3600)
    cache_small.put("key1", result)
    cache_small.put("key2", result)
    cache_small.put("key3", result)  # Should evict key1
    assert cache_small.get("key1") is None, "LRU should evict oldest"
    assert cache_small.get("key3") is not None, "New entry should be kept"
    print("  ✓ LRU eviction works")

    print("  ✓ HTTP Pattern Cache works")


def test_custom_signatures():
    """Test adding custom signatures."""
    print("[TEST] Custom Signatures")
    analyzer = HTTPPatternAnalyzer()

    # Add custom signature
    analyzer.add_custom_signature(
        name="custom_malware",
        patterns=[b"evilpayload", b"malicious"],
        severity="critical",
    )

    # Test detection
    result = analyzer.analyze(
        method="POST",
        url="/upload",
        headers={"User-Agent": "Mozilla/5.0"},
        body=b"This contains evilpayload data",
    )

    assert len(result.signatures_found) > 0, "Should detect custom signature"
    found_custom = any(s.signature_name == "custom_malware" for s in result.signatures_found)
    assert found_custom, "Should have custom_malware signature"
    print("  ✓ Custom signatures detected")

    print("  ✓ Custom Signatures work")


def test_risk_score_calculation():
    """Test risk score calculation."""
    print("[TEST] Risk Score Calculation")
    analyzer = HTTPPatternAnalyzer()

    # Test critical payload
    result_critical = analyzer.analyze(
        method="POST",
        url="/login",
        headers={"User-Agent": "Mozilla/5.0"},
        body=b"<script>alert('xss')</script> UNION SELECT * FROM users--",
    )

    assert result_critical.risk_score > 0.5, "Should have high risk"
    print("  ✓ Critical payload has high risk score")

    # Test normal payload
    result_normal = analyzer.analyze(
        method="GET",
        url="/index.html",
        headers={"User-Agent": "Mozilla/5.0"},
        body=b"Hello World",
    )

    assert result_normal.risk_score < 0.3, "Normal should have low risk"
    print("  ✓ Normal request has low risk score")

    print("  ✓ Risk Score Calculation works")


def test_global_singleton():
    """Test global singleton pattern."""
    print("[TEST] Global Singleton")

    analyzer1 = get_http_analyzer()
    analyzer2 = get_http_analyzer()

    assert analyzer1 is analyzer2, "Should return same instance"
    print("  ✓ Singleton pattern works")

    # Test custom initialization
    analyzer_custom = init_http_analyzer(cache_size=5000, cache_ttl=1800)
    assert analyzer_custom is not analyzer1, "Should create new instance"
    print("  ✓ Custom initialization works")

    print("  ✓ Global Singleton works")


def test_eve_json_enrichment():
    """Test EVE JSON enrichment."""
    print("[TEST] EVE JSON Enrichment")

    result = HTTPAnalysisResult(risk_score=0.85)
    eve_event = {"event_type": "http"}

    enriched = enrich_eve_event_with_http(eve_event, result)

    assert "http" in enriched, "Should have http field"
    assert "analysis" in enriched["http"]
    assert "threat_level" in enriched["http"]
    assert enriched["http"]["threat_level"] == "high", "0.85 risk = high threat"
    print("  ✓ EVE JSON enrichment works")

    # Test critical threat
    result_crit = HTTPAnalysisResult(risk_score=0.95)
    eve_crit = {"event_type": "http"}
    enriched_crit = enrich_eve_event_with_http(eve_crit, result_crit)
    assert enriched_crit["http"]["threat_level"] == "critical"
    print("  ✓ Critical threat level assigned")

    print("  ✓ EVE JSON Enrichment works")


def test_http_stats():
    """Test HTTP statistics."""
    print("[TEST] HTTP Statistics")

    stats = HTTPStats(
        total_analyzed=100,
        signatures_found=25,
        anomalies_detected=15,
        bots_detected=5,
        high_risk=10,
        critical_risk=3,
    )

    assert stats.total_analyzed == 100
    assert stats.signatures_found == 25
    assert stats.critical_risk == 3
    print("  ✓ HTTP Statistics structure works")

    print("  ✓ HTTP Statistics works")


def test_signature_match_dataclass():
    """Test HTTPSignatureMatch dataclass."""
    print("[TEST] HTTPSignatureMatch Dataclass")

    match = HTTPSignatureMatch(
        signature_name="sql_injection",
        pattern_type="regex",
        severity="critical",
        matched_pattern="UNION SELECT",
        position=42,
    )

    assert match.signature_name == "sql_injection"
    assert match.severity == "critical"

    match_dict = match.to_dict()
    assert "signature_name" in match_dict
    assert "timestamp" in match_dict
    print("  ✓ HTTPSignatureMatch works")

    print("  ✓ HTTPSignatureMatch Dataclass works")


def test_anomaly_result_dataclass():
    """Test HTTPAnomalyResult dataclass."""
    print("[TEST] HTTPAnomalyResult Dataclass")

    anomaly = HTTPAnomalyResult(
        anomaly_type="missing_user_agent",
        description="HTTP request missing User-Agent",
        severity="medium",
        confidence=0.9,
    )

    assert anomaly.anomaly_type == "missing_user_agent"
    assert anomaly.confidence == 0.9

    anomaly_dict = anomaly.to_dict()
    assert "anomaly_type" in anomaly_dict
    assert "timestamp" in anomaly_dict
    print("  ✓ HTTPAnomalyResult works")

    print("  ✓ HTTPAnomalyResult Dataclass works")


def test_analysis_result_to_dict():
    """Test HTTPAnalysisResult to_dict conversion."""
    print("[TEST] HTTPAnalysisResult to_dict")

    result = HTTPAnalysisResult(
        bot_detected=True,
        bot_type="sqlmap",
        bot_confidence=0.85,
        risk_score=0.75,
        encoding_detected=["base64", "url_encoded"],
    )

    result_dict = result.to_dict()

    assert "signatures_found" in result_dict
    assert "anomalies_detected" in result_dict
    assert "bot_detected" in result_dict
    assert result_dict["bot_type"] == "sqlmap"
    assert result_dict["risk_score"] == 0.75
    assert len(result_dict["encoding_detected"]) == 2
    print("  ✓ HTTPAnalysisResult to_dict works")

    print("  ✓ HTTPAnalysisResult to_dict works")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE F PART 4: HTTP PATTERN DETECTION - TEST SUITE")
    print("="*70 + "\n")

    tests = [
        test_http_signature_detector,
        test_bot_detector,
        test_http_anomaly_detector,
        test_encoding_detector,
        test_http_pattern_analyzer,
        test_http_pattern_cache,
        test_custom_signatures,
        test_risk_score_calculation,
        test_global_singleton,
        test_eve_json_enrichment,
        test_http_stats,
        test_signature_match_dataclass,
        test_anomaly_result_dataclass,
        test_analysis_result_to_dict,
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
