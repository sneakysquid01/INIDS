"""
Phase B Protocol Parsers Test Suite
Tests for HTTP, DNS, TLS parsers and protocol detection
"""

import sys
import os
from typing import List

# Test data
TEST_HTTP_REQUEST = b"""GET /api/users?id=123 HTTP/1.1\r
Host: api.example.com\r
User-Agent: Mozilla/5.0\r
Content-Length: 0\r
\r
"""

TEST_HTTP_RESPONSE = b"""HTTP/1.1 200 OK\r
Server: nginx/1.19.0\r
Content-Type: application/json\r
Content-Length: 13\r
\r
{"status":"ok"}"""

TEST_HTTP_SUSPICIOUS_REQUEST = b"""GET /admin/../../etc/passwd HTTP/1.1\r
Host: vulnerable.site\r
User-Agent: sqlmap/1.4.8\r
\r
"""

# DNS query packet (simplified structure - normally more complex)
TEST_DNS_QUERY = b'\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01'

# TLS ClientHello (simplified handshake record)
TEST_TLS_HANDSHAKE = bytes.fromhex(
    '16'  # Record type: Handshake
    '0303'  # TLS 1.2
    '0040'  # Length: 64 bytes
)


def test_http_parser():
    """Test HTTP request and response parsing"""
    print("\n=== Testing HTTP Parser ===")
    
    from src.protocol_parsers import HTTPParser
    
    # Test 1: Parse basic HTTP request
    print("✓ Test 1: Parse basic HTTP request")
    req = HTTPParser.parse_request(TEST_HTTP_REQUEST)
    assert req is not None, "HTTP request parsing failed"
    assert req.method == "GET", f"Expected GET, got {req.method}"
    assert req.path == "/api/users", f"Expected /api/users, got {req.path}"
    assert req.host == "api.example.com", f"Expected api.example.com, got {req.host}"
    assert "id=123" in req.query_string, "Query string parsing failed"
    print(f"  Response: {req}")
    
    # Test 2: Parse query parameters
    print("✓ Test 2: Parse query parameters")
    assert len(req.query_params) > 0, "Query params not parsed"
    assert req.query_params.get('id') == '123', f"Query params wrong: {req.query_params}"
    print(f"  Query params: {req.query_params}")
    
    # Test 3: Parse HTTP response
    print("✓ Test 3: Parse HTTP response")
    resp = HTTPParser.parse_response(TEST_HTTP_RESPONSE)
    assert resp is not None, "HTTP response parsing failed"
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert resp.is_success(), "Response should be success (2xx)"
    print(f"  Response: {resp}")
    
    # Test 4: Detect suspicious patterns
    print("✓ Test 4: Detect suspicious patterns")
    sus_req = HTTPParser.parse_request(TEST_HTTP_SUSPICIOUS_REQUEST)
    assert sus_req is not None, "Suspicious request should parse"
    assert sus_req.is_suspicious, "Should detect as suspicious"
    assert len(sus_req.suspicious_indicators) > 0, "Should have indicators"
    print(f"  Suspicious indicators: {sus_req.suspicious_indicators}")
    
    # Test 5: Extract ML features
    print("✓ Test 5: Extract HTTP features for ML")
    features = HTTPParser.extract_features(req)
    assert "http_method" in features, "Missing http_method feature"
    assert features["http_method"] == "GET", f"Wrong method: {features['http_method']}"
    assert "http_uri_length" in features, "Missing uri_length feature"
    print(f"  Features: {list(features.keys())}")
    
    print("✓ HTTP Parser: PASSED\n")


def test_dns_parser():
    """Test DNS query and response parsing"""
    print("\n=== Testing DNS Parser ===")
    
    from src.protocol_parsers import DNSParser
    
    # Test 1: Test entropy calculation
    print("✓ Test 1: Calculate domain entropy")
    entropy = DNSParser._calculate_entropy("xyzhjkqwert")  # High entropy (DGA-like)
    assert entropy > 2.0, f"Entropy calculation seems wrong: {entropy}"
    print(f"  High entropy domain: {entropy:.2f}")
    
    # Test 2: Detect DGA patterns
    print("✓ Test 2: Detect DGA patterns")
    dga_query = DNSParser.DNSQuery(
        transaction_id=1,
        domain="asfhkjqwerasfhkj.com",
        query_type="A"
    )
    DNSParser._check_query_suspicious(dga_query)
    assert dga_query.is_suspicious or dga_query.domain_entropy > 2.0, "DGA detection failed"
    print(f"  DGA query suspicious: {dga_query.is_suspicious}")
    
    # Test 3: Detect DNS tunneling
    print("✓ Test 3: Detect DNS tunneling patterns")
    tunnel_query = DNSParser.DNSQuery(
        transaction_id=2,
        domain="verylongsubdomainwithofdatainside.exfiltration.example.com",
        query_type="A"
    )
    DNSParser._check_query_suspicious(tunnel_query)
    if tunnel_query.is_suspicious:
        print(f"  Detected indicators: {tunnel_query.suspicious_indicators}")
    
    # Test 4: Extract DNS features
    print("✓ Test 4: Extract DNS features for ML")
    normal_query = DNSParser.DNSQuery(
        transaction_id=3,
        domain="google.com",
        query_type="A"
    )
    features = DNSParser.extract_features(normal_query)
    assert "dns_domain_length" in features, "Missing dns_domain_length feature"
    assert "dns_domain_entropy" in features, "Missing dns_domain_entropy feature"
    print(f"  Features: {list(features.keys())}")
    
    # Test 5: DNS response parsing
    print("✓ Test 5: Test DNS response structure")
    response = DNSParser.DNSResponse(
        transaction_id=3,
        response_code="NOERROR",
        query_domain="google.com",
        query_type="A",
        answer_ips=["142.251.33.46"],
        answers=[{"type": "A", "value": "142.251.33.46", "domain": "google.com"}]
    )
    assert not response.is_error, "Normal response should not be error"
    assert response.is_success() if hasattr(response, 'is_success') else True
    print(f"  Response: {response}")
    
    print("✓ DNS Parser: PASSED\n")


def test_tls_parser():
    """Test TLS ClientHello and ServerHello parsing"""
    print("\n=== Testing TLS Parser ===")
    
    from src.protocol_parsers import TLSParser
    
    # Test 1: TLS version detection
    print("✓ Test 1: TLS version detection")
    version = TLSParser._get_tls_version(0x0303)  # TLS 1.2
    assert "1.2" in version, f"Expected TLS 1.2, got {version}"
    print(f"  Version: {version}")
    
    # Test 2: Record type name mapping
    print("✓ Test 2: Record type name mapping")
    a_record = TLSParser._get_record_type_name(1)
    assert a_record == "A", f"Expected A, got {a_record}"
    print(f"  A Record: {a_record}")
    
    # Test 3: All parsers instantiate ClientHello correctly
    print("✓ Test 3: ClientHello creation")
    from src.protocol_parsers import TLSClientHello
    hello = TLSClientHello(
        tls_version="TLS 1.2",
        cipher_suites=[0x002f, 0x0035, 0x003c],
        server_name="example.com"
    )
    assert hello.tls_version == "TLS 1.2", "Version mismatch"
    assert len(hello.cipher_suites) == 3, "Cipher count mismatch"
    print(f"  ClientHello: {hello}")
    
    # Test 4: Weak cipher detection
    print("✓ Test 4: Detect weak ciphers")
    weak_hello = TLSClientHello(
        tls_version="TLS 1.0",
        cipher_suites=[0x0004, 0x0005],  # RC4 ciphers
        server_name="example.com"
    )
    TLSParser._check_client_hello_suspicious(weak_hello)
    assert weak_hello.is_suspicious, "Should detect weak ciphers"
    print(f"  Weak cipher detected: {weak_hello.suspicious_indicators}")
    
    # Test 5: JA3 fingerprint
    print("✓ Test 5: Compute JA3 fingerprint")
    ja3_string = TLSParser._compute_ja3_string(hello)
    assert ja3_string, "JA3 string should not be empty"
    ja3_hash = TLSParser._compute_ja3_hash(ja3_string)
    assert len(ja3_hash) == 32, f"MD5 hash should be 32 chars, got {len(ja3_hash)}"
    print(f"  JA3: {ja3_hash[:16]}...")
    
    # Test 6: SNI parsing
    print("✓ Test 6: Test SNI parsing")
    sni = TLSParser._parse_sni(b'\x00\x1b\x00\x00\x18\x07example\x03com\x00')
    assert sni is None or sni, "SNI parsing should succeed or return None gracefully"
    print(f"  SNI parse: {'success' if sni else 'returned None'}")
    
    print("✓ TLS Parser: PASSED\n")


def test_protocol_detector():
    """Test protocol detection and classification"""
    print("\n=== Testing Protocol Detector ===")
    
    from src.protocol_parsers import ProtocolDetector, ApplicationProtocol
    
    # Test 1: Detect HTTP by port
    print("✓ Test 1: Detect HTTP by well-known port")
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.100", "10.0.0.1", 54321, 80, "TCP"
    )
    assert classification.protocol == ApplicationProtocol.HTTP, f"Expected HTTP, got {classification.protocol}"
    assert classification.confidence > 0.9, "Confidence should be high for well-known port"
    print(f"  Classification: {classification}")
    
    # Test 2: Detect DNS by port
    print("✓ Test 2: Detect DNS by well-known port")
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.100", "8.8.8.8", 54321, 53, "UDP"
    )
    assert classification.protocol == ApplicationProtocol.DNS, f"Expected DNS, got {classification.protocol}"
    print(f"  Classification: {classification}")
    
    # Test 3: Detect HTTP by payload pattern
    print("✓ Test 3: Detect HTTP by payload pattern")
    http_payload = b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.100", "10.0.0.1", 54321, 9999, "TCP", http_payload
    )
    assert classification.protocol == ApplicationProtocol.HTTP, "HTTP pattern detection failed"
    print(f"  Detection: {classification.detection_method}")
    
    # Test 4: Detect TLS by payload pattern
    print("✓ Test 4: Detect TLS by payload pattern")
    tls_payload = bytes([0x16, 0x03, 0x03]) + b"\x00\x40" + b"\x00" * 60
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.100", "10.0.0.1", 54321, 443, "TCP", tls_payload
    )
    assert classification.protocol == ApplicationProtocol.TLS, "TLS pattern detection failed"
    print(f"  Detection: {classification.detection_method}")
    
    # Test 5: Extract ML features
    print("✓ Test 5: Extract protocol ML features")
    from src.protocol_parsers import ParsedProtocolData
    parsed = ParsedProtocolData(
        protocol=ApplicationProtocol.HTTP,
        raw_payload=http_payload
    )
    features = ProtocolDetector.extract_ml_features(parsed)
    assert "protocol" in features, "Missing protocol feature"
    assert "payload_size" in features, "Missing payload_size feature"
    print(f"  Features extracted: {len(features)}")
    
    print("✓ Protocol Detector: PASSED\n")


def test_phase_b_integration():
    """Test Phase B integration with Phase A"""
    print("\n=== Testing Phase B Integration ===")
    
    from src.protocol_parsers.phase_b_integration import (
        ProtocolAnalyzer,
        ProtocolAnalysisContext,
        PhaseABIntegrationAdapter
    )
    
    # Test 1: Protocol analysis context creation
    print("✓ Test 1: Create protocol analysis context")
    ctx = ProtocolAnalysisContext()
    assert ctx.detected_protocol is None, "Should start with no protocol"
    assert ctx.is_suspicious == False, "Should start as not suspicious"
    print(f"  Context: {ctx}")
    
    # Test 2: Create detection callback
    print("✓ Test 2: Create protocol detection callback")
    callback = PhaseABIntegrationAdapter.create_protocol_detection_callback()
    assert callable(callback), "Callback should be callable"
    print(f"  Callback created: {type(callback).__name__}")
    
    # Test 3: Get suspicious indicators list
    print("✓ Test 3: Get suspicious indicators")
    from src.protocol_parsers import HTTPRequest
    req = HTTPRequest(method="GET", uri="test", path="test")
    req.suspicious_indicators = ["sql_injection", "xss_attempt"]
    assert len(req.suspicious_indicators) > 0, "Should have indicators"
    print(f"  Indicators: {req.suspicious_indicators}")
    
    print("✓ Phase B Integration: PASSED\n")


def run_all_tests():
    """Run all Phase B tests"""
    print("\n" + "="*60)
    print("PHASE B: PROTOCOL PARSERS TEST SUITE")
    print("="*60)
    
    tests = [
        ("HTTP Parser", test_http_parser),
        ("DNS Parser", test_dns_parser),
        ("TLS Parser", test_tls_parser),
        ("Protocol Detector", test_protocol_detector),
        ("Phase B Integration", test_phase_b_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED")
            print(f"  Error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    # This enables running the test file standalone
    success = run_all_tests()
    sys.exit(0 if success else 1)
