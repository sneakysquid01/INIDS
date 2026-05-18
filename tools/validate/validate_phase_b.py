"""
Phase B Validation Script
Standalone validation of protocol parsers without pytest dependency
"""

import sys
import os


class ValidationError(Exception):
    """Validation error"""
    pass


def validate_imports():
    """Validate all modules import correctly"""
    print("\n>>> Validating Imports...")
    
    try:
        from src.protocol_parsers import (
            HTTPParser, HTTPRequest, HTTPResponse, HTTPMethod,
            DNSParser, DNSQuery, DNSResponse,
            TLSParser, TLSClientHello, TLSServerHello,
            ProtocolDetector, ApplicationProtocol,
            ProtocolAnalyzer
        )
        print("✓ All protocol parser imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_http_parser():
    """Validate HTTP parser functionality"""
    print("\n>>> Validating HTTP Parser...")
    
    from src.protocol_parsers import HTTPParser
    
    # Test 1: Parse request
    request_bytes = b"GET /api/test HTTP/1.1\r\nHost: example.com\r\n\r\n"
    req = HTTPParser.parse_request(request_bytes)
    if not req or req.method != "GET":
        raise ValidationError("HTTP request parsing failed")
    print("✓ HTTP request parsing works")
    
    # Test 2: Parse response
    response_bytes = b"HTTP/1.1 200 OK\r\nServer: nginx\r\n\r\n"
    resp = HTTPParser.parse_response(response_bytes)
    if not resp or resp.status_code != 200:
        raise ValidationError("HTTP response parsing failed")
    print("✓ HTTP response parsing works")
    
    # Test 3: Detect suspicious patterns
    sus_bytes = b"GET /admin' OR '1'='1 HTTP/1.1\r\n\r\n"
    sus_req = HTTPParser.parse_request(sus_bytes)
    if not sus_req or not sus_req.is_suspicious:
        raise ValidationError("HTTP suspicious pattern detection failed")
    print("✓ HTTP suspicious pattern detection works")
    
    # Test 4: Extract features
    features = HTTPParser.extract_features(req)
    if "http_method" not in features:
        raise ValidationError("HTTP feature extraction incomplete")
    print("✓ HTTP feature extraction works")
    
    return True


def validate_dns_parser():
    """Validate DNS parser functionality"""
    print("\n>>> Validating DNS Parser...")
    
    from src.protocol_parsers import DNSParser, DNSQuery
    
    # Test 1: Entropy calculation
    entropy = DNSParser._calculate_entropy("randomstring123")
    if entropy < 0 or entropy > 10:
        raise ValidationError("DNS entropy calculation out of bounds")
    print("✓ DNS entropy calculation works")
    
    # Test 2: DGA detection
    query = DNSQuery(
        transaction_id=1,
        domain="xyzjhqasdfl.xyz",
        query_type="A"
    )
    DNSParser._check_query_suspicious(query)
    # High entropy domains should be flagged as suspicious
    if query.domain_entropy > 3.0 and not query.is_suspicious:
        raise ValidationError("DNS DGA detection failed")
    print("✓ DNS DGA detection works")
    
    # Test 3: Extract features
    normal_query = DNSQuery(transaction_id=2, domain="google.com", query_type="A")
    features = DNSParser.extract_features(normal_query)
    if "dns_domain_length" not in features:
        raise ValidationError("DNS feature extraction incomplete")
    print("✓ DNS feature extraction works")
    
    return True


def validate_tls_parser():
    """Validate TLS parser functionality"""
    print("\n>>> Validating TLS Parser...")
    
    from src.protocol_parsers import TLSParser, TLSClientHello
    
    # Test 1: Version detection
    version = TLSParser._get_tls_version(0x0303)
    if "1.2" not in version:
        raise ValidationError("TLS version detection failed")
    print("✓ TLS version detection works")
    
    # Test 2: Weak cipher detection
    hello = TLSClientHello(
        tls_version="TLS 1.0",
        cipher_suites=[0x0004, 0x0005]  # RC4
    )
    TLSParser._check_client_hello_suspicious(hello)
    if not hello.is_suspicious:
        raise ValidationError("TLS weak cipher detection failed")
    print("✓ TLS weak cipher detection works")
    
    # Test 3: JA3 fingerprinting
    normal_hello = TLSClientHello(
        tls_version="TLS 1.2",
        cipher_suites=[0x002f, 0x0035]
    )
    ja3_string = TLSParser._compute_ja3_string(normal_hello)
    ja3_hash = TLSParser._compute_ja3_hash(ja3_string)
    if len(ja3_hash) != 32:
        raise ValidationError("TLS JA3 fingerprinting failed")
    print("✓ TLS JA3 fingerprinting works")
    
    # Test 4: Extract features
    features = TLSParser.extract_features(normal_hello)
    if "tls_cipher_count" not in features:
        raise ValidationError("TLS feature extraction incomplete")
    print("✓ TLS feature extraction works")
    
    return True


def validate_protocol_detector():
    """Validate protocol detector functionality"""
    print("\n>>> Validating Protocol Detector...")
    
    from src.protocol_parsers import ProtocolDetector, ApplicationProtocol
    
    # Test 1: Detect HTTP by port
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.1", "10.0.0.1", 12345, 80, "TCP"
    )
    if classification.protocol != ApplicationProtocol.HTTP:
        raise ValidationError("Protocol detection by port failed")
    print("✓ Protocol detection by port works")
    
    # Test 2: Detect DNS by port
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.1", "8.8.8.8", 12345, 53, "UDP"
    )
    if classification.protocol != ApplicationProtocol.DNS:
        raise ValidationError("DNS protocol detection failed")
    print("✓ DNS protocol detection works")
    
    # Test 3: Detect HTTP by payload
    payload = b"GET / HTTP/1.1\r\n\r\n"
    classification = ProtocolDetector.classify_protocol(
        "192.168.1.1", "10.0.0.1", 12345, 9999, "TCP", payload
    )
    if classification.protocol != ApplicationProtocol.HTTP:
        raise ValidationError("HTTP payload detection failed")
    print("✓ HTTP payload detection works")
    
    # Test 4: Feature extraction
    from src.protocol_parsers import ParsedProtocolData
    parsed = ParsedProtocolData(protocol=ApplicationProtocol.HTTP, raw_payload=payload)
    features = ProtocolDetector.extract_ml_features(parsed)
    if "protocol" not in features:
        raise ValidationError("Protocol feature extraction failed")
    print("✓ Protocol feature extraction works")
    
    return True


def validate_phase_b_integration():
    """Validate Phase B integration"""
    print("\n>>> Validating Phase B Integration...")
    
    from src.protocol_parsers.phase_b_integration import (
        ProtocolAnalyzer,
        ProtocolAnalysisContext,
        PhaseABIntegrationAdapter
    )
    
    # Test 1: Protocol analysis context
    ctx = ProtocolAnalysisContext()
    if ctx.packets_analyzed != 0:
        raise ValidationError("Protocol analysis context initialization failed")
    print("✓ Protocol analysis context works")
    
    # Test 2: Detection callback creation
    callback = PhaseABIntegrationAdapter.create_protocol_detection_callback()
    if not callable(callback):
        raise ValidationError("Detection callback creation failed")
    print("✓ Detection callback creation works")
    
    # Test 3: Helper functions
    from src.protocol_parsers.phase_b_integration import (
        analyze_flow_protocol,
        get_flow_protocol,
        is_flow_protocol_suspicious,
        get_protocol_suspicious_indicators
    )
    print("✓ Integration helper functions available")
    
    return True


def validate_backward_compatibility():
    """Validate Phase B doesn't break Phase A"""
    print("\n>>> Validating Backward Compatibility...")
    
    try:
        # Verify Phase A still imports
        from src.packet_capture import PacketSource, PCAPReader
        from src.decoding import PacketDecoder
        from src.flow_tracking import FlowTable
        from src.integration import PacketProcessingPipeline
        print("✓ Phase A modules still import correctly")
        
        # Verify Phase A classes still exist
        if not hasattr(FlowTable, 'get_or_create_flow'):
            raise ValidationError("Phase A FlowTable API broken")
        print("✓ Phase A API unchanged")
        
        return True
    except Exception as e:
        raise ValidationError(f"Phase A backward compatibility broken: {e}")


def validate_protocol_parser_files():
    """Validate all Phase B files exist"""
    print("\n>>> Validating File Structure...")
    
    files = [
        "src/protocol_parsers/__init__.py",
        "src/protocol_parsers/http_parser.py",
        "src/protocol_parsers/dns_parser.py",
        "src/protocol_parsers/tls_parser.py",
        "src/protocol_parsers/protocol_detector.py",
        "src/protocol_parsers/phase_b_integration.py",
        "tests/test_phase_b_protocol_parsers.py",
    ]
    
    for file in files:
        full_path = os.path.join(os.path.dirname(__file__), "..", file)
        if not os.path.exists(full_path):
            raise ValidationError(f"Missing file: {file}")
        print(f"✓ {file}")
    
    return True


def main():
    """Run all validations"""
    print("\n" + "="*70)
    print("PHASE B: PROTOCOL PARSERS VALIDATION")
    print("="*70)
    
    validations = [
        ("File Structure", validate_protocol_parser_files),
        ("Imports", validate_imports),
        ("HTTP Parser", validate_http_parser),
        ("DNS Parser", validate_dns_parser),
        ("TLS Parser", validate_tls_parser),
        ("Protocol Detector", validate_protocol_detector),
        ("Phase B Integration", validate_phase_b_integration),
        ("Backward Compatibility", validate_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, validation_func in validations:
        try:
            if validation_func():
                passed += 1
            else:
                failed += 1
                errors.append(name)
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
            errors.append(f"{name} ({str(e)})")
    
    print("\n" + "="*70)
    print(f"VALIDATION RESULTS")
    print("="*70)
    print(f"✓ Passed: {passed}/{len(validations)}")
    print(f"✗ Failed: {failed}/{len(validations)}")
    
    if errors:
        print("\nFailed validations:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False
    else:
        print("\n✓ ALL VALIDATIONS PASSED")
        print("="*70 + "\n")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
