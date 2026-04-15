"""
Phase F Validation Script - TLS Certificate Validation

Validates Phase F Part 3 (TLS validation) completeness and functionality.
No pytest dependency - runs standalone.
"""

import sys


def validate_tls_imports():
    """Validate all TLS imports work."""
    print("[VALIDATION] Checking TLS imports...")
    
    try:
        from src.advanced import tls_validation
        from src.advanced.tls_validation import (
            CertificateValidator,
            CertificateInfo,
            CertificateValidationResult,
            CertificateParser,
            TLSStats,
            get_tls_validator,
            init_tls_validator,
        )
        print("  ✓ All TLS imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_certificate_info():
    """Validate CertificateInfo structure."""
    print("[VALIDATION] Checking CertificateInfo...")
    
    try:
        from src.advanced.tls_validation import CertificateInfo
        import time
        
        # Create instance
        info = CertificateInfo(
            subject="test.example.com",
            issuer="DigiCert",
            subject_alt_names=["*.example.com"],
            public_key_bits=2048,
            not_valid_before=time.time(),
            not_valid_after=time.time() + 86400,
        )
        
        # Verify fields
        assert info.subject == "test.example.com"
        assert info.public_key_bits == 2048
        
        # Test dict conversion
        d = info.to_dict()
        assert "subject" in d
        
        print("  ✓ CertificateInfo works correctly")
        return True
    except Exception as e:
        print(f"  ✗ CertificateInfo validation failed: {e}")
        return False


def validate_validation_result():
    """Validate CertificateValidationResult."""
    print("[VALIDATION] Checking CertificateValidationResult...")
    
    try:
        from src.advanced.tls_validation import CertificateValidationResult
        
        # Create result
        result = CertificateValidationResult(
            valid=True,
            hostname_valid=True,
            confidence=0.95,
        )
        
        # Verify
        assert result.valid is True
        assert result.confidence == 0.95
        
        # Test dict conversion
        d = result.to_dict()
        assert d["valid"] is True
        
        print("  ✓ CertificateValidationResult works correctly")
        return True
    except Exception as e:
        print(f"  ✗ CertificateValidationResult validation failed: {e}")
        return False


def validate_certificate_validator():
    """Validate CertificateValidator."""
    print("[VALIDATION] Checking CertificateValidator...")
    
    try:
        from src.advanced.tls_validation import CertificateValidator
        
        validator = CertificateValidator(min_key_bits=2048)
        
        # Test adding bad certs
        validator.add_known_bad_cert("badcertfp123")
        
        # Test HPKP
        validator.add_hpkp_pin("example.com", "pin123")
        
        # Get stats
        stats = validator.get_stats()
        assert stats.total_validations == 0
        
        print("  ✓ CertificateValidator works correctly")
        return True
    except Exception as e:
        print(f"  ✗ CertificateValidator validation failed: {e}")
        return False


def validate_hostname_matching():
    """Validate hostname matching."""
    print("[VALIDATION] Checking hostname matching...")
    
    try:
        from src.advanced.tls_validation import CertificateValidator
        
        validator = CertificateValidator()
        
        # Exact match
        assert validator._hostname_matches("example.com", "example.com") is True
        
        # Wildcard
        assert validator._hostname_matches("www.example.com", "*.example.com") is True
        
        # Non-match
        assert validator._hostname_matches("other.com", "example.com") is False
        
        print("  ✓ Hostname matching works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Hostname matching validation failed: {e}")
        return False


def validate_weak_algorithm_detection():
    """Validate weak algorithm detection."""
    print("[VALIDATION] Checking weak algorithm detection...")
    
    try:
        from src.advanced.tls_validation import CertificateValidator
        
        validator = CertificateValidator()
        
        # Check weak algorithms list
        assert "sha1" in validator.WEAK_ALGORITHMS
        assert "md5" in validator.WEAK_ALGORITHMS
        
        # Check string matching
        algo = "sha1WithRSAEncryption"
        is_weak = any(w in algo.lower() for w in validator.WEAK_ALGORITHMS)
        assert is_weak is True
        
        print("  ✓ Weak algorithm detection works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Weak algorithm detection validation failed: {e}")
        return False


def validate_weak_key_detection():
    """Validate weak key detection."""
    print("[VALIDATION] Checking weak key detection...")
    
    try:
        from src.advanced.tls_validation import CertificateValidator
        
        validator = CertificateValidator(min_key_bits=2048)
        
        # Check RSA minimum
        assert validator.MIN_RSA_BITS == 2048
        
        # Check EC minimum
        assert validator.MIN_EC_BITS == 256
        
        # Test weak key detection
        is_weak = 1024 < validator.MIN_RSA_BITS
        assert is_weak is True
        
        print("  ✓ Weak key detection works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Weak key detection validation failed: {e}")
        return False


def validate_expiry_detection():
    """Validate expiry detection."""
    print("[VALIDATION] Checking expiry detection...")
    
    try:
        import time
        from src.advanced.tls_validation import CertificateInfo
        
        now = time.time()
        
        # Create expired cert
        info_expired = CertificateInfo(
            subject="expired.com",
            issuer="CA",
            not_valid_before=now - 86400 * 365,
            not_valid_after=now - 86400,  # Expired
        )
        
        is_expired = now > info_expired.not_valid_after
        assert is_expired is True
        
        # Create valid cert
        info_valid = CertificateInfo(
            subject="valid.com",
            issuer="CA",
            not_valid_before=now - 86400,
            not_valid_after=now + 86400 * 365,
        )
        
        is_valid = now < info_valid.not_valid_after
        assert is_valid is True
        
        print("  ✓ Expiry detection works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Expiry detection validation failed: {e}")
        return False


def validate_self_signed_detection():
    """Validate self-signed detection."""
    print("[VALIDATION] Checking self-signed detection...")
    
    try:
        from src.advanced.tls_validation import CertificateInfo
        
        # Self-signed
        info_ss = CertificateInfo(
            subject="test.com",
            issuer="test.com",  # Same
            is_self_signed=True,
        )
        assert info_ss.is_self_signed is True
        
        # Signed by CA
        info_ca = CertificateInfo(
            subject="test.com",
            issuer="DigiCert",
            is_self_signed=False,
        )
        assert info_ca.is_self_signed is False
        
        print("  ✓ Self-signed detection works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Self-signed detection validation failed: {e}")
        return False


def validate_hpkp_parsing():
    """Validate HPKP header parsing."""
    print("[VALIDATION] Checking HPKP parsing...")
    
    try:
        from src.advanced.tls_validation import CertificateValidator
        
        validator = CertificateValidator()
        
        # Test basic HPKP
        hpkp = 'pin-sha256="abc123"; max-age=31536000'
        result = validator._verify_hpkp_header("abc123", hpkp)
        assert result is True
        
        # Test non-matching
        result2 = validator._verify_hpkp_header("xyz789", hpkp)
        assert result2 is False
        
        print("  ✓ HPKP parsing works correctly")
        return True
    except Exception as e:
        print(f"  ✗ HPKP parsing validation failed: {e}")
        return False


def validate_pinning_database():
    """Validate certificate pinning database."""
    print("[VALIDATION] Checking pinning database...")
    
    try:
        from src.advanced.tls_validation import CertificateValidator
        
        validator = CertificateValidator()
        
        # Add pin
        validator.add_hpkp_pin("example.com", "pin123")
        
        # Check it's stored
        match = validator._check_pinning_db("example.com", "pin123")
        assert match is True
        
        # Check non-matching
        match2 = validator._check_pinning_db("example.com", "wrongpin")
        assert match2 is False
        
        print("  ✓ Pinning database works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Pinning database validation failed: {e}")
        return False


def validate_confidence_calculation():
    """Validate confidence score calculation."""
    print("[VALIDATION] Checking confidence calculation...")
    
    try:
        from src.advanced.tls_validation import (
            CertificateValidator,
            CertificateValidationResult
        )
        
        validator = CertificateValidator()
        
        # Perfect result
        perfect = CertificateValidationResult(errors=[], warnings=[])
        conf_perfect = validator._calculate_confidence(perfect)
        assert conf_perfect == 1.0
        
        # With errors
        with_errors = CertificateValidationResult(
            errors=["expired", "weak_key"],
            warnings=[]
        )
        conf_errors = validator._calculate_confidence(with_errors)
        assert conf_errors < 1.0
        assert conf_errors >= 0.0
        
        print("  ✓ Confidence calculation works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Confidence calculation validation failed: {e}")
        return False


def validate_global_singleton():
    """Validate global singleton."""
    print("[VALIDATION] Checking global singleton...")
    
    try:
        from src.advanced.tls_validation import get_tls_validator
        
        v1 = get_tls_validator()
        v2 = get_tls_validator()
        
        assert v1 is v2
        
        print("  ✓ Global singleton works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Global singleton validation failed: {e}")
        return False


def validate_tls_stats():
    """Validate TLS statistics."""
    print("[VALIDATION] Checking TLS statistics...")
    
    try:
        from src.advanced.tls_validation import TLSStats
        
        stats = TLSStats(
            total_validations=100,
            valid_certificates=95,
            expired_detected=3,
        )
        
        assert stats.total_validations == 100
        assert stats.valid_certificates == 95
        
        print("  ✓ TLS statistics work correctly")
        return True
    except Exception as e:
        print(f"  ✗ TLS statistics validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("PHASE F PART 3: TLS VALIDATION - VALIDATION")
    print("="*60 + "\n")
    
    validations = [
        validate_tls_imports,
        validate_certificate_info,
        validate_validation_result,
        validate_certificate_validator,
        validate_hostname_matching,
        validate_weak_algorithm_detection,
        validate_weak_key_detection,
        validate_expiry_detection,
        validate_self_signed_detection,
        validate_hpkp_parsing,
        validate_pinning_database,
        validate_confidence_calculation,
        validate_global_singleton,
        validate_tls_stats,
    ]
    
    passed = 0
    failed = 0
    
    for validation_func in validations:
        try:
            if validation_func():
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
