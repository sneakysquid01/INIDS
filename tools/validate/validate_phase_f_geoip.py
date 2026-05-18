"""
Phase F Validation Script - GeoIP Enrichment

Validates Phase F Part 1 (GeoIP enrichment) completeness and functionality.
No pytest dependency - runs standalone.
"""

import sys


def validate_geoip_imports():
    """Validate all GeoIP imports work."""
    print("[VALIDATION] Checking GeoIP imports...")
    
    try:
        from src.advanced import (
            GeoIPLookup,
            GeoIPData,
            GeoIPCache,
            GeoIPDatabase,
            GeoIPStats,
            RiskDetector,
            get_geoip_lookup,
            init_geoip,
            enrich_eve_event_with_geoip,
            ip_to_int,
            int_to_ip,
            is_private_ip,
            is_loopback_ip,
        )
        print("  ✓ All GeoIP imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_ip_utils():
    """Validate IP utility functions."""
    print("[VALIDATION] Checking IP utilities...")
    
    try:
        from src.advanced import ip_to_int, int_to_ip, is_private_ip, is_loopback_ip
        
        # Test conversions
        ip_int = ip_to_int("192.0.2.1")
        assert ip_int > 0
        
        ip_str = int_to_ip(ip_int)
        assert ip_str == "192.0.2.1"
        
        # Test private IP detection
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("203.0.113.1") is False
        
        # Test loopback
        assert is_loopback_ip("127.0.0.1") is True
        assert is_loopback_ip("8.8.8.8") is False
        
        print("  ✓ IP utilities work correctly")
        return True
    except Exception as e:
        print(f"  ✗ IP utilities validation failed: {e}")
        return False


def validate_geoip_cache():
    """Validate GeoIP cache functionality."""
    print("[VALIDATION] Checking GeoIP cache...")
    
    try:
        from src.advanced import GeoIPCache, GeoIPData
        
        # Create small cache
        cache = GeoIPCache(max_size=5)
        
        # Create test entry
        data = GeoIPData(
            ip="8.8.8.8",
            country="US",
            country_name="United States",
            region="",
            city="",
            latitude=0,
            longitude=0,
            timezone="UTC",
            postal_code="",
            asn="AS15169",
            as_name="Google",
            isp="Google"
        )
        
        # Put and get
        cache.put("8.8.8.8", data)
        result = cache.get("8.8.8.8")
        assert result is not None
        assert result.country == "US"
        
        # Verify hit
        assert cache.hits > 0
        
        print("  ✓ GeoIP cache works correctly")
        return True
    except Exception as e:
        print(f"  ✗ GeoIP cache validation failed: {e}")
        return False


def validate_geoip_data():
    """Validate GeoIPData structure."""
    print("[VALIDATION] Checking GeoIPData...")
    
    try:
        from src.advanced import GeoIPData
        
        # Create instance
        data = GeoIPData(
            ip="203.0.113.1",
            country="US",
            country_name="United States",
            region="CA",
            city="Mountain View",
            latitude=37.4192,
            longitude=-122.0574,
            timezone="America/Los_Angeles",
            postal_code="94043",
            asn="AS15169",
            as_name="Google Inc.",
            isp="Google",
        )
        
        # Verify fields
        assert data.ip == "203.0.113.1"
        assert data.latitude == 37.4192
        
        # Test dict conversion
        d = data.to_dict()
        assert d["city"] == "Mountain View"
        
        # Test from_dict
        data2 = GeoIPData.from_dict(d)
        assert data2.asn == "AS15169"
        
        print("  ✓ GeoIPData works correctly")
        return True
    except Exception as e:
        print(f"  ✗ GeoIPData validation failed: {e}")
        return False


def validate_risk_detector():
    """Validate risk detection."""
    print("[VALIDATION] Checking risk detector...")
    
    try:
        from src.advanced import RiskDetector, GeoIPData
        
        detector = RiskDetector()
        
        # Add Tor exit node
        detector.add_tor_exit_node("203.0.113.50")
        
        # Create VPN data
        vpn_data = GeoIPData(
            ip="10.0.0.1",
            country="US",
            country_name="United States",
            region="",
            city="",
            latitude=0,
            longitude=0,
            timezone="UTC",
            postal_code="",
            asn="AS39798",  # ExpressVPN
            as_name="ExpressVPN",
            isp="ExpressVPN"
        )
        
        # Analyze
        is_vpn, is_proxy, is_tor, is_dc, risk = detector.analyze(vpn_data)
        assert is_vpn is True
        assert risk > 0.0
        
        # Create Tor data
        tor_data = GeoIPData(
            ip="203.0.113.50",
            country="US",
            country_name="United States",
            region="",
            city="",
            latitude=0,
            longitude=0,
            timezone="UTC",
            postal_code="",
            asn="AS1234",
            as_name="ISP",
            isp="ISP"
        )
        
        # Analyze Tor
        _, _, is_tor2, _, _ = detector.analyze(tor_data)
        assert is_tor2 is True
        
        print("  ✓ Risk detector works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Risk detector validation failed: {e}")
        return False


def validate_geoip_stats():
    """Validate GeoIPStats."""
    print("[VALIDATION] Checking GeoIPStats...")
    
    try:
        from src.advanced import GeoIPStats
        
        # Create stats
        stats = GeoIPStats(
            total_lookups=100,
            cache_hits=80,
            cache_misses=20,
            database_hits=19,
            database_misses=1,
        )
        
        # Verify cache hit rate
        rate = stats.cache_hit_rate
        assert 75 < rate < 85  # Should be ~80%
        
        print("  ✓ GeoIPStats works correctly")
        return True
    except Exception as e:
        print(f"  ✗ GeoIPStats validation failed: {e}")
        return False


def validate_thread_safety():
    """Validate thread-safe operations."""
    print("[VALIDATION] Checking thread safety...")
    
    try:
        import threading
        from src.advanced import GeoIPCache, GeoIPData
        
        cache = GeoIPCache(max_size=100)
        
        data = GeoIPData(
            ip="8.8.8.8",
            country="US",
            country_name="United States",
            region="",
            city="",
            latitude=0,
            longitude=0,
            timezone="UTC",
            postal_code="",
            asn="AS15169",
            as_name="Google",
            isp="Google"
        )
        
        errors = []
        
        def worker():
            try:
                for i in range(50):
                    cache.put(f"8.8.8.{i}", data)
                    result = cache.get(f"8.8.8.{i}")
                    if result is None and i < 10:
                        errors.append("Failed to retrieve")
            except Exception as e:
                errors.append(str(e))
        
        # Run threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        print("  ✓ Thread safety validated")
        return True
    except Exception as e:
        print(f"  ✗ Thread safety validation failed: {e}")
        return False


def validate_eve_integration():
    """Validate EVE JSON integration."""
    print("[VALIDATION] Checking EVE integration...")
    
    try:
        from src.advanced import enrich_eve_event_with_geoip, GeoIPLookup
        
        # Create simple lookup (without database)
        lookup = GeoIPLookup()
        
        # Create EVE event
        event = {
            "src_ip": "10.0.0.1",
            "dest_ip": "203.0.113.1",
        }
        
        # This should not crash
        enrich_eve_event_with_geoip(event, lookup)
        
        # Event should still be valid
        assert "src_ip" in event
        
        print("  ✓ EVE integration works correctly")
        return True
    except Exception as e:
        print(f"  ✗ EVE integration validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("PHASE F PART 1: GEOIP ENRICHMENT - VALIDATION")
    print("="*60 + "\n")
    
    validations = [
        validate_geoip_imports,
        validate_ip_utils,
        validate_geoip_cache,
        validate_geoip_data,
        validate_risk_detector,
        validate_geoip_stats,
        validate_thread_safety,
        validate_eve_integration,
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
