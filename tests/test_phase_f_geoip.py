"""
Phase F Tests: GeoIP Enrichment

Comprehensive tests for GeoIP lookup, caching, risk detection, and EVE integration.
No pytest dependency - runs standalone.
"""

import sys
import time
import json
import tempfile
from pathlib import Path


def test_geoip_data():
    """Test GeoIPData structure."""
    print("[TEST] GeoIPData creation...")
    
    from src.advanced import GeoIPData
    
    # Create GeoIPData
    data = GeoIPData(
        ip="203.0.113.42",
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
    assert data.ip == "203.0.113.42"
    assert data.country == "US"
    assert data.city == "Mountain View"
    
    # Test dictionary conversion
    d = data.to_dict()
    assert d["ip"] == "203.0.113.42"
    
    # Test from_dict
    data2 = GeoIPData.from_dict(d)
    assert data2.city == "Mountain View"
    
    print("  ✓ GeoIPData works correctly")
    return True


def test_ip_utilities():
    """Test IP conversion utilities."""
    print("[TEST] IP utilities...")
    
    from src.advanced import ip_to_int, int_to_ip, is_private_ip, is_loopback_ip
    
    # Test IP to int
    assert ip_to_int("192.0.2.1") > 0
    assert ip_to_int("10.0.0.1") > 0
    
    # Test int to IP
    ip_int = ip_to_int("203.0.113.42")
    ip_str = int_to_ip(ip_int)
    assert ip_str == "203.0.113.42"
    
    # Test private IP detection
    assert is_private_ip("10.0.0.1") is True
    assert is_private_ip("192.168.1.1") is True
    assert is_private_ip("172.16.0.1") is True
    assert is_private_ip("203.0.113.1") is False
    
    # Test loopback detection
    assert is_loopback_ip("127.0.0.1") is True
    assert is_loopback_ip("203.0.113.1") is False
    
    print("  ✓ IP utilities work correctly")
    return True


def test_geoip_cache():
    """Test GeoIP cache with LRU eviction."""
    print("[TEST] GeoIP cache...")
    
    from src.advanced import GeoIPCache, GeoIPData
    
    # Create cache
    cache = GeoIPCache(max_size=3, ttl_seconds=1000)
    
    # Create test data
    data1 = GeoIPData(ip="10.0.0.1", country="US", country_name="United States",
                      region="", city="", latitude=0, longitude=0, timezone="UTC",
                      postal_code="", asn="", as_name="", isp="")
    data2 = GeoIPData(ip="10.0.0.2", country="GB", country_name="United Kingdom",
                      region="", city="", latitude=0, longitude=0, timezone="UTC",
                      postal_code="", asn="", as_name="", isp="")
    data3 = GeoIPData(ip="10.0.0.3", country="DE", country_name="Germany",
                      region="", city="", latitude=0, longitude=0, timezone="UTC",
                      postal_code="", asn="", as_name="", isp="")
    data4 = GeoIPData(ip="10.0.0.4", country="FR", country_name="France",
                      region="", city="", latitude=0, longitude=0, timezone="UTC",
                      postal_code="", asn="", as_name="", isp="")
    
    # Add to cache
    cache.put("10.0.0.1", data1)
    cache.put("10.0.0.2", data2)
    cache.put("10.0.0.3", data3)
    
    # Verify size
    assert cache.size() == 3
    
    # Add fourth item (should evict first)
    cache.put("10.0.0.4", data4)
    
    # Verify size and eviction
    assert cache.size() == 3
    assert cache.evictions >= 1
    
    # Verify get
    result = cache.get("10.0.0.2")
    assert result is not None
    assert result.country == "GB"
    
    # Verify hit rate
    assert cache.hit_rate() > 0
    
    # Test TTL expiry
    cache2 = GeoIPCache(max_size=10, ttl_seconds=1)
    cache2.put("10.0.0.1", data1)
    time.sleep(1.1)
    result = cache2.get("10.0.0.1")
    assert result is None  # Should have expired
    
    print("  ✓ GeoIP cache works correctly")
    return True


def test_geoip_database():
    """Test GeoIP database with file loading."""
    print("[TEST] GeoIP database...")
    
    from src.advanced import GeoIPDatabase
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_data = {
            "entries": [
                {
                    "prefix_start": 3232235776,  # 192.0.2.0
                    "prefix_end": 3232235777,    # 192.0.2.1
                    "country": "US",
                    "country_name": "United States",
                    "region": "CA",
                    "city": "Mountain View",
                    "latitude": 37.4192,
                    "longitude": -122.0574,
                    "timezone": "America/Los_Angeles",
                    "postal_code": "94043",
                    "asn": "AS15169",
                    "as_name": "Google Inc.",
                    "isp": "Google"
                }
            ]
        }
        json.dump(db_data, f)
        temp_path = f.name
    
    try:
        # Load database
        db = GeoIPDatabase()
        loaded = db.load_from_file(temp_path)
        assert loaded is True
        
        # Lookup IP
        result = db.lookup("192.0.2.1")
        assert result is not None
        assert result.country == "US"
        assert result.city == "Mountain View"
        
        # Non-existent IP
        result = db.lookup("203.0.113.1")
        assert result is None
        
        print("  ✓ GeoIP database works correctly")
        return True
    finally:
        Path(temp_path).unlink()


def test_risk_detector():
    """Test risk detection for VPN, proxy, Tor, datacenter."""
    print("[TEST] Risk detector...")
    
    from src.advanced import RiskDetector, GeoIPData
    
    # Create detector
    detector = RiskDetector()
    
    # Add Tor exit node
    detector.add_tor_exit_node("203.0.113.10")
    
    # Create test data
    vpn_data = GeoIPData(ip="10.0.0.1", country="US", country_name="United States",
                         region="", city="", latitude=0, longitude=0, timezone="UTC",
                         postal_code="", asn="AS39798", as_name="ExpressVPN",
                         isp="ExpressVPN")
    
    # Analyze VPN
    is_vpn, is_proxy, is_tor, is_dc, risk = detector.analyze(vpn_data)
    assert is_vpn is True
    assert risk > 0.0
    
    # Create Tor data
    tor_data = GeoIPData(ip="203.0.113.10", country="US", country_name="United States",
                         region="", city="", latitude=0, longitude=0, timezone="UTC",
                         postal_code="", asn="AS1234", as_name="ISP",
                         isp="ISP")
    
    # Analyze Tor
    is_vpn2, is_proxy2, is_tor2, is_dc2, risk2 = detector.analyze(tor_data)
    assert is_tor2 is True
    assert risk2 > 0.0
    
    print("  ✓ Risk detector works correctly")
    return True


def test_geoip_lookup():
    """Test main GeoIP lookup service."""
    print("[TEST] GeoIP lookup...")
    
    from src.advanced import GeoIPLookup, GeoIPDatabase
    import json
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_data = {
            "entries": [
                {
                    "prefix_start": 3232235776,  # 192.0.2.0
                    "prefix_end": 3232235777,    # 192.0.2.1
                    "country": "US",
                    "country_name": "United States",
                    "region": "CA",
                    "city": "Mountain View",
                    "latitude": 37.4192,
                    "longitude": -122.0574,
                    "timezone": "America/Los_Angeles",
                    "postal_code": "94043",
                    "asn": "AS15169",
                    "as_name": "Google Inc.",
                    "isp": "Google"
                }
            ]
        }
        json.dump(db_data, f)
        temp_path = f.name
    
    try:
        # Create lookup
        lookup = GeoIPLookup(database_path=temp_path, cache_size=100)
        
        # Lookup IP
        result = lookup.lookup("192.0.2.1")
        assert result is not None
        assert result.country == "US"
        
        # Verify cache hit on second lookup
        result2 = lookup.lookup("192.0.2.1")
        assert result2 is not None
        
        # Get statistics
        stats = lookup.get_stats()
        assert stats.cache_hits > 0
        assert stats.total_lookups >= 2
        
        # Skip private IP
        result3 = lookup.lookup("10.0.0.1")
        assert result3 is None
        
        # Skip loopback
        result4 = lookup.lookup("127.0.0.1")
        assert result4 is None
        
        print("  ✓ GeoIP lookup works correctly")
        return True
    finally:
        Path(temp_path).unlink()


def test_geoip_lookup_bulk():
    """Test bulk IP lookup."""
    print("[TEST] GeoIP bulk lookup...")
    
    from src.advanced import GeoIPLookup
    import json
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_data = {
            "entries": [
                {
                    "prefix_start": 3232235776,  # 192.0.2.0
                    "prefix_end": 3232235777,    # 192.0.2.1
                    "country": "US",
                    "country_name": "United States",
                    "region": "CA",
                    "city": "Mountain View",
                    "latitude": 37.4192,
                    "longitude": -122.0574,
                    "timezone": "America/Los_Angeles",
                    "postal_code": "94043",
                    "asn": "AS15169",
                    "as_name": "Google Inc.",
                    "isp": "Google"
                }
            ]
        }
        json.dump(db_data, f)
        temp_path = f.name
    
    try:
        # Create lookup
        lookup = GeoIPLookup(database_path=temp_path)
        
        # Bulk lookup
        ips = ["192.0.2.1", "203.0.113.1", "10.0.0.1"]
        results = lookup.lookup_bulk(ips)
        
        assert len(results) == 3
        assert results["192.0.2.1"] is not None
        assert results["203.0.113.1"] is None
        assert results["10.0.0.1"] is None
        
        print("  ✓ GeoIP bulk lookup works correctly")
        return True
    finally:
        Path(temp_path).unlink()


def test_eve_enrichment():
    """Test EVE JSON enrichment with GeoIP."""
    print("[TEST] EVE enrichment...")
    
    from src.advanced import GeoIPLookup, enrich_eve_event_with_geoip
    import json
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_data = {
            "entries": [
                {
                    "prefix_start": 3232235776,
                    "prefix_end": 3232235777,
                    "country": "US",
                    "country_name": "United States",
                    "region": "CA",
                    "city": "Mountain View",
                    "latitude": 37.4192,
                    "longitude": -122.0574,
                    "timezone": "America/Los_Angeles",
                    "postal_code": "94043",
                    "asn": "AS15169",
                    "as_name": "Google Inc.",
                    "isp": "Google"
                }
            ]
        }
        json.dump(db_data, f)
        temp_path = f.name
    
    try:
        # Create lookup
        lookup = GeoIPLookup(database_path=temp_path)
        
        # Create EVE event
        event = {
            "src_ip": "192.0.2.1",
            "dest_ip": "203.0.113.1",
            "protocol": "tcp",
        }
        
        # Enrich event
        enrich_eve_event_with_geoip(event, lookup)
        
        # Verify enrichment
        assert "geoip_source" in event
        assert event["geoip_source"]["country"] == "US"
        assert "geoip_dest" not in event  # Not found in database
        
        print("  ✓ EVE enrichment works correctly")
        return True
    finally:
        Path(temp_path).unlink()


def test_global_singleton():
    """Test global GeoIP singleton."""
    print("[TEST] Global singleton...")
    
    from src.advanced import get_geoip_lookup
    
    # Get first instance
    lookup1 = get_geoip_lookup()
    
    # Get second instance
    lookup2 = get_geoip_lookup()
    
    # Should be same instance
    assert lookup1 is lookup2
    
    print("  ✓ Global singleton works correctly")
    return True


def test_statistics():
    """Test GeoIP statistics and performance."""
    print("[TEST] Statistics and performance...")
    
    from src.advanced import GeoIPLookup, GeoIPCache
    import json
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_data = {
            "entries": [
                {
                    "prefix_start": 3232235776,
                    "prefix_end": 3232235777,
                    "country": "US",
                    "country_name": "United States",
                    "region": "CA",
                    "city": "Mountain View",
                    "latitude": 37.4192,
                    "longitude": -122.0574,
                    "timezone": "America/Los_Angeles",
                    "postal_code": "94043",
                    "asn": "AS15169",
                    "as_name": "Google Inc.",
                    "isp": "Google"
                }
            ]
        }
        json.dump(db_data, f)
        temp_path = f.name
    
    try:
        # Create lookup
        lookup = GeoIPLookup(database_path=temp_path)
        
        # Perform multiple lookups
        for _ in range(10):
            lookup.lookup("192.0.2.1")
        
        # Get statistics
        stats = lookup.get_stats()
        
        assert stats.total_lookups == 10
        assert stats.cache_hits >= 9  # First miss, rest hits
        assert stats.database_hits >= 1
        
        # Verify cache hit rate
        assert stats.cache_hit_rate >= 80.0
        
        print("  ✓ Statistics and performance work correctly")
        return True
    finally:
        Path(temp_path).unlink()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE F: GEOIP ENRICHMENT - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_geoip_data,
        test_ip_utilities,
        test_geoip_cache,
        test_geoip_database,
        test_risk_detector,
        test_geoip_lookup,
        test_geoip_lookup_bulk,
        test_eve_enrichment,
        test_global_singleton,
        test_statistics,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("STATUS: ✓ ALL TESTS PASSED")
    else:
        print(f"STATUS: ✗ {failed} TESTS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
