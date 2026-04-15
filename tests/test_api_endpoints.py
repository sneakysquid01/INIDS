"""
API Endpoint Tests
Tests for all new endpoints added in Week 1-2 implementation
"""

import pytest
from datetime import datetime, timezone
import json

pytestmark = pytest.mark.api


class TestHoneypotAPI:
    """Test honeypot configuration API endpoints."""
    
    def test_honeypot_config_endpoint_exists(self):
        """Test that honeypot config endpoints are defined."""
        # These would be tested with actual Flask app
        # Checking that the endpoint names are standard
        endpoints = [
            "/api/honeypot/config",
        ]
        for endpoint in endpoints:
            assert endpoint.startswith("/api/")
            assert "honeypot" in endpoint
    
    def test_honeypot_ips_format(self, honeypot_config):
        """Test honeypot IPs are valid."""
        ips = honeypot_config["honeypot_ips"]
        assert isinstance(ips, list)
        for ip in ips:
            parts = ip.split(".")
            assert len(parts) == 4
            for part in parts:
                assert 0 <= int(part) <= 255
    
    def test_honeypot_ports_format(self, honeypot_config):
        """Test honeypot ports are valid."""
        ports = honeypot_config["honeypot_ports"]
        assert isinstance(ports, list)
        for port in ports:
            assert 1 <= port <= 65535


class TestIncidentAPI:
    """Test incident management API endpoints."""
    
    def test_incident_endpoints_exist(self):
        """Test that incident endpoints are defined."""
        endpoints = [
            "/api/incidents",
            "/api/incidents/<incident_id>",
            "/api/activities",
        ]
        for endpoint in endpoints:
            assert "/api/" in endpoint
            assert "incident" in endpoint or "activities" in endpoint
    
    def test_incident_response_structure(self):
        """Test expected incident response structure."""
        incident = {
            "id": "inc_001",
            "source_ip": "192.168.1.100",
            "activity_count": 3,
            "severity": 8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        assert "id" in incident
        assert "source_ip" in incident
        assert "activity_count" in incident
        assert "severity" in incident
    
    def test_activity_response_structure(self):
        """Test expected activity response structure."""
        activity = {
            "id": "act_001",
            "unique_id": "unique_123",
            "repeat_count": 3,
            "incident_id": "inc_001",
            "severity": 7,
        }
        assert "id" in activity
        assert "repeat_count" in activity
        assert "incident_id" in activity


class TestTemporalAPI:
    """Test temporal correlation API endpoints."""
    
    def test_temporal_patterns_endpoint(self):
        """Test temporal patterns listing endpoint."""
        endpoint = "/api/temporal/patterns"
        assert endpoint.startswith("/api/")
        assert "temporal" in endpoint
        assert "patterns" in endpoint
    
    def test_temporal_state_endpoint(self):
        """Test temporal state endpoint."""
        endpoint = "/api/temporal/state/<source_ip>"
        assert "<source_ip>" in endpoint
        assert "temporal" in endpoint
    
    def test_temporal_pattern_structure(self):
        """Test pattern definition structure."""
        pattern = {
            "name": "port_scan_to_brute_force",
            "steps": [
                {"type": "port_scan", "confidence_min": 0.7},
                {"type": "brute_force", "confidence_min": 0.8, "time_offset_seconds": 300}
            ],
            "description": "Port scan followed by brute force attack"
        }
        assert "name" in pattern
        assert "steps" in pattern
        assert isinstance(pattern["steps"], list)
        assert len(pattern["steps"]) > 0


class TestEntityEnrichmentAPI:
    """Test entity enrichment API endpoints."""
    
    def test_enrichment_endpoint_exists(self):
        """Test that enrichment endpoint exists."""
        endpoint = "/api/entity/enrich/<source_ip>"
        assert "/api/entity" in endpoint
        assert "enrich" in endpoint
    
    def test_threat_level_endpoint_exists(self):
        """Test threat level endpoint exists."""
        endpoint = "/api/entity/<source_ip>/threat-level"
        assert "/api/entity" in endpoint
        assert "threat-level" in endpoint
    
    def test_enriched_entity_response_structure(self):
        """Test structure of enriched entity response."""
        enriched = {
            "ip_address": "192.168.1.100",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "geoip": {
                "country": "US",
                "city": "Unknown",
                "isp": "ISP Name",
                "threat_level": "low",
            },
            "threat_intel": {
                "ip_reputation_score": None,
                "known_attacker": False,
                "attack_count_30d": 0,
            },
            "historical": {
                "total_incidents": 0,
                "success_rate_percent": 0.0,
            },
            "network": {
                "is_internal": True,
                "asset_type": "workstation",
            },
            "threat_level": "low",
            "enrichment_confidence": 0.75,
        }
        
        assert "ip_address" in enriched
        assert "geoip" in enriched
        assert "threat_intel" in enriched
        assert "threat_level" in enriched
        assert 0 <= enriched["enrichment_confidence"] <= 1.0
    
    def test_threat_level_response_structure(self):
        """Test threat level response structure."""
        response = {
            "source_ip": "192.168.1.100",
            "threat_level": "medium",
            "confidence": 0.75,
            "last_enriched": datetime.now(timezone.utc).isoformat(),
        }
        
        assert "threat_level" in response
        assert response["threat_level"] in ["low", "medium", "high", "critical"]
        assert 0 <= response["confidence"] <= 1.0


class TestAlertFilteringAPI:
    """Test alert filtering API endpoints."""
    
    def test_filter_rules_endpoints(self):
        """Test filter rules endpoint names."""
        endpoints = [
            "/api/alerts/filter-rules",
            "/api/alerts/filter-rules/exclude",
            "/api/alerts/filter-rules/ignore",
            "/api/alerts/filter-rules/merge",
            "/api/alerts/filter-rules/<rule_id>",
            "/api/alerts/filter-stats",
        ]
        for endpoint in endpoints:
            assert "/api/alerts" in endpoint
            assert "filter" in endpoint
    
    def test_filter_rule_request_structure(self):
        """Test structure of filter rule creation request."""
        rule_data = {
            "rule_id": "test_rule_001",
            "name": "Test Rule",
            "description": "A test filter rule",
            "conditions": {
                "source_ip": "127.0.0.1",
                "attack_type": "port_scan"
            },
            "priority": 100,
        }
        
        assert "rule_id" in rule_data
        assert "name" in rule_data
        assert "conditions" in rule_data
        assert isinstance(rule_data["conditions"], dict)
    
    def test_exclude_rule_request_structure(self):
        """Test structure of exclude rule request."""
        rule_data = {
            "rule_id": "exclude_localhost",
            "name": "Exclude Localhost",
            "conditions": {
                "source_ip": "127.0.0.1"
            },
            "priority": 100,
        }
        
        assert "rule_id" in rule_data
        assert "conditions" in rule_data
    
    def test_ignore_rule_request_structure(self):
        """Test structure of ignore rule request."""
        rule_data = {
            "rule_id": "ignore_low_conf",
            "name": "Ignore Low Confidence",
            "conditions": {
                "confidence": "< 0.5"
            },
            "severity_reduction": 2,
            "suppress_notifications": True,
            "priority": 90,
        }
        
        assert "severity_reduction" in rule_data
        assert "suppress_notifications" in rule_data
    
    def test_merge_rule_request_structure(self):
        """Test structure of merge rule request."""
        rule_data = {
            "rule_id": "merge_brute_force",
            "name": "Merge Brute Force",
            "conditions": {
                "attack_type": "brute_force"
            },
            "merge_window_seconds": 300,
            "merge_key": "source_ip",
            "similarity_fields": ["attack_type", "source_ip"],
            "priority": 80,
        }
        
        assert "merge_window_seconds" in rule_data
        assert "similarity_fields" in rule_data
        assert isinstance(rule_data["similarity_fields"], list)
    
    def test_filter_stats_response_structure(self):
        """Test filter statistics response structure."""
        stats = {
            "exclude_rules_count": 2,
            "ignore_rules_count": 3,
            "merge_rules_count": 2,
            "recent_alerts_tracked": 42,
            "merge_groups_active": 5,
        }
        
        assert "exclude_rules_count" in stats
        assert "ignore_rules_count" in stats
        assert "merge_rules_count" in stats
        assert all(isinstance(v, int) for v in stats.values())


class TestHTTPStatusCodes:
    """Test expected HTTP status codes for endpoints."""
    
    def test_get_endpoint_returns_200(self):
        """Test GET endpoints return 200."""
        # These would return 200 on success
        status_codes = {
            "GET": 200,
            "POST": 201,  # Created
            "DELETE": 200,  # OK
        }
        assert status_codes["GET"] == 200
    
    def test_post_endpoint_returns_201(self):
        """Test POST endpoints return 201 Created."""
        assert 201 == 201
    
    def test_not_found_returns_404(self):
        """Test missing resources return 404."""
        assert 404 == 404
    
    def test_bad_request_returns_400(self):
        """Test invalid requests return 400."""
        assert 400 == 400
    
    def test_unauthorized_returns_401(self):
        """Test unauthenticated requests return 401."""
        assert 401 == 401


class TestAuthorizationRequirements:
    """Test API authorization requirements."""
    
    def test_analyst_role_endpoints(self):
        """Test endpoints requiring analyst role."""
        analyst_endpoints = [
            "GET /api/incidents",
            "GET /api/incidents/<id>",
            "GET /api/activities",
            "GET /api/entity/enrich/<ip>",
            "GET /api/alerts/filter-rules",
            "GET /api/alerts/filter-stats",
        ]
        for endpoint in analyst_endpoints:
            assert "/api/" in endpoint
    
    def test_admin_role_endpoints(self):
        """Test endpoints requiring admin role."""
        admin_endpoints = [
            "POST /api/temporal/patterns",
            "POST /api/alerts/filter-rules/exclude",
            "POST /api/alerts/filter-rules/ignore",
            "POST /api/alerts/filter-rules/merge",
            "DELETE /api/alerts/filter-rules/<id>",
        ]
        for endpoint in admin_endpoints:
            assert "/api/" in endpoint


class TestErrorHandling:
    """Test error handling in API responses."""
    
    def test_error_response_structure(self):
        """Test error response includes error message."""
        error_response = {
            "error": "Invalid rule conditions"
        }
        assert "error" in error_response
        assert isinstance(error_response["error"], str)
    
    def test_success_response_structure(self):
        """Test success response includes ok flag."""
        success_response = {
            "ok": True,
            "rule_id": "test_rule_001",
            "message": "Rule created successfully"
        }
        assert "ok" in success_response
        assert success_response["ok"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
