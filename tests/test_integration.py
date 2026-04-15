"""
Integration tests for INIDS
End-to-end testing of:
- Alert filtering pipeline
- Entity enrichment
- Hierarchical aggregation
- Temporal correlation
- API endpoints
"""

import pytest
from datetime import datetime, timedelta, timezone
import json

pytestmark = pytest.mark.integration


class TestAlertFilteringPipeline:
    """Test the three-layer alert filtering pipeline."""
    
    def test_filter_alert_exclude_layer(self):
        """Test EXCLUDE layer blocks alerts completely."""
        from src.ips.alert_filter import ThreeLayerAlertFilter, ExcludeRule
        
        filter_engine = ThreeLayerAlertFilter()
        
        # Add rule to exclude localhost
        rule = ExcludeRule(
            rule_id="test_exclude",
            name="Exclude localhost",
            conditions={"source_ip": "127.0.0.1"},
        )
        filter_engine.add_exclude_rule(rule)
        
        # Alert from localhost
        alert = {
            "id": "test_001",
            "source_ip": "127.0.0.1",
            "attack_type": "port_scan",
        }
        
        result = filter_engine.filter_alert(alert)
        assert result.action.value == "exclude"
    
    def test_filter_alert_ignore_layer(self):
        """Test IGNORE layer deprioritizes alerts."""
        from src.ips.alert_filter import ThreeLayerAlertFilter, IgnoreRule
        
        filter_engine = ThreeLayerAlertFilter()
        
        # Add rule to deprioritize internal scans
        rule = IgnoreRule(
            rule_id="test_ignore",
            name="Ignore internal scans",
            conditions={"attack_type": "network_scan"},
            severity_reduction=2,
        )
        filter_engine.add_ignore_rule(rule)
        
        # Alert for network scan
        alert = {
            "id": "test_001",
            "attack_type": "network_scan",
            "severity": 7,
        }
        
        result = filter_engine.filter_alert(alert)
        assert result.action.value == "ignore"
        assert result.modified_severity == 5  # 7 - 2
    
    def test_filter_alert_merge_layer(self):
        """Test MERGE layer combines similar alerts."""
        from src.ips.alert_filter import ThreeLayerAlertFilter, MergeRule
        
        filter_engine = ThreeLayerAlertFilter()
        
        # Add rule to merge brute force attempts
        rule = MergeRule(
            rule_id="test_merge",
            name="Merge brute force",
            conditions={"attack_type": "brute_force"},
            merge_window_seconds=300,
        )
        filter_engine.add_merge_rule(rule)
        
        # First alert
        alert1 = {
            "id": "test_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_type": "brute_force",
            "source_ip": "192.168.1.100",
        }
        filter_engine.track_alert(alert1)
        
        # Second similar alert within window
        alert2 = {
            "id": "test_002",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_type": "brute_force",
            "source_ip": "192.168.1.100",
        }
        
        result = filter_engine.filter_alert(alert2)
        assert result.action.value == "merge"
        assert result.merged_with_alert_id == "test_001"


class TestEntityEnrichmentPipeline:
    """Test entity enrichment in the detection pipeline."""
    
    def test_entity_enrichment_basic(self, mock_ops_store):
        """Test basic entity enrichment."""
        from src.ips.entity_enrichment import EntityEnrichmentEngine
        
        enricher = EntityEnrichmentEngine(ops_store=mock_ops_store)
        
        enriched = enricher.enrich("192.168.1.100")
        
        assert enriched.ip_address == "192.168.1.100"
        assert enriched.geoip is not None
        assert enriched.threat_intel is not None
        assert enriched.historical is not None
        assert enriched.network is not None
        assert 0 <= enriched.enrichment_confidence <= 1.0
    
    def test_entity_enrichment_threat_level(self, mock_ops_store):
        """Test threat level assessment."""
        from src.ips.entity_enrichment import EntityEnrichmentEngine
        
        enricher = EntityEnrichmentEngine(ops_store=mock_ops_store)
        enriched = enricher.enrich("192.168.1.100")
        
        threat_level = enricher.get_threat_level(enriched)
        assert threat_level in ["low", "medium", "high", "critical"]
    
    def test_entity_enrichment_internal_detection(self):
        """Test internal IP detection."""
        from src.ips.entity_enrichment import EntityEnrichmentEngine
        
        enricher = EntityEnrichmentEngine(
            internal_cidrs=["192.168.0.0/16", "10.0.0.0/8"]
        )
        
        # Test internal IP
        enriched_internal = enricher.enrich("192.168.1.100")
        assert enriched_internal.network.is_internal is True
        
        # Test external IP
        enriched_external = enricher.enrich("8.8.8.8")
        assert enriched_external.network.is_internal is False


class TestIncidentAggregationPipeline:
    """Test complete incident aggregation flow."""
    
    def test_aggregation_single_source_ip(self, mock_ops_store):
        """Test aggregating alerts from single source IP."""
        from src.ips.incident_aggregator import IncidentAggregator
        
        agg = IncidentAggregator(mock_ops_store)
        
        source_ip = "192.168.1.100"
        incident_ids = []
        
        # Generate 3 alerts from same IP
        for i in range(3):
            activity_id, incident_id = agg.aggregate_alert(
                alert_id=f"al_{i:03d}",
                alert_code=f"CODE_ALERT_{i}",
                source_ip=source_ip,
                attack_type="port_scan",
                severity=7,
                timestamp=(datetime.now(timezone.utc) + timedelta(minutes=i)).isoformat(),
                description=f"Alert {i}"
            )
            incident_ids.append(incident_id)
        
        # All should be grouped in same incident
        assert incident_ids[0] == incident_ids[1] == incident_ids[2]
    
    def test_aggregation_multiple_source_ips(self, mock_ops_store):
        """Test alerts from different IPs create different incidents."""
        from src.ips.incident_aggregator import IncidentAggregator
        
        agg = IncidentAggregator(mock_ops_store)
        
        incident_ids = []
        
        # Alerts from different IPs
        for i in range(2):
            source_ip = f"192.168.1.{100 + i}"
            _, incident_id = agg.aggregate_alert(
                alert_id=f"al_{i:03d}",
                alert_code=f"CODE_ALERT_{i}",
                source_ip=source_ip,
                attack_type="port_scan",
                severity=7,
                timestamp=datetime.now(timezone.utc).isoformat(),
                description=f"Alert {i}"
            )
            incident_ids.append(incident_id)
        
        # Should be different incidents
        assert incident_ids[0] != incident_ids[1]


class TestTemporalCorrelationPipeline:
    """Test multi-stage attack correlation."""
    
    def test_sequential_attack_pattern(self):
        """Test detection of port scan followed by brute force."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        
        engine.register_pattern(
            "scan_to_brute",
            [
                {"type": "port_scan", "confidence_min": 0.7},
                {"type": "brute_force", "confidence_min": 0.8, "time_offset_seconds": 600}
            ]
        )
        
        now = datetime.now(timezone.utc)
        source_ip = "192.168.1.100"
        
        # Stage 1: Port scan
        result1 = engine.evaluate({
            "type": "port_scan",
            "source_ip": source_ip,
            "confidence": 0.85,
            "timestamp": now.isoformat(),
        })
        assert result1 is None  # Single event doesn't trigger pattern
        
        # Stage 2: Brute force within 10 minutes
        result2 = engine.evaluate({
            "type": "brute_force",
            "source_ip": source_ip,
            "confidence": 0.90,
            "timestamp": (now + timedelta(minutes=5)).isoformat(),
        })
        assert result2 is not None  # Pattern matched!
        assert result2[0] == "scan_to_brute"
    
    def test_time_window_violation(self):
        """Test that attacks outside time window don't match."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        
        engine.register_pattern(
            "quick_pattern",
            [
                {"type": "event_a", "confidence_min": 0.7},
                {"type": "event_b", "confidence_min": 0.7, "time_offset_seconds": 60}
            ]
        )
        
        now = datetime.now(timezone.utc)
        source_ip = "192.168.1.100"
        
        # First event
        engine.evaluate({
            "type": "event_a",
            "source_ip": source_ip,
            "confidence": 0.8,
            "timestamp": now.isoformat(),
        })
        
        # Second event OUTSIDE window (2 minutes later)
        result = engine.evaluate({
            "type": "event_b",
            "source_ip": source_ip,
            "confidence": 0.8,
            "timestamp": (now + timedelta(minutes=2)).isoformat(),
        })
        
        # Should NOT match because outside time window
        assert result is None


class TestEndToEndAlertFlow:
    """Test complete alert flow through all layers."""
    
    def test_complete_alert_lifecycle(self, mock_ops_store):
        """Test alert from detection through enrichment and aggregation."""
        from src.ips.alert_filter import ThreeLayerAlertFilter
        from src.ips.entity_enrichment import EntityEnrichmentEngine
        from src.ips.incident_aggregator import IncidentAggregator
        
        # Setup all components
        alert_filter = ThreeLayerAlertFilter(mock_ops_store)
        enricher = EntityEnrichmentEngine(mock_ops_store)
        aggregator = IncidentAggregator(mock_ops_store)
        
        # Create alert
        alert = {
            "id": "end_to_end_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "10.0.0.100",
            "attack_type": "brute_force",
            "severity": 8,
            "confidence": 0.95,
        }
        
        # Step 1: Filter
        filter_result = alert_filter.filter_alert(alert)
        assert filter_result.action.value != "exclude"  # Should pass through
        
        # Step 2: Enrich
        enriched_entity = enricher.enrich(alert["source_ip"])
        alert["enriched_entity"] = enriched_entity.to_dict()
        alert["threat_level"] = enricher.get_threat_level(enriched_entity)
        
        # Step 3: Aggregate
        activity_id, incident_id = aggregator.aggregate_alert(
            alert_id=alert["id"],
            alert_code="CODE_BRUTEFORCE",
            source_ip=alert["source_ip"],
            attack_type=alert["attack_type"],
            severity=alert["severity"],
            timestamp=alert["timestamp"],
            description="Brute force attack"
        )
        
        assert activity_id is not None
        assert incident_id is not None


class TestRuleEngineIntegration:
    """Test rule compiler integrated with signature engine."""
    
    def test_compiled_rule_evaluation(self):
        """Test that compiled rules evaluate correctly."""
        from src.detection.rule_compiler import RuleCompiler
        
        compiler = RuleCompiler()
        
        # Complex rule with multiple conditions
        rule = {
            "conditions": {
                "logic": "AND",
                "source_ip": "regex:^192\\.168\\.",
                "destination_port": "range:1-1024",
                "severity": ">= 7"
            }
        }
        
        compiled = compiler.compile_rule(rule)
        assert compiled is not None
        
        # This would normally be used in signature engine
        # We're just testing it compiles without errors


class TestAPIEndpointIntegration:
    """Test API endpoint functionality."""
    
    def test_can_import_detection_event(self):
        """Test importing detection event type."""
        from src.core.event_bus import DetectionEvent
        assert DetectionEvent is not None
    
    def test_can_create_detection_event(self):
        """Test creating detection event."""
        from src.core.event_bus import DetectionEvent
        
        event = DetectionEvent(
            source_ip="192.168.1.100",
            prediction="attack",
            confidence=0.95,
            severity=8,
            attack_type="port_scan",
            profile="scanner",
            reason="Multiple ports scanned",
            timestamp=datetime.now(timezone.utc).isoformat(),
            features={},
        )
        
        assert event.source_ip == "192.168.1.100"
        assert event.prediction == "attack"


class TestDatabaseOperations:
    """Test database operations for persistence."""
    
    def test_save_and_retrieve_alert(self, mock_ops_store):
        """Test saving and retrieving alerts."""
        alert = {
            "id": "db_test_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": 7,
            "prediction": "attack",
            "confidence": 0.85,
            "profile": "port_scan",
            "reason": "Port scan detected",
            "source_ip": "192.168.1.100",
            "attack_type": "network_scan",
        }
        
        # Save
        mock_ops_store.save_alert(alert)
        
        # Retrieve
        retrieved = mock_ops_store._fetchone(
            "SELECT * FROM alerts WHERE id = ?",
            ("db_test_001",)
        )
        
        assert retrieved is not None
        assert retrieved["source_ip"] == "192.168.1.100"
        assert retrieved["attack_type"] == "network_scan"
    
    def test_schema_exists(self, mock_ops_store):
        """Test that all required tables exist."""
        tables = ["alerts", "activities", "incidents", "alert_filter_rules"]
        
        for table in tables:
            result = mock_ops_store._fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            assert result is not None, f"Table {table} doesn't exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
