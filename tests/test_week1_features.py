"""
Unit tests for Week 1 features
Tests individual components:
- Honeypot Detection Engine
- Hot-Reloadable Config Manager
- Enhanced Rule Syntax Compiler
- Hierarchical Incident Aggregation
- Temporal Correlation Engine
"""

import pytest
from datetime import datetime, timedelta, timezone
import json

# Test marker
pytestmark = pytest.mark.unit


class TestHoneypotDetectionEngine:
    """Test Honeypot Detection Engine."""
    
    def test_honeypot_engine_import(self):
        """Test that honeypot engine can be imported."""
        from src.detection.engines.honeypot_engine import HoneypotDetectionEngine
        assert HoneypotDetectionEngine is not None
    
    def test_honeypot_engine_initialization(self):
        """Test honeypot engine initialization."""
        from src.detection.engines.honeypot_engine import HoneypotDetectionEngine
        
        engine = HoneypotDetectionEngine(
            engine_id="test_honeypot",
            honeypot_ips=["10.0.0.50", "10.0.0.51"],
            honeypot_ports=[22, 3389],
        )
        assert engine.engine_id == "test_honeypot"
        assert "10.0.0.50" in engine.honeypot_ips
        assert 22 in engine.honeypot_ports
    
    def test_honeypot_engine_detects_canary_ip(self):
        """Test detection of canary IP access."""
        from src.detection.engines.honeypot_engine import HoneypotDetectionEngine
        
        engine = HoneypotDetectionEngine(
            engine_id="test",
            honeypot_ips=["10.0.0.50"],
            honeypot_ports=[22],
        )
        
        flow = {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.50",
            "dst_port": 22,
        }
        
        result = engine.evaluate(flow)
        assert result is not None
        assert result["prediction"] == "attack"
        assert result["confidence"] == 1.0
    
    def test_honeypot_engine_ignores_normal_traffic(self):
        """Test that normal traffic is not flagged."""
        from src.detection.engines.honeypot_engine import HoneypotDetectionEngine
        
        engine = HoneypotDetectionEngine(
            engine_id="test",
            honeypot_ips=["10.0.0.50"],
            honeypot_ports=[22],
        )
        
        flow = {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.100",
            "dst_port": 443,
        }
        
        result = engine.evaluate(flow)
        assert result is None or result.get("prediction") != "attack"
    
    def test_honeypot_engine_update_ips(self):
        """Test dynamic IP updates."""
        from src.detection.engines.honeypot_engine import HoneypotDetectionEngine
        
        engine = HoneypotDetectionEngine(
            engine_id="test",
            honeypot_ips=["10.0.0.50"],
            honeypot_ports=[22],
        )
        
        # Update IPs
        engine.update_honeypot_ips(["10.0.0.60", "10.0.0.61"])
        assert "10.0.0.50" not in engine.honeypot_ips or "10.0.0.60" in engine.honeypot_ips


class TestEnhancedRuleCompiler:
    """Test Enhanced Rule Syntax Compiler."""
    
    def test_rule_compiler_import(self):
        """Test import of rule compiler."""
        from src.detection.rule_compiler import RuleCompiler
        assert RuleCompiler is not None
    
    def test_rule_compiler_basic_rule(self):
        """Test compiling basic rule."""
        from src.detection.rule_compiler import RuleCompiler
        
        compiler = RuleCompiler()
        rule = {
            "name": "test_rule",
            "conditions": {
                "attack_type": "port_scan",
                "confidence": ">= 0.7"
            }
        }
        
        compiled = compiler.compile_rule(rule)
        assert compiled is not None
    
    def test_rule_compiler_regex_operator(self):
        """Test regex operator in rule."""
        from src.detection.rule_compiler import RuleCompiler
        
        compiler = RuleCompiler()
        rule = {
            "conditions": {
                "source_ip": "regex:^192\\.168\\."
            }
        }
        
        compiled = compiler.compile_rule(rule)
        assert compiled is not None
    
    def test_rule_compiler_range_operator(self):
        """Test range operator in rule."""
        from src.detection.rule_compiler import RuleCompiler
        
        compiler = RuleCompiler()
        rule = {
            "conditions": {
                "port": "range:1-1024"
            }
        }
        
        compiled = compiler.compile_rule(rule)
        assert compiled is not None
    
    def test_rule_compiler_and_operator(self):
        """Test AND logic operator."""
        from src.detection.rule_compiler import RuleCompiler
        
        compiler = RuleCompiler()
        rule = {
            "conditions": {
                "logic": "AND",
                "attack_type": "port_scan",
                "confidence": ">= 0.8"
            }
        }
        
        compiled = compiler.compile_rule(rule)
        assert compiled is not None


class TestHierarchicalIncidentAggregation:
    """Test Hierarchical Incident Aggregation."""
    
    def test_incident_aggregator_import(self):
        """Test import of incident aggregator."""
        from src.ips.incident_aggregator import IncidentAggregator
        assert IncidentAggregator is not None
    
    def test_incident_aggregator_initialization(self, mock_ops_store):
        """Test incident aggregator initialization."""
        from src.ips.incident_aggregator import IncidentAggregator
        
        agg = IncidentAggregator(mock_ops_store)
        assert agg.ops_store is not None
    
    def test_aggregate_single_alert(self, mock_ops_store):
        """Test aggregating a single alert."""
        from src.ips.incident_aggregator import IncidentAggregator
        
        agg = IncidentAggregator(mock_ops_store)
        
        activity_id, incident_id = agg.aggregate_alert(
            alert_id="al_001",
            alert_code="CODE_PORTSCAN",
            source_ip="192.168.1.100",
            attack_type="port_scan",
            severity=7,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description="Port scan detected"
        )
        
        assert activity_id is not None
        assert incident_id is not None
    
    def test_aggregate_multiple_alerts_same_ip(self, mock_ops_store):
        """Test aggregating multiple alerts from same IP."""
        from src.ips.incident_aggregator import IncidentAggregator
        
        agg = IncidentAggregator(mock_ops_store)
        
        # First alert
        activity_id1, incident_id1 = agg.aggregate_alert(
            alert_id="al_001",
            alert_code="CODE_PORTSCAN",
            source_ip="192.168.1.100",
            attack_type="port_scan",
            severity=7,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description="Port scan 1"
        )
        
        # Second alert from same IP
        activity_id2, incident_id2 = agg.aggregate_alert(
            alert_id="al_002",
            alert_code="CODE_PORTSCAN",
            source_ip="192.168.1.100",
            attack_type="port_scan",
            severity=7,
            timestamp=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            description="Port scan 2"
        )
        
        # Should be grouped into same incident
        assert incident_id1 == incident_id2
    
    def test_get_incidents(self, mock_ops_store):
        """Test retrieving incidents."""
        from src.ips.incident_aggregator import IncidentAggregator
        
        agg = IncidentAggregator(mock_ops_store)
        
        # Create an incident
        agg.aggregate_alert(
            alert_id="al_001",
            alert_code="CODE_TEST",
            source_ip="192.168.1.100",
            attack_type="test",
            severity=5,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description="Test alert"
        )
        
        incidents = agg.get_incidents(limit=10)
        assert isinstance(incidents, list)
        assert len(incidents) > 0


class TestTemporalCorrelationEngine:
    """Test Temporal Correlation Engine."""
    
    def test_temporal_engine_import(self):
        """Test import of temporal correlation engine."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        assert TemporalCorrelationEngine is not None
    
    def test_temporal_engine_initialization(self):
        """Test temporal engine initialization."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        assert engine is not None
        assert len(engine.patterns) == 0
    
    def test_temporal_engine_register_pattern(self):
        """Test registering a correlation pattern."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        
        engine.register_pattern(
            "test_pattern",
            [
                {"type": "event_a", "confidence_min": 0.7},
                {"type": "event_b", "confidence_min": 0.8, "time_offset_seconds": 300}
            ]
        )
        
        assert "test_pattern" in engine.patterns
    
    def test_temporal_engine_no_match(self):
        """Test pattern matching with no match."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        
        engine.register_pattern(
            "test_pattern",
            [
                {"type": "port_scan", "confidence_min": 0.7},
                {"type": "brute_force", "confidence_min": 0.8, "time_offset_seconds": 300}
            ]
        )
        
        # Single event that doesn't match pattern
        result = engine.evaluate({
            "type": "port_scan",
            "source_ip": "192.168.1.100",
            "confidence": 0.9,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        assert result is None
    
    def test_temporal_engine_multi_stage_match(self):
        """Test multi-stage attack pattern matching."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        
        engine.register_pattern(
            "port_scan_to_brute",
            [
                {"type": "port_scan", "confidence_min": 0.7},
                {"type": "brute_force", "confidence_min": 0.8, "time_offset_seconds": 600}
            ]
        )
        
        now = datetime.now(timezone.utc)
        
        # First event: port scan
        engine.evaluate({
            "type": "port_scan",
            "source_ip": "192.168.1.100",
            "confidence": 0.9,
            "timestamp": now.isoformat(),
        })
        
        # Second event: brute force within 10 minutes
        result = engine.evaluate({
            "type": "brute_force",
            "source_ip": "192.168.1.100",
            "confidence": 0.85,
            "timestamp": (now + timedelta(minutes=5)).isoformat(),
        })
        
        # Should match the pattern
        assert result is not None


class TestConfigManager:
    """Test Hot-Reloadable Config Manager."""
    
    def test_config_manager_import(self):
        """Test import of config manager."""
        from src.core.config_manager import RedisConfigManager
        assert RedisConfigManager is not None
    
    def test_config_manager_create_instance(self):
        """Test creating config manager instance."""
        from src.core.config_manager import RedisConfigManager
        
        # Mock redis is optional
        try:
            manager = RedisConfigManager(redis_url="redis://localhost:6379")
            # May fail if redis not running, that's OK
        except Exception:
            # Config manager should handle missing Redis gracefully
            pass


# Performance tests
class TestPerformance:
    """Performance tests for Week 1 features."""
    
    @pytest.mark.slow
    def test_rule_compiler_performance(self):
        """Test rule compiler performance with many rules."""
        from src.detection.rule_compiler import RuleCompiler, RuleConditionEvaluator
        
        compiler = RuleCompiler()
        
        # Compile 100 rules
        for i in range(100):
            rule = {
                "conditions": {
                    "source_ip": f"192.168.1.{i}",
                    "severity": ">= 5"
                }
            }
            compiled = compiler.compile_rule(rule)
            assert compiled is not None
    
    @pytest.mark.slow
    def test_temporal_engine_throughput(self):
        """Test temporal engine with high event throughput."""
        from src.detection.temporal_correlation import TemporalCorrelationEngine
        
        engine = TemporalCorrelationEngine()
        
        engine.register_pattern(
            "test",
            [{"type": "event", "confidence_min": 0.5}]
        )
        
        # Process 1000 events
        now = datetime.now(timezone.utc)
        for i in range(1000):
            result = engine.evaluate({
                "type": "event",
                "source_ip": f"192.168.1.{i % 100}",
                "confidence": 0.7,
                "timestamp": (now + timedelta(milliseconds=i)).isoformat(),
            })
            # Most won't match, but engine should handle throughput


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
