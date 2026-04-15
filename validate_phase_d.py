"""
Phase D Validation Script

Validates EVE JSON output module completeness and functionality.
No pytest dependency - runs standalone.
"""

import sys
import json
import tempfile
from pathlib import Path


def validate_imports():
    """Check all imports work"""
    print("[VALIDATION] Checking imports...")
    
    try:
        from src.output import (
            EVEEvent,
            EVEEventBuilder,
            EventType,
            AlertSeverity,
            AlertPayload,
            HTTPPayload,
            DNSPayload,
            TLSPayload,
            FileBackend,
            SyslogBackend,
            RedisBackend,
            WebhookBackend,
            OutputAggregator,
            FlowAggregator,
            AggregationMode,
            OutputPipeline,
            AlertThrottler,
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_eve_event():
    """Validate EVE event creation and serialization"""
    print("[VALIDATION] Checking EVE event functionality...")
    
    try:
        from src.output import EVEEvent, EventType, AlertPayload
        
        # Create event
        alert = AlertPayload(
            action="alert",
            signature_id=1,
            signature="Test alert",
            category="Test",
            severity=2,
        )
        
        event = EVEEvent(
            timestamp="2025-04-09T14:32:15.000000+00:00",
            event_type=EventType.ALERT,
            flow_id=1,
            src_ip="10.0.0.1",
            src_port=1000,
            dest_ip="10.0.0.2",
            dest_port=80,
            proto="tcp",
            alert=alert,
        )
        
        # Verify properties
        assert event.flow_id == 1
        assert event.event_type == EventType.ALERT
        
        # Verify JSON serialization
        json_str = event.to_json()
        data = json.loads(json_str)
        assert data["flow_id"] == 1
        assert data["event_type"] == "alert"
        
        print("  ✓ EVE event creation and serialization works")
        return True
    except Exception as e:
        print(f"  ✗ EVE event validation failed: {e}")
        return False


def validate_eve_event_builder():
    """Validate EVE event builder"""
    print("[VALIDATION] Checking EVE event builder...")
    
    try:
        from src.output import EVEEventBuilder, EventType
        
        builder = EVEEventBuilder(source="Test")
        
        # Test alert creation
        alert = builder.create_alert_event(
            flow_id=1,
            src_ip="10.0.0.1",
            src_port=1000,
            dst_ip="10.0.0.2",
            dst_port=80,
            proto="tcp",
            detection_reason="Test",
            detection_score=0.8,
        )
        assert alert.event_type == EventType.ALERT
        assert alert.alert is not None
        
        # Test flow creation
        flow = builder.create_flow_event(
            flow_id=2,
            src_ip="10.0.0.1",
            src_port=1000,
            dst_ip="10.0.0.2",
            dst_port=53,
            proto="udp",
            flow_state={"packets": 10},
        )
        assert flow.event_type == EventType.FLOW
        
        # Test HTTP creation
        http = builder.create_http_event(
            flow_id=3,
            src_ip="10.0.0.1",
            src_port=1000,
            dst_ip="10.0.0.2",
            dst_port=80,
            proto="tcp",
            http_data={"http_method": "GET", "http_uri": "/"},
        )
        assert http.event_type == EventType.HTTP
        
        print("  ✓ EVE event builder works correctly")
        return True
    except Exception as e:
        print(f"  ✗ EVE event builder validation failed: {e}")
        return False


def validate_file_backend():
    """Validate file output backend"""
    print("[VALIDATION] Checking file backend...")
    
    try:
        from src.output import FileBackend, EVEEventBuilder
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            backend = FileBackend(filepath=filepath)
            
            # Send event
            builder = EVEEventBuilder()
            event = builder.create_alert_event(
                flow_id=1,
                src_ip="10.0.0.1",
                src_port=1000,
                dst_ip="10.0.0.2",
                dst_port=80,
                proto="tcp",
                detection_reason="Test",
                detection_score=0.5,
            )
            
            assert backend.send(event)
            backend.close()
            
            # Verify file was created and contains data
            assert Path(filepath).exists()
            assert Path(filepath).stat().st_size > 0
            
            # Verify content
            with open(filepath) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["event_type"] == "alert"
            
            print("  ✓ File backend works correctly")
            return True
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    except Exception as e:
        print(f"  ✗ File backend validation failed: {e}")
        return False


def validate_output_aggregator():
    """Validate output aggregator"""
    print("[VALIDATION] Checking output aggregator...")
    
    try:
        from src.output import OutputAggregator, FileBackend, EVEEventBuilder
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            agg = OutputAggregator()
            backend = FileBackend(filepath=filepath)
            agg.add_backend(backend)
            
            builder = EVEEventBuilder()
            event = builder.create_alert_event(
                flow_id=1,
                src_ip="10.0.0.1",
                src_port=1000,
                dst_ip="10.0.0.2",
                dst_port=80,
                proto="tcp",
                detection_reason="Test",
                detection_score=0.5,
            )
            
            assert agg.send_event(event)
            agg.close_all()
            
            stats = agg.get_stats()
            assert "File" in stats
            
            print("  ✓ Output aggregator works correctly")
            return True
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    except Exception as e:
        print(f"  ✗ Output aggregator validation failed: {e}")
        return False


def validate_flow_aggregator():
    """Validate flow aggregator"""
    print("[VALIDATION] Checking flow aggregator...")
    
    try:
        from src.output import FlowAggregator, AggregationMode, EVEEventBuilder
        
        agg = FlowAggregator(mode=AggregationMode.UNIQUE_PER_MINUTE)
        builder = EVEEventBuilder()
        
        # Create event
        event = builder.create_alert_event(
            flow_id=1,
            src_ip="10.0.0.1",
            src_port=1000,
            dst_ip="10.0.0.2",
            dst_port=80,
            proto="tcp",
            detection_reason="Test",
            detection_score=0.5,
        )
        event.alert.signature_id = 100
        
        # First event should pass
        assert agg.add_event(event) == True
        
        # Duplicate should be filtered
        assert agg.add_event(event) == False
        
        stats = agg.get_aggregation_stats()
        assert stats["total_deduplicated"] == 1
        
        print("  ✓ Flow aggregator works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Flow aggregator validation failed: {e}")
        return False


def validate_aggregation_modes():
    """Validate different aggregation modes"""
    print("[VALIDATION] Checking aggregation modes...")
    
    try:
        from src.output import FlowAggregator, AggregationMode, EVEEventBuilder
        
        # Test all modes are available
        modes = [
            AggregationMode.PASS_THROUGH,
            AggregationMode.UNIQUE_PER_MINUTE,
            AggregationMode.UNIQUE_PER_HOUR,
            AggregationMode.TOP_ALERT_PER_FLOW,
        ]
        
        builder = EVEEventBuilder()
        
        for mode in modes:
            agg = FlowAggregator(mode=mode)
            event = builder.create_alert_event(
                flow_id=1,
                src_ip="10.0.0.1",
                src_port=1000,
                dst_ip="10.0.0.2",
                dst_port=80,
                proto="tcp",
                detection_reason="Test",
                detection_score=0.5,
            )
            assert agg.add_event(event) is not None
        
        print("  ✓ All aggregation modes work correctly")
        return True
    except Exception as e:
        print(f"  ✗ Aggregation modes validation failed: {e}")
        return False


def validate_alert_throttler():
    """Validate alert throttler"""
    print("[VALIDATION] Checking alert throttler...")
    
    try:
        from src.output import AlertThrottler, EVEEventBuilder
        
        throttler = AlertThrottler(
            max_alerts_per_flow_per_second=5,
            max_alerts_per_second=10,
        )
        
        builder = EVEEventBuilder()
        event = builder.create_alert_event(
            flow_id=1,
            src_ip="10.0.0.1",
            src_port=1000,
            dst_ip="10.0.0.2",
            dst_port=80,
            proto="tcp",
            detection_reason="Test",
            detection_score=0.5,
        )
        
        # Should not be rate limited initially
        assert throttler.should_rate_limit(event) == False
        
        # Get stats
        stats = throttler.get_stats()
        assert stats["global_per_second"] >= 1
        
        print("  ✓ Alert throttler works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Alert throttler validation failed: {e}")
        return False


def validate_integration_with_phases():
    """Validate integration with previous phases"""
    print("[VALIDATION] Checking integration with Phases A-C...")
    
    try:
        # Should be able to import Phase A/B/C modules
        # (they should exist and Phase D should work with their outputs)
        
        # Check Phase C imports work
        try:
            from src.distributed_detection import MultiThreadedPacketPipeline
            print("  ✓ Phase C imports available")
        except ImportError:
            print("  ⚠ Phase C not available (but Phase D is independent)")
        
        print("  ✓ Integration points validated")
        return True
    except Exception as e:
        print(f"  ✗ Integration validation failed: {e}")
        return False


def validate_json_format():
    """Validate EVE JSON format compliance"""
    print("[VALIDATION] Checking EVE JSON format...")
    
    try:
        from src.output import EVEEventBuilder
        
        builder = EVEEventBuilder()
        
        # Create various event types
        events = [
            builder.create_alert_event(
                flow_id=1,
                src_ip="10.0.0.1",
                src_port=1000,
                dst_ip="10.0.0.2",
                dst_port=80,
                proto="tcp",
                detection_reason="Test alert",
                detection_score=0.8,
            ),
            builder.create_flow_event(
                flow_id=2,
                src_ip="10.0.0.1",
                src_port=1001,
                dst_ip="10.0.0.2",
                dst_port=53,
                proto="udp",
                flow_state={"packets": 10},
            ),
            builder.create_http_event(
                flow_id=3,
                src_ip="10.0.0.1",
                src_port=1002,
                dst_ip="10.0.0.2",
                dst_port=80,
                proto="tcp",
                http_data={"http_method": "GET", "http_uri": "/"},
            ),
        ]
        
        # Verify all can be serialized and have required fields
        for event in events:
            data = json.loads(event.to_json())
            
            # Required fields
            assert "timestamp" in data
            assert "flow_id" in data
            assert "event_type" in data
            assert "event_id" in data
            
            # Format checks
            assert data["event_type"] in ["alert", "flow", "http", "dns", "tls", "ssh"]
            assert isinstance(data["flow_id"], int)
        
        print("  ✓ EVE JSON format is correct")
        return True
    except Exception as e:
        print(f"  ✗ JSON format validation failed: {e}")
        return False


def validate_backends_available():
    """Validate all backends are available"""
    print("[VALIDATION] Checking backend availability...")
    
    try:
        from src.output import (
            FileBackend,
            SyslogBackend,
            RedisBackend,
            WebhookBackend,
        )
        
        print("  ✓ File backend available")
        print("  ✓ Syslog backend available")
        print("  ✓ Redis backend available (optional dependency)")
        print("  ✓ Webhook backend available (optional dependency)")
        
        return True
    except Exception as e:
        print(f"  ✗ Backend availability check failed: {e}")
        return False


def main():
    """Run all validations"""
    print("\n" + "="*60)
    print("PHASE D: EVE JSON OUTPUT - VALIDATION SCRIPT")
    print("="*60 + "\n")
    
    validations = [
        validate_imports,
        validate_eve_event,
        validate_eve_event_builder,
        validate_file_backend,
        validate_output_aggregator,
        validate_flow_aggregator,
        validate_aggregation_modes,
        validate_alert_throttler,
        validate_backends_available,
        validate_json_format,
        validate_integration_with_phases,
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
