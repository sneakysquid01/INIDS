"""
Phase C Validation Script
Standalone validation of multi-threaded detection components
"""

import sys
import os


class ValidationError(Exception):
    """Validation error"""
    pass


def validate_imports():
    """Validate all distributed detection modules import correctly"""
    print("\n>>> Validating Imports...")
    
    try:
        from src.distributed_detection import (
            WorkerPool, DetectionWorker, WorkerStats, FlowHasher,
            PacketDistributor, DistributionStats, WorkerDetectionCallback,
            MultiLayerFeatureAggregator,
            MultiThreadedPacketPipeline, PipelineStats, create_multithreaded_pipeline
        )
        print("✓ All distributed detection imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_worker_pool():
    """Validate worker pool functionality"""
    print("\n>>> Validating Worker Pool...")
    
    from src.distributed_detection import WorkerPool, FlowHasher
    from src.decoding import PacketDecoder
    from src.flow_tracking import FlowTable
    
    # Test 1: Pool creation
    pool = WorkerPool(num_workers=2, packet_decoder_class=PacketDecoder, flow_table_class=FlowTable)
    if len(pool.workers) != 2:
        raise ValidationError("Pool should have 2 workers")
    print("✓ Pool creation works")
    
    # Test 2: Worker initialization
    for worker in pool.workers:
        if worker.worker_id < 0 or worker.worker_id >= 2:
            raise ValidationError("Worker ID should be valid")
    print("✓ Worker initialization works")
    
    # Test 3: Flow hashing
    worker_id = FlowHasher.compute_flow_partition("test_flow_123", 2)
    if not (0 <= worker_id < 2):
        raise ValidationError("Worker ID should be in range")
    print("✓ Flow hashing works")
    
    return True


def validate_packet_distributor():
    """Validate packet distributor"""
    print("\n>>> Validating Packet Distributor...")
    
    from src.distributed_detection import WorkerPool, PacketDistributor
    from src.decoding import PacketDecoder
    from src.flow_tracking import FlowTable
    
    pool = WorkerPool(num_workers=2, packet_decoder_class=PacketDecoder, flow_table_class=FlowTable)
    distributor = PacketDistributor(pool, batch_timeout_ms=100, batch_size=4)
    
    if distributor.batch_size != 4:
        raise ValidationError("Batch size should match")
    
    stats = distributor.get_stats()
    if stats.total_packets_queued != 0:
        raise ValidationError("Initial stats should be zero")
    
    print("✓ Packet distributor works")
    return True


def validate_feature_aggregation():
    """Validate feature aggregation"""
    print("\n>>> Validating Feature Aggregation...")
    
    from src.distributed_detection import MultiLayerFeatureAggregator
    
    # Create mock packet
    class MockPacket:
        def __init__(self):
            self.l2_info = None
            self.l3_info = type('obj', (object,), {'src_ip': '192.168.1.1', 'dst_ip': '10.0.0.1'})()
            self.l4_info = type('obj', (object,), {'src_port': 80})()
            self.payload_data = b"test"
    
    packet = MockPacket()
    features = MultiLayerFeatureAggregator.aggregate_packet_features(packet)
    
    if 'pkt_payload_size' not in features:
        raise ValidationError("Missing payload size feature")
    
    print("✓ Feature aggregation works")
    return True


def validate_detection_callback():
    """Validate detection callback creation"""
    print("\n>>> Validating Detection Callback...")
    
    from src.distributed_detection import WorkerDetectionCallback
    
    callback = WorkerDetectionCallback.create_detection_callback()
    if not callable(callback):
        raise ValidationError("Callback should be callable")
    
    print("✓ Detection callback works")
    return True


def validate_multithreaded_pipeline():
    """Validate multi-threaded pipeline"""
    print("\n>>> Validating Multi-Threaded Pipeline...")
    
    from src.distributed_detection import MultiThreadedPacketPipeline, create_multithreaded_pipeline
    from src.packet_capture import InMemorySource
    
    # Test 1: Pipeline creation
    source = InMemorySource([])
    pipeline = create_multithreaded_pipeline(source, num_workers=2)
    
    if pipeline.worker_pool.num_workers != 2:
        raise ValidationError("Pipeline should have correct worker count")
    print("✓ Pipeline creation works")
    
    # Test 2: Pipeline stats
    stats = pipeline.get_stats()
    if stats.packets_input != 0:
        raise ValidationError("Initial stats should be zero")
    print("✓ Pipeline statistics work")
    
    return True


def validate_backward_compatibility():
    """Validate Phase C doesn't break Phase A/B"""
    print("\n>>> Validating Backward Compatibility...")
    
    try:
        # Verify Phase A still works
        from src.packet_capture import PacketSource
        from src.decoding import PacketDecoder
        from src.flow_tracking import FlowTable
        from src.integration import PacketProcessingPipeline
        print("✓ Phase A modules still import correctly")
        
        # Verify Phase B still works
        from src.protocol_parsers import HTTPParser, DNSParser, TLSParser
        from src.protocol_parsers.phase_b_integration import ProtocolAnalyzer
        print("✓ Phase B modules still import correctly")
        
        return True
    except Exception as e:
        raise ValidationError(f"Backend compatibility broken: {e}")


def validate_file_structure():
    """Validate all Phase C files exist"""
    print("\n>>> Validating File Structure...")
    
    files = [
        "src/distributed_detection/__init__.py",
        "src/distributed_detection/worker_pool.py",
        "src/distributed_detection/packet_distributor.py",
        "src/distributed_detection/multi_threaded_pipeline.py",
        "tests/test_phase_c_multithreading.py",
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
    print("PHASE C: MULTI-THREADED DETECTION VALIDATION")
    print("="*70)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports),
        ("Worker Pool", validate_worker_pool),
        ("Packet Distributor", validate_packet_distributor),
        ("Feature Aggregation", validate_feature_aggregation),
        ("Detection Callback", validate_detection_callback),
        ("Multi-threaded Pipeline", validate_multithreaded_pipeline),
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
