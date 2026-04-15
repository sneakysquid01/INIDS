"""
Phase C: Multi-Threaded Detection Test Suite
Tests for worker pool, distribution, and multi-threaded pipeline
"""

import sys
import time
import threading
from typing import List

# Mock classes for testing
class MockDecodedPacket:
    def __init__(self, flow_id="test_flow", payload_size=100):
        self.flow_id = flow_id
        self.payload_data = b"x" * payload_size
        
        # Mock L2
        self.l2_info = type('obj', (object,), {
            'ethertype': 0x0800,
            'vlan_tag': None
        })()
        
        # Mock L3
        self.l3_info = type('obj', (object,), {
            'src_ip': '192.168.1.100',
            'dst_ip': '10.0.0.1',
            'ttl': 64,
            'flags': 0
        })()
        
        # Mock L4
        self.l4_info = type('obj', (object,), {
            'src_port': 54321,
            'dst_port': 80,
            'protocol': 'TCP',
            'tcp_flags': 0x02  # SYN
        })()


class MockFlowContext:
    def __init__(self, flow_id="test_flow"):
        self.flow_id = flow_id
        self.src_ip = '192.168.1.100'
        self.dst_ip = '10.0.0.1'
        self.src_port = 54321
        self.dst_port = 80
        self.protocol = 'TCP'
        self.packet_count = 1
        self.bytes_toserver = 100
        self.bytes_toclient = 0
        self.action = "ALLOW"
        self.escalation_level = 0
        self.features_cache = {}


def test_flow_hasher():
    """Test flow hashing and worker assignment"""
    print("\n=== Testing Flow Hasher ===")
    
    from src.distributed_detection import FlowHasher
    
    # Test 1: Deterministic hashing
    print("✓ Test 1: Deterministic flow hashing")
    flow_id = "abcd1234567890ef"
    worker1 = FlowHasher.compute_flow_partition(flow_id, 4)
    worker2 = FlowHasher.compute_flow_partition(flow_id, 4)
    assert worker1 == worker2, "Hashing should be deterministic"
    assert 0 <= worker1 < 4, "Worker ID should be in range"
    print(f"  Flow {flow_id} → Worker {worker1}")
    
    # Test 2: Distribution across workers
    print("✓ Test 2: Distribution across workers")
    flow_ids = [f"flow_{i:04d}" for i in range(100)]
    distribution = {}
    for fid in flow_ids:
        worker = FlowHasher.compute_flow_partition(fid, 4)
        distribution[worker] = distribution.get(worker, 0) + 1
    
    assert len(distribution) > 1, "Should use multiple workers"
    print(f"  Distribution: {distribution}")
    
    # Test 3: Compute from tuple
    print("✓ Test 3: Compute worker from 5-tuple")
    worker = FlowHasher.compute_flow_partition_from_tuple(
        "192.168.1.100", "10.0.0.1", 54321, 80, "TCP", 4
    )
    assert 0 <= worker < 4, "Worker should be valid"
    print(f"  Tuple → Worker {worker}")
    
    print("✓ Flow Hasher: PASSED\n")


def test_detection_worker():
    """Test detection worker thread"""
    print("\n=== Testing Detection Worker ===")
    
    from src.distributed_detection.worker_pool import DetectionWorker, WorkerPacketBatch
    from src.decoding import PacketDecoder
    from src.flow_tracking import FlowTable
    
    # Test 1: Create worker
    print("✓ Test 1: Create detection worker")
    import queue
    q = queue.Queue()
    worker = DetectionWorker(
        worker_id=0,
        input_queue=q,
        packet_decoder_class=PacketDecoder,
        flow_table_class=FlowTable
    )
    assert worker.worker_id == 0, "Worker ID should match"
    assert worker.state.value == "initializing", "Initial state should be initializing"
    print(f"  Worker created: {worker}")
    
    # Test 2: Start/stop worker
    print("✓ Test 2: Start and stop worker")
    worker.start()
    assert worker.thread is not None, "Thread should be created"
    assert worker.running, "Worker should be running"
    time.sleep(0.1)  # Give thread time to start
    worker.stop()
    time.sleep(0.1)  # Give thread time to stop
    assert not worker.running, "Worker should stop"
    print(f"  Worker stopped cleanly")
    
    # Test 3: Process batch
    print("✓ Test 3: Process packet batch")
    q2 = queue.Queue()
    worker2 = DetectionWorker(
        worker_id=1,
        input_queue=q2,
        packet_decoder_class=PacketDecoder,
        flow_table_class=FlowTable
    )
    worker2.start()
    
    # Create batch
    batch = WorkerPacketBatch(batch_id=1)
    packet = MockDecodedPacket()
    flow_ctx = MockFlowContext()
    batch.packets.append(("flow_1", packet))
    batch.flow_contexts["flow_1"] = flow_ctx
    
    # Send batch
    q2.put(batch)
    time.sleep(0.2)
    
    # Check stats
    stats = worker2.get_stats()
    assert stats.packets_processed > 0, "Should process packets"
    print(f"  Processed: {stats.packets_processed} packets")
    
    worker2.stop()
    print("✓ Detection Worker: PASSED\n")


def test_worker_pool():
    """Test worker pool"""
    print("\n=== Testing Worker Pool ===")
    
    from src.distributed_detection import WorkerPool
    from src.decoding import PacketDecoder
    from src.flow_tracking import FlowTable
    
    # Test 1: Create pool
    print("✓ Test 1: Create worker pool")
    pool = WorkerPool(
        num_workers=4,
        packet_decoder_class=PacketDecoder,
        flow_table_class=FlowTable
    )
    assert len(pool.workers) == 4, "Should have 4 workers"
    print(f"  Pool created with {pool.num_workers} workers")
    
    # Test 2: Start/stop pool
    print("✓ Test 2: Start and stop pool")
    pool.start()
    assert pool.workers[0].running, "Workers should be running"
    time.sleep(0.1)
    pool.stop()
    time.sleep(0.1)
    assert not pool.workers[0].running, "Workers should stop"
    print(f"  Pool stopped cleanly")
    
    # Test 3: Distribute packets
    print("✓ Test 3: Distribute packets to workers")
    pool2 = WorkerPool(num_workers=2, packet_decoder_class=PacketDecoder, flow_table_class=FlowTable)
    pool2.start()
    
    packets = []
    flow_ctxs = {}
    for i in range(10):
        flow_id = f"flow_{i}"
        packet = MockDecodedPacket(flow_id)
        packets.append((flow_id, packet))
        flow_ctxs[flow_id] = MockFlowContext(flow_id)
    
    pool2.distribute_batch(packets, flow_ctxs)
    time.sleep(0.2)
    
    stats = pool2.get_stats()
    assert stats['pool_packets'] > 0, "Should process packets"
    print(f"  Distributed {len(packets)} packets")
    print(f"  Total packets processed: {stats['pool_packets']}")
    
    pool2.stop()
    print("✓ Worker Pool: PASSED\n")


def test_packet_distributor():
    """Test packet distributor"""
    print("\n=== Testing Packet Distributor ===")
    
    from src.distributed_detection import WorkerPool, PacketDistributor
    from src.decoding import PacketDecoder
    from src.flow_tracking import FlowTable
    
    # Test 1: Create distributor
    print("✓ Test 1: Create packet distributor")
    pool = WorkerPool(num_workers=2, packet_decoder_class=PacketDecoder, flow_table_class=FlowTable)
    dist = PacketDistributor(pool, batch_timeout_ms=100, batch_size=4)
    assert dist.batch_size == 4, "Batch size should match"
    print(f"  Distributor created: batch_size={dist.batch_size}")
    
    # Test 2: Queue packets
    print("✓ Test 2: Queue packets for distribution")
    pool.start()
    
    for i in range(10):
        flow_id = f"flow_{i}"
        packet = MockDecodedPacket(flow_id)
        flow_ctx = MockFlowContext(flow_id)
        dist.queue_packet(flow_id, packet, flow_ctx)
    
    # Flush pending
    dist.flush_pending()
    time.sleep(0.2)
    
    stats = dist.get_stats()
    assert stats.total_packets_queued == 10, "Should queue all packets"
    print(f"  Queued: {stats.total_packets_queued} packets")
    print(f"  Batches: {stats.total_batches_created}")
    
    pool.stop()
    print("✓ Packet Distributor: PASSED\n")


def test_feature_aggregation():
    """Test multi-layer feature aggregation"""
    print("\n=== Testing Feature Aggregation ===")
    
    from src.distributed_detection import MultiLayerFeatureAggregator
    
    # Test 1: Aggregate packet features
    print("✓ Test 1: Aggregate packet features")
    packet = MockDecodedPacket()
    features = MultiLayerFeatureAggregator.aggregate_packet_features(packet)
    assert 'pkt_src_ip' in features, "Should extract IP"
    assert 'pkt_src_port' in features, "Should extract port"
    assert 'pkt_payload_size' in features, "Should extract payload size"
    print(f"  Features extracted: {len(features)}")
    
    # Test 2: Aggregate flow features
    print("✓ Test 2: Aggregate flow features")
    flow_ctx = MockFlowContext()
    features = MultiLayerFeatureAggregator.aggregate_flow_features(flow_ctx)
    assert 'flow_src_ip' in features, "Should extract src IP"
    assert 'flow_packets_count' in features, "Should extract packet count"
    assert 'flow_bytes_toserver' in features, "Should extract bytes"
    print(f"  Flow features extracted: {len(features)}")
    
    # Test 3: Aggregate all features
    print("✓ Test 3: Aggregate combined features")
    all_features = MultiLayerFeatureAggregator.aggregate_all_features(packet, flow_ctx)
    assert 'pkt_src_ip' in all_features, "Should have packet features"
    assert 'flow_src_ip' in all_features, "Should have flow features"
    print(f"  Combined features: {len(all_features)}")
    
    print("✓ Feature Aggregation: PASSED\n")


def test_detection_callback():
    """Test worker detection callback"""
    print("\n=== Testing Detection Callback ===")
    
    from src.distributed_detection import WorkerDetectionCallback
    
    # Test 1: Create callback
    print("✓ Test 1: Create detection callback")
    callback = WorkerDetectionCallback.create_detection_callback(ml_models=None)
    assert callable(callback), "Should return callable"
    print(f"  Callback created")
    
    # Test 2: Execute callback
    print("✓ Test 2: Execute detection callback")
    packet = MockDecodedPacket()
    flow_ctx = MockFlowContext()
    
    score, reason = callback(flow_ctx, packet)
    assert isinstance(score, (int, float)), "Score should be numeric"
    print(f"  Callback result: score={score}, reason={reason}")
    
    print("✓ Detection Callback: PASSED\n")


def test_multithreaded_pipeline_basic():
    """Test multi-threaded pipeline basic functionality"""
    print("\n=== Testing Multi-Threaded Pipeline ===")
    
    from src.distributed_detection import MultiThreadedPacketPipeline, create_multithreaded_pipeline
    from src.packet_capture import InMemorySource
    
    # Test 1: Create pipeline
    print("✓ Test 1: Create multi-threaded pipeline")
    
    # Create test packet source
    packets_data = []
    for i in range(10):
        packet = MockDecodedPacket(flow_id=f"flow_{i % 2}")  # 2 flows
        packets_data.append(packet)
    
    source = InMemorySource(packets_data)
    
    pipeline = create_multithreaded_pipeline(source, num_workers=2)
    assert pipeline.worker_pool.num_workers == 2, "Should have 2 workers"
    print(f"  Pipeline created: {pipeline.worker_pool.num_workers} workers")
    
    # Test 2: Run pipeline
    print("✓ Test 2: Run pipeline with small dataset")
    pipeline.run(max_packets=10)
    
    stats = pipeline.get_stats()
    assert stats.packets_input > 0, "Should process packets"
    assert stats.packets_distributed > 0, "Should distribute packets"
    print(f"  Processed: {stats.packets_input} packets")
    print(f"  Throughput: {stats.get_throughput_pps():.0f} pps")
    
    print("✓ Multi-Threaded Pipeline: PASSED\n")


def run_all_tests():
    """Run all Phase C tests"""
    print("\n" + "="*70)
    print("PHASE C: MULTI-THREADED DETECTION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Flow Hasher", test_flow_hasher),
        ("Detection Worker", test_detection_worker),
        ("Worker Pool", test_worker_pool),
        ("Packet Distributor", test_packet_distributor),
        ("Feature Aggregation", test_feature_aggregation),
        ("Detection Callback", test_detection_callback),
        ("Multi-threaded Pipeline", test_multithreaded_pipeline_basic),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
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
            errors.append(test_name)
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
