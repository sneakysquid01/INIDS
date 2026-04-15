"""
Phase C: Multi-Threaded Packet Processing Pipeline
Integrates worker pool with Phase A infrastructure
Lock-free flow pinning (autofp mode)
"""

import logging
import time
from typing import List, Optional, Callable, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for packet processing pipeline"""
    packets_input: int = 0
    packets_decoded: int = 0
    packets_distributed: int = 0
    
    flows_created: int = 0
    flows_active: int = 0
    flows_closed: int = 0
    
    detections_made: int = 0
    alerts_raised: int = 0
    blocks_applied: int = 0
    
    bytes_processed: int = 0
    
    # Timing (seconds)
    start_time: Optional[float] = None
    decode_time_total: float = 0.0
    distribution_time_total: float = 0.0
    detection_time_total: float = 0.0
    
    # Performance
    latency_decode_ms: float = 0.0
    latency_distribution_ms: float = 0.0
    latency_detection_ms: float = 0.0
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_throughput_pps(self) -> float:
        """Get packets per second"""
        uptime = self.get_uptime()
        if uptime > 0:
            return self.packets_input / uptime
        return 0.0
    
    def get_throughput_mbps(self) -> float:
        """Get megabits per second"""
        uptime = self.get_uptime()
        if uptime > 0:
            bits = self.bytes_processed * 8
            megabits = bits / 1_000_000
            return megabits / uptime
        return 0.0
    
    def __repr__(self):
        return (f"PipelineStats(packets={self.packets_input}, "
                f"flows={self.flows_created}, detections={self.detections_made}, "
                f"throughput={self.get_throughput_pps():.0f} pps)")


class MultiThreadedPacketPipeline:
    """
    Multi-threaded variant of Phase A pipeline
    Uses worker pool for lock-free parallel detection
    """
    
    def __init__(self, packet_source, worker_pool, packet_decoder_class=None,
                 flow_table_class=None, detection_callbacks: List[Callable] = None):
        """
        Initialize multi-threaded pipeline
        
        Args:
            packet_source: Packet source (PCAP, live, etc.) from Phase A
            worker_pool: WorkerPool instance
            packet_decoder_class: PacketDecoder class
            flow_table_class: FlowTable class
            detection_callbacks: Optional detection callbacks for workers
        """
        self.packet_source = packet_source
        self.worker_pool = worker_pool
        self.detection_callbacks = detection_callbacks or []
        
        # Phase A components (main thread only)
        if packet_decoder_class is None:
            from src.decoding import PacketDecoder
            packet_decoder_class = PacketDecoder
        
        if flow_table_class is None:
            from src.flow_tracking import FlowTable
            flow_table_class = FlowTable
        
        self.packet_decoder = packet_decoder_class()
        self.flow_table = flow_table_class()  # Main thread flow tracking
        
        # Distributor
        from src.distributed_detection.packet_distributor import PacketDistributor
        self.distributor = PacketDistributor(worker_pool, batch_timeout_ms=50, batch_size=64)
        
        # Statistics
        self.stats = PipelineStats()
        self.stats.start_time = time.time()
        
        # Configuration
        self.cleanup_interval = 1000  # Cleanup every N packets
        self.max_flows = 100000  # Maximum concurrent flows
    
    def run(self, max_packets: int = 0, max_duration_seconds: int = 0) -> None:
        """
        Run packet processing pipeline
        
        Args:
            max_packets: Max packets to process (0 = unlimited)
            max_duration_seconds: Max duration (0 = unlimited)
        """
        logger.info(f"Pipeline starting with {self.worker_pool.num_workers} workers")
        self.worker_pool.start()
        
        packets_processed = 0
        start_time = time.time()
        
        try:
            for decoded_packet in self.packet_source.read_packets():
                if decoded_packet is None:
                    break
                
                self.stats.packets_input += 1
                
                # Update flow table and get context
                flow_context = self._process_packet_main_thread(decoded_packet)
                
                if flow_context:
                    # Queue for worker threads
                    self.distributor.queue_packet(
                        flow_id=decoded_packet.flow_id,
                        decoded_packet=decoded_packet,
                        flow_context=flow_context
                    )
                    self.stats.packets_distributed += 1
                
                packets_processed += 1
                
                # Periodic statistics and cleanup
                if packets_processed % self.cleanup_interval == 0:
                    self._cleanup_and_collect(packets_processed)
                
                # Check termination conditions
                if max_packets > 0 and packets_processed >= max_packets:
                    logger.info(f"Reached max packets limit: {max_packets}")
                    break
                
                if max_duration_seconds > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= max_duration_seconds:
                        logger.info(f"Reached max duration: {max_duration_seconds}s")
                        break
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Flush final batch
            self.distributor.flush_pending()
            
            # Wait for workers to finish
            logger.info("Waiting for workers to finish...")
            time.sleep(1)  # Give workers time to process
            
            # Collect final results
            self._cleanup_and_collect(packets_processed)
            
            # Stop workers
            self.worker_pool.stop()
            
            logger.info("Pipeline finished")
    
    def _process_packet_main_thread(self, decoded_packet) -> Optional['FlowContext']:
        """
        Process packet in main thread (Phase A operations)
        Returns flow context for distribution to workers
        """
        try:
            if not decoded_packet or not decoded_packet.l4_info:
                return None
            
            decode_start = time.time()
            
            # Get or create flow
            flow_context = self.flow_table.get_or_create_flow(
                src_ip=decoded_packet.l3_info.src_ip if decoded_packet.l3_info else "",
                dst_ip=decoded_packet.l3_info.dst_ip if decoded_packet.l3_info else "",
                src_port=decoded_packet.l4_info.src_port,
                dst_port=decoded_packet.l4_info.dst_port,
                protocol=decoded_packet.l4_info.protocol
            )
            
            # Update packet stats
            self.flow_table.update_packet_stats(
                flow_context,
                len(decoded_packet.payload_data) if decoded_packet.payload_data else 0
            )
            
            # Update TCP state
            if hasattr(decoded_packet.l4_info, 'tcp_flags'):
                self.flow_table.update_tcp_state(flow_context, decoded_packet.l4_info.tcp_flags)
            
            # Track statistics
            self.stats.packets_decoded += 1
            self.stats.flows_active = len(self.flow_table.flows)
            self.stats.bytes_processed += len(decoded_packet.payload_data) if decoded_packet.payload_data else 0
            
            decode_time = time.time() - decode_start
            self.stats.decode_time_total += decode_time
            self.stats.latency_decode_ms = (decode_time * 1000) if time.time() > 0 else 0
            
            return flow_context
        
        except Exception as e:
            logger.debug(f"Packet processing error: {e}")
            return None
    
    def _cleanup_and_collect(self, packets_processed: int) -> None:
        """Periodic cleanup and detection collection"""
        # Flush pending packets
        self.distributor.flush_pending()
        
        # Collect detections from workers
        detections = self.worker_pool.collect_detections()
        self.stats.detections_made += len(detections)
        
        # Update flow states based on detections
        for detection in detections:
            flow_id = detection['flow_id']
            score = detection['detection_score']
            reason = detection['alert_reason']
            
            if flow_id in self.flow_table.flows:
                flow = self.flow_table.flows[flow_id]
                
                # Update flow action based on detection
                if score > 0.8:
                    self.flow_table.set_flow_action(flow, self.flow_table.FlowAction.BLOCK)
                    self.stats.blocks_applied += 1
                elif score > 0.5:
                    self.flow_table.set_flow_action(flow, self.flow_table.FlowAction.ALERT)
                    self.stats.alerts_raised += 1
        
        # Cleanup expired flows
        expired_count = self.flow_table.cleanup_expired_flows()
        self.stats.flows_closed += expired_count
        
        # Log progress
        throughput = self.stats.get_throughput_pps()
        mbps = self.stats.get_throughput_mbps()
        logger.info(f"Progress: {packets_processed} packets, {len(self.flow_table.flows)} active flows, "
                   f"{throughput:.0f} pps, {mbps:.2f} Mbps, {self.stats.detections_made} detections")
    
    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics"""
        return self.stats
    
    def print_stats(self) -> None:
        """Pretty-print statistics"""
        stats = self.get_stats()
        worker_stats = self.worker_pool.get_stats()
        
        print("\n" + "="*80)
        print("MULTI-THREADED PIPELINE STATISTICS")
        print("="*80)
        print(f"\n[PIPELINE OVERVIEW]")
        print(f"  Uptime: {stats.get_uptime():.1f}s")
        print(f"  Packets input: {stats.packets_input}")
        print(f"  Packets decoded (main thread): {stats.packets_decoded}")
        print(f"  Packets distributed to workers: {stats.packets_distributed}")
        print(f"  Throughput: {stats.get_throughput_pps():.0f} pps ({stats.get_throughput_mbps():.2f} Mbps)")
        
        print(f"\n[FLOWS]")
        print(f"  Flows created: {stats.flows_created}")
        print(f"  Flows active: {stats.flows_active}")
        print(f"  Flows closed: {stats.flows_closed}")
        print(f"  Total bytes: {stats.bytes_processed / 1_000_000:.2f} MB")
        
        print(f"\n[DETECTIONS]")
        print(f"  Total detections: {stats.detections_made}")
        print(f"  Alerts raised: {stats.alerts_raised}")
        print(f"  Blocks applied: {stats.blocks_applied}")
        
        print(f"\n[LATENCY]")
        print(f"  Avg decode: {stats.latency_decode_ms:.2f}ms")
        print(f"  Avg distribution: {stats.latency_distribution_ms:.2f}ms")
        print(f"  Avg detection (workers): {stats.latency_detection_ms:.2f}ms")
        
        print(f"\n[WORKER POOL ({worker_stats['num_workers']} workers)]")
        print(f"  Total worker packets: {worker_stats['total_worker_packets']}")
        print(f"  Total worker detections: {worker_stats['total_worker_detections']}")
        print(f"  Avg worker latency: {worker_stats['avg_worker_latency_ms']:.2f}ms")
        
        for ws in worker_stats['worker_stats']:
            print(f"    Worker {ws.worker_id}: {ws.packets_processed} packets, "
                  f"{ws.detections_made} detections, {ws.get_throughput():.0f} pps")
        
        dist_stats = self.distributor.get_stats()
        print(f"\n[DISTRIBUTION]")
        print(f"  Batches created: {dist_stats.total_batches_created}")
        print(f"  Avg batch size: {dist_stats.average_batch_size:.1f}")
        print(f"  Max batch size: {dist_stats.max_batch_size}")
        print(f"  Unique flows: {dist_stats.flow_count}")
        
        print("="*80 + "\n")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.packet_source:
            self.packet_source.close()
        self.worker_pool.stop()


# Convenience function for easy usage
def create_multithreaded_pipeline(packet_source, num_workers: int = 4, 
                                  ml_models=None) -> MultiThreadedPacketPipeline:
    """
    Factory function to create multi-threaded pipeline with defaults
    
    Args:
        packet_source: PacketSource instance
        num_workers: Number of worker threads
        ml_models: Tuple of (ensemble_model, risk_scorer) or None
    
    Returns:
        MultiThreadedPacketPipeline ready to run
    """
    from src.distributed_detection.worker_pool import WorkerPool
    from src.distributed_detection.packet_distributor import WorkerDetectionCallback
    from src.protocol_parsers.phase_b_integration import ProtocolAnalyzer
    
    # Create worker pool
    worker_pool = WorkerPool(num_workers=num_workers)
    
    # Create detection callback
    detection_callback = WorkerDetectionCallback.create_detection_callback(
        ml_models=ml_models,
        protocol_analyzer=ProtocolAnalyzer
    )
    
    # Register callback with workers
    worker_pool.detection_callbacks.append(detection_callback)
    
    # Create pipeline
    pipeline = MultiThreadedPacketPipeline(
        packet_source=packet_source,
        worker_pool=worker_pool,
        detection_callbacks=[detection_callback]
    )
    
    return pipeline
