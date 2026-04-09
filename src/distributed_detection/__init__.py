"""
Distributed Detection Module
Multi-threaded packet processing with lock-free flow pinning
"""

from .worker_pool import WorkerPool, DetectionWorker, WorkerStats, FlowHasher
from .packet_distributor import PacketDistributor, DistributionStats, WorkerDetectionCallback, MultiLayerFeatureAggregator
from .multi_threaded_pipeline import MultiThreadedPacketPipeline, PipelineStats, create_multithreaded_pipeline

__all__ = [
    # Worker Pool
    'WorkerPool',
    'DetectionWorker',
    'WorkerStats',
    'FlowHasher',
    
    # Distribution
    'PacketDistributor',
    'DistributionStats',
    'WorkerDetectionCallback',
    'MultiLayerFeatureAggregator',
    
    # Pipeline
    'MultiThreadedPacketPipeline',
    'PipelineStats',
    'create_multithreaded_pipeline',
]
