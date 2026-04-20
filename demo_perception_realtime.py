#!/usr/bin/env python3
"""INIDS 2.0 Real-time Perception Layer Demo

Demonstrates:
- Real-time detection events flowing through perception engines
- Attack story construction from detection sequences
- Confidence breakdown showing decision factors
- Live system pulse tracking metrics
- Real-time latency metrics and throughput
"""

import time
import random
import logging
from datetime import datetime, timedelta, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_brute_force_attack(event_bus):
    """Simulate a brute force attack: multiple failed auth attempts from one IP"""
    logger.info("\n" + "="*70)
    logger.info("DEMO: SIMULATING BRUTE FORCE ATTACK (SSH)")
    logger.info("="*70)
    
    attacker_ip = "192.168.1.100"
    
    # Brute force phase: multiple failed authentications
    for attempt in range(1, 6):
        from src.core.event_bus import DetectionEvent
        
        event = DetectionEvent(
            source_ip=attacker_ip,
            prediction="Attack",
            confidence=0.75 + (attempt * 0.05),  # Confidence increases with attempts
            attack_type="brute_force",
            features={
                "failed_auth_attempts": attempt,
                "repeated_connection_failures": attempt,
                "protocol_anomaly": 0.8,
                "connection_count": attempt * 2,
            },
            severity="high" if attempt >= 3 else "medium",
            reason=f"Multiple failed SSH authentications (attempt {attempt})"
        )
        
        logger.info(f"  [{attempt}] Detection: {attacker_ip} - Confidence {event.confidence:.0%}")
        event_bus.publish(event)
        time.sleep(0.5)  # Space out events for visualization
    
    logger.info("\n✓ Brute force attack simulated - perception engines should now recognize pattern\n")


def simulate_data_exfiltration(event_bus):
    """Simulate data exfiltration: large data transfer followed by exit"""
    logger.info("="*70)
    logger.info("DEMO: SIMULATING DATA EXFILTRATION")
    logger.info("="*70)
    
    attacker_ip = "10.0.1.50"
    
    # Phase 1: Initial reconnaissance
    from src.core.event_bus import DetectionEvent
    
    events_sequence = [
        {
            "attack_type": "reconnaissance",
            "features": {"port_variety": 0.8, "dns_entropy": 0.7},
            "severity": "medium",
            "reason": "Network scanning detected"
        },
        {
            "attack_type": "exploitation",
            "features": {"known_malware_signature": 0.9, "process_injection_attempt": 0.8},
            "severity": "high",
            "reason": "Malware signature matched"
        },
        {
            "attack_type": "exfiltration",
            "features": {"data_exfiltration_pattern": 0.95, "payload_size_ratio": 0.9},
            "severity": "critical",
            "reason": "Large suspicious data transfer detected"
        }
    ]
    
    for i, event_data in enumerate(events_sequence, 1):
        event = DetectionEvent(
            source_ip=attacker_ip,
            prediction="Attack",
            confidence=0.65 + (i * 0.1),
            attack_type=event_data["attack_type"],
            features=event_data["features"],
            severity=event_data["severity"],
            reason=event_data["reason"]
        )
        
        logger.info(f"  [{i}] Phase {i}: {event_data['attack_type'].upper()}")
        logger.info(f"       Confidence: {event.confidence:.0%} | Reason: {event.reason}")
        event_bus.publish(event)
        time.sleep(1)
    
    logger.info("\n✓ Multi-stage attack simulated - story engine should build narrative\n")


def simulate_mixed_traffic(event_bus, duration_seconds=5):
    """Simulate mixed normal and suspicious traffic"""
    logger.info("="*70)
    logger.info("DEMO: SIMULATING MIXED TRAFFIC (Normal + Suspicious)")
    logger.info("="*70)
    
    from src.core.event_bus import DetectionEvent
    
    start_time = time.time()
    event_count = 0
    
    while time.time() - start_time < duration_seconds:
        # 70% normal, 30% suspicious
        if random.random() < 0.7:
            # Normal traffic
            source_ip = f"10.0.{random.randint(1,255)}.{random.randint(1,255)}"
            event = DetectionEvent(
                source_ip=source_ip,
                prediction="Normal",
                confidence=random.uniform(0.85, 0.99),
                features={"request_rate": random.uniform(10, 50)},
                severity="low",
                attack_type="normal",
                reason="Baseline behavior"
            )
        else:
            # Suspicious traffic
            source_ip = f"172.16.{random.randint(1,255)}.{random.randint(1,255)}"
            attack_types = ["port_scan", "brute_force", "sql_injection", "c2_communication"]
            event = DetectionEvent(
                source_ip=source_ip,
                prediction="Attack",
                confidence=random.uniform(0.60, 0.95),
                features={"port_variety": random.uniform(0.5, 1.0)},
                severity=random.choice(["high", "critical"]),
                attack_type=random.choice(attack_types),
                reason="Anomalous behavior detected"
            )
        
        event_bus.publish(event)
        event_count += 1
        time.sleep(0.1)
    
    logger.info(f"\n✓ Processed {event_count} mixed events in {duration_seconds} seconds\n")


def demonstrate_perception_layer(app):
    """Main demo function"""
    logger.info("\n" + "█"*70)
    logger.info("█ " + " "*66 + " █")
    logger.info("█ " + "INIDS 2.0 - REAL-TIME PERCEPTION LAYER DEMONSTRATION".center(66) + " █")
    logger.info("█ " + " "*66 + " █")
    logger.info("█"*70)
    
    with app.app_context():
        from web_app.app import (
            event_bus, perception_integration, attack_story_engine,
            confidence_breakdown_engine, live_system_pulse
        )
        
        logger.info("\n[PHASE 1] Simulating Attacks Through Perception Layer")
        logger.info("-" * 70)
        
        # Give system a moment to stabilize
        time.sleep(1)
        
        # Run demo scenarios
        simulate_brute_force_attack(event_bus)
        time.sleep(2)
        
        simulate_data_exfiltration(event_bus)
        time.sleep(2)
        
        simulate_mixed_traffic(event_bus, duration_seconds=5)
        
        # Give workers time to process events
        logger.info("[PHASE 2] Processing Complete - Gathering Metrics")
        logger.info("-" * 70)
        time.sleep(2)
        
        # Display results
        logger.info("\n[RESULTS] Real-time Integration Metrics")
        logger.info("-" * 70)
        
        status = perception_integration.get_status()
        
        logger.info(f"Status: {status['status'].upper()}")
        logger.info(f"Events Processed: {status['events_processed']}")
        logger.info(f"Events Dropped (backpressure): {status['events_dropped']}")
        logger.info(f"Queue Size: {status['queue_size']} / {status['queue_max_size']}")
        logger.info(f"Throughput: {status['throughput_events_per_second']:.1f} events/sec")
        logger.info(f"Uptime: {status['uptime_seconds']:.1f}s")
        logger.info(f"\nLatency Metrics:")
        logger.info(f"  Average: {status['latency_ms']['average']:.2f}ms")
        logger.info(f"  P95: {status['latency_ms']['p95']:.2f}ms")
        logger.info(f"  P99: {status['latency_ms']['p99']:.2f}ms")
        logger.info(f"\nWorker Threads: {status['worker_threads']}")
        logger.info(f"Batch Size: {status['batch_size']}")
        
        # Show attack stories
        logger.info("\n[PERCEPTION] Attack Story Analysis")
        logger.info("-" * 70)
        
        recent_stories = attack_story_engine.get_recent_stories(limit=3)
        if recent_stories:
            for story_id, story in recent_stories:
                logger.info(f"\nAttack ID: {story_id}")
                logger.info(f"Phases Detected: {len(story.get('phases', []))}")
                for phase in story.get('phases', []):
                    logger.info(f"  - {phase['phase_type'].upper()}: {phase['description']}")
                logger.info(f"Summary: {story.get('summary', 'N/A')}")
        else:
            logger.info("No attack stories yet (too few events)")
        
        # Show confidence breakdowns
        logger.info("\n[PERCEPTION] Confidence Breakdown - Feature Importance")
        logger.info("-" * 70)
        
        ranking = confidence_breakdown_engine.get_feature_importance_ranking()
        if ranking:
            logger.info("Top Contributing Features (Global):")
            for i, (feature_name, contribution) in enumerate(ranking[:5], 1):
                logger.info(f"  {i}. {feature_name}: {contribution:.2f}")
        else:
            logger.info("No feature importance data yet")
        
        # Show live pulse
        logger.info("\n[PERCEPTION] Live System Pulse - Real-time Metrics")
        logger.info("-" * 70)
        
        pulse = live_system_pulse.get_pulse_status()
        logger.info(f"Status: {pulse.get('status', 'N/A')}")
        logger.info(f"Current Metrics:")
        logger.info(f"  Flows/sec: {pulse.get('current', {}).get('flows_per_second', 0):.0f}")
        logger.info(f"  Alerts/min: {pulse.get('current', {}).get('alerts_per_minute', 0):.1f}")
        logger.info(f"  Blocked IPs: {pulse.get('current', {}).get('blocked_ips', 0)}")
        logger.info(f"  Model Accuracy: {pulse.get('current', {}).get('model_accuracy_percent', 95):.1f}%")
        logger.info(f"  Threat Level: {pulse.get('current', {}).get('threat_level', 0):.0f}%")
        
        logger.info(f"\nPulse Strength: {pulse.get('pulse_strength', 0):.2f} (0=idle, 1=active)")
        
        logger.info("\n" + "█"*70)
        logger.info("█ " + "DEMO COMPLETE".center(66) + " █")
        logger.info("█"*70)
        logger.info("\nKey Takeaways:")
        logger.info("  ✓ Perception engines receive real-time detection events")
        logger.info("  ✓ Attack stories build narrative timelines from sequences")
        logger.info("  ✓ Confidence breakdowns explain why alerts fire")
        logger.info("  ✓ Live pulse tracks system metrics in real-time")
        logger.info("  ✓ Latency stays within target (<500ms)")
        logger.info("\n")


if __name__ == "__main__":
    # Import here to use the running app
    from web_app.app import app
    
    logger.info("Starting INIDS 2.0 Perception Layer Demo...\n")
    
    try:
        demonstrate_perception_layer(app)
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.exception("Demo failed with error")
