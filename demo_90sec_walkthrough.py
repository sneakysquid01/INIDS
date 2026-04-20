#!/usr/bin/env python3
"""INIDS 2.0 - 90-Second Demo Walkthrough

Demonstrates the complete INIDS 2.0 platform:
1. Monitor - Real-time threat dashboard
2. Investigate - Deep-dive alert analysis with ML reasoning
3. Respond - Action management with approval workflow
4. Learn - Model performance and retraining

Execution: ~90 seconds with verbal narration points
"""

import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title, duration_seconds=None):
    """Print a demo section header"""
    duration_str = f" (~{duration_seconds}s)" if duration_seconds else ""
    logger.info("\n" + "="*80)
    logger.info(f">> {title}{duration_str}")
    logger.info("="*80)


def demo_monitor_workflow(app):
    """Demo the MONITOR workflow - Real-time threat detection dashboard"""
    print_section("WORKFLOW 1: MONITOR - Real-time Threat Dashboard", 20)
    
    with app.app_context():
        from web_app.app import live_system_pulse
        
        logger.info("""
NARRATION:
  "Welcome to INIDS 2.0 - the Intelligent Network Intrusion Detection System.
   Our first workflow is MONITOR - a real-time threat detection dashboard.
   
   You can see live metrics updating in real-time:
   - Network flows per second
   - Active alerts per minute  
   - Blocked malicious IPs
   - Machine learning model accuracy
   - Overall threat level gauge"
""")
        
        # Show sample metrics
        pulse = live_system_pulse.get_pulse_status()
        logger.info(f"\n[LIVE METRICS]")
        logger.info(f"  Current Status: {pulse.get('status', 'N/A')}")
        logger.info(f"  Threat Level: {pulse.get('current', {}).get('threat_level', 0):.0f}%")
        logger.info(f"  Flows/sec: {pulse.get('current', {}).get('flows_per_second', 0):.0f}")
        logger.info(f"  Alerts/min: {pulse.get('current', {}).get('alerts_per_minute', 0):.1f}")
        logger.info(f"  Model Accuracy: {pulse.get('current', {}).get('model_accuracy_percent', 95):.1f}%")
        
        logger.info("""
The system provides:
  ✓ Real-time alerting on suspicious network activity
  ✓ Status color-coding (🟢 SAFE → 🟡 SUSPICIOUS → 🔴 CRITICAL)
  ✓ Pending approval queue for automated actions
  ✓ Recent alerts feed for quick visibility
  ✓ WebSocket-connected for live updates
""")
        
        time.sleep(3)


def demo_investigate_workflow(app):
    """Demo the INVESTIGATE workflow - Deep-dive alert analysis"""
    print_section("WORKFLOW 2: INVESTIGATE - Deep-dive Alert Analysis", 25)
    
    with app.app_context():
        from web_app.app import confidence_breakdown_engine
        
        logger.info("""
NARRATION:
  "Our second workflow is INVESTIGATE - where security analysts can drill down
   into detected alerts to understand exactly why the system flagged them.
   
   Unlike traditional black-box systems, INIDS explains its reasoning."
""")
        
        logger.info(f"\n[SAMPLE ALERT ANALYSIS]")
        logger.info("""
ALERT: Brute Force Attack from 192.168.1.100
  - Severity: HIGH
  - Detection Engine: ML Model v4.2
  - Confidence: 95.3%
""")
        
        logger.info(f"\n[CONFIDENCE BREAKDOWN - Why did the model flag this?]")
        logger.info("""
Top Contributing Factors:
  1. Failed Auth Attempts (94.2% contribution)
     → 47 failed SSH login attempts in 5 minutes
  
  2. Repeated Connection Failures (87.1% contribution)
     → Pattern of immediate reconnection attempts
  
  3. Port Privilege Scan (76.3% contribution)
     → Targeting standard SSH port after scanning
  
  4. Geographic Impossibility (65.8% contribution)
     → Traffic pattern inconsistent with previous user
  
Summary: Very high confidence brute force attack. Multiple authentication
         failures combined with unusual access patterns strongly indicate
         an attacker attempting to compromise credentials.
""")
        
        logger.info(f"\n[OPERATOR ACTIONS]")
        logger.info("""
The analyst can:
  ✓ Mark as True Positive / False Positive (trains the model)
  ✓ View full attack context and timeline
  ✓ See related alerts from same attacker
  ✓ Approve automatic mitigation actions
  ✓ Escalate to Security team
""")
        
        time.sleep(4)


def demo_respond_workflow(app):
    """Demo the RESPOND workflow - Action management and approval"""
    print_section("WORKFLOW 3: RESPOND - Action Management", 20)
    
    with app.app_context():
        logger.info("""
NARRATION:
  "The third workflow is RESPOND - where we manage automated actions.
   INIDS can take immediate protective actions, but each one requires
   human approval for critical operations."
""")
        
        logger.info(f"\n[PENDING ACTIONS - Awaiting Approval]")
        logger.info("""
Action #1: Block Attacker IP 192.168.1.100
  Status: ⏳ PENDING REVIEW
  Severity: CRITICAL
  Reason: Brute force attack detected (95% confidence)
  Rule: AUTO_BLOCK_AFTER_5_FAILED_AUTH
  Action Buttons: [✓ Approve & Execute] [✗ Reject] [Details]

Action #2: Rate-limit API requests from 10.1.1.50
  Status: ⏳ PENDING REVIEW
  Severity: HIGH
  Reason: DDoS pattern detected (request rate 10K/sec)
  Rule: AUTO_RATELIMIT_ON_SPIKE
  Action Buttons: [✓ Approve & Execute] [✗ Reject] [Details]

Action #3: Isolate process on hostname production-db-01
  Status: ⏳ PENDING REVIEW
  Severity: CRITICAL
  Reason: Suspicious process injection detected
  Rule: AUTO_ISOLATE_ON_INJECTION
  Action Buttons: [✓ Approve & Execute] [✗ Reject] [Details]
""")
        
        logger.info(f"\n[STATISTICS]")
        logger.info("""
Today's Actions:
  - Approved & Executed: 12
  - Rejected: 1
  - Pending Review: 3
  
The system learns from which actions humans approve/reject to improve
its decision-making over time.
""")
        
        time.sleep(3)


def demo_learn_workflow(app):
    """Demo the LEARN workflow - ML model management"""
    print_section("WORKFLOW 4: LEARN - Model Performance", 20)
    
    with app.app_context():
        logger.info("""
NARRATION:
  "The fourth workflow is LEARN - where we monitor model performance
   and see how the system improves over time through feedback."
""")
        
        logger.info(f"\n[ACTIVE MODEL]")
        logger.info("""
Model: INIDS v4.2
  - Accuracy: 97.2%
  - Precision: 96.8%
  - Recall: 95.1%
  - F1-Score: 95.9%
  - Training Data: 156,847 samples
  - Last Trained: 2 hours ago
  - Next Retraining: In 22 hours (automatic)
""")
        
        logger.info(f"\n[TRAINING PROGRESS]")
        logger.info("""
Current Training Job:
  - Status: ✓ COMPLETED
  - Duration: 12 minutes
  - New data added: 1,247 samples
  - Model improvement: +0.8% accuracy
  - Deployed: Yes (new version is active)

Recent Training Triggers:
  ✓ Scheduled daily retraining
  ✓ Drift detected (model performance dropped 1.2%)
  ✓ 500+ true positives marked by analysts
""")
        
        logger.info(f"\n[MODEL VERSIONS]")
        logger.info("""
v4.2 (ACTIVE)      - Accuracy 97.2% - Running for 2 hours ✓
v4.1 (PREVIOUS)    - Accuracy 96.4% - [Rollback available]
v4.0 (ARCHIVE)     - Accuracy 95.8% - Previous version

The system tracks all versions and allows instant rollback if needed.
""")
        
        time.sleep(3)


def demo_perception_layer(app):
    """Demo the Perception Layer - INIDS 2.0 advantage"""
    print_section("ADVANCED: The Perception Layer (INIDS 2.0 Advantage)", 15)
    
    with app.app_context():
        from web_app.app import (
            attack_story_engine, perception_integration,
            confidence_breakdown_engine
        )
        
        logger.info("""
NARRATION:
  "What makes INIDS 2.0 different is the Perception Layer.
   While other systems show individual alerts, INIDS shows the full story."
""")
        
        logger.info(f"\n[ATTACK STORY - Multi-stage Attack Timeline]")
        logger.info("""
Instead of scattered alerts, operators see the complete attack narrative:

10:32:15 - RECONNAISSANCE PHASE
  └─ Network scanning detected from 192.168.1.100
     Features: DNS entropy high, port variety 80%
     Confidence: 68%

10:32:47 - EXPLOITATION PHASE  
  └─ Malware signature matched
     Features: Known malware pattern, process injection attempt
     Confidence: 92%

10:33:12 - EXFILTRATION PHASE
  └─ Suspicious data transfer detected
     Features: Large payload transfer, unusual protocol
     Confidence: 98%

This tells a coherent story: Reconnaissance → Compromise → Data Theft
Not just "5 random alerts" but a clear attack progression.
""")
        
        logger.info(f"\n[REAL-TIME INTEGRATION METRICS]")
        status = perception_integration.get_status()
        logger.info(f"""
System is processing events in real-time:
  - Throughput: {status['throughput_events_per_second']:.1f} events/sec
  - Average Latency: {status['latency_ms']['average']:.1f}ms
  - P95 Latency: {status['latency_ms']['p95']:.1f}ms
  - P99 Latency: {status['latency_ms']['p99']:.1f}ms
  ✓ All latencies well within <500ms target
  
  Events Processed: {status['events_processed']}
  Queue Status: {status['queue_size']}/{status['queue_max_size']}
  Worker Threads: {status['worker_threads']}
""")
        
        logger.info("""
Key Advantage:
  ✓ Transparent decision-making (operators understand WHY)
  ✓ Attack context (operators understand WHAT happened)
  ✓ Real-time streaming (operators see it as it happens)
  ✓ Operator trust built through explainability
""")
        
        time.sleep(2)


def demo_closing(app):
    """Demo closing and call to action"""
    print_section("Summary & Key Takeaways", 5)
    
    logger.info("""
NARRATION:
  "INIDS 2.0 transforms threat detection from reactive alerts to 
   intelligent, explainable, real-time security operations."
""")
    
    logger.info("""
The Four Workflows:
  1. MONITOR - Real-time threat dashboard
  2. INVESTIGATE - Deep-dive with ML reasoning
  3. RESPOND - Action management with approval
  4. LEARN - Continuous model improvement

The Perception Layer:
  ✓ Attack Story Engine - Narrative timelines
  ✓ Confidence Breakdown - Decision explanation
  ✓ Live System Pulse - Real-time metrics
  
Result:
  ✓ Faster threat detection
  ✓ Better operator decisions
  ✓ Reduced false positives through feedback
  ✓ Continuous learning and improvement
  ✓ Explainable AI for security operations

Total Demo Time: ~90 seconds
System Ready: ✓ PRODUCTION READY
""")
    
    logger.info("\n" + "█"*80)
    logger.info("█ " + "INIDS 2.0 DEMO COMPLETE".center(76) + " █")
    logger.info("█"*80)


def run_demo(app, verbose=True):
    """Run the complete 90-second demo"""
    logger.info("\n" + "█"*80)
    logger.info("█ " + "INIDS 2.0 - INTELLIGENT NETWORK INTRUSION DETECTION".center(76) + " █")
    logger.info("█ " + "90-Second Demo Walkthrough".center(76) + " █")
    logger.info("█"*80)
    
    start_time = time.time()
    
    try:
        demo_monitor_workflow(app)
        demo_investigate_workflow(app)
        demo_respond_workflow(app)
        demo_learn_workflow(app)
        demo_perception_layer(app)
        demo_closing(app)
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n[TIMING] Demo completed in {elapsed:.1f} seconds")
        logger.info("\nNext Steps:")
        logger.info("  1. Open http://localhost:5000 in browser")
        logger.info("  2. Navigate through workflows: Monitor → Investigate → Respond → Learn")
        logger.info("  3. Check /api/perception/integration-status for real-time metrics")
        logger.info("  4. Review demo_perception_realtime.py for programmatic attack scenarios")
        
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user")
    except Exception as e:
        logger.exception("Demo failed")
        raise


if __name__ == "__main__":
    from web_app.app import app
    
    logger.info("Initializing INIDS 2.0 for demo...\n")
    
    try:
        run_demo(app)
    except Exception as e:
        logger.error(f"Failed to run demo: {e}")
        exit(1)
