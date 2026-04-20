"""
Attack Story Engine - Generates narrative timelines of attack progression.

Analyzes sequences of detections to tell a coherent story of how an attack
unfolded, making it easy for operators to understand the attack flow.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from threading import RLock

logger = logging.getLogger(__name__)


@dataclass
class AttackPhase:
    """Represents a phase in an attack story."""
    timestamp: str
    phase_type: str  # "reconnaissance", "exploitation", "exfiltration", "persistence"
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    indicators: List[str]
    affected_ips: List[str]
    action_taken: Optional[str] = None


class AttackStoryEngine:
    """
    Analyzes sequences of detections to construct a narrative story
    of attack progression.
    """

    # Known attack patterns (reconnaissance → exploitation → exfiltration)
    ATTACK_PHASES = {
        "reconnaissance": {
            "patterns": ["port_scan", "network_scan", "service_enumeration"],
            "title": "🔍 Reconnaissance",
            "description": "Attacker probing network for vulnerabilities"
        },
        "exploitation": {
            "patterns": ["brute_force", "sql_injection", "rce_attempt", "privilege_escalation"],
            "title": "⚔️ Exploitation",
            "description": "Attacker attempting to gain access"
        },
        "exfiltration": {
            "patterns": ["data_exfiltration", "dns_tunneling", "http_tunnel", "c2_communication"],
            "title": "📤 Exfiltration",
            "description": "Attacker stealing data or communicating with C&C"
        },
        "persistence": {
            "patterns": ["backdoor", "persistence_mechanism", "rootkit", "scheduled_task"],
            "title": "🔐 Persistence",
            "description": "Attacker establishing long-term access"
        }
    }

    def __init__(self):
        """Initialize the Attack Story Engine."""
        self.lock = RLock()
        self.detection_cache = {}  # detection_id → detection details
        self.stories = {}  # attack_id → story phases
        logger.info("AttackStoryEngine initialized")

    def record_detection(self, detection_id: str, detection: Dict[str, Any]) -> None:
        """Record a detection for analysis."""
        with self.lock:
            self.detection_cache[detection_id] = {
                **detection,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def analyze_attack_progression(self, detections: List[Dict[str, Any]]) -> List[AttackPhase]:
        """
        Analyze a sequence of detections to identify attack phases.

        Args:
            detections: List of detection records (chronologically ordered)

        Returns:
            List of AttackPhase objects representing the attack story
        """
        with self.lock:
            if not detections:
                return []

            # Sort by timestamp
            sorted_detections = sorted(
                detections,
                key=lambda d: d.get("timestamp", "")
            )

            phases = []
            current_phase = None
            phase_detections = []

            for detection in sorted_detections:
                detected_phase = self._classify_detection(detection)

                # If phase changed, finalize current phase and start new one
                if detected_phase != current_phase and current_phase is not None:
                    phase_obj = self._create_phase(
                        current_phase,
                        phase_detections
                    )
                    if phase_obj:
                        phases.append(phase_obj)
                    phase_detections = []

                current_phase = detected_phase
                phase_detections.append(detection)

            # Add final phase
            if current_phase and phase_detections:
                phase_obj = self._create_phase(current_phase, phase_detections)
                if phase_obj:
                    phases.append(phase_obj)

            return phases

    def _classify_detection(self, detection: Dict[str, Any]) -> Optional[str]:
        """Classify a detection into an attack phase."""
        detection_type = detection.get("attack_type", "").lower()
        reason = detection.get("reason", "").lower()
        combined = f"{detection_type} {reason}".lower()

        # Check against known patterns
        for phase_name, phase_info in self.ATTACK_PHASES.items():
            for pattern in phase_info["patterns"]:
                if pattern in combined:
                    return phase_name

        return None

    def _create_phase(
        self,
        phase_type: str,
        detections: List[Dict[str, Any]]
    ) -> Optional[AttackPhase]:
        """Create an AttackPhase from a group of detections."""
        if not detections:
            return None

        phase_info = self.ATTACK_PHASES.get(phase_type)
        if not phase_info:
            return None

        # Calculate aggregate metrics
        confidences = [
            det.get("confidence", 0.5)
            for det in detections
            if "confidence" in det
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Determine severity based on number of detections and confidence
        if avg_confidence > 0.9 and len(detections) > 5:
            severity = "critical"
        elif avg_confidence > 0.8 or len(detections) > 3:
            severity = "high"
        elif avg_confidence > 0.6:
            severity = "medium"
        else:
            severity = "low"

        # Collect indicators
        indicators = []
        affected_ips = set()

        for det in detections:
            indicators.append(det.get("reason", ""))
            if "source_ip" in det:
                affected_ips.add(det["source_ip"])
            if "target_ip" in det:
                affected_ips.add(det["target_ip"])

        # Use timestamp from first detection
        timestamp = detections[0].get("timestamp", datetime.now(timezone.utc).isoformat())

        return AttackPhase(
            timestamp=timestamp,
            phase_type=phase_type,
            title=phase_info["title"],
            description=phase_info["description"],
            severity=severity,
            confidence=avg_confidence,
            indicators=list(set(indicators))[:5],  # Top 5 unique indicators
            affected_ips=list(affected_ips),
        )

    def get_attack_story(self, attack_id: str) -> Dict[str, Any]:
        """
        Get the complete story for an attack.

        Returns a narrative structure that UI can render as a timeline.
        """
        with self.lock:
            if attack_id not in self.stories:
                return {"phases": [], "summary": "No story available"}

            phases = self.stories[attack_id]

            return {
                "attack_id": attack_id,
                "phases": [
                    {
                        "timestamp": phase.timestamp,
                        "phase_type": phase.phase_type,
                        "title": phase.title,
                        "description": phase.description,
                        "severity": phase.severity,
                        "confidence": phase.confidence,
                        "indicators": phase.indicators,
                        "affected_ips": phase.affected_ips,
                        "action_taken": phase.action_taken,
                    }
                    for phase in phases
                ],
                "summary": self._generate_summary(phases),
            }

    def _generate_summary(self, phases: List[AttackPhase]) -> str:
        """Generate a natural language summary of the attack story."""
        if not phases:
            return "No attack detected"

        phase_names = " → ".join(p.title for p in phases)
        return f"Attack progression: {phase_names}"

    def store_story(self, attack_id: str, phases: List[AttackPhase]) -> None:
        """Store an attack story for later retrieval."""
        with self.lock:
            self.stories[attack_id] = phases
            logger.info(f"Stored attack story for {attack_id} with {len(phases)} phases")

    def get_recent_stories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent attack stories."""
        with self.lock:
            # Return stories in reverse order (newest first)
            stories = []
            for attack_id in list(self.stories.keys())[-limit:]:
                stories.append(self.get_attack_story(attack_id))
            return list(reversed(stories))
