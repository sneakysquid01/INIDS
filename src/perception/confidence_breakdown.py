"""
Confidence Breakdown Engine - Explains model reasoning and decision factors.

Displays which features most influenced a detection, making the model
less of a "black box" and helping operators understand why alerts fire.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from threading import RLock

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Represents how much a feature influenced the decision."""
    feature_name: str
    value: Any
    contribution_score: float  # -1.0 to 1.0 (negative = suggests normal, positive = suggests attack)
    importance_rank: int  # 1 = most important
    explanation: str


class ConfidenceBreakdownEngine:
    """
    Analyzes model decisions to explain which features drove the prediction.

    This makes the model's reasoning transparent to operators, building trust
    and helping them understand when alerts are justified.
    """

    # Feature explanations (how each feature relates to attack probability)
    FEATURE_EXPLANATIONS = {
        "connection_count": "Number of concurrent connections from source",
        "failed_auth_attempts": "Failed authentication attempts (higher = more suspicious)",
        "port_variety": "How many different destination ports scanned",
        "protocol_anomaly": "Detection of unusual protocol usage",
        "payload_size_ratio": "Unusual ratio of incoming vs outgoing data",
        "request_rate": "Rate of requests per second",
        "geographic_impossibility": "Source IP location inconsistent with network",
        "known_malware_signature": "Match against known malware signatures",
        "dns_entropy": "Randomness in DNS queries (high = tunneling)",
        "tls_certificate_anomaly": "Invalid or mismatched TLS certificate",
        "port_privilege_scan": "Scanning for privileged ports (1-1024)",
        "repeated_connection_failures": "Multiple failed connection attempts",
        "c2_ip_reputation": "IP matches known C&C server list",
        "data_exfiltration_pattern": "Pattern matches known data theft behavior",
        "process_injection_attempt": "Process memory modification detected",
    }

    # Feature impact thresholds
    STRONG_INDICATOR_THRESHOLD = 0.7
    WEAK_INDICATOR_THRESHOLD = 0.3

    def __init__(self):
        """Initialize the Confidence Breakdown Engine."""
        self.lock = RLock()
        self.decision_cache = {}  # detection_id → breakdown
        logger.info("ConfidenceBreakdownEngine initialized")

    def analyze_detection(
        self,
        detection_id: str,
        detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a detection to break down confidence into contributing factors.

        Args:
            detection_id: Unique identifier for this detection
            detection: Detection data with features and model output

        Returns:
            Dictionary with breakdown analysis
        """
        with self.lock:
            # Extract features
            features = detection.get("features", {})
            confidence = detection.get("confidence", 0.0)
            attack_type = detection.get("attack_type", "Unknown")

            # Calculate feature contributions
            contributions = self._calculate_feature_contributions(
                features,
                confidence,
                attack_type
            )

            # Sort by importance
            contributions.sort(
                key=lambda c: abs(c.contribution_score),
                reverse=True
            )

            # Assign ranks
            for i, contribution in enumerate(contributions, 1):
                contribution.importance_rank = i

            breakdown = {
                "detection_id": detection_id,
                "overall_confidence": confidence,
                "attack_type": attack_type,
                "total_features_analyzed": len(features),
                "strong_indicators_count": len([c for c in contributions if abs(c.contribution_score) > self.STRONG_INDICATOR_THRESHOLD]),
                "weak_indicators_count": len([c for c in contributions if abs(c.contribution_score) <= self.WEAK_INDICATOR_THRESHOLD]),
                "top_features": [
                    {
                        "rank": c.importance_rank,
                        "feature_name": c.feature_name,
                        "value": str(c.value)[:100],  # Truncate long values
                        "contribution": c.contribution_score,
                        "explanation": c.explanation,
                        "indicator_type": "strong" if abs(c.contribution_score) > self.STRONG_INDICATOR_THRESHOLD else "weak"
                    }
                    for c in contributions[:10]  # Top 10 features
                ],
                "summary": self._generate_summary(contributions, confidence, attack_type),
            }

            # Cache for quick retrieval
            self.decision_cache[detection_id] = breakdown

            return breakdown

    def _calculate_feature_contributions(
        self,
        features: Dict[str, Any],
        confidence: float,
        attack_type: str
    ) -> List[FeatureContribution]:
        """
        Calculate how much each feature contributed to the decision.

        This is a simplified analysis. In production, this would use
        SHAP values or similar explainability methods.
        """
        contributions = []

        # Known attack-specific features
        attack_indicators = {
            "brute_force": ["failed_auth_attempts", "repeated_connection_failures", "port_privilege_scan"],
            "port_scan": ["port_variety", "connection_count", "port_privilege_scan"],
            "data_exfiltration": ["payload_size_ratio", "dns_entropy", "data_exfiltration_pattern"],
            "c2_communication": ["c2_ip_reputation", "dns_entropy", "tls_certificate_anomaly"],
            "malware": ["known_malware_signature", "process_injection_attempt", "repeated_connection_failures"],
        }

        expected_features = attack_indicators.get(attack_type, [])

        for feature_name, feature_value in features.items():
            # Normalize numeric features to 0-1 range
            normalized_value = self._normalize_feature(feature_name, feature_value)

            # Calculate contribution (higher for features matching attack type)
            if feature_name in expected_features:
                contribution = normalized_value * confidence
            else:
                contribution = normalized_value * (confidence * 0.5)

            # Get explanation
            explanation = self.FEATURE_EXPLANATIONS.get(
                feature_name,
                f"Automated detection: {feature_name}"
            )

            contributions.append(
                FeatureContribution(
                    feature_name=feature_name,
                    value=feature_value,
                    contribution_score=contribution,
                    importance_rank=0,  # Will be assigned later
                    explanation=explanation
                )
            )

        return contributions

    def _normalize_feature(self, feature_name: str, value: Any) -> float:
        """
        Normalize a feature value to 0-1 range.

        Converts various types to a suspicious score.
        """
        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float)):
            # Clamp to 0-1
            return min(1.0, max(0.0, float(value)))

        if isinstance(value, str):
            # String length as proxy for suspicion
            return min(1.0, len(value) / 100.0)

        if isinstance(value, list):
            return min(1.0, len(value) / 10.0)

        return 0.5  # Default uncertainty

    def _generate_summary(
        self,
        contributions: List[FeatureContribution],
        confidence: float,
        attack_type: str
    ) -> str:
        """Generate natural language summary of the breakdown."""
        if not contributions:
            return f"Detection confidence: {confidence:.1%}"

        top_features = [c.feature_name for c in contributions[:3]]
        features_str = ", ".join(top_features)

        confidence_level = "Very High" if confidence > 0.9 else "High" if confidence > 0.75 else "Medium" if confidence > 0.5 else "Low"

        return f"{confidence_level} confidence {attack_type} detection ({confidence:.1%}). Key indicators: {features_str}"

    def get_breakdown(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached breakdown."""
        with self.lock:
            return self.decision_cache.get(detection_id)

    def compare_detections(
        self,
        detection_id_1: str,
        detection_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare confidence breakdowns of two detections.

        Useful for understanding why two similar-looking alerts have
        different confidence scores.
        """
        with self.lock:
            breakdown_1 = self.decision_cache.get(detection_id_1)
            breakdown_2 = self.decision_cache.get(detection_id_2)

            if not breakdown_1 or not breakdown_2:
                return {"error": "One or both detections not found"}

            return {
                "detection_1": detection_id_1,
                "detection_2": detection_id_2,
                "confidence_1": breakdown_1.get("overall_confidence"),
                "confidence_2": breakdown_2.get("overall_confidence"),
                "confidence_difference": abs(
                    breakdown_1.get("overall_confidence", 0) -
                    breakdown_2.get("overall_confidence", 0)
                ),
                "top_features_1": breakdown_1.get("top_features", [])[:5],
                "top_features_2": breakdown_2.get("top_features", [])[:5],
                "summary": f"Detection 1 is {abs(breakdown_1.get('overall_confidence', 0) - breakdown_2.get('overall_confidence', 0)):.1%} more confident"
            }

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Get overall importance ranking of features across all cached decisions.

        Shows which features are most impactful globally.
        """
        with self.lock:
            feature_scores = {}

            for breakdown in self.decision_cache.values():
                for feature in breakdown.get("top_features", []):
                    fname = feature.get("feature_name", "")
                    contrib = abs(feature.get("contribution", 0))
                    feature_scores[fname] = feature_scores.get(fname, 0) + contrib

            # Sort by score
            ranked = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return ranked[:15]  # Top 15 features
