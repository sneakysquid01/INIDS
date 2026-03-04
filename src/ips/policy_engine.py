from __future__ import annotations

from src.core.event_bus import PolicyDecisionEvent, RiskScoreEvent


class PolicyEngine:
    """Maps risk signals to policy decisions."""

    def decide(self, risk_event: RiskScoreEvent, policy) -> PolicyDecisionEvent:
        mode = str(policy.mode).strip().lower()
        prediction = str(risk_event.detection.prediction).strip().lower()
        confidence = float(risk_event.detection.confidence)
        risk_score = float(risk_event.risk_score)

        alert_threshold = float(getattr(policy, "risk_alert_threshold", 0.4))
        rate_limit_threshold = float(getattr(policy, "risk_rate_limit_threshold", 0.6))
        temp_block_threshold = float(getattr(policy, "risk_temp_block_threshold", 0.75))
        block_threshold = float(getattr(policy, "risk_block_threshold", 0.85))
        confidence_block_threshold = float(getattr(policy, "confidence_block_threshold", 85.0))
        ttl_seconds = int(getattr(policy, "block_ttl_seconds", 300))

        if mode in {"monitor", "detect_only", "ids_only"}:
            if prediction == "attack" or risk_score >= alert_threshold:
                return PolicyDecisionEvent(
                    risk=risk_event,
                    decision="ALERT",
                    reason="monitor_mode_alert",
                    ttl_seconds=None,
                )
            return PolicyDecisionEvent(
                risk=risk_event,
                decision="ALLOW",
                reason="monitor_mode_allow",
                ttl_seconds=None,
            )

        if prediction != "attack":
            if risk_score >= alert_threshold:
                return PolicyDecisionEvent(
                    risk=risk_event,
                    decision="ALERT",
                    reason="non_attack_high_risk_alert",
                    ttl_seconds=None,
                )
            return PolicyDecisionEvent(
                risk=risk_event,
                decision="ALLOW",
                reason="non_attack_allow",
                ttl_seconds=None,
            )

        if confidence >= confidence_block_threshold and risk_score >= block_threshold:
            return PolicyDecisionEvent(
                risk=risk_event,
                decision="BLOCK",
                reason="attack_high_confidence_high_risk",
                ttl_seconds=ttl_seconds,
            )
        if risk_score >= temp_block_threshold:
            return PolicyDecisionEvent(
                risk=risk_event,
                decision="TEMP_BLOCK",
                reason="attack_high_risk_temp_block",
                ttl_seconds=max(60, ttl_seconds),
            )
        if risk_score >= rate_limit_threshold:
            return PolicyDecisionEvent(
                risk=risk_event,
                decision="RATE_LIMIT",
                reason="attack_medium_risk",
                ttl_seconds=max(30, min(ttl_seconds, 120)),
            )
        if risk_score >= alert_threshold:
            return PolicyDecisionEvent(
                risk=risk_event,
                decision="ALERT",
                reason="attack_alert_only",
                ttl_seconds=None,
            )

        return PolicyDecisionEvent(
            risk=risk_event,
            decision="ALLOW",
            reason="attack_low_risk_allow",
            ttl_seconds=None,
        )
