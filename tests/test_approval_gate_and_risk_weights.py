"""Tests for PENDING_BLOCK approval gate and configurable risk weights."""
from __future__ import annotations

import pytest

from src.ips.policy_engine import PolicyEngine
from src.ips.risk_engine import RiskEngine, RiskWeights
from src.prevention_service import PolicyConfig
from src.core.event_bus import DetectionEvent, RiskScoreEvent


def _make_risk_event(prediction="attack", confidence=95.0, risk_score=0.9):
    detection = DetectionEvent(
        source_ip="10.0.0.1",
        prediction=prediction,
        confidence=confidence,
        severity="high",
    )
    return RiskScoreEvent(
        detection=detection,
        risk_score=risk_score,
        components={"confidence": 0.95, "severity": 0.85, "frequency": 0.5},
    )


class _StrictPolicy:
    """Policy that requires approval before blocks."""
    mode = "auto_block"
    risk_alert_threshold = 0.4
    risk_rate_limit_threshold = 0.6
    risk_temp_block_threshold = 0.75
    risk_block_threshold = 0.85
    confidence_block_threshold = 85.0
    block_ttl_seconds = 300
    block_requires_approval = True
    dry_run = False


class _PermissivePolicy:
    """Policy that auto-blocks without approval."""
    mode = "auto_block"
    risk_alert_threshold = 0.4
    risk_rate_limit_threshold = 0.6
    risk_temp_block_threshold = 0.75
    risk_block_threshold = 0.85
    confidence_block_threshold = 85.0
    block_ttl_seconds = 300
    block_requires_approval = False
    dry_run = False


# ---------------------------------------------------------------------------
# PolicyEngine PENDING_BLOCK gate
# ---------------------------------------------------------------------------

def test_pending_block_when_approval_required():
    engine = PolicyEngine()
    risk = _make_risk_event(confidence=95.0, risk_score=0.95)
    decision = engine.decide(risk, _StrictPolicy())
    assert decision.decision == "PENDING_BLOCK"


def test_direct_block_when_no_approval_required():
    engine = PolicyEngine()
    risk = _make_risk_event(confidence=95.0, risk_score=0.95)
    decision = engine.decide(risk, _PermissivePolicy())
    assert decision.decision == "BLOCK"


def test_temp_block_becomes_pending_with_approval():
    engine = PolicyEngine()
    # risk_score in [0.75, 0.85) range → TEMP_BLOCK normally
    risk = _make_risk_event(confidence=50.0, risk_score=0.78)
    decision = engine.decide(risk, _StrictPolicy())
    assert decision.decision == "PENDING_BLOCK"


def test_rate_limit_not_affected_by_approval_flag():
    engine = PolicyEngine()
    risk = _make_risk_event(confidence=50.0, risk_score=0.65)
    decision = engine.decide(risk, _StrictPolicy())
    # RATE_LIMIT does not go through the PENDING_BLOCK gate
    assert decision.decision == "RATE_LIMIT"


def test_monitor_mode_never_pending_block():
    class MonitorPolicy(_StrictPolicy):
        mode = "monitor"

    engine = PolicyEngine()
    risk = _make_risk_event(confidence=99.0, risk_score=0.99)
    decision = engine.decide(risk, MonitorPolicy())
    assert decision.decision == "ALERT"


# ---------------------------------------------------------------------------
# Configurable risk weights
# ---------------------------------------------------------------------------

def test_risk_weights_override_changes_score():
    engine = RiskEngine()
    detection = DetectionEvent(
        source_ip="10.0.0.2", prediction="attack", confidence=95.0, severity="high"
    )
    default_result = engine.calculate(detection)
    # Increase confidence weight significantly
    heavy_confidence = RiskWeights(confidence=1.0, severity=0.0, frequency=0.0)
    heavy_result = engine.calculate(detection, weights_override=heavy_confidence)
    assert heavy_result.risk_score != default_result.risk_score


def test_risk_weights_override_confidence_only():
    engine = RiskEngine()
    detection = DetectionEvent(
        source_ip="10.0.0.3", prediction="attack", confidence=60.0, severity="low"
    )
    w = RiskWeights(confidence=1.0, severity=0.0, frequency=0.0)
    result = engine.calculate(detection, weights_override=w)
    # With confidence=1.0, 0.0, 0.0 the score should equal normalized confidence ≈ 0.60
    assert abs(result.risk_score - 0.60) < 0.05


def test_risk_default_weights_unchanged():
    """Passing no override should use the default weights."""
    engine = RiskEngine(weights=RiskWeights(confidence=0.5, severity=0.3, frequency=0.2))
    # Use different IPs so the frequency counter starts at the same level for both.
    det1 = DetectionEvent(
        source_ip="10.0.0.41", prediction="normal", confidence=10.0, severity="low"
    )
    det2 = DetectionEvent(
        source_ip="10.0.0.42", prediction="normal", confidence=10.0, severity="low"
    )
    result1 = engine.calculate(det1)
    result2 = engine.calculate(det2)
    assert abs(result1.risk_score - result2.risk_score) < 0.001


# ---------------------------------------------------------------------------
# PolicyConfig integration
# ---------------------------------------------------------------------------

def test_policy_config_defaults():
    cfg = PolicyConfig()
    assert cfg.block_requires_approval is False
    assert cfg.risk_weight_confidence == 0.5
    assert cfg.risk_weight_severity == 0.3
    assert cfg.risk_weight_frequency == 0.2


def test_set_policy_updates_approval_flag():
    from src.prevention_service import PreventionService
    svc = PreventionService()
    policy = svc.set_policy(block_requires_approval=True)
    assert policy.block_requires_approval is True


def test_set_policy_updates_risk_weights():
    from src.prevention_service import PreventionService
    svc = PreventionService()
    policy = svc.set_policy(
        risk_weight_confidence=0.6, risk_weight_severity=0.2, risk_weight_frequency=0.2
    )
    assert policy.risk_weight_confidence == 0.6
    assert policy.risk_weight_severity == 0.2


def test_set_policy_rejects_invalid_weight():
    from src.prevention_service import PreventionService
    svc = PreventionService()
    with pytest.raises(ValueError):
        svc.set_policy(risk_weight_confidence=1.5)
