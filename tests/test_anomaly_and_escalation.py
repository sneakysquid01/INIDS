"""Tests for AnomalyEngine auto-fit rolling buffer and EscalationTracker memory safety."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.detection.engines.anomaly_engine import AnomalyEngine
from src.prevention.escalation_tracker import EscalationTracker, EscalationLevel


# ---------------------------------------------------------------------------
# AnomalyEngine auto-fit
# ---------------------------------------------------------------------------

def test_anomaly_engine_not_ready_before_fit():
    engine = AnomalyEngine(buffer_size=10)
    assert not engine.is_ready()


def test_anomaly_engine_manual_fit():
    engine = AnomalyEngine(buffer_size=10)
    X = np.zeros((20, len(engine._feature_names)))
    engine.fit(X)
    assert engine.is_ready()


def test_anomaly_engine_auto_fit_from_buffer():
    engine = AnomalyEngine(buffer_size=5)
    sample = {f: 0.0 for f in engine._feature_names}
    for i in range(4):
        newly_fit = engine.add_sample(sample)
        assert newly_fit is False
    newly_fit = engine.add_sample(sample)  # 5th sample → triggers fit
    assert newly_fit is True
    assert engine.is_ready()


def test_anomaly_engine_buffer_resets_after_fit():
    engine = AnomalyEngine(buffer_size=3)
    sample = {f: 0.0 for f in engine._feature_names}
    for _ in range(3):
        engine.add_sample(sample)
    status = engine.buffer_status()
    assert status["fitted"] is True
    assert status["buffer_collected"] == 0  # reset after fit


def test_anomaly_engine_buffer_status_progress():
    engine = AnomalyEngine(buffer_size=10)
    sample = {f: 0.0 for f in engine._feature_names}
    engine.add_sample(sample)
    engine.add_sample(sample)
    status = engine.buffer_status()
    assert status["buffer_collected"] == 2
    assert status["buffer_pct"] == 20.0
    assert status["fitted"] is False


def test_anomaly_engine_disabled_buffer():
    engine = AnomalyEngine(buffer_size=0)
    sample = {f: 0.0 for f in engine._feature_names}
    result = engine.add_sample(sample)
    assert result is False
    assert not engine.is_ready()


def test_anomaly_engine_persist_and_load(tmp_path):
    model_path = str(tmp_path / "anomaly.pkl")
    engine = AnomalyEngine(buffer_size=3, model_path=model_path)
    sample = {f: 0.0 for f in engine._feature_names}
    for _ in range(3):
        engine.add_sample(sample)
    assert engine.is_ready()

    # Create a new engine instance pointing to same path — should load persisted model.
    engine2 = AnomalyEngine(buffer_size=3, model_path=model_path)
    assert engine2.is_ready()


def test_fitted_anomaly_engine_evaluate():
    engine = AnomalyEngine(buffer_size=3)
    sample = {f: 0.0 for f in engine._feature_names}
    for _ in range(3):
        engine.add_sample(sample)
    result = engine.evaluate(sample)
    assert result.verdict in ("normal", "suspicious")
    assert 0.0 <= result.confidence <= 100.0


# ---------------------------------------------------------------------------
# EscalationTracker memory safety
# ---------------------------------------------------------------------------

def test_escalation_tracker_evict_stale():
    tracker = EscalationTracker(cooldown_seconds=0.01)
    tracker.record_hit("10.0.0.1", "low")
    time.sleep(0.02)  # wait past cooldown
    evicted = tracker.evict_stale()
    assert evicted >= 1
    assert tracker.get_level("10.0.0.1") == EscalationLevel.CLEAN


def test_escalation_evict_keeps_active_ips():
    tracker = EscalationTracker(cooldown_seconds=60.0)
    tracker.record_hit("10.0.0.10", "high")
    tracker.record_hit("10.0.0.11", "critical")
    evicted = tracker.evict_stale()
    # Active IPs (within cooldown) must not be removed
    assert evicted == 0
    assert "10.0.0.10" in tracker._states
    assert "10.0.0.11" in tracker._states


def test_escalation_max_tracked_cap():
    # max_tracked has an internal floor of 100; use 100 and exceed it by 10.
    tracker = EscalationTracker(cooldown_seconds=300.0, max_tracked=100)
    for i in range(110):
        tracker.record_hit(f"192.168.{i // 256}.{i % 256}", "low")
    # Hard-cap path evicts one entry per new IP beyond the limit.
    assert len(tracker._states) <= 100


def test_escalation_tracker_normal_behavior_preserved():
    tracker = EscalationTracker(cooldown_seconds=300.0, max_tracked=1000)
    tracker.record_hit("10.1.1.1", "high")
    level = tracker.get_level("10.1.1.1")
    # high severity jump=2 from CLEAN → RATE_LIMIT
    assert level >= EscalationLevel.RATE_LIMIT


def test_evict_stale_returns_zero_for_empty():
    tracker = EscalationTracker()
    assert tracker.evict_stale() == 0
