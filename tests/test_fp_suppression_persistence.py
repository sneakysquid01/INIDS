"""Tests for FalsePositiveManager persistence and enforcement in SignatureEngine."""
from __future__ import annotations

import pytest

from src.ops_store import OpsStore
from src.prevention.false_positive_manager import FalsePositiveManager
from src.detection.engines.signature_engine import SignatureEngine


@pytest.fixture()
def store(tmp_path):
    return OpsStore(str(tmp_path / "test.db"))


@pytest.fixture()
def fp_mgr(store):
    return FalsePositiveManager(suppress_threshold=0.7, min_samples=2, ops_store=store)


# ---------------------------------------------------------------------------
# OpsStore FP suppression CRUD
# ---------------------------------------------------------------------------

def test_save_and_list_fp_suppression(store):
    store.save_fp_suppression("signature", "SIG-001")
    rows = store.list_fp_suppressions()
    assert any(r["engine_id"] == "signature" and r["rule_id"] == "SIG-001" for r in rows)


def test_delete_fp_suppression(store):
    store.save_fp_suppression("signature", "SIG-002")
    assert store.delete_fp_suppression("signature", "SIG-002")
    rows = store.list_fp_suppressions()
    assert not any(r["rule_id"] == "SIG-002" for r in rows)


def test_fp_suppression_idempotent(store):
    store.save_fp_suppression("signature", "SIG-005")
    store.save_fp_suppression("signature", "SIG-005")  # second insert should not raise
    rows = store.list_fp_suppressions()
    sig005 = [r for r in rows if r["rule_id"] == "SIG-005"]
    assert len(sig005) == 1


# ---------------------------------------------------------------------------
# FalsePositiveManager persistence
# ---------------------------------------------------------------------------

def test_auto_suppress_persists(fp_mgr, store):
    fp_mgr.report_fp("signature", "SIG-001")
    fp_mgr.report_fp("signature", "SIG-001")  # 2/2 ≥ 0.7 threshold
    assert fp_mgr.is_suppressed("signature", "SIG-001")
    rows = store.list_fp_suppressions()
    assert any(r["rule_id"] == "SIG-001" for r in rows)


def test_explicit_suppress_persists(fp_mgr, store):
    newly = fp_mgr.suppress("signature", "SIG-003")
    assert newly is True
    assert fp_mgr.is_suppressed("signature", "SIG-003")
    rows = store.list_fp_suppressions()
    assert any(r["rule_id"] == "SIG-003" for r in rows)


def test_unsuppress_removes_from_store(fp_mgr, store):
    fp_mgr.suppress("signature", "SIG-010")
    removed = fp_mgr.unsuppress("signature", "SIG-010")
    assert removed is True
    assert not fp_mgr.is_suppressed("signature", "SIG-010")
    rows = store.list_fp_suppressions()
    assert not any(r["rule_id"] == "SIG-010" for r in rows)


def test_load_from_store_restores_state(store):
    store.save_fp_suppression("signature", "SIG-007")
    mgr2 = FalsePositiveManager(ops_store=store)
    loaded = mgr2.load_from_store()
    assert loaded >= 1
    assert mgr2.is_suppressed("signature", "SIG-007")


# ---------------------------------------------------------------------------
# SignatureEngine enforcement
# ---------------------------------------------------------------------------

def test_signature_engine_skips_suppressed_rule():
    """Suppressed rules must not produce attack verdicts."""
    fp_mgr = FalsePositiveManager(suppress_threshold=0.7, min_samples=1)
    fp_mgr.suppress("signature", "SIG-001")

    engine = SignatureEngine(fp_manager=fp_mgr)
    engine.add_rule({
        "id": "SIG-001",
        "name": "Test Rule",
        "severity": "high",
        "attack_type": "dos",
        "conditions": [{"field": "count", "op": ">", "value": 0}],
    })
    result = engine.evaluate({"count": 999})
    assert result.verdict == "normal", "Suppressed rule must not trigger attack verdict"


def test_signature_engine_fires_unsuppressed_rule():
    """Non-suppressed rules must still work normally."""
    fp_mgr = FalsePositiveManager()
    engine = SignatureEngine(fp_manager=fp_mgr)
    engine.add_rule({
        "id": "SIG-002",
        "name": "Active Rule",
        "severity": "high",
        "attack_type": "dos",
        "conditions": [{"field": "count", "op": ">", "value": 0}],
    })
    result = engine.evaluate({"count": 999})
    assert result.verdict == "attack"


def test_signature_engine_skips_suppressed_falls_through_to_next():
    """After suppressing first-match rule, second rule should match."""
    fp_mgr = FalsePositiveManager()
    fp_mgr.suppress("signature", "SIG-A")

    engine = SignatureEngine(fp_manager=fp_mgr)
    engine.add_rule({
        "id": "SIG-A",
        "name": "Suppressed Rule",
        "severity": "high",
        "attack_type": "dos",
        "conditions": [{"field": "count", "op": ">", "value": 0}],
    })
    engine.add_rule({
        "id": "SIG-B",
        "name": "Active Rule",
        "severity": "medium",
        "attack_type": "probe",
        "conditions": [{"field": "count", "op": ">", "value": 0}],
    })
    result = engine.evaluate({"count": 999})
    assert result.verdict == "attack"
    assert result.rule_id == "SIG-B"
