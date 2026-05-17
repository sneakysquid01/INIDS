"""Regression tests for B-06: Eliminate duplicate alert/action stores.

Validation checkpoints per PLAN.md:
1. After sub-deploy 2: GET /api/alerts count matches OpsStore
2. After sub-deploy 3: no InMemoryAlertStore in DetectionService constructor,
   no InMemoryPreventionStore in PreventionService
3. Dual-write: alerts saved to ops_store when ops_store is provided
4. drain_to_ops_store() drains all records, raises on failures
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Sub-deploy 1: Dual-write — DetectionService writes to ops_store
# ---------------------------------------------------------------------------


class FakeModel:
    def predict(self, _df):
        return [1]  # always "Attack"

    def predict_proba(self, _df):
        return [[0.01, 0.99]]


class TestDetectionServiceDualWrite:
    def test_alert_written_to_ops_store_on_attack(self):
        from src.detection_service import DetectionService

        ops_mock = MagicMock()
        svc = DetectionService(model=FakeModel(), ops_store=ops_mock)
        result = svc.predict_from_features({"duration": 1.0}, source_ip="1.2.3.4")

        assert result.alert is not None
        ops_mock.save_alert.assert_called_once()
        saved = ops_mock.save_alert.call_args[0][0]
        assert saved["id"] == result.alert.id
        assert saved["source_ip"] == "1.2.3.4"
        assert saved["prediction"] == "Attack"

    def test_alert_not_written_when_ops_store_is_none(self):
        from src.detection_service import DetectionService

        svc = DetectionService(model=FakeModel(), ops_store=None)
        result = svc.predict_from_features({"duration": 1.0})

        assert result.alert is not None  # alert still returned in result

    def test_ops_store_failure_does_not_crash_service(self):
        from src.detection_service import DetectionService

        ops_mock = MagicMock()
        ops_mock.save_alert.side_effect = Exception("db down")
        svc = DetectionService(model=FakeModel(), ops_store=ops_mock)

        result = svc.predict_from_features({"duration": 1.0})
        assert result.alert is not None  # PredictionResult still populated

    def test_dual_write_to_both_alert_store_and_ops_store(self):
        from src.detection_service import DetectionService, InMemoryAlertStore

        ops_mock = MagicMock()
        legacy_store = InMemoryAlertStore()
        svc = DetectionService(model=FakeModel(), alert_store=legacy_store, ops_store=ops_mock)
        result = svc.predict_from_features({"duration": 1.0})

        assert result.alert is not None
        ops_mock.save_alert.assert_called_once()
        assert len(legacy_store.list_alerts()) == 1


# ---------------------------------------------------------------------------
# Sub-deploy 3: No InMemoryPreventionStore in PreventionService
# ---------------------------------------------------------------------------


class TestPreventionServiceStoreRemoved:
    def test_prevention_service_has_no_store_attribute(self):
        from src.prevention_service import PreventionService

        svc = PreventionService()
        assert not hasattr(svc, "store"), (
            "InMemoryPreventionStore removed from PreventionService in B-06 sub-deploy 3"
        )

    def test_prevention_service_init_no_store_param(self):
        import inspect
        from src.prevention_service import PreventionService

        params = inspect.signature(PreventionService.__init__).parameters
        assert "store" not in params, (
            "'store' parameter must not exist in PreventionService after B-06 sub-deploy 3"
        )

    def test_inmemory_prevention_store_class_still_importable(self):
        from src.prevention_service import InMemoryPreventionStore

        store = InMemoryPreventionStore()
        assert hasattr(store, "add_action")


# ---------------------------------------------------------------------------
# Sub-deploy 3: DetectionService constructor no longer requires alert_store
# ---------------------------------------------------------------------------


class TestDetectionServiceNoAlertStore:
    def test_detection_service_requires_no_alert_store(self):
        from src.detection_service import DetectionService

        svc = DetectionService(model=FakeModel())
        result = svc.predict_from_features({"duration": 1.0})
        assert result.alert is not None

    def test_detection_service_ops_store_is_primary_write_path(self):
        from src.detection_service import DetectionService

        writes = []
        ops_mock = MagicMock()
        ops_mock.save_alert.side_effect = lambda p: writes.append(p)

        svc = DetectionService(model=FakeModel(), ops_store=ops_mock)
        result = svc.predict_from_features({"duration": 1.0})

        assert len(writes) == 1
        assert writes[0]["id"] == result.alert.id


# ---------------------------------------------------------------------------
# drain_to_ops_store()
# ---------------------------------------------------------------------------


class TestDrainToOpsStore:
    def _make_source_store(self):
        from src.detection_service import InMemoryAlertStore, Alert

        store = InMemoryAlertStore(max_items=100)
        for i in range(3):
            store.add(Alert(
                id=f"al_{i:05d}",
                timestamp="2026-01-01T00:00:00+00:00",
                severity="high",
                prediction="Attack",
                confidence=91.0,
                profile="balanced",
                reason="model_prediction",
            ))
        return store

    def test_drain_returns_correct_count(self):
        from src.detection_service import drain_to_ops_store

        source = self._make_source_store()
        ops_mock = MagicMock()
        count = drain_to_ops_store(source, ops_mock)
        assert count == 3

    def test_drain_calls_save_alert_for_each_record(self):
        from src.detection_service import drain_to_ops_store

        source = self._make_source_store()
        ops_mock = MagicMock()
        drain_to_ops_store(source, ops_mock)
        assert ops_mock.save_alert.call_count == 3

    def test_drain_raises_on_any_failure(self):
        import pytest
        from src.detection_service import drain_to_ops_store

        source = self._make_source_store()
        ops_mock = MagicMock()
        ops_mock.save_alert.side_effect = Exception("db error")

        with pytest.raises(RuntimeError, match="drain failures"):
            drain_to_ops_store(source, ops_mock)

    def test_drain_empty_store_returns_zero(self):
        from src.detection_service import drain_to_ops_store, InMemoryAlertStore

        source = InMemoryAlertStore(max_items=100)
        ops_mock = MagicMock()
        count = drain_to_ops_store(source, ops_mock)
        assert count == 0
        ops_mock.save_alert.assert_not_called()
