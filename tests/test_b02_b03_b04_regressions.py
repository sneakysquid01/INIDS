"""Regression tests for B-02, B-03, and B-04.

B-02: ML inference graceful degradation
  - Injected predict() failure → 200 with verdict=unknown, fallback=True
  - ml_unknown_verdict_total increments on each fallback
  - Normal inference still returns correct verdicts
  - AnomalyEngine.evaluate() returns unknown when model not fitted

B-03: PolicyConfig race fix
  - PolicyConfig is frozen (direct assignment raises FrozenInstanceError)
  - config_manager.update() succeeds and returns new frozen config
  - Concurrent reads see consistent snapshots

B-04: AnomalyEngine model swap race
  - AnomalyEngine uses _get_model() / _set_model() (no direct _model attr)
  - Concurrent fit + evaluate produces no AttributeError
"""
from __future__ import annotations

import dataclasses
import threading
import time
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# B-02: ML graceful degradation — MLEngine
# ---------------------------------------------------------------------------

class TestMLEngineGracefulDegradation:
    def _make_engine(self):
        from unittest.mock import MagicMock
        model = MagicMock()
        model.predict.return_value = [0]
        model.predict_proba.return_value = [[0.9, 0.1]]
        from src.detection.engines.ml_engine import MLEngine
        return MLEngine(model=model), model

    def _full_features(self) -> dict:
        from src.schema import DEFAULT_FEATURE_ROW
        return dict(DEFAULT_FEATURE_ROW)

    def test_predict_failure_returns_unknown_verdict(self):
        from src.detection.engines.ml_engine import MLEngine
        model = mock.MagicMock()
        model.predict.side_effect = ValueError("model corrupt")
        engine = MLEngine(model=model)

        result = engine.evaluate(self._full_features())

        assert result.verdict == "unknown"
        assert result.metadata.get("fallback") is True
        assert "error" in result.metadata

    def test_predict_proba_failure_returns_unknown_verdict(self):
        from src.detection.engines.ml_engine import MLEngine
        model = mock.MagicMock()
        model.predict.return_value = [0]
        model.predict_proba.side_effect = RuntimeError("proba broken")
        engine = MLEngine(model=model)

        result = engine.evaluate(self._full_features())
        assert result.verdict == "unknown"
        assert result.metadata.get("fallback") is True

    def test_unknown_verdict_counter_increments(self):
        from src.detection.engines.ml_engine import MLEngine, get_unknown_verdict_total

        model = mock.MagicMock()
        model.predict.side_effect = ValueError("boom")
        engine = MLEngine(model=model)

        before = get_unknown_verdict_total()
        engine.evaluate(self._full_features())
        after = get_unknown_verdict_total()
        assert after == before + 1

    def test_normal_inference_unaffected(self):
        import pandas as pd
        from unittest.mock import MagicMock
        from src.detection.engines.ml_engine import MLEngine
        from src.schema import FEATURE_COLUMNS, DEFAULT_FEATURE_ROW

        model = MagicMock()
        model.predict.return_value = [0]
        model.predict_proba.return_value = [[0.9, 0.1]]
        engine = MLEngine(model=model)

        features = dict(DEFAULT_FEATURE_ROW)
        result = engine.evaluate(features)
        assert result.verdict in ("attack", "normal")
        assert result.metadata.get("fallback") is None

    def test_anomaly_engine_returns_unknown_when_not_fitted(self):
        from src.detection.engines.anomaly_engine import AnomalyEngine
        engine = AnomalyEngine(buffer_size=0)
        assert not engine.is_ready()

        result = engine.evaluate({"source_ip": "1.2.3.4"})
        assert result.verdict == "unknown"
        assert result.metadata.get("fallback") is True

    def test_anomaly_engine_inference_exception_returns_unknown(self):
        from src.detection.engines.anomaly_engine import AnomalyEngine

        engine = AnomalyEngine(buffer_size=0)
        bad_model = mock.MagicMock()
        bad_model.predict.side_effect = RuntimeError("model broken")
        bad_model.n_features_in_ = 5
        engine.set_model(bad_model)
        assert engine.is_ready()

        result = engine.evaluate({"source_ip": "1.2.3.4"})
        assert result.verdict == "unknown"
        assert result.metadata.get("fallback") is True


# ---------------------------------------------------------------------------
# B-03: PolicyConfig frozen + PolicyConfigManager
# ---------------------------------------------------------------------------

class TestPolicyConfigFrozen:
    def test_policy_config_is_frozen(self):
        from src.prevention_service import PolicyConfig
        cfg = PolicyConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            cfg.mode = "auto_block"  # type: ignore[misc]

    def test_policy_config_manager_get_returns_snapshot(self):
        from src.prevention_service import PolicyConfig, PolicyConfigManager
        mgr = PolicyConfigManager(PolicyConfig(mode="monitor"))
        cfg = mgr.get()
        assert cfg.mode == "monitor"

    def test_policy_config_manager_update_returns_new_config(self):
        from src.prevention_service import PolicyConfig, PolicyConfigManager
        mgr = PolicyConfigManager()
        new_cfg = mgr.update(mode="auto_block", dry_run=False)
        assert new_cfg.mode == "auto_block"
        assert new_cfg.dry_run is False
        # Old config untouched (frozen)
        assert mgr.get() is new_cfg

    def test_policy_config_manager_concurrent_reads_consistent(self):
        from src.prevention_service import PolicyConfig, PolicyConfigManager
        mgr = PolicyConfigManager(PolicyConfig(mode="monitor"))
        errors: list[str] = []

        def reader():
            for _ in range(100):
                cfg = mgr.get()
                if cfg.mode not in ("monitor", "auto_block"):
                    errors.append(f"unexpected mode: {cfg.mode}")

        def writer():
            modes = ["auto_block", "monitor"] * 50
            for mode in modes:
                mgr.update(mode=mode)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent read errors: {errors}"

    def test_prevention_service_policy_property_readable(self):
        from src.prevention_service import PreventionService
        svc = PreventionService()
        assert svc.policy.mode == "monitor"

    def test_prevention_service_set_policy_uses_manager(self):
        from src.prevention_service import PreventionService
        svc = PreventionService()
        result = svc.set_policy(mode="auto_block", dry_run=False)
        assert result.mode == "auto_block"
        assert svc.policy.mode == "auto_block"


# ---------------------------------------------------------------------------
# B-04: AnomalyEngine model swap race
# ---------------------------------------------------------------------------

class TestAnomalyEngineModelSwapRace:
    def test_no_direct_model_attribute(self):
        from src.detection.engines.anomaly_engine import AnomalyEngine
        engine = AnomalyEngine()
        assert not hasattr(engine, "_model"), (
            "B-04: AnomalyEngine must not expose _model directly; use _model_ref"
        )

    def test_get_model_returns_none_before_fit(self):
        from src.detection.engines.anomaly_engine import AnomalyEngine
        engine = AnomalyEngine()
        assert engine._get_model() is None

    def test_set_model_then_get_model(self):
        from src.detection.engines.anomaly_engine import AnomalyEngine
        engine = AnomalyEngine()
        sentinel = object()
        engine._set_model(sentinel)
        assert engine._get_model() is sentinel

    def test_concurrent_fit_and_evaluate_no_error(self):
        from src.detection.engines.anomaly_engine import AnomalyEngine
        engine = AnomalyEngine(buffer_size=0)

        # Fit with a small dataset first so evaluate can run
        X = np.random.rand(50, 5)
        feature_names = engine._feature_names[:5]

        errors: list[str] = []

        def fitter():
            for _ in range(20):
                try:
                    engine.fit(X)
                except Exception as exc:
                    errors.append(f"fit: {exc}")
                time.sleep(0.001)

        def evaluator():
            features = {f: 0.5 for f in feature_names}
            for _ in range(50):
                try:
                    engine.evaluate(features)
                except AttributeError as exc:
                    errors.append(f"eval AttributeError: {exc}")
                except Exception:
                    pass  # unknown verdict is fine

        t_fit = threading.Thread(target=fitter)
        t_eval = threading.Thread(target=evaluator)
        t_fit.start()
        t_eval.start()
        t_fit.join()
        t_eval.join()

        assert not errors, f"Concurrent errors: {errors}"
