"""FIX-017 regression: temporal_correlation_engine only registered when patterns loaded."""
import inspect


def test_pattern_count_method_exists():
    from src.detection.temporal_correlation import TemporalCorrelationEngine
    assert hasattr(TemporalCorrelationEngine, "pattern_count"), (
        "TemporalCorrelationEngine must have a pattern_count() method"
    )


def test_pattern_count_returns_zero_when_empty():
    from src.detection.temporal_correlation import TemporalCorrelationEngine
    engine = TemporalCorrelationEngine()
    assert engine.pattern_count() == 0


def test_app_registers_temporal_only_when_patterns_present():
    """app.py must guard engine_registry.register(temporal_correlation_engine) on pattern_count() > 0."""
    import web_app.app as app_mod
    src = inspect.getsource(app_mod)
    assert "pattern_count() > 0" in src, (
        "app.py must guard temporal engine registration with pattern_count() > 0"
    )


def test_app_logs_skip_when_no_patterns():
    """app.py must log when temporal engine is skipped."""
    import web_app.app as app_mod
    src = inspect.getsource(app_mod)
    assert "temporal" in src and "skipped" in src, (
        "app.py must log a skip message when temporal engine has no patterns"
    )


def test_temporal_engine_not_registered_by_default():
    """Engine registry must not contain the temporal engine when it has 0 patterns."""
    import web_app.app as app_mod
    from src.detection.temporal_correlation import TemporalCorrelationEngine
    # The temporal engine was conditionally registered at import time.
    # Since no patterns are loaded, it should not be in the registry.
    with app_mod.engine_registry._lock:
        registered_engines = list(app_mod.engine_registry._engines.values())
    temporal_engines = [e for e in registered_engines if isinstance(e, TemporalCorrelationEngine)]
    assert temporal_engines == [], (
        "TemporalCorrelationEngine must not be registered when pattern_count() == 0"
    )
