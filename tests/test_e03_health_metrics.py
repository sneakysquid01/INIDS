"""E-03: Health endpoint must return real uptime_seconds, not hardcoded values."""
import time
import pytest


def _get_app():
    from web_app.app import app, _APP_START_TIME
    return app, _APP_START_TIME


class TestHealthMetrics:
    def test_app_start_time_is_float(self):
        _, start_time = _get_app()
        assert isinstance(start_time, float)

    def test_app_start_time_is_recent(self):
        _, start_time = _get_app()
        assert start_time <= time.time()
        # Should have started within the last 10 minutes (test run constraint)
        assert time.time() - start_time < 600

    def test_dashboard_metrics_uptime_is_numeric(self):
        """_build_dashboard_metrics_payload must return a numeric uptime."""
        from web_app.app import _build_dashboard_metrics_payload, ops_store
        alerts = []
        recent_actions = []
        metrics = _build_dashboard_metrics_payload(alerts=alerts, recent_actions=recent_actions)
        uptime = metrics.get("system_uptime")
        assert uptime is not None, "system_uptime key missing"
        assert isinstance(uptime, (int, float)), (
            f"system_uptime must be numeric, got {type(uptime).__name__}: {uptime!r}"
        )

    def test_dashboard_metrics_uptime_not_hardcoded_string(self):
        """system_uptime must not be the hardcoded '4.2h' placeholder."""
        from web_app.app import _build_dashboard_metrics_payload
        metrics = _build_dashboard_metrics_payload(alerts=[], recent_actions=[])
        uptime = metrics.get("system_uptime")
        assert uptime != "4.2h", "system_uptime is still the hardcoded placeholder '4.2h'"

    def test_dashboard_metrics_uptime_is_non_negative(self):
        from web_app.app import _build_dashboard_metrics_payload
        metrics = _build_dashboard_metrics_payload(alerts=[], recent_actions=[])
        assert metrics["system_uptime"] >= 0
