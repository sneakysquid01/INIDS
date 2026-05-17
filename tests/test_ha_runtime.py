from __future__ import annotations

import os
from unittest.mock import patch


def test_leader_election_without_redis_is_leader():
    # B-05: single-instance (no Redis) mode requires INIDS_REDIS_REQUIRED=false
    from src.ha.leader_election import LeaderElection

    with patch.dict(os.environ, {"INIDS_REDIS_REQUIRED": "false"}):
        le = LeaderElection(redis_client=None, instance_id="unit-test")
        le.start()
    assert le.is_leader is True
    status = le.status()
    assert status["is_leader"] is True
    assert status["redis_connected"] is False
    le.stop()


def test_health_check_degraded_when_one_probe_fails():
    from src.ha.health_check import HealthCheck

    hc = HealthCheck()
    hc.register("ok", lambda: {"ready": True})
    hc.register("bad", lambda: {"ready": False})

    report = hc.check()
    assert report["status"] == "degraded"
    assert report["checks"]["ok"]["ready"] is True
    assert report["checks"]["bad"]["ready"] is False


def test_health_live_endpoint(monkeypatch):
    import web_app.app as app_module

    client = app_module.app.test_client()
    resp = client.get("/api/health/live")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "live"
    assert data["process_up"] is True


def test_health_ready_endpoint_healthy(monkeypatch):
    import web_app.app as app_module

    class _Healthy:
        def check(self):
            return {"status": "healthy", "checks": {"model": {"ready": True}}}

    monkeypatch.setattr(app_module, "health_check", _Healthy())
    monkeypatch.setattr(app_module, "_ensure_pipeline_started", lambda: True)

    client = app_module.app.test_client()
    resp = client.get("/api/health/ready")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_health_ready_endpoint_degraded(monkeypatch):
    import web_app.app as app_module

    class _Degraded:
        def check(self):
            return {"status": "degraded", "checks": {"model": {"ready": False}}}

    monkeypatch.setattr(app_module, "health_check", _Degraded())
    monkeypatch.setattr(app_module, "_ensure_pipeline_started", lambda: False)

    client = app_module.app.test_client()
    resp = client.get("/api/health/ready")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "degraded"


def test_api_health_contains_readiness_and_leader(monkeypatch):
    import web_app.app as app_module

    class _Healthy:
        def check(self):
            return {"status": "healthy", "checks": {"model": {"ready": True}}}

    class _Leader:
        def status(self):
            return {"instance_id": "x", "is_leader": True, "redis_connected": False}

    monkeypatch.setattr(app_module, "health_check", _Healthy())
    monkeypatch.setattr(app_module, "leader_election", _Leader())
    monkeypatch.setattr(app_module, "ensure_detection_service", lambda: True)
    monkeypatch.setattr(app_module, "_ensure_pipeline_started", lambda: True)
    monkeypatch.setattr(app_module, "_pipeline_status", lambda: {"running": False, "enabled": False})

    client = app_module.app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "readiness" in data
    assert "leader_election" in data


def test_shutdown_runtime_stops_scheduler_and_leader(monkeypatch):
    import web_app.app as app_module

    calls = {"scheduler": 0, "leader": 0, "pipeline": 0, "redis": 0}

    class _Scheduler:
        def stop(self):
            calls["scheduler"] += 1

    class _Leader:
        def stop(self):
            calls["leader"] += 1

    monkeypatch.setattr(app_module, "prevention_scheduler", _Scheduler())
    monkeypatch.setattr(app_module, "leader_election", _Leader())
    monkeypatch.setattr(app_module, "_stop_pipeline_runtime", lambda: calls.__setitem__("pipeline", calls["pipeline"] + 1))
    monkeypatch.setattr(app_module, "_close_redis_client", lambda: calls.__setitem__("redis", calls["redis"] + 1))

    app_module.app._shutdown_started = False
    app_module._shutdown_runtime()

    assert calls["scheduler"] == 1
    assert calls["leader"] == 1
    assert calls["pipeline"] == 1
    assert calls["redis"] == 1


def test_scheduler_skips_when_not_leader(monkeypatch):
    from src.ips.scheduler import PreventionScheduler

    class _Executor:
        def __init__(self):
            self.cleanup_calls = 0
            self.reconcile_calls = 0

        def cleanup_expired_actions(self):
            self.cleanup_calls += 1
            return 0

        def reconcile(self):
            self.reconcile_calls += 1
            return {}

    ex = _Executor()
    leader_state = {"leader": False}

    sched = PreventionScheduler(
        ex,
        interval_seconds=1,
        reconcile_every=1,
        is_leader_fn=lambda: leader_state["leader"],
    )

    # Run a tiny controlled loop by monkeypatching sleep to stop quickly.
    import src.ips.scheduler as sched_mod

    ticks = {"n": 0}

    def _sleep(_seconds):
        ticks["n"] += 1
        if ticks["n"] >= 2:
            sched._stop_event.set()

    monkeypatch.setattr(sched_mod, "sleep", _sleep)

    sched._run()
    assert ex.cleanup_calls == 0
    assert ex.reconcile_calls == 0
