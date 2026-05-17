from dataclasses import replace

from src.core.event_bus import DetectionEvent, EventBus
from src.detection.aggregator import AggregatedResult
from src.detection.engine_base import EngineResult
from src.pipeline.backpressure import BackpressureController, BackpressureLevel
import web_app.app as app_module


class FakeRedisClient:
    def __init__(self):
        self.xadd_calls = []
        self.closed = False

    def ping(self):
        return True

    def xadd(self, stream_key, fields, **kwargs):
        self.xadd_calls.append((stream_key, fields))
        return "1-0"

    def xlen(self, stream_key):
        return 0

    def close(self):
        self.closed = True


class FakeStreamProcessor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.stopped = False

    def run(self):
        return None

    def stop(self):
        self.stopped = True

    def pending_count(self):
        return 0

    def lag(self):
        return 0


class FakePipelineWorker:
    def __init__(self, processor, backpressure):
        self.processor = processor
        self.backpressure = backpressure
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        self.processor.stop()

    def status(self):
        return {
            "running": self.started and not self.stopped,
            "backpressure": self.backpressure.status(),
            "pending": 0,
        }


def _reset_pipeline_state(monkeypatch):
    monkeypatch.setattr(app_module.app, "_pipeline_worker", None, raising=False)
    monkeypatch.setattr(app_module.app, "_pipeline_backpressure", None, raising=False)
    monkeypatch.setattr(app_module.app, "_pipeline_processor", None, raising=False)
    monkeypatch.setattr(app_module.app, "_pipeline_worker_started", False, raising=False)
    monkeypatch.setattr(app_module.app, "_redis_client", None, raising=False)
    monkeypatch.setattr(app_module.app, "_redis_client_initialized", False, raising=False)


def test_stream_result_callback_publishes_detection_event(monkeypatch):
    bus = EventBus()
    captured = []
    bus.subscribe(DetectionEvent, captured.append)
    monkeypatch.setattr(app_module, "event_bus", bus)

    aggregated = AggregatedResult(
        verdict="attack",
        confidence=91.5,
        severity="high",
        attack_type="dos",
        engine_results=[
            EngineResult(
                engine_id="threshold_primary",
                engine_type="threshold",
                verdict="attack",
                confidence=91.5,
                severity="high",
                attack_type="dos",
            )
        ],
        strategy="any_trigger",
    )

    app_module._stream_result_callback(aggregated, {"source_ip": "1.1.1.1", "count": 900})

    assert len(captured) == 1
    event = captured[0]
    assert event.source_ip == "1.1.1.1"
    assert event.prediction == "Attack"
    assert event.reason == "stream_pipeline"
    assert event.features["count"] == 900


def test_pipeline_startup_skips_when_redis_unavailable(monkeypatch):
    _reset_pipeline_state(monkeypatch)
    monkeypatch.setattr(
        app_module,
        "SETTINGS",
        replace(app_module.SETTINGS, pipeline_enabled=True, redis_url="redis://localhost:6379"),
    )
    monkeypatch.setattr(app_module, "_get_redis_client", lambda: None)

    started = app_module._ensure_pipeline_started()

    assert started is False
    assert app_module.app._pipeline_worker is None
    assert app_module._pipeline_status()["running"] is False


def test_pipeline_startup_and_ingest_stream_path(monkeypatch, tmp_path):
    from src.ops_store import OpsStore

    _reset_pipeline_state(monkeypatch)
    fake_redis = FakeRedisClient()
    monkeypatch.setattr(
        app_module,
        "SETTINGS",
        replace(
            app_module.SETTINGS,
            pipeline_enabled=True,
            redis_url="redis://localhost:6379",
            pipeline_stream_key="inids:flows",
            pipeline_batch_size=10,
        ),
    )
    _analyst_key = "pipeline-test-analyst-key"
    monkeypatch.setenv("INIDS_ANALYST_API_KEY", _analyst_key)
    store = OpsStore(str(tmp_path / "ops_pipeline.db"))
    monkeypatch.setattr(app_module, "ops_store", store)
    app_module.app.ops_store = store
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "StreamProcessor", FakeStreamProcessor)
    monkeypatch.setattr(app_module, "PipelineWorker", FakePipelineWorker)
    monkeypatch.setattr(app_module, "_get_redis_client", lambda: fake_redis)

    client = app_module.app.test_client()

    response = client.post(
        "/api/ingest",
        json={
            "source": "sensor-a",
            "features": {
                "duration": 1,
                "src_bytes": 10,
                "dst_bytes": 5,
                "count": 20,
                "source_ip": "1.1.1.1",
            },
        },
        headers={"X-API-Key": _analyst_key},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["queued"] == 1
    assert payload["queue_size"] == 0
    assert payload["pipeline"]["running"] is True
    assert len(fake_redis.xadd_calls) == 1
    stream_key, fields = fake_redis.xadd_calls[0]
    assert stream_key == "inids:flows"
    assert "payload" in fields

    app_module._stop_pipeline_runtime()
    assert app_module.app._pipeline_worker is None


def test_backpressure_mode_transitions():
    controller = BackpressureController(
        sampling_threshold=10,
        shedding_threshold=20,
        cooldown_seconds=0,
    )

    assert controller.update(0) == BackpressureLevel.NORMAL
    assert controller.update(12) == BackpressureLevel.SAMPLING
    assert controller.update(25) == BackpressureLevel.SHEDDING
    assert controller.update(1) == BackpressureLevel.NORMAL
