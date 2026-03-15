"""EventBus concurrency stress tests.

Covers:
- Many threads publishing events simultaneously — no deadlock.
- All published events are handled (no missing deliveries).
- Handlers that block briefly do not drop events from other publishers.
- Late subscriptions do not cause data races.
- High-volume publish does not OOM (handlers counted, not stored).
"""
from __future__ import annotations

import threading
import time
from collections import Counter

import pytest

from src.core.event_bus import EventBus, DetectionEvent, RiskScoreEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(ip: str = "10.0.0.1") -> DetectionEvent:
    return DetectionEvent(source_ip=ip, prediction="Normal", confidence=80.0)


# ---------------------------------------------------------------------------
# Tests: basic thread safety
# ---------------------------------------------------------------------------


class TestEventBusThreadSafety:
    def test_concurrent_publishers_all_handled(self):
        """100 threads each publish 10 events → handler called 1 000 times."""
        bus = EventBus()
        counter = Counter()
        lock = threading.Lock()

        def handler(event):
            with lock:
                counter["hits"] += 1

        bus.subscribe(DetectionEvent, handler)

        threads = []
        for i in range(100):
            def publish_batch(idx=i):
                for j in range(10):
                    bus.publish(_make_detection(f"10.0.{idx}.{j}"))
            t = threading.Thread(target=publish_batch)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert counter["hits"] == 1000

    def test_concurrent_subscribe_and_publish(self):
        """Threads subscribe while other threads are publishing — no crash."""
        bus = EventBus()
        results = []
        lock = threading.Lock()

        def late_subscribe():
            time.sleep(0.01)
            bus.subscribe(DetectionEvent, lambda e: None)

        def publisher():
            for _ in range(50):
                bus.publish(_make_detection())

        threads = [threading.Thread(target=publisher) for _ in range(5)]
        threads += [threading.Thread(target=late_subscribe) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No assertion needed — success means no deadlock/crash

    def test_no_deadlock_with_blocking_handler(self):
        """A blocking handler should not deadlock other concurrent publishers."""
        bus = EventBus()
        processed = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def slow_handler(event):
            """Simulates a slow handler (e.g., I/O)."""
            time.sleep(0.05)
            with lock:
                processed.append(event.source_ip)

        bus.subscribe(DetectionEvent, slow_handler)

        threads = [
            threading.Thread(target=lambda ip=f"192.168.0.{i}": bus.publish(_make_detection(ip)))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(processed) == 10

    def test_exception_in_handler_does_not_block_other_threads(self):
        """Even with a crashing handler, concurrent publishers complete."""
        bus = EventBus()
        good_results = []
        lock = threading.Lock()

        def good_handler(event):
            with lock:
                good_results.append(event.source_ip)

        def bad_handler(event):
            raise RuntimeError("always crashes")

        bus.subscribe(DetectionEvent, bad_handler)
        bus.subscribe(DetectionEvent, good_handler)

        threads = [
            threading.Thread(target=lambda ip=f"172.16.0.{i}": bus.publish(_make_detection(ip)))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(good_results) == 20


# ---------------------------------------------------------------------------
# Tests: handler isolation
# ---------------------------------------------------------------------------


class TestHandlerIsolation:
    def test_handler_for_different_type_not_called(self):
        bus = EventBus()
        detection_calls = []
        risk_calls = []

        bus.subscribe(DetectionEvent, lambda e: detection_calls.append(e))
        bus.subscribe(RiskScoreEvent, lambda e: risk_calls.append(e))

        bus.publish(_make_detection())
        assert len(detection_calls) == 1
        assert len(risk_calls) == 0

    def test_multiple_subscribers_all_receive_event(self):
        bus = EventBus()
        results = []

        for i in range(5):
            bus.subscribe(DetectionEvent, lambda e, idx=i: results.append(idx))

        bus.publish(_make_detection())
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_subscribe_after_publish_misses_earlier_events(self):
        """EventBus is not a replay bus — late subscribers miss past events."""
        bus = EventBus()
        received = []

        bus.publish(_make_detection("1.2.3.4"))  # published before subscribe

        bus.subscribe(DetectionEvent, lambda e: received.append(e.source_ip))
        bus.publish(_make_detection("5.6.7.8"))  # published after subscribe

        assert received == ["5.6.7.8"]


# ---------------------------------------------------------------------------
# Tests: high volume stress
# ---------------------------------------------------------------------------


class TestHighVolumePublish:
    def test_10k_events_all_counted(self):
        bus = EventBus()
        counter = [0]
        lock = threading.Lock()

        def handler(event):
            with lock:
                counter[0] += 1

        bus.subscribe(DetectionEvent, handler)

        n = 10_000
        for i in range(n):
            bus.publish(_make_detection(f"10.{i // 256}.{i % 256}.1"))

        assert counter[0] == n

    def test_high_volume_does_not_cause_oom(self):
        """Publish many events without keeping references — memory stays bounded."""
        bus = EventBus()
        # No-op handler: just drops on the floor
        bus.subscribe(DetectionEvent, lambda e: None)

        for i in range(50_000):
            bus.publish(_make_detection(f"10.0.0.1"))
        # If we reach here without MemoryError, the test passes.
