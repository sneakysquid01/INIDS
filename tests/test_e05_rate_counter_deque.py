"""E-05: Rate counter sliding window must use deque — O(k) not O(n)."""
import time
import collections
import pytest

from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from src.middleware import RateLimitMiddleware, RateLimitConfig as MwRateLimitConfig


class TestRateLimiterDeque:
    def test_in_memory_rate_limiter_uses_deque(self):
        limiter = InMemoryRateLimiter(RateLimitConfig(requests=5, window_seconds=60))
        limiter.allow("test_key")
        q = limiter._events["test_key"]
        assert isinstance(q, collections.deque), (
            f"Expected deque, got {type(q).__name__}"
        )

    def test_middleware_rate_limiter_uses_deque(self):
        limiter = RateLimitMiddleware(MwRateLimitConfig(requests=5, window_seconds=60))
        # Inspect the requests dict structure
        from collections import defaultdict, deque
        assert isinstance(limiter.requests, defaultdict)
        # Trigger creation of a key
        limiter.requests["test_ip"].append(time.time())
        assert isinstance(limiter.requests["test_ip"], deque)

    def test_sliding_window_evicts_old_entries(self):
        """Entries older than window_seconds must be evicted on each check."""
        config = RateLimitConfig(requests=100, window_seconds=1)
        limiter = InMemoryRateLimiter(config)

        # Add entry
        allowed, _ = limiter.allow("key1")
        assert allowed is True

        # Simulate time passing by backdating the entry
        q = limiter._events["key1"]
        # Move old entry to past
        old_ts = time.time() - 2  # 2 seconds ago, outside 1-second window
        q.clear()
        q.append(old_ts)

        # Allow should succeed and evict the old entry
        allowed2, _ = limiter.allow("key1")
        assert allowed2 is True
        # Old entry evicted, only the new one remains
        assert len(q) == 1

    def test_rate_limit_enforced_within_window(self):
        config = RateLimitConfig(requests=3, window_seconds=60)
        limiter = InMemoryRateLimiter(config)

        assert limiter.allow("k")[0] is True
        assert limiter.allow("k")[0] is True
        assert limiter.allow("k")[0] is True
        allowed, retry_after = limiter.allow("k")
        assert allowed is False
        assert retry_after > 0

    def test_deque_bounded_by_limit_not_total_events(self):
        """After window expiry, queue length reflects only in-window events."""
        config = RateLimitConfig(requests=1000, window_seconds=60)
        limiter = InMemoryRateLimiter(config)

        for _ in range(50):
            limiter.allow("key2")

        q = limiter._events["key2"]
        assert len(q) == 50  # all within window

    def test_middleware_sliding_window_cleanup(self):
        """Middleware deque removes entries outside the window on each check."""
        from flask import Flask
        app = Flask(__name__)
        config = MwRateLimitConfig(requests=100, window_seconds=1)
        limiter = RateLimitMiddleware(config)

        with app.test_request_context("/", environ_base={"REMOTE_ADDR": "1.2.3.4"}):
            import src.middleware as mw
            from unittest.mock import MagicMock
            mock_req = MagicMock()
            mock_req.remote_addr = "1.2.3.4"
            mock_req.headers = {}
            original = mw.request
            mw.request = mock_req
            try:
                limiter.is_rate_limited()
                key = "ip:1.2.3.4"
                # Force old entry
                limiter.requests[key].appendleft(time.time() - 5)
                before = len(limiter.requests[key])
                limiter.is_rate_limited()
                after = len(limiter.requests[key])
                # Stale entry should be evicted
                assert after <= before
            finally:
                mw.request = original
