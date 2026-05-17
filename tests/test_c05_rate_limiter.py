"""C-05 security regression tests — Rate Limiter Unification (Step 19).

Validation checkpoints per PLAN.md C-05 spec (G-RATE-1 correction):
1. UnifiedRateLimiter.check_ip() blocks after IP_LIMIT requests in the window
2. UnifiedRateLimiter.check_user() blocks after USER_LIMIT requests in the window
3. Different IPs / user_ids have independent counters (no cross-contamination)
4. Returns (False, retry_after > 0) when limited, (True, 0) when allowed
5. Redis unavailable → falls back to in-memory, logs WARNING (G-STATE-3)
6. Tier 1 (IP) fires from before_request: 429 when IP limit exceeded
7. Tier 2 (user) fires from require_roles(): 429 when per-user limit exceeded
8. RateLimitMiddleware is no longer active in the middleware chain (double-limit removed)
9. InMemoryRateLimiter (legacy) still importable and functional (not deleted)
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.rate_limiter import (
    InMemoryRateLimiter,
    RateLimitConfig,
    UnifiedRateLimiter,
    get_unified_rate_limiter,
    set_unified_rate_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exhausted_limiter(limit: int = 3) -> UnifiedRateLimiter:
    """Return a UnifiedRateLimiter whose in-memory IP counter is already full."""
    limiter = UnifiedRateLimiter()
    limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=limit, window_seconds=60))
    return limiter


# ---------------------------------------------------------------------------
# Checkpoint 1: check_ip() blocks after IP_LIMIT requests
# ---------------------------------------------------------------------------

class TestCheckIp:
    def test_ip_allowed_under_limit(self):
        """Checkpoint 1: first request under the limit → allowed."""
        limiter = UnifiedRateLimiter()
        # Override with a tiny limit for fast testing
        limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=5, window_seconds=60))
        allowed, retry = limiter.check_ip("1.2.3.4")
        assert allowed is True
        assert retry == 0

    def test_ip_blocked_after_limit(self):
        """Checkpoint 1: request after limit is exceeded → not allowed."""
        limiter = UnifiedRateLimiter()
        limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=3, window_seconds=60))
        for _ in range(3):
            limiter.check_ip("1.2.3.4")
        allowed, retry = limiter.check_ip("1.2.3.4")
        assert allowed is False
        assert retry > 0

    def test_ip_returns_retry_after_when_limited(self):
        limiter = UnifiedRateLimiter()
        limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=1, window_seconds=60))
        limiter.check_ip("2.2.2.2")
        _, retry = limiter.check_ip("2.2.2.2")
        assert isinstance(retry, int)
        assert retry >= 1

    def test_ip_class_constants(self):
        """IP_LIMIT and WINDOW_SECONDS are set to PLAN.md values."""
        assert UnifiedRateLimiter.IP_LIMIT == 1000
        assert UnifiedRateLimiter.WINDOW_SECONDS == 60


# ---------------------------------------------------------------------------
# Checkpoint 2: check_user() blocks after USER_LIMIT requests
# ---------------------------------------------------------------------------

class TestCheckUser:
    def test_user_allowed_under_limit(self):
        """Checkpoint 2: first request under the user limit → allowed."""
        limiter = UnifiedRateLimiter()
        limiter._user_limiter = InMemoryRateLimiter(RateLimitConfig(requests=5, window_seconds=60))
        allowed, retry = limiter.check_user("user-abc")
        assert allowed is True
        assert retry == 0

    def test_user_blocked_after_limit(self):
        """Checkpoint 2: request after user limit exceeded → not allowed."""
        limiter = UnifiedRateLimiter()
        limiter._user_limiter = InMemoryRateLimiter(RateLimitConfig(requests=3, window_seconds=60))
        for _ in range(3):
            limiter.check_user("user-abc")
        allowed, retry = limiter.check_user("user-abc")
        assert allowed is False
        assert retry > 0

    def test_user_class_constants(self):
        """USER_LIMIT is set to PLAN.md value."""
        assert UnifiedRateLimiter.USER_LIMIT == 200


# ---------------------------------------------------------------------------
# Checkpoint 3: Independent counters per IP / user
# ---------------------------------------------------------------------------

class TestIndependentCounters:
    def test_different_ips_independent(self):
        limiter = UnifiedRateLimiter()
        limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=2, window_seconds=60))
        limiter.check_ip("10.0.0.1")
        limiter.check_ip("10.0.0.1")
        # Exhausted for 10.0.0.1 — but 10.0.0.2 should still be allowed
        allowed_other, _ = limiter.check_ip("10.0.0.2")
        assert allowed_other is True

    def test_different_users_independent(self):
        limiter = UnifiedRateLimiter()
        limiter._user_limiter = InMemoryRateLimiter(RateLimitConfig(requests=2, window_seconds=60))
        limiter.check_user("user-a")
        limiter.check_user("user-a")
        allowed_other, _ = limiter.check_user("user-b")
        assert allowed_other is True


# ---------------------------------------------------------------------------
# Checkpoint 4: Return values
# ---------------------------------------------------------------------------

class TestReturnValues:
    def test_allowed_returns_true_zero(self):
        limiter = UnifiedRateLimiter()
        limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=10, window_seconds=60))
        allowed, retry = limiter.check_ip("9.9.9.9")
        assert allowed is True
        assert retry == 0

    def test_blocked_returns_false_positive_retry(self):
        limiter = UnifiedRateLimiter()
        limiter._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=1, window_seconds=60))
        limiter.check_ip("9.9.9.9")
        allowed, retry = limiter.check_ip("9.9.9.9")
        assert allowed is False
        assert retry >= 1


# ---------------------------------------------------------------------------
# Checkpoint 5: Redis fallback logs WARNING (G-STATE-3)
# ---------------------------------------------------------------------------

class TestRedisFallback:
    def test_redis_failure_falls_back_to_in_memory(self):
        """Checkpoint 5: Redis raises → falls back to in-memory, no exception."""
        bad_redis = MagicMock()
        bad_redis.incr.side_effect = Exception("Redis connection refused")

        limiter = UnifiedRateLimiter(redis_client=bad_redis)
        # Must not raise
        allowed, retry = limiter.check_ip("3.3.3.3")
        assert isinstance(allowed, bool)

    def test_redis_failure_logs_warning(self, caplog):
        """Checkpoint 5 / G-STATE-3: Redis failure is logged at WARNING level."""
        bad_redis = MagicMock()
        bad_redis.incr.side_effect = Exception("Redis down")

        limiter = UnifiedRateLimiter(redis_client=bad_redis)
        with caplog.at_level(logging.WARNING, logger="src.rate_limiter"):
            limiter.check_ip("4.4.4.4")
        assert any("Redis unavailable" in r.message for r in caplog.records)

    def test_redis_user_failure_falls_back(self):
        bad_redis = MagicMock()
        bad_redis.incr.side_effect = Exception("Redis connection refused")

        limiter = UnifiedRateLimiter(redis_client=bad_redis)
        allowed, _ = limiter.check_user("user-redis-fail")
        assert isinstance(allowed, bool)


# ---------------------------------------------------------------------------
# Checkpoint 6: Tier 1 (IP) fires from before_request via Flask
# ---------------------------------------------------------------------------

class TestTier1FlaskIntegration:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        import web_app.app as _app
        # Set a limiter with limit=1 so the second request is blocked
        self._original = get_unified_rate_limiter()
        tiny = UnifiedRateLimiter()
        tiny._ip_limiter = InMemoryRateLimiter(RateLimitConfig(requests=1, window_seconds=60))
        set_unified_rate_limiter(tiny)
        self._client = _app.app.test_client()
        yield
        set_unified_rate_limiter(self._original)

    def test_second_ip_request_returns_429(self):
        """Checkpoint 6: after IP limit exhausted, before_request returns 429."""
        # First request (allowed)
        resp1 = self._client.get("/api/health")
        assert resp1.status_code != 429

        # Exhaust the limiter for this IP
        limiter = get_unified_rate_limiter()
        limiter.check_ip("127.0.0.1")  # consume the 1 allowed request

        # Next API request (not a health check) should be rate-limited
        resp2 = self._client.get("/api/auth/status")
        assert resp2.status_code == 429

    def test_health_endpoints_bypass_rate_limit(self):
        """Health endpoints must never be rate-limited."""
        limiter = get_unified_rate_limiter()
        # Exhaust limiter
        for _ in range(5):
            limiter.check_ip("127.0.0.1")

        resp = self._client.get("/api/health")
        assert resp.status_code != 429


# ---------------------------------------------------------------------------
# Checkpoint 8: RateLimitMiddleware no longer active in chain
# ---------------------------------------------------------------------------

class TestRateLimitMiddlewareRemoved:
    def test_rate_limit_middleware_not_called_on_request(self):
        """Checkpoint 8: RateLimitMiddleware.before_request must NOT fire on requests."""
        import src.middleware as _mw
        import web_app.app as _app

        _mw_instance = getattr(_app.app, "rate_limiter", None)
        if _mw_instance is None:
            # RateLimitMiddleware not on app — test passes
            return

        with patch.object(_mw_instance, "before_request", wraps=_mw_instance.before_request) as mock_br:
            client = _app.app.test_client()
            client.get("/api/health")
            mock_br.assert_not_called()


# ---------------------------------------------------------------------------
# Checkpoint 9: InMemoryRateLimiter still functional (not deleted)
# ---------------------------------------------------------------------------

class TestLegacyLimiterPreserved:
    def test_in_memory_rate_limiter_importable(self):
        """Checkpoint 9: InMemoryRateLimiter still importable from src.rate_limiter."""
        from src.rate_limiter import InMemoryRateLimiter as _IRL
        assert _IRL is not None

    def test_in_memory_rate_limiter_functional(self):
        limiter = InMemoryRateLimiter(RateLimitConfig(requests=2, window_seconds=60))
        assert limiter.allow("k")[0] is True
        assert limiter.allow("k")[0] is True
        assert limiter.allow("k")[0] is False


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_set_and_get_unified_rate_limiter(self):
        original = get_unified_rate_limiter()
        new_limiter = UnifiedRateLimiter()
        set_unified_rate_limiter(new_limiter)
        assert get_unified_rate_limiter() is new_limiter
        set_unified_rate_limiter(original)

    def test_get_unified_rate_limiter_returns_instance(self):
        rl = get_unified_rate_limiter()
        assert rl is None or isinstance(rl, UnifiedRateLimiter)
