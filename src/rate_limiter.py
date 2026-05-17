from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
import logging
from threading import Lock
from time import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitConfig:
    requests: int = 120
    window_seconds: int = 60


class InMemoryRateLimiter:
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time()
        window_start = now - self.config.window_seconds
        with self._lock:
            if len(self._events) > 50000:
                stale_keys = [k for k, q in self._events.items() if (not q) or q[-1] < window_start]
                for stale_key in stale_keys[:10000]:
                    self._events.pop(stale_key, None)
            q = self._events[key]
            while q and q[0] < window_start:
                q.popleft()
            if len(q) >= self.config.requests:
                retry_after = int(max(1, self.config.window_seconds - (now - q[0])))
                return False, retry_after
            q.append(now)
            return True, 0


class UnifiedRateLimiter:
    """Two-tier rate limiter — C-05 (Step 19).

    Tier 1 (check_ip):   global IP limit — IP_LIMIT req/min, called from before_request.
    Tier 2 (check_user): per-user limit — USER_LIMIT req/min, called from require_roles().

    Redis-backed when a client is provided; falls back to in-memory with a WARNING
    log on each fallback event (G-STATE-3).
    """

    IP_LIMIT: int = 1000
    USER_LIMIT: int = 200
    WINDOW_SECONDS: int = 60

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._ip_limiter = InMemoryRateLimiter(
            RateLimitConfig(requests=self.IP_LIMIT, window_seconds=self.WINDOW_SECONDS)
        )
        self._user_limiter = InMemoryRateLimiter(
            RateLimitConfig(requests=self.USER_LIMIT, window_seconds=self.WINDOW_SECONDS)
        )

    def check_ip(self, ip: str) -> tuple[bool, int]:
        """Tier 1: global IP rate check. Returns (allowed, retry_after_seconds)."""
        if self._redis is not None:
            try:
                return self._redis_check(f"rl:ip:{ip}", self.IP_LIMIT)
            except Exception as exc:
                logger.warning(
                    "UnifiedRateLimiter: Redis unavailable for IP check (in-memory fallback): %s", exc
                )
        return self._ip_limiter.allow(ip)

    def check_user(self, user_id: str) -> tuple[bool, int]:
        """Tier 2: per-user rate check. Returns (allowed, retry_after_seconds)."""
        if self._redis is not None:
            try:
                return self._redis_check(f"rl:user:{user_id}", self.USER_LIMIT)
            except Exception as exc:
                logger.warning(
                    "UnifiedRateLimiter: Redis unavailable for user check (in-memory fallback): %s", exc
                )
        return self._user_limiter.allow(user_id)

    def _redis_check(self, key: str, limit: int) -> tuple[bool, int]:
        """Fixed-window rate check via Redis INCR + EXPIRE."""
        count = int(self._redis.incr(key))
        if count == 1:
            self._redis.expire(key, self.WINDOW_SECONDS)
        if count > limit:
            ttl = self._redis.ttl(key)
            retry_after = max(1, ttl if ttl > 0 else self.WINDOW_SECONDS)
            return False, retry_after
        return True, 0


# ---------------------------------------------------------------------------
# Module-level singleton — set once at app startup by web_app/app.py
# ---------------------------------------------------------------------------

_unified_limiter: UnifiedRateLimiter | None = None


def get_unified_rate_limiter() -> UnifiedRateLimiter | None:
    """Return the active UnifiedRateLimiter singleton, or None if not yet initialised."""
    return _unified_limiter


def set_unified_rate_limiter(limiter: UnifiedRateLimiter) -> None:
    """Register the app-wide UnifiedRateLimiter (called once at startup)."""
    global _unified_limiter
    _unified_limiter = limiter
