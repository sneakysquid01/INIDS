from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class RateLimitConfig:
    requests: int = 120
    window_seconds: int = 60


class InMemoryRateLimiter:
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time()
        window_start = now - self.config.window_seconds
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
