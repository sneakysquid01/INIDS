from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from collections import defaultdict, deque
from time import time


def test_rate_limiter_blocks_after_threshold():
    limiter = InMemoryRateLimiter(RateLimitConfig(requests=2, window_seconds=60))
    assert limiter.allow("k")[0] is True
    assert limiter.allow("k")[0] is True
    allowed, retry = limiter.allow("k")
    assert allowed is False
    assert retry >= 1


def test_rate_limiter_prunes_stale_keys_when_cardinality_is_high():
    limiter = InMemoryRateLimiter(RateLimitConfig(requests=2, window_seconds=60))

    # Create high-cardinality stale keys to trigger cleanup path.
    limiter._events = defaultdict(deque)
    stale_ts = time() - 120
    for i in range(50010):
        limiter._events[f"old-{i}"] = deque([stale_ts])

    allowed, _ = limiter.allow("fresh")
    assert allowed is True
    assert len(limiter._events) < 50010
