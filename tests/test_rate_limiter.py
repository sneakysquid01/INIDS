from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig


def test_rate_limiter_blocks_after_threshold():
    limiter = InMemoryRateLimiter(RateLimitConfig(requests=2, window_seconds=60))
    assert limiter.allow("k")[0] is True
    assert limiter.allow("k")[0] is True
    allowed, retry = limiter.allow("k")
    assert allowed is False
    assert retry >= 1
