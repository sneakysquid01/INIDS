"""E-01: IPBlockingMiddleware must pass 127.0.0.1 and ::1 without blocking."""
import pytest
from unittest.mock import MagicMock

from src.middleware import IPBlockingMiddleware


class TestLocalhostWhitelist:
    def setup_method(self):
        self.middleware = IPBlockingMiddleware(max_failures=1, block_time_seconds=300)

    def test_127_0_0_1_not_blocked_after_failures(self):
        for _ in range(10):
            self.middleware.add_failure("127.0.0.1")
        assert self.middleware.is_blocked("127.0.0.1") is False

    def test_ipv6_loopback_not_blocked_after_failures(self):
        for _ in range(10):
            self.middleware.add_failure("::1")
        assert self.middleware.is_blocked("::1") is False

    def test_localhost_not_blocked_after_failures(self):
        for _ in range(10):
            self.middleware.add_failure("localhost")
        assert self.middleware.is_blocked("localhost") is False

    def test_non_loopback_is_blocked_after_failures(self):
        for _ in range(5):
            self.middleware.add_failure("10.0.0.1")
        assert self.middleware.is_blocked("10.0.0.1") is True

    def test_loopback_whitelist_present(self):
        assert "127.0.0.1" in self.middleware.whitelist
        assert "::1" in self.middleware.whitelist

    def test_before_request_passes_loopback(self):
        """before_request() must return None (allow) for loopback IPs."""
        for _ in range(10):
            self.middleware.add_failure("127.0.0.1")

        mock_request = MagicMock()
        mock_request.remote_addr = "127.0.0.1"

        import src.middleware as mw
        original = mw.request
        mw.request = mock_request
        try:
            result = self.middleware.before_request()
        finally:
            mw.request = original

        assert result is None

    def test_before_request_blocks_non_loopback(self):
        """before_request() must return 403 for a blocked non-loopback IP."""
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context("/", environ_base={"REMOTE_ADDR": "10.0.0.99"}):
            for _ in range(10):
                self.middleware.add_failure("10.0.0.99")
            result = self.middleware.before_request()
            assert result is not None
            response, status = result
            assert status == 403
