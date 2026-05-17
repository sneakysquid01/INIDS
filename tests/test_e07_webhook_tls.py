"""E-07: WebhookFirewallAdapter must reject HTTP (non-HTTPS) webhook URLs."""
import pytest

from src.firewall_adapters import WebhookFirewallAdapter


class TestWebhookTLS:
    def test_https_url_accepted(self):
        adapter = WebhookFirewallAdapter(webhook_url="https://example.com/webhook")
        assert adapter.webhook_url == "https://example.com/webhook"

    def test_http_url_rejected(self):
        with pytest.raises(ValueError, match="HTTPS"):
            WebhookFirewallAdapter(webhook_url="http://example.com/webhook")

    def test_empty_url_accepted(self):
        """Empty URL is allowed — unconfigured adapter is a no-op."""
        adapter = WebhookFirewallAdapter(webhook_url="")
        assert adapter.webhook_url == ""

    def test_no_url_accepted(self):
        adapter = WebhookFirewallAdapter()
        assert adapter.webhook_url == ""

    def test_ftp_url_rejected(self):
        with pytest.raises(ValueError, match="HTTPS"):
            WebhookFirewallAdapter(webhook_url="ftp://evil.com/hook")

    def test_https_with_port_accepted(self):
        adapter = WebhookFirewallAdapter(webhook_url="https://example.com:8443/hook")
        assert adapter.webhook_url.startswith("https://")

    def test_error_message_contains_bad_url(self):
        bad_url = "http://bad.example.com/hook"
        with pytest.raises(ValueError) as exc_info:
            WebhookFirewallAdapter(webhook_url=bad_url)
        assert bad_url in str(exc_info.value)

    def test_block_with_valid_https_url(self):
        """block() on a valid-HTTPS adapter should attempt the POST (network call)."""
        import unittest.mock as mock
        with mock.patch("urllib.request.urlopen") as mock_open:
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            adapter = WebhookFirewallAdapter(webhook_url="https://example.com/hook")
            result = adapter.block("1.2.3.4")
            assert result is True
