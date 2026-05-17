"""D-04: Prevention defaults + TI feed security regression tests.

Verifies:
1. dry_run requires explicit INIDS_DRY_RUN configuration (no silent default).
2. TI feed loader rejects RFC-1918/internal IP indicators.
3. Rollback: INIDS_DRY_RUN=true re-enables dry_run immediately.
"""
import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# D-04-1: dry_run default behaviour
# ---------------------------------------------------------------------------

class TestDryRunDefault:
    def test_dry_run_true_when_env_unset(self):
        """Without INIDS_DRY_RUN, dry_run must default to True (fail-safe)."""
        env = {k: v for k, v in os.environ.items() if k != "INIDS_DRY_RUN"}
        with patch.dict(os.environ, env, clear=True):
            from src.prevention_service import PolicyConfig, _read_dry_run_from_env
            assert _read_dry_run_from_env() is True

    def test_dry_run_true_when_env_empty(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": ""}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is True

    def test_dry_run_false_when_env_false(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "false"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is False

    def test_dry_run_false_when_env_zero(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "0"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is False

    def test_dry_run_false_when_env_no(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "no"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is False

    def test_dry_run_false_when_env_off(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "off"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is False

    def test_dry_run_true_when_env_true(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "true"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is True

    def test_dry_run_true_when_env_one(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "1"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is True

    def test_dry_run_case_insensitive(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "FALSE"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is False

    def test_dry_run_unknown_value_defaults_to_true(self):
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "maybe"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is True

    def test_policy_config_uses_env_for_dry_run(self):
        """PolicyConfig() must read INIDS_DRY_RUN at construction time."""
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "false"}):
            from src.prevention_service import PolicyConfig
            cfg = PolicyConfig()
            assert cfg.dry_run is False

    def test_policy_config_dry_run_true_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "INIDS_DRY_RUN"}
        with patch.dict(os.environ, env, clear=True):
            from src.prevention_service import PolicyConfig
            cfg = PolicyConfig()
            assert cfg.dry_run is True

    def test_rollback_set_env_true_re_enables_dry_run(self):
        """Rollback: INIDS_DRY_RUN=true must restore dry_run=True immediately."""
        with patch.dict(os.environ, {"INIDS_DRY_RUN": "true"}):
            from src.prevention_service import _read_dry_run_from_env
            assert _read_dry_run_from_env() is True


# ---------------------------------------------------------------------------
# D-04-2: TI feed RFC-1918 rejection
# ---------------------------------------------------------------------------

class TestTiFeedRfc1918Rejection:
    def setup_method(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        self.ti = ThreatIntelManager()

    def test_rfc1918_10_rejected_from_csv(self):
        """10.x.x.x must not be loaded into the TI cache."""
        csv_data = "indicator,severity\n10.0.0.1,high\n1.2.3.4,high\n"
        count = self.ti.load_csv_feed(csv_data, source="test")
        assert self.ti.lookup_ip("10.0.0.1") is None
        assert self.ti.lookup_ip("1.2.3.4") is not None
        assert count == 1  # only the public IP was loaded

    def test_rfc1918_192168_rejected_from_csv(self):
        """192.168.x.x must not be loaded into the TI cache."""
        csv_data = "indicator,severity\n192.168.1.100,critical\n"
        self.ti.load_csv_feed(csv_data, source="test")
        assert self.ti.lookup_ip("192.168.1.100") is None

    def test_rfc1918_172_rejected_from_csv(self):
        """172.16-31.x.x must not be loaded into the TI cache."""
        csv_data = "indicator,severity\n172.20.10.5,medium\n"
        self.ti.load_csv_feed(csv_data, source="test")
        assert self.ti.lookup_ip("172.20.10.5") is None

    def test_loopback_rejected_from_csv(self):
        """127.x.x.x must not be loaded."""
        csv_data = "indicator,severity\n127.0.0.1,high\n"
        self.ti.load_csv_feed(csv_data, source="test")
        assert self.ti.lookup_ip("127.0.0.1") is None

    def test_public_ip_accepted(self):
        """Public IPs must still load normally."""
        csv_data = "indicator,severity\n8.8.8.8,high\n"
        count = self.ti.load_csv_feed(csv_data, source="test")
        assert count == 1
        assert self.ti.lookup_ip("8.8.8.8") is not None

    def test_rfc1918_rejected_from_json(self):
        """10.x.x.x in JSON feed must not be loaded."""
        import json
        payload = json.dumps([
            {"ip": "10.1.2.3", "severity": "high"},
            {"ip": "5.6.7.8", "severity": "medium"},
        ])
        count = self.ti.load_json_feed(payload, source="test-json")
        assert self.ti.lookup_ip("10.1.2.3") is None
        assert self.ti.lookup_ip("5.6.7.8") is not None
        assert count == 1

    def test_link_local_rejected_from_json(self):
        """169.254.x.x link-local addresses must not be loaded."""
        import json
        payload = json.dumps([{"ip": "169.254.1.1", "severity": "low"}])
        self.ti.load_json_feed(payload, source="test-json")
        assert self.ti.lookup_ip("169.254.1.1") is None

    def test_ipv6_loopback_rejected_from_json(self):
        """IPv6 loopback (::1) must not be loaded."""
        import json
        payload = json.dumps([{"ip": "::1", "type": "ip", "severity": "high"}])
        self.ti.load_json_feed(payload, source="test-json")
        assert self.ti.lookup_ip("::1") is None

    def test_mixed_feed_only_public_ips_loaded(self):
        """A mixed feed with both public and RFC-1918 IPs: only public loaded."""
        csv_data = (
            "indicator,severity\n"
            "1.1.1.1,high\n"
            "10.0.0.1,high\n"
            "8.8.8.8,medium\n"
            "192.168.0.1,low\n"
            "203.0.113.5,high\n"
        )
        count = self.ti.load_csv_feed(csv_data, source="mixed")
        assert count == 3
        assert self.ti.lookup_ip("1.1.1.1") is not None
        assert self.ti.lookup_ip("8.8.8.8") is not None
        assert self.ti.lookup_ip("203.0.113.5") is not None
        assert self.ti.lookup_ip("10.0.0.1") is None
        assert self.ti.lookup_ip("192.168.0.1") is None


# ---------------------------------------------------------------------------
# D-04-3: Helper function unit tests
# ---------------------------------------------------------------------------

class TestIsRfc1918:
    def test_10_block(self):
        from src.threat_intel.feed_manager import _is_rfc1918
        assert _is_rfc1918("10.255.255.255") is True

    def test_172_16_31(self):
        from src.threat_intel.feed_manager import _is_rfc1918
        assert _is_rfc1918("172.16.0.1") is True
        assert _is_rfc1918("172.31.255.255") is True
        assert _is_rfc1918("172.15.0.1") is False
        assert _is_rfc1918("172.32.0.1") is False

    def test_192_168(self):
        from src.threat_intel.feed_manager import _is_rfc1918
        assert _is_rfc1918("192.168.0.1") is True
        assert _is_rfc1918("192.169.0.1") is False

    def test_public_ip_not_rfc1918(self):
        from src.threat_intel.feed_manager import _is_rfc1918
        assert _is_rfc1918("8.8.8.8") is False
        assert _is_rfc1918("1.1.1.1") is False
        assert _is_rfc1918("203.0.113.1") is False

    def test_invalid_address_not_rfc1918(self):
        from src.threat_intel.feed_manager import _is_rfc1918
        assert _is_rfc1918("not-an-ip") is False
