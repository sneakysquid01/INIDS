"""E-06: HoneypotDetectionEngine must read INIDS_HONEYPOT_CONFIDENCE from env."""
import os
import pytest

from src.detection.engines.honeypot_engine import HoneypotDetectionEngine


def _engine_with_ip(ip="10.1.1.254"):
    return HoneypotDetectionEngine(
        engine_id="honeypot_test",
        honeypot_ips=[ip],
        honeypot_ports=[],
    )


class TestHoneypotConfidence:
    def test_default_confidence_is_100(self, monkeypatch):
        monkeypatch.delenv("INIDS_HONEYPOT_CONFIDENCE", raising=False)
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 100.0

    def test_env_var_sets_confidence(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "75.0")
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 75.0

    def test_env_var_100_sets_confidence(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "100.0")
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 100.0

    def test_env_var_0_sets_confidence(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "0.0")
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 0.0

    def test_invalid_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "not_a_float")
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 100.0

    def test_out_of_range_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "150.0")
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 100.0

    def test_negative_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "-5.0")
        engine = _engine_with_ip()
        result = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result.confidence == 100.0

    def test_confidence_read_at_call_time_not_module_load(self, monkeypatch):
        """Env var change after engine init must take effect immediately."""
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "50.0")
        engine = _engine_with_ip()
        result1 = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result1.confidence == 50.0

        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "80.0")
        result2 = engine.evaluate({"dest_ip": "10.1.1.254"})
        assert result2.confidence == 80.0

    def test_src_ip_match_uses_env_confidence(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "42.5")
        engine = _engine_with_ip("10.1.1.254")
        result = engine.evaluate({"src_ip": "10.1.1.254", "dest_ip": "1.2.3.4"})
        assert result.confidence == 42.5

    def test_port_match_uses_env_confidence(self, monkeypatch):
        monkeypatch.setenv("INIDS_HONEYPOT_CONFIDENCE", "60.0")
        engine = HoneypotDetectionEngine(
            engine_id="test",
            honeypot_ips=[],
            honeypot_ports=[9999],
        )
        result = engine.evaluate({"dst_port": 9999})
        assert result.confidence == 60.0

    def test_get_confidence_method_exists(self):
        assert callable(HoneypotDetectionEngine._get_confidence)
