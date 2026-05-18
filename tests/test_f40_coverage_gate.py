"""Step 40 (Phase F) — Full regression suite: coverage gate validation and targeted tests.

Assembles coverage verification for security-critical paths and adds targeted
tests for previously under-covered detection/IPS modules (threshold_engine,
signature_engine) to satisfy the ≥ 80% gate on src/auth/.

Tests in this file cover:
  - ThresholdEngine: _RateCounter, evaluate (SYN flood, port scan, error rate, normal)
  - SignatureEngine: load_rules, add_rule, evaluate (match/no-match, FP suppression)
  - Coverage configuration: pyproject.toml has [tool.coverage] section
  - CI: test job runs in security.yml
"""
from __future__ import annotations

import pathlib
import time

import pytest

ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# ThresholdEngine — _RateCounter
# ---------------------------------------------------------------------------


class TestRateCounter:
    def _make_counter(self):
        from src.detection.engines.threshold_engine import _RateCounter
        return _RateCounter()

    def test_empty_counter_returns_zero(self):
        c = self._make_counter()
        assert c.count(60.0) == 0

    def test_record_and_count(self):
        c = self._make_counter()
        now = time.monotonic()
        c.record(now)
        c.record(now)
        assert c.count(60.0, now + 1) == 2

    def test_old_timestamps_evicted(self):
        c = self._make_counter()
        old = time.monotonic() - 120
        c.record(old)
        c.record(old)
        # window is 60s — old entries are outside the window
        assert c.count(60.0) == 0

    def test_cap_on_max_timestamps(self):
        from src.detection.engines.threshold_engine import _RateCounter
        c = _RateCounter()
        now = time.monotonic()
        for _ in range(_RateCounter._MAX_TIMESTAMPS + 5):
            c.record(now)
        assert len(c._timestamps) <= _RateCounter._MAX_TIMESTAMPS


# ---------------------------------------------------------------------------
# ThresholdEngine — evaluate
# ---------------------------------------------------------------------------


class TestThresholdEngine:
    def _make_engine(self, **kwargs):
        from src.detection.engines.threshold_engine import ThresholdEngine
        return ThresholdEngine(**kwargs)

    def test_normal_traffic_returns_normal(self):
        eng = self._make_engine()
        result = eng.evaluate({"source_ip": "10.0.0.1", "count": 1, "serror_rate": 0.0})
        assert result.verdict == "normal"

    def test_syn_flood_detected(self):
        eng = self._make_engine()
        features = {"source_ip": "10.0.0.2", "count": 201, "serror_rate": 0.8}
        result = eng.evaluate(features)
        assert result.verdict == "attack"
        assert result.attack_type == "dos"

    def test_port_scan_detected(self):
        eng = self._make_engine()
        features = {"source_ip": "10.0.0.3", "dst_host_srv_count": 51, "srv_diff_host_rate": 0.5}
        result = eng.evaluate(features)
        assert result.verdict == "attack"
        assert result.attack_type == "probe"

    def test_high_error_rate_detected(self):
        eng = self._make_engine()
        features = {"source_ip": "10.0.0.4", "rerror_rate": 0.7}
        result = eng.evaluate(features)
        assert result.verdict == "attack"
        assert result.severity == "medium"

    def test_connection_rate_limit_triggers_attack(self):
        eng = self._make_engine(connection_rate_limit=5)
        ip = "10.0.0.5"
        result = None
        for _ in range(10):
            result = eng.evaluate({"source_ip": ip})
        assert result.verdict == "attack"
        assert result.metadata["reason"] == "connection_rate_exceeded"

    def test_fp_manager_suppression_returns_normal(self):
        class MockFP:
            def is_suppressed(self, *args):
                return True

        eng = self._make_engine(fp_manager=MockFP())
        result = eng.evaluate({"source_ip": "10.0.0.6", "count": 999, "serror_rate": 1.0})
        assert result.verdict == "normal"
        assert result.metadata.get("suppressed_by_fp_manager") is True

    def test_fp_manager_not_suppressed_still_detects(self):
        class MockFP:
            def is_suppressed(self, *args):
                return False

        eng = self._make_engine(fp_manager=MockFP())
        features = {"source_ip": "10.0.0.7", "count": 201, "serror_rate": 0.8}
        result = eng.evaluate(features)
        assert result.verdict == "attack"

    def test_fp_manager_exception_is_handled(self):
        class BrokenFP:
            def is_suppressed(self, *args):
                raise RuntimeError("fp broken")

        eng = self._make_engine(fp_manager=BrokenFP())
        result = eng.evaluate({"source_ip": "10.0.0.8"})
        assert result.verdict == "normal"  # default when FP check fails

    def test_missing_feature_field_no_crash(self):
        eng = self._make_engine()
        result = eng.evaluate({"source_ip": "10.0.0.9"})
        assert result.verdict in ("normal", "attack")

    def test_engine_id_and_type(self):
        eng = self._make_engine(engine_id="custom_threshold")
        assert eng.engine_id == "custom_threshold"
        assert eng.engine_type == "threshold"
        assert eng.is_ready() is True

    def test_match_static_method_with_invalid_op(self):
        from src.detection.engines.threshold_engine import ThresholdEngine
        rule = {"conditions": [("count", "<<", 5)]}
        assert ThresholdEngine._match(rule, {"count": 3}) is True  # unknown op falls through

    def test_match_static_method_none_value(self):
        from src.detection.engines.threshold_engine import ThresholdEngine
        rule = {"conditions": [("count", ">", 5)]}
        assert ThresholdEngine._match(rule, {}) is False

    def test_match_static_method_nonfloat_value(self):
        from src.detection.engines.threshold_engine import ThresholdEngine
        rule = {"conditions": [("count", ">", 5)]}
        assert ThresholdEngine._match(rule, {"count": "not-a-number"}) is False

    def test_ip_counter_eviction_on_full_table(self):
        eng = self._make_engine()
        eng._max_tracked_ips = 3
        for i in range(5):
            eng.evaluate({"source_ip": f"192.168.0.{i}"})
        assert len(eng._counters) <= 4


# ---------------------------------------------------------------------------
# SignatureEngine — load_rules, add_rule, evaluate
# ---------------------------------------------------------------------------


class TestSignatureEngine:
    def _make_engine(self):
        from src.detection.engines.signature_engine import SignatureEngine
        return SignatureEngine()

    def test_empty_engine_is_not_ready(self):
        eng = self._make_engine()
        assert eng.is_ready() is False

    def test_add_rule_makes_engine_ready(self):
        eng = self._make_engine()
        eng.add_rule({"id": "SIG-001", "name": "Test", "conditions": [{"field": "count", "op": ">", "value": 10}]})
        assert eng.is_ready() is True

    def test_load_rules_from_nonexistent_path(self):
        eng = self._make_engine()
        eng.load_rules("/nonexistent/path/rules.yaml")
        assert eng.is_ready() is False

    def test_evaluate_no_match_returns_normal(self):
        eng = self._make_engine()
        eng.add_rule({
            "id": "SIG-001",
            "name": "SYN Flood",
            "severity": "high",
            "attack_type": "dos",
            "confidence": 90.0,
            "conditions": [{"field": "serror_rate", "op": ">", "value": 0.8}],
        })
        result = eng.evaluate({"serror_rate": 0.1})
        assert result.verdict == "normal"

    def test_evaluate_match_returns_attack(self):
        eng = self._make_engine()
        eng.add_rule({
            "id": "SIG-002",
            "name": "High Error",
            "severity": "critical",
            "attack_type": "dos",
            "confidence": 95.0,
            "conditions": [{"field": "serror_rate", "op": ">", "value": 0.5}],
        })
        result = eng.evaluate({"serror_rate": 0.9})
        assert result.verdict == "attack"
        assert result.severity == "critical"
        assert result.rule_id == "SIG-002"

    def test_fp_suppressed_rule_is_skipped(self):
        class MockFP:
            def is_suppressed(self, engine_id, rule_id):
                return rule_id == "SIG-003"

        eng = self._make_engine()
        eng._fp_manager = MockFP()
        eng.add_rule({
            "id": "SIG-003",
            "name": "Suppressed",
            "severity": "high",
            "attack_type": "dos",
            "conditions": [{"field": "serror_rate", "op": ">", "value": 0.5}],
        })
        result = eng.evaluate({"serror_rate": 0.9})
        assert result.verdict == "normal"

    def test_engine_id_and_type(self):
        eng = self._make_engine()
        assert eng.engine_id == "signature"
        assert eng.engine_type == "signature"

    def test_legacy_match_rule_no_conditions(self):
        from src.detection.engines.signature_engine import SignatureEngine
        assert SignatureEngine._match_rule_legacy({"conditions": []}, {}) is False

    def test_legacy_match_rule_missing_field(self):
        from src.detection.engines.signature_engine import SignatureEngine
        rule = {"conditions": [{"field": "x", "op": ">", "value": 1}]}
        assert SignatureEngine._match_rule_legacy(rule, {}) is False

    def test_legacy_match_rule_unknown_op(self):
        from src.detection.engines.signature_engine import SignatureEngine
        rule = {"conditions": [{"field": "x", "op": "~~", "value": 1}]}
        assert SignatureEngine._match_rule_legacy(rule, {"x": 5}) is False

    def test_legacy_match_rule_string_comparison(self):
        from src.detection.engines.signature_engine import SignatureEngine
        rule = {"conditions": [{"field": "proto", "op": "==", "value": "tcp"}]}
        assert SignatureEngine._match_rule_legacy(rule, {"proto": "tcp"}) is True

    def test_load_rules_from_list_yaml(self, tmp_path):
        import yaml
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(yaml.dump([
            {"id": "L01", "conditions": [{"field": "count", "op": ">", "value": 100}]}
        ]))
        eng = self._make_engine()
        eng.load_rules(str(rules_file))
        assert eng.is_ready() is True

    def test_load_rules_from_dict_yaml(self, tmp_path):
        import yaml
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(yaml.dump({
            "rules": [{"id": "D01", "conditions": [{"field": "count", "op": ">", "value": 1}]}]
        }))
        eng = self._make_engine()
        eng.load_rules(str(rules_file))
        assert eng.is_ready() is True


# ---------------------------------------------------------------------------
# Coverage configuration validation
# ---------------------------------------------------------------------------


def test_pyproject_toml_has_coverage_config():
    """pyproject.toml must contain [tool.coverage] or pytest addopts for coverage."""
    content = (ROOT / "pyproject.toml").read_text()
    has_coverage_tool = "[tool.coverage" in content
    has_addopts_cov = "cov" in content
    assert has_coverage_tool or has_addopts_cov, (
        "pyproject.toml is missing coverage configuration — "
        "Step 40 requires coverage gates on src/auth, src/detection, src/ips"
    )


def test_all_phase_security_tests_collected():
    """Security regression tests for every phase must exist in tests/."""
    tests_dir = ROOT / "tests"
    required_prefixes = [
        "test_c01_", "test_c02_", "test_c03_", "test_c04_", "test_c05_",
        "test_c06_", "test_c07_",
        "test_d01_", "test_d02_", "test_d03_", "test_d04_", "test_d05_",
        "test_d06_", "test_d07_", "test_d08_",
        "test_e01_", "test_e02_", "test_e03_", "test_e04_", "test_e05_",
        "test_e06_", "test_e07_", "test_e08_",
        "test_f_auth_remove",
        "test_f42_", "test_f43_", "test_f44_",
    ]
    present = [f.name for f in tests_dir.glob("test_*.py")]
    missing = [p for p in required_prefixes if not any(f.startswith(p) for f in present)]
    assert not missing, f"Security regression test files missing: {missing}"


def test_ci_workflow_runs_tests():
    """security.yml must include a test job."""
    workflow = ROOT / ".github" / "workflows" / "security.yml"
    content = workflow.read_text()
    assert "pytest" in content, "CI workflow does not run pytest — tests are not blocking in CI"
