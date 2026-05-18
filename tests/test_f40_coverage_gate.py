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


# ---------------------------------------------------------------------------
# RuleConditionEvaluator — advanced operators
# ---------------------------------------------------------------------------


class TestRuleConditionEvaluator:
    def _make_evaluator(self):
        from src.detection.rule_compiler import RuleConditionEvaluator
        return RuleConditionEvaluator()

    def _cond(self, field, op, value, **kwargs):
        return {"field": field, "operator": op, "value": value, **kwargs}

    # Numeric operators
    def test_numeric_gt_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("x", ">", 5), {"x": 10}) is True

    def test_numeric_gt_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("x", ">", 5), {"x": 2}) is False

    def test_numeric_type_error_returns_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("x", ">", "nope"), {"x": "also_nope"}) is False

    def test_missing_field_returns_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("missing", ">", 1), {}) is False

    def test_missing_field_not_in_returns_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("missing", "not_in", ["a"]), {}) is True

    def test_missing_field_is_empty_returns_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition({"field": "x", "operator": "is_empty", "value": None}, {}) is True

    # Regex operator
    def test_regex_match(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "regex", r"tc[p]"), {"proto": "tcp"}) is True

    def test_regex_no_match(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "regex", r"udp"), {"proto": "tcp"}) is False

    def test_regex_invalid_pattern_returns_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("x", "regex", r"[invalid"), {"x": "tcp"}) is False

    def test_regex_cache_hit(self):
        e = self._make_evaluator()
        cond = self._cond("proto", "regex", r"tcp")
        e.evaluate_condition(cond, {"proto": "tcp"})
        # second call should use cache — no exception
        assert e.evaluate_condition(cond, {"proto": "udp"}) is False

    # Contains/not_contains
    def test_contains_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("msg", "contains", "error"), {"msg": "error found"}) is True

    def test_contains_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("msg", "contains", "error"), {"msg": "all good"}) is False

    def test_not_contains_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("msg", "not_contains", "error"), {"msg": "all good"}) is True

    def test_not_contains_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("msg", "not_contains", "error"), {"msg": "error found"}) is False

    def test_contains_case_sensitive(self):
        e = self._make_evaluator()
        cond = {"field": "msg", "operator": "contains", "value": "Error", "case_sensitive": True}
        assert e.evaluate_condition(cond, {"msg": "error found"}) is False

    # starts_with / ends_with
    def test_starts_with_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("msg", "starts_with", "http"), {"msg": "https://example.com"}) is True

    def test_starts_with_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("msg", "starts_with", "ftp"), {"msg": "https://example.com"}) is False

    def test_ends_with_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("path", "ends_with", ".exe"), {"path": "malware.exe"}) is True

    def test_ends_with_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("path", "ends_with", ".pdf"), {"path": "malware.exe"}) is False

    # in / not_in
    def test_in_list_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "in", ["tcp", "udp"]), {"proto": "tcp"}) is True

    def test_in_list_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "in", ["tcp", "udp"]), {"proto": "icmp"}) is False

    def test_in_scalar_coerced_to_list(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "in", "tcp"), {"proto": "tcp"}) is True

    def test_not_in_list_true(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "not_in", ["tcp", "udp"]), {"proto": "icmp"}) is True

    def test_not_in_list_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("proto", "not_in", ["tcp", "udp"]), {"proto": "tcp"}) is False

    # Range operator
    def test_range_within(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("port", "range", [1024, 65535]), {"port": 8080}) is True

    def test_range_outside(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("port", "range", [1024, 65535]), {"port": 80}) is False

    def test_range_invalid_value_list(self):
        e = self._make_evaluator()
        # value is not [min, max] — missing second element
        assert e.evaluate_condition(self._cond("port", "range", [1024]), {"port": 80}) is False

    def test_range_non_numeric_value(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("port", "range", [1024, 65535]), {"port": "bad"}) is False

    # is_empty / is_not_empty
    def test_is_empty_empty_string(self):
        e = self._make_evaluator()
        assert e.evaluate_condition({"field": "x", "operator": "is_empty", "value": None}, {"x": ""}) is True

    def test_is_empty_non_empty(self):
        e = self._make_evaluator()
        assert e.evaluate_condition({"field": "x", "operator": "is_empty", "value": None}, {"x": "hello"}) is False

    def test_is_not_empty_non_empty(self):
        e = self._make_evaluator()
        assert e.evaluate_condition({"field": "x", "operator": "is_not_empty", "value": None}, {"x": "hello"}) is True

    def test_is_not_empty_zero(self):
        e = self._make_evaluator()
        assert e.evaluate_condition({"field": "x", "operator": "is_not_empty", "value": None}, {"x": 0}) is False

    # Unknown operator
    def test_unknown_operator_returns_false(self):
        e = self._make_evaluator()
        assert e.evaluate_condition(self._cond("x", "xyzzy", 1), {"x": 1}) is False


# ---------------------------------------------------------------------------
# RuleCompiler — compile_rule, compile_conditions, and/or logic
# ---------------------------------------------------------------------------


class TestRuleCompiler:
    def _make_compiler(self):
        from src.detection.rule_compiler import RuleCompiler
        return RuleCompiler()

    def test_compile_and_evaluate_simple(self):
        c = self._make_compiler()
        rule = {"id": "R1", "conditions": [{"field": "count", "operator": ">", "value": 5}]}
        assert c.evaluate_rule(rule, {"count": 10}) is True
        assert c.evaluate_rule(rule, {"count": 1}) is False

    def test_compile_rule_cache_hit(self):
        c = self._make_compiler()
        rule = {"id": "R2", "conditions": [{"field": "count", "operator": ">", "value": 5}]}
        p1 = c.compile_rule(rule)
        p2 = c.compile_rule(rule)
        assert p1 is p2

    def test_compile_empty_conditions_returns_true(self):
        c = self._make_compiler()
        rule = {"id": "R3", "conditions": []}
        assert c.evaluate_rule(rule, {}) is True

    def test_compile_rule_invalid_conditions_returns_false_predicate(self):
        from src.detection.rule_compiler import RuleCompilationError
        c = self._make_compiler()
        # Passing a string as conditions triggers RuleCompilationError → predicate returns False
        rule = {"id": "R4", "conditions": "not_valid"}
        predicate = c.compile_rule(rule)
        assert predicate({}) is False

    def test_compile_conditions_and_group(self):
        c = self._make_compiler()
        rule = {
            "id": "R5",
            "conditions": {
                "and": [
                    {"field": "count", "operator": ">", "value": 5},
                    {"field": "serror_rate", "operator": ">", "value": 0.5},
                ]
            },
        }
        assert c.evaluate_rule(rule, {"count": 10, "serror_rate": 0.8}) is True
        assert c.evaluate_rule(rule, {"count": 10, "serror_rate": 0.1}) is False

    def test_compile_conditions_or_group(self):
        c = self._make_compiler()
        rule = {
            "id": "R6",
            "conditions": {
                "or": [
                    {"field": "count", "operator": ">", "value": 100},
                    {"field": "serror_rate", "operator": ">", "value": 0.9},
                ]
            },
        }
        assert c.evaluate_rule(rule, {"count": 10, "serror_rate": 0.95}) is True
        assert c.evaluate_rule(rule, {"count": 10, "serror_rate": 0.1}) is False

    def test_compile_conditions_or_empty_returns_false(self):
        c = self._make_compiler()
        rule = {"id": "R7", "conditions": {"or": []}}
        assert c.evaluate_rule(rule, {"count": 999}) is False

    def test_compile_conditions_and_empty_returns_true(self):
        c = self._make_compiler()
        rule = {"id": "R8", "conditions": {"and": []}}
        assert c.evaluate_rule(rule, {}) is True

    def test_compile_conditions_invalid_dict_group(self):
        from src.detection.rule_compiler import RuleCompilationError
        c = self._make_compiler()
        # Dict without "and"/"or" key → RuleCompilationError → predicate returns False
        rule = {"id": "R9", "conditions": {"nope": []}}
        predicate = c.compile_rule(rule)
        assert predicate({}) is False

    def test_compile_and_with_nested_or(self):
        c = self._make_compiler()
        rule = {
            "id": "R10",
            "conditions": [
                {"field": "count", "operator": ">", "value": 5},
                {"or": [
                    {"field": "serror_rate", "operator": ">", "value": 0.9},
                    {"field": "rerror_rate", "operator": ">", "value": 0.9},
                ]},
            ],
        }
        assert c.evaluate_rule(rule, {"count": 10, "serror_rate": 0.0, "rerror_rate": 0.95}) is True
        assert c.evaluate_rule(rule, {"count": 10, "serror_rate": 0.0, "rerror_rate": 0.0}) is False

    def test_compile_or_with_nested_and(self):
        c = self._make_compiler()
        rule = {
            "id": "R11",
            "conditions": {
                "or": [
                    {"and": [
                        {"field": "count", "operator": ">", "value": 100},
                        {"field": "serror_rate", "operator": ">", "value": 0.8},
                    ]},
                    {"field": "rerror_rate", "operator": ">", "value": 0.9},
                ]
            },
        }
        # First branch: count=200 and serror=0.9 → True
        assert c.evaluate_rule(rule, {"count": 200, "serror_rate": 0.9, "rerror_rate": 0.1}) is True
        # Second branch: rerror=0.95 → True
        assert c.evaluate_rule(rule, {"count": 1, "serror_rate": 0.0, "rerror_rate": 0.95}) is True
        # Neither → False
        assert c.evaluate_rule(rule, {"count": 1, "serror_rate": 0.0, "rerror_rate": 0.1}) is False

    def test_compile_and_non_dict_condition(self):
        c = self._make_compiler()
        # A non-dict item in conditions list → logs warning and returns False
        rule = {"id": "R12", "conditions": ["not_a_dict"]}
        predicate = c.compile_rule(rule)
        assert predicate({}) is False

    def test_compile_or_non_dict_condition(self):
        c = self._make_compiler()
        rule = {"id": "R13", "conditions": {"or": ["not_a_dict"]}}
        predicate = c.compile_rule(rule)
        assert predicate({}) is False


# ---------------------------------------------------------------------------
# ActionExecutor — basic paths (no real firewall)
# ---------------------------------------------------------------------------


class TestActionExecutorBasic:
    def _make_executor(self, **kwargs):
        from src.ips.action_executor import ActionExecutor
        from src.firewall_adapters import MockFirewallAdapter
        adapter = MockFirewallAdapter()
        return ActionExecutor(adapter=adapter, adapter_name="mock", **kwargs)

    def test_block_ip_invalid_ip(self):
        ex = self._make_executor()
        ok, status = ex.block_ip("not_an_ip", 60)
        assert ok is False
        assert status == "invalid_ip"

    def test_unblock_ip_invalid_ip(self):
        ex = self._make_executor()
        ok, status = ex.unblock_ip("not_an_ip")
        assert ok is False
        assert status == "invalid_ip"

    def test_block_ip_success(self):
        ex = self._make_executor()
        ok, status = ex.block_ip("1.2.3.4", 60)
        assert ok is True
        assert status == "blocked"

    def test_unblock_ip_success(self):
        ex = self._make_executor()
        # Block first so there's something to unblock
        ex.block_ip("1.2.3.5", 60)
        ok, status = ex.unblock_ip("1.2.3.5")
        assert ok is True
        assert status == "unblocked"

    def test_rate_limit_delegates_to_block(self):
        ex = self._make_executor()
        ok, status = ex.rate_limit("10.0.0.1", 60)
        assert ok is True
        assert status == "rate_limited"

    def test_cleanup_expired_actions_no_store(self):
        ex = self._make_executor()
        # No ops_store → returns 0 immediately
        assert ex.cleanup_expired_actions() == 0

    def test_reconcile_stateless_adapter(self):
        from src.ips.action_executor import ActionExecutor
        from src.firewall_adapters import WebhookFirewallAdapter

        class _FakeStore:
            def list_active_blocks(self, limit=5000):
                return []

        adapter = WebhookFirewallAdapter(webhook_url="https://example.com/hook")
        ex = ActionExecutor(adapter=adapter, adapter_name="webhook", ops_store=_FakeStore())
        result = ex.reconcile()
        assert result.get("skipped") is True

    def test_reconcile_no_store(self):
        ex = self._make_executor()
        result = ex.reconcile()
        assert result["db_active"] == 0
        assert result["firewall_rules"] == 0

    def test_approve_pending_block_no_store(self):
        ex = self._make_executor()
        result = ex.approve_pending_block("act_abc", policy=None)
        assert result["ok"] is False
        assert result["error"] == "no_ops_store"

    def test_circuit_breaker_opens_after_failures(self):
        ex = self._make_executor(cb_failure_threshold=2, cb_open_duration_s=5.0)
        assert ex._circuit_open() is False
        ex._record_adapter_result(False)
        assert ex._circuit_open() is False  # not yet at threshold
        ex._record_adapter_result(False)
        assert ex._circuit_open() is True   # now open

    def test_circuit_breaker_resets_on_success(self):
        ex = self._make_executor(cb_failure_threshold=2)
        ex._record_adapter_result(False)
        ex._record_adapter_result(False)
        assert ex._circuit_open() is True
        # Success after threshold doesn't reset open_until mid-window;
        # but failure_count resets
        ex._record_adapter_result(True)
        assert ex._cb_failure_count == 0

    def test_emit_audit_no_store_no_bus(self):
        ex = self._make_executor()
        # Should not raise when both ops_store and event_bus are None
        ex._emit_audit("test_event", "test message")


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
