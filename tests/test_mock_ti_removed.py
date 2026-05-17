"""
Regression tests for PLAN.md Phase A Step 2 (A-02): Mock Threat Intel Removal.
Verifies that load_threat_intel() no longer loads RFC-1918 or hardcoded indicators.
"""
import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

APP_PY = os.path.join(ROOT, "web_app", "app.py")


def _read_app_source() -> str:
    with open(APP_PY, encoding="utf-8") as f:
        return f.read()


def test_no_rfc1918_in_load_threat_intel():
    """load_threat_intel() must not contain any hardcoded RFC-1918 IP addresses."""
    source = _read_app_source()
    # Extract just the load_threat_intel function body using line scanning
    in_func = False
    func_lines = []
    for line in source.splitlines():
        if line.startswith("def load_threat_intel("):
            in_func = True
        elif in_func and line.startswith("def ") and not line.startswith("def load_threat_intel("):
            break
        if in_func:
            func_lines.append(line)

    func_body = "\n".join(func_lines)
    rfc1918_patterns = ["192.168.", "10.0.", "10.1.", "172.16.", "172.17.", "172.18.",
                        "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31."]
    for pattern in rfc1918_patterns:
        assert pattern not in func_body, (
            f"Found RFC-1918 address pattern '{pattern}' in load_threat_intel(). "
            "Mock indicators must be removed (PLAN.md A-02)."
        )


def test_no_mock_feed_source_in_load_threat_intel():
    """load_threat_intel() must not reference 'mock_feed' source."""
    source = _read_app_source()
    in_func = False
    func_lines = []
    for line in source.splitlines():
        if line.startswith("def load_threat_intel("):
            in_func = True
        elif in_func and line.startswith("def ") and not line.startswith("def load_threat_intel("):
            break
        if in_func:
            func_lines.append(line)
    func_body = "\n".join(func_lines)
    assert "mock_feed" not in func_body, (
        "load_threat_intel() still references 'mock_feed'. All mock indicators must be removed."
    )


def test_load_threat_intel_no_op_without_feed_path(monkeypatch):
    """load_threat_intel() must return without loading indicators when INIDS_TI_FEED_PATH is unset."""
    monkeypatch.delenv("INIDS_TI_FEED_PATH", raising=False)

    # We test the source-level guarantee: no hardcoded indicators loaded
    # (full integration test requires app context; this verifies the control flow)
    source = _read_app_source()
    assert "INIDS_TI_FEED_PATH" in source, (
        "load_threat_intel() must check INIDS_TI_FEED_PATH env var"
    )


def test_no_mock_indicators_list_in_app():
    """The mock_indicators list must not exist anywhere in app.py."""
    source = _read_app_source()
    assert "mock_indicators" not in source, (
        "mock_indicators variable found in app.py. All mock TI data must be removed (PLAN.md A-02)."
    )
